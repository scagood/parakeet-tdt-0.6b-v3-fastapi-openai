"""FastAPI routes for OpenAI-compatible transcription."""
from __future__ import annotations

import asyncio
import functools
import math
import re
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse, PlainTextResponse, Response

from . import aligner, spoken
from .audio import load_audio
from .chunker import auto_chunk, slice_chunks
from .config import (
    CPU_INFO,
    MAX_AUDIO_SECONDS,
    MAX_BATCH_BYTES,
    MAX_BATCH_FILES,
    MAX_REQUEST_CHUNKS,
    MAX_UPLOAD_BYTES,
    MODEL_ALIASES,
    MODEL_CONFIGS,
    SPOKEN_NUMBERS,
    TARGET_SR,
    UPLOAD_READ_CHUNK_BYTES,
    logger,
)
from .model import default_model_name, loaded_models

router = APIRouter()
_ALLOWED_FORMATS = {"json", "text", "srt", "vtt", "verbose_json"}

# Parakeet TDT reports token START times only (80 ms encoder frames); its
# duration head emits at most 4 frames per token, so a token's audio never
# extends more than 0.32 s past its start.
_WORD_TAIL_SEC = 0.32


@dataclass(slots=True)
class _PreparedAudio:
    waveform: Any
    ranges: List[Tuple[int, int]]
    pieces: List[Any]
    duration: float


class _AudioTooLong(ValueError):
    pass


def _clean_text(text: str) -> str:
    if not text:
        return ""
    text = text.replace("\u2581", " ").strip()
    text = re.sub(r"\s+", " ", text)
    return text.replace(" '", "'")


def _fmt_srt_time(seconds: float) -> str:
    total_ms = max(0, int(round(float(seconds) * 1000)))
    hours, remainder = divmod(total_ms, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    secs, millis = divmod(remainder, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def _segments_to_srt(segments: Sequence[Dict[str, Any]]) -> str:
    lines: List[str] = []
    index = 1
    for segment in segments:
        text = segment["segment"].strip()
        if not text:
            continue
        lines.extend(
            [
                str(index),
                f"{_fmt_srt_time(segment['start'])} --> {_fmt_srt_time(segment['end'])}",
                text,
                "",
            ]
        )
        index += 1
    return "\n".join(lines)


def _segments_to_vtt(segments: Sequence[Dict[str, Any]]) -> str:
    output = ["WEBVTT", ""]
    for segment in segments:
        text = segment["segment"].strip()
        if not text:
            continue
        start = _fmt_srt_time(segment["start"]).replace(",", ".")
        end = _fmt_srt_time(segment["end"]).replace(",", ".")
        output.extend([f"{start} --> {end}", text, ""])
    return "\n".join(output)


def _extract(result: Any) -> Dict[str, Any]:
    text = _clean_text(getattr(result, "text", str(result)))
    tokens = [str(token) for token in (getattr(result, "tokens", []) or [])]
    raw_timestamps = list(getattr(result, "timestamps", []) or [])
    if tokens and len(raw_timestamps) != len(tokens):
        logger.warning(
            "token/timestamp length mismatch: %d tokens vs %d timestamps",
            len(tokens),
            len(raw_timestamps),
        )
    # Substitute the previous timestamp for missing/invalid entries instead of
    # dropping them, so tokens and timestamps always stay aligned 1:1.
    timestamps: List[float] = []
    previous = 0.0
    for index in range(len(tokens)):
        try:
            timestamp = float(raw_timestamps[index])
        except (IndexError, TypeError, ValueError):
            timestamp = previous
        if not math.isfinite(timestamp) or timestamp < 0.0:
            timestamp = previous
        timestamps.append(timestamp)
        previous = timestamp
    return {"text": text, "tokens": tokens, "timestamps": timestamps}


def _validate_model(model: Optional[str]) -> str:
    normalized = (model or default_model_name()).strip().lower()
    normalized = MODEL_ALIASES.get(normalized, normalized)
    if normalized not in MODEL_CONFIGS:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown model {model!r}. Available models: {sorted(MODEL_CONFIGS)}",
        )
    return normalized


def _validate_format(response_format: str) -> str:
    normalized = (response_format or "json").strip().lower()
    if normalized not in _ALLOWED_FORMATS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported response_format {response_format!r}",
        )
    return normalized


async def _read_upload_limited(upload: UploadFile) -> bytes:
    if not upload or not upload.filename:
        raise HTTPException(status_code=400, detail="No file provided")
    declared_size = getattr(upload, "size", None)
    if declared_size is not None and declared_size > MAX_UPLOAD_BYTES:
        await upload.close()
        raise HTTPException(
            status_code=413,
            detail=f"File exceeds the {MAX_UPLOAD_BYTES} byte upload limit",
        )

    payload = bytearray()
    try:
        while True:
            chunk = await upload.read(UPLOAD_READ_CHUNK_BYTES)
            if not chunk:
                break
            payload.extend(chunk)
            if len(payload) > MAX_UPLOAD_BYTES:
                raise HTTPException(
                    status_code=413,
                    detail=f"File exceeds the {MAX_UPLOAD_BYTES} byte upload limit",
                )
    finally:
        await upload.close()
    if not payload:
        raise HTTPException(status_code=400, detail="Empty file")
    return bytes(payload)


def _prepare_audio(raw: bytes) -> _PreparedAudio:
    waveform = load_audio(raw)
    duration = float(waveform.size) / TARGET_SR
    if duration <= 0:
        raise ValueError("decoded audio is empty")
    if duration > MAX_AUDIO_SECONDS:
        raise _AudioTooLong(
            f"audio duration {duration:.1f}s exceeds limit {MAX_AUDIO_SECONDS:.1f}s"
        )
    ranges = auto_chunk(waveform)
    pieces = slice_chunks(waveform, ranges)
    if len(pieces) > MAX_REQUEST_CHUNKS:
        raise _AudioTooLong(
            f"audio produced {len(pieces)} chunks; limit is {MAX_REQUEST_CHUNKS}"
        )
    return _PreparedAudio(
        waveform=waveform,
        ranges=ranges,
        pieces=pieces,
        duration=duration,
    )


async def _prepare_in_pool(request: Request, raw: bytes) -> _PreparedAudio:
    loop = asyncio.get_running_loop()
    try:
        return await loop.run_in_executor(
            request.app.state.audio_pool, _prepare_audio, raw
        )
    except _AudioTooLong as exc:
        raise HTTPException(status_code=413, detail=str(exc)) from exc
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("audio decode/preprocessing failed")
        raise HTTPException(status_code=415, detail="Audio could not be decoded") from exc


def _apply_alignment(
    words: List[Dict[str, Any]],
    spans: Optional[Sequence[Optional[aligner.Span]]],
    chunk_start: float,
    chunk_end: float,
) -> None:
    """Re-time one chunk's words from aligner spans (seconds from chunk start).

    Words the aligner could not place keep their model times, squeezed between
    their aligned neighbours so the word list stays in order.
    """
    if not spans:
        return
    for word, span in zip(words, spans):
        if span is not None:
            word["start"] = min(chunk_end, chunk_start + span[0])
            word["end"] = min(chunk_end, max(word["start"], chunk_start + span[1]))
    for index, (word, span) in enumerate(zip(words, spans)):
        if span is not None:
            continue
        low = words[index - 1]["end"] if index else chunk_start
        high = next(
            (w["start"] for w, s in zip(words[index + 1 :], spans[index + 1 :]) if s is not None),
            chunk_end,
        )
        high = max(low, high)
        word["start"] = min(max(word["start"], low), high)
        word["end"] = min(max(word["end"], word["start"]), high)


def _choose_readings(
    texts: List[str], options: List[List[str]], chunk: aligner.ChunkAligner
) -> List[str]:
    """Pick, by ear, which reading of each ambiguous word was said.

    The default readings are aligned first to find each word's neighbours; each
    ambiguous word is then heard between its neighbours' edges, so a wrong
    default only costs its own slot.
    """
    choices = [said[0] for said in options]
    spans = chunk.spans(spoken.spoken_words(texts)) or [None] * len(texts)
    for index, said in enumerate(options):
        if len(said) < 2:
            continue
        before = next((s[1] for s in reversed(spans[:index]) if s is not None), 0.0)
        after = next((s[0] for s in spans[index + 1 :] if s is not None), float("inf"))
        choices[index] = said[chunk.best(said, before, after)]
    return choices


def _speak_numbers(
    words: List[Dict[str, Any]], chunk: Callable[[], Optional[aligner.ChunkAligner]]
) -> Optional[List[Dict[str, Any]]]:
    """One chunk's words with numbers, money and units said out ("$5" -> "five
    dollars"), or None if nothing changed (PARAKEET_SPOKEN_NUMBERS).

    Where the text allows several readings ("£2.10": "two pounds ten", "two
    pounds and ten pence", ...) the audio decides, via `chunk()` (called only
    then, as it loads the aligner); without it the first reading is used. A word
    that becomes several shares its time span out by length; when the aligner
    runs next it re-times each of them from the audio.
    """
    texts = [w["word"] for w in words]
    options = [spoken.readings(text) for text in texts]
    if all(said == [text] for said, text in zip(options, texts)):
        return None
    choices: List[Optional[str]] = [said[0] for said in options]
    if any(len(said) > 1 for said in options) and (heard := chunk()) is not None:
        choices = _choose_readings(texts, options, heard)
    said = spoken.spoken_words(texts, choices=choices)
    out: List[Dict[str, Any]] = []
    for index, (word, text) in enumerate(zip(words, said)):
        if text == word["word"]:
            out.append(word)
            continue
        if spoken.starts_sentence(words[index - 1]["word"] if index else None):
            text = spoken.capitalize(text)
        parts = text.split()
        total = sum(len(part) for part in parts)
        start, span = word["start"], word["end"] - word["start"]
        for position, part in enumerate(parts):
            end = word["end"] if position == len(parts) - 1 else start + span * len(part) / total
            out.append({"start": start, "end": end, "word": part})
            start = end
    return out


def _stitch(
    prepared: _PreparedAudio,
    results: Sequence[Any],
    *,
    align: bool = False,
    speak: bool = False,
    language: Optional[str] = None,
) -> Tuple[str, List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Chunk results -> (text, segments, words).

    `align` re-times words with the forced aligner; `speak` says numbers out
    (PARAKEET_SPOKEN_NUMBERS). Both may run the aligner's model, so a caller
    setting either should call this off the event loop.
    """
    if len(results) != len(prepared.ranges):
        raise RuntimeError(
            f"inference returned {len(results)} results for "
            f"{len(prepared.ranges)} chunks"
        )

    segments: List[Dict[str, Any]] = []
    words: List[Dict[str, Any]] = []
    for (start_sample, end_sample), chunk_wav, result in zip(
        prepared.ranges, prepared.pieces, results
    ):
        chunk_start = start_sample / TARGET_SR
        chunk_end = min(prepared.duration, end_sample / TARGET_SR)
        info = _extract(result)
        if not info["text"]:
            continue

        timestamps = info["timestamps"]
        segment_start = chunk_start
        if timestamps:
            segment_start = min(chunk_end, max(chunk_start, chunk_start + timestamps[0]))
        segment_end = max(segment_start, chunk_end)
        if timestamps:
            segment_end = min(
                segment_end, max(segment_start, chunk_start + timestamps[-1] + _WORD_TAIL_SEC)
            )
        segments.append(
            {
                "start": segment_start,
                "end": segment_end,
                "segment": info["text"],
            }
        )

        # Group BPE pieces into words: a piece starting with the word marker
        # ("\u2581" or a plain space, depending on export) opens a new word.
        grouped: List[Tuple[str, float, float]] = []  # (word, first_ts, last_ts)
        for token, timestamp in zip(info["tokens"], timestamps):
            piece = token.replace("\u2581", " ")
            starts_word = piece.startswith(" ")
            piece = piece.strip()
            if not piece:
                continue
            if grouped and not starts_word:
                word, first_ts, _last_ts = grouped[-1]
                grouped[-1] = (word + piece, first_ts, timestamp)
            else:
                grouped.append((piece, timestamp, timestamp))

        chunk_words: List[Dict[str, Any]] = []
        for index, (word, first_ts, last_ts) in enumerate(grouped):
            word_start = min(chunk_end, max(chunk_start, chunk_start + first_ts))
            if index + 1 < len(grouped):
                next_start = chunk_start + grouped[index + 1][1]
            else:
                next_start = chunk_end
            # The model only reports token start times, so bound the end by the
            # last token's start plus the model's maximum token duration rather
            # than letting a word absorb the silence before the next word.
            word_end = min(next_start, chunk_start + last_ts + _WORD_TAIL_SEC)
            word_end = min(chunk_end, max(word_start, word_end))
            chunk_words.append({"start": word_start, "end": word_end, "word": word})

        heard: List[Optional[aligner.ChunkAligner]] = []

        def chunk(wav: Any = chunk_wav) -> Optional[aligner.ChunkAligner]:
            """The chunk's aligner, made on first use (it loads the model) and shared."""
            if not heard:
                heard.append(aligner.for_chunk(wav, language))
            return heard[0]

        if speak and (said := _speak_numbers(chunk_words, chunk)) is not None:
            # Text and words are rewritten together so they always agree.
            chunk_words = said
            segments[-1]["segment"] = _clean_text(" ".join(w["word"] for w in chunk_words))

        if align and chunk_words and (timer := chunk()) is not None:
            _apply_alignment(
                chunk_words,
                timer.spans([w["word"] for w in chunk_words]),
                chunk_start,
                chunk_end,
            )
            # Segment bounds came from the model's estimates; keep them covering
            # the re-timed words so a cue never ends before its last word.
            segment = segments[-1]
            segment["start"] = min(segment["start"], chunk_words[0]["start"])
            segment["end"] = max(segment["end"], chunk_words[-1]["end"])
        words.extend(chunk_words)

    full_text = _clean_text(" ".join(item["segment"] for item in segments))
    return full_text, segments, words


async def _infer_prepared(request: Request, prepared: _PreparedAudio, model_name: str):
    worker = request.app.state.worker
    if worker is None or not getattr(request.app.state, "ready", False):
        raise HTTPException(status_code=503, detail="Model is not ready")
    return await worker.submit_many(prepared.pieces, model_name)


# parakeet-tdt-0.6b-v3 language coverage; v2 is English-only.
_V3_LANGUAGES = [
    "bg", "hr", "cs", "da", "nl", "en", "et", "fi", "fr", "de", "el", "hu",
    "it", "lv", "lt", "mt", "pl", "pt", "ro", "ru", "sk", "sl", "es", "sv",
    "uk",
]
_MODEL_CREATED = 1785888000  # catalog introduction (2026-08-05), fixed for stable output


def _model_card(name: str) -> Dict[str, Any]:
    hf_id = MODEL_CONFIGS[name]["hf_id"]
    return {
        "id": name,
        "object": "model",
        "created": _MODEL_CREATED,
        "owned_by": hf_id.split("/")[0] if "/" in hf_id else "istupakov",
        "language": ["en"] if name.startswith("parakeet-v2") else _V3_LANGUAGES,
        "task": "automatic-speech-recognition",
        "aliases": sorted(a for a, t in MODEL_ALIASES.items() if t == name),
    }


@router.get("/v1/models")
def list_models():
    return {"object": "list", "data": [_model_card(name) for name in MODEL_CONFIGS]}


@router.get("/v1/models/{model_id:path}")
def retrieve_model(model_id: str):
    try:
        name = _validate_model(model_id)
    except HTTPException as exc:
        raise HTTPException(
            status_code=404, detail=f"Model {model_id!r} not found"
        ) from exc
    return _model_card(name)


@router.get("/health")
def health(request: Request):
    ready = bool(getattr(request.app.state, "ready", False))
    return {
        "status": "healthy" if ready else "starting",
        "ready": ready,
        "models": list(MODEL_CONFIGS.keys()),
        "loaded": loaded_models(),
        "default_model": default_model_name(),
        "cpu": CPU_INFO,
        "aligner": aligner.status(),
    }


@router.get("/healthz")
def healthz(request: Request):
    if not getattr(request.app.state, "ready", False):
        raise HTTPException(status_code=503, detail="not ready")
    return {"status": "ok"}


@router.post("/v1/audio/transcriptions")
async def transcribe(
    request: Request,
    file: UploadFile = File(...),
    model: Optional[str] = Form(None),
    response_format: str = Form("json"),
    timestamp_granularities: Optional[List[str]] = Form(
        None, alias="timestamp_granularities[]"
    ),
    timestamp_granularities_plain: Optional[List[str]] = Form(
        None, alias="timestamp_granularities"
    ),
    language: Optional[str] = Form(None),
    prompt: Optional[str] = Form(None),
    temperature: Optional[float] = Form(None),
):
    del prompt, temperature  # accepted for OpenAI client compatibility
    model_name = _validate_model(model)
    output_format = _validate_format(response_format)
    granularities = set(timestamp_granularities or []) | set(
        timestamp_granularities_plain or []
    )
    want_words = output_format == "verbose_json" and "word" in granularities
    speak = SPOKEN_NUMBERS and aligner.language_code(language) == "en"
    raw = await _read_upload_limited(file)

    started = time.perf_counter()
    prepared = await _prepare_in_pool(request, raw)
    decode_ms = (time.perf_counter() - started) * 1000

    infer_started = time.perf_counter()
    results = await _infer_prepared(request, prepared, model_name)
    infer_ms = (time.perf_counter() - infer_started) * 1000

    stitch_started = time.perf_counter()
    align = want_words and aligner.supports(language)
    stitch = functools.partial(_stitch, align=align, speak=speak, language=language)
    if align or (speak and aligner.supports(language)):
        # Aligning, or choosing a number's reading by ear, runs a second ONNX model
        # over the audio: keep it off the loop, and off the audio pool so it never
        # holds up other requests' decoding.
        full_text, segments, words = await asyncio.get_running_loop().run_in_executor(
            request.app.state.align_pool, stitch, prepared, results
        )
    else:
        full_text, segments, words = stitch(prepared, results)
    stitch_ms = (time.perf_counter() - stitch_started) * 1000

    logger.info(
        "transcribe model=%s dur=%.2fs chunks=%d decode=%.0fms infer=%.0fms "
        "stitch=%.0fms total=%.0fms",
        model_name,
        prepared.duration,
        len(prepared.pieces),
        decode_ms,
        infer_ms,
        stitch_ms,
        (time.perf_counter() - started) * 1000,
    )

    if output_format == "text":
        return PlainTextResponse(full_text)
    if output_format == "srt":
        return Response(_segments_to_srt(segments), media_type="application/x-subrip")
    if output_format == "vtt":
        return Response(_segments_to_vtt(segments), media_type="text/vtt")
    if output_format == "verbose_json":
        return JSONResponse(
            {
                "task": "transcribe",
                "language": (language or "").strip() or "auto",
                "duration": prepared.duration,
                "text": full_text,
                "segments": [
                    {
                        "id": index,
                        "seek": 0,
                        "start": segment["start"],
                        "end": segment["end"],
                        "text": segment["segment"],
                        "tokens": [],
                        "temperature": 0.0,
                        "avg_logprob": 0.0,
                        "compression_ratio": 0.0,
                        "no_speech_prob": 0.0,
                    }
                    for index, segment in enumerate(segments)
                ],
                "words": words if want_words else None,
            }
        )
    return JSONResponse({"text": full_text})


@router.post("/v1/audio/transcriptions/batch")
async def transcribe_batch(
    request: Request,
    files: List[UploadFile] = File(...),
    model: Optional[str] = Form(None),
):
    if not files:
        raise HTTPException(status_code=400, detail="No files provided")
    if len(files) > MAX_BATCH_FILES:
        raise HTTPException(
            status_code=413,
            detail=f"Batch contains {len(files)} files; limit is {MAX_BATCH_FILES}",
        )
    model_name = _validate_model(model)
    filenames = [upload.filename or "unnamed" for upload in files]

    raws: List[bytes] = []
    total_bytes = 0
    for upload in files:
        raw = await _read_upload_limited(upload)
        total_bytes += len(raw)
        if total_bytes > MAX_BATCH_BYTES:
            raise HTTPException(
                status_code=413,
                detail=f"Batch exceeds the {MAX_BATCH_BYTES} byte limit",
            )
        raws.append(raw)

    loop = asyncio.get_running_loop()
    futures = [
        loop.run_in_executor(request.app.state.audio_pool, _prepare_audio, raw)
        for raw in raws
    ]
    prepared_or_errors = await asyncio.gather(*futures, return_exceptions=True)
    prepared_files: List[_PreparedAudio] = []
    for filename, item in zip(filenames, prepared_or_errors):
        if isinstance(item, _AudioTooLong):
            raise HTTPException(status_code=413, detail=f"{filename}: {item}")
        if isinstance(item, BaseException):
            logger.exception(
                "batch audio preprocessing failed for %s",
                filename,
                exc_info=(type(item), item, item.__traceback__),
            )
            raise HTTPException(
                status_code=415, detail=f"{filename}: audio could not be decoded"
            )
        prepared_files.append(item)

    total_chunks = sum(len(item.pieces) for item in prepared_files)
    if total_chunks > MAX_REQUEST_CHUNKS:
        raise HTTPException(
            status_code=413,
            detail=f"Batch produced {total_chunks} chunks; limit is {MAX_REQUEST_CHUNKS}",
        )

    flattened = [piece for item in prepared_files for piece in item.pieces]
    worker = request.app.state.worker
    if worker is None or not getattr(request.app.state, "ready", False):
        raise HTTPException(status_code=503, detail="Model is not ready")
    flat_results = await worker.submit_many(flattened, model_name)

    # The batch endpoint takes no `language`: the default decides.
    speak = SPOKEN_NUMBERS and aligner.language_code(None) == "en"
    stitch = functools.partial(_stitch, speak=speak)
    cursor = 0
    response_items = []
    for filename, prepared in zip(filenames, prepared_files):
        count = len(prepared.pieces)
        item_results = flat_results[cursor : cursor + count]
        cursor += count
        if speak and aligner.supports(None):  # may hear readings: off the loop
            text, _segments, _words = await asyncio.get_running_loop().run_in_executor(
                request.app.state.align_pool, stitch, prepared, item_results
            )
        else:
            text, _segments, _words = stitch(prepared, item_results)
        response_items.append(
            {"filename": filename, "text": text, "duration": prepared.duration}
        )

    if cursor != len(flat_results):
        raise RuntimeError("inference result accounting mismatch")
    return {"results": response_items, "batch_size": len(response_items)}
