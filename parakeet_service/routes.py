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
    ALIGNER_CONFIGS,
    CHUNK_MIN_SEC,
    CPU_INFO,
    MAX_AUDIO_SECONDS,
    MAX_BATCH_BYTES,
    MAX_BATCH_FILES,
    MAX_REQUEST_CHUNKS,
    MAX_UPLOAD_BYTES,
    MODEL_CONFIGS,
    SPOKEN_NUMBERS,
    TARGET_SR,
    UPLOAD_READ_CHUNK_BYTES,
    logger,
)
from .model import loaded_models, variant_key

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
    # onnx_asr drops the space before any non-word character, meaning to join
    # punctuation, and so glues currency to the word before: "was\u00a31.10".
    text = re.sub(r"(?<=[^\W\d_])(?=[$\u00a3\u20ac])", " ", text)
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


def _validate_model(model: str) -> str:
    normalized = model.strip().lower()
    if normalized not in MODEL_CONFIGS:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown model {model!r}. Available models: {sorted(MODEL_CONFIGS)}",
        )
    return normalized


def _variant(model_name: str, quantization: Optional[str]) -> str:
    """The "model:quant" key a request runs on; fp32 unless it asks otherwise."""
    try:
        return variant_key(model_name, quantization)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from None


def _family(model_name: str) -> str:
    return MODEL_CONFIGS[model_name]["family"]


def _chunk_bounds(model_name: str) -> Tuple[float, float, float]:
    """(target, max, min) seconds for chunking this model (models.yaml)."""
    config = MODEL_CONFIGS[model_name]
    target = config["chunk_target_sec"]
    return target, config["chunk_max_sec"], min(CHUNK_MIN_SEC, target)


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


def _prepare_audio(
    raw: bytes,
    target_sec: float,
    max_sec: float,
    min_sec: float,
) -> _PreparedAudio:
    waveform = load_audio(raw)
    duration = float(waveform.size) / TARGET_SR
    if duration <= 0:
        raise ValueError("decoded audio is empty")
    if duration > MAX_AUDIO_SECONDS:
        raise _AudioTooLong(
            f"audio duration {duration:.1f}s exceeds limit {MAX_AUDIO_SECONDS:.1f}s"
        )
    ranges = auto_chunk(waveform, target_sec=target_sec, max_sec=max_sec, min_sec=min_sec)
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


async def _prepare_in_pool(
    request: Request,
    raw: bytes,
    target_sec: float,
    max_sec: float,
    min_sec: float,
) -> _PreparedAudio:
    loop = asyncio.get_running_loop()
    try:
        return await loop.run_in_executor(
            request.app.state.audio_pool,
            _prepare_audio,
            raw,
            target_sec,
            max_sec,
            min_sec,
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


# Parakeet's number is trusted unless the audio prefers a mishearing of it
# (spoken.rivals) by at least this much, in log-prob over the phrase's window.
# Measured on the TTS benchmarks (381 clips of alternative readings, 306-clip
# corpus) and on spoken lists of 1-99 (571 numbers): no mishearing beat a
# right number by more than 11.7, and at 10 some did ("three nil" to "two
# nil"); the smallest real fix led by 16.3. At 20, 9 + 3 misheard numbers are
# fixed and none broken; 15 fixes one more, 40 and above lose most of them.
_CORRECTION_MARGIN = 20.0
_ZERO = re.compile(rf"(\W*)({'|'.join(spoken.ZERO_WORDS)})(\W*)")


def _phrases(texts: Sequence[str], most: Optional[int] = None) -> List[spoken.Phrase]:
    """The numbers to say in one chunk's words (spoken.phrases): none in a
    chunk mostly in another alphabet (Cyrillic, Greek, ...), which is not
    English whatever the request's `language` said; the aligner leaves those
    untimed by the same rule. Checked second: a chunk with no digit has no
    phrase, and that is the cheaper test."""
    found = spoken.phrases(texts, most=most)
    return [] if found and aligner.other_alphabet(texts) else found


def _undecided(texts: Sequence[str], phrase: spoken.Phrase) -> bool:
    """Whether the audio has anything to decide for `phrase`: which of its
    readings was said, or whether a misheard number was."""
    return len(phrase.options) > 1 or bool(spoken.rivals(texts, phrase))


def _choose_readings(
    texts: List[str], found: Sequence[spoken.Phrase], chunk: aligner.ChunkAligner
) -> List[str]:
    """Pick, by ear, how each phrase was said, or which number was.

    The default readings are aligned first to find each phrase's neighbours;
    each phrase is then heard from the end of the word before it to the start
    of the word after it, so a wrong default only costs its own slot.
    """
    # ponytail: the wav2vec2 pass covers the whole chunk (~4 s of CPU per 60 s)
    # even to hear one number (~0.3 s for its window plus _CONTEXT). For
    # requests without word times, run it over each phrase's window from
    # Parakeet's own times instead, if the cost matters.
    chosen = [phrase.options[0] for phrase in found]
    said = list(texts)  # spoken.spoken_words(texts), without finding the phrases again
    for phrase in found:
        said[phrase.start : phrase.end] = phrase.readings[0]
    spans = chunk.spans(said) or [None] * len(texts)
    for index, phrase in enumerate(found):
        rivals = spoken.rivals(texts, phrase)
        if len(phrase.options) < 2 and not rivals:
            continue
        start = next((s[1] for s in reversed(spans[: phrase.start]) if s is not None), 0.0)
        end = next((s[0] for s in spans[phrase.end :] if s is not None), float("inf"))
        own = chunk.scores(phrase.options, start, end)
        choice = phrase.options[own.index(max(own))]
        if rivals and max(own) > float("-inf"):
            theirs = chunk.scores(rivals, start, end)
            if max(theirs) > max(own) + _CORRECTION_MARGIN:
                choice = rivals[theirs.index(max(theirs))]
        chosen[index] = _each_zero(choice, chunk, start, end)
    return chosen


def _each_zero(said: str, chunk: aligner.ChunkAligner, start: float, end: float) -> str:
    """Speakers mix their zeros within one number ("nine zero oh one"): try
    each zero the other ways too, one at a time, punctuation kept. The zero
    already said comes first, so it stays when the audio can't tell."""
    tokens = said.split()
    for slot, token in enumerate(tokens):
        if match := _ZERO.fullmatch(token):
            opening, said_zero, closing = match.groups()
            zeros = [said_zero, *(zero for zero in spoken.ZERO_WORDS if zero != said_zero)]
            ways = [[*tokens[:slot], f"{opening}{zero}{closing}", *tokens[slot + 1 :]] for zero in zeros]
            tokens = ways[chunk.best([" ".join(way) for way in ways], start, end)]
    return " ".join(tokens)


def _speak_numbers(
    words: List[Dict[str, Any]], chunk: Callable[[], Optional[aligner.ChunkAligner]]
) -> Optional[List[Dict[str, Any]]]:
    """One chunk's words with numbers, money and units said out ("$5" -> "five
    dollars"), or None if it has none (PARAKEET_SPOKEN_NUMBERS).

    Each number is a phrase with the words that change how it is said
    (spoken.phrases). The audio, via `chunk()` (called only when a phrase has
    something to decide, as it loads the aligner), picks how it was said
    ("£2.10": "two pounds ten", "two pounds and ten pence", ...) and may, by a
    clear margin, correct a misheard number; without it the first reading is
    used. A phrase shares its time span out by length among its spoken words;
    when the aligner runs next it re-times each of them from the audio.
    """
    texts = [w["word"] for w in words]
    found = _phrases(texts, most=2)  # every reading only if the audio will choose
    if not found:
        return None
    chosen = [phrase.options[0] for phrase in found]
    if any(_undecided(texts, phrase) for phrase in found) and (ear := chunk()) is not None:
        chosen = _choose_readings(texts, _phrases(texts), ear)
    out: List[Dict[str, Any]] = []
    done = 0
    for phrase, text in zip(found, chosen):
        out += words[done : phrase.start]
        if spoken.starts_sentence(texts[phrase.start - 1] if phrase.start else None):
            text = spoken.capitalize(text)
        out += _spread(text.split(), words[phrase.start]["start"], words[phrase.end - 1]["end"])
        done = phrase.end
    return out + words[done:]


def _spread(parts: List[str], start: float, end: float) -> List[Dict[str, Any]]:
    """Words over the span [start, end], each a share by its length."""
    share = (end - start) / sum(len(part) for part in parts)
    out = []
    for position, part in enumerate(parts):
        stop = end if position == len(parts) - 1 else start + share * len(part)
        out.append({"start": start, "end": stop, "word": part})
        start = stop
    return out


def _group_words(info: Dict[str, Any]) -> List[Tuple[str, float, float]]:
    """One chunk's BPE pieces (from _extract) as (word, first_ts, last_ts).

    A piece starting with the word marker ("\u2581" or a plain space, depending
    on export) opens a new word. Parakeet emits the marker as a token of its
    own before digits and currency signs (" was", " ", "\u00a3", "1"...): it
    opens the next piece's word.
    """
    grouped: List[Tuple[str, float, float]] = []
    pending_break = False
    for token, timestamp in zip(info["tokens"], info["timestamps"]):
        piece = token.replace("\u2581", " ")
        starts_word = pending_break or piece.startswith(" ")
        piece = piece.strip()
        pending_break = starts_word and not piece
        if not piece:
            continue
        if grouped and not starts_word:
            word, first_ts, _last_ts = grouped[-1]
            grouped[-1] = (word + piece, first_ts, timestamp)
        else:
            grouped.append((piece, timestamp, timestamp))
    return grouped


def _stitch(
    prepared: _PreparedAudio,
    results: Sequence[Any],
    *,
    align: bool = False,
    speak: bool = False,
    language: Optional[str] = None,
    aligner_choice: Optional[Tuple[str, str]] = None,
) -> Tuple[str, List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Chunk results -> (text, segments, words).

    `align` re-times words with the request's aligner (`aligner_choice`, its
    name and quantization); `speak` says numbers out (PARAKEET_SPOKEN_NUMBERS),
    hearing which reading was said through that aligner if there is one. Either
    may run the aligner's model: call it through _stitch_request, which keeps
    that off the event loop.
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

        grouped = _group_words(info)
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
                heard.append(aligner.for_chunk(wav, language, *aligner_choice) if aligner_choice else None)
            return heard[0]

        said = None
        if speak:
            try:
                said = _speak_numbers(chunk_words, chunk)
            except Exception:  # it refines a finished transcript: never a 500 for it
                logger.exception("spoken numbers failed; keeping Parakeet's text")
        if said is not None:
            # Text and words are rewritten together so they always agree.
            chunk_words = said
            segments[-1]["segment"] = _clean_text(" ".join(w["word"] for w in chunk_words))

        # A text-only result (Whisper) has no tokens, so no word spans exist to
        # align. When alignment will run, split the transcript into words with
        # placeholder spans for the aligner to re-time from the audio.
        if align and not chunk_words and info["text"]:
            chunk_words = _spread(info["text"].split(), chunk_start, chunk_end)

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


def _needs_aligner(
    results: Sequence[Any],
    *,
    align: bool = False,
    speak: bool = False,
    language: Optional[str] = None,
    aligner_choice: Optional[Tuple[str, str]] = None,
) -> bool:
    """Whether _stitch, with these flags, runs the aligner's model: for word
    times, or to hear how a number was said (a phrase with something to
    decide). The same words and phrases as _stitch's, so they agree; two
    readings of a phrase are enough to know it has a choice."""
    if align:
        return True
    if not (speak and aligner_choice):
        return False  # for_chunk answers None without loading anything
    try:
        for result in results:
            texts = [word for word, _first, _last in _group_words(_extract(result))]
            if any(_undecided(texts, phrase) for phrase in _phrases(texts, most=2)):
                return True
    except Exception:  # can't tell: _stitch will log it and keep Parakeet's text
        return True
    return False


async def _stitch_request(
    request: Request, prepared: _PreparedAudio, results: Sequence[Any], **flags: Any
) -> Tuple[str, List[Dict[str, Any]], List[Dict[str, Any]]]:
    """_stitch(prepared, results, **flags), off the event loop unless it is
    plain text work (no word times, no digit to say).

    Saying numbers reads their readings back through number_parse, which is
    CPU work: it, and deciding whether the aligner's model is needed, run on
    the audio pool. With the model (a second ONNX model over the audio) it runs
    on the one-worker align pool, off the audio pool so it never holds up other
    requests' decoding; only those requests queue behind other alignment."""
    stitch = functools.partial(_stitch, prepared, results, **flags)
    says_numbers = flags.get("speak") and any(
        char.isdigit() for result in results for char in str(getattr(result, "text", result))
    )
    if not (flags.get("align") or says_numbers):
        return stitch()
    loop, state = asyncio.get_running_loop(), request.app.state
    needs = flags.get("align") or await loop.run_in_executor(
        state.audio_pool, functools.partial(_needs_aligner, results, **flags)
    )
    return await loop.run_in_executor(state.align_pool if needs else state.audio_pool, stitch)


async def _infer_prepared(request: Request, prepared: _PreparedAudio, model_key: str):
    worker = request.app.state.worker
    if worker is None or not getattr(request.app.state, "ready", False):
        raise HTTPException(status_code=503, detail="Model is not ready")
    return await worker.submit_many(prepared.pieces, model_key)


_MODEL_CREATED = 1785888000  # catalog introduction (2026-08-05), fixed for stable output


def _model_card(name: str) -> Dict[str, Any]:
    config = MODEL_CONFIGS[name]
    return {
        "id": name,
        "object": "model",
        "created": _MODEL_CREATED,
        # The model's author; each quantization is someone's ONNX export of it.
        "owned_by": "openai" if _family(name) == "whisper" else "nvidia",
        "language": config["languages"],
        "quantizations": list(config["quantizations"]),
        "task": "automatic-speech-recognition",
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


def _aligner_card(name: str) -> Dict[str, Any]:
    config = ALIGNER_CONFIGS[name]
    return {
        "id": name,
        "object": "aligner",
        "created": _MODEL_CREATED,
        "language": list(config["languages"]),
        "quantizations": list(config["quantizations"]),
        "default_quantization": config["default_quantization"],
        "task": "forced-alignment",
    }


@router.get("/v1/aligners")
def list_aligners():
    """The aligners a request may name (`aligner`), as /v1/models lists models."""
    return {"object": "list", "data": [_aligner_card(name) for name in ALIGNER_CONFIGS]}


@router.get("/v1/aligners/{aligner_id:path}")
def retrieve_aligner(aligner_id: str):
    name = aligner_id.strip().lower()
    if name not in ALIGNER_CONFIGS:
        raise HTTPException(status_code=404, detail=f"Aligner {aligner_id!r} not found")
    return _aligner_card(name)


@router.get("/health")
def health(request: Request):
    ready = bool(getattr(request.app.state, "ready", False))
    return {
        "status": "healthy" if ready else "starting",
        "ready": ready,
        "models": list(MODEL_CONFIGS.keys()),
        "loaded": loaded_models(),
        "cpu": CPU_INFO,
        "aligner": aligner.status(),
    }


@router.get("/healthz")
def healthz(request: Request):
    if not getattr(request.app.state, "ready", False):
        raise HTTPException(status_code=503, detail="not ready")
    return {"status": "ok"}


def _validate_aligner(
    name: Optional[str], quantization: Optional[str], language: Optional[str]
) -> Optional[Tuple[str, str]]:
    """The request's aligner as (name, quantization), or None if it names none:
    in the catalog, at one of its quantizations (else its
    default_quantization), aligning the request's language if it has one
    (none: nothing is aligned, as with PARAKEET_ALIGN_DEFAULT_LANGUAGE empty)."""
    if name is None:
        if quantization is not None:
            raise HTTPException(status_code=400, detail="aligner_quantization needs an aligner")
        return None
    normalized = name.strip().lower()
    if normalized not in ALIGNER_CONFIGS:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown aligner {name!r}. Available aligners: {sorted(ALIGNER_CONFIGS)}",
        )
    spec = ALIGNER_CONFIGS[normalized]
    quant = (quantization or spec["default_quantization"]).strip().lower()
    if quant not in spec["quantizations"]:
        raise HTTPException(
            status_code=400,
            detail=f"Aligner {normalized!r} has no {quant!r} quantization. Available: {list(spec['quantizations'])}",
        )
    if aligner.language_code(language) and not aligner.aligns(normalized, language):
        raise HTTPException(
            status_code=400,
            detail=f"Aligner {normalized!r} does not align {aligner.language_code(language)!r}; "
            f"it aligns {list(spec['languages'])}",
        )
    return normalized, quant


def _whisper_language_known(model_name: str, language: Optional[str]) -> bool:
    """Whether a Whisper request's language is *known*, so its words may be
    aligned: an English-only model, or an explicit `language` — never inferred
    from a multilingual model's auto-detection, which could be any language
    and would mis-time the audio."""
    return MODEL_CONFIGS[model_name]["languages"] == ["en"] or (language or "").strip().lower() not in ("", "auto")


def _speaks(spoken_numbers: Optional[bool], language: Optional[str]) -> bool:
    """Say numbers in words: the request's `spoken_numbers`, else the server's
    PARAKEET_SPOKEN_NUMBERS; English only."""
    wanted = SPOKEN_NUMBERS if spoken_numbers is None else spoken_numbers
    return wanted and aligner.language_code(language) == "en"


@router.post("/v1/audio/transcriptions")
async def transcribe(
    request: Request,
    file: UploadFile = File(...),
    model: str = Form(...),
    quantization: Optional[str] = Form(None),
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
    spoken_numbers: Optional[bool] = Form(None),
    aligner_name: Optional[str] = Form(None, alias="aligner"),
    aligner_quantization: Optional[str] = Form(None),
):
    del prompt, temperature  # accepted for OpenAI client compatibility
    model_name = _validate_model(model)
    model_key = _variant(model_name, quantization)
    family = _family(model_name)
    target_sec, max_sec, min_sec = _chunk_bounds(model_name)
    output_format = _validate_format(response_format)
    granularities = set(timestamp_granularities or []) | set(
        timestamp_granularities_plain or []
    )
    # Naming an aligner is asking for aligned word times; there is no default.
    choice = _validate_aligner(aligner_name, aligner_quantization, language)
    speak = _speaks(spoken_numbers, language)
    # Word timestamps: Parakeet emits them from its TDT tokens. Whisper returns
    # text only, so its words come purely from forced-aligning the transcript,
    # and only when the request names an aligner and its language is known —
    # never guessed for a multilingual Whisper request.
    want_words = output_format == "verbose_json" and "word" in granularities and (
        family == "parakeet"
        or (family == "whisper" and choice is not None and _whisper_language_known(model_name, language))
    )
    raw = await _read_upload_limited(file)

    started = time.perf_counter()
    prepared = await _prepare_in_pool(request, raw, target_sec, max_sec, min_sec)
    decode_ms = (time.perf_counter() - started) * 1000

    infer_started = time.perf_counter()
    results = await _infer_prepared(request, prepared, model_key)
    infer_ms = (time.perf_counter() - infer_started) * 1000

    stitch_started = time.perf_counter()
    full_text, segments, words = await _stitch_request(
        request,
        prepared,
        results,
        align=want_words and choice is not None,
        speak=speak,
        language=language,
        aligner_choice=choice,
    )
    stitch_ms = (time.perf_counter() - stitch_started) * 1000

    logger.info(
        "transcribe model=%s dur=%.2fs chunks=%d decode=%.0fms infer=%.0fms "
        "stitch=%.0fms total=%.0fms",
        model_key,
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
    model: str = Form(...),
    quantization: Optional[str] = Form(None),
    spoken_numbers: Optional[bool] = Form(None),
    aligner_name: Optional[str] = Form(None, alias="aligner"),
    aligner_quantization: Optional[str] = Form(None),
):
    if not files:
        raise HTTPException(status_code=400, detail="No files provided")
    if len(files) > MAX_BATCH_FILES:
        raise HTTPException(
            status_code=413,
            detail=f"Batch contains {len(files)} files; limit is {MAX_BATCH_FILES}",
        )
    model_name = _validate_model(model)
    model_key = _variant(model_name, quantization)
    choice = _validate_aligner(aligner_name, aligner_quantization, None)  # no `language`: the default
    target_sec, max_sec, min_sec = _chunk_bounds(model_name)
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
        loop.run_in_executor(
            request.app.state.audio_pool,
            _prepare_audio,
            raw,
            target_sec,
            max_sec,
            min_sec,
        )
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
    flat_results = await worker.submit_many(flattened, model_key)

    # The batch endpoint takes no `language`: the default decides.
    speak = _speaks(spoken_numbers, None)
    cursor = 0
    response_items = []
    for filename, prepared in zip(filenames, prepared_files):
        count = len(prepared.pieces)
        item_results = flat_results[cursor : cursor + count]
        cursor += count
        text, _segments, _words = await _stitch_request(
            request, prepared, item_results, speak=speak, aligner_choice=choice
        )
        response_items.append(
            {"filename": filename, "text": text, "duration": prepared.duration}
        )

    if cursor != len(flat_results):
        raise RuntimeError("inference result accounting mismatch")
    return {"results": response_items, "batch_size": len(response_items)}
