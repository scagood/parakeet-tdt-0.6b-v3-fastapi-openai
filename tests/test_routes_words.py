"""The /v1/audio/transcriptions word-timestamp and spoken-number paths (and the
batch endpoint), with Parakeet and the aligner faked.

The handlers are called directly (CI has no httpx for TestClient); everything
between the form fields and the response body is the real code. The fake
audio's bytes are the transcript Parakeet "hears".
"""
from __future__ import annotations

import inspect
import io
import json
import threading
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest
from fastapi import UploadFile, params

from parakeet_service import aligner, routes
from parakeet_service.config import TARGET_SR

WORDS = ["hello", "world"]


class _Worker:
    async def submit_many(self, pieces, _model_name):
        return [
            SimpleNamespace(
                text=piece,
                tokens=[" " + word for word in piece.split()],
                timestamps=[0.8 * i for i in range(len(piece.split()))],
            )
            for piece in pieces
        ]


def _prepared(raw, *_bounds):
    return routes._PreparedAudio(
        waveform=None, ranges=[(0, 2 * TARGET_SR)], pieces=[raw.decode()], duration=2.0
    )


@pytest.fixture
def calls(monkeypatch):
    """Record aligner calls; the fake aligner re-times words to 0.1 s + 1.5 s each
    and hears every reading equally well (so the first is kept)."""
    recorded = []

    class FakeChunk:
        def __init__(self, language):
            self.language = language

        def spans(self, words):
            recorded.append(
                {"words": list(words), "language": self.language, "thread": threading.current_thread().name}
            )
            return [(0.1 + 1.5 * i, 0.4 + 1.5 * i) for i in range(len(words))]

        def scores(self, options, _start, _end):
            recorded.append({"options": list(options), "thread": threading.current_thread().name})
            return [0.0] * len(options)

        def best(self, _options, _start, _end):
            return 0

    def fake_for_chunk(_wav, language):
        return FakeChunk(language) if aligner.supports(language) else None

    async def fake_prepare(_request, raw, *_bounds):
        return _prepared(raw)

    monkeypatch.setattr(aligner, "for_chunk", fake_for_chunk)
    monkeypatch.setattr(routes, "ALIGN_WORDS", False)
    monkeypatch.setattr(aligner, "ALIGN_DEFAULT_LANGUAGE", "en")
    monkeypatch.setattr(routes, "_prepare_in_pool", fake_prepare)
    monkeypatch.setattr(routes, "_prepare_audio", _prepared)
    return recorded


def _pool(thread):
    """Where a thread belongs: "align" or "audio" pool, else "inline" (the event loop)."""
    return next((pool for pool in ("align", "audio") if thread.startswith(pool)), "inline")


@pytest.fixture
def stitched(monkeypatch):
    """Where each _stitch ran: "align", "audio" or "inline" (see _pool)."""
    threads = []
    stitch = routes._stitch

    def recording(*args, **kwargs):
        threads.append(_pool(threading.current_thread().name))
        return stitch(*args, **kwargs)

    monkeypatch.setattr(routes, "_stitch", recording)
    return threads


@pytest.fixture
def speak(monkeypatch):
    monkeypatch.setattr(routes, "SPOKEN_NUMBERS", True)


def _state():
    return SimpleNamespace(
        worker=_Worker(),
        ready=True,
        audio_pool=ThreadPoolExecutor(max_workers=2, thread_name_prefix="audio"),
        align_pool=ThreadPoolExecutor(max_workers=1, thread_name_prefix="align"),
    )


async def _transcribe(
    response_format="verbose_json",
    granularity="word",
    language=None,
    text="hello world",
    align_words=True,
    spoken_numbers=None,
):
    state = _state()
    try:
        response = await routes.transcribe(
            request=SimpleNamespace(app=SimpleNamespace(state=state)),
            file=UploadFile(io.BytesIO(text.encode()), filename="a.wav"),
            model="parakeet-v3",
            quantization=None,
            response_format=response_format,
            timestamp_granularities=[granularity] if granularity else None,
            timestamp_granularities_plain=None,
            language=language,
            prompt=None,
            temperature=None,
            align_words=align_words,
            spoken_numbers=spoken_numbers,
        )
    finally:
        state.audio_pool.shutdown()
        state.align_pool.shutdown()
    return json.loads(response.body) if response_format.endswith("json") else response.body.decode()


async def _batch(*texts, spoken_numbers=None):
    state = _state()
    try:
        body = await routes.transcribe_batch(
            request=SimpleNamespace(app=SimpleNamespace(state=state)),
            files=[UploadFile(io.BytesIO(text.encode()), filename=f"{i}.wav") for i, text in enumerate(texts)],
            model="parakeet-v3",
            quantization=None,
            spoken_numbers=spoken_numbers,
        )
    finally:
        state.audio_pool.shutdown()
        state.align_pool.shutdown()
    return [item["text"] for item in body["results"]]


@pytest.mark.asyncio
async def test_word_request_is_aligned_on_the_align_pool(calls):
    body = await _transcribe(language="en-US")
    assert [c["words"] for c in calls] == [WORDS]
    assert calls[0]["language"] == "en-US"
    assert calls[0]["thread"].startswith("align")  # not the audio pool, not the loop
    assert [(w["word"], w["start"], w["end"]) for w in body["words"]] == [
        ("hello", 0.1, 0.4),
        ("world", 1.6, 1.9),
    ]
    assert body["language"] == "en-US"
    # the segment was widened to cover the re-timed last word
    assert body["segments"][0]["end"] >= body["words"][-1]["end"]


@pytest.mark.asyncio
async def test_unsupported_language_keeps_model_times(calls):
    body = await _transcribe(language="fr")
    assert calls == []
    assert body["words"][0]["start"] == 0.0 and body["words"][1]["start"] == 0.8
    assert body["language"] == "fr"


@pytest.mark.asyncio
async def test_missing_language_reports_auto_and_uses_the_default(calls):
    body = await _transcribe(language=None)
    assert len(calls) == 1 and body["language"] == "auto"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("response_format", "granularity"),
    [("verbose_json", None), ("verbose_json", "segment"), ("json", "word"), ("srt", "word")],
)
async def test_alignment_only_runs_when_words_are_returned(calls, stitched, response_format, granularity):
    await _transcribe(response_format=response_format, granularity=granularity)
    assert calls == []
    assert stitched == ["inline"]  # nothing to align: no queue


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("server_default", "align_words", "aligned"),
    [(False, None, False), (False, True, True), (True, None, True), (True, False, False)],
)
async def test_the_request_opts_in_or_out_else_the_server_default(
    calls, monkeypatch, server_default, align_words, aligned
):
    monkeypatch.setattr(routes, "ALIGN_WORDS", server_default)
    body = await _transcribe(align_words=align_words)
    assert bool(calls) == aligned
    assert (body["words"][1]["start"] == 1.6) == aligned  # else Parakeet's 0.8


@pytest.mark.parametrize(
    ("handler", "name"),
    [
        (routes.transcribe, "align_words"),
        (routes.transcribe, "spoken_numbers"),
        (routes.transcribe_batch, "spoken_numbers"),
    ],
)
def test_the_switches_are_optional_form_fields(handler, name):
    # the handlers are called directly here, so pin what FastAPI will parse
    field = inspect.signature(handler).parameters[name]
    assert isinstance(field.default, params.Form) and field.default.default is None
    assert field.annotation in ("Optional[bool]", "bool | None")


def test_health_reports_aligner_state(monkeypatch):
    monkeypatch.setattr(aligner, "status", lambda: {"en": "failed"})
    request = SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(ready=True)))
    assert routes.health(request)["aligner"] == {"en": "failed"}


# --------------------------------------------------------------------------- #
# Spoken numbers (PARAKEET_SPOKEN_NUMBERS)
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_spoken_numbers_are_off_by_default(calls):
    body = await _transcribe(response_format="json", text="It cost $5 today.")
    assert body["text"] == "It cost $5 today."
    assert await _batch("It cost $5 today.") == ["It cost $5 today."]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("server_default", "spoken_numbers", "said"),
    [(False, None, False), (False, True, True), (True, None, True), (True, False, False)],
)
async def test_the_request_says_numbers_or_not_else_the_server_default(
    calls, monkeypatch, server_default, spoken_numbers, said
):
    monkeypatch.setattr(routes, "SPOKEN_NUMBERS", server_default)
    text = "It cost five dollars today." if said else "It cost $5 today."
    body = await _transcribe(response_format="json", text="It cost $5 today.", spoken_numbers=spoken_numbers)
    assert body["text"] == text
    assert await _batch("It cost $5 today.", spoken_numbers=spoken_numbers) == [text]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("language", "text"),
    [
        (None, "It cost five dollars today."),  # PARAKEET_ALIGN_DEFAULT_LANGUAGE
        ("en", "It cost five dollars today."),
        ("en-GB", "It cost five dollars today."),
        ("English", "It cost five dollars today."),
        ("fr", "It cost $5 today."),
        ("de-DE", "It cost $5 today."),
    ],
)
async def test_spoken_numbers_are_english_only(calls, speak, language, text):
    body = await _transcribe(response_format="json", language=language, text="It cost $5 today.")
    assert body["text"] == text


@pytest.mark.asyncio
async def test_spoken_numbers_follow_the_default_language(calls, speak, monkeypatch):
    monkeypatch.setattr(aligner, "ALIGN_DEFAULT_LANGUAGE", "")
    assert (await _transcribe(response_format="json", text="It cost $5."))["text"] == "It cost $5."
    assert await _batch("It cost $5.") == ["It cost $5."]
    assert (await _transcribe(response_format="json", language="en", text="It cost $5."))["text"] == (
        "It cost five dollars."
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("response_format", "granularity"),
    [
        ("json", None), ("text", None), ("srt", None), ("vtt", None),
        ("verbose_json", "segment"), ("verbose_json", "word"),
    ],
)
async def test_spoken_numbers_reach_every_response_format(calls, speak, response_format, granularity):
    body = await _transcribe(response_format=response_format, granularity=granularity, text="It cost $5 today.")
    if response_format == "verbose_json":
        assert body["text"] == body["segments"][0]["text"] == "It cost five dollars today."
        if granularity == "word":
            assert [w["word"] for w in body["words"]] == ["It", "cost", "five", "dollars", "today."]
    else:
        body = body["text"] if response_format == "json" else body
        assert "It cost five dollars today." in body and "$5" not in body


@pytest.mark.asyncio
async def test_batch_says_numbers_too(calls, speak):
    texts = await _batch("It cost $5 today.", "No numbers here.")
    assert texts == ["It cost five dollars today.", "No numbers here."]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("text", "pool"),
    [
        ("No numbers here.", "inline"),
        ("No numbers in an MP3 here.", "audio"),  # a digit: phrases are looked for off the loop
        ("It opens at 6pm.", "audio"),  # one reading, no rivals: nothing to hear
        ("It opens at 6 pm.", "audio"),  # nor spaced
        ("That'll be £2.10 please.", "align"),
    ],
)
async def test_spoken_numbers_queue_on_the_align_pool_only_to_hear_one(calls, stitched, speak, text, pool):
    await _transcribe(response_format="json", text=text)
    await _batch(text)
    assert stitched == [pool, pool]
    assert all(call["thread"].startswith("align") for call in calls)
    assert bool(calls) == (pool == "align")


@pytest.mark.asyncio
async def test_numbers_are_never_read_on_the_event_loop(calls, speak, monkeypatch):
    # Reading every number's readings back through number_parse is CPU work
    # (seconds for a long request of codes): deciding and saying both run off the loop.
    threads = []
    phrases = routes.spoken.phrases

    def recording(*args, **kwargs):
        threads.append(_pool(threading.current_thread().name))
        return phrases(*args, **kwargs)

    monkeypatch.setattr(routes.spoken, "phrases", recording)
    text = "Order 001100110011 cost £2.10 at 6pm."
    await _transcribe(response_format="json", text=text)
    await _batch(text)
    assert threads and "inline" not in threads


@pytest.mark.asyncio
async def test_a_spoken_numbers_bug_is_never_a_500(calls, speak, monkeypatch, caplog):
    def broken(_words, **_options):
        raise RuntimeError("a bug in spoken.phrases")

    monkeypatch.setattr(routes.spoken, "phrases", broken)
    body = await _transcribe(text="It cost $5 today.")
    assert body["text"] == "It cost $5 today."
    assert [w["word"] for w in body["words"]] == ["It", "cost", "$5", "today."]
    assert await _batch("It cost $5 today.") == ["It cost $5 today."]
    assert "spoken numbers failed" in caplog.text


# --- Whisper word alignment (Whisper returns text only; words come from the
# --- English aligner, and only when English is known) --------------------------

class _WhisperWorker:
    """Whisper returns a bare transcript string per chunk: no tokens."""

    async def submit_many(self, pieces, _model_name):
        return list(pieces)


async def _transcribe_whisper(model, *, language=None, align_words=True, text="hello world"):
    state = _state()
    state.worker = _WhisperWorker()
    try:
        response = await routes.transcribe(
            request=SimpleNamespace(app=SimpleNamespace(state=state)),
            file=UploadFile(io.BytesIO(text.encode()), filename="a.wav"),
            model=model,
            quantization=None,
            response_format="verbose_json",
            timestamp_granularities=["word"],
            timestamp_granularities_plain=None,
            language=language,
            prompt=None,
            temperature=None,
            align_words=align_words,
            spoken_numbers=None,
        )
    finally:
        state.audio_pool.shutdown()
        state.align_pool.shutdown()
    return json.loads(response.body)


@pytest.mark.asyncio
async def test_whisper_english_transcript_is_split_and_aligned(calls):
    # No tokens, so the words come purely from splitting the text and aligning.
    body = await _transcribe_whisper("whisper-base", language="en")
    assert [c["words"] for c in calls] == [WORDS]
    assert [(w["word"], w["start"], w["end"]) for w in body["words"]] == [
        ("hello", 0.1, 0.4),
        ("world", 1.6, 1.9),
    ]


@pytest.mark.asyncio
async def test_whisper_multilingual_without_language_returns_no_words(calls):
    # Auto-detect could be any language; never align a multilingual model blindly.
    body = await _transcribe_whisper("whisper-base", language=None)
    assert calls == []
    assert body["words"] is None


@pytest.mark.asyncio
async def test_whisper_english_only_model_aligns_without_language(calls):
    # A .en model is English by construction, so no language field is needed.
    body = await _transcribe_whisper("whisper-base.en", language=None)
    assert [c["words"] for c in calls] == [WORDS]
    assert body["words"] is not None


@pytest.mark.asyncio
async def test_whisper_words_off_when_align_disabled(calls):
    # Whisper has no native word times, so align_words=false means no words.
    body = await _transcribe_whisper("whisper-base", language="en", align_words=False)
    assert calls == []
    assert body["words"] is None
