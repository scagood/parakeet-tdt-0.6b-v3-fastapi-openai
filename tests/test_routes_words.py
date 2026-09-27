"""The /v1/audio/transcriptions word-timestamp path, with Parakeet and the aligner faked.

The handler is called directly (CI has no httpx for TestClient); everything
between the form fields and the JSON body is the real code.
"""
from __future__ import annotations

import io
import json
import threading
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest
from fastapi import UploadFile

from parakeet_service import aligner, routes
from parakeet_service.config import TARGET_SR

WORDS = ["hello", "world"]


class _Worker:
    async def submit_many(self, pieces, _model_name):
        return [
            SimpleNamespace(text="hello world", tokens=[" hello", " world"], timestamps=[0.0, 0.8])
            for _ in pieces
        ]


@pytest.fixture
def calls(monkeypatch):
    """Record aligner calls; the fake aligner re-times words to 0.1 s + 1.5 s each."""
    recorded = []

    def fake_align(chunk_wav, words, language=None):
        recorded.append({"words": words, "language": language, "thread": threading.current_thread().name})
        return [(0.1 + 1.5 * i, 0.4 + 1.5 * i) for i in range(len(words))]

    async def fake_prepare(_request, _raw):
        return routes._PreparedAudio(
            waveform=None, ranges=[(0, 2 * TARGET_SR)], pieces=["chunk"], duration=2.0
        )

    monkeypatch.setattr(aligner, "align_words", fake_align)
    monkeypatch.setattr(aligner, "ALIGN_WORDS", True)
    monkeypatch.setattr(aligner, "ALIGN_DEFAULT_LANGUAGE", "en")
    monkeypatch.setattr(routes, "_prepare_in_pool", fake_prepare)
    return recorded


async def _transcribe(response_format="verbose_json", granularity="word", language=None):
    align_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="align")
    state = SimpleNamespace(worker=_Worker(), ready=True, audio_pool=None, align_pool=align_pool)
    try:
        response = await routes.transcribe(
            request=SimpleNamespace(app=SimpleNamespace(state=state)),
            file=UploadFile(io.BytesIO(b"audio"), filename="a.wav"),
            model=None,
            response_format=response_format,
            timestamp_granularities=[granularity] if granularity else None,
            timestamp_granularities_plain=None,
            language=language,
            prompt=None,
            temperature=None,
        )
    finally:
        align_pool.shutdown()
    return json.loads(response.body) if response_format.endswith("json") else response.body


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
async def test_alignment_only_runs_when_words_are_returned(calls, response_format, granularity):
    await _transcribe(response_format=response_format, granularity=granularity)
    assert calls == []


@pytest.mark.asyncio
async def test_disabled_alignment_keeps_model_times(calls, monkeypatch):
    monkeypatch.setattr(aligner, "ALIGN_WORDS", False)
    body = await _transcribe()
    assert calls == [] and body["words"][1]["start"] == 0.8


def test_health_reports_aligner_state(monkeypatch):
    monkeypatch.setattr(aligner, "status", lambda: {"en": "failed"})
    request = SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(ready=True)))
    assert routes.health(request)["aligner"] == {"en": "failed"}
