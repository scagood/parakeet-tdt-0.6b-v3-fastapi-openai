"""Startup warm-up pass.

ORT defers kernel selection to the first inference, so the warm-up moves that
cost ahead of the readiness probe. Because the warm-up chunk is exactly what a
real request looks like, a replica that cannot run it must not report ready.
"""
from __future__ import annotations

import asyncio
import types

import numpy as np
import pytest

from parakeet_service import main
from parakeet_service.config import TARGET_SR
from parakeet_service.model import warmup_waveform


def _app_with_worker(submit):
    worker = types.SimpleNamespace(submit=submit)
    return types.SimpleNamespace(state=types.SimpleNamespace(worker=worker))


def test_warmup_waveform_shape_and_dtype():
    wav = warmup_waveform(2.0)
    assert wav.dtype == np.float32
    assert wav.shape == (2 * TARGET_SR,)


def test_warmup_waveform_is_deterministic():
    assert np.array_equal(warmup_waveform(0.5), warmup_waveform(0.5))


def test_warmup_waveform_stays_in_range():
    wav = warmup_waveform(1.0)
    assert np.all(np.isfinite(wav))
    assert np.abs(wav).max() <= 1.0


def test_warmup_waveform_is_never_empty():
    assert warmup_waveform(0.0).size >= 1


@pytest.mark.asyncio
async def test_warmup_submits_the_default_model():
    seen = {}

    async def submit(wav, model_name):
        seen["samples"] = wav.size
        seen["model"] = model_name
        return "ok"

    await main._warmup(_app_with_worker(submit))

    assert seen["samples"] > 0
    assert seen["model"]


@pytest.mark.asyncio
async def test_warmup_propagates_inference_errors():
    async def submit(_wav, _model_name):
        raise RuntimeError("model exploded")

    with pytest.raises(RuntimeError, match="warm-up inference failed") as info:
        await main._warmup(_app_with_worker(submit))

    assert isinstance(info.value.__cause__, RuntimeError)
    assert "model exploded" in str(info.value.__cause__)


@pytest.mark.asyncio
async def test_warmup_exits_the_process_on_a_stuck_worker(monkeypatch):
    """A wedged native call cannot be joined, so the timeout path must not
    fall through to an orderly shutdown that would hang on it."""
    monkeypatch.setattr(main, "WARMUP_TIMEOUT_SEC", 0.01)
    exits = []
    monkeypatch.setattr(main, "_exit_without_join", exits.append)
    started = asyncio.Event()

    async def submit(_wav, _model_name):
        started.set()
        await asyncio.sleep(60)

    with pytest.raises(RuntimeError, match="did not finish within"):
        await main._warmup(_app_with_worker(submit))

    assert started.is_set()
    assert len(exits) == 1 and "PARAKEET_WARMUP_TIMEOUT_SEC" in exits[0]


@pytest.mark.asyncio
async def test_lifespan_warms_up_before_reporting_ready(monkeypatch):
    """The point of the warm-up is that it lands *before* readiness.

    Exercises the real lifespan wiring — load, worker start, warm-up, ready —
    with a stub standing in for the ONNX model, since the model itself is not
    available in unit-test environments.
    """
    from fastapi import FastAPI

    ready_during_warmup = []
    app = FastAPI()

    class _StubModel:
        def recognize(self, wav):
            ready_during_warmup.append(bool(getattr(app.state, "ready", False)))
            assert getattr(wav, "size", 0) > 0
            return types.SimpleNamespace(text="", tokens=[], timestamps=[])

    stub = _StubModel()
    monkeypatch.setattr(main, "load_model", lambda *a, **k: stub)
    monkeypatch.setattr(main, "get_model", lambda *a, **k: stub)

    async with main.lifespan(app):
        assert app.state.ready is True
        # Exactly one warm-up pass, and readiness was still False during it.
        assert ready_during_warmup == [False]

    assert app.state.ready is False


@pytest.mark.asyncio
async def test_lifespan_reports_ready_when_warmup_is_disabled(monkeypatch):
    from fastapi import FastAPI

    recognized = []
    app = FastAPI()

    class _StubModel:
        def recognize(self, wav):
            recognized.append(wav)
            return types.SimpleNamespace(text="", tokens=[], timestamps=[])

    stub = _StubModel()
    monkeypatch.setattr(main, "load_model", lambda *a, **k: stub)
    monkeypatch.setattr(main, "get_model", lambda *a, **k: stub)
    monkeypatch.setattr(main, "WARMUP", False)

    async with main.lifespan(app):
        assert app.state.ready is True
        assert recognized == []


@pytest.mark.asyncio
async def test_lifespan_fails_startup_when_warmup_fails(monkeypatch):
    """A model that cannot run the warm-up chunk would 500 every request.

    Failing startup makes that visible to the orchestrator instead of letting
    the replica pass readiness and take traffic.
    """
    from fastapi import FastAPI

    app = FastAPI()

    class _BrokenModel:
        def recognize(self, _wav):
            raise RuntimeError("kernel selection failed")

    stub = _BrokenModel()
    monkeypatch.setattr(main, "load_model", lambda *a, **k: stub)
    monkeypatch.setattr(main, "get_model", lambda *a, **k: stub)

    with pytest.raises(RuntimeError, match="warm-up inference failed"):
        async with main.lifespan(app):
            pytest.fail("lifespan must not reach the serving phase")

    assert app.state.ready is False


def test_warmup_input_matches_a_real_chunk():
    """Closest available proxy for "the model will accept this".

    The real model is not importable in unit-test environments, so instead
    assert the warm-up input is indistinguishable from what `slice_chunks`
    hands the worker on a normal request.
    """
    from parakeet_service.chunker import slice_chunks

    real = slice_chunks(np.zeros(TARGET_SR * 30, dtype=np.float32), [(0, TARGET_SR * 30)])[0]
    warm = warmup_waveform(5.0)

    assert (warm.dtype, warm.ndim) == (real.dtype, real.ndim)
    assert warm.flags.c_contiguous == real.flags.c_contiguous
