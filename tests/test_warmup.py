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
async def test_warmup_runs_each_calibration_duration(monkeypatch):
    monkeypatch.setattr(main, "CALIB_SECS", [30.0, 2.0, 5.0])
    seen = []

    async def submit(wav, _model_name):
        seen.append(round(wav.size / TARGET_SR))
        return "ok"

    await main._warmup(_app_with_worker(submit))

    # Runs every requested length, ascending (arena grows monotonically).
    assert seen == [2, 5, 30]


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


def _batch_worker(per_sec_mib):
    from parakeet_service.batchworker import BatchWorker

    return BatchWorker(
        lambda _name: object(), budget_mib=1000.0, per_sec_mib=per_sec_mib
    )


def test_fit_cost_single_point_is_linear_through_origin():
    # One (5s, 50MiB) sample => 10 MiB/s, no quadratic term.
    coeffs = main._fit_cost([(5.0, 50)])
    assert list(coeffs) == pytest.approx([0.0, 10.0, 0.0])


def test_fit_cost_recovers_a_quadratic_curve():
    # Samples drawn from cost(sec) = 0.5*sec^2 + 2*sec.
    pts = [(s, int(0.5 * s * s + 2 * s)) for s in (2, 5, 30, 60, 300)]
    coeffs = main._fit_cost(pts)
    assert coeffs[0] == pytest.approx(0.5, abs=0.05)  # quadratic term
    assert coeffs[1] == pytest.approx(2.0, abs=1.0)  # linear term


def test_fit_cost_drops_nonpositive_samples():
    # A negative delta (another process freed VRAM) is discarded; the two good
    # points still fit a line.
    coeffs = main._fit_cost([(2.0, -100), (5.0, 50), (30.0, 300)])
    assert coeffs is not None
    assert main_np_polyval(coeffs, 5.0) == pytest.approx(50, abs=5)


def test_fit_cost_returns_none_without_usable_samples():
    assert main._fit_cost([(2.0, -5), (5.0, 0)]) is None


def test_apply_calibration_hands_curve_to_worker():
    worker = _batch_worker(per_sec_mib=8.0)
    main._apply_calibration(worker, [(5.0, 50), (30.0, 300), (300.0, 45000)])
    # Estimate now follows the fitted curve, not the 8 MiB/s default.
    est_300 = worker._est_mib(np.zeros(300 * TARGET_SR, dtype=np.float32))
    assert est_300 == pytest.approx(45000, rel=0.1)


def test_apply_calibration_noop_on_bad_samples():
    worker = _batch_worker(per_sec_mib=8.0)
    main._apply_calibration(worker, [(5.0, -10)])
    # Kept the default: 1s clip is 8 MiB.
    assert worker._est_mib(np.zeros(TARGET_SR, dtype=np.float32)) == pytest.approx(8.0)


def main_np_polyval(coeffs, x):
    import numpy as np

    return float(np.polyval(coeffs, x))


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
