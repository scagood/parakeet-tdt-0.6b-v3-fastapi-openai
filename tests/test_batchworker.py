from __future__ import annotations

import asyncio

import numpy as np
import pytest

from parakeet_service.batchworker import BatchWorker, InferencePool


class _Model:
    def recognize(self, value):
        if isinstance(value, list):
            return [float(item.sum()) for item in value]
        return float(value.sum())


@pytest.mark.asyncio
async def test_inference_pool_runs_and_rejects_after_stop():
    pool = InferencePool(lambda _name: _Model(), workers=2)
    results = await pool.submit_many(
        [np.ones(3, dtype=np.float32), np.ones(5, dtype=np.float32)], "model"
    )
    assert results == [3.0, 5.0]
    await pool.stop()
    with pytest.raises(RuntimeError, match="stopped"):
        await pool.submit(np.ones(1, dtype=np.float32), "model")


@pytest.mark.asyncio
async def test_batch_worker_batches_same_model():
    worker = BatchWorker(lambda _name: _Model(), max_batch=4, window_ms=20)
    await worker.start()
    try:
        results = await worker.submit_many(
            [np.ones(2, dtype=np.float32), np.ones(4, dtype=np.float32)], "a"
        )
        assert results == [2.0, 4.0]
    finally:
        await worker.stop()


@pytest.mark.asyncio
async def test_batch_worker_preserves_mixed_model_jobs():
    worker = BatchWorker(lambda _name: _Model(), max_batch=4, window_ms=20)
    await worker.start()
    try:
        first, second = await asyncio.gather(
            worker.submit(np.ones(2, dtype=np.float32), "a"),
            worker.submit(np.ones(3, dtype=np.float32), "b"),
        )
        assert (first, second) == (2.0, 3.0)
    finally:
        await worker.stop()


class _RecordingModel:
    """Reports the size of each batch it is handed."""

    def __init__(self, sizes):
        self._sizes = sizes

    def recognize(self, value):
        if isinstance(value, list):
            self._sizes.append(len(value))
            return [float(item.sum()) for item in value]
        self._sizes.append(1)
        return float(value.sum())


# TARGET_SR samples == 1 second, and per_sec_mib=1.0 makes 1s cost 1 MiB, so a
# clip of N seconds costs N MiB — keeps the arithmetic obvious below.
from parakeet_service.config import TARGET_SR


def _clip(seconds: float) -> np.ndarray:
    return np.ones(int(seconds * TARGET_SR), dtype=np.float32)


@pytest.mark.asyncio
async def test_budget_splits_batch_by_estimated_vram():
    sizes: list[int] = []
    # Budget 2.5 MiB at 1 MiB/s with a high count cap: three 1s clips must
    # split 2 / 1, not pack into one batch of 3.
    worker = BatchWorker(
        lambda _name: _RecordingModel(sizes),
        max_batch=8,
        window_ms=50,
        budget_mib=2.5,
        per_sec_mib=1.0,
    )
    await worker.start()
    try:
        results = await worker.submit_many([_clip(1)] * 3, "a")
        assert results == [float(_clip(1).sum())] * 3
        assert sizes == [2, 1]
    finally:
        await worker.stop()


@pytest.mark.asyncio
async def test_budget_runs_lone_oversize_clip():
    sizes: list[int] = []
    worker = BatchWorker(
        lambda _name: _RecordingModel(sizes),
        max_batch=8,
        window_ms=20,
        budget_mib=2.0,
        per_sec_mib=1.0,
    )
    await worker.start()
    try:
        # A single 9s clip estimated well over the 2 MiB budget must still run.
        clip = _clip(9)
        assert await worker.submit(clip, "a") == float(clip.sum())
        assert sizes == [1]
    finally:
        await worker.stop()


def test_describe_reports_implied_clips_per_batch():
    worker = BatchWorker(
        lambda _name: object(), max_batch=32, budget_mib=1000.0, per_sec_mib=10.0
    )
    # 1000 MiB budget, 10 MiB/s * 5s = 50 MiB/clip => 20 clips.
    text = worker.describe(5.0)
    assert "~20 clips/batch" in text
    assert "budget=1000MiB" in text and "cost[hi..lo]=[0, 10, 0]" in text


def test_describe_falls_back_to_count_cap_without_budget():
    worker = BatchWorker(
        lambda _name: object(), max_batch=8, budget_mib=0.0, per_sec_mib=10.0
    )
    assert "count-capped at 8" in worker.describe(5.0)


def test_recalibrate_applies_quadratic_curve():
    worker = BatchWorker(lambda _name: object(), budget_mib=1000.0, per_sec_mib=8.0)
    # cost(sec) = 2*sec^2 + 1*sec + 0
    worker.recalibrate([2.0, 1.0, 0.0])
    # 3s clip => 2*9 + 3 = 21 MiB
    assert worker._est_mib(np.ones(3 * TARGET_SR, dtype=np.float32)) == pytest.approx(21.0)


def test_recalibrate_ignores_zero_curve():
    worker = BatchWorker(lambda _name: object(), budget_mib=1000.0, per_sec_mib=8.0)
    worker.recalibrate([0.0, 0.0, 0.0])
    # Unchanged: default linear 8 MiB/s => 1s clip is 8 MiB.
    assert worker._est_mib(np.ones(TARGET_SR, dtype=np.float32)) == pytest.approx(8.0)


@pytest.mark.asyncio
async def test_batch_worker_rejects_submission_before_start():
    worker = BatchWorker(lambda _name: _Model(), max_batch=2, window_ms=0)
    with pytest.raises(RuntimeError, match="not running"):
        await worker.submit(np.ones(1, dtype=np.float32), "a")
    await worker.stop()
