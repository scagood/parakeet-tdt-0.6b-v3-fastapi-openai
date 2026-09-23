"""FastAPI application factory and resource lifespan."""
from __future__ import annotations

import asyncio
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager

import numpy as np
from fastapi import FastAPI

from .batchworker import BatchWorker, build_worker
from .config import (
    AUDIO_WORKERS,
    CALIB_SECS,
    USE_GPU,
    VRAM_PER_SEC_EXPLICIT,
    WARMUP,
    WARMUP_SEC,
    WARMUP_TIMEOUT_SEC,
    logger,
    query_gpu_mib,
)
from .model import default_model_name, get_model, load_model, warmup_waveform
from .routes import router


def _shutdown_pool(pool: ThreadPoolExecutor) -> None:
    pool.shutdown(wait=True, cancel_futures=True)


def _exit_without_join(message: str) -> None:
    """Terminate the process while a native call is still running.

    A wedged ORT call cannot be interrupted from Python. Every orderly path
    out — the lifespan's ``worker.stop()`` and the interpreter's own exit —
    joins the executor thread and would hang on it, so leave without one.
    """
    logger.critical(message)
    logging.shutdown()
    os._exit(1)


def _calib_durations() -> list[float]:
    """Clip lengths to push through warm-up: the calibration sweep if the
    operator asked for one, else the single readiness clip."""
    return CALIB_SECS if CALIB_SECS else [WARMUP_SEC]


def _calibration_active() -> bool:
    return USE_GPU != "false" and not VRAM_PER_SEC_EXPLICIT


async def _warmup(app: FastAPI) -> None:
    """Push synthetic chunks through the real inference path.

    Failure is fatal. The chunk is what every real request looks like, so a
    model that cannot run it would 500 every request while passing readiness.
    A timeout is fatal too, and exits the process outright: cancelling the
    wait leaves the ORT call running on a thread nothing can interrupt, so
    the replica could neither serve past it nor shut down cleanly. The
    orchestrator restarts it. ``PARAKEET_WARMUP_TIMEOUT_SEC`` bounds the wait.

    When ``PARAKEET_CALIB_SECS`` lists several durations the clips run ascending
    and the VRAM each consumes is recorded, so the batch memory estimate is
    fitted to the real cost curve rather than one point (see _apply_calibration).
    """
    started = time.perf_counter()
    measure = _calibration_active()
    baseline = query_gpu_mib("memory.used") if measure else None
    points: list[tuple[float, int]] = []
    for seconds in sorted(_calib_durations()):
        try:
            await asyncio.wait_for(
                app.state.worker.submit(warmup_waveform(seconds), default_model_name()),
                timeout=WARMUP_TIMEOUT_SEC,
            )
        except asyncio.TimeoutError:
            message = (
                f"warm-up did not finish within {WARMUP_TIMEOUT_SEC:.0f}s; "
                "raise PARAKEET_WARMUP_TIMEOUT_SEC or set PARAKEET_WARMUP=false"
            )
            _exit_without_join(message)
            raise RuntimeError(message) from None  # only reached when _exit_without_join is stubbed
        except Exception as exc:
            raise RuntimeError("warm-up inference failed") from exc
        # Ascending durations grow the (cache-only) ORT arena monotonically, so
        # each reading minus the baseline is that clip's activation footprint.
        if baseline is not None:
            used = query_gpu_mib("memory.used")
            if used is not None:
                points.append((seconds, used - baseline))
    logger.info("Warm-up completed in %.2fs", time.perf_counter() - started)
    _apply_calibration(app.state.worker, points)


def _fit_cost(points: list[tuple[float, int]]):
    """Fit a per-clip MiB(seconds) curve to (duration, VRAM-delta) samples.

    Returns highest-degree-first polynomial coefficients (numpy.polyval order),
    or None if the samples yield nothing usable. Coefficients are clamped to be
    non-negative so the estimate is monotonic and never predicts freeing memory
    — a conservative bias that errs toward smaller, safer batches.
    """
    usable = [(d, m) for d, m in points if d > 0 and m > 0]
    if not usable:
        return None
    if len(usable) == 1:
        d, m = usable[0]
        return np.array([0.0, m / d, 0.0])
    degree = 2 if len(usable) >= 3 else 1
    ds = np.array([d for d, _ in usable], dtype=float)
    ms = np.array([m for _, m in usable], dtype=float)
    coeffs = np.polyfit(ds, ms, degree)
    coeffs = np.concatenate([np.zeros(3 - coeffs.size), coeffs])  # pad to quadratic
    coeffs = np.clip(coeffs, 0.0, None)
    return coeffs if coeffs.any() else None


def _apply_calibration(worker, points: list[tuple[float, int]]) -> None:
    """Fit the measured VRAM samples and hand the curve to the batch worker."""
    if not points or not isinstance(worker, BatchWorker):
        return
    coeffs = _fit_cost(points)
    if coeffs is None:
        logger.warning(
            "Batch memory calibration skipped: VRAM samples %s are implausible "
            "(another process on the card?); using configured PARAKEET_VRAM_PER_SEC_MIB",
            points,
        )
        return
    worker.recalibrate(coeffs)
    logger.info(
        "Calibrated batch memory from %d point(s) %s: cost[hi..lo]=%s",
        len(points),
        points,
        np.array2string(coeffs, precision=3, suppress_small=True),
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.ready = False
    app.state.worker = None
    app.state.audio_pool = ThreadPoolExecutor(
        max_workers=AUDIO_WORKERS, thread_name_prefix="audio"
    )
    try:
        logger.info("Lifespan startup: loading default model")
        await asyncio.to_thread(load_model)
        app.state.worker = build_worker(get_model)
        await app.state.worker.start()
        if WARMUP and WARMUP_SEC > 0:
            await _warmup(app)
        if isinstance(app.state.worker, BatchWorker):
            logger.info(
                "Batch memory config: %s", app.state.worker.describe(WARMUP_SEC)
            )
        app.state.ready = True
        logger.info("Service ready")
        yield
    finally:
        app.state.ready = False
        logger.info("Lifespan shutdown")
        if app.state.worker is not None:
            await app.state.worker.stop()
        await asyncio.to_thread(_shutdown_pool, app.state.audio_pool)


def create_app() -> FastAPI:
    app = FastAPI(
        title="Parakeet TDT 0.6B v3 (optimized)",
        version="1.5.0",  # x-release-please-version
        description=(
            "High-throughput OpenAI-compatible ASR service for "
            "Parakeet TDT 0.6B v3."
        ),
        lifespan=lifespan,
    )
    app.include_router(router)
    return app


app = create_app()
