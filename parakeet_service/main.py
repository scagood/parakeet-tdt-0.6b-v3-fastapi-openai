"""FastAPI application factory and resource lifespan."""
from __future__ import annotations

import asyncio
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager

from fastapi import FastAPI

from .batchworker import build_worker
from .config import AUDIO_WORKERS, WARMUP, WARMUP_SEC, WARMUP_TIMEOUT_SEC, logger
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


async def _warmup(app: FastAPI) -> None:
    """Push one synthetic chunk through the real inference path.

    Failure is fatal. The chunk is what every real request looks like, so a
    model that cannot run it would 500 every request while passing readiness.
    A timeout is fatal too, and exits the process outright: cancelling the
    wait leaves the ORT call running on a thread nothing can interrupt, so
    the replica could neither serve past it nor shut down cleanly. The
    orchestrator restarts it. ``PARAKEET_WARMUP_TIMEOUT_SEC`` bounds the wait.
    """
    started = time.perf_counter()
    try:
        await asyncio.wait_for(
            app.state.worker.submit(warmup_waveform(), default_model_name()),
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
    logger.info("Warm-up completed in %.2fs", time.perf_counter() - started)


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
        version="2.0.0",  # x-release-please-version
        description=(
            "High-throughput OpenAI-compatible ASR service for "
            "Parakeet TDT 0.6B v3."
        ),
        lifespan=lifespan,
    )
    app.include_router(router)
    return app


app = create_app()
