"""FastAPI application factory and resource lifespan."""
from __future__ import annotations

import asyncio
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager

from fastapi import FastAPI

from .batchworker import build_worker
from .config import AUDIO_WORKERS, WARMUP, WARMUP_SEC, logger
from .model import default_model_name, get_model, load_model, warmup_waveform
from .routes import router

# Generous: a warm-up that takes this long signals a stuck worker, not a slow
# one. It only bounds the wait — a timeout is logged and startup continues.
_WARMUP_TIMEOUT_SEC = 120.0


def _shutdown_pool(pool: ThreadPoolExecutor) -> None:
    pool.shutdown(wait=True, cancel_futures=True)


async def _warmup(app: FastAPI) -> None:
    """Push one synthetic chunk through the real inference path.

    Best-effort: a warm-up failure must not keep an otherwise healthy replica
    from serving, so every error is logged and swallowed.
    """
    started = time.perf_counter()
    try:
        await asyncio.wait_for(
            app.state.worker.submit(warmup_waveform(), default_model_name()),
            timeout=_WARMUP_TIMEOUT_SEC,
        )
    except asyncio.TimeoutError:
        logger.warning("Warm-up timed out after %.0fs; continuing", _WARMUP_TIMEOUT_SEC)
    except Exception:
        logger.warning("Warm-up pass failed; continuing", exc_info=True)
    else:
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
        version="1.4.0",  # x-release-please-version
        description=(
            "High-throughput OpenAI-compatible ASR service for "
            "Parakeet TDT 0.6B v3."
        ),
        lifespan=lifespan,
    )
    app.include_router(router)
    return app


app = create_app()
