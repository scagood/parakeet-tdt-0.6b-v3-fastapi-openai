"""Startup warm-up pass.

ORT defers kernel selection to the first inference, so the warm-up moves that
cost ahead of the readiness probe. It must never be able to keep an otherwise
healthy replica from serving.
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
async def test_warmup_swallows_inference_errors():
    async def submit(_wav, _model_name):
        raise RuntimeError("model exploded")

    # Must return normally: startup continues even when warm-up fails.
    await main._warmup(_app_with_worker(submit))


@pytest.mark.asyncio
async def test_warmup_gives_up_on_a_stuck_worker(monkeypatch):
    monkeypatch.setattr(main, "_WARMUP_TIMEOUT_SEC", 0.01)
    started = asyncio.Event()

    async def submit(_wav, _model_name):
        started.set()
        await asyncio.sleep(60)

    await main._warmup(_app_with_worker(submit))

    assert started.is_set()
