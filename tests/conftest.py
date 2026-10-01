"""Let route-level tests import parakeet_service without the ONNX stack.

CI installs lightweight deps only, so the unit-test job stays fast and does
not pull the multi-hundred-megabyte ONNX runtime wheels; the docker builds
exercise the real dependency set. parakeet_service.model only touches these
modules at call time, so import-level stubs are enough; tests that exercise
provider probing monkeypatch the attributes they need.
"""
from __future__ import annotations

import os
import sys
import types
from collections import OrderedDict

import pytest

# Tests never download models: a code path that reaches the real aligner loader
# fails fast (and falls back) instead of fetching ~95 MB.
os.environ["HF_HUB_OFFLINE"] = "1"

try:
    import onnx_asr  # noqa: F401
    import onnxruntime  # noqa: F401
except ImportError:
    ort_stub = types.ModuleType("onnxruntime")
    ort_stub.get_available_providers = lambda: ["CPUExecutionProvider"]
    sys.modules.setdefault("onnx_asr", types.ModuleType("onnx_asr"))
    sys.modules.setdefault("onnxruntime", ort_stub)


@pytest.fixture(autouse=True)
def _hermetic_aligner(monkeypatch):
    """Nothing reaches the real aligner loader unless a test fakes it (fake
    aligner.for_chunk, or the hub as test_aligner's loader tests do): it would
    download ~95 MB or quietly load a cached model, so a test that does fails.
    Load state starts empty, so no test sees another's failed or faked load."""
    from parakeet_service import aligner

    reached = []

    def download(repo, filename, revision=None):
        reached.append(filename)
        raise OSError("tests never download models")

    monkeypatch.setitem(sys.modules, "huggingface_hub", types.SimpleNamespace(hf_hub_download=download))
    # CI's onnxruntime stub has no InferenceSession: without one the loader
    # fails before its download, and this guard would never see it.
    monkeypatch.setattr(aligner.ort, "InferenceSession", lambda *a, **k: None, raising=False)
    monkeypatch.setattr(aligner, "_loaded", OrderedDict())
    monkeypatch.setattr(aligner, "_failed_at", {})
    yield
    assert not reached, f"reached the real aligner loader for {reached}: fake aligner.for_chunk"
