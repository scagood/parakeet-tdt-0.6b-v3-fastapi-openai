"""Let route-level tests import parakeet_service without the ONNX stack.

CI installs lightweight deps only (no onnx-asr/onnxruntime, which lacks
wheels for parts of the tested Python matrix). parakeet_service.model only
touches these modules at call time, so import-level stubs are enough; tests
that exercise provider probing monkeypatch the attributes they need.
"""
from __future__ import annotations

import sys
import types

try:
    import onnx_asr  # noqa: F401
    import onnxruntime  # noqa: F401
except ImportError:
    ort_stub = types.ModuleType("onnxruntime")
    ort_stub.get_available_providers = lambda: ["CPUExecutionProvider"]
    sys.modules.setdefault("onnx_asr", types.ModuleType("onnx_asr"))
    sys.modules.setdefault("onnxruntime", ort_stub)
