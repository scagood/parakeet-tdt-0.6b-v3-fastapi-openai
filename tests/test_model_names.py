from __future__ import annotations

import pytest
from fastapi import HTTPException

from parakeet_service.config import MODEL_ALIASES, MODEL_CONFIGS
from parakeet_service.routes import _validate_model


def test_short_names_resolve():
    for name in MODEL_CONFIGS:
        assert _validate_model(name) == name


def test_old_names_still_work():
    for alias, target in MODEL_ALIASES.items():
        assert target in MODEL_CONFIGS
        assert _validate_model(alias) == target
        assert _validate_model(alias.upper()) == target


def test_unknown_model_rejected():
    with pytest.raises(HTTPException):
        _validate_model("parakeet-v99")


def test_explicit_default_skips_probe(monkeypatch):
    from parakeet_service import model as m

    monkeypatch.setattr(m, "_DEFAULT_MODEL_NAME", None)
    monkeypatch.setattr(m, "DEFAULT_MODEL_EXPLICIT", True)
    monkeypatch.setattr(m, "DEFAULT_MODEL", "parakeet-v2")
    monkeypatch.setattr(m, "USE_GPU", "auto")
    monkeypatch.setattr(
        m, "_preload_cuda_libraries",
        lambda: (_ for _ in ()).throw(AssertionError("must not probe")),
    )
    assert m.default_model_name() == "parakeet-v2"


def test_auto_default_probes_cuda(monkeypatch):
    from parakeet_service import model as m

    monkeypatch.setattr(m, "DEFAULT_MODEL_EXPLICIT", False)
    monkeypatch.setattr(m, "USE_GPU", "auto")
    monkeypatch.setattr(m, "_preload_cuda_libraries", lambda: True)

    monkeypatch.setattr(m, "_DEFAULT_MODEL_NAME", None)
    monkeypatch.setattr(
        m.ort, "get_available_providers",
        lambda: ["CUDAExecutionProvider", "CPUExecutionProvider"],
    )
    assert m.default_model_name() == m.GPU_DEFAULT_MODEL

    monkeypatch.setattr(m, "_DEFAULT_MODEL_NAME", None)
    monkeypatch.setattr(m.ort, "get_available_providers", lambda: ["CPUExecutionProvider"])
    assert m.default_model_name() == m.CPU_DEFAULT_MODEL
