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
