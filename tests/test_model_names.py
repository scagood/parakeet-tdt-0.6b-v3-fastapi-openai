from __future__ import annotations

import inspect

import pytest
from fastapi import HTTPException, params

from parakeet_service import config, routes
from parakeet_service.config import MODEL_CONFIGS
from parakeet_service.model import variant_key
from parakeet_service.routes import _validate_model


def test_short_names_resolve():
    for name in MODEL_CONFIGS:
        assert _validate_model(name) == name


@pytest.mark.parametrize("handler", [routes.transcribe, routes.transcribe_batch])
def test_model_is_required_and_quantization_optional(handler):
    # No default model: a request without one is a 422 from FastAPI.
    parameters = inspect.signature(handler).parameters
    model, quantization = parameters["model"].default, parameters["quantization"].default
    assert isinstance(model, params.Form) and model.is_required()
    assert isinstance(quantization, params.Form) and quantization.default is None


def test_unknown_model_rejected():
    with pytest.raises(HTTPException):
        _validate_model("parakeet-v99")


def test_quantization_defaults_to_fp32_whatever_the_hardware():
    assert variant_key("parakeet-v3") == "parakeet-v3:fp32"
    assert variant_key("Whisper-Tiny", "FP16") == "whisper-tiny:fp16"
    # fp32 is the default, so every model must have it.
    for name, entry in MODEL_CONFIGS.items():
        assert "fp32" in entry["quantizations"], name


def test_unknown_quantization_is_a_400_naming_the_choices():
    with pytest.raises(HTTPException) as err:
        routes._variant("parakeet-v3", "q4")
    assert err.value.status_code == 400
    assert "fp16" in err.value.detail and "int8" in err.value.detail


def test_models_endpoint_lists_catalog():
    listing = routes.list_models()
    assert listing["object"] == "list"
    assert [card["id"] for card in listing["data"]] == list(MODEL_CONFIGS)
    for card in listing["data"]:
        assert card["object"] == "model"
        assert card["task"] == "automatic-speech-recognition"
        assert card["language"]
        assert card["quantizations"] == ["fp32", "fp16", "int8"]
    cards_by_id = {c["id"]: c for c in listing["data"]}
    assert cards_by_id["parakeet-v2"]["language"] == ["en"]
    assert cards_by_id["parakeet-v3"]["owned_by"] == "nvidia"


def test_models_endpoint_retrieve():
    assert routes.retrieve_model("Parakeet-V3")["id"] == "parakeet-v3"
    for gone in ("parakeet-v99", "parakeet-v3-fp32"):  # 2.0 dropped the -quant names
        with pytest.raises(HTTPException) as err:
            routes.retrieve_model(gone)
        assert err.value.status_code == 404


def test_whisper_registered_and_card_is_not_parakeet():
    card = routes.retrieve_model("whisper-base")
    assert card["owned_by"] == "openai"
    # Whisper's real multilingual set, not the Parakeet list or a bare ["auto"].
    assert card["language"] == config._WHISPER_LANGUAGES
    assert len(card["language"]) == 99 and "zh" in card["language"]
    assert card["language"] != config._V3_LANGUAGES


def test_whisper_quantizations_share_one_repo():
    assert MODEL_CONFIGS["whisper-small"]["quantizations"] == {
        "fp32": ("onnx-community/whisper-small", None),
        "fp16": ("onnx-community/whisper-small", "fp16"),
        "int8": ("onnx-community/whisper-small", "int8"),
    }


def test_whisper_english_model_reports_en():
    assert routes.retrieve_model("whisper-base.en")["language"] == ["en"]
    assert routes.retrieve_model("whisper-base")["language"] == config._WHISPER_LANGUAGES


def test_chunk_bounds_are_tighter_for_whisper_and_parakeet_v2():
    p_target, p_max, _ = routes._chunk_bounds("parakeet-v3")
    for name in ("whisper-base", "parakeet-v2"):
        target, maximum, _ = routes._chunk_bounds(name)
        assert maximum <= 30.0 < p_max
        assert target < p_target


def test_every_model_states_its_chunk_bounds():
    for name, entry in MODEL_CONFIGS.items():
        assert 0 < entry["chunk_target_sec"] <= entry["chunk_max_sec"], name
