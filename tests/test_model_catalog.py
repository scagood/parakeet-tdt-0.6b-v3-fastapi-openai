from __future__ import annotations

import copy

import pytest

from parakeet_service.config import load_catalog, validate_catalog

_ENTRY = {
    "family": "parakeet",
    "onnx_asr_type": "nemo-conformer-tdt",
    "languages": ["en"],
    "chunk_target_sec": 25.0,
    "chunk_max_sec": 30.0,
    "quantizations": {
        "fp32": {
            "repo": "me/model",
            "revision": "0" * 40,
            "files": {
                "encoder-model.onnx": "encoder-model.onnx",
                "encoder-model.onnx.data.000": "encoder-model.onnx.data.000",
                "decoder_joint-model.onnx": "decoder_joint-model.onnx",
                "vocab.txt": "vocab.txt",
                "config.json": "config.json",
            },
        }
    },
}


def test_a_replacement_file_loads_with_anchors_and_quoted_codes(tmp_path):
    path = tmp_path / "models.yaml"
    path.write_text(
        'languages:\n  nordic: &nordic ["da", "no", "sv"]\n'
        "models:\n  my-model:\n"
        "    family: parakeet\n    onnx_asr_type: nemo-conformer-tdt\n    languages: *nordic\n"
        "    chunk_target_sec: 25\n    chunk_max_sec: 30\n"
        '    quantizations:\n      fp32:\n        repo: me/model\n        revision: "' + "a" * 40 + '"\n'
        "        files:\n          encoder-model.onnx: encoder-model.onnx\n"
        "          decoder_joint-model.onnx: decoder_joint-model.onnx\n"
        "          vocab.txt: vocab.txt\n          config.json: config.json\n"
    )
    models = load_catalog(path)
    assert list(models) == ["my-model"]
    assert models["my-model"]["languages"] == ["da", "no", "sv"]


def _broken(change):
    models = {"my-model": copy.deepcopy(_ENTRY)}
    change(models["my-model"], models["my-model"]["quantizations"]["fp32"])
    return models


@pytest.mark.parametrize(
    ("change", "complaint"),
    [
        (lambda e, v: e["quantizations"].pop("fp32"), "fp32"),
        (lambda e, v: v.update(revision=1234), "revision"),
        (lambda e, v: v["files"].pop("vocab.txt"), "missing ['vocab.txt']"),
        (lambda e, v: v["files"].update({"notes.txt": "notes.txt"}), "neither"),
        (lambda e, v: e.update(onnx_asr_type="whisper-ort"), "onnx_asr_type"),
        (lambda e, v: e.update(family=["parakeet"]), "family"),
        (lambda e, v: e.update(languages=["da", False]), "languages"),  # a bare `no`
        (lambda e, v: e.update(chunk_target_sec=40.0), "chunk_target_sec"),
        (lambda e, v: e.pop("languages"), "missing ['languages']"),
    ],
)
def test_a_broken_catalog_names_the_problem(change, complaint):
    with pytest.raises(ValueError, match=complaint.replace("[", r"\[").replace("]", r"\]")):
        validate_catalog(_broken(change))


def test_model_names_must_be_lowercase():
    with pytest.raises(ValueError, match="lowercase"):
        validate_catalog({"My-Model": copy.deepcopy(_ENTRY)})


def test_an_invalid_file_stops_startup_naming_the_file(tmp_path):
    path = tmp_path / "models.yaml"
    path.write_text("models: {}\n")
    with pytest.raises(RuntimeError, match=str(path)):
        load_catalog(path)
