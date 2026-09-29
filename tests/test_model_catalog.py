from __future__ import annotations

import copy

import pytest

from parakeet_service.config import load_catalog, validate_aligners, validate_catalog

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
            "files": {"encoder-model.onnx.data.000": "encoder-model.onnx.data.000"},
        },
        "int8": {
            "repo": "me/model",
            "revision": "0" * 40,
            "files": {
                "encoder-model.onnx": "int8/encoder-model.int8.onnx",
                "decoder_joint-model.onnx": "int8/decoder_joint-model.int8.onnx",
            },
        },
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
        '      fp16:\n        repo: me/model\n        revision: "' + "a" * 40 + '"\n'
        "        files:\n          encoder-model.onnx: encoder-model.fp16.onnx\n"
        "aligners:\n  my-aligner:\n    aligner_type: sherpa-onnx\n"
        "    languages: *nordic\n    normalisers: [letters, lower]\n"
        '    blank: "<s>"\n    separator: " "\n    default_quantization: int8\n'
        '    quantizations:\n      int8:\n        repo: me/aligner\n        revision: "' + "b" * 40 + '"\n'
        "        files:\n          model.onnx: model.int8.onnx\n"
    )
    catalog = load_catalog(path)
    models = catalog["models"]
    # As for models, an aligner's default files are spelled out on load.
    assert catalog["aligners"]["my-aligner"]["quantizations"]["int8"]["files"] == {
        "model.onnx": "model.int8.onnx",
        "tokens.txt": "tokens.txt",
    }
    assert list(models) == ["my-model"]
    assert models["my-model"]["languages"] == ["da", "no", "sv"]
    # Defaults are spelled out on load; `files` overrides only what it names.
    quantizations = models["my-model"]["quantizations"]
    assert quantizations["fp32"]["files"] == {
        "encoder-model.onnx": "encoder-model.onnx",
        "decoder_joint-model.onnx": "decoder_joint-model.onnx",
        "vocab.txt": "vocab.txt",
        "config.json": "config.json",
    }
    assert quantizations["fp16"]["files"]["encoder-model.onnx"] == "encoder-model.fp16.onnx"
    assert quantizations["fp16"]["files"]["decoder_joint-model.onnx"] == "decoder_joint-model.onnx"


def _broken(change):
    models = {"my-model": copy.deepcopy(_ENTRY)}
    change(models["my-model"], models["my-model"]["quantizations"]["fp32"])
    return models


@pytest.mark.parametrize(
    ("change", "complaint"),
    [
        (lambda e, v: e["quantizations"].pop("fp32"), "fp32"),
        (lambda e, v: v.update(revision=1234), "revision"),
        (lambda e, v: e["quantizations"]["int8"].update(files=dict(v["files"])), "same files as my-model:fp32"),
        (lambda e, v: v.update(files=["encoder-model.onnx"]), "files must map"),
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


_ALIGNER = {
    "aligner_type": "transformers-js",
    "languages": ["en"],
    "normalisers": ["english", "upper"],
    "blank": "<pad>",
    "separator": "|",
    "default_quantization": "int8",
    "quantizations": {
        "fp32": {"repo": "me/aligner", "revision": "0" * 40},
        "int8": {"repo": "me/aligner", "revision": "0" * 40, "files": {"model.onnx": "onnx/model_int8.onnx"}},
    },
}


@pytest.mark.parametrize(
    ("change", "complaint"),
    [
        (lambda a: a.update(aligner_type="sherpa"), "aligner_type 'sherpa'"),
        (lambda a: a.update(normalisers=["english", "title"]), "normalisers must be a list"),
        (lambda a: a.update(normalisers="english"), "normalisers must be a list"),
        (lambda a: a.update(languages={"en": ["english"]}), "languages must list"),
        (lambda a: a.update(languages=["da", False]), "languages"),  # a bare `no`
        (lambda a: a.update(languages=["pt-BR"]), "without a region"),  # never matches a request
        (lambda a: a.update(languages=["EN"]), "lowercase"),
        (lambda a: a.update(normalisers=[{"english": True}]), "normalisers must be a list"),
        (lambda a: a.update(default_quantization=["int8"]), "default_quantization"),
        (lambda a: a.update(separator=0), "separator"),
        (lambda a: a.pop("separator"), "missing ['separator']"),  # null, not left out
        (lambda a: a.update(default_quantization="fp16"), "default_quantization"),
        (lambda a: a["quantizations"]["int8"].update(revision=1234), "revision"),
        (lambda a: a["quantizations"]["int8"]["files"].update({"tokens.txt": "tokens.txt"}), "not a file"),
        (lambda a: a["quantizations"]["int8"].pop("files"), "same files as my-aligner:fp32"),
    ],
)
def test_a_broken_aligner_names_the_problem(change, complaint):
    entry = copy.deepcopy(_ALIGNER)
    change(entry)
    with pytest.raises(ValueError, match=complaint.replace("[", r"\[").replace("]", r"\]")):
        validate_aligners({"my-aligner": entry})


def test_aligners_may_share_a_language_and_there_may_be_none():
    validate_aligners({"first": copy.deepcopy(_ALIGNER), "second": copy.deepcopy(_ALIGNER)})
    validate_aligners({})  # none: every word keeps the model's times
    with pytest.raises(ValueError, match="lowercase"):
        validate_aligners({"My-Aligner": copy.deepcopy(_ALIGNER)})


def test_an_invalid_file_stops_startup_naming_the_file(tmp_path):
    path = tmp_path / "models.yaml"
    path.write_text("models: {}\n")
    with pytest.raises(RuntimeError, match=str(path)):
        load_catalog(path)
