from __future__ import annotations

from collections import OrderedDict

from parakeet_service import model as m


def _stub_loader(monkeypatch, cache_size, calls=None):
    monkeypatch.setattr(m, "MODEL_CACHE_SIZE", cache_size)
    monkeypatch.setattr(m, "_MODELS", OrderedDict())
    monkeypatch.setattr(m, "_resolve_providers", lambda: ["CPUExecutionProvider"])
    monkeypatch.setattr(m, "_validate_gpu_binding", lambda *a, **k: None)
    monkeypatch.setattr(m, "_build_sess_options", lambda *a, **k: None)

    def load(repo, quantization=None, **_kwargs):
        if calls is not None:
            calls.append((repo, quantization))
        return object()

    monkeypatch.setattr(m.onnx_asr, "load_model", load, raising=False)


def test_cache_evicts_least_recent_when_capped(monkeypatch):
    _stub_loader(monkeypatch, cache_size=2)
    a = m.load_model("whisper-tiny:fp32", with_timestamps=False)
    m.load_model("whisper-base:fp32", with_timestamps=False)
    assert m.load_model("whisper-tiny:fp32", with_timestamps=False) is a  # hit keeps it warm
    m.load_model("whisper-small:fp32", with_timestamps=False)  # evicts base (least-recent)
    assert m.loaded_models() == ["whisper-small:fp32", "whisper-tiny:fp32"]


def test_cache_unbounded_by_default(monkeypatch):
    _stub_loader(monkeypatch, cache_size=0)
    for name in ("whisper-tiny", "whisper-base", "whisper-small"):
        m.load_model(f"{name}:fp32", with_timestamps=False)
    assert len(m.loaded_models()) == 3


def test_each_quantization_loads_its_own_repo_and_file(monkeypatch):
    calls = []
    _stub_loader(monkeypatch, cache_size=0, calls=calls)
    m.load_model("whisper-medium.en:int8", with_timestamps=False)
    m.load_model("whisper-medium.en:fp32", with_timestamps=False)
    assert calls == [("Xenova/whisper-medium.en", "quantized"), ("Xenova/whisper-medium.en", None)]
    assert m.loaded_models() == ["whisper-medium.en:fp32", "whisper-medium.en:int8"]
