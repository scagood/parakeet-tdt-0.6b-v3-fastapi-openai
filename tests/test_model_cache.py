from __future__ import annotations

from collections import OrderedDict

from parakeet_service import model as m


def _stub_loader(monkeypatch, cache_size):
    monkeypatch.setattr(m, "MODEL_CACHE_SIZE", cache_size)
    monkeypatch.setattr(m, "_MODELS", OrderedDict())
    monkeypatch.setattr(m, "_resolve_providers", lambda: ["CPUExecutionProvider"])
    monkeypatch.setattr(m, "_validate_gpu_binding", lambda *a, **k: None)
    monkeypatch.setattr(m, "_build_sess_options", lambda *a, **k: None)
    monkeypatch.setattr(m.onnx_asr, "load_model", lambda *a, **k: object(), raising=False)


def test_cache_evicts_least_recent_when_capped(monkeypatch):
    _stub_loader(monkeypatch, cache_size=2)
    a = m.load_model("whisper-tiny", with_timestamps=False)
    m.load_model("whisper-base", with_timestamps=False)
    assert m.load_model("whisper-tiny", with_timestamps=False) is a  # hit keeps it warm
    m.load_model("whisper-small", with_timestamps=False)  # evicts base (least-recent)
    assert m.loaded_models() == ["whisper-small", "whisper-tiny"]


def test_cache_unbounded_by_default(monkeypatch):
    _stub_loader(monkeypatch, cache_size=0)
    for name in ("whisper-tiny", "whisper-base", "whisper-small"):
        m.load_model(name, with_timestamps=False)
    assert len(m.loaded_models()) == 3
