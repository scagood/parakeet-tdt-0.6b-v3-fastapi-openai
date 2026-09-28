"""Thread-safe ONNX Runtime model loading for Parakeet TDT."""
from __future__ import annotations

import os
import tempfile
import threading
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, List, Tuple

# Import config before ONNX Runtime so thread-pool environment limits are active.
from .config import (
    GPU_DEVICE_ID,
    MODEL_CACHE_SIZE,
    MODEL_CONFIGS,
    MODELS_DIR,
    ORT_INTER_THREADS,
    ORT_INTRA_THREADS,
    TARGET_SR,
    USE_GPU,
    WARMUP_SEC,
    logger,
)

import numpy as np
import onnx_asr
import onnxruntime as ort

_ModelKey = Tuple[str, bool]
_MODELS: "OrderedDict[_ModelKey, object]" = OrderedDict()
_MODEL_LOCK = threading.RLock()
_CUDA_PRELOADED = False


def _preload_cuda_libraries() -> bool:
    """Load CUDA/cuDNN libraries before creating any ORT session."""
    global _CUDA_PRELOADED
    if USE_GPU not in {"true", "auto"}:
        return False
    if _CUDA_PRELOADED:
        return True
    with _MODEL_LOCK:
        if _CUDA_PRELOADED:
            return True
        preload = getattr(ort, "preload_dlls", None)
        if preload is None:
            if USE_GPU == "true":
                raise RuntimeError(
                    "onnxruntime-gpu does not expose preload_dlls; install a "
                    "compatible ONNX Runtime GPU package"
                )
            return True
        try:
            # Empty directory explicitly prefers NVIDIA runtime wheels installed
            # by the [cuda,cudnn] extra over potentially incompatible system libs.
            preload(cuda=True, cudnn=True, msvc=False, directory="")
            _CUDA_PRELOADED = True
            logger.info("Preloaded CUDA/cuDNN libraries for ONNX Runtime")
            return True
        except Exception as exc:
            if USE_GPU == "true":
                raise RuntimeError("failed to preload CUDA/cuDNN libraries") from exc
            logger.warning("CUDA/cuDNN preload failed; using CPU fallback: %s", exc)
            return False


def _build_sess_options(
    intra_threads: int = ORT_INTRA_THREADS, spinning: bool = True
) -> ort.SessionOptions:
    options = ort.SessionOptions()
    options.intra_op_num_threads = intra_threads
    options.inter_op_num_threads = ORT_INTER_THREADS
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    options.add_session_config_entry("session.set_denormal_as_zero", "1")
    options.add_session_config_entry(
        "session.intra_op.allow_spinning", "1" if spinning else "0"
    )
    options.add_session_config_entry("session.inter_op.allow_spinning", "0")
    return options


def _resolve_providers() -> List[Any]:
    cuda_runtime_ready = _preload_cuda_libraries()
    available = set(ort.get_available_providers())
    has_cuda = cuda_runtime_ready and "CUDAExecutionProvider" in available

    if USE_GPU == "true" and not has_cuda:
        raise RuntimeError(
            "PARAKEET_USE_GPU=true but CUDAExecutionProvider is unavailable; "
            f"available providers: {sorted(available)}"
        )
    if USE_GPU == "false" or not has_cuda:
        return ["CPUExecutionProvider"]

    cuda = (
        "CUDAExecutionProvider",
        {
            "device_id": GPU_DEVICE_ID,
            "cudnn_conv_algo_search": "EXHAUSTIVE",
            "cudnn_conv_use_max_workspace": "1",
            "do_copy_in_default_stream": "1",
        },
    )
    return [cuda] if USE_GPU == "true" else [cuda, "CPUExecutionProvider"]


def _session_provider_report(model: Any) -> Dict[str, List[str]]:
    report: Dict[str, List[str]] = {}
    candidates = [("model", model), ("asr", getattr(model, "asr", None))]
    for prefix, candidate in candidates:
        if candidate is None:
            continue
        for attribute in ("_model", "_encoder", "_decoder", "_decoder_joint"):
            session = getattr(candidate, attribute, None)
            get_providers = getattr(session, "get_providers", None)
            if callable(get_providers):
                report[f"{prefix}.{attribute}"] = list(get_providers())
    return report


def _validate_gpu_binding(name: str, model: Any) -> None:
    report = _session_provider_report(model)
    if report:
        logger.info("Session providers for %s: %s", name, report)
    if USE_GPU != "true":
        return
    if not report:
        raise RuntimeError(
            f"PARAKEET_USE_GPU=true but ORT providers for {name} could not be inspected"
        )
    if not all(
        providers
        and providers[0] in {"CUDAExecutionProvider", "TensorrtExecutionProvider"}
        for providers in report.values()
    ):
        raise RuntimeError(
            f"PARAKEET_USE_GPU=true but {name} did not bind all sessions to GPU: {report}"
        )


def variant_key(model: str, quantization: str | None = None) -> str:
    """Resolve a model and optional quantization to its "model:quant" key.

    Without a quantization the answer is fp32, the reference precision, never
    something picked from the hardware: the same request gets the same numbers
    on every deployment. Raises ValueError naming the valid choices.
    """
    name = model.strip().lower()
    if name not in MODEL_CONFIGS:
        raise ValueError(f"Unknown model {model!r}. Available models: {sorted(MODEL_CONFIGS)}")
    quant = (quantization or "fp32").strip().lower()
    available = MODEL_CONFIGS[name]["quantizations"]
    if quant not in available:
        raise ValueError(
            f"Model {name!r} has no {quant!r} quantization. Available: {list(available)}"
        )
    return f"{name}:{quant}"


def _link_files(variant: Dict[str, Any], folder: Path) -> None:
    """Fetch a variant's files and link them into `folder` under the names
    onnx-asr expects (models.yaml "files").

    Hard links, not symlinks: onnxruntime resolves a symlinked .onnx to its
    cache blob and refuses external data that resolves anywhere else (#35).
    """
    from huggingface_hub import hf_hub_download

    for name, path in variant["files"].items():
        blob = os.path.realpath(
            hf_hub_download(variant["repo"], path, revision=variant["revision"])
        )
        try:
            os.link(blob, folder / name)
        except OSError:
            # Cache on another filesystem. Per-repo blobs keep a model and its
            # data side by side (HF_HUB_DISABLE_SHARED_BLOBS), so this still loads.
            os.symlink(blob, folder / name)


def load_model(key: str, *, with_timestamps: bool = True):
    """Load (or return the cached) model for a variant_key() key."""
    cache_key = (key, with_timestamps)

    with _MODEL_LOCK:
        cached = _MODELS.get(cache_key)
        if cached is not None:
            _MODELS.move_to_end(cache_key)
            return cached

        name, _, quant = key.partition(":")
        config = MODEL_CONFIGS[name]
        variant = config["quantizations"][quant]
        providers = _resolve_providers()
        session_options = _build_sess_options()
        logger.info(
            "Loading %s from %s providers=%s intra=%d inter=%d",
            key,
            variant["repo"],
            providers,
            ORT_INTRA_THREADS,
            ORT_INTER_THREADS,
        )
        # onnx-asr reads every file while it loads, so the links only need to
        # live that long; a folder per load keeps replicas sharing the models
        # volume out of each other's way.
        with tempfile.TemporaryDirectory(dir=MODELS_DIR, prefix=".load-") as folder:
            _link_files(variant, Path(folder))
            model = onnx_asr.load_model(
                config["onnx_asr_type"],
                folder,
                providers=providers,
                sess_options=session_options,
            )
        # Verified on onnx_asr 0.12.0 (whisper-tiny): .with_timestamps() works
        # for Whisper, but the standard onnx-community/whisper-* repos carry no
        # alignment heads, so it returns empty tokens/timestamps — text only.
        # The token/timestamp shape _stitch consumes is Parakeet TDT's anyway,
        # so keep Whisper on the plain text adapter (its .recognize() returns the
        # transcript string). Whisper word times would come from forced-aligning
        # that transcript against the audio, not from the model.
        if with_timestamps and config["family"] == "parakeet":
            model = model.with_timestamps()
        _validate_gpu_binding(key, model)
        _MODELS[cache_key] = model
        # ponytail: LRU cap, drop least-recent so a many-model sweep fits RAM.
        while MODEL_CACHE_SIZE and len(_MODELS) > MODEL_CACHE_SIZE:
            evicted, _ = _MODELS.popitem(last=False)
            logger.info("Evicted %s (cache size %d)", evicted, MODEL_CACHE_SIZE)
        logger.info("Loaded %s", key)
        return model


def get_model(key: str):
    return load_model(key, with_timestamps=True)


def warmup_waveform(seconds: float | None = None) -> np.ndarray:
    """Build the synthetic 16 kHz waveform used for the startup warm-up pass.

    Deterministic on purpose: warm-up cost should not vary run to run. The
    tones sit in the speech band so the encoder, decoder and joint networks
    all get exercised, while the low amplitude keeps the decoder from emitting
    much text.
    """
    duration = WARMUP_SEC if seconds is None else seconds
    samples = max(1, int(duration * TARGET_SR))
    time_axis = np.arange(samples, dtype=np.float32) / TARGET_SR
    tones = sum(
        np.sin(2.0 * np.pi * frequency * time_axis) for frequency in (110.0, 220.0, 440.0)
    )
    return (0.05 * tones).astype(np.float32)


def loaded_models() -> List[str]:
    with _MODEL_LOCK:
        return sorted({key for key, _timestamps in _MODELS})
