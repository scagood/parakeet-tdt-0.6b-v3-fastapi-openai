"""Configuration for the optimized Parakeet v3 service.

Configuration is validated at import time so invalid deployments fail before a
large model is downloaded or loaded into memory.
"""
from __future__ import annotations

import logging
import math
import os
import sys
from pathlib import Path
from typing import Iterable, Optional


# Set numeric-library limits before importing NumPy/ONNX Runtime in other modules.
for _name in (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
):
    os.environ.setdefault(_name, "1")


def _env_int(name: str, default: int, *, minimum: int = 1) -> int:
    raw = os.getenv(name)
    try:
        value = default if raw is None else int(raw)
    except ValueError as exc:
        raise RuntimeError(f"{name} must be an integer, got {raw!r}") from exc
    if value < minimum:
        raise RuntimeError(f"{name} must be >= {minimum}, got {value}")
    return value


def _env_float(name: str, default: float, *, minimum: float = 0.0) -> float:
    raw = os.getenv(name)
    try:
        value = default if raw is None else float(raw)
    except ValueError as exc:
        raise RuntimeError(f"{name} must be numeric, got {raw!r}") from exc
    if value < minimum:
        raise RuntimeError(f"{name} must be >= {minimum}, got {value}")
    return value


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    normalized = raw.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise RuntimeError(f"{name} must be a boolean, got {raw!r}")


def _env_choice(name: str, default: str, choices: Iterable[str]) -> str:
    allowed = {choice.lower() for choice in choices}
    value = os.getenv(name, default).strip().lower()
    if value not in allowed:
        raise RuntimeError(
            f"{name} must be one of {sorted(allowed)}, got {value!r}"
        )
    return value


# ---------------------------------------------------------------------------
# Paths & models
# ---------------------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = Path(os.getenv("PARAKEET_MODELS_DIR", ROOT_DIR / "models")).expanduser()
MODELS_DIR.mkdir(parents=True, exist_ok=True)

os.environ.setdefault("HF_HOME", str(MODELS_DIR))
os.environ.setdefault("HF_HUB_CACHE", str(MODELS_DIR))
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "true")

# Even with a fully warm cache, huggingface_hub makes a revision-check request
# to huggingface.co on every load. Offline mode skips it and reads the cache
# directly, which matters when many replicas start at once behind a
# rate-limited or firewalled egress. Only enable it where the cache is
# pre-seeded out of band; an incomplete cache fails the load instead of
# downloading the remainder.
HF_OFFLINE = _env_bool("PARAKEET_HF_OFFLINE", False)
if HF_OFFLINE:
    os.environ.setdefault("HF_HUB_OFFLINE", "1")

MODEL_CONFIGS = {
    "parakeet-v3-int8": {
        "hf_id": "nemo-parakeet-tdt-0.6b-v3",
        "quantization": "int8",
        "description": "INT8 CPU profile",
    },
    "parakeet-v3-fp32": {
        "hf_id": "istupakov/parakeet-tdt-0.6b-v3-onnx",
        "quantization": None,
        "description": "FP32 GPU profile",
    },
    "parakeet-v3-fp16": {
        "hf_id": "grikdotnet/parakeet-tdt-0.6b-fp16",
        "quantization": "fp16",
        "description": "FP16 GPU profile",
    },
    "parakeet-v2-int8": {
        "hf_id": "nemo-parakeet-tdt-0.6b-v2",
        "quantization": "int8",
        "description": "INT8 CPU profile (English-only v2)",
    },
    "parakeet-v2-fp32": {
        "hf_id": "istupakov/parakeet-tdt-0.6b-v2-onnx",
        "quantization": None,
        "description": "FP32 profile (English-only v2)",
    },
    "parakeet-v2-fp16": {
        "hf_id": "ysdede/parakeet-tdt-0.6b-v2-onnx",
        "quantization": "fp16",
        "description": "FP16 GPU profile (English-only v2)",
    },
}
# Former API names, kept working. Keys are lowercase; lookups are normalized.
MODEL_ALIASES = {
    "parakeet-v3": "parakeet-v3-fp32",
    "parakeet-tdt-0.6b-v3": "parakeet-v3-int8",
    "istupakov/parakeet-tdt-0.6b-v3-onnx": "parakeet-v3-fp32",
    "grikdotnet/parakeet-tdt-0.6b-fp16": "parakeet-v3-fp16",
    "parakeet-v2": "parakeet-v2-fp32",
    "parakeet-tdt-0.6b-v2": "parakeet-v2-int8",
    "istupakov/parakeet-tdt-0.6b-v2-onnx": "parakeet-v2-fp32",
}
# FP16 halves VRAM at identical output on GPU; on CPU it upcasts (slower), so
# CPU deployments default to FP32. int8 measurably drops words after silences.
GPU_DEFAULT_MODEL = "parakeet-v3-fp16"
CPU_DEFAULT_MODEL = "parakeet-v3-fp32"

USE_GPU = _env_choice("PARAKEET_USE_GPU", "true", {"auto", "true", "false"})
_default_model_fallback = CPU_DEFAULT_MODEL if USE_GPU == "false" else GPU_DEFAULT_MODEL
DEFAULT_MODEL_EXPLICIT = os.getenv("PARAKEET_DEFAULT_MODEL") is not None
DEFAULT_MODEL = os.getenv("PARAKEET_DEFAULT_MODEL", _default_model_fallback).strip().lower()
DEFAULT_MODEL = MODEL_ALIASES.get(DEFAULT_MODEL, DEFAULT_MODEL)
if DEFAULT_MODEL not in MODEL_CONFIGS:
    raise RuntimeError(
        "PARAKEET_DEFAULT_MODEL must be one of "
        f"{sorted(MODEL_CONFIGS)}, got {DEFAULT_MODEL!r}"
    )


# ---------------------------------------------------------------------------
# Performance and safety knobs
# ---------------------------------------------------------------------------
TARGET_SR = 16_000

CHUNK_TARGET_SEC = _env_float("PARAKEET_CHUNK_TARGET_SEC", 60.0, minimum=0.1)
CHUNK_MAX_SEC = _env_float("PARAKEET_CHUNK_MAX_SEC", 75.0, minimum=0.1)
CHUNK_MIN_SEC = _env_float("PARAKEET_CHUNK_MIN_SEC", 20.0, minimum=0.0)
if not CHUNK_MIN_SEC <= CHUNK_TARGET_SEC <= CHUNK_MAX_SEC:
    raise RuntimeError(
        "chunk durations must satisfy PARAKEET_CHUNK_MIN_SEC <= "
        "PARAKEET_CHUNK_TARGET_SEC <= PARAKEET_CHUNK_MAX_SEC"
    )

# Silence gaps at least this long are cut out of chunks instead of being fed
# to the model; long in-chunk silence measurably degrades recognition of the
# speech that follows it (int8 TDT drops words after multi-second pauses).
CHUNK_TRIM_SILENCE_SEC = _env_float("PARAKEET_CHUNK_TRIM_SILENCE_SEC", 3.0, minimum=0.5)

VAD_THRESHOLD = _env_float("PARAKEET_VAD_THRESHOLD", 0.5, minimum=0.0)
if VAD_THRESHOLD > 1.0:
    raise RuntimeError("PARAKEET_VAD_THRESHOLD must be <= 1.0")
VAD_MIN_SILENCE_MS = _env_int("PARAKEET_VAD_MIN_SILENCE_MS", 400, minimum=1)
VAD_SPEECH_PAD_MS = _env_int("PARAKEET_VAD_SPEECH_PAD_MS", 120, minimum=0)

GPU_DEVICE_ID = _env_int("PARAKEET_GPU_DEVICE_ID", 0, minimum=0)
BATCHED = _env_bool("PARAKEET_BATCHED", USE_GPU != "false")
MAX_BATCH_SIZE = _env_int("PARAKEET_MAX_BATCH_SIZE", 4)
BATCH_WINDOW_MS = _env_float("PARAKEET_BATCH_WINDOW_MS", 4.0, minimum=0.0)

# ONNX Runtime defers kernel selection and arena allocation to the first
# inference, so a freshly started replica serves its first real request well
# below steady-state speed. Pushing one synthetic chunk through before
# reporting ready moves that cost into startup, where an orchestrator is
# already waiting on the readiness probe.
WARMUP = _env_bool("PARAKEET_WARMUP", True)
WARMUP_SEC = _env_float("PARAKEET_WARMUP_SEC", 5.0, minimum=0.0)
# A warm-up that fails or exceeds this bound fails startup: a replica whose
# model cannot run one synthetic chunk would 500 every real request, and an
# orchestrator restarts a crashed replica faster than it notices a sick one.
WARMUP_TIMEOUT_SEC = _env_float("PARAKEET_WARMUP_TIMEOUT_SEC", 120.0, minimum=1.0)

MAX_UPLOAD_BYTES = _env_int(
    "PARAKEET_MAX_UPLOAD_BYTES", 256 * 1024 * 1024, minimum=1
)
MAX_BATCH_FILES = _env_int("PARAKEET_MAX_BATCH_FILES", 16)
MAX_BATCH_BYTES = _env_int(
    "PARAKEET_MAX_BATCH_BYTES", 512 * 1024 * 1024, minimum=1
)
MAX_AUDIO_SECONDS = _env_float("PARAKEET_MAX_AUDIO_SECONDS", 2 * 60 * 60, minimum=1.0)
MAX_REQUEST_CHUNKS = _env_int("PARAKEET_MAX_REQUEST_CHUNKS", 512)
FFMPEG_TIMEOUT_SEC = _env_float("PARAKEET_FFMPEG_TIMEOUT_SEC", 180.0, minimum=1.0)
UPLOAD_READ_CHUNK_BYTES = min(1024 * 1024, MAX_UPLOAD_BYTES)


# ---------------------------------------------------------------------------
# CPU/ORT threading
# ---------------------------------------------------------------------------
CGROUP_ROOT = Path("/sys/fs/cgroup")
PROC_CGROUP = Path("/proc/self/cgroup")


def _read_cgroup_file(path: Path) -> Optional[str]:
    try:
        return path.read_text().strip()
    except OSError:
        return None


def _quota_to_cpus(quota_raw: str, period_raw: str) -> Optional[int]:
    try:
        quota = int(quota_raw)
        period = int(period_raw)
    except ValueError:
        return None
    if quota <= 0 or period <= 0:  # -1 (v1) and 0 both mean "no limit"
        return None
    # Round up: a 3.5-core budget still runs 4 threads without oversubscribing,
    # since they timeshare within the same quota.
    return max(1, math.ceil(quota / period))


def _own_cgroup_paths(proc_cgroup: Path) -> tuple[Optional[str], Optional[str]]:
    """Return this process's (v2, v1 cpu) cgroup paths from ``/proc/self/cgroup``.

    Each line is ``<id>:<controllers>:<path>``; the v2 (unified) entry has an
    empty controller list. Either path is ``None`` when its hierarchy is absent.
    """
    v2_path = v1_path = None
    text = _read_cgroup_file(proc_cgroup)
    for line in (text or "").splitlines():
        parts = line.split(":", 2)
        if len(parts) != 3:
            continue
        _hierarchy_id, controllers, path = parts
        if controllers == "":
            v2_path = path
        elif "cpu" in controllers.split(","):
            v1_path = path
    return v2_path, v1_path


def _cgroup_and_ancestors(path: Optional[str]) -> list[Path]:
    """Relative cgroup path followed by each ancestor up to the root."""
    relative = Path((path or "/").lstrip("/"))
    return [relative, *relative.parents]


def cgroup_cpu_limit(
    root: Path = CGROUP_ROOT, proc_cgroup: Path = PROC_CGROUP
) -> Optional[int]:
    """Return the CPU count this cgroup's CFS quota allows, else ``None``.

    A Kubernetes ``resources.limits.cpu`` is a CFS *quota*, not a cpuset, so
    both ``os.sched_getaffinity()`` and ``psutil.cpu_count()`` report the
    node's full core count from inside a limited pod. Sizing thread pools from
    those numbers oversubscribes the quota badly — a 4-core pod on a 64-core
    node would otherwise start 64 ORT intra-op threads and thrash.

    The quota is looked up on the process's own cgroup, resolved through
    ``/proc/self/cgroup``, and on every ancestor: with a private cgroup
    namespace (the Docker and Kubernetes default) the process sits at the
    mount root, but under systemd ``CPUQuota=`` or ``--cgroupns=host`` it is
    nested several levels down and the mount root reports no limit at all.
    Nested quotas compose as a minimum, so the tightest one wins.
    """
    v2_path, v1_path = _own_cgroup_paths(proc_cgroup)
    limits: list[int] = []

    # cgroup v2: a single "<quota> <period>" line; quota is "max" when unset.
    unified = False
    for relative in _cgroup_and_ancestors(v2_path):
        raw = _read_cgroup_file(root / relative / "cpu.max")
        if raw is None:
            continue
        unified = True
        parts = raw.split()
        if len(parts) == 2 and parts[0] != "max":
            cpus = _quota_to_cpus(parts[0], parts[1])
            if cpus is not None:
                limits.append(cpus)
    if unified:
        return min(limits) if limits else None

    # cgroup v1: separate quota/period files, quota of -1 when unset. The cpu
    # controller is mounted as "cpu" or co-mounted as "cpu,cpuacct".
    for relative in _cgroup_and_ancestors(v1_path):
        for controller_dir in ("cpu", "cpu,cpuacct"):
            cpu_dir = root / controller_dir / relative
            quota = _read_cgroup_file(cpu_dir / "cpu.cfs_quota_us")
            period = _read_cgroup_file(cpu_dir / "cpu.cfs_period_us")
            if quota is None or period is None:
                continue
            cpus = _quota_to_cpus(quota, period)
            if cpus is not None:
                limits.append(cpus)
            break
    return min(limits) if limits else None


try:
    _detected_logical = len(os.sched_getaffinity(0))
except (AttributeError, OSError):
    _detected_logical = os.cpu_count() or 1

try:
    import psutil  # type: ignore

    _detected_physical = psutil.cpu_count(logical=False) or _detected_logical
except Exception:
    _detected_physical = _detected_logical

def effective_cpu_counts(
    detected_physical: int, detected_logical: int, quota: Optional[int]
) -> tuple[int, int]:
    """Clamp detected CPU counts to the cgroup quota, when one applies."""
    if quota is None:
        return detected_physical, detected_logical
    logical = max(1, min(detected_logical, quota))
    physical = max(1, min(detected_physical, logical))
    return physical, logical


CPU_QUOTA = cgroup_cpu_limit()
_physical, _available_logical = effective_cpu_counts(
    _detected_physical, _detected_logical, CPU_QUOTA
)

DEFAULT_INTRA = 1 if USE_GPU != "false" else min(_physical, _available_logical)
ORT_INTRA_THREADS = _env_int("PARAKEET_ORT_INTRA_THREADS", DEFAULT_INTRA)
ORT_INTER_THREADS = _env_int("PARAKEET_ORT_INTER_THREADS", 1)
AUDIO_WORKERS = _env_int("PARAKEET_AUDIO_WORKERS", min(8, _physical))
# Each InferencePool worker runs its own ORT call with ORT_INTRA_THREADS
# threads, so the pool must not outnumber the CPUs the quota actually grants:
# a 1-core pod running four concurrent sessions just timeshares one core.
INFER_WORKERS = _env_int("PARAKEET_INFER_WORKERS", max(1, min(4, _available_logical)))


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s  %(levelname)-7s  %(name)s: %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger("parakeet_v3")

CPU_INFO = {
    "physical": _physical,
    "logical": _available_logical,
    "detected_physical": _detected_physical,
    "detected_logical": _detected_logical,
    "cgroup_quota": CPU_QUOTA,
    "ort_intra": ORT_INTRA_THREADS,
    "ort_inter": ORT_INTER_THREADS,
    "audio_workers": AUDIO_WORKERS,
    "infer_workers": INFER_WORKERS,
}

if CPU_QUOTA is not None and CPU_QUOTA < _detected_logical:
    logger.info(
        "cgroup CPU quota %d is below the %d detected logical CPUs; "
        "sizing thread pools from the quota",
        CPU_QUOTA,
        _detected_logical,
    )
