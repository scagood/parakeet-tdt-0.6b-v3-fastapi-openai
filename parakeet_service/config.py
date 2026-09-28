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
# huggingface_hub 2.x files Xet blobs in a cache-wide store sharded by hash
# prefix, so a model .onnx and its external-data file resolve into different
# directories and onnxruntime 1.30 refuses the data as escaping the model
# directory (#35). Per-repo blobs keep each pair side by side.
os.environ.setdefault("HF_HUB_DISABLE_SHARED_BLOBS", "1")

# Even with a fully warm cache, huggingface_hub makes a revision-check request
# to huggingface.co on every load. Offline mode skips it and reads the cache
# directly, which matters when many replicas start at once behind a
# rate-limited or firewalled egress. Only enable it where the cache is
# pre-seeded out of band; an incomplete cache fails the load instead of
# downloading the remainder.
HF_OFFLINE = _env_bool("PARAKEET_HF_OFFLINE", False)
if HF_OFFLINE:
    # Assign, not setdefault: an explicit operator request must win over a
    # base image that exports HF_HUB_OFFLINE=0.
    os.environ["HF_HUB_OFFLINE"] = "1"

# parakeet-tdt-0.6b-v3 language coverage; v2 is English-only.
_V3_LANGUAGES = [
    "bg", "hr", "cs", "da", "nl", "en", "et", "fi", "fr", "de", "el", "hu",
    "it", "lv", "lt", "mt", "pl", "pt", "ro", "ru", "sk", "sl", "es", "sv",
    "uk",
]
# Whisper's 99 languages (multilingual exports); the .en exports are English-only.
_WHISPER_LANGUAGES = [
    "en", "zh", "de", "es", "ru", "ko", "fr", "ja", "pt", "tr", "pl", "ca",
    "nl", "ar", "sv", "it", "id", "hi", "fi", "vi", "he", "uk", "el", "ms",
    "cs", "ro", "da", "hu", "ta", "no", "th", "ur", "hr", "bg", "lt", "la",
    "mi", "ml", "cy", "sk", "te", "fa", "lv", "bn", "sr", "az", "sl", "kn",
    "et", "mk", "br", "eu", "is", "hy", "ne", "mn", "bs", "kk", "sq", "sw",
    "gl", "mr", "pa", "si", "km", "sn", "yo", "so", "af", "oc", "ka", "be",
    "tg", "sd", "gu", "am", "yi", "lo", "uz", "fo", "ht", "ps", "tk", "nn",
    "mt", "sa", "lb", "my", "bo", "tl", "mg", "as", "tt", "haw", "ln", "ha",
    "ba", "jw", "su",
]

# One entry per model: what every precision of it shares (family, languages,
# chunk lengths) and, per quantization, the repo and onnx_asr quantization to
# load (https://github.com/istupakov/onnx-asr). A request names the model and
# may pick a quantization; without one it gets fp32, the reference precision,
# whatever the hardware. FP16 halves VRAM (identical output measured on
# Parakeet v3) but ONNX Runtime upcasts it on CPU, which is slower; int8
# measurably drops words after silences.
#
# Whisper repos are onnx-community's except where its export is broken (#35):
# the bare medium and large-v3 repos are empty (the exports live under -ONNX),
# there is no medium.en, and the .en fp16 merged decoders fail onnxruntime's
# graph check. Those come from Xenova, whose 8-bit files are _quantized (it has
# no int8 encoder). The Whisper encoder sees a fixed 30 s window and the export
# silently drops audio past it, so a longer chunk would lose its tail. Whisper
# returns text only; word times come from forced alignment of the transcript
# (#26).
MODEL_CONFIGS = {
    "parakeet-v3": {
        "family": "parakeet",
        "languages": _V3_LANGUAGES,
        "chunk_target_sec": 60.0,
        "chunk_max_sec": 75.0,
        "quantizations": {
            "fp32": ("istupakov/parakeet-tdt-0.6b-v3-onnx", None),
            "fp16": ("grikdotnet/parakeet-tdt-0.6b-fp16", "fp16"),
            "int8": ("istupakov/parakeet-tdt-0.6b-v3-onnx", "int8"),
        },
    },
    "parakeet-v2": {
        "family": "parakeet",
        "languages": ["en"],
        "chunk_target_sec": 25.0,
        "chunk_max_sec": 30.0,
        "quantizations": {
            "fp32": ("istupakov/parakeet-tdt-0.6b-v2-onnx", None),
            "fp16": ("ysdede/parakeet-tdt-0.6b-v2-onnx", "fp16"),
            "int8": ("istupakov/parakeet-tdt-0.6b-v2-onnx", "int8"),
        },
    },
    "whisper-tiny": {
        "family": "whisper",
        "languages": _WHISPER_LANGUAGES,
        "chunk_target_sec": 25.0,
        "chunk_max_sec": 30.0,
        "quantizations": {
            "fp32": ("onnx-community/whisper-tiny", None),
            "fp16": ("onnx-community/whisper-tiny", "fp16"),
            "int8": ("onnx-community/whisper-tiny", "int8"),
        },
    },
    "whisper-tiny.en": {
        "family": "whisper",
        "languages": ["en"],
        "chunk_target_sec": 25.0,
        "chunk_max_sec": 30.0,
        "quantizations": {
            "fp32": ("onnx-community/whisper-tiny.en", None),
            "fp16": ("Xenova/whisper-tiny.en", "fp16"),
            "int8": ("onnx-community/whisper-tiny.en", "int8"),
        },
    },
    "whisper-base": {
        "family": "whisper",
        "languages": _WHISPER_LANGUAGES,
        "chunk_target_sec": 25.0,
        "chunk_max_sec": 30.0,
        "quantizations": {
            "fp32": ("onnx-community/whisper-base", None),
            "fp16": ("onnx-community/whisper-base", "fp16"),
            "int8": ("onnx-community/whisper-base", "int8"),
        },
    },
    "whisper-base.en": {
        "family": "whisper",
        "languages": ["en"],
        "chunk_target_sec": 25.0,
        "chunk_max_sec": 30.0,
        "quantizations": {
            "fp32": ("onnx-community/whisper-base.en", None),
            "fp16": ("Xenova/whisper-base.en", "fp16"),
            "int8": ("onnx-community/whisper-base.en", "int8"),
        },
    },
    "whisper-small": {
        "family": "whisper",
        "languages": _WHISPER_LANGUAGES,
        "chunk_target_sec": 25.0,
        "chunk_max_sec": 30.0,
        "quantizations": {
            "fp32": ("onnx-community/whisper-small", None),
            "fp16": ("onnx-community/whisper-small", "fp16"),
            "int8": ("onnx-community/whisper-small", "int8"),
        },
    },
    "whisper-small.en": {
        "family": "whisper",
        "languages": ["en"],
        "chunk_target_sec": 25.0,
        "chunk_max_sec": 30.0,
        "quantizations": {
            "fp32": ("onnx-community/whisper-small.en", None),
            "fp16": ("Xenova/whisper-small.en", "fp16"),
            "int8": ("onnx-community/whisper-small.en", "int8"),
        },
    },
    "whisper-medium": {
        "family": "whisper",
        "languages": _WHISPER_LANGUAGES,
        "chunk_target_sec": 25.0,
        "chunk_max_sec": 30.0,
        "quantizations": {
            "fp32": ("onnx-community/whisper-medium-ONNX", None),
            "fp16": ("onnx-community/whisper-medium-ONNX", "fp16"),
            "int8": ("onnx-community/whisper-medium-ONNX", "int8"),
        },
    },
    "whisper-medium.en": {
        "family": "whisper",
        "languages": ["en"],
        "chunk_target_sec": 25.0,
        "chunk_max_sec": 30.0,
        "quantizations": {
            "fp32": ("Xenova/whisper-medium.en", None),
            "fp16": ("Xenova/whisper-medium.en", "fp16"),
            "int8": ("Xenova/whisper-medium.en", "quantized"),
        },
    },
    "whisper-large-v3": {
        "family": "whisper",
        "languages": _WHISPER_LANGUAGES,
        "chunk_target_sec": 25.0,
        "chunk_max_sec": 30.0,
        "quantizations": {
            "fp32": ("onnx-community/whisper-large-v3-ONNX", None),
            "fp16": ("onnx-community/whisper-large-v3-ONNX", "fp16"),
            "int8": ("onnx-community/whisper-large-v3-ONNX", "int8"),
        },
    },
    "whisper-large-v3-turbo": {
        "family": "whisper",
        "languages": _WHISPER_LANGUAGES,
        "chunk_target_sec": 25.0,
        "chunk_max_sec": 30.0,
        "quantizations": {
            "fp32": ("onnx-community/whisper-large-v3-turbo", None),
            "fp16": ("onnx-community/whisper-large-v3-turbo", "fp16"),
            "int8": ("onnx-community/whisper-large-v3-turbo", "int8"),
        },
    },
}
# Every entry lists the languages it transcribes: served on its model card, and
# ["en"] marks an English-only model whose word times the aligner may take.
# Every entry states the chunk length long audio is cut to: "chunk_target_sec"
# preferred, "chunk_max_sec" at most (and audio no longer than that is not cut
# at all). Parakeet v2, fp32 and int8 alike, hears whole stretches of clear
# speech in a chunk of 45 s or more as silence, which ones depending
# chaotically on where the chunk starts; no 20 or 30 s chunk did (#36).
# Parakeet v3 is the other way round: it loses more speech in 20-30 s chunks
# than in 60 s ones.

USE_GPU = _env_choice("PARAKEET_USE_GPU", "true", {"auto", "true", "false"})

# Models loaded (and warmed up) before the service reports ready, as "model"
# (fp32) or "model:quantization". Requests must still name their model; nothing
# here is used as a fallback. Entries are validated at startup, like requests.
PRELOAD_MODELS = [
    entry.strip().lower()
    for entry in os.getenv("PARAKEET_PRELOAD_MODELS", "").split(",")
    if entry.strip()
]


# ---------------------------------------------------------------------------
# Performance and safety knobs
# ---------------------------------------------------------------------------
TARGET_SR = 16_000

# Chunk lengths are per model (MODEL_CONFIGS). This is the shortest chunk cut
# at a pause, capped at the model's own target.
CHUNK_MIN_SEC = _env_float("PARAKEET_CHUNK_MIN_SEC", 20.0, minimum=0.0)

# Silence gaps at least this long are cut out of chunks instead of being fed
# to the model; long in-chunk silence measurably degrades recognition of the
# speech that follows it (int8 TDT drops words after multi-second pauses).
CHUNK_TRIM_SILENCE_SEC = _env_float("PARAKEET_CHUNK_TRIM_SILENCE_SEC", 3.0, minimum=0.5)

VAD_THRESHOLD = _env_float("PARAKEET_VAD_THRESHOLD", 0.5, minimum=0.0)
if VAD_THRESHOLD > 1.0:
    raise RuntimeError("PARAKEET_VAD_THRESHOLD must be <= 1.0")
VAD_MIN_SILENCE_MS = _env_int("PARAKEET_VAD_MIN_SILENCE_MS", 400, minimum=1)
VAD_SPEECH_PAD_MS = _env_int("PARAKEET_VAD_SPEECH_PAD_MS", 120, minimum=0)

# Loaded models are cached forever by default (0 = unbounded). Set a small N to
# LRU-evict all but the N most-recent when sweeping many models on limited RAM.
MODEL_CACHE_SIZE = _env_int("PARAKEET_MODEL_CACHE_SIZE", 0, minimum=0)

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
# Word timestamps can be re-timed by a wav2vec2 forced aligner (aligner.py). A
# request opts in with `align_words=true`; this is the answer for requests that
# don't say. Off by default: it costs ~2 s of CPU per 30 s of audio, and not
# every client wants it. The aligner downloads on the first request that uses it.
ALIGN_WORDS = _env_bool("PARAKEET_ALIGN_WORDS", False)
# Language assumed for alignment (and spoken numbers) when a request sends no
# `language`. Parakeet v3 is multilingual and nothing here detects the language,
# so this is an operator's statement about their audio. Empty means only align
# (or say numbers) when the request names a language.
ALIGN_DEFAULT_LANGUAGE = os.getenv("PARAKEET_ALIGN_DEFAULT_LANGUAGE", "en").strip().lower()
# Parakeet writes numbers the way it chooses, and not consistently: "twenty-five
# pounds" may come back as "£25" or "25 lb", "five dollars" as "$5". On, English
# transcripts say numbers, money and units in words instead (spoken.py), using
# the aligner to hear how each was said. A request opts in or out with
# `spoken_numbers=true|false`; this is the answer for requests that don't say.
SPOKEN_NUMBERS = _env_bool("PARAKEET_SPOKEN_NUMBERS", False)

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


def _own_cgroup_paths(
    proc_cgroup: Path,
) -> tuple[Optional[str], Optional[str], Optional[str]]:
    """Return this process's (v2 path, v1 cpu path, v1 cpu controllers).

    Parsed from ``/proc/self/cgroup``, where each line is
    ``<id>:<controllers>:<path>``; the v2 (unified) entry has an empty
    controller list. Entries are ``None`` when their hierarchy is absent. The
    v1 controller string is returned verbatim because it doubles as the mount
    directory name when the cpu controller is co-mounted (``cpu,cpuacct`` on
    most hosts, ``cpuacct,cpu`` on some).
    """
    v2_path = v1_path = v1_controllers = None
    text = _read_cgroup_file(proc_cgroup)
    for line in (text or "").splitlines():
        parts = line.split(":", 2)
        if len(parts) != 3:
            continue
        _hierarchy_id, controllers, path = parts
        if controllers == "":
            v2_path = path
        elif "cpu" in controllers.split(","):
            v1_path, v1_controllers = path, controllers
    return v2_path, v1_path, v1_controllers


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
    v2_path, v1_path, v1_controllers = _own_cgroup_paths(proc_cgroup)
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
    # controller is mounted under its own name or a co-mounted one.
    controller_dirs = dict.fromkeys(filter(None, (v1_controllers, "cpu", "cpu,cpuacct")))
    for relative in _cgroup_and_ancestors(v1_path):
        for controller_dir in controller_dirs:
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
# The word aligner always runs on CPU, even when Parakeet has the GPU, so it
# cannot share ORT_INTRA_THREADS (1 in GPU mode). Its int8 kernels stop scaling
# at about four threads.
ALIGN_THREADS = _env_int("PARAKEET_ALIGN_THREADS", min(4, _physical))
# Each InferencePool worker runs its own ORT call with ORT_INTRA_THREADS
# spinning threads, so workers x intra-op threads is what has to fit the CPUs
# the quota actually grants; four workers on four intra-op threads would put
# sixteen spinning threads on a 4-core budget.
INFER_WORKERS = _env_int(
    "PARAKEET_INFER_WORKERS",
    max(1, min(4, _available_logical // ORT_INTRA_THREADS)),
)


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
    "align_threads": ALIGN_THREADS,
}

if CPU_QUOTA is not None and CPU_QUOTA < _detected_logical:
    logger.info(
        "cgroup CPU quota %d is below the %d detected logical CPUs; "
        "sizing thread pools from the quota",
        CPU_QUOTA,
        _detected_logical,
    )
