"""Configuration for the optimized Parakeet v3 service.

Configuration is validated at import time so invalid deployments fail before a
large model is downloaded or loaded into memory.
"""
from __future__ import annotations

import logging
import math
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import yaml


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

# The model catalog: every model the service serves, and how to load each
# precision of it. The entries are data, in models.yaml next to this file, or in
# the YAML file PARAKEET_MODEL_CATALOG names (a k8s ConfigMap, say), which then
# replaces the built-in catalog wholesale. Either way it is validated here, so a
# broken catalog stops the service at startup rather than at a request.

# What onnx-asr reads from a model folder, by the model type that runs it, and
# where each file sits in the repo unless an entry's "files" says otherwise: the
# fp32 layout of that type's usual exports (istupakov's for Parakeet,
# onnx-community's for Whisper). Other precisions name their own files.
ONNX_ASR_DEFAULT_FILES = {
    "nemo-conformer-tdt": {
        "encoder-model.onnx": "encoder-model.onnx",
        "decoder_joint-model.onnx": "decoder_joint-model.onnx",
        "vocab.txt": "vocab.txt",
        "config.json": "config.json",
    },
    "whisper": {
        "encoder_model.onnx": "onnx/encoder_model.onnx",
        "decoder_model_merged.onnx": "onnx/decoder_model_merged.onnx",
        "vocab.json": "vocab.json",
        "added_tokens.json": "added_tokens.json",
        "config.json": "config.json",
    },
}
# How routes.py treats a model's output: Parakeet's TDT tokens carry word times,
# Whisper returns text only.
MODEL_FAMILIES = {"parakeet", "whisper"}
_KEYS = {"family", "onnx_asr_type", "languages", "chunk_target_sec", "chunk_max_sec", "quantizations"}
_REVISION = re.compile(r"[0-9a-f]{40}")


def validate_catalog(models: Dict[str, Any]) -> None:
    """Raise ValueError naming the first thing wrong with a catalog's models."""
    if not isinstance(models, dict) or not models:
        raise ValueError("`models` must map model names to entries")
    for name, entry in models.items():
        # Requests are matched lowercased, so a mixed-case name could never be asked for.
        if not isinstance(name, str) or name != name.lower():
            raise ValueError(f"{name!r}: model names must be lowercase strings")
        if not isinstance(entry, dict):
            raise ValueError(f"{name}: an entry must be a mapping")
        if missing := _KEYS - entry.keys():
            raise ValueError(f"{name}: missing {sorted(missing)}")
        if not isinstance(entry["family"], str) or entry["family"] not in MODEL_FAMILIES:
            raise ValueError(f"{name}: family {entry['family']!r} is not one of {sorted(MODEL_FAMILIES)}")
        onnx_asr_type = entry["onnx_asr_type"]
        if not isinstance(onnx_asr_type, str) or onnx_asr_type not in ONNX_ASR_DEFAULT_FILES:
            raise ValueError(
                f"{name}: onnx_asr_type {onnx_asr_type!r} is not one of {sorted(ONNX_ASR_DEFAULT_FILES)}"
            )
        defaults = ONNX_ASR_DEFAULT_FILES[onnx_asr_type]
        languages = entry["languages"]
        if not isinstance(languages, list) or not languages or not all(isinstance(x, str) for x in languages):
            # A bare `no` (Norwegian) arrives as False: quote language codes.
            raise ValueError(f"{name}: languages must be a non-empty list of quoted codes")
        target, maximum = entry["chunk_target_sec"], entry["chunk_max_sec"]
        if not all(isinstance(x, (int, float)) for x in (target, maximum)) or not 0 < target <= maximum:
            raise ValueError(f"{name}: need 0 < chunk_target_sec <= chunk_max_sec")
        quantizations = entry["quantizations"]
        if not isinstance(quantizations, dict) or "fp32" not in quantizations:
            raise ValueError(f"{name}: quantizations must include fp32, the default")
        graphs = [f for f in defaults if f.endswith(".onnx")]
        loads: Dict[tuple, str] = {}
        for quant, variant in quantizations.items():
            where = f"{name}:{quant}"
            if not isinstance(quant, str) or quant != quant.lower():
                raise ValueError(f"{where}: quantization names must be lowercase strings")
            if not isinstance(variant, dict) or not isinstance(variant.get("repo"), str):
                raise ValueError(f"{where}: needs a repo")
            # Quote it: an unquoted SHA can load as a number.
            if not isinstance(variant.get("revision"), str) or not _REVISION.fullmatch(variant["revision"]):
                raise ValueError(f"{where}: revision must be a quoted 40-character commit SHA")
            files = variant.get("files", {})
            if not isinstance(files, dict) or not all(isinstance(x, str) for x in (*files, *files.values())):
                raise ValueError(f"{where}: files must map onnx-asr names to repo paths")
            for extra in files.keys() - defaults.keys():
                # Anything else is external data (possibly sharded, .data.000),
                # named as its .onnx refers to it.
                if not (any(extra.startswith(g) for g in graphs) and "data" in extra):
                    raise ValueError(f"{where}: {extra!r} is neither a file onnx-asr reads nor external data")
            # A quantization that forgot its `files` would quietly load another's.
            key = (variant["repo"], variant["revision"], tuple(sorted({**defaults, **files}.items())))
            if key in loads:
                raise ValueError(f"{where}: loads the same files as {name}:{loads[key]}; name its own in `files`")
            loads[key] = quant


# The fp32 files an aligner reads, by `aligner_type`, as ONNX_ASR_DEFAULT_FILES is for models.
ALIGNER_DEFAULT_FILES = {
    "transformers-js": {"model.onnx": "onnx/model.onnx", "vocab.json": "vocab.json"},  # {token: id}
    "sherpa-onnx": {"model.onnx": "model.onnx", "tokens.txt": "tokens.txt"},  # "token id" lines
}
# The steps an aligner's `normalisers` may list (aligner.py writes them).
ALIGN_NORMALISERS = {"english", "letters", "upper", "lower"}
_ALIGNER_KEYS = {
    "aligner_type", "languages", "normalisers", "blank", "separator", "default_quantization", "quantizations"
}


def validate_aligners(aligners: Dict[str, Any]) -> None:
    """Raise ValueError naming the first thing wrong with a catalog's aligners."""
    if not isinstance(aligners, dict):
        raise ValueError("`aligners` must map aligner names to entries ({} for none)")
    for name, entry in aligners.items():
        # Requests are matched lowercased, as model names are.
        if not isinstance(name, str) or name != name.lower():
            raise ValueError(f"{name!r}: aligner names must be lowercase strings")
        if not isinstance(entry, dict):
            raise ValueError(f"{name}: an aligner must be a mapping")
        if missing := _ALIGNER_KEYS - entry.keys():
            raise ValueError(f"{name}: missing {sorted(missing)}")
        aligner_type = entry["aligner_type"]
        if not isinstance(aligner_type, str) or aligner_type not in ALIGNER_DEFAULT_FILES:
            raise ValueError(
                f"{name}: aligner_type {aligner_type!r} is not one of {sorted(ALIGNER_DEFAULT_FILES)}"
            )
        languages, steps = entry["languages"], entry["normalisers"]
        # Requests match on a lowercase code without its region ("fr-CA" is "fr").
        # A bare `no` (Norwegian) arrives as False: quote language codes.
        if not isinstance(languages, list) or not languages or not all(
            isinstance(x, str) and re.fullmatch(r"[a-z]{2,3}", x) for x in languages
        ):
            raise ValueError(f"{name}: languages must list quoted lowercase codes, without a region")
        if not isinstance(steps, list) or not steps or not all(
            isinstance(step, str) and step in ALIGN_NORMALISERS for step in steps
        ):
            raise ValueError(f"{name}: normalisers must be a list of {sorted(ALIGN_NORMALISERS)}")
        if not isinstance(entry["blank"], str):
            raise ValueError(f"{name}: blank must be a quoted token")
        if entry["separator"] is not None and not isinstance(entry["separator"], str):
            raise ValueError(f"{name}: separator must be a quoted token, or null for none")
        quantizations = entry["quantizations"]
        default = entry["default_quantization"]
        if not isinstance(quantizations, dict) or not isinstance(default, str) or default not in quantizations:
            raise ValueError(f"{name}: quantizations must include default_quantization")
        defaults = ALIGNER_DEFAULT_FILES[aligner_type]
        loads: Dict[tuple, str] = {}
        for quant, variant in quantizations.items():
            where = f"{name}:{quant}"
            if not isinstance(quant, str) or quant != quant.lower():
                raise ValueError(f"{where}: quantization names must be lowercase strings")
            if not isinstance(variant, dict) or not isinstance(variant.get("repo"), str):
                raise ValueError(f"{where}: needs a repo")
            if not isinstance(variant.get("revision"), str) or not _REVISION.fullmatch(variant["revision"]):
                raise ValueError(f"{where}: revision must be a quoted 40-character commit SHA")
            files = variant.get("files", {})
            if not isinstance(files, dict) or not all(isinstance(x, str) for x in (*files, *files.values())):
                raise ValueError(f"{where}: files must map aligner file names to repo paths")
            if extra := files.keys() - defaults.keys():
                raise ValueError(f"{where}: {sorted(extra)} is not a file a {aligner_type} aligner reads")
            # A quantization that forgot its `files` would quietly load another's.
            key = (variant["repo"], variant["revision"], tuple(sorted({**defaults, **files}.items())))
            if key in loads:
                raise ValueError(f"{where}: loads the same files as {name}:{loads[key]}; name its own in `files`")
            loads[key] = quant


def load_catalog(path: Path) -> Dict[str, Any]:
    """Read and validate a catalog file; return its models and aligners."""
    with open(path, encoding="utf-8") as handle:
        catalog = yaml.safe_load(handle)
    try:
        if not isinstance(catalog, dict):
            raise ValueError("expected a mapping with `models` and `aligners` keys")
        validate_catalog(catalog.get("models"))
        validate_aligners(catalog.get("aligners"))
    except ValueError as exc:
        raise RuntimeError(f"invalid model catalog {path}: {exc}") from None
    # Spell out every file here, so loading never has to know about defaults.
    for entry in catalog["models"].values():
        for variant in entry["quantizations"].values():
            variant["files"] = {**ONNX_ASR_DEFAULT_FILES[entry["onnx_asr_type"]], **variant.get("files", {})}
    for entry in catalog["aligners"].values():
        for variant in entry["quantizations"].values():
            variant["files"] = {**ALIGNER_DEFAULT_FILES[entry["aligner_type"]], **variant.get("files", {})}
    return {"models": catalog["models"], "aligners": catalog["aligners"]}


CATALOG_PATH = Path(os.getenv("PARAKEET_MODEL_CATALOG") or Path(__file__).with_name("models.yaml"))
_CATALOG = load_catalog(CATALOG_PATH)
MODEL_CONFIGS = _CATALOG["models"]
ALIGNER_CONFIGS = _CATALOG["aligners"]

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

# Chunk lengths are per model (models.yaml). This is the shortest chunk cut
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

# Loaded models and aligners are cached forever by default (0 = unbounded). Set
# a small N to LRU-evict all but the N most-recent models, and likewise aligners,
# when clients can ask for more than fits in RAM.
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
