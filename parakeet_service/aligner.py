"""Word boundaries by CTC forced alignment (WhisperX-style, without torch).

Parakeet decides *what* was said; a character-level wav2vec2 CTC model, run
through ONNX Runtime, decides *when*. Parakeet's own word times sit on 80 ms
encoder frames and its word ends are estimated (see routes._WORD_TAIL_SEC).
Aligned times sit on 20 ms frames and the ends come from the audio.
"""
from __future__ import annotations

import json
import threading
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .config import ALIGN_THREADS, ALIGN_WORDS, TARGET_SR, logger

Span = Tuple[float, float]

# One char-level wav2vec2 CTC export (ONNX + vocab.json) per language. Adding a
# language is adding an entry here.
ALIGN_MODELS: Dict[str, Dict[str, str]] = {
    "en": {
        "repo": "onnx-community/wav2vec2-base-960h-ONNX",
        "revision": "729c1a6730fb549c20a1c73a3d3f96f11020225e",
        "onnx": "onnx/model_int8.onnx",
        "vocab": "vocab.json",
    },
}

_STRIDE = 320  # wav2vec2's feature encoder emits one frame per 320 samples (20 ms)
_MIN_SAMPLES = 400  # receptive field of that encoder; shorter input has no frames
# ponytail: wav2vec2 self-attention is O(frames²), so long chunks run in 30 s
# windows. A word straddling a window edge loses some context; overlap the
# windows if that shows up in practice.
_WINDOW = 30 * TARGET_SR
_DIGITS = "ZERO ONE TWO THREE FOUR FIVE SIX SEVEN EIGHT NINE".split()
_NEG_INF = -1e30

_lock = threading.Lock()
_loaded: Dict[str, Optional[Tuple[Any, Dict[str, int]]]] = {}


def supports(language: Optional[str]) -> bool:
    return ALIGN_WORDS and _language(language) in ALIGN_MODELS


def _language(language: Optional[str]) -> str:
    # ponytail: no language sent means English. Non-English audio on v3 should
    # send `language` so it keeps Parakeet's own times instead.
    return (language or "en").strip().lower()


def _load(language: str) -> Optional[Tuple[Any, Dict[str, int]]]:
    with _lock:
        if language in _loaded:
            return _loaded[language]
        spec = ALIGN_MODELS[language]
        loaded = None
        try:
            import onnxruntime as ort
            from huggingface_hub import hf_hub_download

            def fetch(filename: str) -> str:
                return hf_hub_download(spec["repo"], filename, revision=spec["revision"])

            options = ort.SessionOptions()
            options.intra_op_num_threads = ALIGN_THREADS
            options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            # Only word requests use it, so don't leave threads spinning between calls.
            options.add_session_config_entry("session.intra_op.allow_spinning", "0")
            # ponytail: CPU only. wav2vec2 on CUDA (model_fp16.onnx) would cut ~1.8 s
            # per 30 s of audio to tens of ms on GPU hosts; untested, so not wired.
            session = ort.InferenceSession(
                fetch(spec["onnx"]), sess_options=options, providers=["CPUExecutionProvider"]
            )
            with open(fetch(spec["vocab"]), encoding="utf-8") as handle:
                loaded = (session, json.load(handle))
            logger.info("Loaded %s word aligner %s", language, spec["repo"])
        except Exception:
            # Cached as None so a missing model is logged once, not per request.
            logger.exception("word aligner for %r failed to load; keeping model word times", language)
        _loaded[language] = loaded
        return loaded


def _normalize(word: str) -> str:
    # ponytail: digits are spelled one by one ("2026" -> TWO ZERO TWO SIX). Speech
    # says "twenty twenty-six", so the word lands in the right place with looser
    # edges. Swap in a number-to-words pass if numbers need tight boundaries.
    return "".join(_DIGITS[int(c)] if c.isdigit() and c.isascii() else c for c in word.upper())


def _emission(session: Any, wav: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """CTC log-probs (T, V) for `wav`, and each frame's start time in seconds."""
    name = session.get_inputs()[0].name
    bounds = list(range(0, wav.size, _WINDOW)) + [wav.size]
    if len(bounds) > 2 and bounds[-1] - bounds[-2] < TARGET_SR:
        bounds.pop(-2)  # fold a short tail into the previous window
    logits, times = [], []
    for start, end in zip(bounds, bounds[1:]):
        piece = wav[start:end].astype(np.float32)
        if piece.size < _MIN_SAMPLES:
            continue
        piece = (piece - piece.mean()) / np.sqrt(piece.var() + 1e-7)  # do_normalize
        out = session.run(None, {name: piece[None, :]})[0][0]
        logits.append(out)
        times.append((start + _STRIDE * np.arange(out.shape[0])) / TARGET_SR)
    if not logits:
        return np.empty((0, 0), dtype=np.float32), np.empty(0)
    stacked = np.concatenate(logits).astype(np.float64)
    stacked -= stacked.max(axis=-1, keepdims=True)
    stacked -= np.log(np.exp(stacked).sum(axis=-1, keepdims=True))
    return stacked, np.concatenate(times)


def forced_align(
    emission: np.ndarray, targets: Sequence[int], blank: int = 0
) -> Optional[List[Tuple[int, int]]]:
    """Viterbi path of `targets` through CTC log-probs `emission` (T, V).

    Returns one (start_frame, end_frame_exclusive) per target, or None when the
    audio has too few frames to hold the targets.
    """
    n = len(targets)
    if n == 0:
        return []
    repeats = sum(a == b for a, b in zip(targets, targets[1:]))
    frames = emission.shape[0]
    if frames < n + repeats:  # repeats need a blank between them
        return None

    ext = np.full(2 * n + 1, blank, dtype=np.int64)  # blank, t0, blank, t1, ..., blank
    ext[1::2] = targets
    width = ext.size
    emit = emission[:, ext]
    can_skip = np.zeros(width, dtype=bool)  # jump over a blank into a new token
    can_skip[2:] = (ext[2:] != blank) & (ext[2:] != ext[:-2])
    columns = np.arange(width)

    back = np.zeros((frames, width), dtype=np.int8)  # 0 stay, 1 step, 2 skip
    score = np.full(width, _NEG_INF)
    score[:2] = emit[0, :2]
    for t in range(1, frames):
        step = np.concatenate(([_NEG_INF], score[:-1]))
        skip = np.where(can_skip, np.concatenate(([_NEG_INF, _NEG_INF], score[:-2])), _NEG_INF)
        options = np.stack((score, step, skip))
        back[t] = choice = options.argmax(axis=0)
        score = options[choice, columns] + emit[t]

    state = width - 1 if score[-1] >= score[-2] else width - 2
    path = np.empty(frames, dtype=np.int64)
    for t in range(frames - 1, -1, -1):
        path[t] = state
        state -= int(back[t, state])  # int(): int8 arithmetic would overflow past 127

    spans = []
    for index in range(n):
        hits = np.flatnonzero(path == 2 * index + 1)
        spans.append((int(hits[0]), int(hits[-1]) + 1))
    return spans


def align_words(
    wav: np.ndarray, words: Sequence[str], language: Optional[str] = None
) -> Optional[List[Optional[Span]]]:
    """(start, end) seconds from the start of `wav` for each of `words`.

    A word with no characters the model knows gets None. Returns None outright
    when no aligner is available or the audio cannot hold the text, so the
    caller keeps its own times.
    """
    loaded = _load(_language(language))
    if loaded is None:
        return None
    session, vocab = loaded
    separator = vocab.get("|")

    targets: List[int] = []
    owners: List[int] = []  # index into `words`; -1 for the word separator
    for index, word in enumerate(words):
        ids = [vocab[c] for c in _normalize(word) if c in vocab]
        if not ids:
            continue
        if targets and separator is not None:
            targets.append(separator)
            owners.append(-1)
        targets.extend(ids)
        owners.extend([index] * len(ids))
    if not targets:
        return None

    try:
        emission, times = _emission(session, wav)
        spans = forced_align(emission, targets, blank=vocab.get("<pad>", 0))
    except Exception:
        # The transcript is already done; a failed refinement must not fail it.
        logger.exception("word alignment failed; keeping model word times")
        return None
    if spans is None:
        return None

    aligned: List[Optional[Span]] = [None] * len(words)
    frame_sec = _STRIDE / TARGET_SR
    for (first, last), owner in zip(spans, owners):
        if owner < 0:
            continue
        start, end = float(times[first]), float(times[last - 1]) + frame_sec
        current = aligned[owner]
        aligned[owner] = (start, end) if current is None else (current[0], end)
    return aligned
