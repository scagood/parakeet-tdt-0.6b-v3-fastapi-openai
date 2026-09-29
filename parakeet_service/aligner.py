"""Word boundaries by CTC forced alignment (WhisperX-style, without torch).

Parakeet decides *what* was said; a character-level wav2vec2 CTC model, run
through ONNX Runtime, decides *when*. Parakeet's own word times sit on 80 ms
encoder frames and its word ends are estimated (see routes._WORD_TAIL_SEC).
Aligned times sit on 20 ms frames and the ends come from the audio.
"""
from __future__ import annotations

import json
import re
import string
import threading
import time
import unicodedata
from collections import OrderedDict
from typing import Any, Callable, Container, Optional, Sequence

import numpy as np
import onnxruntime as ort

from . import spoken
from .config import ALIGN_DEFAULT_LANGUAGE, ALIGN_THREADS, ALIGNER_CONFIGS, MODEL_CACHE_SIZE, TARGET_SR, logger
from .model import _build_sess_options

Span = tuple[float, float]

# --------------------------------------------------------------------------- #
# English text -> the characters wav2vec2-base-960h was trained on (A-Z, ').
# Parakeet writes numbers and symbols; spoken.py says them out, then this maps
# the result onto the model's alphabet.
# --------------------------------------------------------------------------- #
_SYMBOLS = {"%": " PERCENT", "&": " AND ", "+": " PLUS ", "@": " AT "}
_APOSTROPHES = str.maketrans({"’": "'", "‘": "'", "ʼ": "'"})
# Words whose letters don't sound like them to a character model: a lone letter
# is said as its name ("ten p" is "ten pee", "Plan B" is "plan bee").
_SAID_AS = {
    "NOUGHT": "NAWT",
    **dict(zip(
        "BCDEFGHJKLMNPQRSTUVWXYZ",
        "BEE SEE DEE EE EF JEE AYCH JAY KAY EL EM EN PEE KYOO AR ES TEE YOU VEE DOUBLEYOU EX WHY ZEE".split(),
    )),
}
# Letter names that depend on the speaker: the audio picks ("zee" or "zed").
# The transcript keeps the letter either way; only its timing listens for both.
_ACCENTS = {"ZEE": "ZED", "AYCH": "HAYCH"}


def _letters(text: str) -> str:
    """Spoken text as the aligner's letters: symbols said, accents folded (café ->
    CAFE), hyphens as word breaks, lone letters as their names; other punctuation
    is dropped by the vocab."""
    for symbol, said in _SYMBOLS.items():
        text = text.replace(symbol, said)
    text = unicodedata.normalize("NFKD", text.upper().replace("-", " "))
    text = "".join(c for c in text if not unicodedata.combining(c))
    return " ".join(_SAID_AS.get(token.strip(".,!?;:\"'()"), token) for token in text.split())


def _normalize_english(words: Sequence[str]) -> list[str]:
    """Spoken letters for each of `words`, in the order they are said."""
    said = spoken.spoken_words([word.translate(_APOSTROPHES) for word in words], everywhere=True)
    return [_letters(text) for text in said]


def _accent(part: str) -> str:
    """The other name of a speaker-dependent letter name ("ZEE" -> "ZED"), in
    the case `part` is in; any other part as it is."""
    other = _ACCENTS.get(part.upper())
    return part if other is None else other if part.isupper() else other.lower()


def _mostly_unknown(words: Sequence[str], said: Sequence[str], known: Container[str]) -> bool:
    """Whether most of `words` that have letters have none in `known` once
    said (`said`, one text per word): Cyrillic, Greek, ... in a model of Latin
    letters. That is not its language, and forcing it would be worse than
    leaving the words alone."""
    lettered = [text for word, text in zip(words, said) if any(c.isalpha() for c in word)]
    return 2 * sum(not any(c in known for c in text) for text in lettered) > len(lettered)


def other_alphabet(words: Sequence[str]) -> bool:
    """Whether `words` are mostly in an alphabet English is not written in, by
    the rule the aligner uses to leave them untimed (spoken numbers leave them
    as written too). Accents are folded first: "café" is Latin."""
    return _mostly_unknown(words, [_letters(word) for word in words], string.ascii_uppercase)


def _normalize_letters(words: Sequence[str]) -> list[str]:
    """Each of `words` as its letters, in any language: spelled the way
    Omnilingual ASR's training transcripts are (upstream's text_normalize,
    default config), NFKC, apostrophes and hyphens kept, other punctuation a
    word break."""
    said = []
    for word in words:
        text = unicodedata.normalize("NFKC", word).translate(_APOSTROPHES)
        # ponytail: digits are dropped, not said, so a number keeps its model times.
        text = "".join(c if c in "'-" or unicodedata.category(c)[0] in "LM" else " " for c in text)
        said.append(" ".join(part.strip("'-") for part in text.split()))
    return said


# --------------------------------------------------------------------------- #
# Models: the catalog's `aligners` (models.yaml)
# --------------------------------------------------------------------------- #
# An aligner's `normalisers`, run in order over a chunk's words (the whole list:
# a word's spoken form can depend on its neighbours, "$5 million").
_NORMALISERS: dict[str, Callable[[Sequence[str]], list[str]]] = {
    "english": _normalize_english,
    "letters": _normalize_letters,
    "upper": lambda said: [text.upper() for text in said],
    "lower": lambda said: [text.lower() for text in said],
}
# Full names OpenAI clients may send instead of ISO 639-1 codes.
_LANGUAGE_NAMES = {"english": "en"}

_STRIDE = 320  # wav2vec2's feature encoder emits one frame per 320 samples (20 ms)
_MIN_SAMPLES = 400  # receptive field of that encoder; shorter input has no frames
# wav2vec2 self-attention is O(frames²), so long chunks run in 30 s windows.
# Each window also sees 2 s of audio either side, and only its own 30 s of
# frames are kept, so no word is aligned without context. Both are multiples
# of _STRIDE so every window's frames land on one shared 20 ms grid.
_WINDOW = 30 * TARGET_SR
_CONTEXT = 2 * TARGET_SR
# Speech that Parakeet missed has to go somewhere on the forced path; without a
# place for it, the word before stretches over it (measured +720 ms for a missed
# "curiosity"). So the separator between words, plus one before the first and
# after the last, is a "star" state (as in MMS forced alignment): it also takes
# any frame at the best token's log-prob minus this penalty. Measured on the
# tutorial clip: clean, 0.1-2.0 keeps the correct transcript unchanged and
# every single missed word within 20 ms; with white noise at 5 dB SNR, a missed
# word still moves a neighbour >100 ms in 2/45 runs at 0.5 (5/45 at 1.0, 9/45
# at 2.0), while 0.25 starts taking frames from correct words.
_STAR_PENALTY = 0.5
# Choosing between readings of a number needs the opposite trade-off: with a
# cheap star the shortest reading wins ("ten p" over "and ten pence" that was
# said), so there the star costs more. Measured on a 246-clip TTS benchmark of
# alternative readings (3 voices, real Parakeet v3 transcripts): 230 right at
# 0.5, 235 at 1.0, 236 at 2.0-5.0; 3.0 sits mid-plateau.
_CHOICE_STAR_PENALTY = 3.0
_NEG_INF = -1e30
# A failed load (no network, no cached model) is retried after this long.
_RETRY_SEC = 300.0

_lock = threading.Lock()
_loaded: "OrderedDict[str, tuple[Any, dict[str, int]]]" = OrderedDict()
_failed_at: dict[str, float] = {}


def language_code(language: Optional[str]) -> str:
    """ISO 639-1 code for a request's `language` ("en-US", "English" -> "en")."""
    code = (language or "").strip().lower()
    if code in ("", "auto"):
        code = ALIGN_DEFAULT_LANGUAGE
    code = re.split(r"[-_]", code, maxsplit=1)[0]
    return _LANGUAGE_NAMES.get(code, code)


def aligns(name: str, language: Optional[str]) -> bool:
    """Whether aligner `name` (a catalog name) aligns a request's `language`."""
    return language_code(language) in ALIGNER_CONFIGS[name]["languages"]


def status() -> dict[str, str]:
    """Per-aligner, per-quantization state ("name:quant"), for /health."""
    keys = [f"{name}:{quant}" for name, spec in ALIGNER_CONFIGS.items() for quant in spec["quantizations"]]
    return {key: "loaded" if key in _loaded else "failed" if key in _failed_at else "not loaded" for key in keys}


def _read_vocab(files: dict[str, str]) -> dict[str, int]:
    """token -> id from the fetched vocab file: a vocab.json ({token: id}), or a
    tokens.txt of "token id" lines, where the token may itself be a space."""
    if "vocab.json" in files:
        with open(files["vocab.json"], encoding="utf-8") as handle:
            return json.load(handle)
    with open(files["tokens.txt"], encoding="utf-8") as handle:
        lines = (line.rstrip("\n").rpartition(" ") for line in handle)
        return {token: int(index) for token, _, index in lines}


def _load(name: str, quant: str) -> Optional[tuple[Any, dict[str, int]]]:
    key = f"{name}:{quant}"
    with _lock:
        if key in _loaded:
            _loaded.move_to_end(key)
            return _loaded[key]
        failed = _failed_at.get(key)
        if failed is not None and time.monotonic() - failed < _RETRY_SEC:
            return None
        spec = ALIGNER_CONFIGS[name]
        variant = spec["quantizations"][quant]
        try:
            from huggingface_hub import hf_hub_download

            files = {
                file: hf_hub_download(variant["repo"], path, revision=variant["revision"])
                for file, path in variant["files"].items()
            }
            # ponytail: CPU only. On CUDA (an fp16 export) the pass would drop from
            # ~1.8 s per 30 s of audio to tens of ms; untested, so not wired (#32).
            session = ort.InferenceSession(
                files["model.onnx"],
                # Only word and spoken-number requests use it: no threads spinning between calls.
                sess_options=_build_sess_options(ALIGN_THREADS, spinning=False),
                providers=["CPUExecutionProvider"],
            )
            vocab = _read_vocab(files)
            for token in (spec["blank"], spec["separator"]):
                if token is not None and token not in vocab:
                    raise KeyError(f"{token!r} is not in the vocab")
        except Exception:
            _failed_at[key] = time.monotonic()
            logger.exception(
                "word aligner %s failed to load; keeping model word times, retrying in %.0fs",
                key,
                _RETRY_SEC,
            )
            return None
        _failed_at.pop(key, None)
        _loaded[key] = loaded = (session, vocab)
        # The models' LRU cap (PARAKEET_MODEL_CACHE_SIZE), counted separately.
        while MODEL_CACHE_SIZE and len(_loaded) > MODEL_CACHE_SIZE:
            evicted, _ = _loaded.popitem(last=False)
            logger.info("Evicted word aligner %s (cache size %d)", evicted, MODEL_CACHE_SIZE)
        logger.info("Loaded word aligner %s (%s)", key, variant["repo"])
        return loaded


def _emission(session: Any, wav: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """CTC log-probs (T, V) for `wav`, and each frame's start time in seconds."""
    name = session.get_inputs()[0].name
    bounds = list(range(0, wav.size, _WINDOW)) + [wav.size]
    if len(bounds) > 2 and bounds[-1] - bounds[-2] < TARGET_SR:
        bounds.pop(-2)  # fold a short tail into the previous window
    log_probs, starts = [], []
    for core_start, core_end in zip(bounds, bounds[1:]):
        start = max(0, core_start - _CONTEXT)
        piece = wav[start : min(wav.size, core_end + _CONTEXT)].astype(np.float32)
        if piece.size < _MIN_SAMPLES:
            continue
        piece = (piece - piece.mean()) / np.sqrt(piece.var() + 1e-7)  # do_normalize
        out = session.run(None, {name: piece[None, :]})[0][0]
        frame_starts = start + _STRIDE * np.arange(out.shape[0])
        keep = (frame_starts >= core_start) & (frame_starts < core_end)
        # Log-softmax per window in float32: a float64 copy is 300 MB at 10k tokens.
        out = out[keep].astype(np.float32, copy=False)
        out -= out.max(axis=-1, keepdims=True)
        out -= np.log(np.exp(out).sum(axis=-1, keepdims=True))
        log_probs.append(out)
        starts.append(frame_starts[keep])
    if not log_probs:
        return np.empty((0, 0), np.float32), np.empty(0)
    return np.concatenate(log_probs), np.concatenate(starts) / TARGET_SR


def _viterbi(
    emission: np.ndarray, targets: Sequence[int], *, blank: int
) -> Optional[tuple[list[tuple[int, int]], float]]:
    """Viterbi path of `targets` through CTC log-probs `emission` (T, V).

    Returns one (start_frame, end_frame_exclusive) per target and the path's
    total log-prob, or None when the audio has too few frames to hold the targets.
    """
    n = len(targets)
    if n == 0:
        return [], 0.0
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

    best = score[-1] if score[-1] >= score[-2] else score[-2]
    state = width - 1 if score[-1] >= score[-2] else width - 2
    path = np.empty(frames, dtype=np.int64)
    for t in range(frames - 1, -1, -1):
        path[t] = state
        state -= int(back[t, state])  # int(): int8 arithmetic would overflow past 127

    spans = []
    for index in range(n):
        hits = np.flatnonzero(path == 2 * index + 1)
        spans.append((int(hits[0]), int(hits[-1]) + 1))
    return spans, float(best)


def _star_path(
    emission: np.ndarray,
    spoken: Sequence[tuple[int, Sequence[int]]],
    *,
    blank: int,
    separator: Optional[int],
    penalty: Optional[float] = None,
) -> Optional[tuple[list[tuple[int, int, int]], float]]:
    """Force-align spoken words between "star" separators (`penalty`: per frame
    the star takes, default _STAR_PENALTY).

    Returns (owner, first frame, end frame) per character, frames counted in
    `emission`, and the path's total log-prob; None if the audio cannot hold it.
    """
    if not spoken or emission.shape[0] == 0:  # no text, or audio too short for a frame
        return None
    anything = emission.max(axis=1) - (_STAR_PENALTY if penalty is None else penalty)
    stars = anything if separator is None else np.maximum(emission[:, separator], anything)
    # Widen only the columns the path can visit, not the whole vocab.
    tokens = sorted({blank, *(token for _owner, ids in spoken for token in ids)})
    column = {token: index for index, token in enumerate(tokens)}
    emission = np.column_stack([emission[:, tokens], stars]).astype(np.float64)
    star = len(tokens)
    # The edge stars must each take a frame; give them a free one either side so
    # speech starting on the first frame keeps it.
    pad = np.full((1, emission.shape[1]), _NEG_INF)
    pad[0, star] = 0.0
    emission = np.concatenate([pad, emission, pad])

    targets, owners = [star], [-1]
    for owner, ids in spoken:
        targets.extend([*(column[token] for token in ids), star])
        owners.extend([owner] * len(ids) + [-1])
    path = _viterbi(emission, targets, blank=column[blank])
    if path is None:
        return None
    spans, score = path
    # -1: the leading pad frame shifts every real frame by one
    return [(owner, first - 1, last - 1) for (first, last), owner in zip(spans, owners) if owner >= 0], score


def word_spans(
    emission: np.ndarray,
    frame_starts: np.ndarray,
    spoken: Sequence[tuple[int, Sequence[int]]],
    n_words: int,
    *,
    blank: int,
    separator: Optional[int],
) -> Optional[list[Optional[Span]]]:
    """Force-align spoken words and return (start, end) seconds per word.

    `spoken` holds (word index, character ids) in order; a word may appear more
    than once ("2026" is said as three words). Words absent from it get None.
    """
    path = _star_path(emission, spoken, blank=blank, separator=separator)
    if path is None:
        return None
    frame_sec = _STRIDE / TARGET_SR
    words: list[Optional[Span]] = [None] * n_words
    for owner, first, last in path[0]:
        start, end = float(frame_starts[first]), float(frame_starts[last - 1]) + frame_sec
        current = words[owner]
        words[owner] = (start, end) if current is None else (current[0], end)
    return words


class ChunkAligner:
    """One chunk's audio, ready to time its words or to hear which of several
    readings of a word was said. The wav2vec2 pass runs once, on first use,
    and a failed one is not retried.

    Everything here refines a finished transcript: failures are logged and
    answered with "don't know" (None, or the first reading), never raised.
    """

    def __init__(self, wav: np.ndarray, session: Any, vocab: dict[str, int], spec: dict[str, Any]):
        self._wav = wav
        self._session = session
        self._vocab = vocab
        self._steps = [_NORMALISERS[step] for step in spec["normalisers"]]
        self._letter_names = "english" in spec["normalisers"]  # "zee" or "zed"
        self._blank = vocab[spec["blank"]]
        self._separator = None if spec["separator"] is None else vocab[spec["separator"]]
        self._frames: Optional[tuple[np.ndarray, np.ndarray]] = None

    def _normalize(self, words: Sequence[str]) -> list[str]:
        """`words` through the aligner's normaliser steps, in order."""
        said = list(words)
        for step in self._steps:
            said = step(said)
        return said

    def _emission(self) -> tuple[np.ndarray, np.ndarray]:
        if self._frames is None:
            try:
                self._frames = _emission(self._session, self._wav)
            except Exception:
                # Once per chunk, not once per question: with no frames, every
                # later span or score of this chunk is "don't know".
                logger.exception("wav2vec2 pass failed; keeping model word times and default readings")
                self._frames = np.empty((0, 0)), np.empty(0)
        return self._frames

    def _spoken(self, said: Sequence[str]) -> list[tuple[int, list[int]]]:
        """(word index, character ids) for each spoken part of each word's `said` text."""
        spoken: list[tuple[int, list[int]]] = []
        for index, text in enumerate(said):
            for part in text.split():
                if ids := [self._vocab[c] for c in part if c in self._vocab]:
                    spoken.append((index, ids))
        return spoken

    def spans(self, words: Sequence[str]) -> Optional[list[Optional[Span]]]:
        """(start, end) seconds from the chunk start for each of `words`.

        A word with no characters the model knows gets None. None outright when
        the text is not in the model's alphabet or the audio cannot hold it.
        """
        try:
            said = self._normalize(words)
            spoken = self._spoken(said)
            if not spoken or _mostly_unknown(words, said, self._vocab):
                return None
            emission, frame_starts = self._emission()
            timed = word_spans(
                emission, frame_starts, spoken, len(words), blank=self._blank, separator=self._separator
            )
            if timed and self._letter_names and (accented := self._accented(said, timed)) != said:
                timed = word_spans(
                    emission, frame_starts, self._spoken(accented), len(words),
                    blank=self._blank, separator=self._separator,
                )
            return timed
        except Exception:
            logger.exception("word alignment failed; keeping model word times")
            return None

    def _accented(self, said: Sequence[str], timed: Sequence[Optional[Span]]) -> list[str]:
        """`said` with each lone letter named the way the audio says it ("zee" or
        "zed"), heard between the words either side of it."""
        said = list(said)
        for index, text in enumerate(said):
            other = " ".join(_accent(part) for part in text.split())
            if other == text or timed[index] is None:
                continue
            start = max((span[1] for span in timed[:index] if span), default=0.0)
            end = min((span[0] for span in timed[index + 1 :] if span), default=float("inf"))
            default, accent = self.scores([text, other], start, end)
            said[index] = other if accent > default else text
        return said

    def scores(self, options: Sequence[str], start: float, end: float) -> list[float]:
        """How well each reading in `options` matches the audio between `start`
        and `end` seconds: its best path's log-prob, -inf where it does not fit
        (all -inf if it cannot tell).

        Every reading is scored over the same frames, with the star states on
        either side: a reading that leaves out a spoken word pays for the audio
        it cannot explain, one that adds a word has to find letters for it.
        """
        try:
            emission, frame_starts = self._emission()
            window = emission[(frame_starts >= start) & (frame_starts < end)]
            scores = []
            for option in options:
                path = _star_path(
                    window,
                    self._spoken(self._normalize([option])),
                    blank=self._blank,
                    separator=self._separator,
                    penalty=_CHOICE_STAR_PENALTY,
                )
                scores.append(float("-inf") if path is None else path[1])
            return scores
        except Exception:
            logger.exception("reading choice failed; keeping the default reading")
            return [float("-inf")] * len(options)

    def best(self, options: Sequence[str], start: float, end: float) -> int:
        """Index of the reading in `options` that best matches the audio between
        `start` and `end` seconds (0, the default reading, if it cannot tell)."""
        return int(np.argmax(self.scores(options, start, end)))


def for_chunk(
    wav: np.ndarray, language: Optional[str] = None, name: Optional[str] = None, quantization: Optional[str] = None
) -> Optional[ChunkAligner]:
    """A ChunkAligner for `wav` from aligner `name` at `quantization` (else its
    default_quantization), or None when the request named no aligner, it does
    not align `language`, or it could not load."""
    if name is None or not aligns(name, language):
        return None
    spec = ALIGNER_CONFIGS[name]
    loaded = _load(name, quantization or spec["default_quantization"])
    if loaded is None:
        return None
    session, vocab = loaded
    return ChunkAligner(wav, session, vocab, spec)
