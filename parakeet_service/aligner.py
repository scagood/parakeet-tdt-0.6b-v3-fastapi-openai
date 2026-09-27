"""Word boundaries by CTC forced alignment (WhisperX-style, without torch).

Parakeet decides *what* was said; a character-level wav2vec2 CTC model, run
through ONNX Runtime, decides *when*. Parakeet's own word times sit on 80 ms
encoder frames and its word ends are estimated (see routes._WORD_TAIL_SEC).
Aligned times sit on 20 ms frames and the ends come from the audio.
"""
from __future__ import annotations

import json
import re
import threading
import time
import unicodedata
from dataclasses import dataclass
from typing import Any, Callable, Optional, Sequence

import numpy as np
import onnxruntime as ort

from . import spoken
from .config import ALIGN_DEFAULT_LANGUAGE, ALIGN_THREADS, ALIGN_WORDS, TARGET_SR, logger
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
    **dict(zip("BCDEFGHJKLMNPQRSTUVWXYZ", "BEE SEE DEE EE EF JEE AYCH JAY KAY EL EM EN PEE KYOO AR ES TEE YOU VEE DOUBLEYOU EX WHY ZEE".split())),
}


def _letters(text: str) -> str:
    """Spoken text as the aligner's letters: symbols said, accents folded (café ->
    CAFE), hyphens as word breaks, lone letters as their names; other punctuation
    is dropped by the vocab."""
    for symbol, said in _SYMBOLS.items():
        text = text.replace(symbol, said)
    text = unicodedata.normalize("NFKD", text.upper().replace("-", " "))
    text = "".join(c for c in text if not unicodedata.combining(c))
    return " ".join(_SAID_AS.get(token.strip(".,!?;:\"'()"), token) for token in text.split())


def _spoken_english(word: str) -> str:
    """One Parakeet word as spoken English letters, spaces between spoken words."""
    return _letters(spoken.spoken_word(word.translate(_APOSTROPHES), everywhere=True))


def _normalize_english(words: Sequence[str]) -> list[str]:
    """Spoken letters for each of `words`, in the order they are said."""
    said = spoken.spoken_words([word.translate(_APOSTROPHES) for word in words], everywhere=True)
    return [_letters(text) for text in said]


# --------------------------------------------------------------------------- #
# Models
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class AlignModel:
    """A char-level wav2vec2 CTC export (ONNX + vocab.json) for one language."""

    repo: str
    revision: str
    onnx: str
    vocab: str
    # All of a chunk's words -> one spoken string per word. Takes the whole list
    # because a word's spoken form can depend on its neighbours ("$5 million").
    normalize: Callable[[Sequence[str]], list[str]]


# Adding a language is adding an entry here.
ALIGN_MODELS: dict[str, AlignModel] = {
    "en": AlignModel(
        repo="onnx-community/wav2vec2-base-960h-ONNX",
        revision="729c1a6730fb549c20a1c73a3d3f96f11020225e",
        onnx="onnx/model_int8.onnx",
        vocab="vocab.json",
        normalize=_normalize_english,
    ),
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
_loaded: dict[str, tuple[Any, dict[str, int]]] = {}
_failed_at: dict[str, float] = {}


def language_code(language: Optional[str]) -> str:
    """ISO 639-1 code for a request's `language` ("en-US", "English" -> "en")."""
    code = (language or "").strip().lower()
    if code in ("", "auto"):
        code = ALIGN_DEFAULT_LANGUAGE
    code = re.split(r"[-_]", code, maxsplit=1)[0]
    return _LANGUAGE_NAMES.get(code, code)


def supports(language: Optional[str]) -> bool:
    return ALIGN_WORDS and language_code(language) in ALIGN_MODELS


def status() -> dict[str, str]:
    """Per-language aligner state, for /health."""
    if not ALIGN_WORDS:
        return {code: "disabled" for code in ALIGN_MODELS}
    return {
        code: "loaded" if code in _loaded else "failed" if code in _failed_at else "not loaded"
        for code in ALIGN_MODELS
    }


def _load(language: str) -> Optional[tuple[Any, dict[str, int]]]:
    with _lock:
        if language in _loaded:
            return _loaded[language]
        failed = _failed_at.get(language)
        if failed is not None and time.monotonic() - failed < _RETRY_SEC:
            return None
        spec = ALIGN_MODELS[language]
        try:
            from huggingface_hub import hf_hub_download

            def fetch(filename: str) -> str:
                return hf_hub_download(spec.repo, filename, revision=spec.revision)

            # ponytail: CPU only. wav2vec2 on CUDA (model_fp16.onnx) would cut ~1.8 s
            # per 30 s of audio to tens of ms on GPU hosts; untested, so not wired.
            session = ort.InferenceSession(
                fetch(spec.onnx),
                # Only word requests use it, so don't leave threads spinning between calls.
                sess_options=_build_sess_options(ALIGN_THREADS, spinning=False),
                providers=["CPUExecutionProvider"],
            )
            with open(fetch(spec.vocab), encoding="utf-8") as handle:
                vocab = json.load(handle)
        except Exception:
            _failed_at[language] = time.monotonic()
            logger.exception(
                "word aligner for %r failed to load; keeping model word times, retrying in %.0fs",
                language,
                _RETRY_SEC,
            )
            return None
        _failed_at.pop(language, None)
        _loaded[language] = (session, vocab)
        logger.info("Loaded %s word aligner %s", language, spec.repo)
        return _loaded[language]


def _emission(session: Any, wav: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """CTC log-probs (T, V) for `wav`, and each frame's start time in seconds."""
    name = session.get_inputs()[0].name
    bounds = list(range(0, wav.size, _WINDOW)) + [wav.size]
    if len(bounds) > 2 and bounds[-1] - bounds[-2] < TARGET_SR:
        bounds.pop(-2)  # fold a short tail into the previous window
    logits, starts = [], []
    for core_start, core_end in zip(bounds, bounds[1:]):
        start = max(0, core_start - _CONTEXT)
        piece = wav[start : min(wav.size, core_end + _CONTEXT)].astype(np.float32)
        if piece.size < _MIN_SAMPLES:
            continue
        piece = (piece - piece.mean()) / np.sqrt(piece.var() + 1e-7)  # do_normalize
        out = session.run(None, {name: piece[None, :]})[0][0]
        frame_starts = start + _STRIDE * np.arange(out.shape[0])
        keep = (frame_starts >= core_start) & (frame_starts < core_end)
        logits.append(out[keep])
        starts.append(frame_starts[keep])
    if not logits:
        return np.empty((0, 0)), np.empty(0)
    stacked = np.concatenate(logits).astype(np.float64)
    stacked -= stacked.max(axis=-1, keepdims=True)
    stacked -= np.log(np.exp(stacked).sum(axis=-1, keepdims=True))
    return stacked, np.concatenate(starts) / TARGET_SR


def forced_align(
    emission: np.ndarray, targets: Sequence[int], *, blank: int
) -> Optional[list[tuple[int, int]]]:
    """Viterbi path of `targets` through CTC log-probs `emission` (T, V).

    Returns one (start_frame, end_frame_exclusive) per target, or None when the
    audio has too few frames to hold the targets.
    """
    path = _viterbi(emission, targets, blank=blank)
    return None if path is None else path[0]


def _viterbi(
    emission: np.ndarray, targets: Sequence[int], *, blank: int
) -> Optional[tuple[list[tuple[int, int]], float]]:
    """forced_align(), plus the best path's total log-prob."""
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
    star = emission.shape[1]
    anything = emission.max(axis=1) - (_STAR_PENALTY if penalty is None else penalty)
    column = anything if separator is None else np.maximum(emission[:, separator], anything)
    emission = np.concatenate([emission, column[:, None]], axis=1)
    # The edge stars must each take a frame; give them a free one either side so
    # speech starting on the first frame keeps it.
    pad = np.full((1, emission.shape[1]), _NEG_INF)
    pad[0, star] = 0.0
    emission = np.concatenate([pad, emission, pad])

    targets, owners = [star], [-1]
    for owner, ids in spoken:
        targets.extend([*ids, star])
        owners.extend([owner] * len(ids) + [-1])
    path = _viterbi(emission, targets, blank=blank)
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
    readings of a word was said. The wav2vec2 pass runs once, on first use.

    Everything here refines a finished transcript: failures are logged and
    answered with "don't know" (None, or the first reading), never raised.
    """

    def __init__(self, wav: np.ndarray, session: Any, vocab: dict[str, int], normalize):
        self._wav = wav
        self._session = session
        self._vocab = vocab
        self._normalize = normalize
        self._blank = vocab.get("<pad>", 0)
        self._separator = vocab.get("|")
        self._frames: Optional[tuple[np.ndarray, np.ndarray]] = None

    def _emission(self) -> tuple[np.ndarray, np.ndarray]:
        if self._frames is None:
            self._frames = _emission(self._session, self._wav)
        return self._frames

    def _spoken(self, words: Sequence[str]) -> list[tuple[int, list[int]]]:
        spoken: list[tuple[int, list[int]]] = []
        for index, text in enumerate(self._normalize(words)):
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
            spoken = self._spoken(words)
            lettered = [i for i, word in enumerate(words) if any(c.isalpha() for c in word)]
            placed = {owner for owner, _ids in spoken}
            unplaceable = sum(i not in placed for i in lettered)
            # Mostly letters the model has never seen (Cyrillic, Greek, ...): this is
            # not its language, and forcing it would be worse than the model's times.
            if not spoken or unplaceable * 2 > len(lettered):
                return None
            emission, frame_starts = self._emission()
            return word_spans(
                emission, frame_starts, spoken, len(words), blank=self._blank, separator=self._separator
            )
        except Exception:
            logger.exception("word alignment failed; keeping model word times")
            return None

    def best(self, options: Sequence[str], start: float, end: float) -> int:
        """Index of the reading in `options` that best matches the audio between
        `start` and `end` seconds (0 if it cannot tell).

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
                    self._spoken([option]),
                    blank=self._blank,
                    separator=self._separator,
                    penalty=_CHOICE_STAR_PENALTY,
                )
                scores.append(_NEG_INF if path is None else path[1])
            return int(np.argmax(scores)) if max(scores) > _NEG_INF else 0
        except Exception:
            logger.exception("reading choice failed; keeping the default reading")
            return 0


def for_chunk(wav: np.ndarray, language: Optional[str] = None) -> Optional[ChunkAligner]:
    """A ChunkAligner for `wav`, or None when no aligner serves `language`."""
    if not supports(language):
        return None
    code = language_code(language)
    loaded = _load(code)
    if loaded is None:
        return None
    session, vocab = loaded
    return ChunkAligner(wav, session, vocab, ALIGN_MODELS[code].normalize)


def align_words(
    wav: np.ndarray, words: Sequence[str], language: Optional[str] = None
) -> Optional[list[Optional[Span]]]:
    """(start, end) seconds from the start of `wav` for each of `words`, or None
    when no aligner is available or it cannot place them."""
    chunk = for_chunk(wav, language)
    return None if chunk is None else chunk.spans(words)
