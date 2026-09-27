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

from .config import ALIGN_DEFAULT_LANGUAGE, ALIGN_THREADS, TARGET_SR, logger
from .model import _build_sess_options

Span = tuple[float, float]

# --------------------------------------------------------------------------- #
# English text -> the characters wav2vec2-base-960h was trained on (A-Z, ').
# Parakeet writes numbers and symbols; the aligner needs them as spoken words.
# --------------------------------------------------------------------------- #
_ONES = (
    "ZERO ONE TWO THREE FOUR FIVE SIX SEVEN EIGHT NINE TEN ELEVEN TWELVE THIRTEEN "
    "FOURTEEN FIFTEEN SIXTEEN SEVENTEEN EIGHTEEN NINETEEN"
).split()
_TENS = "_ _ TWENTY THIRTY FORTY FIFTY SIXTY SEVENTY EIGHTY NINETY".split()
_SCALES = ((10**12, "TRILLION"), (10**9, "BILLION"), (10**6, "MILLION"), (1000, "THOUSAND"))
_SCALE_WORDS = {"HUNDRED", "THOUSAND", "MILLION", "BILLION", "TRILLION"}
_ORDINALS = {
    "ONE": "FIRST", "TWO": "SECOND", "THREE": "THIRD", "FIVE": "FIFTH",
    "EIGHT": "EIGHTH", "NINE": "NINTH", "TWELVE": "TWELFTH",
}
_CURRENCIES = {"$": "DOLLAR", "£": "POUND", "€": "EURO"}
_UNITS = {unit + plural for unit in _CURRENCIES.values() for plural in ("", "S")}
_SYMBOLS = {"%": " PERCENT", "&": " AND ", "+": " PLUS ", "@": " AT "}
_APOSTROPHES = str.maketrans({"’": "'", "‘": "'", "ʼ": "'"})
# Unit abbreviations after a number, as (singular, plural) spoken words. What
# matters for timing is the sound: "£25", "25 lb" and "25lb" are all "twenty
# five pounds", and ASR often writes money as weight or the other way round.
_POUNDS = ("POUND", "POUNDS")
_KM_PER_HOUR = ("KILOMETER PER HOUR", "KILOMETERS PER HOUR")
_UNIT_WORDS = {
    "lb": _POUNDS, "lbs": _POUNDS,
    "oz": ("OUNCE", "OUNCES"),
    "kg": ("KILOGRAM", "KILOGRAMS"), "kgs": ("KILOGRAM", "KILOGRAMS"),
    "km": ("KILOMETER", "KILOMETERS"),
    "cm": ("CENTIMETER", "CENTIMETERS"),
    "mm": ("MILLIMETER", "MILLIMETERS"),
    "ml": ("MILLILITER", "MILLILITERS"),
    "ft": ("FOOT", "FEET"),
    "mph": ("MILE PER HOUR", "MILES PER HOUR"),
    "kph": _KM_PER_HOUR, "km/h": _KM_PER_HOUR,
    "hr": ("HOUR", "HOURS"), "hrs": ("HOUR", "HOURS"),
    "min": ("MINUTE", "MINUTES"), "mins": ("MINUTE", "MINUTES"),
    "°c": ("DEGREE CELSIUS", "DEGREES CELSIUS"),
    "°f": ("DEGREE FAHRENHEIT", "DEGREES FAHRENHEIT"),
    "°": ("DEGREE", "DEGREES"),
}
# Scale abbreviations only count after a currency: "$5m" is five million
# dollars, but a bare "5m" could be metres, so it keeps its letter.
_SCALE_ABBREVIATIONS = {"k": "THOUSAND", "m": "MILLION", "bn": "BILLION"}
_SUFFIXES = "|".join(
    re.escape(s) for s in sorted({*_UNIT_WORDS, *_SCALE_ABBREVIATIONS}, key=len, reverse=True)
)
# A leading minus counts only at the start of a word: "mid-2020s" is not negative.
_NUMBER = re.compile(
    r"(?:(?<!\w)(-))?([$£€])?(\d{1,3}(?:,\d{3})+|\d+)(\.\d+)?(st|nd|rd|th)?"
    rf"(?:({_SUFFIXES})(?![a-z]))?",
    re.IGNORECASE,
)
_TRAILING_PUNCTUATION = re.compile(r"[.,!?;:]+$")


def _cardinal(n: int) -> str:
    if n < 20:
        return _ONES[n]
    if n < 100:
        return _TENS[n // 10] + ("" if n % 10 == 0 else " " + _ONES[n % 10])
    if n < 1000:
        rest = "" if n % 100 == 0 else " " + _cardinal(n % 100)
        return f"{_ONES[n // 100]} HUNDRED{rest}"
    for scale, name in _SCALES:
        if n >= scale:
            head, rest = divmod(n, scale)
            return f"{_cardinal(head)} {name}" + ("" if rest == 0 else " " + _cardinal(rest))
    raise AssertionError("unreachable")


def _year(n: int) -> str:
    # 1999 -> NINETEEN NINETY NINE, 1905 -> NINETEEN OH FIVE, 1900 -> NINETEEN HUNDRED
    head, tail = divmod(n, 100)
    if tail == 0:
        return f"{_cardinal(head)} HUNDRED"
    return f"{_cardinal(head)} {'OH ' if tail < 10 else ''}{_cardinal(tail)}"


def _ordinal(words: str) -> str:
    head, _, last = words.rpartition(" ")
    if last in _ORDINALS:
        last = _ORDINALS[last]
    elif last.endswith("Y"):
        last = last[:-1] + "IETH"
    else:
        last += "TH"
    return f"{head} {last}".strip()


def _digits(digits: str) -> str:
    return " ".join(_ONES[int(d)] for d in digits)


def _unit(key: str, count: str) -> str:
    singular, plural = _UNIT_WORDS[key]
    return singular if count == "1" else plural


def _say_number(match: re.Match) -> str:
    sign, currency, digits, decimals, ordinal, suffix = match.groups()
    suffix = (suffix or "").lower()
    scale = _SCALE_ABBREVIATIONS.get(suffix) if currency else None
    plain = digits.replace(",", "")
    if len(plain) > 15 or ("," not in digits and plain.startswith("0") and len(plain) > 1):
        # Codes, IDs and phone numbers are read digit by digit.
        spoken = _digits(plain)
    elif len(digits) == 4 and not (currency or ordinal or decimals) and (
        1100 <= int(plain) <= 1999 or 2010 <= int(plain) <= 2099
    ):
        # ponytail: 4-digit numbers in year range read as years ("twenty twenty
        # six"). A count said "one thousand five hundred" then starts ~160 ms
        # late (measured). Needs context to tell a year from a count.
        spoken = _year(int(plain))
    else:
        spoken = _cardinal(int(plain))
    if ordinal:
        spoken = _ordinal(spoken)
    fraction = (decimals or ".")[1:]
    if fraction and not (currency and len(fraction) == 2 and not scale):
        # "$2.5 million" is "two point five million dollars"; only 2 digits are cents
        spoken += " POINT " + _digits(fraction)
        fraction = ""
    if scale:
        spoken += " " + scale
    if currency:
        unit = _CURRENCIES[currency] + ("" if plain == "1" and not decimals and not scale else "S")
        spoken = f"{spoken} {unit}" + (f" {_cardinal(int(fraction))}" if fraction.strip("0") else "")
    if suffix in _UNIT_WORDS:
        spoken += " " + _unit(suffix, digits if not decimals else "")
    elif suffix and not scale:
        spoken += " " + suffix.upper()  # "5m", "5k": left as written
    if sign:
        spoken = "MINUS " + spoken
    return f" {spoken} "


def _spoken_english(word: str) -> str:
    """One Parakeet word as spoken English letters, spaces between spoken words.

    Numbers, currency and symbols are said out; accents are folded (café ->
    CAFE); remaining punctuation is dropped by the caller's vocab filter.
    """
    text = _NUMBER.sub(_say_number, word.translate(_APOSTROPHES))
    for symbol, spoken in _SYMBOLS.items():
        text = text.replace(symbol, spoken)
    text = unicodedata.normalize("NFKD", text.upper())
    return "".join(c for c in text if not unicodedata.combining(c))


def _normalize_english(words: Sequence[str]) -> list[str]:
    """Spoken letters for each of `words`, in the order they are said."""
    spoken = [_spoken_english(word).split() for word in words]
    # A unit written as its own word after a number: "25 lb" is "twenty five pounds".
    for index in range(1, len(words)):
        key = _TRAILING_PUNCTUATION.sub("", words[index]).lower()
        previous = _TRAILING_PUNCTUATION.sub("", words[index - 1])
        if key in _UNIT_WORDS and any(c.isdigit() for c in previous):
            spoken[index] = _unit(key, previous).split()
    # "$5 million" is said "five million dollars": the unit follows the scale word.
    for current, following in zip(spoken, spoken[1:]):
        if (
            len(current) > 1
            and current[-1] in _UNITS
            and following
            and re.sub(r"[^A-Z]", "", following[0]) in _SCALE_WORDS
        ):
            unit = current.pop()
            following.insert(1, unit if unit.endswith("S") else unit + "S")
    return [" ".join(parts) for parts in spoken]


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
_NEG_INF = -1e30
# A failed load (no network, no cached model) is retried after this long.
_RETRY_SEC = 300.0

_lock = threading.Lock()
_loaded: dict[str, tuple[Any, dict[str, int]]] = {}
_failed_at: dict[str, float] = {}


def _language(language: Optional[str]) -> str:
    """ISO 639-1 code for a request's `language` ("en-US", "English" -> "en")."""
    code = (language or "").strip().lower()
    if code in ("", "auto"):
        code = ALIGN_DEFAULT_LANGUAGE
    code = re.split(r"[-_]", code, maxsplit=1)[0]
    return _LANGUAGE_NAMES.get(code, code)


def supports(language: Optional[str]) -> bool:
    return _language(language) in ALIGN_MODELS


def status() -> dict[str, str]:
    """Per-language aligner state, for /health."""
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
    if not spoken or emission.shape[0] == 0:  # no text, or audio too short for a frame
        return None
    star = emission.shape[1]
    anything = emission.max(axis=1) - _STAR_PENALTY
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
    spans = forced_align(emission, targets, blank=blank)
    if spans is None:
        return None

    frame_sec = _STRIDE / TARGET_SR
    words: list[Optional[Span]] = [None] * n_words
    for (first, last), owner in zip(spans, owners):
        if owner < 0:
            continue
        # -1: the leading pad frame shifts every real frame by one
        start = float(frame_starts[first - 1])
        end = float(frame_starts[last - 2]) + frame_sec
        current = words[owner]
        words[owner] = (start, end) if current is None else (current[0], end)
    return words


def align_words(
    wav: np.ndarray, words: Sequence[str], language: Optional[str] = None
) -> Optional[list[Optional[Span]]]:
    """(start, end) seconds from the start of `wav` for each of `words`.

    A word with no characters the model knows gets None. Returns None outright
    when no aligner is available, the text is not in the model's alphabet, or
    the audio cannot hold the text, so the caller keeps its own times.
    """
    code = _language(language)
    loaded = _load(code)
    if loaded is None:
        return None
    session, vocab = loaded

    # Everything below is refinement of a finished transcript: any failure
    # falls back to the model's own times rather than failing the request.
    try:
        spoken: list[tuple[int, list[int]]] = []
        lettered = unplaceable = 0
        for index, text in enumerate(ALIGN_MODELS[code].normalize(words)):
            parts = [ids for part in text.split() if (ids := [vocab[c] for c in part if c in vocab])]
            spoken.extend((index, ids) for ids in parts)
            if any(c.isalpha() for c in words[index]):
                lettered += 1
                unplaceable += not parts
        # Mostly letters the model has never seen (Cyrillic, Greek, ...): this is
        # not its language, and forcing it would be worse than the model's times.
        if not spoken or unplaceable * 2 > lettered:
            return None
        emission, frame_starts = _emission(session, wav)
        return word_spans(
            emission,
            frame_starts,
            spoken,
            len(words),
            blank=vocab.get("<pad>", 0),
            separator=vocab.get("|"),
        )
    except Exception:
        # The transcript is already done; a failed refinement must not fail it.
        logger.exception("word alignment failed; keeping model word times")
        return None
