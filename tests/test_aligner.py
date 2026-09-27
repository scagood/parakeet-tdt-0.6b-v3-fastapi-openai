from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from parakeet_service import aligner, routes
from parakeet_service.config import TARGET_SR


def _emission(runs, frames, vocab_size=5, blank=0):
    """Log-probs where each (token, n_frames) run dominates its frames; the rest is blank."""
    logits = np.zeros((frames, vocab_size))
    t = 0
    for token, count in runs:
        logits[t : t + count, token] = 8.0
        t += count
    logits[t:, blank] = 8.0
    return logits - np.log(np.exp(logits).sum(axis=1, keepdims=True))


def test_forced_align_recovers_token_spans():
    emission = _emission([(0, 2), (1, 3), (2, 4), (3, 3)], frames=15)
    assert aligner.forced_align(emission, [1, 2, 3]) == [(2, 5), (5, 9), (9, 12)]


def test_forced_align_keeps_repeated_tokens_apart():
    emission = _emission([(1, 3), (0, 1), (1, 3)], frames=7)
    first, second = aligner.forced_align(emission, [1, 1])
    assert first[1] <= second[0]


def test_forced_align_handles_long_transcripts():
    # 200 tokens -> a 401-state lattice, past what int8 state arithmetic can hold.
    targets = [1 + (i % 3) for i in range(200)]
    emission = _emission([(token, 2) for token in targets], frames=410)
    spans = aligner.forced_align(emission, targets)
    assert spans == [(2 * i, 2 * i + 2) for i in range(200)]


def test_forced_align_rejects_too_few_frames():
    # "aa" needs a blank between the two tokens: three frames minimum.
    assert aligner.forced_align(_emission([(1, 2)], frames=2), [1, 1]) is None


def test_normalize_spells_digits():
    assert aligner._normalize("it's 42") == "IT'S FOURTWO"


def test_apply_alignment_keeps_unplaced_words_in_order():
    words = [
        {"word": "a", "start": 0.0, "end": 0.3},
        {"word": "%", "start": 0.1, "end": 0.9},  # nothing the aligner can anchor
        {"word": "b", "start": 0.5, "end": 0.8},
    ]
    routes._apply_alignment(words, [(0.10, 0.20), None, (0.40, 0.60)], 10.0, 20.0)
    assert (words[0]["start"], words[0]["end"]) == (10.1, 10.2)
    assert (words[2]["start"], words[2]["end"]) == (10.4, 10.6)
    assert words[0]["end"] <= words[1]["start"] <= words[1]["end"] <= words[2]["start"]


def test_stitch_uses_aligner_times_per_chunk():
    ranges = [(0, 2 * TARGET_SR), (5 * TARGET_SR, 7 * TARGET_SR)]
    pieces = ["chunk0", "chunk1"]
    prepared = routes._PreparedAudio(waveform=None, ranges=ranges, pieces=pieces, duration=7.0)
    results = [
        SimpleNamespace(text="hi there", tokens=[" hi", " there"], timestamps=[0.0, 0.8]),
        SimpleNamespace(text="bye", tokens=[" bye"], timestamps=[0.0]),
    ]
    calls = []

    def fake_align(piece, words):
        calls.append((piece, words))
        return [(0.25 + i, 0.5 + i) for i in range(len(words))]

    _text, _segments, words = routes._stitch(prepared, results, fake_align)
    assert calls == [("chunk0", ["hi", "there"]), ("chunk1", ["bye"])]
    assert [(w["word"], w["start"], w["end"]) for w in words] == [
        ("hi", 0.25, 0.5),
        ("there", 1.25, 1.5),
        ("bye", 5.25, 5.5),
    ]


def test_stitch_keeps_model_times_when_aligner_unavailable():
    prepared = routes._PreparedAudio(
        waveform=None, ranges=[(0, 2 * TARGET_SR)], pieces=[None], duration=2.0
    )
    results = [SimpleNamespace(text="hi", tokens=[" hi"], timestamps=[0.4])]
    plain = routes._stitch(prepared, results)
    aligned = routes._stitch(prepared, results, lambda _piece, _words: None)
    assert plain == aligned
