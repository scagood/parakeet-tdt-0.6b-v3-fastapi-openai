from __future__ import annotations

import sys
from types import SimpleNamespace

import numpy as np
import pytest

from parakeet_service import aligner, model
from parakeet_service.config import ORT_INTRA_THREADS, TARGET_SR

BLANK, A, B, C, SEP = 0, 1, 2, 3, 4


def _emission(runs, frames, vocab_size=5):
    """Log-probs where each (token, n_frames) run dominates its frames; the rest is blank."""
    logits = np.zeros((frames, vocab_size))
    t = 0
    for token, count in runs:
        logits[t : t + count, token] = 8.0
        t += count
    logits[t:, BLANK] = 8.0
    return logits - np.log(np.exp(logits).sum(axis=1, keepdims=True))


def test_forced_align_recovers_token_spans():
    emission = _emission([(BLANK, 2), (A, 3), (B, 4), (C, 3)], frames=15)
    assert aligner.forced_align(emission, [A, B, C], blank=BLANK) == [(2, 5), (5, 9), (9, 12)]


def test_forced_align_keeps_repeated_tokens_apart():
    emission = _emission([(A, 3), (BLANK, 1), (A, 3)], frames=7)
    first, second = aligner.forced_align(emission, [A, A], blank=BLANK)
    assert first[1] <= second[0]


def test_forced_align_handles_long_transcripts():
    # 200 tokens -> a 401-state lattice, past what int8 state arithmetic can hold.
    targets = [1 + (i % 3) for i in range(200)]
    emission = _emission([(token, 2) for token in targets], frames=410)
    spans = aligner.forced_align(emission, targets, blank=BLANK)
    assert spans == [(2 * i, 2 * i + 2) for i in range(200)]


def test_forced_align_rejects_too_few_frames():
    # "aa" needs a blank between the two tokens: three frames minimum.
    assert aligner.forced_align(_emission([(A, 2)], frames=2), [A, A], blank=BLANK) is None


def _frame_starts(frames):
    return np.arange(frames) * aligner._STRIDE / TARGET_SR


def test_word_spans_do_not_absorb_a_missed_word():
    # "A <C C C, which the transcript lacks> B": A must not stretch over the Cs.
    runs = [(BLANK, 2), (A, 3), (BLANK, 1), (C, 2), (BLANK, 1), (C, 2), (BLANK, 1), (C, 2), (B, 3)]
    emission = _emission(runs, frames=20)
    spans = aligner.word_spans(
        emission, _frame_starts(20), [(0, [A]), (1, [B])], 2, blank=BLANK, separator=SEP
    )
    frame = aligner._STRIDE / TARGET_SR
    assert spans[0] == pytest.approx((2 * frame, 5 * frame))
    assert spans[1] == pytest.approx((14 * frame, 17 * frame))


def test_word_spans_join_spoken_parts_and_skip_unspoken_words():
    emission = _emission([(A, 2), (BLANK, 1), (B, 2)], frames=8)
    # word 0 is said as two parts; word 1 had no letters the model knows
    spans = aligner.word_spans(
        emission, _frame_starts(8), [(0, [A]), (0, [B])], 2, blank=BLANK, separator=SEP
    )
    frame = aligner._STRIDE / TARGET_SR
    assert spans[0] == pytest.approx((0.0, 5 * frame))
    assert spans[1] is None


def test_word_spans_return_none_for_audio_without_frames():
    assert aligner.word_spans(
        np.empty((0, 0)), np.empty(0), [(0, [A])], 1, blank=BLANK, separator=SEP
    ) is None


@pytest.mark.parametrize(
    ("word", "spoken"),
    [
        ("Hello,", "HELLO,"),
        ("don’t", "DON'T"),
        ("café", "CAFE"),
        ("42", "FORTY TWO"),
        ("2026", "TWENTY TWENTY SIX"),
        ("1905", "NINETEEN OH FIVE"),
        ("2005", "TWO THOUSAND FIVE"),
        ("1,250", "ONE THOUSAND TWO HUNDRED FIFTY"),
        ("21st", "TWENTY FIRST"),
        ("11th", "ELEVENTH"),
        ("3.14", "THREE POINT ONE FOUR"),
        ("$5.50", "FIVE DOLLARS FIFTY"),
        ("$2.5", "TWO POINT FIVE DOLLARS"),
        ("$1", "ONE DOLLAR"),
        ("$1,000,000,000,000", "ONE TRILLION DOLLARS"),
        ("-5", "MINUS FIVE"),
        ("mid-2020s", "MID- TWENTY TWENTY S"),
        ("50%", "FIFTY PERCENT"),
        ("007", "ZERO ZERO SEVEN"),
        ("R&D", "R AND D"),
        ("20lb", "TWENTY POUNDS"),
        ("1lb", "ONE POUND"),
        ("5kg", "FIVE KILOGRAMS"),
        ("70mph", "SEVENTY MILES PER HOUR"),
        ("20°C", "TWENTY DEGREES CELSIUS"),
        ("$5m", "FIVE MILLION DOLLARS"),
        ("$5.5m", "FIVE POINT FIVE MILLION DOLLARS"),
        ("£5bn", "FIVE BILLION POUNDS"),
        ("$20k", "TWENTY THOUSAND DOLLARS"),
        ("5k", "FIVE K"),  # a race, not money
        ("5m", "FIVE M"),  # metres or million: left as written
        ("5kb", "FIVE KB"),  # not a known unit
    ],
)
def test_spoken_english_says_it_as_spoken(word, spoken):
    assert aligner._spoken_english(word).split() == spoken.split()


@pytest.mark.parametrize("written", [["£25"], ["25", "lb"], ["25lb"], ["25", "lbs."], ["25", "pounds"]])
def test_money_and_weight_pounds_align_as_the_same_speech(written):
    # ASR often writes spoken "twenty five pounds" (money) as "25 lb", or back:
    # the text stays as written, and either way the aligner looks for the same sound.
    assert " ".join(aligner._normalize_english(written)).split() == "TWENTY FIVE POUNDS".split()


def test_unit_words_only_follow_numbers():
    assert aligner._normalize_english(["1", "lb"]) == ["ONE", "POUND"]
    assert aligner._normalize_english(["the", "lb", "key"]) == ["THE", "LB", "KEY"]


def test_normalize_english_moves_the_currency_after_its_scale_word():
    # "$5 million" is said "five million dollars"
    assert aligner._normalize_english(["cost", "$5", "million.", "now"]) == [
        "COST", "FIVE", "MILLION. DOLLARS", "NOW",
    ]
    assert aligner._normalize_english(["$1", "billion"]) == ["ONE", "BILLION DOLLARS"]
    assert aligner._normalize_english(["$5", "each"]) == ["FIVE DOLLARS", "EACH"]


def test_normalize_english_never_raises_on_absurd_numbers():
    aligner._normalize_english(["$1." + "9" * 5000, "9" * 5000, "1," * 2000 + "000"])


@pytest.mark.parametrize(
    ("language", "code"),
    [("en", "en"), ("EN", "en"), ("en-US", "en"), ("en_GB", "en"), ("English", "en"), ("fr", "fr")],
)
def test_language_codes(language, code):
    assert aligner._language(language) == code


def test_missing_language_uses_the_configured_default(monkeypatch):
    monkeypatch.setattr(aligner, "ALIGN_DEFAULT_LANGUAGE", "en")
    assert all(aligner.supports(value) for value in (None, "", "  ", "auto"))
    monkeypatch.setattr(aligner, "ALIGN_DEFAULT_LANGUAGE", "")
    assert not any(aligner.supports(value) for value in (None, "", "auto"))
    assert aligner.supports("en-US") and not aligner.supports("fr")


VOCAB = {"<pad>": 0, "|": 4, "'": 5, **{chr(ord("A") + i): 6 + i for i in range(26)}}


def test_text_outside_the_model_alphabet_keeps_model_times(monkeypatch):
    monkeypatch.setitem(aligner._loaded, "en", (object(), VOCAB))
    ran = []

    def fake_emission(_session, _wav):
        # recorded rather than raised: align_words would swallow an exception
        ran.append(True)
        return np.empty((0, 0)), np.empty(0)

    monkeypatch.setattr(aligner, "_emission", fake_emission)
    # Russian, sent without `language`: numbers are spellable, the words are not
    assert aligner.align_words(np.zeros(16000), ["привет", "2026", "мир"]) is None
    assert not ran, "must not run the model for text it cannot spell"


def test_one_foreign_word_in_english_is_still_aligned(monkeypatch):
    monkeypatch.setitem(aligner._loaded, "en", (object(), VOCAB))
    seen = []

    def fake_emission(_session, _wav):
        seen.append(True)
        return np.empty((0, 0)), np.empty(0)

    monkeypatch.setattr(aligner, "_emission", fake_emission)
    aligner.align_words(np.zeros(16000), ["hello", "Москва", "friend"])
    assert seen


class _Download:
    def __init__(self, tmp_path, fail_times):
        self.calls = 0
        self.fail_times = fail_times
        (tmp_path / "vocab.json").write_text('{"<pad>": 0}')
        self.tmp_path = tmp_path

    def __call__(self, repo, filename, revision):
        self.calls += 1
        if self.calls <= self.fail_times:
            raise OSError("offline")
        return str(self.tmp_path / filename.split("/")[-1])


def _fake_hub(monkeypatch, download):
    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(hf_hub_download=download))
    monkeypatch.setattr(aligner, "_loaded", {})
    monkeypatch.setattr(aligner, "_failed_at", {})


def test_failed_load_is_retried_after_a_cooldown(monkeypatch, tmp_path):
    download = _Download(tmp_path, fail_times=1)
    _fake_hub(monkeypatch, download)
    monkeypatch.setattr(aligner.ort, "InferenceSession", lambda *a, **k: "session", raising=False)
    monkeypatch.setattr(aligner, "_build_sess_options", lambda *a, **k: None)
    now = [1000.0]
    monkeypatch.setattr(aligner.time, "monotonic", lambda: now[0])

    assert aligner._load("en") is None
    assert aligner.status() == {"en": "failed"}
    assert aligner._load("en") is None and download.calls == 1  # no retry storm
    now[0] += aligner._RETRY_SEC
    assert aligner._load("en") == ("session", {"<pad>": 0})
    assert aligner.status() == {"en": "loaded"}


def test_aligner_session_uses_its_own_threads_without_spinning(monkeypatch, tmp_path):
    _fake_hub(monkeypatch, _Download(tmp_path, fail_times=0))
    built = []
    monkeypatch.setattr(aligner, "_build_sess_options", lambda *a, **k: built.append((a, k)))
    monkeypatch.setattr(aligner.ort, "InferenceSession", lambda *a, **k: "session", raising=False)
    aligner._load("en")
    assert built == [((aligner.ALIGN_THREADS,), {"spinning": False})]


def test_parakeet_session_options_are_unchanged_by_default(monkeypatch):
    class Options:
        def __init__(self):
            self.entries = {}

        def add_session_config_entry(self, key, value):
            self.entries[key] = value

    fake_ort = SimpleNamespace(
        SessionOptions=Options,
        ExecutionMode=SimpleNamespace(ORT_SEQUENTIAL="sequential"),
        GraphOptimizationLevel=SimpleNamespace(ORT_ENABLE_ALL="all"),
    )
    monkeypatch.setattr(model, "ort", fake_ort)
    options = model._build_sess_options()
    assert options.intra_op_num_threads == ORT_INTRA_THREADS
    assert options.entries["session.intra_op.allow_spinning"] == "1"
    assert model._build_sess_options(3, spinning=False).entries[
        "session.intra_op.allow_spinning"
    ] == "0"


class _PositionSession:
    """Fake wav2vec2 that reports which absolute frame each output frame saw.

    The test audio holds one spike per 20 ms frame, at offset (frame % V) inside
    it; per-window normalization keeps the spike the loudest sample, so each
    output frame's one-hot class tells the test where in the audio it came from.
    """

    V = 64

    def __init__(self):
        self.lengths = []

    def get_inputs(self):
        return [SimpleNamespace(name="input_values")]

    def run(self, _outputs, feeds):
        x = feeds["input_values"][0]
        self.lengths.append(x.size)
        frames = (x.size - aligner._MIN_SAMPLES) // aligner._STRIDE + 1
        out = np.zeros((1, frames, self.V), np.float32)
        for k in range(frames):
            block = x[k * aligner._STRIDE : (k + 1) * aligner._STRIDE]
            out[0, k, int(block.argmax())] = 50.0
        return [out]


@pytest.mark.parametrize("seconds", [3.4, 61.5, 70.0, 95.3])
def test_windows_keep_each_frame_once_from_the_right_audio(seconds):
    samples = int(seconds * TARGET_SR)
    total = (samples - aligner._MIN_SAMPLES) // aligner._STRIDE + 1
    wav = np.zeros(samples, np.float32)
    for frame in range(total):
        wav[frame * aligner._STRIDE + frame % _PositionSession.V] = 1.0

    session = _PositionSession()
    emission, starts = aligner._emission(session, wav)
    assert np.allclose(starts, _frame_starts(total))  # one gap-free 20 ms grid
    assert (emission.argmax(axis=1) == np.arange(total) % _PositionSession.V).all()
    assert max(session.lengths) <= aligner._WINDOW + 2 * aligner._CONTEXT + TARGET_SR
