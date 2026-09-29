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


def _forced_align(emission, targets):
    """The Viterbi path's (start, end) frames per target, or None."""
    path = aligner._viterbi(emission, targets, blank=BLANK)
    return None if path is None else path[0]


def test_forced_align_recovers_token_spans():
    emission = _emission([(BLANK, 2), (A, 3), (B, 4), (C, 3)], frames=15)
    assert _forced_align(emission, [A, B, C]) == [(2, 5), (5, 9), (9, 12)]


def test_forced_align_keeps_repeated_tokens_apart():
    emission = _emission([(A, 3), (BLANK, 1), (A, 3)], frames=7)
    first, second = _forced_align(emission, [A, A])
    assert first[1] <= second[0]


def test_forced_align_handles_long_transcripts():
    # 200 tokens -> a 401-state lattice, past what int8 state arithmetic can hold.
    targets = [1 + (i % 3) for i in range(200)]
    emission = _emission([(token, 2) for token in targets], frames=410)
    spans = _forced_align(emission, targets)
    assert spans == [(2 * i, 2 * i + 2) for i in range(200)]


def test_forced_align_rejects_too_few_frames():
    # "aa" needs a blank between the two tokens: three frames minimum.
    assert _forced_align(_emission([(A, 2)], frames=2), [A, A]) is None


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
        ("mid-2020s", "MID TWENTY TWENTIES"),
        ("50%", "FIFTY PERCENT"),
        ("007", "ZERO ZERO SEVEN"),
        ("R&D", "AR AND DEE"),  # lone letters are said as their names
        ("20lb", "TWENTY POUNDS"),
        ("1lb", "ONE POUND"),
        ("5kg", "FIVE KILOGRAMS"),
        ("70mph", "SEVENTY MILES PER HOUR"),
        ("20°C", "TWENTY DEGREES CELSIUS"),
        ("$5m", "FIVE MILLION DOLLARS"),
        ("$5.5m", "FIVE POINT FIVE MILLION DOLLARS"),
        ("£5bn", "FIVE BILLION POUNDS"),
        ("$20k", "TWENTY THOUSAND DOLLARS"),
        ("5k", "FIVE KAY"),  # a race, not money
        ("5m", "FIVE EM"),  # metres or million: left as written
        ("5kb", "FIVE KB"),  # not a known unit
        ("12C", "TWELVE SEE"),  # a seat, not cents
        ("£11.40p", "ELEVEN POUNDS FORTY PEE"),  # the "p" was said
        ("7.15am", "SEVEN FIFTEEN AM"),  # a time, not a decimal
        ("12€", "TWELVE EUROS"),
        ("150¢", "ONE HUNDRED FIFTY CENTS"),
        ("007p", "SEVEN PENCE"),
        ("an", "AN"),
        ("MP3", "MP THREE"),  # digits inside a name are said too
        ("nought", "NAWT"),  # routes tries "nought" for every zero: its letters don't say it
        ("100-200", "ONE HUNDRED TO TWO HUNDRED"),  # a range, not a phone number
        ("108-99.", "ONE HUNDRED EIGHT NINETY NINE."),
        ("555-1234", "FIVE FIVE FIVE ONE TWO THREE FOUR"),
    ],
)
def test_normalize_english_says_it_as_spoken(word, spoken):
    [said] = aligner._normalize_english([word])
    assert said.split() == spoken.split()


def test_normalize_english_says_a_number_with_the_words_around_it():
    # #30: the same phrases as spoken numbers, each word keeping its own part
    assert aligner._normalize_english(["on", "5", "May."]) == ["ON", "THE FIFTH OF", "MAY."]
    assert aligner._normalize_english(["the", "5", "May"]) == ["THE", "FIFTH OF", "MAY"]  # one "the"
    assert aligner._normalize_english(["£12,500", "million."]) == [
        "TWELVE THOUSAND FIVE HUNDRED", "MILLION POUNDS.",
    ]
    assert aligner._normalize_english(["July", "4,"]) == ["JULY", "FOURTH,"]
    assert aligner._normalize_english(["90", "mph"]) == ["NINETY", "MILES PER HOUR"]
    assert aligner._normalize_english(["715", "a.m."]) == ["SEVEN FIFTEEN", "A.M."]
    assert aligner._normalize_english(["(1", "lb)"]) == ["(ONE", "POUND)"]
    assert aligner._normalize_english(["30", "€"]) == ["THIRTY EUROS", ""]


@pytest.mark.parametrize("written", [["£25"], ["25", "lb"], ["25lb"], ["25", "lbs."], ["25", "pounds"]])
def test_money_and_weight_pounds_align_as_the_same_speech(written):
    # ASR often writes spoken "twenty five pounds" (money) as "25 lb", or back:
    # the text stays as written, and either way the aligner looks for the same sound.
    assert " ".join(aligner._normalize_english(written)).rstrip(".").split() == "TWENTY FIVE POUNDS".split()


def test_unit_words_only_follow_numbers():
    assert aligner._normalize_english(["1", "lb"]) == ["ONE", "POUND"]
    assert aligner._normalize_english(["the", "lb", "key"]) == ["THE", "LB", "KEY"]


def test_normalize_english_moves_the_currency_after_its_scale_word():
    # "$5 million" is said "five million dollars"
    assert aligner._normalize_english(["cost", "$5", "million.", "now"]) == [
        "COST", "FIVE", "MILLION DOLLARS.", "NOW",
    ]
    assert aligner._normalize_english(["$1", "billion"]) == ["ONE", "BILLION DOLLARS"]
    assert aligner._normalize_english(["$5", "each"]) == ["FIVE DOLLARS", "EACH"]


def test_normalize_english_never_raises_on_absurd_numbers():
    # the digit cap keeps this fast: fail on it here rather than hang below
    assert aligner.spoken.phrases(["0" + "27" * 10]) == []
    aligner._normalize_english(["$1." + "9" * 5000, "9" * 5000, "1," * 2000 + "000"])


@pytest.mark.parametrize(
    ("word", "spelled"),
    [
        ("L’Été,", "l'été"),  # lower case, the apostrophe its plain form
        ("été", "été"),  # NFKC composes: the vocab holds "é", not e + accent
        ("peut-être", "peut-être"),  # hyphens and apostrophes are letters here
        ("«Москва»", "москва"),
        ("Straße.", "straße"),
        ("fünf/sechs", "fünf sechs"),  # other punctuation breaks the word
        ("25%", ""),  # numbers are not said
        ("COVID-19", "covid"),
        ("'Allo'", "allo"),  # quotes are not apostrophes
        ("-", ""),
    ],
)
def test_normalize_omnilingual_spells_as_its_training_text(word, spelled):
    assert aligner._normalize_omnilingual([word]) == [spelled]


def test_the_catalog_aligns_every_parakeet_v3_language():
    from parakeet_service.config import MODEL_CONFIGS

    assert set(MODEL_CONFIGS["parakeet-v3"]["languages"]) <= set(aligner.ALIGN_LANGUAGES)
    assert aligner.ALIGN_LANGUAGES["en"] == "wav2vec2-base-960h"


def test_every_text_the_catalog_may_name_is_spelled_here():
    from parakeet_service.config import ALIGNER_TEXTS

    assert set(aligner._TEXTS) == ALIGNER_TEXTS


def test_tokens_txt_vocab_keeps_the_space_token(tmp_path):
    path = tmp_path / "tokens.txt"
    path.write_text("<s> 0\n<pad> 1\n  2\né 3\n", encoding="utf-8")
    assert aligner._read_vocab(str(path)) == {"<s>": 0, "<pad>": 1, " ": 2, "é": 3}


@pytest.mark.parametrize(
    ("language", "code"),
    [("en", "en"), ("EN", "en"), ("en-US", "en"), ("en_GB", "en"), ("English", "en"), ("fr", "fr")],
)
def test_language_codes(language, code):
    assert aligner.language_code(language) == code


def test_missing_language_uses_the_configured_default(monkeypatch):
    monkeypatch.setattr(aligner, "ALIGN_DEFAULT_LANGUAGE", "en")
    assert all(aligner.supports(value) for value in (None, "", "  ", "auto"))
    monkeypatch.setattr(aligner, "ALIGN_DEFAULT_LANGUAGE", "")
    assert not any(aligner.supports(value) for value in (None, "", "auto"))
    assert aligner.supports("en-US") and aligner.supports("fr") and not aligner.supports("ja")


VOCAB = {"<pad>": 0, "|": 4, "'": 5, **{chr(ord("A") + i): 6 + i for i in range(26)}}
VOCAB_SIZE = max(VOCAB.values()) + 1
# The shape of the catalog's wav2vec2-base-960h, as ChunkAligner reads it.
ENGLISH = {"blank": "<pad>", "separator": "|", "text": "english"}


def test_text_outside_the_model_alphabet_keeps_model_times(monkeypatch):
    monkeypatch.setitem(aligner._loaded, "wav2vec2-base-960h", (object(), VOCAB))
    ran = []

    def fake_emission(_session, _wav):
        # recorded rather than raised: spans() would swallow an exception
        ran.append(True)
        return np.empty((0, 0)), np.empty(0)

    monkeypatch.setattr(aligner, "_emission", fake_emission)
    # Russian, sent without `language`: numbers are spellable, the words are not
    assert aligner.for_chunk(np.zeros(16000)).spans(["привет", "2026", "мир"]) is None
    assert not ran, "must not run the model for text it cannot spell"


def _chunk_hearing(runs, frames):
    """A ChunkAligner whose audio is `runs` of (letter, frames)."""
    ids = [(VOCAB[token] if token != BLANK else BLANK, count) for token, count in runs]
    chunk = aligner.ChunkAligner(None, None, VOCAB, ENGLISH)
    chunk._frames = (_emission(ids, frames, vocab_size=VOCAB_SIZE), _frame_starts(frames))
    return chunk


def test_scores_prefer_the_reading_the_audio_spells():
    chunk = _chunk_hearing([(BLANK, 2), ("T", 3), (BLANK, 1), ("E", 3), (BLANK, 1), ("N", 3)], frames=16)
    scores = chunk.scores(["ten", "tan", "t"], 0.0, 1.0)
    assert scores[0] > scores[1] and scores[0] > scores[2]  # "t" pays for the audio it leaves unexplained
    assert chunk.best(["t", "ten"], 0.0, 1.0) == 1


@pytest.mark.parametrize(("written", "heard"), [("Z", "ZED"), ("Z", "ZEE"), ("H", "HAYCH"), ("H", "AYCH")])
def test_a_lone_letter_is_timed_the_way_it_was_named(written, heard):
    # the transcript says "Z" either way; the timing listens for "zee" and "zed"
    runs = [(BLANK, 2)] + [run for letter in heard for run in ((letter, 3), (BLANK, 1))]
    chunk = _chunk_hearing(runs, frames=4 * len(heard) + 6)
    assert chunk._accented(aligner._normalize_english([written]), [(0.0, 1.0)]) == [heard]
    assert chunk.spans([written])[0] is not None


def test_scores_hear_only_their_window():
    # "ten", a pause, then "pence": heard up to the pause "ten" was said; the
    # next word's audio would make it "ten pence".
    ten, pence = [("T", 3), ("E", 3), ("N", 3)], [("P", 3), ("E", 3), ("N", 3), ("C", 3), ("E", 3)]
    chunk = _chunk_hearing([(BLANK, 1), *ten, (BLANK, 10), *pence], frames=36)
    pause = 12 * aligner._STRIDE / TARGET_SR
    assert chunk.best(["ten", "ten pence"], 0.0, pause) == 0
    assert chunk.best(["ten", "ten pence"], 0.0, 1.0) == 1
    # whichever comes first: a reading pays for all of its letters, not only
    # for the prefix the audio holds ("five million dollars" is listed first)
    assert chunk.best(["ten pence", "ten"], 0.0, pause) == 1


def test_scores_fail_soft(monkeypatch):
    chunk = aligner.ChunkAligner(None, None, VOCAB, ENGLISH)
    monkeypatch.setattr(chunk, "_emission", lambda: 1 / 0)
    assert chunk.scores(["one", "two"], 0.0, 1.0) == [float("-inf")] * 2
    assert chunk.best(["one", "two"], 0.0, 1.0) == 0


def test_viterbi_scores_the_path_it_returns():
    emission = _emission([(BLANK, 1), (A, 2), (B, 2)], frames=6)
    spans, score = aligner._viterbi(emission, [A, B], blank=BLANK)
    assert spans == [(1, 3), (3, 5)]
    path = [BLANK, A, A, B, B, BLANK]
    assert score == pytest.approx(sum(emission[t, token] for t, token in enumerate(path)))
    assert aligner._viterbi(emission, [A, C], blank=BLANK)[1] < score - 5  # C is never heard


def _ox_cat():
    """Log-probs of "ox cat" said with a faint "x": its frame's likeliest token
    is blank, the "x" 4 below it. Every other token dominates its frames."""
    tokens = [BLANK, "O", "O", BLANK, "|", "C", "C", "A", "A", "T", "T", BLANK]
    logits = np.zeros((len(tokens), VOCAB_SIZE))
    for t, token in enumerate(tokens):
        logits[t, VOCAB.get(token, BLANK)] = 8.0
    logits[3, VOCAB["X"]] = 4.0
    return logits - np.log(np.exp(logits).sum(axis=1, keepdims=True))


def test_star_penalty_decides_whether_unexplained_speech_is_cheap():
    emission = _ox_cat()
    ox, cat = [VOCAB[c] for c in "OX"], [VOCAB[c] for c in "CAT"]

    def score(spoken, penalty):
        return aligner._star_path(emission, spoken, blank=BLANK, separator=VOCAB["|"], penalty=penalty)[1]

    full, short = [(0, ox), (1, cat)], [(0, cat)]
    # A cheap star (the timing default) takes "o x" as noise for less than the
    # faint "x" costs: the shortest reading wins by leaving speech unexplained.
    assert score(short, aligner._STAR_PENALTY) > score(full, aligner._STAR_PENALTY)
    # The choice penalty makes each unexplained frame cost more than that.
    assert score(full, aligner._CHOICE_STAR_PENALTY) > score(short, aligner._CHOICE_STAR_PENALTY)


def test_best_hears_the_whole_reading_under_the_choice_penalty():
    chunk = aligner.ChunkAligner(None, None, VOCAB, ENGLISH)
    chunk._frames = (_ox_cat(), _frame_starts(12))
    assert chunk.best(["cat", "ox cat", "ox cap"], 0.0, 1.0) == 1


class _CountingSession:
    """Fake wav2vec2 that hears nothing (all blank) and counts its runs."""

    def __init__(self, fail=False):
        self.runs, self.fail = 0, fail

    def get_inputs(self):
        return [SimpleNamespace(name="input_values")]

    def run(self, _outputs, feeds):
        self.runs += 1
        if self.fail:
            raise RuntimeError("onnxruntime failed")
        frames = (feeds["input_values"].shape[1] - aligner._MIN_SAMPLES) // aligner._STRIDE + 1
        out = np.zeros((1, frames, VOCAB_SIZE), np.float32)
        out[..., BLANK] = 8.0
        return [out]


def _ask_everything(chunk):
    """What routes asks of one chunk: timing, then choosing, then zeros."""
    chunk.spans(["It", "cost", "$2.10."])
    chunk.scores(["two pounds ten.", "two ten."], 0.0, 3.0)
    chunk.best(["zero", "oh"], 0.0, 3.0)
    return chunk.spans(["It", "cost", "two", "pounds", "ten."])


def test_the_audio_is_heard_once_per_chunk():
    session = _CountingSession()
    chunk = aligner.ChunkAligner(np.zeros(3 * TARGET_SR, np.float32), session, VOCAB, ENGLISH)
    assert _ask_everything(chunk) is not None
    assert session.runs == 1


def test_a_failed_pass_is_not_run_again(caplog):
    session = _CountingSession(fail=True)
    chunk = aligner.ChunkAligner(np.zeros(3 * TARGET_SR, np.float32), session, VOCAB, ENGLISH)
    assert _ask_everything(chunk) is None
    assert chunk.scores(["one", "two"], 0.0, 3.0) == [float("-inf")] * 2
    assert chunk.best(["one", "two"], 0.0, 3.0) == 0
    assert session.runs == 1 and caplog.text.count("wav2vec2 pass failed") == 1


def test_spoken_numbers_share_the_aligners_other_alphabet_rule():
    assert aligner.other_alphabet(["привет", "2026", "мир"])
    assert aligner.other_alphabet(["Это", "стоило", "$5", "and"])
    assert not aligner.other_alphabet(["hello", "Москва", "friend"])
    assert not aligner.other_alphabet(["café", "crème", "brûlée"])  # accents fold into Latin
    assert not aligner.other_alphabet(["2026", "$5"])  # no letters at all


def test_one_foreign_word_in_english_is_still_aligned(monkeypatch):
    monkeypatch.setitem(aligner._loaded, "wav2vec2-base-960h", (object(), VOCAB))
    seen = []

    def fake_emission(_session, _wav):
        seen.append(True)
        return np.empty((0, 0)), np.empty(0)

    monkeypatch.setattr(aligner, "_emission", fake_emission)
    aligner.for_chunk(np.zeros(16000)).spans(["hello", "Москва", "friend"])
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

    assert aligner._load("wav2vec2-base-960h") is None
    assert aligner.status() == {"wav2vec2-base-960h": "failed", "omnilingual-ctc-300m": "not loaded"}
    assert aligner._load("wav2vec2-base-960h") is None and download.calls == 1  # no retry storm
    now[0] += aligner._RETRY_SEC
    assert aligner._load("wav2vec2-base-960h") == ("session", {"<pad>": 0})
    assert aligner.status()["wav2vec2-base-960h"] == "loaded"


def test_aligner_session_uses_its_own_threads_without_spinning(monkeypatch, tmp_path):
    _fake_hub(monkeypatch, _Download(tmp_path, fail_times=0))
    built = []
    monkeypatch.setattr(aligner, "_build_sess_options", lambda *a, **k: built.append((a, k)))
    monkeypatch.setattr(aligner.ort, "InferenceSession", lambda *a, **k: "session", raising=False)
    aligner._load("wav2vec2-base-960h")
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
