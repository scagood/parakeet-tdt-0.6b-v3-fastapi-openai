from __future__ import annotations

from types import SimpleNamespace

import pytest

from parakeet_service import routes, spoken
from parakeet_service.config import TARGET_SR


def _say(text):
    return " ".join(spoken.spoken_words(text.split()))


@pytest.mark.parametrize(
    ("written", "said"),
    [
        ("It cost $5 today.", "It cost five dollars today."),
        ("It cost $1.", "It cost one dollar."),
        ("We sold 25 lb of it.", "We sold twenty-five pounds of it."),
        ("We sold 25lb of it.", "We sold twenty-five pounds of it."),
        ("That is £25.", "That is twenty-five pounds."),
        ("They raised $5 million.", "They raised five million dollars."),
        ("They raised $5m.", "They raised five million dollars."),
        ("It was 20°C and 50% humid.", "It was twenty degrees Celsius and fifty percent humid."),
        ("The 21st of May 2026.", "The twenty-first of May twenty twenty-six."),
        ("About 1,250 people.", "About one thousand two hundred fifty people."),
        ("It fell to -5 overnight.", "It fell to minus five overnight."),
        ("Pi is 3.14 roughly.", "Pi is three point one four roughly."),
        ("(25 lb)", "(twenty-five pounds)"),
    ],
)
def test_whole_numbers_are_said_out(written, said):
    assert _say(written) == said


@pytest.mark.parametrize("written", ["an MP3 player", "COVID-19 cases", "it is 5m long", "a 5k run", "B2B sales"])
def test_names_and_ambiguous_numbers_are_left_as_written(written):
    assert _say(written) == written


def test_everywhere_mode_reads_digits_inside_names_for_the_aligner():
    assert spoken.spoken_word("MP3", everywhere=True) == "MP three"
    assert spoken.spoken_word("5m", everywhere=True) == "five m"


def test_capitalize_and_sentence_starts():
    assert spoken.capitalize("five dollars") == "Five dollars"
    assert spoken.capitalize("(five") == "(Five"
    assert spoken.starts_sentence(None) and spoken.starts_sentence("done.") and spoken.starts_sentence('said."')
    assert not spoken.starts_sentence("cost")


def _words(*items):
    return [{"word": w, "start": s, "end": e} for w, s, e in items]


def test_speak_numbers_splits_a_word_span_by_length():
    words = _words(("It", 0.0, 0.2), ("cost", 0.2, 0.5), ("$5", 0.5, 1.0))
    said = routes._speak_numbers(words)
    assert [w["word"] for w in said] == ["It", "cost", "five", "dollars"]
    five, dollars = said[2], said[3]
    assert five["start"] == 0.5 and dollars["end"] == 1.0
    assert five["end"] == dollars["start"] == pytest.approx(0.5 + 0.5 * 4 / 11)


def test_speak_numbers_capitalizes_at_sentence_start_only():
    said = routes._speak_numbers(_words(("$5", 0.0, 1.0), ("is", 1.0, 1.2), ("fine.", 1.2, 1.5), ("$5", 1.5, 2.0)))
    assert [w["word"] for w in said] == ["Five", "dollars", "is", "fine.", "Five", "dollars"]


def test_speak_numbers_returns_none_when_nothing_changes():
    assert routes._speak_numbers(_words(("hello", 0.0, 0.5), ("MP3", 0.5, 1.0))) is None


def _prepared():
    return routes._PreparedAudio(
        waveform=None, ranges=[(0, 3 * TARGET_SR)], pieces=["chunk"], duration=3.0
    )


def _result(text):
    tokens = [" " + t for t in text.split()]
    return SimpleNamespace(text=text, tokens=tokens, timestamps=[0.4 * i for i in range(len(tokens))])


def test_stitch_rewrites_text_segments_and_words_together():
    text, segments, words = routes._stitch(_prepared(), [_result("It cost $5 today.")], speak=True)
    assert text == segments[0]["segment"] == "It cost five dollars today."
    assert [w["word"] for w in words] == ["It", "cost", "five", "dollars", "today."]


def test_stitch_leaves_text_alone_when_off_or_nothing_to_say():
    assert routes._stitch(_prepared(), [_result("It cost $5 today.")])[0] == "It cost $5 today."
    assert routes._stitch(_prepared(), [_result("No numbers here.")], speak=True)[0] == "No numbers here."


def test_aligner_times_the_spoken_words():
    seen = []

    def fake_align(_wav, words):
        seen.append(words)
        return [(0.1 * i, 0.1 * i + 0.05) for i in range(len(words))]

    routes._stitch(_prepared(), [_result("It cost $5 today.")], fake_align, speak=True)
    assert seen == [["It", "cost", "five", "dollars", "today."]]
