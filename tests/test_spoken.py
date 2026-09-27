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


def _plain(text):
    return " ".join(text.lower().replace("-", " ").replace(",", "").split())


# Every way transcript-align's number strategy knows a numeral can be said
# (@darksheep/transcript-align, strategies/numbers/number.test.js), turned
# around: the spoken form must be among the readings of the written one, so
# the audio gets the chance to pick it.
@pytest.mark.parametrize(
    ("written", "said"),
    [
        ("9", "nine"), ("4th", "fourth"), ("50s", "fifties"), ("45", "forty five"),
        ("21st", "twenty first"), ("300", "three hundred"), ("1100", "eleven hundred"),
        ("145", "a hundred and forty five"), ("160", "a hundred and sixty"),
        ("3000", "three thousand"), ("1,000", "a thousand"), ("3000000", "three million"),
        ("1000000", "a million"), ("2000000000", "two billion"),
        ("1003005", "one million three thousand and five"),
        ("1105", "eleven hundred five"), ("1105", "eleven hundred and five"),
        ("1105", "one thousand one hundred and five"), ("1984", "nineteen eighty four"),
        ("1105", "one one oh five"),
        ("007", "double oh seven"), ("333", "treble three"), ("7733", "double seven double three"),
        ("566644", "five triple six double four"), ("0800", "oh eight hundred"),
        ("01111", "zero one triple one"), ("273377", "two seven double three double seven"),
        ("273377", "twenty seven thirty three seventy seven"),
        ("0776611333", "oh double seven six six double one triple three"),
        ("1:14", "one fourteen"), ("3:16", "three sixteen"), ("1:10", "one ten"),
        ("1970s", "seventies"), ("1970s", "nineteen seventies"), ("1980s", "eighties"),
        ("2020s", "twenties"), ("2020s", "twenty twenties"),
        ("£2.50", "two pounds fifty"), ("$2.50", "two dollars fifty"), ("2.50", "two fifty"),
        ("£2", "two pounds"), ("£2", "two quid"), ("£2.50", "two fifty"),
        ("$2.22", "two twenty two"), ("$2.22", "two dollars and twenty two cents"),
        ("50p", "fifty pence"), ("$0.50", "fifty cents"), ("£1.50", "a pound fifty"),
        ("£250", "two hundred and fifty pounds"), ("£250", "two hundred and fifty"),
        ("£1,105.50", "eleven hundred and five pounds fifty"),
        ("10.5", "ten point five"), ("10.5", "ten and a half"), ("10.25", "ten and a quarter"),
        ("10.75", "ten and three quarters"), ("10.25", "ten and a fourth"),
        ("10.75", "ten and three fourths"), ("1.05", "one point oh five"),
        ("156.5", "one hundred and fifty six point five"),
        ("227.6", "two hundred and twenty seven point six"),
    ],
)
def test_every_known_way_of_saying_it_is_a_reading(written, said):
    assert _plain(said) in [_plain(option) for option in spoken.readings(written)]


# What Parakeet actually wrote for these, in the TTS corpus.
@pytest.mark.parametrize(
    ("written", "said"),
    [
        ("cost$25", "cost twenty five dollars"), ("paid$3.50", "paid three dollars and fifty cents"),
        ("£11.40", "eleven pounds and forty pence"), ("£150", "a hundred and fifty pounds"),
        ("1030", "ten thirty"), ("715am", "seven fifteen am"), ("1145", "eleven forty five"),
        ("1500", "fifteen hundred"), ("1500", "one thousand five hundred"),
        ("999", "nine nine nine"), ("911", "nine one one"), ("911", "nine eleven"),
        ("4421", "four four two one"), ("101", "one oh one"), ("7772", "triple seven two"),
        ("747", "seven four seven"), ("550", "five fifty"), ("5.50", "five fifty"),
        ("2-1", "two one"), ("108-99", "one hundred and eight to ninety nine"),
        ("555-1234", "five five five one two three four"), ("3.12", "three point twelve"),
        ("0.05", "zero point zero five"), ("64,", "six four"),
        ("£11.40p", "eleven pounds forty p"), ("1500", "one thousand and five hundred"),
        ("£1.5", "one and a half pounds"), ("0.5", "nought point five"),
    ],
)
def test_what_parakeet_writes_can_be_heard_as_what_was_said(written, said):
    assert _plain(said) in [_plain(option) for option in spoken.readings(written)]


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


def _no_audio():
    return None


def test_speak_numbers_splits_a_word_span_by_length():
    words = _words(("It", 0.0, 0.2), ("cost", 0.2, 0.5), ("$5", 0.5, 1.0))
    said = routes._speak_numbers(words, _no_audio)
    assert [w["word"] for w in said] == ["It", "cost", "five", "dollars"]
    five, dollars = said[2], said[3]
    assert five["start"] == 0.5 and dollars["end"] == 1.0
    assert five["end"] == dollars["start"] == pytest.approx(0.5 + 0.5 * 4 / 11)


def test_speak_numbers_capitalizes_at_sentence_start_only():
    said = routes._speak_numbers(_words(("$5", 0.0, 1.0), ("is", 1.0, 1.2), ("fine.", 1.2, 1.5), ("$5", 1.5, 2.0)), _no_audio)
    assert [w["word"] for w in said] == ["Five", "dollars", "is", "fine.", "Five", "dollars"]


def test_speak_numbers_returns_none_when_nothing_changes():
    assert routes._speak_numbers(_words(("hello", 0.0, 0.5), ("MP3", 0.5, 1.0)), _no_audio) is None


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


class _Ear:
    """Stands in for aligner.ChunkAligner: word i sits at (i, i + 0.5) seconds,
    and best() answers with whichever option contains `heard`."""

    def __init__(self, heard=None):
        self.heard, self.timed, self.windows = heard, [], []

    def spans(self, words):
        self.timed.append(list(words))
        return [(float(i), i + 0.5) for i in range(len(words))]

    def best(self, options, start, end):
        self.windows.append((start, end))
        return next((i for i, option in enumerate(options) if self.heard and self.heard in option), 0)


def test_aligner_times_the_spoken_words(monkeypatch):
    ear = _Ear()
    monkeypatch.setattr(routes.aligner, "for_chunk", lambda _wav, _language: ear)
    routes._stitch(_prepared(), [_result("It cost $5 today.")], align=True, speak=True)
    assert ear.timed[-1] == ["It", "cost", "five", "dollars", "today."]


def test_the_audio_picks_the_reading_that_was_said(monkeypatch):
    ear = _Ear(heard="and ten pence")
    monkeypatch.setattr(routes.aligner, "for_chunk", lambda _wav, _language: ear)
    text = routes._stitch(_prepared(), [_result("That'll be £2.10 please.")], speak=True)[0]
    assert text == "That'll be two pounds and ten pence please."
    # heard between its neighbours' edges: "be" ends at 2.5, "please." starts at 3.0
    assert ear.windows == [(1.5, 3.0)]


def test_each_zero_is_heard_on_its_own(monkeypatch):
    class ZeroEar(_Ear):
        """Hears "oh seven seven zero zero nine zero oh one two three": each
        option is scored by how many of its words match, position by position."""

        said = "oh seven seven zero zero nine zero oh one two three".split()

        def best(self, options, start, end):
            scores = [sum(a == b for a, b in zip(option.split(), self.said)) for option in options]
            return scores.index(max(scores))

    ear = ZeroEar()
    monkeypatch.setattr(routes.aligner, "for_chunk", lambda _wav, _language: ear)
    text = routes._stitch(_prepared(), [_result("Call me on 07700900123.")], speak=True)[0]
    assert text == "Call me on oh seven seven zero zero nine zero oh one two three."


def test_a_lone_letter_is_aligned_as_its_name():
    from parakeet_service import aligner

    assert aligner._letters("ten p") == "TEN PEE"
    assert aligner._letters("nought point five") == "NAWT POINT FIVE"
    assert aligner._letters("I have a plan B.") == "I HAVE A PLAN BEE"


def test_without_audio_the_first_reading_is_used(monkeypatch):
    monkeypatch.setattr(routes.aligner, "for_chunk", lambda _wav, _language: None)
    assert routes._stitch(_prepared(), [_result("That'll be £2.10 please.")], speak=True)[0] == (
        "That'll be two pounds ten please."
    )


def test_unambiguous_numbers_never_load_the_aligner(monkeypatch):
    def refuse(_wav, _language):
        raise AssertionError("no ambiguity, no model")

    monkeypatch.setattr(routes.aligner, "for_chunk", refuse)
    assert routes._stitch(_prepared(), [_result("It was 50% on the 21st.")], speak=True)[0] == (
        "It was fifty percent on the twenty-first."
    )
