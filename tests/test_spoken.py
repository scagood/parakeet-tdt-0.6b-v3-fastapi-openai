from __future__ import annotations

import random
from types import SimpleNamespace

import pytest

from parakeet_service import number_parse, routes, spoken
from parakeet_service.config import TARGET_SR
from tests.test_number_parse import _say as _long_form


def _say(text):
    """`text` as the transcript says it with no audio: each phrase's first reading."""
    return " ".join(filter(None, spoken.spoken_words(text.split())))


def _options(written):
    """Every reading of `written`, which must be one phrase."""
    words = written.split()
    [phrase] = spoken.phrases(words)
    assert (phrase.start, phrase.end) == (0, len(words))
    return phrase.options


def _plain(text):
    return " ".join(text.lower().replace("-", " ").replace(",", "").split())


# The no-audio default of representative inputs, the README's examples first.
@pytest.mark.parametrize(
    ("written", "said"),
    [
        ("It cost $5 today.", "It cost five dollars today."),
        ("It cost$25 today.", "It cost twenty-five dollars today."),
        ("That is £25.", "That is twenty-five pounds."),
        ("We sold 25 lb of it.", "We sold twenty-five pounds of it."),
        ("They raised $5 million.", "They raised five million dollars."),
        ("They raised $5m.", "They raised five million dollars."),
        ("It was 20°C and 50% humid.", "It was twenty degrees Celsius and fifty percent humid."),
        ("On the 21st.", "On the twenty-first."),
        ("That'll be £2.10 please.", "That'll be two pounds ten please."),
        ("About 1500 people.", "About fifteen hundred people."),
        ("Call 911 now.", "Call nine hundred eleven now."),
        ("At 1030 sharp.", "At one thousand thirty sharp."),
        ("It cost $1.", "It cost one dollar."),
        ("We sold 25lb of it.", "We sold twenty-five pounds of it."),
        ("The 21st of May 2026.", "The twenty-first of May twenty twenty-six."),
        ("About 1,250 people.", "About one thousand two hundred fifty people."),
        ("It fell to -5 overnight.", "It fell to minus five overnight."),
        ("Pi is 3.14 roughly.", "Pi is three point one four roughly."),
        ("(25 lb)", "(twenty-five pounds)"),
        ("(1 lb)", "(one pound)"),
        ("The house sold for$1.25 million.", "The house sold for one point two five million dollars."),
        ("His fastball hit 90 mph.", "His fastball hit ninety miles per hour."),
        ("The train leaves at 715 a.m. daily.", "The train leaves at seven fifteen a.m. daily."),
        ("The shop closes at 5.30 pm.", "The shop closes at five thirty pm."),
        ("It opens at 7.15am.", "It opens at seven fifteen am."),
        ("The party is on 5 May.", "The party is on the fifth of May."),
        ("We watched on July 4,", "We watched on July fourth,"),
        ("She won the 100 meters.", "She won the one hundred meters."),
        ("Dinner was 30 € each.", "Dinner was thirty euros each."),
        ("Dinner was 12€ each.", "Dinner was twelve euros each."),
        ("From 1939-1945.", "From nineteen thirty-nine to nineteen forty-five."),
        ("The 2024-25 season.", "The twenty twenty-four twenty-five season."),
        ("It was 99c or 50p.", "It was ninety-nine cents or fifty pence."),
        ("It was £11.40p.", "It was eleven pounds forty p."),
        ("Call 07700900123.", "Call zero seven seven zero zero nine zero zero one two three."),
        ("It won 2-1.", "It won two one."),
        ("In the 1970s.", "In the nineteen seventies."),
        ("It is $1.05.", "It is one dollar five."),
        # a range is no phone number, and says "to"; a score says the amounts
        ("We expect 100-200 people.", "We expect one hundred to two hundred people."),
        ("It takes 10-15 minutes.", "It takes ten to fifteen minutes."),
        ("The 24-25 season.", "The twenty-four twenty-five season."),
        ("The Lakers won 108-99.", "The Lakers won one hundred eight ninety-nine."),
        ("Call 555-1234.", "Call five five five one two three four."),
        # one "the"
        ("On the 5 May deadline.", "On the fifth of May deadline."),
        ("The 3rd May.", "The third of May."),
        # a whole scaled amount, its currency after the scale word
        ("The deficit was £12,500 million.", "The deficit was twelve thousand five hundred million pounds."),
        ("It cost $1,500 million.", "It cost one thousand five hundred million dollars."),
        # punctuation ends a number: "May" and "Million" are other words
        ("Tom got 5, May got 3.", "Tom got five, May got three."),
        ("It was $2. Million people watched.", "It was two dollars. Million people watched."),
        # a time however its am/pm is spaced
        ("It starts at 800 p.m.", "It starts at eight p.m."),
        ("It starts at 8.00 pm.", "It starts at eight pm."),
        # two places with a last zero: a price, not a decimal said digit by digit
        ("It cost 12.50.", "It cost twelve fifty."),
        ("It was 5.00.", "It was five."),
        # no "thirteen o'clock"; a season past a century ends in the next one
        ("At 13:00 sharp.", "At thirteen hundred sharp."),
        ("The 1999-00 season.", "The nineteen ninety-nine two thousand season."),
        # a month in brackets still takes its day; money or a bracket keeps them apart
        ("We met (July 4) there.", "We met (July fourth) there."),
        ("It cost $5 May I add.", "It cost five dollars May I add."),
        ("On July (4) we met.", "On July (four) we met."),
        ("Our 100th customer.", "Our one hundredth customer."),
    ],
)
def test_the_first_reading_is_said_without_audio(written, said):
    assert _say(written) == said


@pytest.mark.parametrize(
    "written",
    [
        "an MP3 player", "COVID-19 cases", "it is 5m long", "a 5k run", "B2B sales", "Your seat is 12C.",
        "1080p video",
    ],
)
def test_names_and_ambiguous_numbers_are_left_as_written(written):
    assert spoken.phrases(written.split()) == []
    assert _say(written) == written


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
    assert _plain(said) in [_plain(option) for option in _options(written)]


# What Parakeet actually wrote for these, in the TTS benchmarks.
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
        ("$2 million.", "two million dollars."), ("$2 million.", "two million bucks."),
        ("£80 million.", "eighty million pounds."), ("$1.25 million.", "one point two five million dollars."),
        ("$1.25 million.", "one and a quarter million dollars."),
        ("$1.25 million.", "one point twenty-five million dollars."),
        ("$4.5 billion", "four and a half billion dollars"),
        ("5 May.", "the fifth of May."), ("5 May.", "fifth of May."), ("5 May.", "five May."),
        ("May 5.", "May fifth."), ("May 5.", "May the fifth."), ("July 4th.", "July the fourth."),
        ("21 June", "the twenty-first of June"), ("June 21,", "June twenty-first,"),
        ("715 a.m.", "seven fifteen a.m."), ("7.15 a.m.", "seven fifteen a.m."), ("5.30 pm.", "five thirty pm."),
        ("7:15 a.m.", "seven fifteen a.m."), ("530.", "five thirty."), ("715", "seven fifteen"),
        ("90mph.", "ninety M P H."), ("90 mph.", "ninety M P H."), ("90 mph", "ninety miles per hour"),
        ("30 €", "thirty euros"), ("€30", "thirty euro"), ("€12.50.", "twelve euros and fifty cents."),
        ("€12.50", "twelve euro fifty"), ("1939-1945", "nineteen thirty-nine to nineteen forty-five"),
        ("2024-25", "twenty twenty-four twenty-five"), ("2024-25", "twenty twenty-four to twenty-five"),
        ("24-25", "twenty-four twenty-five"), ("$1.05", "one oh five"), ("$1.50", "a buck fifty"),
        ("1900s", "nineteen hundreds"), ("2000s", "two thousands"), ("2010s", "tens"),
        ("700", "seven o'clock"), ("1000", "ten o'clock"), ("800 p.m.", "eight o'clock p.m."),
        ("8.00 pm", "eight o'clock pm"), ("100-200", "one hundred two hundred"), ("2-0", "two nil"),
    ],
)
def test_what_parakeet_writes_can_be_heard_as_what_was_said(written, said):
    assert _plain(said) in [_plain(option) for option in _options(written)]


def test_the_before_a_round_hundred_allows_the_bare_scale():
    [phrase] = spoken.phrases("She won the 100 meters".split())
    assert "hundred" in phrase.options and "a hundred" not in phrase.options
    assert "hundred" in spoken.phrases("(the 100 meters)".split())[0].options
    assert "hundreds" in spoken.phrases("the 100s".split())[0].options
    assert spoken.phrases(["100s"]) == []  # "one hundreds" is no reading


def _heard(text):
    """Every reading of the one number in `text`."""
    [phrase] = spoken.phrases(text.split())
    return phrase.options


def test_other_determiners_allow_the_bare_scale_too():
    # With only "one hundred" and "a hundred" to choose from, the audio of
    # "her hundred meters" and "our hundredth customer" picked the "a" ones.
    assert "hundred" in _heard("She ran her 100 meters.")
    assert _heard("Our 100th customer.") == ["one hundredth", "hundredth"]
    assert "thousandth" in _heard("Every 1000th visitor wins.")
    assert "a hundred" not in _heard("My 100 meters.")
    assert "a hundred" in _heard("I told her 100 times.")  # "her" may end what comes before
    # "a hundredth" is 1/100: never an ordinal, determiner or not
    assert not any(option.startswith("a ") for option in _options("100th") + _options("1000th"))


@pytest.mark.parametrize(
    ("written", "said"),
    [
        ("a $5 bill.", "a five dollar bill."),
        ("a $100 million project.", "a hundred million dollar project."),
        ("an 11 lb baby", "an eleven pound baby"),
        ("a 100 people", "a hundred people"),
        ("It cost $5 today.", "It cost five dollars today."),
    ],
)
def test_after_a_or_an_the_article_is_said_once_and_the_unit_is_singular(written, said):
    assert " ".join(spoken.spoken_words(written.split())) == said
    assert not any(option.startswith(("a ", "the ")) for option in _heard(written))


def test_a_spoken_p_follows_no_and_unless_it_is_written():
    # "eleven pounds and forty p" beat the "and forty pence" said by 0.3 on the benchmark
    assert "eleven pounds and forty p" not in _options("£11.40")
    assert "eleven pounds forty p" in _options("£11.40")
    assert "eleven pounds and forty p" in _options("£11.40p")


def test_a_scale_word_the_readings_cannot_say_leaves_the_number_as_written():
    # never "five million ... pounds million": no reading means both words
    assert _say("It was £5433052 million.") == "It was £5433052 million."


# Nobody says 80 as "eight oh"; "64," may be a set said "six four".
@pytest.mark.parametrize(("written", "reading"), [("80", "eight oh"), ("10", "one zero"), ("64", "six four")])
def test_two_digits_are_recited_only_without_a_zero(written, reading):
    assert (reading in _options(written)) == ("0" not in written)


# Junk the per-word readings used to offer; the parser round trip drops it.
@pytest.mark.parametrize(
    ("written", "junk"),
    [
        ("2000", "twenty oh zero"), ("1900", "nineteen oh zero"), ("100000", "ten oh zero oh zero"),
        ("$1.50m", "a point five zero million dollars"), ("$1.50m", "a and a half million dollars"),
        ("$0.5", "zero and a half dollars"), ("$1.05", "one five"), ("$150.99", "a hundred and fifty ninety-nine"),
        ("$1.25 million", "one dollar twenty-five million"), ("$1.25 million", "one twenty-five million"),
        ("7.15am", "seven point one five am"), ("10-0", "one zero zero"),
        ("1970s", "one thousand nine hundred seventies"),
        ("the 100 meters", "the a hundred meters"),
    ],
)
def test_readings_that_mean_another_number_are_dropped(written, junk):
    words = written.split()
    assert _plain(junk) not in [_plain(option) for phrase in spoken.phrases(words) for option in phrase.options]


def test_a_phrase_keeps_each_words_part_for_the_aligner():
    [phrase] = spoken.phrases(["$2", "million."])
    assert phrase.readings[0] == ("two", "million dollars.")
    assert spoken.spoken_words(["on", "5", "May."]) == ["on", "the fifth of", "May."]
    assert spoken.spoken_words(["July", "4,"]) == ["July", "fourth,"]


def _written_numbers():
    """Thousands of numbers as Parakeet writes them, each with a way it is said
    (from an independent speller) that must be among its readings."""
    rng = random.Random(30)
    for n in [*range(2200), *(rng.randrange(2200, 10**12) for _ in range(300))]:
        yield str(n), _long_form(n)
    for _ in range(200):
        major, minor = rng.randrange(1, 1000), rng.randrange(1, 100)
        pence = "penny" if minor == 1 else "pence"
        yield f"£{major}.{minor:02d}", f"{_long_form(major)} pounds and {_long_form(minor)} {pence}"
        yield f"${major}", f"{_long_form(major)} {'dollar' if major == 1 else 'dollars'}"
        places = str(rng.randrange(1, 10**4))
        yield f"{major}.{places}", f"{_long_form(major)} point {' '.join(_long_form(int(d)) for d in places)}"
        hour, minute = rng.randrange(1, 13), rng.randrange(10, 60)
        yield f"{hour}:{minute:02d}", f"{_long_form(hour)} {_long_form(minute)}"
        code = "0" + str(rng.randrange(10**6))
        yield code, " ".join("oh" if d == "0" else _long_form(int(d)) for d in code)
        first, second = rng.randrange(0, 10), rng.randrange(0, 10)
        yield f"{first}-{second}", f"{_long_form(first)} to {_long_form(second)}"
        start = rng.randrange(1900, 2000)
        yield f"{start}-{start + 1 + rng.randrange(20)}", None


def _written_phrases():
    """Numbers with the word after them that changes how they are said."""
    rng = random.Random(31)
    for _ in range(100):
        amount, grouped = rng.randrange(2, 1000), rng.randrange(1000, 100000)
        day, hour, minute = rng.randrange(1, 29), rng.randrange(1, 13), rng.randrange(10, 60)
        yield f"${amount} million.", f"{_long_form(amount)} million dollars."
        yield f"£{grouped:,} million", f"{_long_form(grouped)} million pounds"
        yield f"{amount} kg", f"{_long_form(amount)} kilograms"
        yield f"{day} May", f"{_long_form(day)} May"
        yield f"{hour}{minute:02d} p.m.", f"{_long_form(hour)} {_long_form(minute)} p.m."
        yield f"{hour}.{minute:02d} pm", f"{_long_form(hour)} {_long_form(minute)} pm"
        yield f"{hour}00 a.m.", f"{_long_form(hour)} a.m."


@pytest.mark.parametrize("written_forms", [_written_numbers, _written_phrases])
def test_thousands_of_written_numbers_have_readings_that_mean_them(written_forms):
    for written, said in written_forms():
        options = _options(written)
        meaning = number_parse.values(written)
        assert all(not number_parse.values(option).isdisjoint(meaning) for option in options), written
        if said is not None:
            assert _plain(said) in [_plain(option) for option in options], written


def test_everywhere_mode_reads_digits_inside_names_for_the_aligner():
    assert spoken.spoken_words(["MP3"], everywhere=True)[0] == "MP three"
    assert spoken.spoken_words(["5m"], everywhere=True)[0] == "five m"
    assert spoken.spoken_words(["12C"], everywhere=True)[0] == "twelve C"


@pytest.mark.parametrize(
    "word", ["150¢", "007p", "085¢", "000p", "0.00p", "£00p", "99.99am", "$5kg", "4S", "7.5pm", "1080p"]
)
def test_everywhere_mode_says_anything_with_digits(word):
    said = spoken.spoken_words([word], everywhere=True)[0]
    assert said and not any(c.isdigit() for c in said)


def test_everywhere_mode_never_raises():
    rng = random.Random(30)
    alphabet = "0123456789$£€.,-:%°Cpcm¢'sthndr ap"
    for _ in range(3000):
        word = "".join(rng.choice(alphabet) for _ in range(rng.randrange(1, 12)))
        said = spoken.spoken_words(word.split(), everywhere=True)
        assert not any(c.isdigit() for text in said for c in text), word


def test_the_default_reading_checks_only_what_it_uses(monkeypatch):
    # A code like this has ~190 readings; the aligner's default (every English
    # word request) needs the first that means it, not all of them checked.
    words = ["Order", "001100110011", "shipped."]
    [full] = spoken.phrases(words)
    checked = []
    values = number_parse.values
    monkeypatch.setattr(number_parse, "values", lambda text: checked.append(text) or values(text))
    assert spoken.spoken_words(words, everywhere=True)[1] == full.options[0]
    assert len(full.readings) > 100 and len(checked) == 2  # the written number and its first reading
    assert [phrase.readings for phrase in spoken.phrases(words, most=2)] == [full.readings[:2]]
    # no audio to choose (the aligner failed to load): two readings show there
    # was a choice, and the first is said
    checked.clear()
    monkeypatch.setattr(routes.aligner, "for_chunk", lambda _wav, _language: None)
    said = routes._stitch(_prepared(), [_result(" ".join(words))], speak=True)[0]
    assert said == f"Order {full.options[0]} shipped." and len(checked) == 3


def test_a_code_past_the_digit_cap_stays_as_written():
    # Reading a code back costs ~5x more every 4 digits: 21 would take seconds
    # a few caps up, so the cap is held here rather than by a hanging test.
    assert spoken.phrases(["0" + "27" * 10]) == []  # 21 digits
    assert spoken.phrases(["It", "was", "0" + "27" * 9 + "2."])  # 20
    assert spoken.phrases(["£123456789012345.123456"]) == []  # the fraction's digits count too


@pytest.mark.parametrize(
    "word", ["(£1.10).", '"£1.10"', "cost$1", "-15", "$1,250.50", "21st", "(-$5m),", "25lb", "£11.40p"]
)
def test_a_parsed_word_is_written_back_as_it_was(word):
    # a rival is written back this way and replaces the word: nothing around its number may go
    assert spoken._write(spoken._parse(word)) == word


def test_a_trailing_currency_is_written_back_in_front():
    assert spoken._write(spoken._parse("12€.")) == "€12."


def test_capitalize_and_sentence_starts():
    assert spoken.capitalize("five dollars") == "Five dollars"
    assert spoken.capitalize("(five") == "(Five"
    assert spoken.starts_sentence(None) and spoken.starts_sentence("done.") and spoken.starts_sentence('said."')
    assert not spoken.starts_sentence("cost")
    assert not any(spoken.starts_sentence(word) for word in ("No.", "Mr.", "e.g.", "(vs."))


def test_both_modules_read_the_same_written_notation():
    # number_parse reads what spoken writes, so the two tables must agree
    assert number_parse._CURRENCY_SCALES == spoken._SCALE_ABBREVIATIONS
    assert set(number_parse._MINOR_SUFFIXES) == set(spoken._MINOR_SUFFIXES)
    assert number_parse._OPENING == spoken._OPENING
    assert set(number_parse._CURRENCY_SIGNS) == set(spoken._CURRENCIES)


# --------------------------------------------------------------------------- #
# Mishearings
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    ("written", "said"),
    [
        ("£1.10", "two pounds ten"), ("£1 million", "two million pounds"), ("$1", "four dollars"),
        ("$1.5 million", "one and a quarter million dollars"), ("$1.4 million", "one and a quarter million dollars"),
        ("€3", "thirty euros"), ("£5.15", "five pounds fifty"), ("12.50", "twelve euros fifty"),
        # each family on its own: teens and tens, whole and fraction; a zero
        # dropped; the last fraction digit off by one; the currency dropped
        ("15", "fifty"), ("$15", "fifty dollars"), ("£5.13", "five pounds thirty"), ("£40", "four pounds"),
        ("$2.40", "two dollars forty-one"), ("£1.10", "one point one zero"),
        # two places: halves and quarters as hundredths
        ("£1.25", "one pound fifty"), ("$2.75", "two dollars fifty"),
    ],
)
def test_rivals_include_the_number_parakeet_misheard(written, said):
    words = written.split()
    [phrase] = spoken.phrases(words)
    assert _plain(said) in [_plain(option) for option in spoken.rivals(words, phrase)]


# Only a one-digit number may have its digit wrong: in spoken lists, "41"
# heard as "21" was the aligner's mistake, not Parakeet's.
@pytest.mark.parametrize(
    ("written", "misheard"), [("98", "twenty-eight"), ("41", "twenty-one"), ("£47", "twenty-seven")]
)
def test_longer_numbers_keep_their_digits(written, misheard):
    [phrase] = spoken.phrases([written])
    assert not any(option.startswith(misheard) for option in spoken.rivals([written], phrase))
    [long] = spoken.phrases(["1234"])
    assert spoken.rivals(["1234"], long) == []  # nor a zero more or less past three digits


def test_a_rival_replaces_the_same_words():
    # "50" is no day, but its rivals "5" and "15" are: "the fifth of May" would
    # replace "50" alone and say "May" twice.
    words = "Tom got 50 May got 3.".split()
    phrase = spoken.phrases(words)[0]
    assert spoken.rivals(words, phrase) and not any("May" in option for option in spoken.rivals(words, phrase))


def test_rivals_are_bounded_and_each_means_its_own_number():
    for written in ["£1.10", "$2", "12.50", "€3", "$1.25 million", "90 mph", "(£1.10).", "150", "50p", "-5"]:
        words = written.split()
        [phrase] = spoken.phrases(words)
        misheard = spoken._misheard(spoken._parse(words[0]))
        assert 0 < len(misheard) <= spoken._MAX_RIVALS, written
        options = spoken.rivals(words, phrase)
        assert len(options) <= spoken._MAX_RIVALS * spoken._RIVAL_READINGS, written
        assert not set(options) & set(phrase.options), written
        rival_words = [" ".join([spoken._write(n), *words[1:]]) for n in misheard]
        assert all(any(number_parse.means(option, rival) for rival in rival_words) for option in options), written


@pytest.mark.parametrize(
    "written",
    [
        "007", "1970s", "70s", "715am", "MP3 player", "July 4", "10:30", "2-1",
        "6 pm.", "715 a.m.", "7.15 a.m.", "5.30 pm.", "800 p.m.", "5 May", "21 June,",
        # an ordinal is a date or a position ("210st" was one rival)
        "21st", "the 5th of May", "the 1st round",
    ],
)
def test_codes_times_dates_positions_and_names_have_no_rivals(written):
    words = written.split()
    assert all(spoken.rivals(words, phrase) == [] for phrase in spoken.phrases(words))


# --------------------------------------------------------------------------- #
# Routes: choosing by ear
# --------------------------------------------------------------------------- #
def _words(*items):
    return [{"word": w, "start": s, "end": e} for w, s, e in items]


def _no_audio():
    return None


def test_speak_numbers_splits_a_phrase_span_by_length():
    words = _words(("It", 0.0, 0.2), ("cost", 0.2, 0.5), ("$5", 0.5, 0.8), ("million.", 0.8, 1.0))
    said = routes._speak_numbers(words, _no_audio)
    assert [w["word"] for w in said] == ["It", "cost", "five", "million", "dollars."]
    five, dollars = said[2], said[4]
    assert five["start"] == 0.5 and dollars["end"] == 1.0
    assert five["end"] == said[3]["start"] == pytest.approx(0.5 + 0.5 * 4 / 19)


def test_speak_numbers_capitalizes_at_sentence_start_only():
    words = _words(("$5", 0.0, 1.0), ("is", 1.0, 1.2), ("fine.", 1.2, 1.5), ("$5", 1.5, 2.0))
    said = routes._speak_numbers(words, _no_audio)
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


def test_stitch_rewrites_text_segments_and_words_together(monkeypatch):
    monkeypatch.setattr(routes.aligner, "for_chunk", lambda _wav, _language: None)
    text, segments, words = routes._stitch(_prepared(), [_result("It cost $5 today.")], speak=True)
    assert text == segments[0]["segment"] == "It cost five dollars today."
    assert [w["word"] for w in words] == ["It", "cost", "five", "dollars", "today."]


def test_stitch_leaves_text_alone_when_off_or_nothing_to_say():
    assert routes._stitch(_prepared(), [_result("It cost $5 today.")])[0] == "It cost $5 today."
    assert routes._stitch(_prepared(), [_result("No numbers here.")], speak=True)[0] == "No numbers here."


class _Ear:
    """Stands in for aligner.ChunkAligner: word i sits at (i, i + 0.5) seconds,
    and a reading scores 1 for each of the `heard` texts it contains."""

    def __init__(self, *heard):
        self.heard, self.timed, self.windows = heard, [], []

    def spans(self, words):
        self.timed.append(list(words))
        return [(float(i), i + 0.5) for i in range(len(words))]

    def scores(self, options, start, end):
        self.windows.append((start, end))
        return [float(sum(heard in option for heard in self.heard)) for option in options]

    def best(self, options, start, end):
        scores = self.scores(options, start, end)
        return scores.index(max(scores))


def test_aligner_times_the_spoken_words(monkeypatch):
    ear, made = _Ear(), []
    monkeypatch.setattr(routes.aligner, "for_chunk", lambda _wav, _language: made.append(ear) or ear)
    routes._stitch(_prepared(), [_result("It cost $5 today.")], align=True, speak=True)
    assert ear.timed[-1] == ["It", "cost", "five", "dollars", "today."]
    assert len(made) == 1  # one aligner, so one wav2vec2 pass, hears and times the chunk


@pytest.mark.parametrize("said", ["two pounds and ten pence", "two quid ten"])  # the 2nd and 5th reading
def test_the_audio_picks_the_reading_that_was_said(monkeypatch, said):
    ear = _Ear(said)
    monkeypatch.setattr(routes.aligner, "for_chunk", lambda _wav, _language: ear)
    text = routes._stitch(_prepared(), [_result("That'll be £2.10 please.")], speak=True)[0]
    assert text == f"That'll be {said} please."
    # heard between its neighbours' edges: "be" ends at 1.5, "please." starts at 3.0
    assert set(ear.windows) == {(1.5, 3.0)}


def test_a_phrase_is_heard_up_to_the_word_after_it(monkeypatch):
    ear = _Ear("million dollars")
    monkeypatch.setattr(routes.aligner, "for_chunk", lambda _wav, _language: ear)
    text = routes._stitch(_prepared(), [_result("They raised $2 million last year.")], speak=True)[0]
    assert text == "They raised two million dollars last year."
    # "raised" ends at 1.5; "$2 million" is words 2-3, so "last" starts at 4.0
    assert set(ear.windows) == {(1.5, 4.0)}


def test_each_number_in_a_chunk_is_heard_in_its_own_slot(monkeypatch):
    ear = _Ear("and ten pence", "and fifty pence")
    monkeypatch.setattr(routes.aligner, "for_chunk", lambda _wav, _language: ear)
    written = "It opens at 6pm and costs £2.10 then £3.50 today."
    assert routes._stitch(_prepared(), [_result(written)], speak=True)[0] == (
        "It opens at six pm and costs two pounds and ten pence then three pounds and fifty pence today."
    )
    # "6pm" (word 3) has nothing to decide; "£2.10" (6) sits between "costs"
    # and "then", "£3.50" (8) between "then" and "today."
    assert set(ear.windows) == {(5.5, 7.0), (7.5, 9.0)}


def test_a_round_number_after_a_determiner_may_be_heard_bare(monkeypatch):
    ear = _MarginEar(0.0)
    ear.heard = {"hundred": 5.0, "hundredth": 5.0}
    monkeypatch.setattr(routes.aligner, "for_chunk", lambda _wav, _language: ear)
    for written, said in [
        ("She ran her 100 meters.", "She ran her hundred meters."),
        ("Our 100th customer arrived.", "Our hundredth customer arrived."),
    ]:
        assert routes._stitch(_prepared(), [_result(written)], speak=True)[0] == said


def test_each_zero_is_heard_on_its_own(monkeypatch):
    class ZeroEar(_Ear):
        """Hears "oh seven seven zero zero nine zero oh one two oh.": each
        option is scored by how many of its words match, position by position."""

        said = "oh seven seven zero zero nine zero oh one two oh.".split()

        def scores(self, options, start, end):
            self.windows.append((start, end))
            return [float(sum(a == b for a, b in zip(option.split(), self.said))) for option in options]

    ear = ZeroEar()
    monkeypatch.setattr(routes.aligner, "for_chunk", lambda _wav, _language: ear)
    text = routes._stitch(_prepared(), [_result("Call me on 07700900120.")], speak=True)[0]
    assert text == "Call me on oh seven seven zero zero nine zero oh one two oh."  # "oh." too
    assert set(ear.windows) == {(2.5, float("inf"))}  # each zero heard in the number's own slot


class _DeafEar(_Ear):
    """Can't tell anything: no word times, every reading -inf (a failed pass)."""

    def spans(self, words):
        return None

    def scores(self, options, start, end):
        return [float("-inf")] * len(options)


def test_the_first_reading_is_kept_when_the_audio_cannot_tell(monkeypatch):
    monkeypatch.setattr(routes.aligner, "for_chunk", lambda _wav, _language: _DeafEar())
    # "oh" stays: its zero is not swapped for "zero" by a tie
    assert routes._stitch(_prepared(), [_result("Born in 1905 in Leeds.")], speak=True)[0] == (
        "Born in nineteen oh five in Leeds."
    )


def test_without_audio_the_first_reading_is_used(monkeypatch):
    monkeypatch.setattr(routes.aligner, "for_chunk", lambda _wav, _language: None)
    assert routes._stitch(_prepared(), [_result("That'll be £2.10 please.")], speak=True)[0] == (
        "That'll be two pounds ten please."
    )


def _refuse(_wav, _language):
    raise AssertionError("nothing to hear, no model")


@pytest.mark.parametrize(
    ("written", "said"),
    [
        ("No numbers in an MP3 here.", "No numbers in an MP3 here."),
        ("It opens at 6pm.", "It opens at six pm."),  # one reading, no rivals
        ("It opens at 6 pm.", "It opens at six pm."),  # however it is spaced
    ],
)
def test_nothing_to_hear_never_loads_the_model(monkeypatch, written, said):
    monkeypatch.setattr(routes.aligner, "for_chunk", _refuse)
    assert routes._stitch(_prepared(), [_result(written)], speak=True)[0] == said
    assert not routes._needs_aligner([_result(written)], speak=True)


def test_a_number_to_hear_needs_the_model(monkeypatch):
    monkeypatch.setattr(routes.aligner, "ALIGN_DEFAULT_LANGUAGE", "en")
    results = [_result("No numbers here."), _result("That'll be £2.10 please.")]
    assert routes._needs_aligner(results, speak=True)
    assert not routes._needs_aligner(results)  # spoken numbers off
    assert not routes._needs_aligner(results, speak=True, language="ja")  # no aligner for it
    assert routes._needs_aligner([_result("No numbers here.")], align=True)


@pytest.mark.parametrize(
    "written",
    ["Это стоило $5 вчера вечером.", "Αυτό κόστισε £2.10 χθες."],
)
def test_other_alphabets_are_left_as_written(monkeypatch, written):
    # Sent without `language`, so taken for English: the aligner's own rule
    # (aligner.other_alphabet) says the chunk isn't, so no number is said out.
    monkeypatch.setattr(routes.aligner, "for_chunk", _refuse)
    assert routes._stitch(_prepared(), [_result(written)], speak=True)[0] == written
    assert not routes._needs_aligner([_result(written)], speak=True)


def test_one_foreign_word_in_english_still_says_its_numbers(monkeypatch):
    monkeypatch.setattr(routes.aligner, "for_chunk", lambda _wav, _language: None)
    assert routes._stitch(_prepared(), [_result("It cost $5 in Москва.")], speak=True)[0] == (
        "It cost five dollars in Москва."
    )


class _BrokenEar(_Ear):
    def scores(self, options, start, end):
        raise RuntimeError("a bug while hearing")


def test_a_failure_keeps_parakeets_text(monkeypatch, caplog):
    ear = _BrokenEar()
    monkeypatch.setattr(routes.aligner, "for_chunk", lambda _wav, _language: ear)
    results = [_result("That'll be £2.10 please.")]
    text, segments, words = routes._stitch(_prepared(), results, align=True, speak=True)
    assert text == segments[0]["segment"] == "That'll be £2.10 please."
    assert "spoken numbers failed" in caplog.text
    # and the words are still timed, as written
    assert ear.timed[-1] == ["That'll", "be", "£2.10", "please."] and words[2]["start"] == 2.0


class _MarginEar(_Ear):
    """Scores the rival "two pounds ten." `lead`, Parakeet's own "one pound
    and ten pence." `own`, and every other reading `other`."""

    def __init__(self, lead, own=0.0, other=0.0):
        super().__init__()
        self.heard, self.other = {"two pounds ten.": lead, "one pound and ten pence.": own}, other

    def scores(self, options, start, end):
        return [self.heard.get(option, self.other) for option in options]


@pytest.mark.parametrize(
    ("lead", "own", "text"),
    [
        (routes._CORRECTION_MARGIN - 1, 0.0, "It was one pound ten."),
        (routes._CORRECTION_MARGIN, 0.0, "It was one pound ten."),  # a tie with the margin is no clear lead
        (routes._CORRECTION_MARGIN + 1, 0.0, "It was two pounds ten."),
        # the rival beats the default reading by the margin, but not the best one
        (routes._CORRECTION_MARGIN + 1, 10.0, "It was one pound and ten pence."),
        # the measured band (see _CORRECTION_MARGIN): wrong rivals led by up
        # to 11.7, the smallest real fix by 16.3
        (10.0, 0.0, "It was one pound ten."),
        (25.0, 0.0, "It was two pounds ten."),
    ],
)
def test_a_misheard_number_is_corrected_only_by_a_clear_margin(monkeypatch, lead, own, text):
    monkeypatch.setattr(routes.aligner, "for_chunk", lambda _wav, _language: _MarginEar(lead, own))
    assert routes._stitch(_prepared(), [_result("It was £1.10.")], speak=True)[0] == text


def test_a_number_none_of_whose_readings_fit_is_not_corrected(monkeypatch):
    # no reading of Parakeet's number fits the window: that is no evidence against it
    ear = _MarginEar(5.0, own=float("-inf"), other=float("-inf"))
    monkeypatch.setattr(routes.aligner, "for_chunk", lambda _wav, _language: ear)
    assert routes._stitch(_prepared(), [_result("It was £1.10.")], speak=True)[0] == "It was one pound ten."


def test_a_number_with_one_reading_can_still_be_corrected(monkeypatch):
    # "20%" is said one way, but may be a misheard "2%": the model must hear it
    monkeypatch.setattr(routes.aligner, "ALIGN_DEFAULT_LANGUAGE", "en")
    ear = _MarginEar(0.0)
    ear.heard = {"two percent": routes._CORRECTION_MARGIN + 1}
    monkeypatch.setattr(routes.aligner, "for_chunk", lambda _wav, _language: ear)
    assert routes._needs_aligner([_result("It was 20% off.")], speak=True)
    assert routes._stitch(_prepared(), [_result("It was 20% off.")], speak=True)[0] == "It was two percent off."


def test_numbers_are_never_corrected_with_spoken_numbers_off(monkeypatch):
    monkeypatch.setattr(routes.aligner, "for_chunk", lambda _wav, _language: _MarginEar(1000.0))
    assert routes._stitch(_prepared(), [_result("It was £1.10.")])[0] == "It was £1.10."
    assert routes._stitch(_prepared(), [_result("It was £1.10.")], align=True)[0] == "It was £1.10."
