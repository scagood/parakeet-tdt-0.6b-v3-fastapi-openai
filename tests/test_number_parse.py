from __future__ import annotations

import random
from decimal import Decimal

import pytest

from parakeet_service.number_parse import Value, means, values

# @darksheep/transcript-align, strategies/numbers/number.test.js, as (said,
# written). Not ported: its partial matches ("three hundred" against "3" "00")
# and comma pieces split across words, which are about aligning two word
# streams; a Parakeet number is one word ("1,100"), and here a text is read
# whole. Nor its roman numerals: "mm", "cm" and "ml" are well-formed ones.
TRANSCRIPT_ALIGN_VALID = [
    ("nine", "9"), ("fourth", "4"), ("four", "4th"), ("'fifties", "50s"), ("forty five", "45"),
    ("twenty first", "21st"), ("three hundred", "300"), ("eleven hundred", "1100"), ("eleven hundred", "1,100"),
    ("a hundred and forty five", "145"), ("a hundred and sixty", "160"), ("an hundred and twelve", "112"),
    ("three thousand", "3000"), ("three thousand", "3,000"), ("a thousand", "1,000"),
    ("ten thousandth", "10,000th"), ("one thousandth", "1,000th"),
    ("three million", "3000000"), ("a million", "1000000"), ("two billion", "2000000000"),
    ("one million three thousand and five", "1003005"), ("a thousand thousand", "1000000"),
    ("1000 thousand", "1000000"), ("three thousand million", "3000000000"),
    ("eleven hundred five", "1105"), ("eleven hundred and five", "1105"), ("eleven hundred and five", "1,105"),
    ("one thousand one hundred and five", "1105"), ("one thousand one hundred and five", "1,105"),
    ("nineteen eighty four", "1984"), ("one one 'o five", "1105"), ("one one oh five", "1,105"),
    ("double oh seven", "007"), ("treble three", "333"), ("double seven double three", "7733"),
    ("five triple six double four", "566644"), ("oh eight hundred", "0800"), ("zero one triple one", "01111"),
    ("2 7 double three double seven", "273377"),
    ("one fourteen", "1:14"), ("three sixteen", "3:16"), ("one ten", "1:10"),
    # decades
    ("seventies", "1970s"), ("nineteen seventies", "1970s"), ("eighties", "1980s"),
    ("nineteen eighties", "1980s"), ("twenties", "2020s"), ("twenty twenties", "2020s"),
    # money: one flat unit vocabulary, amounts compared exactly
    ("two pounds fifty", "£2.50"), ("two dollars fifty", "$2.50"), ("two dollars fifty.", "$2.50."),
    ("2.50 lb", "£2.50"), ("two pounds fifty", "2.5 lb"), ("two fifty", "2.50"), ("2 quid", "£2"),
    ("two pounds", "£2"), ("two fifty", "£2.50"), ("two twenty two", "$2.22"),
    ("two dollars and twenty two cents", "$2.22"), ("fifty pence", "50p"), ("fifty cents", "$0.50"),
    ("a pound fifty", "£1.50"), ("two hundred and fifty pounds", "£250"),
    ("eleven hundred and five pounds fifty", "£1,105.50"),
    # decimals
    ("ten point five", "10.5"), ("ten and a half", "10.5"), ("ten and a quarter", "10.25"),
    ("ten and three quarters", "10.75"), ("ten and a fourth", "10.25"), ("ten and three fourths", "10.75"),
    ("one point oh five", "1.05"), ("one hundred and fifty six point five", "156.5"),
    ("two hundred and twenty seven point six", "227.6"), ("two point five million", "2.5 million"),
    # its single-case tests
    ("two hundred and fifty", "£250"), ("twenty-first", "21st"), ("three hundred and the", "300"),
    ("twenty seven thirty-three seventy seven", "273377"),
    ("oh double seven six six double one triple three", "0776611333"),
    # identifier.test.js: the digits of a name, however they were recited
    ("L one seven L six three six three", "L17L6363"), ("L seventeen L sixty three sixty three", "L17L6363"),
    ("L 17 L 6363", "L17L6363"), ("bee twelve", "B12"), ("X nine", "X9"), ("sea three", "C3"),
    ("are two", "R2"), ("lima one seven lima six three six three", "L17L6363"), ("kilo nine", "K9"),
    ("A zero zero seven", "A007"), ("golf double seven", "G77"), ("A double oh seven", "A007"),
]

# Also not ported, from identifier.test.js: its rejects of names read as words
# ("covid nineteen") and of text with no digits; here letters aren't checked.
# "C three P O": a "p" straight after a number is pence here, as in number.js.
TRANSCRIPT_ALIGN_REJECTS = [
    ("three hundred", "400"), ("a hundred and forty five", "140"), ("three hundred", "300s"),
    ("three thousand", "4,000"), ("three thousand", "300"), ("three thousand", "3000s"),
    ("three million", "3000"), ("three million thousand", "3000000000"),
    ("fifty", "50s"), ("five forty", "45"), ("twenty twenty", "40"), ("forty five", "46"),
    ("shovel", "9"), ("nine", "shovel"),
    ("double seven", "7"), ("seven double", "77"), ("double twenty", "2020"), ("double shovel", "22"),
    ("seventy", "1970s"), ("nineteen seventy", "1970s"), ("seventies", "1975"),
    ("two hundred and fifty", "£2.50"), ("two hundred and fifty", "2.50"), ("two pounds forty", "£2.50"),
    ("2 50", "£2.50"), ("first fifty", "$1.50"),
    ("ten point six", "10.5"), ("ten point five", "105"), ("ten and a third", "10.33"),
    ("L one seven L six three six", "L17L6363"), ("bee thirteen", "B12"),
]

# What spoken.py and the phrase rework say, beyond transcript-align. Where
# transcript-align differs: "the hundred" meets "the 100" (it reads a bare scale
# as prose), and a decimal may have any number of places (it stops at two).
SPOKEN_VALID = [
    ("the hundred meters", "the 100 meters"), ("three point one four one", "3.141"),
    ("three point twelve", "3.12"), ("nought point five", "0.5"), ("oh point five", "0.5"),
    ("point five", "0.5"), ("a half", "0.5"), ("zero point zero five", "0.05"),
    ("ten o'clock", "10:00"), ("ten", "10:00"), ("ten hundred", "10:00"), ("seven oh five", "7:05"),
    ("seven fifteen a.m.", "715 a.m."), ("seven fifteen a.m.", "7.15am"), ("five thirty p.m.", "5.30pm"),
    ("ten thirty", "10.30pm"), ("six p.m.", "6pm"), ("six o'clock", "6pm"), ("zero thirty", "0:30"),
    ("two nil", "2-0"), ("two love", "2-0"), ("two oh", "2-0"), ("two to one", "2-1"),
    ("one hundred and eight to ninety-nine", "108-99"), ("five five five one two three four", "555-1234"),
    ("nineteen thirty-nine to nineteen forty-five", "1939-1945"),
    ("twenty twenty-four twenty-five", "2024-25"), ("twenty twenty-four to twenty twenty-five", "2024-25"),
    ("nineteen ninety-nine two thousand", "1999-00"), ("the hundredth", "the 100th"),
    ("thirteen hundred", "13:00"),
    ("minus five", "-5"), ("negative ten point five", "-10.5"), ("minus forty degrees Celsius", "-40°C"),
    ("the fifth of May", "5 May"), ("five May", "5 May"), ("July the fourth,", "July 4,"),
    ("twenty-first", "21st"), ("a hundredth", "100th"), ("one hundred and first", "101st"),
    ("nineteen hundreds", "1900s"), ("the hundreds", "100s"), ("two thousands", "2000s"),
    ("two thousand twenties", "2020s"), ("twenty tens", "2010s"), ("tens", "2010s"),
    ("two million dollars.", "$2 million."), ("two million bucks.", "$2 million."),
    ("one point two five million dollars", "$1.25 million"), ("one and a quarter million dollars", "$1.25 million"),
    ("five million dollars", "$5m"), ("two point five billion euros", "€2.5bn"), ("five thousand pounds", "£5k"),
    ("ninety miles per hour", "90 mph"), ("ninety M P H", "90 mph"), ("twenty-five pounds", "25 lb"),
    ("(one pound)", "(1 lb)"), ("fifty percent", "50%"), ("twelve and a half percent", "12.5%"),
    ("twelve euros", "12€"), ("thirty euros", "30 €"), ("cost twenty-five dollars", "cost$25"),
    ("eleven pounds forty p", "£11.40p"), ("eleven pounds and forty pence", "£11.40"), ("eleven forty", "£11.40"),
    ("twelve ninety-nine", "$12.99"), ("one oh five", "1.05"), ("a buck fifty", "$1.50"),
    ("one penny", "1p"), ("a penny", "1p"), ("a cent", "$0.01"), ("ninety-nine cents", "99c"),
    ("ninety-nine cents", "99¢"), ("a quid", "£1"),
    ("fifty quid", "£50"), ("five bucks", "$5"), ("fifteen hundred", "1500"),
    ("one thousand and five hundred", "1500"), ("nine eleven", "911"), ("nine one one", "911"),
    ("nineteen oh five", "1905"), ("twenty oh five", "2005"), ("twenty twenty-six", "2026"),
    ("forty-four twenty-one", "4421"), ("triple seven two", "7772"), ("one oh one", "101"),
    ("oh seven seven zero zero nine zero oh one two three", "07700900123"),
    ("MP three", "MP3"), ("MP3", "MP3"), ("twelve C", "12C"), ("five m", "5m"),
    ("one thousand and eighty", "1080p"), ("eight p.m.", "800 p.m."), ("eight o'clock pm", "8.00 pm"),
    ("six o'clock", "6 pm"), ("seven fifteen a.m.", "7.15 a.m."),
    ("twelve thousand five hundred million pounds", "£12,500 million"),
    ("one thousand five hundred million dollars", "$1,500 million"),
]

# Junk spoken.py generated before this parser, and plain wrong numbers.
SPOKEN_REJECTS = [
    ("twenty oh zero", "2000"), ("nineteen oh zero", "1900"), ("ten oh zero oh zero", "100000"),
    ("a point five zero million dollars", "$1.50m"), ("a and a half million dollars", "$1.50m"),
    ("zero and a half dollars", "$0.5"), ("one five", "$1.05"),
    ("one hundreds", "100s"), ("a hundreds", "100s"), ("zeros", "2000s"), ("seven zeros", "70s"),
    ("one thousand nine hundred seventies", "1970s"), ("seven point one five", "7.15am"),
    ("twelve cents", "12C"), ("seven", "007"), ("eight hundred", "0800"), ("fifty", "50p"),
    ("fifty", "£0.50"), ("five", "-5"), ("ten", "1000"), ("ten", "10:30"), ("twenty-one", "2-1"),
    ("one two", "2-1"), ("seventy", "70s"), ("two million", "$2"), ("one dollar twenty-five million", "$1.25m"),
    ("one million twenty-five thousand dollars", "$1.25m"),
    # transcript-align matches its "10" as a partial and leaves "point blank" as prose
    ("ten point blank", "10"),
    # each rule that keeps junk out: a minor unit, a bare scale, an ordinal
    # price, falling or equal scales, a year's zero half, a negative decade
    ("fifty p", "50"), ("hundred", "100"), ("two pounds fifth", "£2.05"),
    ("three thousand four thousand", "7000"), ("three thousand four thousand", "3004000"),
    ("nineteen zero", "1900"), ("minus seventies", "-70"),
    ("seven point one five a.m.", "7.15 a.m."),  # a time, however its am/pm is spaced
    # a collapsed price has a whole part; scales only multiply up
    ("zero fifty", "£0.50"), ("oh fifty", "0.50"), ("two point five million thousand", "$2.5bn"),
    # a season's end is in its own century; "hundredth" goes bare only after "the"
    ("nineteen ninety-nine nineteen hundred", "1999-00"), ("hundredth", "100th"),
]


@pytest.mark.parametrize(("said", "written"), TRANSCRIPT_ALIGN_VALID + SPOKEN_VALID)
def test_a_reading_means_the_written_number(said, written):
    assert means(said, written)


@pytest.mark.parametrize(("said", "written"), TRANSCRIPT_ALIGN_REJECTS + SPOKEN_REJECTS)
def test_a_reading_of_something_else_does_not(said, written):
    assert not means(said, written)


def test_values_are_what_the_text_says_left_to_right():
    assert values("£2.50") == {(Value(Decimal("2.5")),)}
    assert values("the fifth of May") == {(Value(5),)}
    assert values("2-1") == values("two to one") == {(Value(2), Value(1))}
    assert values("twenty–five") == {(Value(25),)}  # an en dash joins words too
    assert values("1970s") == {(Value(1970, decade=True),), (Value(70, decade=True),)}
    assert values("007") == {(Value(7, digits="007"),)}
    assert values("10:00") == {(Value(1000),), (Value(10),)}
    assert (Value(1984),) in values("nineteen eighty four")
    assert values("no numbers here") == {()}
    assert values("three quarters of it") == {(Value(Decimal("0.75")),)}  # never (3,)
    assert values("one hundreds") == frozenset()  # can't be read through: junk


def _say(n: int) -> str:
    """n in words the long way, independently of spoken.py: 1105 -> "one
    thousand one hundred and five"."""
    ones = ("zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen "
            "sixteen seventeen eighteen nineteen").split()
    tens = "_ _ twenty thirty forty fifty sixty seventy eighty ninety".split()
    if n < 20:
        return ones[n]
    if n < 100:
        return tens[n // 10] + (f"-{ones[n % 10]}" if n % 10 else "")
    for scale, name in ((10**9, "billion"), (10**6, "million"), (1000, "thousand"), (100, "hundred")):
        if n >= scale:
            head, rest = divmod(n, scale)
            tail = f" and {_say(rest)}" if 0 < rest < 100 else f" {_say(rest)}" if rest else ""
            return f"{_say(head)} {name}{tail}"
    raise AssertionError("unreachable")


def test_thousands_of_written_numbers_read_back():
    rng = random.Random(30)
    numbers = [*range(3000), *(rng.randrange(3000, 10**12) for _ in range(1000))]
    for n in numbers:
        assert (Value(n),) in values(_say(n)), n
        assert not means(_say(n + 1), str(n)), n
    for _ in range(500):
        pounds, pence = rng.randrange(1, 1000), rng.randrange(1, 100)
        assert means(f"{_say(pounds)} pounds and {_say(pence)} pence", f"£{pounds}.{pence:02d}")
        assert not means(f"{_say(pounds)} pounds {_say(pence)}", f"£{pounds}.{(pence + 1) % 100:02d}")
        places = str(rng.randrange(1, 10**6))
        said = " ".join(_say(int(d)) for d in places)
        assert means(f"{_say(pounds)} point {said}", f"{pounds}.{places}")
        hour, minute = rng.randrange(1, 13), rng.randrange(10, 60)
        assert means(f"{_say(hour)} {_say(minute)}", f"{hour}:{minute:02d}")
        code = "0" + str(rng.randrange(10**6))
        assert means(" ".join("oh" if d == "0" else _say(int(d)) for d in code), code)
        assert not means(_say(int(code)), code)
