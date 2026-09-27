"""English numbers, money and units as spoken words.

Parakeet writes what it hears in written form ("$5", "25 lb", "10:30"), and
different speech often comes out the same: "two pounds ten" and "two pounds
and ten pence" are both "£2.10", "ten thirty" is "1030". Two consumers want the
spoken form back:

- the word aligner needs letters it can find in the audio; it reads digits
  everywhere (spoken_words(..., everywhere=True)), even inside "MP3";
- PARAKEET_SPOKEN_NUMBERS returns a transcript that says what was said.

Both start from phrases(): each number with the words next to it that change
how it is said ("$2 million" is "two million dollars", "5 May" is "the fifth
of May"), and every plausible way it could have been said, so the audio can
pick (routes._speak_numbers). The first reading is used when there is no
audio. Every reading is read back through number_parse and kept only if it
still means the number written, so junk ("twenty oh zero") never reaches the
audio or the transcript.
"""
from __future__ import annotations

import itertools
import re
from dataclasses import dataclass
from typing import NamedTuple, Optional, Sequence

from . import number_parse

_ONES = (
    "zero one two three four five six seven eight nine ten eleven twelve thirteen "
    "fourteen fifteen sixteen seventeen eighteen nineteen"
).split()
_TENS = "_ _ twenty thirty forty fifty sixty seventy eighty ninety".split()
_SCALES = ((10**12, "trillion"), (10**9, "billion"), (10**6, "million"), (1000, "thousand"))
_SCALE_WORDS = {"hundred", "thousand", "million", "billion", "trillion"}
_ORDINALS = {
    "one": "first", "two": "second", "three": "third", "five": "fifth",
    "eight": "eighth", "nine": "ninth", "twelve": "twelfth",
}


class _Currency(NamedTuple):
    one: str
    many: str
    minor_one: str
    minor_many: str
    slang_one: str  # "a quid", "a buck"
    slang_many: str


_CURRENCIES = {
    "$": _Currency("dollar", "dollars", "cent", "cents", "buck", "bucks"),
    "£": _Currency("pound", "pounds", "penny", "pence", "quid", "quid"),
    "€": _Currency("euro", "euros", "cent", "cents", "euro", "euro"),  # "thirty euro"
}
# Unit abbreviations after a number, as (singular, plural). "£25", "25 lb" and
# "25lb" are all said "twenty-five pounds", and ASR often writes money as weight
# or the other way round.
_POUNDS = ("pound", "pounds")
_KM_PER_HOUR = ("kilometer per hour", "kilometers per hour")
_UNIT_WORDS = {
    "lb": _POUNDS, "lbs": _POUNDS,
    "oz": ("ounce", "ounces"),
    "kg": ("kilogram", "kilograms"), "kgs": ("kilogram", "kilograms"),
    "km": ("kilometer", "kilometers"),
    "cm": ("centimeter", "centimeters"),
    "mm": ("millimeter", "millimeters"),
    "ml": ("milliliter", "milliliters"),
    "ft": ("foot", "feet"),
    "mph": ("mile per hour", "miles per hour"),
    "kph": _KM_PER_HOUR, "km/h": _KM_PER_HOUR,
    "hr": ("hour", "hours"), "hrs": ("hour", "hours"),
    "min": ("minute", "minutes"), "mins": ("minute", "minutes"),
    "°c": ("degree Celsius", "degrees Celsius"),
    "°f": ("degree Fahrenheit", "degrees Fahrenheit"),
    "°": ("degree", "degrees"),
    "%": ("percent", "percent"),
}
# A number before a noun takes a singular unit: "a five dollar bill".
_SINGULAR = {
    plural_word: singular_word
    for singular, plural in [
        *_UNIT_WORDS.values(),
        *((c.one, c.many) for c in _CURRENCIES.values()),
        *((c.slang_one, c.slang_many) for c in _CURRENCIES.values()),
    ]
    for singular_word, plural_word in zip(singular.split(), plural.split())
    if singular_word != plural_word
}
_SINGULAR_UNIT = re.compile(rf"\b({'|'.join(map(re.escape, _SINGULAR))})\b")
# Units also said as their letters: "ninety M P H".
_UNIT_LETTERS = {"mph": "M P H", "kph": "K P H"}
# Scale abbreviations only count after a currency: "$5m" is five million
# dollars, but a bare "5m" could be metres, so it keeps its letter.
_SCALE_ABBREVIATIONS = {"k": "thousand", "m": "million", "bn": "billion"}
# Hundredths said as a fraction: "10.5" is "ten and a half".
_FRACTION_WORDS = {50: ["a half"], 25: ["a quarter", "a fourth"], 75: ["three quarters", "three fourths"]}
# Minor currency after a number: "50p", "99c", "99¢". Case matters: "12C" is a seat.
_MINOR_SUFFIXES = {"p": "£", "P": "£", "c": "$", "¢": "$"}
_MONTHS = {
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
}
_MAX_DIGITS = 15  # a longer run of digits is a code, read one by one
# ponytail: a number with more digits stays as written. Reading a recited code
# back through number_parse enumerates its parses, which doubles every two
# digits or so: ~10 ms per word at 20 digits, ~0.2 s at 30. Match the readings
# against the written digits instead (as identifier.js's readDigits does) to
# say longer ones.
_MAX_SAID_DIGITS = 20
_SUFFIXES = "|".join(
    re.escape(s)
    for s in sorted(
        {*_UNIT_WORDS, *_SCALE_ABBREVIATIONS, *_MINOR_SUFFIXES, *_CURRENCIES,
         "am", "pm", "a.m.", "p.m.", "s", "'s"},
        key=len,
        reverse=True,
    )
)
_SIGN = f"[{re.escape(''.join(_CURRENCIES))}]"
# A leading minus counts only at the start of a word: "mid-2020s" is not negative.
_NUMBER_PATTERN = (
    rf"(?:(?<!\w)(-))?({_SIGN})?(\d{{1,3}}(?:,\d{{3}})+|\d+)(?:\.(\d+))?(st|nd|rd|th)?"
    rf"(?:({_SUFFIXES})(?![a-z]))?"
)
_NUMBER = re.compile(_NUMBER_PATTERN, re.IGNORECASE)
_OPENING = "\"'([“‘"
_OPEN = r"([\"'(\[“‘]*)"
_CLOSE = r"([\"')\]”’.,!?;:]*)"
# A whole word that is just a number (with its currency, unit and punctuation),
# possibly glued to the word before by a currency sign: Parakeet writes "cost$25".
_WHOLE = re.compile(rf"{_OPEN}(?:([A-Za-z']+)(?={_SIGN}))?{_NUMBER_PATTERN}{_CLOSE}", re.IGNORECASE)
_TIME = re.compile(rf"{_OPEN}(\d{{1,2}}):(\d{{2}})(?:\s?(am|pm|a\.m\.|p\.m\.))?{_CLOSE}", re.IGNORECASE)
_TIME_SUFFIX = re.compile(r"[ap]\.?m\.?", re.IGNORECASE)
_PAIR = re.compile(rf"{_OPEN}(\d+)-(\d+){_CLOSE}")  # scores, ranges, phone numbers: "2-1", "555-1234"
_TRAILING_PUNCTUATION = re.compile(r"[\"')\]”’.,!?;:]+$")
_SENTENCE_END = re.compile(r"[.!?][\"')\]”’]*$")
# Full stops that end no sentence before a number.
_ABBREVIATIONS = {
    "no.", "nos.", "mr.", "mrs.", "ms.", "dr.", "st.", "vs.", "e.g.", "i.e.", "approx.", "fig.", "vol.", "p.",
}
# Words a round 100 or 1000 may follow bare: "the hundred meters", "her
# hundredth birthday". No article follows most of them ("our a hundredth"), but
# "her", "this" and "that" may also end what comes before one: "I told her a
# hundred times".
_DETERMINERS = {
    "the", "my", "your", "his", "her", "its", "our", "their", "this", "that", "these", "those",
    "every", "each", "first", "last", "next",
}
_ARTICLE_MAY_FOLLOW = {"her", "this", "that"}
ZERO_WORDS = ("zero", "oh", "nought")  # routes tries each zero of a chosen reading every way


# --------------------------------------------------------------------------- #
# Number words
# --------------------------------------------------------------------------- #
def _cardinal(n: int, *, british: bool = False) -> str:
    """942 -> "nine hundred forty-two" (british: "nine hundred and forty-two")."""
    if n < 20:
        return _ONES[n]
    if n < 100:
        return _TENS[n // 10] + ("" if n % 10 == 0 else "-" + _ONES[n % 10])
    if n < 1000:
        if n % 100 == 0:
            return f"{_ONES[n // 100]} hundred"
        return f"{_ONES[n // 100]} hundred {'and ' if british else ''}{_cardinal(n % 100, british=british)}"
    for scale, name in _SCALES:
        if n >= scale:
            head, rest = divmod(n, scale)
            if rest == 0:
                return f"{_cardinal(head, british=british)} {name}"
            joiner = " and " if british and rest < 100 else " "
            return f"{_cardinal(head, british=british)} {name}{joiner}{_cardinal(rest, british=british)}"
    raise AssertionError("unreachable")


def _in_pairs(digits: str) -> Optional[str]:
    """Said two digits at a time: "1905" -> "nineteen oh five", "911" -> "nine
    eleven", "1900" -> "nineteen hundred", "273377" -> "twenty-seven
    thirty-three seventy-seven"; None if a pair can't be said ("190005")."""
    first = len(digits) % 2 or 2
    pairs = [digits[:first]] + [digits[i : i + 2] for i in range(first, len(digits), 2)]
    said = []
    for index, pair in enumerate(pairs):
        if pair == "00" and not 0 < index == len(pairs) - 1:
            return None
        if pair == "00":
            said.append("hundred")
        else:
            said.append(f"oh {_ONES[int(pair)]}" if len(pair) == 2 and pair[0] == "0" else _cardinal(int(pair)))
    return " ".join(said)


def _ordinal(words: str) -> str:
    head, separator, last = max(
        (words.rpartition(" "), words.rpartition("-")), key=lambda parts: len(parts[0])
    )
    if last in _ORDINALS:
        last = _ORDINALS[last]
    elif last.endswith("y"):
        last = last[:-1] + "ieth"
    else:
        last += "th"
    return f"{head}{separator}{last}"


def _plural(words: str) -> str:
    """Decades: "nineteen ninety" -> "nineteen nineties", "eighty" -> "eighties"."""
    head, separator, last = max(
        (words.rpartition(" "), words.rpartition("-")), key=lambda parts: len(parts[0])
    )
    last = last[:-1] + "ies" if last.endswith("y") else last + "s"
    return f"{head}{separator}{last}"


def _digits(digits: str, *, zero: str = "zero") -> str:
    return " ".join(zero if d == "0" else _ONES[int(d)] for d in digits)


def _cents(cents: int) -> str:
    """Two decimal places said as one number, as prices are: 50 -> "fifty", 5 -> "oh five"."""
    return f"oh {_ONES[cents]}" if cents < 10 else _cardinal(cents)


# ponytail: every mix of grouped and ungrouped runs ("oh double seven six six
# double one") up to this many per zero word; past it only all-ungrouped and
# all-grouped. Scoring one reading costs well under a millisecond. If long
# codes need every mix, choose each run's grouping by ear as routes does each
# zero, instead of listing the product.
_MAX_RECITATIONS = 64


def _run_ways(name: str, size: int) -> list[str]:
    """A run of one digit said every way it splits into ones, doubles and
    triples: "1111" -> "one one one one", "one triple one", "double one double one"..."""
    if size == 0:
        return [""]
    if size > 6:  # nobody says ten zeros any other way than one by one or in triples
        return [" ".join([name] * size), " ".join([f"triple {name}"] * (size // 3) + [name] * (size % 3))]
    ways = []
    for part, said in ((1, [name]), (2, [f"double {name}"]), (3, [f"triple {name}", f"treble {name}"])):
        if part <= size:
            ways += [f"{head} {tail}".strip() for head in said for tail in _run_ways(name, size - part)]
    return ways


def _recitations(digits: str) -> list[str]:
    """A digit string read out as a code or phone number, every usual way:
    "zero"/"oh"/"nought", runs as "double"/"triple"/"treble" or one by one,
    a trailing "hundred" ("0800" -> "oh eight hundred"), and pairs ("273377" ->
    "twenty-seven thirty-three seventy-seven")."""
    runs = [(d, len(list(group))) for d, group in itertools.groupby(digits)]
    options = []
    for zero in ZERO_WORDS:
        name = lambda d: zero if d == "0" else _ONES[int(d)]  # noqa: E731
        ways = [_run_ways(name(d), size) for d, size in runs]
        combinations = 1
        for way in ways:
            combinations *= len(way)
        if combinations <= _MAX_RECITATIONS:
            options += [" ".join(choice) for choice in itertools.product(*ways)]
        else:
            options += [" ".join(way[0] for way in ways), " ".join(way[-1] for way in ways)]
        if len(digits) >= 3 and digits.endswith("00"):
            options.append(f"{' '.join(name(d) for d in digits[:-2])} hundred")
    if len(digits) >= 4 and len(digits) % 2 == 0:
        options.append(_in_pairs(digits))
    return _unique(options)


def _unique(items) -> list:
    return list(dict.fromkeys(item for item in items if item))


def _amounts(n: int) -> list[str]:
    """Ways to say n as a quantity, most likely first."""
    options = [_cardinal(n), _cardinal(n, british=True)]
    if 1000 <= n < 10**6 and n % 1000 >= 100:  # "one thousand and five hundred"
        options.append(f"{_cardinal(n // 1000)} thousand and {_cardinal(n % 1000, british=True)}")
    for said in list(options):
        if said.startswith("one "):  # "a hundred and fifty", "a thousand"
            options.append("a " + said[4:])
    if 1100 <= n <= 9999 and (n // 100) % 10:  # "fifteen hundred (and fifty)"
        rest = n % 100
        hundreds = f"{_cardinal(n // 100)} hundred"
        options += [hundreds] if rest == 0 else [f"{hundreds}{join}{_cardinal(rest)}" for join in (" ", " and ")]
    return _unique(options)


# --------------------------------------------------------------------------- #
# One written number -> its readings
# --------------------------------------------------------------------------- #
class _Numeral(NamedTuple):
    """One number as written, split up: "(cost$1,250.50." has opening "(",
    glued "cost", currency "$", whole "1250" (grouped), fraction "50" and
    closing "."."""

    opening: str
    glued: str  # the word Parakeet glued on before a currency: "cost$25"
    sign: str
    currency: str  # written before the digits or after them ("12€")
    whole: str
    grouped: bool  # "1,250": an amount, never a code or a year
    fraction: str  # the digits after the point
    ordinal: str  # "st", "nd", "rd", "th"
    suffix: str  # as written: "lb", "m", "p", "am", "s", ...
    closing: str


def _numeral(sign, currency, whole, fraction, ordinal, suffix, opening="", glued="", closing="") -> _Numeral:
    if suffix in _CURRENCIES:  # "12€": the currency is said after the number anyway
        currency, suffix = currency or suffix, ""
    return _Numeral(
        opening, glued, sign, currency, whole.replace(",", ""), "," in whole,
        fraction, ordinal.lower(), suffix, closing,
    )


def _parse(word: str) -> Optional[_Numeral]:
    """A word that is one number, split up; None if it is anything else."""
    if not (match := _WHOLE.fullmatch(word)):
        return None
    opening, glued, *number, closing = (group or "" for group in match.groups())
    return _numeral(*number, opening=opening, glued=glued, closing=closing)


def _write(n: _Numeral) -> str:
    """The word `n` was parsed from (the currency before the digits)."""
    whole = f"{int(n.whole):,}" if n.grouped else n.whole
    fraction = f".{n.fraction}" if n.fraction else ""
    return f"{n.opening}{n.glued}{n.sign}{n.currency}{whole}{fraction}{n.ordinal}{n.suffix}{n.closing}"


def _front(n: _Numeral, said: str) -> list[str]:
    """`said` after what is written before n's digits: its opening punctuation,
    a glued word ("cost five dollars") and its sign ("minus"/"negative")."""
    glued = f"{n.glued} " if n.glued else ""
    return [f"{n.opening}{glued}{sign}{said}" for sign in (("minus ", "negative ") if n.sign else ("",))]


def _integer(digits: str, *, grouped: bool = False) -> list[str]:
    """Ways to say a whole number, most likely first. Four digits in year range
    are a year first ("1999": "nineteen ninety-nine"), three or four may be
    said in pairs ("1030": "ten thirty", "911": "nine eleven") or read out, and
    a leading zero makes a code ("0800"), read digit by digit."""
    if len(digits) > _MAX_DIGITS or (digits[0] == "0" and len(digits) > 1 and not grouped):
        return _recitations(digits)
    n = int(digits)
    options = _amounts(n)
    if grouped:
        return options
    if len(digits) == 4 and (1100 <= n <= 1999 or 2010 <= n <= 2099):
        # ponytail: 4-digit numbers in year range default to the year reading
        # ("twenty twenty-six"); with spoken numbers on, the audio picks
        # otherwise. Word times (spoken numbers off) always align the year
        # reading, so a count said "one thousand five hundred" starts ~160 ms
        # late (measured). Telling a year from a count needs context ("in",
        # "people").
        options.insert(0, _in_pairs(digits))
    if len(digits) in (3, 4):
        if n % 100:
            options.append(_in_pairs(digits))  # "ten thirty", "nine eleven", "forty-four twenty-one"
        elif 1 <= n // 100 <= 12:
            options.append(f"{_cardinal(n // 100)} o'clock")
    # "64," may be a set said "six four", but nobody says 80 as "eight oh": in a
    # spoken list, "eighty" heard with a pause after it scored as "eight oh".
    if len(digits) > 2 or (len(digits) == 2 and digits[1] != "0"):
        options += _recitations(digits)
    return _unique(options)


def _point(whole: str, fraction: str) -> list[str]:
    """A number said with its decimal point: "3.14" -> "three point one four",
    "3.12" -> "three point twelve", "10.5" -> "ten and a half", "0.5" ->
    "nought point five", "point five", "a half"."""
    n = int(whole)
    heads = _amounts(n) if n else ["zero", "nought", "oh", ""]
    tails = [_digits(fraction), _digits(fraction, zero="oh")]
    if len(fraction) == 2 and fraction[0] != "0":
        tails.append(_cardinal(int(fraction)))  # version "3.12"
    options = [f"{head} point {tail}".strip() for head in heads for tail in tails]
    for part in _FRACTION_WORDS.get(int(fraction.ljust(2, "0")) if len(fraction) <= 2 else 0, []):
        options += [f"{head} and {part}" for head in heads] if n else [part]
    return _unique(options)


def _decimal(whole: str, fraction: str) -> list[str]:
    """A decimal on its own: _point(), and two places also as a price or a
    time is said ("5.50": "five fifty", "1.05": "one oh five", "5.00":
    "five"). A written last zero is no decimal's ("twelve point five" is
    "12.5"), so "12.50" is "twelve fifty" first."""
    options = _point(whole, fraction)
    if len(fraction) == 2 and int(whole):
        cents = int(fraction)
        prices = [f"{head} {_cents(cents)}" if cents else head for head in _amounts(int(whole))]
        options = prices + options if fraction[1] == "0" else options + prices
    return _unique(options)


def _amount_and_unit(n: _Numeral, *, scaled: bool) -> list[tuple[str, str]]:
    """n's number and its currency as said, the currency after any scale word:
    "$1.25" (million) -> ("one point two five", "dollars"), ("one and a
    quarter", "dollars"), ...; the currency may go unsaid ("£250" as "two
    hundred and fifty"), and so may a missing one."""
    if n.fraction:
        heads = _point(n.whole, n.fraction)
    else:
        heads = [*_amounts(int(n.whole)), *(["a"] if n.whole == "1" else [])]
    if not n.currency:
        return [(head, "") for head in heads]
    name = _CURRENCIES[n.currency]
    one = n.whole == "1" and not n.fraction and not scaled
    units = (name.one, name.slang_one) if one else (name.many, name.slang_many)
    pairs = [(head, unit) for unit in units for head in heads]
    pairs += [(head, "") for head in heads if scaled or head != "a"]  # "a million", never a bare "a"
    return _unique(pairs)


def _pounds_and_pence(name: _Currency, major: int, cents: int, *, minor: str = "") -> list[str]:
    """"£2.10" -> "two pounds ten", "two pounds and ten pence", "two pounds ten
    pence", "two pounds ten p", "two quid ten", "two ten"; "$0.50" -> "fifty
    cents". `minor`: only the ways that say it, as written ("£11.40p")."""
    said = _cardinal(cents)
    minors = [name.minor_one if cents == 1 else name.minor_many, *(["p"] if name.minor_many == "pence" else [])]
    if not major:
        return _unique([f"{said} {each}" for each in minors] + ([f"a {name.minor_one}"] if cents == 1 else []))
    heads = [*_amounts(major), *(["a"] if major == 1 else [])]
    unit, slang = (name.one, name.slang_one) if major == 1 else (name.many, name.slang_many)
    if not cents:  # "$5.00"
        return _unique(f"{head} {each}" for each in (unit, slang) for head in heads)
    if minor:
        return _unique(f"{head} {unit}{joiner}{said} {minor}" for joiner in (" ", " and ") for head in heads)
    options: list[str] = []
    for head in heads:
        options.append(f"{head} {unit} {said}")
        # "and forty pence", but "forty p" only without the "and" (unless the p is written, above)
        options.append(f"{head} {unit} and {said} {minors[0]}")
        options += [f"{head} {unit} {said} {each}" for each in minors]
    options += [f"{head} {slang} {said}" for head in heads]  # "a quid fifty", "a buck fifty"
    options += [f"{head} {_cents(cents)}" for head in _amounts(major)]  # "two ten", "one oh five"
    return _unique(options)


def _money(n: _Numeral, scale: str = "") -> list[str]:
    """Ways to say an amount of money, most likely first: "£2.10" -> "two
    pounds ten", ...; "$5" (million) -> "five million dollars", "five million
    bucks", "five million"."""
    if len(n.fraction) == 2 and not scale:
        return _pounds_and_pence(_CURRENCIES[n.currency], int(n.whole), int(n.fraction))
    pairs = _amount_and_unit(n, scaled=bool(scale))
    return _unique(" ".join(filter(None, (head, scale, unit))) for head, unit in pairs)


def _pence(n: _Numeral) -> list[str]:
    """"50p", "99c", "99¢" in minor units ("fifty pence"); "£11.40p" as pounds
    and pence with its "p" said."""
    if n.currency:
        if len(n.fraction) != 2:
            return []
        minor = "p" if n.suffix in "pP" else ""
        return _pounds_and_pence(_CURRENCIES[n.currency], int(n.whole), int(n.fraction), minor=minor)
    if n.fraction or n.ordinal or (int(n.whole) >= 100 and n.suffix != "¢"):
        return []  # "1080p": a resolution
    return _pounds_and_pence(_CURRENCIES[_MINOR_SUFFIXES[n.suffix]], 0, int(n.whole))


def _measure(n: _Numeral, key: str) -> list[tuple[str, str]]:
    """A number and the unit after it: "25 lb" -> ("twenty-five", "pounds");
    "90 mph" -> ("ninety", "miles per hour"), ("ninety", "M P H")."""
    heads = _point(n.whole, n.fraction) if n.fraction else _amounts(int(n.whole))
    singular, plural = _UNIT_WORDS[key]
    units = [singular if n.whole == "1" and not n.fraction else plural]
    units += [_UNIT_LETTERS[key]] if key in _UNIT_LETTERS else []
    return [(head, unit) for unit in units for head in heads]


def _clock(hour: int, minute: Optional[int], *, suffixed: bool = False) -> list[str]:
    """"7:15" -> "seven fifteen", "7:05" -> "seven oh five", "10:00" -> "ten
    o'clock", "ten", "ten hundred"; before "am"/"pm" the bare hour comes first;
    "13:00" -> "thirteen hundred", "thirteen", never "thirteen o'clock"."""
    said = _cardinal(hour)
    if minute is None:
        return [said]
    if minute == 0 and not 1 <= hour <= 12:
        return [f"{said} hundred", said]
    if minute == 0:
        return [said, f"{said} o'clock"] if suffixed else [f"{said} o'clock", said, f"{said} hundred"]
    return [f"{said} {'oh ' if minute < 10 else ''}{_cardinal(minute)}"]


def _clock_of(n: _Numeral) -> list[str]:
    """A number before "am"/"pm" as a time of day: "715", "7.15" -> "seven
    fifteen", "6" -> "six"; [] if it can't be one."""
    if n.currency or n.ordinal or n.sign or n.grouped or len(n.whole) > 4:
        return []
    if n.fraction:  # "7.15am"
        hour, minute = n.whole, n.fraction
    else:
        hour, minute = (n.whole, "") if len(n.whole) <= 2 else (n.whole[:-2], n.whole[-2:])
    if len(minute) not in (0, 2) or int(hour) > 24 or (minute and int(minute) > 59):
        return []
    return _clock(int(hour), int(minute) if minute else None, suffixed=True)


def _decade(digits: str) -> list[str]:
    """"1970s" -> "nineteen seventies", "seventies"; "2000s" -> "two
    thousands"; "100s" -> "hundreds" (after "the")."""
    n = int(digits)
    options = [_plural(said) for said in _integer(digits)[:2]]
    if n in (100, 1000):
        options.append(_plural(_cardinal(n).split()[-1]))
    if n >= 100 and n % 100 >= 10:
        options.append(_plural(_cardinal(n % 100)))
    return _unique(options)


def _ranges(first: str, second: str) -> list[str]:
    """Two numbers joined by a dash: a range ("10-15": "ten to fifteen" first),
    a score ("2-1": "two one", "two to one", "two nil"), years ("1939-1945",
    the seasons "2024-25" and "24-25": the years first) or a phone number
    ("555-1234", "0161-496": digit by digit first)."""
    if len(first) > _MAX_DIGITS or len(second) > _MAX_DIGITS or len(first + second) > _MAX_SAID_DIGITS:
        return []
    if _is_year(first) and (len(second) == 2 or _is_year(second)):
        starts = _integer(first)[:2]
        if len(second) == 4:
            ends, joins = _integer(second)[:2], (" to ", " ")
        else:
            end_year = _integer(number_parse.season_end(first, second))[0]  # "1999-00": two thousand
            ends, joins = [_cardinal(int(second)), end_year], (" ", " to ")
        return _unique(f"{start}{join}{end}" for join in joins for start in starts for end in ends)
    digits = [f"{_digits(first)} {_digits(second)}", f"{_digits(first, zero='oh')} {_digits(second, zero='oh')}"]
    season = len(first) == len(second) == 2 and int(second) == int(first) + 1  # "24-25": two years
    joins = (" to ", " ") if int(first) < int(second) and not season else (" ", " to ")
    amounts = []
    for said in _amounts(int(first)):
        amounts += [f"{said}{join}{other}" for join in joins for other in _amounts(int(second))]
        amounts += [f"{said} nil"] if int(second) == 0 else []
    phone = len(second) >= 4 or any(len(side) > 1 and side[0] == "0" for side in (first, second))
    return _unique(digits + amounts if phone else amounts + digits)


def _is_year(digits: str) -> bool:
    return len(digits) == 4 and 1000 <= int(digits) <= 2099


def _day(n: Optional[_Numeral]) -> Optional[int]:
    """The day of the month `n` could be, or None."""
    if n is None or n.currency or n.fraction or n.sign or n.suffix or n.grouped or len(n.whole) > 2:
        return None
    return int(n.whole) if 1 <= int(n.whole) <= 31 else None


def _readings(n: _Numeral) -> list[str]:
    """Ways to say n (its sign, glued word and punctuation aside), most likely
    first; [] if it could be something else ("5m": metres or million? "12C": a seat)."""
    suffix = n.suffix.lower()
    if len(n.whole) > _MAX_DIGITS:
        return [] if n.currency or n.fraction or n.ordinal or n.suffix else _recitations(n.whole)
    if n.suffix in _MINOR_SUFFIXES:
        return _pence(n)
    if _TIME_SUFFIX.fullmatch(n.suffix):  # "715am", "7.15am"
        return [f"{said} {n.suffix}" for said in _clock_of(n)]
    if n.currency:
        scale = _SCALE_ABBREVIATIONS.get(suffix, "")
        return [] if n.ordinal or (suffix and not scale) else _money(n, scale)
    if suffix in _UNIT_WORDS:
        return [] if n.ordinal else [f"{head} {unit}" for head, unit in _measure(n, suffix)]
    if n.suffix in ("s", "'s"):
        return [] if n.fraction or n.ordinal else _decade(n.whole)
    if n.suffix:
        return []  # "5m", "5k", "12C": a name, or more than one thing
    if n.ordinal:  # never "a hundredth": that is 1/100
        amounts = [said for said in _amounts(int(n.whole)) if not said.startswith("a ")]
        return [] if n.fraction else [_ordinal(said) for said in amounts]
    return _decimal(n.whole, n.fraction) if n.fraction else _integer(n.whole, grouped=n.grouped)


def _said_anyway(match: re.Match) -> str:
    """The number `match` found inside a word, said somehow: the aligner needs
    letters even for "5m" ("five m") or "MP3"."""
    n = _numeral(*(group or "" for group in match.groups()))
    if not (said := _readings(n)[:1]):
        whole = _recitations(n.whole)[0] if len(n.whole) > _MAX_DIGITS else _amounts(int(n.whole))[0]
        point = f" point {_digits(n.fraction)}" if n.fraction else ""
        said = [f"{whole}{point} {n.suffix}".strip()]
    return _front(n, said[0])[0]


# --------------------------------------------------------------------------- #
# Phrases: a number with the words that change how it is said
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Phrase:
    """Parakeet's words[start:end], which hold one number, and every way they
    could have been said, most likely first. Each reading gives the spoken text
    of each of the words, so the aligner can still time them one by one:
    ("two", "million dollars.") for ["$2", "million."]."""

    start: int
    end: int
    readings: tuple[tuple[str, ...], ...]

    @property
    def options(self) -> list[str]:
        """Each reading as one text: "two million dollars."."""
        return [" ".join(filter(None, parts)) for parts in self.readings]


def phrases(words: Sequence[str], *, most: Optional[int] = None) -> list[Phrase]:
    """The numbers in `words` (one chunk, as Parakeet wrote it), in order.

    A number takes in the word after it when that changes how it is said: a
    scale word ("$2 million." -> "two million dollars."), a unit ("90 mph"), a
    time of day ("715 a.m."), a month ("5 May" -> "the fifth of May"); a month
    takes in the day after it ("July 4," -> "July fourth,"). Words that are
    not numbers, and names such as "MP3", "COVID-19", "5m" or "12C", are in none.

    `most` keeps each phrase's first `most` readings only: reading one back
    through number_parse costs far more than making it, and spoken_words
    needs one, routes two to know whether the audio has a choice.
    """
    if not any(char.isdigit() for word in words for char in word):
        return []  # every phrase has digits (a month needs its day): prose costs one scan
    found: list[Phrase] = []
    at = 0
    while at < len(words):
        phrase = _phrase(words, at, most)
        found += [phrase] if phrase else []
        at = phrase.end if phrase else at + 1
    return found


def _phrase(words: Sequence[str], at: int, most: Optional[int] = None) -> Optional[Phrase]:
    """The phrase starting at words[at], or None."""
    before = words[at - 1].lstrip(_OPENING).lower() if at else ""
    article = before in ("a", "an")
    determiner = before if before in _DETERMINERS or article else ""
    if at + 1 < len(words) and (two := _two_words(words[at], words[at + 1])):
        # None if no reading means them: "£12,500 million" is never "twelve
        # thousand five hundred pounds" and a bare "million".
        readings = two
    else:
        readings = [(said,) for said in _one_word(words[at], bool(determiner))]
    return _meaning(words, at, _after_article(readings) if article else readings, determiner, most)


def _after_article(readings: list[tuple[str, ...]]) -> list[tuple[str, ...]]:
    """After "a"/"an" the article is already said and the number is a modifier:
    "a $100 million project" -> "a hundred million dollar project", "a $5
    bill" -> "a five dollar bill"."""
    own = [(parts[0][2:], *parts[1:]) for parts in readings if parts[0].startswith("a ")]
    return [
        tuple(_SINGULAR_UNIT.sub(lambda match: _SINGULAR[match[0]], part) for part in parts)
        for parts in own + readings
    ]


def _meaning(
    words: Sequence[str], at: int, readings: list[tuple[str, ...]], determiner: str, most: Optional[int]
) -> Optional[Phrase]:
    """A Phrase of the (first `most`) readings that still mean the words they
    replace, read back through number_parse; None if none do. After a
    determiner they are read as after "the", the one place number_parse takes
    a bare "hundred", and say no article of their own ("the the fifth of May",
    "our a hundredth") unless one may follow it ("told her a hundred times")."""
    if determiner and determiner not in _ARTICLE_MAY_FOLLOW:
        readings = [parts for parts in readings if parts[0].split()[:1] not in (["a"], ["the"])]
    readings = _unique(readings)
    if not readings:
        return None
    end = at + len(readings[0])
    context = "a " if determiner in ("a", "an") else "the " if determiner else ""
    written = number_parse.values(context + " ".join(words[at:end]))
    meant = (
        parts for parts in readings
        if not written.isdisjoint(number_parse.values(context + " ".join(filter(None, parts))))
    )
    kept = tuple(itertools.islice(meant, most))
    return Phrase(at, end, kept) if kept else None


def _one_word(word: str, bare: bool) -> list[str]:
    """Readings of a word that is a number on its own, punctuation kept.
    `bare`: after a determiner, where a round 100 or 1000 may go without its
    "one" ("the hundred meters", "her hundredth birthday")."""
    if match := _TIME.fullmatch(word):
        opening, hour, minute, suffix, closing = (group or "" for group in match.groups())
        tail = f" {suffix}" if suffix else ""
        said = _clock(int(hour), int(minute), suffixed=bool(suffix))
        return [f"{opening}{each}{tail}{closing}" for each in said]
    if match := _PAIR.fullmatch(word):
        opening, first, second, closing = match.groups()
        return [f"{opening}{said}{closing}" for said in _ranges(first, second)]
    n = _parse(word)
    if n is None or len(n.whole) + len(n.fraction) > _MAX_SAID_DIGITS:
        return []
    options = _readings(n)
    if bare and n.whole in ("100", "1000") and not (n.currency or n.fraction or n.suffix):
        scale = _cardinal(int(n.whole)).split()[-1]
        options.insert(1, _ordinal(scale) if n.ordinal else scale)
    return [f"{front}{n.closing}" for said in options for front in _front(n, said)]


def _two_words(word: str, after: str) -> list[tuple[str, str]]:
    """Readings of a number and the word after it that changes how it is said,
    or of a month and the day after it, as the spoken text of each word:
    ("$2", "million.") -> ("two", "million dollars."), ("5", "May") -> ("the
    fifth of", "May"), ("July", "4,") -> ("July", "fourth,"); [] if they are
    not one phrase."""
    if not after or after[0] in _OPENING or _TRAILING_PUNCTUATION.search(word):
        return []
    core = _TRAILING_PUNCTUATION.sub("", after)
    closing = after[len(core) :]
    if word.lstrip(_OPENING) in _MONTHS:  # "July 4," -> "July fourth,"; "(July 4)" too
        day = _day(n := _parse(after))
        return [] if day is None else [(word, f"{said}{closing}") for said in _month_day(day, bool(n.ordinal))]
    if (clock := _TIME.fullmatch(word)) and not clock[4] and _TIME_SUFFIX.fullmatch(core):  # "7:15 a.m."
        hour, minute = int(clock[2]), int(clock[3])
        return [(f"{clock[1]}{said}", after) for said in _clock(hour, minute, suffixed=True)]
    n = _parse(word)
    if n is None or n.suffix or len(n.whole) > _MAX_DIGITS or len(n.whole + n.fraction) > _MAX_SAID_DIGITS:
        return []
    key = core.lower()
    if key in _SCALE_WORDS and not n.ordinal:  # "$2 million." -> "two million dollars."
        pairs = [(head, f"{core} {unit}".strip()) for head, unit in _amount_and_unit(n, scaled=True)]
    elif core in _MONTHS and (day := _day(n)):  # "5 May" -> "the fifth of May"
        pairs = [(said, core) for said in _day_month(day, bool(n.ordinal))]
    elif n.currency or n.ordinal:
        return []
    elif key in _UNIT_WORDS:  # "25 lb", "90 mph"
        pairs = _measure(n, key)
    elif core in _CURRENCIES:  # "30 €": all of it said on the number's word
        money = _money(n._replace(currency=core))
        return [(f"{front}{closing}", "") for said in money for front in _front(n, said)]
    elif _TIME_SUFFIX.fullmatch(core):  # "715 a.m.", "5.30 pm."
        pairs = [(said, core) for said in _clock_of(n)]
    else:
        return []
    return [(front, f"{tail}{closing}") for head, tail in pairs for front in _front(n, head)]


def _day_month(day: int, ordinal: bool) -> list[str]:
    """A day before its month: "5 May" is "the fifth of May", "fifth of May" or "five May"."""
    said = _ordinal(_cardinal(day))
    return [f"the {said} of", f"{said} of", said if ordinal else _cardinal(day)]


def _month_day(day: int, ordinal: bool) -> list[str]:
    """A day after its month: "July 4" is "July fourth", "July the fourth" or "July four"."""
    said = _ordinal(_cardinal(day))
    return [said, f"the {said}", *([] if ordinal else [_cardinal(day)])]


# --------------------------------------------------------------------------- #
# Mishearings
# --------------------------------------------------------------------------- #
# ponytail: a fixed menu of likely mishearings, bounded so the audio scores at
# most _MAX_RIVALS * _RIVAL_READINGS more readings per number. Decoding the
# number from the audio would find any error, at far higher cost.
_MAX_RIVALS = 12
_RIVAL_READINGS = 4
_TEENS_AND_TENS = {f"1{d}": f"{d}0" for d in "3456789"} | {f"{d}0": f"1{d}" for d in "3456789"}


def _misheard(n: _Numeral) -> list[_Numeral]:
    """Numbers Parakeet may have written in place of the one said, most likely
    first: a currency missed or invented, the fraction's halves and quarters or
    its last digit off by one, teens and tens swapped (15/50), a zero dropped or
    added, a one-digit whole number wrong ("£1.10" for "two pounds ten").
    Longer numbers keep their digits: in spoken lists, "41" heard as "21" was
    wav2vec2's mistake, not Parakeet's."""
    rivals = []
    if n.currency:
        rivals.append(n._replace(currency=""))
    elif len(n.fraction) == 2 and not n.suffix:  # "12.50" said "twelve euros fifty"
        rivals += [n._replace(currency=currency) for currency in _CURRENCIES]
    if n.fraction:
        last = int(n.fraction[-1])
        family = ["25", "50", "75"] if len(n.fraction) == 2 else ["25", "5", "75"]
        near = [n.fraction[:-1] + str(d) for d in (last - 1, last + 1) if 0 <= d <= 9]
        rivals += [n._replace(fraction=fraction) for fraction in family + near]
    if n.whole in _TEENS_AND_TENS:
        rivals.append(n._replace(whole=_TEENS_AND_TENS[n.whole]))
    if n.fraction in _TEENS_AND_TENS:
        rivals.append(n._replace(fraction=_TEENS_AND_TENS[n.fraction]))
    if len(n.whole) <= 3 and not n.fraction:
        rivals.append(n._replace(whole=n.whole + "0"))
        rivals += [n._replace(whole=n.whole[:-1])] if n.whole.endswith("0") and len(n.whole) > 1 else []
    if len(n.whole) == 1:
        rivals += [n._replace(whole=digit) for digit in "123456789" if digit != n.whole]
    return [rival for rival in _unique(rivals) if rival != n][:_MAX_RIVALS]


def rivals(words: Sequence[str], phrase: Phrase) -> list[str]:
    """Readings of numbers Parakeet may have misheard as the one in `phrase`
    ("£1.10" for "two pounds ten", "€3" for "thirty euros"), most likely first,
    each read back as its own number and none of them one of the phrase's own
    readings. [] unless the phrase is an amount: not a code, a decade, a time
    ("6pm", "6 pm"), a date or position ("5 May", "May 5", "the 21st"), a
    range or a name."""
    n = _parse(words[phrase.start])
    after = _TRAILING_PUNCTUATION.sub("", words[phrase.end - 1]) if phrase.end - phrase.start > 1 else ""
    if n is None or n.ordinal or any(_TIME_SUFFIX.fullmatch(s) for s in (n.suffix, after)) or after in _MONTHS:
        return []  # a time, a date or a position: its hour or day is no amount
    if n.suffix in ("s", "'s") or len(n.whole) > _MAX_DIGITS or (n.whole[0] == "0" and len(n.whole) > 1):
        return []  # a decade or a code: "70s", "007"
    own = set(phrase.options)
    options = []
    for rival in _misheard(n):
        changed = [*words[: phrase.start], _write(rival), *words[phrase.start + 1 :]]
        if (found := _phrase(changed, phrase.start)) and found.end == phrase.end:
            options += [said for said in found.options[:_RIVAL_READINGS] if said not in own]
    return _unique(options)


# --------------------------------------------------------------------------- #
# Words
# --------------------------------------------------------------------------- #
def spoken_words(words: Sequence[str], *, everywhere: bool = False) -> list[str]:
    """Each of `words` as said by default (each phrase's first reading), one
    text per word, in speaking order; other words as written.

    `everywhere` also reads digits inside other words ("MP3", "mid-2020s") and
    ambiguous suffixes ("5m"), which the aligner wants and a transcript doesn't.
    """
    said = list(words)
    for phrase in phrases(words, most=1):
        said[phrase.start : phrase.end] = phrase.readings[0]
    if not everywhere:
        return said
    return [_NUMBER.sub(lambda match: f" {_said_anyway(match)} ", text).strip() for text in said]


def starts_sentence(previous_word: Optional[str]) -> bool:
    """Whether the word after `previous_word` begins a sentence (None: first
    word). An abbreviation's full stop ends none: "No. 5", "Mr. 5", "e.g. 5"."""
    if previous_word is None:
        return True
    abbreviation = previous_word.lower().lstrip(_OPENING) in _ABBREVIATIONS
    return bool(_SENTENCE_END.search(previous_word)) and not abbreviation


def capitalize(text: str) -> str:
    """Upper-case the first letter, past any leading punctuation ("(five" -> "(Five")."""
    for index, char in enumerate(text):
        if char.isalpha():
            return text[:index] + char.upper() + text[index + 1 :]
    return text
