"""Numbers read back into values, from what was said and from how it was written.

Ported from @darksheep/transcript-align's numbers strategy (src/strategies/numbers/
parse.js, number.js, identifier.js; Unlicense, by this project's author), which
pairs a book's written numbers with a transcript's spoken ones by reading both
sides into values. Here the same reading checks that a spoken form of a Parakeet
number still means that number: means("two pounds and ten pence", "£2.10").

values() reads any mix of words and numerals, so one reader serves both sides.
Words that are not numbers are skipped ("the fifth of May" is 5), but never a
word that would change the number next to it ("fifty pence" is 0.5, never 50),
and text that can't be read through has no values at all (junk such as "one
hundreds" or "twenty oh zero").

When two readings are the same number:
- Amounts are exact decimals, so money, units and decimal points don't change a
  value: "£2.50", "$2.50", "2.50 lb", "2.5", "two pounds fifty", "two fifty" and
  "two point five" are all 2.5, and "two hundred and fifty" (250) never meets
  them. Unit words are one flat vocabulary, as in transcript-align (pounds,
  dollars, lb, quid and bucks are interchangeable), and a unit may go unsaid
  ("£250" as "two hundred and fifty"). Minor units are hundredths: "50p",
  "fifty pence" and "fifty p" are 0.5, never 50.
- Decades never meet plain numbers: "1970s" is the decade 1970 ("nineteen
  seventies") or, said short, the decade 70 ("seventies"); "seventy" is neither.
- A leading zero makes digits a code: "007" is only met by a recitation that
  says the zeros ("double oh seven"), never by "seven". Without one, a
  recitation is its number: "one one oh five" is 1105.
- Ordinals are their cardinals: "21st" and "twenty-first" are 21.
- A time is its digits: "10:30" is 1030 ("ten thirty"), "o'clock" is the hour
  times 100, a time on the hour may leave its minutes unsaid ("10:00" is also
  10) and am/pm are ignored.
- Several numbers are a sequence, in order: "2-1" is (2, 1), said "two one",
  "two to one" or "two nil". A phone-like pair ("555-1234") is also its digits
  run together, and a season ("2024-25") also (2024, 2025).
- Letters in a name are not checked, only its digits: "MP3" and "MP three" are
  both (3,).
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from decimal import Decimal
from itertools import product
from typing import NamedTuple, Optional, Sequence


@dataclass(frozen=True)
class Value:
    """One number as read. `number` is exact, so 2.50 and 2.5 are equal and
    Value(1984) works as well as Value(Decimal(1984)); `decade` marks "the
    seventies"; `digits` keeps a code's leading zeros, which `number` loses."""

    number: Decimal
    decade: bool = False
    digits: str = ""


Parse = tuple[Value, ...]
_Read = Optional[tuple[Decimal | int, int]]  # a value and how many words it took


class _Word(NamedTuple):
    value: int
    kind: str = "other"  # a "tens" word takes a "unit" after it: "twenty five"
    ordinal: bool = False
    decade: bool = False


_UNITS = "one two three four five six seven eight nine".split()
_TEENS = "ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen".split()
_TENS = "twenty thirty forty fifty sixty seventy eighty ninety".split()
_WORDS: dict[str, _Word] = {
    **{name: _Word(n, "unit") for n, name in enumerate(_UNITS, 1)},
    **{name: _Word(n, "unit", ordinal=True) for n, name in enumerate(
        "first second third fourth fifth sixth seventh eighth ninth".split(), 1)},
    **{name: _Word(n) for n, name in enumerate(_TEENS, 10)},
    **{name + "th": _Word(n, ordinal=True) for n, name in enumerate(_TEENS, 10) if n != 12},
    "twelfth": _Word(12, ordinal=True),
    **{name: _Word(n, "tens") for n, name in zip(range(20, 100, 10), _TENS)},
    **{name[:-1] + "ieth": _Word(n, ordinal=True) for n, name in zip(range(20, 100, 10), _TENS)},
    **{name[:-1] + "ies": _Word(n, decade=True) for n, name in zip(range(20, 100, 10), _TENS)},
    # Not in transcript-align: "the (twenty) tens" are the 2010s, and zero said
    # as a number, in scores ("two nil", "two oh") and before a point ("nought
    # point five").
    "tens": _Word(10, decade=True),
    **dict.fromkeys(("zero", "oh", "nought", "naught", "nil", "love"), _Word(0)),
}
_SCALES = {
    name + suffix: 10**power
    for name, power in (("hundred", 2), ("thousand", 3), ("million", 6), ("billion", 9), ("trillion", 12))
    for suffix in ("", "th")
}
# ponytail: US short scale, capped at trillion (as transcript-align); add entries above for more
_MAJOR_SCALES = (10**12, 10**9, 10**6, 1000)
_PLURAL_SCALES = {"hundreds": 100, "thousands": 1000}  # "the nineteen hundreds", "the two thousands"
_ARTICLES = {"a", "an"}
_ZERO_DIGITS = {"o", "oh", "zero", "nought", "naught"}
_REPEATS = {"double": 2, "triple": 3, "treble": 3}
# Only fractions that land on whole hundredths: a third is no decimal a numeral spells.
_FRACTIONS = {"half": 50, "halves": 50, "quarter": 25, "quarters": 25, "fourth": 25, "fourths": 25}
# ponytail: transcript-align's seed vocabulary; other units ("kilograms") are just skipped
# words. Add a unit here when saying it should change the number next to it.
_MAJOR_UNITS = {
    "pound", "pounds", "dollar", "dollars", "buck", "bucks", "quid", "euro", "euros", "lb", "lbs", "kg",
}
_MINOR_UNITS = {"pence", "penny", "p", "cent", "cents"}
# Words that change the number they follow, so the number never ends before one.
_EXTENDS = {*_SCALES, *_PLURAL_SCALES, *_FRACTIONS, *_MINOR_UNITS, "point", "oclock"}
# Words that mean something to a number, so reading never skips one: "three
# quarters" can't be 3, "minus five" can't be five. "p" is also a letter ("M P
# H") so it only counts straight after a number, above; "a" is also a letter
# ("A007") and only counts where a parser reads it ("a hundred", "a half").
_NEVER_SKIPPED = {
    *_WORDS, *_SCALES, *_PLURAL_SCALES, *_FRACTIONS, *_REPEATS,
    *(_MINOR_UNITS - {"p"}), "point", "oclock", "minus", "negative",
}
_CODE = re.compile(r"0\d+")
_DECIMAL = re.compile(r"\d+\.\d+")
_NUMERAL_WORD = re.compile(r"(0|[1-9]\d*)(s)?")


def _at(words: Sequence[str], index: int) -> str:
    return words[index] if index < len(words) else ""


# --------------------------------------------------------------------------- #
# Spoken words -> values (transcript-align's parse.js; its JS names in quotes)
# --------------------------------------------------------------------------- #
def _token(word: str) -> Optional[_Word]:
    """"parseNumberToken": a number word, or digits ("90s" is a decade)."""
    if entry := _WORDS.get(word):
        return entry
    if match := _NUMERAL_WORD.fullmatch(word):
        return _Word(int(match[1]), decade=bool(match[2]))
    return None


def _small(words: Sequence[str], at: int, *, cardinal: bool = False) -> _Read:
    """"parseSmall": below a hundred, one word or tens then a unit.
    `cardinal`: spoken cardinals only ("the first fifty" is no price)."""
    word = _at(words, at)
    first = _token(word)
    if first is None or first.decade or (cardinal and (first.ordinal or not word.isalpha())):
        return None
    if first.kind == "tens":
        second = _WORDS.get(_at(words, at + 1))
        if second and second.kind == "unit" and not (cardinal and second.ordinal):
            return first.value + second.value, 2
    return first.value, 1


def _group(words: Sequence[str], at: int) -> _Read:
    """"parseGroup": below a thousand, "three hundred and twelve". An article
    only counts before a scale word, so the "a" of "a house" is no one."""
    article = _at(words, at) in _ARTICLES
    lead = (1, 1) if article else _small(words, at)
    if lead is None:
        return None
    value, used = lead
    after = _at(words, at + used)
    if _SCALES.get(after) != 100:
        return None if article and after not in _SCALES else lead
    used += 1
    linked = _at(words, at + used) == "and"
    tail = _small(words, at + used + linked)
    if tail is None:
        return value * 100, used
    return value * 100 + tail[0], used + linked + tail[1]


def _scaled(words: Sequence[str], at: int, smallest: int = 0) -> _Read:
    """"parseScaled": a whole number phrase, one scale at a time, each smaller
    than the last ("three thousand four thousand" is no one number). Rising
    scales multiply: "a thousand thousand" is a million, and so is a larger
    scale after the whole phrase: "twelve thousand five hundred million"."""
    lead = _group(words, at)
    if lead is None:
        return None
    value, used = lead
    scale = _SCALES.get(_at(words, at + used), 0)
    if scale not in _MAJOR_SCALES[smallest:]:
        return lead
    used += 1
    factor = scale
    following = _SCALES.get(_at(words, at + used), 0)
    while following >= factor and scale * following in _MAJOR_SCALES:
        scale, factor, used = scale * following, following, used + 1
        following = _SCALES.get(_at(words, at + used), 0)
    linked = _at(words, at + used) == "and"
    value *= scale
    if tail := _scaled(words, at + used + linked, _MAJOR_SCALES.index(scale) + 1):
        value, used = value + tail[0], used + linked + tail[1]
    larger = _SCALES.get(_at(words, at + used), 0)
    if not smallest and larger > scale and larger in _MAJOR_SCALES:
        return value * larger, used + 1
    return value, used


def _fraction(words: Sequence[str], at: int) -> _Read:
    """"a half", "three quarters": a count (1-9) of a fraction word."""
    word = _at(words, at)
    entry = _WORDS.get(word)
    count = 1 if word in _ARTICLES else entry.value if entry and entry.kind == "unit" and not entry.ordinal else 0
    part = _FRACTIONS.get(_at(words, at + 1))
    return (Decimal(count * part) / 100, 2) if count and part else None


def _decimal(words: Sequence[str], at: int) -> _Read:
    """"parseDecimalPhrase": "ten point five", "ten and a half". Extended with
    a bare "point five" / "a half", zero heads ("nought point five") and any
    number of places ("three point one four one", "three point twelve")."""
    if bare := _fraction(words, at):
        return bare
    whole = (0, 0) if _at(words, at) == "point" else _scaled(words, at)
    if whole is None:
        return None
    value, used = whole
    if _at(words, at + used) == "point":
        run = _digit_run(words, at + used + 1, least=1, hundred=False)
        return None if run is None else (Decimal(f"{value}.{run[0]}"), used + 1 + run[1])
    # "and a half" only after a whole number: "zero and a half" is not said.
    if value and _at(words, at + used) == "and" and (part := _fraction(words, at + used + 1)):
        return value + part[0], used + 1 + part[1]
    return None


def _year(words: Sequence[str], at: int) -> _Read:
    """"parseYearPhrase": two halves, "nineteen eighty four" (the lead >= 10).
    A zero second half is no year ("twenty zero"); "twenty oh five" is recited."""
    lead = _small(words, at)
    if lead is None or lead[0] < 10:
        return None
    tail = _small(words, at + lead[1])
    if tail is None or not tail[0]:
        return None
    return lead[0] * 100 + tail[0], lead[1] + tail[1]


def _decade(words: Sequence[str], at: int) -> _Read:
    """"parseDecadePhrase": "nineteen seventies" is the 1970s. Extended with a
    thousands lead ("two thousand twenties") and plural scales: "the hundreds",
    "the nineteen hundreds", "the two thousands" (never "one hundreds")."""
    word = _at(words, at)
    if word in _PLURAL_SCALES:
        return (_PLURAL_SCALES[word], 1) if at and words[at - 1] == "the" else None
    lead = _small(words, at)
    if lead is None:
        return None
    value, used = lead
    after = _at(words, at + used)
    if after in _PLURAL_SCALES:
        return (value * _PLURAL_SCALES[after], used + 1) if value >= 2 else None
    if value >= 10:
        value *= 100
    elif (thousands := _scaled(words, at)) and thousands[0] and not thousands[0] % 1000:
        value, used = thousands
    else:
        return None
    tail = _token(_at(words, at + used))
    return (value + tail.value, used + 1) if tail and tail.decade else None


def _digit(word: str) -> Optional[str]:
    """"digitFor": one digit; "oh" and "nought" are zero."""
    if word in _ZERO_DIGITS:
        return "0"
    if len(word) == 1 and word.isdigit():
        return word
    entry = _WORDS.get(word)
    return str(entry.value) if entry and entry.kind == "unit" and not entry.ordinal else None


def _pair(word: str, after: str) -> Optional[tuple[str, int]]:
    """"pairFor": a tens or teen word as the two digits it says: "twenty
    seven" is 27, "seventy" 70, "fourteen" 14."""
    entry = _WORDS.get(word)
    if entry is None or entry.ordinal or entry.decade or not 10 <= entry.value <= 90:
        return None
    if entry.kind == "tens":
        unit = _WORDS.get(after)
        if unit and unit.kind == "unit" and not unit.ordinal:
            return str(entry.value + unit.value), 2
        return str(entry.value), 1
    return (str(entry.value), 1) if entry.value < 20 else None


def _digit_run(words: Sequence[str], at: int, *, least: int, hundred: bool) -> Optional[tuple[str, int]]:
    """"parseDigitPhrase": digits recited, as codes and phone numbers are: "four
    oh two", "double five six", pairs ("twenty seven thirty-three") and a
    trailing "hundred" ("oh eight hundred"). Returns the digits as said, leading
    zeros kept. Stricter than transcript-align: once a pair is said the rest
    goes in pairs too ("nineteen oh five", never "nineteen oh zero")."""
    digits, used, paired = "", 0, False
    while at + used < len(words):
        word, after = words[at + used], _at(words, at + used + 1)
        if hundred and word == "hundred" and digits:
            digits, used = digits + "00", used + 1
            break
        if pair := _pair(word, after):
            digits, used, paired = digits + pair[0], used + pair[1], True
            continue
        repeat = _REPEATS.get(word, 1)
        digit = _digit(after if repeat > 1 else word)
        if digit is None:
            break
        if paired and repeat == 1:
            second = _digit(after)
            if digit != "0" or second in (None, "0"):
                break
            digits, used = digits + "0" + second, used + 2
            continue
        digits, used = digits + digit * repeat, used + (1 if repeat == 1 else 2)
    return (digits, used) if len(digits) >= least else None


def _money(words: Sequence[str], at: int) -> _Read:
    """"parseMoneyPhrase": "two pounds fifty", "a dollar and twenty-two cents",
    "fifty pence", and the collapsed "two fifty". The lead may be any number
    (transcript-align: below a thousand)."""
    article = _at(words, at) in _ARTICLES and _at(words, at + 1) in (_MAJOR_UNITS | {"penny", "cent"})
    lead = (1, 1) if article else _scaled(words, at)
    if lead is None:
        return None
    value, used = lead
    unit = _at(words, at + used)
    if unit in _MAJOR_UNITS:
        used += 1
        linked = _at(words, at + used) == "and"
        tail = _small(words, at + used + linked, cardinal=True)
        if tail is None:
            return value, used
        used += linked + tail[1]
        return value + Decimal(tail[0]) / 100, used + (_at(words, at + used) in _MINOR_UNITS)
    if unit in _MINOR_UNITS:
        return Decimal(value) / 100, used + 1
    return _collapsed(words, at)


def _collapsed(words: Sequence[str], at: int) -> _Read:
    """A price with its units unsaid: "two fifty" is 2.50, "one oh five" 1.05.
    transcript-align takes a lead of one to nine; prices said this way run
    higher ("twelve ninety-nine")."""
    lead = _small(words, at, cardinal=True)
    if lead is None or not lead[0]:
        return None
    rest = at + lead[1]
    cents = _digit(_at(words, rest + 1))
    if _at(words, rest) in _ZERO_DIGITS and cents not in (None, "0"):
        return lead[0] + Decimal(cents) / 100, lead[1] + 2
    tail = _small(words, rest, cardinal=True)
    if tail is None or tail[0] < 10:
        return None
    return lead[0] + Decimal(tail[0]) / 100, lead[1] + tail[1]


def _the_scale(words: Sequence[str], at: int) -> _Read:
    """"the hundred meters", "the hundredth": a bare scale word after "the".
    transcript-align refuses it (in a book "the hundred" is prose); here it
    only ever meets a written "the 100" (spoken.py reads "her 100th" as "the
    100th" to check it)."""
    word = _at(words, at)
    bare = word in ("hundred", "thousand", "hundredth", "thousandth")
    return (_SCALES[word], 1) if bare and at and words[at - 1] == "the" else None


def _times_scale(words: Sequence[str], at: int, read: _Read) -> _Read:
    """"two point five million": a decimal times the scale words after it."""
    if read is None:
        return None
    value, used = read
    last = 0
    while (scale := _SCALES.get(_at(words, at + used), 0)) and scale >= last:
        value, last, used = value * scale, scale, used + 1
    return value, used


def _on_the_hour(words: Sequence[str], at: int, read: _Read) -> _Read:
    """"ten o'clock" is 1000, as "10:00" is."""
    if read is None or _at(words, at + read[1]) != "oclock":
        return read
    return read[0] * 100, read[1] + 1


def _starting(words: Sequence[str], at: int) -> set[tuple[Value, int]]:
    """Every number that starts at words[at], with how many words it takes
    (number.js's `readings`)."""
    word = words[at]
    if word in ("minus", "negative"):
        if at + 1 == len(words):
            return set()
        return {
            (Value(-value.number), used + 1)
            for value, used in _starting(words, at + 1)
            if not (value.decade or value.digits)
        }
    found = set()
    if _CODE.fullmatch(word):  # "007": the zeros are part of it
        found.add((Value(Decimal(word), digits=word), 1))
    if (token := _token(word)) and token.decade:
        found.add((Value(Decimal(token.value), decade=True), 1))
    if decade := _decade(words, at):
        found.add((Value(Decimal(decade[0]), decade=True), decade[1]))
    if recited := _digit_run(words, at, least=2, hundred=True):
        digits, used = recited
        found.add((Value(Decimal(digits), digits=digits if digits.startswith("0") else ""), used))
    for read in (
        _times_scale(words, at, (Decimal(word), 1)) if _DECIMAL.fullmatch(word) else None,  # "2.5" as written
        _on_the_hour(words, at, _scaled(words, at)),
        _times_scale(words, at, _decimal(words, at)),
        _year(words, at),
        _money(words, at),
        _the_scale(words, at),
    ):
        if read:
            found.add((Value(Decimal(read[0])), read[1]))
    return found


def _parses(words: Sequence[str]) -> frozenset[Parse]:
    """Every sequence of numbers `words` read through as, left to right (a
    Parse; spoken.py's readings are the texts, this is what they mean)."""
    # ponytail: enumerates every parse. Fine for a phrase; exponential in the
    # number of ambiguous numbers in one text (spoken._MAX_SAID_DIGITS). For
    # whole chunks, match against a target instead (as identifier.js's readDigits does).
    after: list[frozenset[Parse]] = [frozenset()] * len(words) + [frozenset({()})]
    for at in reversed(range(len(words))):
        word, found = words[at], set()
        if word not in _NEVER_SKIPPED and not any(c.isdigit() for c in word):
            found |= after[at + 1]
        for value, used in _starting(words, at):
            if _at(words, at + used) not in _EXTENDS:
                found |= {(value, *rest) for rest in after[at + used]}
        after[at] = frozenset(found)
    return after[0]


# --------------------------------------------------------------------------- #
# Written text -> words (numerals keep their digits; symbols become words)
# --------------------------------------------------------------------------- #
_APOSTROPHES = str.maketrans({"’": "'", "‘": "'", "ʼ": "'"})
_OPENING = "\"'([“‘"
_CLOSING = "\"')]”’.,!?;:"
_DASHES = re.compile("[-\u2010\u2011\u2013\u2014]")  # hyphen, non-breaking hyphen, en and em dash
_CURRENCY_SIGNS = "$£€"  # spoken._CURRENCIES says them; a test keeps the two in step
_SIGN = f"[{re.escape(_CURRENCY_SIGNS)}]"
_GLUED = re.compile(rf"([A-Za-z']+)({_SIGN}\d.*)")  # Parakeet glues a price to the word before: "cost$25"
_CLOCK = re.compile(r"(\d{1,2}?)(?:([:.]?)(\d{2}))?([ap]\.?m\.?)?", re.IGNORECASE)
_AM_PM = re.compile(rf"[ap]\.?m\.?[{re.escape(_CLOSING)}]*", re.IGNORECASE)
_PAIR = re.compile(r"(\d+)-(\d+)")
_NUMERAL = re.compile(rf"(-)?({_SIGN})?(\d{{1,3}}(?:,\d{{3}})+|\d+)(?:\.(\d+))?(\D*)")
# Only after a currency: "5m" may be metres.
_CURRENCY_SCALES = {"k": "thousand", "m": "million", "bn": "billion"}
_MINOR_SUFFIXES = ("p", "P", "c", "¢")  # "50p", "99c"; "12C" is a seat


def _numeral_words(sign: str, currency: str, whole: str, fraction: str, suffix: str) -> list[tuple[str, ...]]:
    whole = whole.replace(",", "")
    lower = suffix.lower()
    plain = not (currency or fraction)
    if plain and lower in ("s", "'s"):  # "1970s" is also said "seventies"
        options = [(whole + "s",)] + ([(f"{int(whole) % 100}s",)] if int(whole) >= 100 else [])
    elif plain and suffix in _MINOR_SUFFIXES:  # "1080p" maybe no price
        options = [(str(Decimal(whole) / 100),)] + ([(whole,)] if int(whole) >= 100 else [])
    elif currency and lower in _CURRENCY_SCALES:
        options = [(f"{whole}.{fraction}" if fraction else whole, _CURRENCY_SCALES[lower])]
    else:  # currencies and units don't change the number; "5m", "12C" are names
        options = [(f"{whole}.{fraction}" if fraction else whole,)]
    return [("minus", *option) if sign else option for option in options]


def _token_words(token: str) -> list[tuple[str, ...]]:
    """One written word as the words the reader takes, in every way it can be
    meant: "£2.50" -> ("2.50",), "10:00" -> ("1000",) or ("10",)."""
    token = token.translate(_APOSTROPHES).lstrip(_OPENING).rstrip(_CLOSING)
    if match := _GLUED.fullmatch(token):
        return [a + b for a in _token_words(match[1]) for b in _token_words(match[2])]
    if (match := _CLOCK.fullmatch(token)) and (match[2] == ":" or match[4]):
        hour, minute = int(match[1]), int(match[3] or 0)
        return [(f"{hour}{minute:02d}",)] + ([(str(hour),)] if not minute else [])
    if match := _PAIR.fullmatch(token):
        first, second = match.groups()
        options = [(first, "to", second)]  # "to" keeps "2-1" from reading as a recited 21
        if len(first) == 4 and len(second) == 2:  # a season: "2024-25", "1999-00"
            options.append((first, "to", season_end(first, second)))
        if len(first) >= 3 or len(second) >= 4:  # a phone number: "555-1234"
            options.append((first + second,))
        return options
    if match := _NUMERAL.fullmatch(token):
        return _numeral_words(*(group or "" for group in match.groups()))
    parts = [part for part in _DASHES.split(token) if part]
    if len(parts) > 1:  # "twenty-five", "COVID-19"
        return [sum(choice, ()) for choice in product(*map(_token_words, parts))]
    plain = re.sub(r"[^a-z0-9]", "", unicodedata.normalize("NFKD", token.lower()))
    if plain.isalnum() and not plain.isalpha() and not plain.isdigit():  # "MP3": the digits of a name
        return [tuple(re.findall(r"\d+", plain))]
    return [(plain,) if plain else ()]


# --------------------------------------------------------------------------- #
# Public
# --------------------------------------------------------------------------- #
def values(text: str) -> frozenset[Parse]:
    """Every way `text` reads as numbers, left to right, spoken or written:
    values("two to one") == values("2-1") == {(Value(2), Value(1))}. Text with
    no numbers reads as {()}; text that can't be read through as frozenset()."""
    tokens = text.split()
    for index in reversed(range(1, len(tokens))):  # "800 p.m." is a time, as "800p.m." is
        if _AM_PM.fullmatch(tokens[index]) and tokens[index - 1][-1:].isdigit():
            tokens[index - 1 : index + 1] = [tokens[index - 1] + tokens[index]]
    word_lists: list[tuple[str, ...]] = [()]
    for token in tokens:
        word_lists = [done + more for done in word_lists for more in _token_words(token)]
    return frozenset().union(*map(_parses, word_lists))


def means(said: str, written: str) -> bool:
    """Whether spoken text `said` can be read as the numbers `written` says."""
    return not values(said).isdisjoint(values(written))


def season_end(first: str, second: str) -> str:
    """The year a season written first-second ends in: ("2024", "25") ->
    "2025", and past a century ("1999", "00") -> "2000"."""
    return f"{int(first[:2]) + (int(second) < int(first[2:]))}{second}"
