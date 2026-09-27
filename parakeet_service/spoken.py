"""English numbers, money and units as spoken words.

Parakeet writes what it hears in written form ("$5", "25 lb", "10:30"), and
different speech often comes out the same: "two pounds ten" and "two pounds
and ten pence" are both "£2.10", "ten thirty" is "1030". Two consumers want the
spoken form back:

- the word aligner needs letters it can find in the audio; it reads digits
  everywhere (spoken_word(..., everywhere=True)), even inside "MP3";
- PARAKEET_SPOKEN_NUMBERS returns a transcript that says what was said; it
  rewrites whole numeric words only, and readings() lists every plausible way
  each one could have been said so the audio can pick (routes._speak_numbers).

The first reading of each word is the one used when there is no audio.
"""
from __future__ import annotations

import itertools
import re
from typing import Optional, Sequence

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
# symbol: (singular, plural, minor singular, minor plural)
_CURRENCIES = {
    "$": ("dollar", "dollars", "cent", "cents"),
    "£": ("pound", "pounds", "penny", "pence"),
    "€": ("euro", "euros", "cent", "cents"),
}
# "dollar" or "dollars" -> "dollars": after a scale word the unit is always plural
_CURRENCY_PLURAL = {word: c[1] for c in _CURRENCIES.values() for word in c[:2]}
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
# Scale abbreviations only count after a currency: "$5m" is five million
# dollars, but a bare "5m" could be metres, so it keeps its letter.
_SCALE_ABBREVIATIONS = {"k": "thousand", "m": "million", "bn": "billion"}
# Minor currency after a small number: "50p", "99c", "99¢".
_MINOR_SUFFIXES = {"p": "£", "c": "$", "¢": "$"}
_TIME_SUFFIXES = {"am", "pm", "a.m.", "p.m."}
_SUFFIXES = "|".join(
    re.escape(s)
    for s in sorted(
        {*_UNIT_WORDS, *_SCALE_ABBREVIATIONS, *_MINOR_SUFFIXES, *_TIME_SUFFIXES, "s", "'s"},
        key=len,
        reverse=True,
    )
)
# A leading minus counts only at the start of a word: "mid-2020s" is not negative.
_NUMBER_PATTERN = (
    r"(?:(?<!\w)(-))?([$£€])?(\d{1,3}(?:,\d{3})+|\d+)(\.\d+)?(st|nd|rd|th)?"
    rf"(?:({_SUFFIXES})(?![a-z]))?"
)
_NUMBER = re.compile(_NUMBER_PATTERN, re.IGNORECASE)
_OPEN = r"([\"'(\[“‘]*)"
_CLOSE = r"([\"')\]”’.,!?;:]*)"
# A whole word that is just a number (with its currency, unit and punctuation),
# possibly glued to the word before by a currency sign: Parakeet writes "cost$25".
_WHOLE = re.compile(rf"{_OPEN}(?:([A-Za-z']+)(?=[$£€]))?{_NUMBER_PATTERN}{_CLOSE}", re.IGNORECASE)
_TIME = re.compile(rf"{_OPEN}(\d{{1,2}}):(\d{{2}})(?:\s?(am|pm|a\.m\.|p\.m\.))?{_CLOSE}", re.IGNORECASE)
_PAIR = re.compile(rf"{_OPEN}(\d+)-(\d+){_CLOSE}")  # scores, phone numbers: "2-1", "555-1234"
_TRAILING_PUNCTUATION = re.compile(r"[\"')\]”’.,!?;:]+$")
_SENTENCE_END = re.compile(r"[.!?][\"')\]”’]*$")


# --------------------------------------------------------------------------- #
# Numbers
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


def _pair(n: int) -> str:
    """Said in two halves: 1999 -> "nineteen ninety-nine", 1905 -> "nineteen oh
    five", 1900 -> "nineteen hundred", 1030 -> "ten thirty", 911 -> "nine eleven"."""
    head, tail = divmod(n, 100)
    if tail == 0:
        return f"{_cardinal(head)} hundred"
    return f"{_cardinal(head)} {'oh ' if tail < 10 else ''}{_cardinal(tail)}"


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


# ponytail: every mix of grouped and ungrouped runs ("oh double seven six six
# double one") up to this many per zero word; past it only all-ungrouped and
# all-grouped. Scoring one reading costs well under a millisecond.
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
    for zero in ("zero", "oh", "nought"):
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
        pairs = [digits[i : i + 2] for i in range(0, len(digits), 2)]
        options.append(" ".join(_cardinal(int(p)) if p[0] != "0" else f"oh {_ONES[int(p[1])]}" for p in pairs))
    return _unique(options)


def _unique(items) -> list[str]:
    return list(dict.fromkeys(item for item in items if item))


def _amounts(n: int) -> list[str]:
    """Ways to say n as a quantity, most likely first."""
    options = [_cardinal(n), _cardinal(n, british=True)]
    for said in list(options):
        if said.startswith("one "):  # "a hundred and fifty", "a thousand"
            options.append("a " + said[4:])
    if 1100 <= n <= 9999 and (n // 100) % 10 and n < 10000:  # "fifteen hundred (and fifty)"
        rest = n % 100
        hundreds = f"{_cardinal(n // 100)} hundred"
        options += [hundreds] if rest == 0 else [f"{hundreds} {_cardinal(rest)}", f"{hundreds} and {_cardinal(rest)}"]
    return _unique(options)


def _number_readings(digits: str, *, amount: bool) -> list[str]:
    """Ways to say a digit string; `amount` limits them to quantities (money,
    units, percent), where "nine one one" or "ten thirty" make no sense."""
    plain = digits.replace(",", "")
    code = "," not in digits and plain.startswith("0") and len(plain) > 1
    if len(plain) > 15 or code:
        # Codes, IDs and phone numbers are read digit by digit.
        return _recitations(plain)
    n = int(plain)
    amounts = _amounts(n)
    if amount or "," in digits:
        return amounts
    options = list(amounts)
    if len(plain) == 4 and (1100 <= n <= 1999 or 2010 <= n <= 2099):
        # ponytail: 4-digit numbers in year range default to the year reading
        # ("twenty twenty-six"); the audio picks otherwise when it is available.
        options.insert(0, _pair(n))
    if len(plain) in (3, 4):
        options.append(_pair(n))  # "ten thirty", "nine eleven", "forty-four twenty-one"
        if n % 100 == 0 and 1 <= n // 100 <= 12:
            options.append(f"{_cardinal(n // 100)} o'clock")
        if 2000 <= n <= 2009:
            options.append(f"twenty oh {_ONES[n % 10]}")
    if len(plain) >= 2:
        options += _recitations(plain)
    return _unique(options)


def _money(groups: Sequence[Optional[str]], integer: list[str]) -> list[str]:
    _sign, currency, digits, decimals, _ordinal_suffix, suffix = groups
    singular, plural, minor_one, minor_many = _CURRENCIES[currency]
    major = int(digits.replace(",", ""))
    bare = integer  # said with no unit after it: never just "a"
    if major == 1:
        integer = [*integer, "a"]  # "a dollar fifty", "a quid"
    fraction = (decimals or ".")[1:]
    scale = _SCALE_ABBREVIATIONS.get((suffix or "").lower())
    if len(fraction) == 2 and not scale:
        cents = int(fraction)
        minor = minor_one if cents == 1 else minor_many
        said_cents = _cardinal(cents)
        if major == 0:
            return _unique([f"{said_cents} {minor}", f"{said_cents} p" if currency == "£" else ""])
        unit = singular if major == 1 else plural
        options = []
        for said in integer:
            if cents == 0:
                options.append(f"{said} {unit}")
                continue
            options += [
                f"{said} {unit} {said_cents}",
                f"{said} {unit} and {said_cents} {minor}",
                f"{said} {unit} {said_cents} {minor}",
            ]
            if currency == "£":
                options.append(f"{said} {unit} {said_cents} p")
        if cents:
            options += [f"{said} {said_cents}" for said in bare]  # "five fifty"
        return _unique(options)
    point = f" point {_digits(fraction)}" if fraction else ""
    unit = singular if major == 1 and not fraction and not scale else plural
    scale_word = f" {scale}" if scale else ""
    options = [f"{said}{point}{scale_word} {unit}" for said in integer]
    if not fraction and not scale:
        slang = {"£": ("quid", "quid"), "$": ("buck", "bucks")}.get(currency)
        if slang:  # "fifty quid", "five bucks"
            options += [f"{said} {slang[0] if major == 1 else slang[1]}" for said in integer]
        options += bare  # the unit left unsaid: "£250" as "two hundred and fifty"
    return _unique(options)


def _readings(groups: Sequence[Optional[str]], *, strict: bool) -> list[str]:
    """Readings of one matched number, most likely first; [] if `strict` and
    it is ambiguous ("5m": metres or million?)."""
    sign, currency, digits, decimals, ordinal, suffix = groups
    suffix = (suffix or "").lower()
    plain = digits.replace(",", "")
    if suffix in _MINOR_SUFFIXES and not currency and not decimals:
        if int(plain) >= 100 and suffix != "¢":
            return [] if strict else [f"{_cardinal(int(plain))} {suffix}"]  # "1080p"
        currency = _MINOR_SUFFIXES[suffix]
        groups = (sign, currency, "0", "." + plain.zfill(2), ordinal, "")
        return _signed(sign, _money(groups, []))
    if strict and suffix and suffix not in _UNIT_WORDS and suffix not in _TIME_SUFFIXES and not (
        currency and suffix in _SCALE_ABBREVIATIONS
    ) and suffix not in ("s", "'s"):
        return []  # "5m", "5k": metres or million, a race or money
    if suffix in _TIME_SUFFIXES and not (currency or decimals or ordinal) and len(plain) <= 4:
        return _signed(sign, [f"{said} {suffix}" for said in _clock(plain)])
    if suffix in ("s", "'s") and not (currency or decimals or ordinal):
        decades = [_plural(said) for said in _number_readings(digits, amount=False)[:2]]
        if int(plain) >= 100:  # "1970s" is often just "seventies"
            decades.append(_plural(_cardinal(int(plain) % 100)))
        return _signed(sign, _unique(decades))
    amount = bool(currency or decimals or ordinal or suffix in _UNIT_WORDS)
    integer = _number_readings(digits, amount=amount)
    if ordinal:
        options = _unique(_ordinal(said) for said in integer)
    elif currency:
        options = _money((sign, currency, digits, decimals, ordinal, suffix), integer)
    else:
        options = integer
        if decimals:
            fraction = decimals[1:]
            tails = [f"point {_digits(fraction)}", f"point {_digits(fraction, zero='oh')}"]
            if len(fraction) == 2 and fraction[0] != "0":
                tails.append(f"point {_cardinal(int(fraction))}")  # version "3.12"
            heads = options
            if int(plain) == 0:
                heads = ["zero", "nought", "oh", ""]  # "point five"
            options = [f"{head} {tail}".strip() for head in heads for tail in tails]
            hundredths = int(fraction.ljust(2, "0")[:2]) if len(fraction) <= 2 else None
            parts = {50: ["a half"], 25: ["a quarter", "a fourth"], 75: ["three quarters", "three fourths"]}
            for part in parts.get(hundredths, []):  # "ten and a half"
                options += [f"{head} and {part}" if int(plain) else part for head in heads]
            if len(fraction) == 2 and int(plain):  # "2.50": "two fifty", "1.05": "one oh five"
                said = f"{'oh ' if fraction[0] == '0' else ''}{_cardinal(int(fraction))}"
                options += [f"{head} {said}" for head in heads]
            options = _unique(options)
        if suffix in _UNIT_WORDS:
            count = digits if not decimals else ""
            options = [f"{said} {_unit(suffix, count)}" for said in options]
        elif suffix and not strict:
            options = [f"{said} {suffix}" for said in options]  # "5m", "5k": left as written
    return _signed(sign, options)


def _clock(digits: str) -> list[str]:
    """"715" -> "seven fifteen", "6" -> "six", "1000" -> "ten o'clock"."""
    n = int(digits)
    if len(digits) <= 2:
        return [_cardinal(n)]
    hour, minute = divmod(n, 100)
    if minute == 0:
        return [f"{_cardinal(hour)} o'clock", _cardinal(hour), f"{_cardinal(hour)} hundred"]
    return [f"{_cardinal(hour)} {'oh ' if minute < 10 else ''}{_cardinal(minute)}"]


def _signed(sign: Optional[str], options: list[str]) -> list[str]:
    if not sign:
        return options
    return _unique([f"minus {said}" for said in options] + [f"negative {said}" for said in options])


def _unit(key: str, count: str) -> str:
    singular, plural = _UNIT_WORDS[key]
    return singular if count == "1" else plural


# --------------------------------------------------------------------------- #
# Words
# --------------------------------------------------------------------------- #
def readings(word: str) -> list[str]:
    """Plausible spoken forms of one whole Parakeet word, most likely first.

    [word] when it is not a whole number (names like "MP3" stay as written).
    """
    if match := _WHOLE.fullmatch(word):
        opening, glued, *number, closing = match.groups()
        options = _readings(number, strict=True)
        prefix = f"{glued} " if glued else ""
        return [f"{opening}{prefix}{said}{closing}" for said in options] or [word]
    if match := _TIME.fullmatch(word):
        opening, hour, minute, suffix, closing = match.groups()
        n = int(hour) * 100 + int(minute)
        options = _clock(f"{n:03d}") + ([f"{_cardinal(int(hour))} {_cardinal(int(minute))}"] if int(minute) >= 10 else [])
        tail = f" {suffix}" if suffix else ""
        return _unique(f"{opening}{said}{tail}{closing}" for said in options)
    if match := _PAIR.fullmatch(word):
        opening, first, second, closing = match.groups()
        phone = len(first) >= 3 or len(second) >= 4
        numbers = [_digits(first) + " " + _digits(second), _digits(first, zero="oh") + " " + _digits(second, zero="oh")]
        score = []
        for said_first in _amounts(int(first)):
            score += [f"{said_first} {said} " for said in _amounts(int(second))]
            score += [f"{said_first} to {said}" for said in _amounts(int(second))]
            if int(second) == 0:
                score.append(f"{said_first} nil")
        score = [said.strip() for said in score]
        options = numbers + score if phone else score + numbers
        return _unique(f"{opening}{said}{closing}" for said in options)
    return [word]


def spoken_word(word: str, *, everywhere: bool = False) -> str:
    """`word` with its number said out (its first reading); unchanged if there
    is nothing to say.

    `everywhere` also reads digits inside other words ("MP3", "mid-2020s") and
    ambiguous suffixes ("5m"), which the aligner wants and a transcript doesn't.
    """
    if everywhere:
        return _NUMBER.sub(lambda m: f" {_readings(m.groups(), strict=False)[0]} ", word).strip()
    return readings(word)[0]


def spoken_words(
    words: Sequence[str],
    *,
    everywhere: bool = False,
    choices: Optional[Sequence[Optional[str]]] = None,
) -> list[str]:
    """Each of `words` as spoken (possibly several words each), in speaking order.

    `choices[i]`, when given, is the reading to use for word i (from readings()).
    """
    spoken = [
        (choices[index] if choices and choices[index] is not None else spoken_word(word, everywhere=everywhere)).split()
        for index, word in enumerate(words)
    ]
    # A unit written as its own word after a number: "25 lb" is "twenty-five pounds".
    for index in range(1, len(words)):
        key = _TRAILING_PUNCTUATION.sub("", words[index]).lower()
        previous = _TRAILING_PUNCTUATION.sub("", words[index - 1])
        if key in _UNIT_WORDS and any(c.isdigit() for c in previous):
            trailing = words[index][len(_TRAILING_PUNCTUATION.sub("", words[index])) :]
            spoken[index] = (_unit(key, previous) + trailing).split()
    # "$5 million" is said "five million dollars": the unit follows the scale word.
    for current, following in zip(spoken, spoken[1:]):
        if len(current) > 1 and current[-1] in _CURRENCY_PLURAL and following:
            scale_word = _TRAILING_PUNCTUATION.sub("", following[0])
            if scale_word.lower() in _SCALE_WORDS:
                unit = _CURRENCY_PLURAL[current.pop()]
                trailing = following[0][len(scale_word) :]  # "million." -> "million dollars."
                following[0] = scale_word
                following.insert(1, unit + trailing)
    return [" ".join(parts) for parts in spoken]


def starts_sentence(previous_word: Optional[str]) -> bool:
    """Whether the word after `previous_word` begins a sentence (None: first word)."""
    return previous_word is None or bool(_SENTENCE_END.search(previous_word))


def capitalize(text: str) -> str:
    """Upper-case the first letter, past any leading punctuation ("(five" -> "(Five")."""
    for index, char in enumerate(text):
        if char.isalpha():
            return text[:index] + char.upper() + text[index + 1 :]
    return text
