"""English numbers, money and units as spoken words.

Parakeet writes what it hears in written form ("$5", "25 lb", "10:30"). Two
consumers want the spoken form back: the word aligner, which needs letters it
can find in the audio, and PARAKEET_SPOKEN_NUMBERS, which returns a transcript
that says what was said. Both use spoken_words(); the aligner in `everywhere`
mode (digits inside names too: "MP3" -> "MP three"), the transcript only for
whole numeric words, so names are left as written.
"""
from __future__ import annotations

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
_CURRENCIES = {"$": ("dollar", "dollars"), "£": ("pound", "pounds"), "€": ("euro", "euros")}
# "dollar" or "dollars" -> "dollars": after a scale word the unit is always plural
_CURRENCY_PLURAL = {word: plural for singular, plural in _CURRENCIES.values() for word in (singular, plural)}
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
_SUFFIXES = "|".join(
    re.escape(s) for s in sorted({*_UNIT_WORDS, *_SCALE_ABBREVIATIONS}, key=len, reverse=True)
)
# A leading minus counts only at the start of a word: "mid-2020s" is not negative.
_NUMBER_PATTERN = (
    r"(?:(?<!\w)(-))?([$£€])?(\d{1,3}(?:,\d{3})+|\d+)(\.\d+)?(st|nd|rd|th)?"
    rf"(?:({_SUFFIXES})(?![a-z]))?"
)
_NUMBER = re.compile(_NUMBER_PATTERN, re.IGNORECASE)
# A whole word that is just a number (with its currency, unit and punctuation).
_WHOLE = re.compile(rf"([\"'(\[“‘]*){_NUMBER_PATTERN}([\"')\]”’.,!?;:]*)", re.IGNORECASE)
_TRAILING_PUNCTUATION = re.compile(r"[\"')\]”’.,!?;:]+$")
_SENTENCE_END = re.compile(r"[.!?][\"')\]”’]*$")


def _cardinal(n: int) -> str:
    if n < 20:
        return _ONES[n]
    if n < 100:
        return _TENS[n // 10] + ("" if n % 10 == 0 else "-" + _ONES[n % 10])
    if n < 1000:
        rest = "" if n % 100 == 0 else " " + _cardinal(n % 100)
        return f"{_ONES[n // 100]} hundred{rest}"
    for scale, name in _SCALES:
        if n >= scale:
            head, rest = divmod(n, scale)
            return f"{_cardinal(head)} {name}" + ("" if rest == 0 else " " + _cardinal(rest))
    raise AssertionError("unreachable")


def _year(n: int) -> str:
    # 1999 -> nineteen ninety-nine, 1905 -> nineteen oh five, 1900 -> nineteen hundred
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


def _digits(digits: str) -> str:
    return " ".join(_ONES[int(d)] for d in digits)


def _unit(key: str, count: str) -> str:
    singular, plural = _UNIT_WORDS[key]
    return singular if count == "1" else plural


def _say(groups: Sequence[Optional[str]], *, strict: bool) -> Optional[str]:
    """Spoken form of one matched number; None if `strict` and it is ambiguous."""
    sign, currency, digits, decimals, ordinal, suffix = groups
    suffix = (suffix or "").lower()
    scale = _SCALE_ABBREVIATIONS.get(suffix) if currency else None
    if strict and suffix and not scale and suffix not in _UNIT_WORDS:
        return None  # "5m", "5k": metres or million, a race or money
    plain = digits.replace(",", "")
    if len(plain) > 15 or ("," not in digits and plain.startswith("0") and len(plain) > 1):
        # Codes, IDs and phone numbers are read digit by digit.
        spoken = _digits(plain)
    elif len(digits) == 4 and not (currency or ordinal or decimals) and (
        1100 <= int(plain) <= 1999 or 2010 <= int(plain) <= 2099
    ):
        # ponytail: 4-digit numbers in year range read as years ("twenty
        # twenty-six"). A count said "one thousand five hundred" then starts
        # ~160 ms late (measured). Needs context to tell a year from a count.
        spoken = _year(int(plain))
    else:
        spoken = _cardinal(int(plain))
    if ordinal:
        spoken = _ordinal(spoken)
    fraction = (decimals or ".")[1:]
    if fraction and not (currency and len(fraction) == 2 and not scale):
        # "$2.5 million" is "two point five million dollars"; only 2 digits are cents
        spoken += " point " + _digits(fraction)
        fraction = ""
    if scale:
        spoken += " " + scale
    if currency:
        singular, plural = _CURRENCIES[currency]
        unit = singular if plain == "1" and not decimals and not scale else plural
        spoken = f"{spoken} {unit}" + (f" {_cardinal(int(fraction))}" if fraction.strip("0") else "")
    if suffix in _UNIT_WORDS:
        spoken += " " + _unit(suffix, digits if not decimals else "")
    elif suffix and not scale:
        spoken += " " + suffix  # "5m", "5k": left as written
    if sign:
        spoken = "minus " + spoken
    return spoken


def spoken_word(word: str, *, everywhere: bool = False) -> str:
    """`word` with its number said out; unchanged if there is nothing to say.

    `everywhere` also reads digits inside other words ("MP3", "mid-2020s") and
    ambiguous suffixes ("5m"), which the aligner wants and a transcript doesn't.
    """
    if everywhere:
        return _NUMBER.sub(lambda m: f" {_say(m.groups(), strict=False)} ", word).strip()
    match = _WHOLE.fullmatch(word)
    if match is None:
        return word
    spoken = _say(match.groups()[1:-1], strict=True)
    return word if spoken is None else f"{match.group(1)}{spoken}{match.groups()[-1]}"


def spoken_words(words: Sequence[str], *, everywhere: bool = False) -> list[str]:
    """Each of `words` as spoken (possibly several words each), in speaking order."""
    spoken = [spoken_word(word, everywhere=everywhere).split() for word in words]
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
