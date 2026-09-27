from __future__ import annotations

from types import SimpleNamespace

from parakeet_service import routes
from parakeet_service.config import TARGET_SR


def _prepared(ranges_sec, pieces=None):
    ranges = [(int(s * TARGET_SR), int(e * TARGET_SR)) for s, e in ranges_sec]
    duration = max(e for _s, e in ranges_sec)
    return routes._PreparedAudio(
        waveform=None,
        ranges=ranges,
        pieces=pieces or [None] * len(ranges),
        duration=duration,
    )


def _result(tokens, timestamps):
    text = "".join(t.replace("▁", " ") for t in tokens).strip()
    return SimpleNamespace(text=text, tokens=tokens, timestamps=timestamps)


def test_word_end_does_not_absorb_pause():
    result = _result(
        [" Hello", " wor", "ld", ".", " Then"],
        [0.0, 0.5, 0.7, 0.9, 21.0],  # 20 s pause inside the chunk
    )
    _text, _segments, words = routes._stitch(_prepared([(0.0, 30.0)]), [result])
    assert [w["word"] for w in words] == ["Hello", "world.", "Then"]
    pre_pause = words[1]
    assert pre_pause["end"] <= 0.9 + routes._WORD_TAIL_SEC + 1e-9
    assert all(w["end"] - w["start"] < 1.5 for w in words)


def test_last_word_does_not_balloon_to_chunk_end():
    result = _result([" Deep", " breath", "."], [0.0, 0.4, 0.8])
    _text, _segments, words = routes._stitch(_prepared([(0.0, 75.0)]), [result])
    assert words[-1]["end"] <= 0.8 + routes._WORD_TAIL_SEC + 1e-9


def test_token_timestamp_mismatch_drops_no_words():
    result = _result([" one", " two", " three", " four"], [0.0, 0.5])
    _text, _segments, words = routes._stitch(_prepared([(0.0, 10.0)]), [result])
    assert [w["word"] for w in words] == ["one", "two", "three", "four"]
    starts = [w["start"] for w in words]
    assert starts == sorted(starts)


def test_invalid_timestamps_reuse_previous_and_drop_nothing():
    result = _result(
        [" one", " two", " three", " four"],
        [0.0, float("nan"), float("inf"), -5.0],
    )
    _text, _segments, words = routes._stitch(_prepared([(0.0, 10.0)]), [result])
    assert [w["word"] for w in words] == ["one", "two", "three", "four"]
    starts = [w["start"] for w in words]
    assert starts == sorted(starts)
    assert all(w["start"] >= 0.0 for w in words)


def test_bpe_pieces_group_into_one_word():
    result = _result(["▁lig", "ht", "ho", "use"], [0.0, 0.1, 0.2, 0.3])
    _text, _segments, words = routes._stitch(_prepared([(0.0, 5.0)]), [result])
    assert [w["word"] for w in words] == ["lighthouse"]
    assert words[0]["start"] == 0.0
    assert words[0]["end"] <= 0.3 + routes._WORD_TAIL_SEC + 1e-9


def test_second_chunk_words_use_chunk_offset():
    first = _result([" one", "."], [0.0, 0.3])
    second = _result([" two", "."], [0.5, 0.8])
    _text, _segments, words = routes._stitch(
        _prepared([(0.0, 10.0), (10.0, 20.0)]), [first, second]
    )
    assert [w["word"] for w in words] == ["one.", "two."]
    assert words[1]["start"] == 10.5


def test_lone_word_marker_before_digits_and_currency_starts_a_word():
    # Parakeet v3 tokens, verbatim: the marker comes alone before "£" and digits,
    # and onnx_asr's text join has already dropped the space before the "£".
    result = SimpleNamespace(
        text="The coffee was£1.10 in 2005.",
        tokens=[" The", " co", "ff", "ee", " was", " ", "£", "1", ".", "1", "0", " in", " ", "2", "0", "0", "5", "."],
        timestamps=[0.1 * i for i in range(18)],
    )
    text, segments, words = routes._stitch(_prepared([(0.0, 5.0)]), [result])
    assert [w["word"] for w in words] == ["The", "coffee", "was", "£1.10", "in", "2005."]
    assert text == segments[0]["segment"] == "The coffee was £1.10 in 2005."


def test_aligner_retimes_each_chunk_from_its_own_audio():
    first = _result([" hi", " there"], [0.0, 0.8])
    second = _result([" bye"], [0.0])
    calls = []

    def fake_align(chunk_wav, words):
        calls.append((chunk_wav, words))
        return [(0.25 + i, 0.5 + i) for i in range(len(words))]

    _text, _segments, words = routes._stitch(
        _prepared([(0.0, 2.0), (5.0, 7.0)], pieces=["chunk0", "chunk1"]),
        [first, second],
        fake_align,
    )
    assert calls == [("chunk0", ["hi", "there"]), ("chunk1", ["bye"])]
    assert [(w["word"], w["start"], w["end"]) for w in words] == [
        ("hi", 0.25, 0.5),
        ("there", 1.25, 1.5),
        ("bye", 5.25, 5.5),
    ]


def test_unavailable_aligner_keeps_model_times():
    prepared = _prepared([(0.0, 2.0)])
    result = _result([" hi"], [0.4])
    assert routes._stitch(prepared, [result], lambda _wav, _words: None) == routes._stitch(
        prepared, [result]
    )


def test_unaligned_words_stay_between_aligned_neighbours():
    words = [
        {"word": "a", "start": 0.0, "end": 0.3},
        {"word": "%", "start": 0.1, "end": 0.9},  # nothing the aligner can anchor
        {"word": "b", "start": 0.5, "end": 0.8},
    ]
    routes._apply_alignment(words, [(0.10, 0.20), None, (0.40, 0.60)], 10.0, 20.0)
    assert (words[0]["start"], words[0]["end"]) == (10.1, 10.2)
    assert (words[2]["start"], words[2]["end"]) == (10.4, 10.6)
    assert words[0]["end"] <= words[1]["start"] <= words[1]["end"] <= words[2]["start"]
