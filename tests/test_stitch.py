from __future__ import annotations

from types import SimpleNamespace

from parakeet_service import routes
from parakeet_service.config import TARGET_SR


def _prepared(ranges_sec):
    ranges = [(int(s * TARGET_SR), int(e * TARGET_SR)) for s, e in ranges_sec]
    duration = max(e for _s, e in ranges_sec)
    return routes._PreparedAudio(
        waveform=None, ranges=ranges, pieces=[None] * len(ranges), duration=duration
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
