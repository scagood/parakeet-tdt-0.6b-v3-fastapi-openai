"""Cover the in-process WAV decode path and where it hands off to ffmpeg.

`audioop` left the stdlib in 3.13 (PEP 594), so the decoder does its own
channel downmix and integer conversion in numpy. Sample-rate conversion is
deliberately not done here: see the measurements in OPTIMIZATION.md.
"""
from __future__ import annotations

import shutil
import struct
import wave
from io import BytesIO

import numpy as np
import pytest

from parakeet_service import audio
from parakeet_service.config import TARGET_SR


def build_wav(samples: bytes, *, rate: int = TARGET_SR, channels: int = 1, width: int = 2) -> bytes:
    buffer = BytesIO()
    with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(channels)
        wav_file.setsampwidth(width)
        wav_file.setframerate(rate)
        wav_file.writeframes(samples)
    return buffer.getvalue()


def pack24(values) -> bytes:
    return b"".join(struct.pack("<i", v)[:3] for v in values)


def decode(data: bytes):
    info = audio._wav_info(data)
    assert info is not None
    return audio._decode_pcm_wav(data, info)


# --- widths, all at 16 kHz mono ---------------------------------------------

def test_decodes_8_bit_unsigned():
    data = build_wav(bytes([128, 192, 64, 255, 0]), width=1)
    np.testing.assert_array_equal(
        decode(data), np.float32([0.0, 0.5, -0.5, 127 / 128, -1.0])
    )


def test_decodes_16_bit_signed():
    pcm = struct.pack("<5h", 0, 16384, -16384, 32767, -32768)
    np.testing.assert_array_equal(
        decode(build_wav(pcm)), np.float32([0.0, 0.5, -0.5, 32767 / 32768, -1.0])
    )


def test_decodes_24_bit_signed():
    pcm = pack24([0, 1 << 22, -(1 << 22), (1 << 23) - 1, -(1 << 23)])
    np.testing.assert_array_equal(
        decode(build_wav(pcm, width=3)),
        np.float32([0.0, 0.5, -0.5, ((1 << 23) - 1) / (1 << 23), -1.0]),
    )


def test_decodes_32_bit_signed():
    pcm = struct.pack("<3i", 0, 1 << 30, -(1 << 30))
    np.testing.assert_array_equal(
        decode(build_wav(pcm, width=4)), np.float32([0.0, 0.5, -0.5])
    )


# --- channels ----------------------------------------------------------------

def test_stereo_is_downmixed_to_the_channel_mean():
    pcm = struct.pack("<6h", 16384, 0, -16384, -16384, 32767, -32768)
    np.testing.assert_allclose(
        decode(build_wav(pcm, channels=2)),
        np.float32([0.25, -0.5, (32767 / 32768 - 1.0) / 2]),
        atol=1e-7,
    )


def test_more_than_two_channels_defers_to_ffmpeg():
    pcm = struct.pack("<6h", *range(6))
    assert decode(build_wav(pcm, channels=6)) is None


# --- handoffs to ffmpeg ------------------------------------------------------

@pytest.mark.parametrize("rate", [8_000, 22_050, 44_100, 48_000])
def test_other_sample_rates_defer_to_ffmpeg(rate):
    assert decode(build_wav(struct.pack("<2h", 0, 1), rate=rate)) is None


def test_compressed_wav_defers_to_ffmpeg():
    data = build_wav(struct.pack("<2h", 0, 1))
    info = audio._wav_info(data)
    info["compression"] = "ULAW"
    assert audio._decode_pcm_wav(data, info) is None


def test_wav_info_rejects_non_wav_bytes():
    assert audio._wav_info(b"ID3\x04not audio at all") is None
    assert audio._wav_info(b"") is None


def test_load_audio_uses_ffmpeg_only_when_conversion_is_needed(monkeypatch):
    calls = []
    monkeypatch.setattr(
        audio, "_ffmpeg_decode", lambda data: calls.append(data) or np.float32([0.25])
    )
    pcm = struct.pack("<2h", 16384, -16384)

    np.testing.assert_array_equal(
        audio.load_audio(build_wav(pcm)), np.float32([0.5, -0.5])
    )
    assert calls == []

    np.testing.assert_array_equal(
        audio.load_audio(build_wav(pcm, rate=44_100)), np.float32([0.25])
    )
    assert len(calls) == 1

    np.testing.assert_array_equal(audio.load_audio(b"OggS\x00 not a wav"), np.float32([0.25]))
    assert len(calls) == 2


def test_load_audio_returns_empty_for_empty_input():
    result = audio.load_audio(b"")
    assert result.dtype == np.float32
    assert result.size == 0


# --- shape guarantees --------------------------------------------------------

def test_result_is_contiguous_float32():
    pcm = struct.pack("<4h", 1, 2, 3, 4)
    for data in (build_wav(pcm), build_wav(pcm, channels=2)):
        result = decode(data)
        assert result.dtype == np.float32
        assert result.ndim == 1
        assert result.flags["C_CONTIGUOUS"]


def test_truncated_final_frame_is_dropped_not_raised():
    # A file cut mid-frame leaves a dangling channel sample; drop it rather
    # than reshaping a ragged buffer.
    data = build_wav(struct.pack("<4h", 1, 2, 3, 4), channels=2)[:-2]
    info = {"compression": "NONE", "sample_rate": TARGET_SR, "channels": 2, "sample_width": 2}
    np.testing.assert_allclose(
        audio._decode_pcm_wav(data, info), np.float32([1.5 / 32768]), atol=1e-9
    )


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")
@pytest.mark.parametrize("channels,width", [(1, 2), (2, 2), (1, 3), (2, 1)])
def test_in_process_decode_agrees_with_ffmpeg(channels, width):
    frames = 4000
    signal = np.sin(np.linspace(0, 120, frames * channels)) * 0.8
    if width == 1:
        pcm = (signal * 127 + 128).astype(np.uint8).tobytes()
    elif width == 2:
        pcm = (signal * 32767).astype("<i2").tobytes()
    else:
        pcm = pack24((signal * 8388607).astype(np.int64))
    data = build_wav(pcm, channels=channels, width=width)

    ours = decode(data)
    theirs = audio._ffmpeg_decode(data)
    assert ours is not None
    assert ours.size == theirs.size
    # ffmpeg emits s16, so it quantizes; one LSB of int16 is the tolerance.
    np.testing.assert_allclose(ours, theirs, atol=2 / 32768)


# --- the ffmpeg path, which carries everything not already at 16 kHz ---------

requires_ffmpeg = pytest.mark.skipif(
    shutil.which("ffmpeg") is None, reason="ffmpeg not installed"
)


def tone_wav(seconds: float, *, rate: int, channels: int) -> bytes:
    frames = int(rate * seconds)
    t = np.arange(frames) / rate
    wave_data = np.sin(2 * np.pi * 440 * t) * 0.6
    if channels == 2:
        wave_data = np.repeat(wave_data, 2)
    return build_wav((wave_data * 32767).astype("<i2").tobytes(), rate=rate, channels=channels)


@requires_ffmpeg
@pytest.mark.parametrize("rate,channels", [(44_100, 2), (48_000, 2), (22_050, 1), (8_000, 1)])
def test_ffmpeg_resamples_to_mono_16k(rate, channels):
    result = audio.load_audio(tone_wav(0.5, rate=rate, channels=channels))
    assert result.dtype == np.float32
    assert result.ndim == 1
    assert abs(result.size - TARGET_SR // 2) < TARGET_SR // 50  # ~0.5 s, ±20 ms
    assert np.isfinite(result).all()
    assert 0.2 < float(np.abs(result).max()) <= 1.0


@requires_ffmpeg
def test_ffmpeg_decode_reports_undecodable_input():
    with pytest.raises(RuntimeError, match="ffmpeg decode failed"):
        audio._ffmpeg_decode(b"RIFF\x00\x00\x00\x00WAVEjunk" + b"\x00" * 64)


@requires_ffmpeg
def test_ffmpeg_decode_handles_a_non_wav_container():
    # Ask ffmpeg for a FLAC, then feed it back through the decoder.
    import subprocess

    source = tone_wav(0.25, rate=44_100, channels=2)
    flac = subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-i", "pipe:0", "-f", "flac", "pipe:1"],
        input=source, capture_output=True, check=True,
    ).stdout
    assert audio._wav_info(flac) is None  # not a WAV, so the fast path declines it
    result = audio.load_audio(flac)
    assert result.ndim == 1 and result.size > 0 and np.isfinite(result).all()
