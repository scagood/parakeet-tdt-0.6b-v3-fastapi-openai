"""In-memory audio decoding and resampling."""
from __future__ import annotations

import subprocess
import wave
from io import BytesIO
from pathlib import Path
from typing import Optional

from .config import FFMPEG_TIMEOUT_SEC, TARGET_SR

import numpy as np


def _wav_info(data: bytes) -> Optional[dict]:
    try:
        with wave.open(BytesIO(data), "rb") as wav_file:
            return {
                "sample_rate": wav_file.getframerate(),
                "channels": wav_file.getnchannels(),
                "sample_width": wav_file.getsampwidth(),
                "compression": wav_file.getcomptype(),
            }
    except (wave.Error, EOFError, OSError):
        return None


_PCM_DTYPES = {2: "<i2", 4: "<i4"}


def _pcm_to_float(pcm: bytes, sample_width: int) -> np.ndarray:
    """Convert interleaved little-endian PCM to float32 in [-1, 1)."""
    if sample_width == 1:  # 8-bit WAV is unsigned; everything wider is signed
        return (np.frombuffer(pcm, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
    if sample_width == 3:
        # numpy has no 24-bit dtype: copy each sample into the top three bytes
        # of an int32, which sign-extends it and scales it by 256 for free.
        raw = np.frombuffer(pcm, dtype=np.uint8)
        usable = raw.size - raw.size % 3
        packed = np.zeros((usable // 3, 4), dtype=np.uint8)
        packed[:, 1:] = raw[:usable].reshape(-1, 3)
        return packed.view("<i4").ravel().astype(np.float32) / 2147483648.0
    count = len(pcm) // sample_width
    samples = np.frombuffer(pcm, dtype=_PCM_DTYPES[sample_width], count=count)
    return samples.astype(np.float32) / float(1 << (8 * sample_width - 1))


def _decode_pcm_wav(data: bytes, info: dict) -> Optional[np.ndarray]:
    """Decode an uncompressed 16 kHz PCM WAV without leaving the process.

    Returns None for anything needing a sample-rate conversion, which
    `_ffmpeg_decode` does faster than numpy can. See OPTIMIZATION.md.
    """
    if info["compression"] != "NONE" or info["sample_rate"] != TARGET_SR:
        return None
    sample_width = info["sample_width"]
    channels = info["channels"]
    if sample_width not in (1, 2, 3, 4):
        return None
    if channels not in (1, 2):
        return None  # ffmpeg downmixes 5.1 and up with per-channel weights
    try:
        with wave.open(BytesIO(data), "rb") as wav_file:
            pcm = wav_file.readframes(wav_file.getnframes())
        samples = _pcm_to_float(pcm, sample_width)
    except (wave.Error, EOFError, OSError, ValueError):
        return None
    if channels == 2:
        usable = samples.size - samples.size % 2
        samples = samples[:usable].reshape(-1, 2).mean(axis=1, dtype=np.float32)
    return np.ascontiguousarray(samples, dtype=np.float32)


def _ffmpeg_decode(data: bytes) -> np.ndarray:
    """Decode a container/codec to mono 16 kHz float32 with one FFmpeg call."""
    command = [
        "ffmpeg",
        "-hide_banner",
        "-nostdin",
        "-loglevel",
        "error",
        "-i",
        "pipe:0",
        "-map",
        "0:a:0",
        "-vn",
        "-sn",
        "-dn",
        "-ac",
        "1",
        "-ar",
        str(TARGET_SR),
        "-f",
        "s16le",
        "pipe:1",
    ]
    try:
        process = subprocess.run(
            command,
            input=data,
            capture_output=True,
            check=False,
            timeout=FFMPEG_TIMEOUT_SEC,
        )
    except FileNotFoundError as exc:
        raise RuntimeError("ffmpeg is not installed or not available on PATH") from exc
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(
            f"ffmpeg decode exceeded {FFMPEG_TIMEOUT_SEC:g} seconds"
        ) from exc
    if process.returncode != 0:
        error = process.stderr.decode(errors="replace").strip()[:500]
        raise RuntimeError(f"ffmpeg decode failed: {error or 'unknown error'}")
    if not process.stdout:
        return np.empty(0, dtype=np.float32)
    result = np.frombuffer(process.stdout, dtype="<i2").astype(np.float32) / 32768.0
    return np.ascontiguousarray(result, dtype=np.float32)


def load_audio(data: bytes) -> np.ndarray:
    """Return a finite mono float32 waveform at 16 kHz."""
    if not data:
        return np.empty(0, dtype=np.float32)
    info = _wav_info(data)
    waveform = _decode_pcm_wav(data, info) if info is not None else None
    if waveform is None:
        waveform = _ffmpeg_decode(data)
    if waveform.ndim != 1:
        waveform = np.ravel(waveform)
    if not np.isfinite(waveform).all():
        waveform = np.nan_to_num(waveform, copy=False)
    return np.ascontiguousarray(waveform, dtype=np.float32)


def load_audio_path(path: Path) -> np.ndarray:
    return load_audio(Path(path).read_bytes())
