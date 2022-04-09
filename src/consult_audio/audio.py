"""Decoding, validation and consistent mono 16-kHz preprocessing."""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from io import BytesIO

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

TARGET_SR = 16_000
SUPPORTED_EXTENSIONS = {".wav", ".flac", ".ogg"}


class AudioError(ValueError):
    """Invalid or unsupported user audio."""


@dataclass(frozen=True)
class AudioInfo:
    sample_rate: int
    channels: int
    frames: int
    duration: float
    peak: float


def validate_audio(x: np.ndarray, sr: int) -> np.ndarray:
    if not isinstance(sr, int) or not 8_000 <= sr <= 192_000:
        raise AudioError("Sample rate must be between 8000 and 192000 Hz")
    x = np.asarray(x)
    if x.ndim not in (1, 2) or (x.ndim == 2 and x.shape[1] not in (1, 2)):
        raise AudioError("Expected mono or two-channel audio, shape [samples, channels]")
    if x.shape[0] == 0:
        raise AudioError("Audio is empty")
    if not np.issubdtype(x.dtype, np.number) or not np.all(np.isfinite(x)):
        raise AudioError("Audio must contain only finite numeric samples")
    return x.astype(np.float32)


def mono(x: np.ndarray) -> np.ndarray:
    """Stereo policy: average channels; original channel count available in metadata."""
    return x.mean(axis=1, dtype=np.float32) if x.ndim == 2 else x


def resample(x: np.ndarray, source_sr: int, target_sr: int = TARGET_SR) -> np.ndarray:
    if source_sr == target_sr:
        return np.asarray(x, np.float32)
    if target_sr <= 0:
        raise AudioError("Target sample rate must be positive")
    ratio = Fraction(target_sr, source_sr)
    return resample_poly(x, ratio.numerator, ratio.denominator).astype(np.float32)


def decode(data: bytes, *, max_seconds: float = 120) -> tuple[np.ndarray, AudioInfo]:
    try:
        with sf.SoundFile(BytesIO(data)) as file:
            sr, frames, channels = file.samplerate, file.frames, file.channels
            if channels not in (1, 2) or sr < 8_000 or sr > 192_000:
                raise AudioError("Only mono/stereo audio at 8–192 kHz is supported")
            if frames <= 0 or frames / sr > max_seconds:
                raise AudioError(f"Audio must contain 0–{max_seconds:g} seconds")
            audio = file.read(dtype="float32", always_2d=False)
    except (RuntimeError, sf.LibsndfileError, ValueError) as exc:
        if isinstance(exc, AudioError):
            raise
        raise AudioError("Audio cannot be decoded") from exc
    audio = validate_audio(audio, sr)
    info = AudioInfo(sr, channels, frames, frames / sr, float(np.max(np.abs(audio))))
    return audio, info


def wav_bytes(x: np.ndarray, sr: int = TARGET_SR) -> bytes:
    x = validate_audio(x, sr)
    buf = BytesIO()
    # IEEE float WAV preserves dynamic range and avoids unintended PCM clipping.
    sf.write(buf, x, sr, format="WAV", subtype="FLOAT")
    return buf.getvalue()


def prepare(x: np.ndarray, sr: int) -> np.ndarray:
    return resample(mono(validate_audio(x, sr)), sr, TARGET_SR)
