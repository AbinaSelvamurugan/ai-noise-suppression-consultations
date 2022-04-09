"""Spectral gating with specified initial or user-selected noise-only calibration."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.signal import istft, stft

from .audio import AudioError, validate_audio, mono


@dataclass(frozen=True)
class GateConfig:
    n_fft: int = 512
    hop: int = 128
    strength: float = 1.4
    noise_seconds: float = 0.35
    noise_start_seconds: float = 0.0
    minimum_gain: float = 0.08
    smooth_time: float = 1.0
    smooth_frequency: float = 1.0

    def check(self) -> None:
        if self.n_fft not in (256, 512, 1024, 2048):
            raise AudioError("n_fft must be 256, 512, 1024, or 2048")
        if not 1 <= self.hop <= self.n_fft // 2:
            raise AudioError("hop must be between 1 and n_fft/2")
        if not 0 <= self.strength <= 4 or not 0 <= self.minimum_gain <= 1:
            raise AudioError("strength or minimum_gain out of range")
        if self.noise_seconds <= 0 or self.noise_start_seconds < 0:
            raise AudioError("Noise calibration segment must be positive")
        if not 0 <= self.smooth_time <= 8 or not 0 <= self.smooth_frequency <= 8:
            raise AudioError("Smoothing must be in [0, 8]")


def spectral_gate(x: np.ndarray, sr: int, cfg: GateConfig | None = None) -> np.ndarray:
    cfg = cfg or GateConfig()
    cfg.check()
    x = mono(validate_audio(x, sr))
    if len(x) < 2 or float(np.max(np.abs(x))) < 1e-10:
        return x.copy()
    noverlap = cfg.n_fft - cfg.hop
    work = np.pad(x, (0, max(0, cfg.n_fft - len(x))))
    _, times, z = stft(
        work, fs=sr, window="hann", nperseg=cfg.n_fft, noverlap=noverlap,
        nfft=cfg.n_fft, boundary="zeros", padded=True,
    )
    start = cfg.noise_start_seconds
    end = start + cfg.noise_seconds
    selection = (times >= start) & (times < end)
    if not np.any(selection):
        raise AudioError("Noise calibration segment falls outside the recording")
    reference = np.median(np.abs(z[:, selection]), axis=1, keepdims=True)
    power = np.abs(z)
    # Conservative subtraction; gain smoothing suppresses isolated musical-noise bins.
    gain = np.maximum(cfg.minimum_gain, 1.0 - cfg.strength * reference / (power + 1e-9))
    gain = gaussian_filter(gain, sigma=(cfg.smooth_frequency, cfg.smooth_time))
    gain = np.clip(gain, cfg.minimum_gain, 1.0)
    _, output = istft(
        z * gain, fs=sr, window="hann", nperseg=cfg.n_fft,
        noverlap=noverlap, nfft=cfg.n_fft, input_onesided=True, boundary=True,
    )
    return output[:len(x)].astype(np.float32)
