"""Reference-based objective speech enhancement metrics; never infer clean truth."""
from __future__ import annotations

import numpy as np
from scipy.signal import stft


def si_sdr(reference: np.ndarray, estimate: np.ndarray) -> float | None:
    reference = np.asarray(reference, dtype=np.float64)
    estimate = np.asarray(estimate, dtype=np.float64)
    if len(reference) != len(estimate) or len(reference) < 4:
        raise ValueError("Reference and estimate must have identical lengths")
    reference = reference - reference.mean()
    estimate = estimate - estimate.mean()
    power = np.dot(reference, reference)
    if power < 1e-10:
        return None
    target = np.dot(estimate, reference) / power * reference
    error = estimate - target
    return float(10 * np.log10((np.dot(target, target) + 1e-12) /
                               (np.dot(error, error) + 1e-12)))


def segmental_snr(reference: np.ndarray, estimate: np.ndarray, sr: int = 16000) -> float | None:
    frame = int(sr * .02)
    values = []
    for i in range(0, len(reference) - frame + 1, frame):
        s, e = reference[i:i + frame], estimate[i:i + frame]
        signal_power = np.mean(s.astype(np.float64) ** 2)
        if signal_power < 1e-8:
            continue
        error_power = np.mean((s.astype(np.float64) - e) ** 2)
        values.append(np.clip(10 * np.log10(signal_power / max(error_power, 1e-12)), -10, 35))
    return float(np.mean(values)) if values else None


def log_spectral_distance(reference: np.ndarray, estimate: np.ndarray, sr: int = 16000) -> float:
    def spectra(x: np.ndarray) -> np.ndarray:
        _, _, z = stft(x, fs=sr, nperseg=512, noverlap=256, boundary="zeros")
        return 20 * np.log10(np.maximum(abs(z), 1e-7))
    a, b = spectra(reference), spectra(estimate)
    return float(np.mean(np.sqrt(np.mean((a - b) ** 2, axis=0))))


def evaluate_pair(reference: np.ndarray, noisy: np.ndarray,
                  enhanced: np.ndarray, sr: int = 16000) -> dict:
    if not (len(reference) == len(noisy) == len(enhanced)):
        raise ValueError("Clean, noisy and enhanced signals must be sample-aligned")
    before, after = si_sdr(reference, noisy), si_sdr(reference, enhanced)
    result = {
        "si_sdr_noisy_db": before, "si_sdr_enhanced_db": after,
        "si_sdr_improvement_db": after - before if before is not None and after is not None else None,
        "segmental_snr_noisy_db": segmental_snr(reference, noisy, sr),
        "segmental_snr_enhanced_db": segmental_snr(reference, enhanced, sr),
        "lsd_noisy_db": log_spectral_distance(reference, noisy, sr),
        "lsd_enhanced_db": log_spectral_distance(reference, enhanced, sr),
        "stoi_noisy": None, "stoi_enhanced": None, "pesq": None,
    }
    try:
        from pystoi import stoi
        if sr in (10000, 16000) and len(reference) >= int(.4 * sr):
            result["stoi_noisy"] = float(stoi(reference, noisy, sr, extended=False))
            result["stoi_enhanced"] = float(stoi(reference, enhanced, sr, extended=False))
    except (ImportError, ValueError, RuntimeError):
        pass
    # PESQ deliberately unavailable: depends on separate compatible, licensed evaluator.
    return result
