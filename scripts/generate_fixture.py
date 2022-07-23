"""Small deterministic harmonic/non-speech fixture: SOFTWARE TESTS ONLY."""
from pathlib import Path
import numpy as np
import soundfile as sf

root = Path('data/synthetic')
rng = np.random.default_rng(123)
sr = 16000
for k in range(9):
    t = np.arange(sr * 2) / sr
    f = 130 + 17 * k
    # Synthetic modulated harmonic sound is NOT a recorded human voice.
    clean = (.08 * (1 + np.sin(2 * np.pi * 3.4 * t)) *
             (np.sin(2 * np.pi * f * t) + .4 * np.sin(2 * np.pi * 2 * f * t)))
    path = root / 'clean' / f'synthetic_speaker_{k:02d}'
    path.mkdir(parents=True, exist_ok=True)
    sf.write(path / 'tone.wav', clean, sr)
    noise_dir = root / 'noise'
    noise_dir.mkdir(parents=True, exist_ok=True)
    noise = rng.normal(0, .09, len(t))
    noise = np.cumsum(noise) if k % 2 else noise
    noise = noise / (max(np.max(np.abs(noise)), 1e-5)) * .1
    sf.write(noise_dir / f'noise_{k:02d}.wav', noise, sr)
print('Created SYNTHETIC, NON-SPEECH software-test dataset:', root)
