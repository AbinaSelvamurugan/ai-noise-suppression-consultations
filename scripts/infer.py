"""Command-line audio enhancement; requires an explicit checkpoint for neural inference."""
import argparse
from pathlib import Path
import time

import soundfile as sf

from consult_audio.audio import TARGET_SR, prepare, wav_bytes
from consult_audio.checkpoint import load_checkpoint
from consult_audio.spectral import GateConfig, spectral_gate
from consult_audio.stream import neural_enhance

p = argparse.ArgumentParser(description='Enhance a soundfile to mono 16 kHz float WAV')
p.add_argument('--input', type=Path, required=True)
p.add_argument('--output', type=Path, required=True)
p.add_argument('--method', choices=['spectral', 'neural'], default='spectral')
p.add_argument('--strength', type=float, default=1.4)
p.add_argument('--noise-start', type=float, default=0)
p.add_argument('--noise-seconds', type=float, default=.35)
p.add_argument('--checkpoint', type=Path)
p.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
a = p.parse_args()
if not a.input.is_file() or a.output.resolve() == a.input.resolve():
    p.error('Input must exist and output must not overwrite input')
x, sr = sf.read(a.input, dtype='float32')
x = prepare(x, sr)
start = time.perf_counter()
if a.method == 'spectral':
    y = spectral_gate(x, TARGET_SR, GateConfig(strength=a.strength,
                  noise_start_seconds=a.noise_start, noise_seconds=a.noise_seconds))
else:
    if a.checkpoint is None:
        p.error('Neural inference requires an explicitly trained compatible --checkpoint')
    model, _ = load_checkpoint(a.checkpoint, a.device)
    y = neural_enhance(x, model, a.device)
seconds = time.perf_counter() - start
a.output.parent.mkdir(parents=True, exist_ok=True)
a.output.write_bytes(wav_bytes(y))
print(f'Saved {len(y)/TARGET_SR:.3f}s enhanced WAV: {a.output}; inference {seconds*1000:.2f} ms; RTF {seconds/(len(y)/TARGET_SR):.4f}')
