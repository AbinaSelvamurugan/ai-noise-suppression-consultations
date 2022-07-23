"""Export one deterministic paired example for listening; human voice depends on your corpus."""
import argparse
from pathlib import Path

from consult_audio.audio import wav_bytes
from consult_audio.data import MixedSpeechDataset
from consult_audio.spectral import spectral_gate

p = argparse.ArgumentParser()
p.add_argument('--root', type=Path, required=True)
p.add_argument('--manifest', type=Path, required=True)
p.add_argument('--output', type=Path, default=Path('artifacts/listening-example'))
p.add_argument('--split', choices=['train', 'val', 'test'], default='test')
p.add_argument('--index', type=int, default=0)
p.add_argument('--snr-db', type=float, default=0)
a = p.parse_args()
dataset = MixedSpeechDataset(a.root, a.manifest, a.split,
                             snr_min=a.snr_db, snr_max=a.snr_db)
if not 0 <= a.index < len(dataset):
    p.error('Example index outside manifest split')
noisy, clean, _ = dataset[a.index]
y = spectral_gate(noisy.numpy(), 16000)
a.output.mkdir(parents=True, exist_ok=True)
for name, samples in [('clean', clean.numpy()), ('noisy', noisy.numpy()),
                      ('spectral', y)]:
    (a.output / f'{name}.wav').write_bytes(wav_bytes(samples))
print(f'Exported aligned original/noisy/spectral WAVs to {a.output}')
