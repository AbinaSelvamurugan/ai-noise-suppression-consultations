import argparse
import json
from pathlib import Path
import torch
import yaml
from consult_audio.benchmark import benchmark

p = argparse.ArgumentParser()
p.add_argument('--config', type=Path, default=Path('configs/evaluation.yaml'))
p.add_argument('--root', type=Path, default=Path('data/synthetic'))
p.add_argument('--manifest', type=Path, default=Path('artifacts/manifest.jsonl'))
p.add_argument('--output', type=Path, default=Path('artifacts/benchmark'))
p.add_argument('--checkpoint', type=Path)
p.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
p.add_argument('--max-examples', type=int)
p.add_argument('--snrs', type=float, nargs='+')
a = p.parse_args()
settings = yaml.safe_load(a.config.read_text())
if a.device == 'cuda' and not torch.cuda.is_available():
    p.error('CUDA unavailable')
print(json.dumps(benchmark(a.root, a.manifest, a.output, a.checkpoint,
                           a.max_examples or settings['max_examples'], a.device,
                           tuple(a.snrs or settings['snrs_db'])), indent=2))
