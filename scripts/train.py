import argparse
import json
from pathlib import Path
import yaml
from consult_audio.model import ModelConfig
from consult_audio.training import train

p = argparse.ArgumentParser()
p.add_argument('--root', type=Path, default=Path('data/synthetic'))
p.add_argument('--manifest', type=Path, default=Path('artifacts/manifest.jsonl'))
p.add_argument('--output', type=Path, default=Path('artifacts/checkpoints'))
p.add_argument('--config', type=Path, default=Path('configs/training.yaml'))
p.add_argument('--model-config', type=Path, default=Path('configs/model.yaml'))
p.add_argument('--epochs', type=int)
p.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
p.add_argument('--resume', type=Path)
a = p.parse_args()
settings = yaml.safe_load(a.config.read_text())
if a.epochs is not None:
    settings['epochs'] = a.epochs
if a.device == 'cuda':
    import torch
    if not torch.cuda.is_available():
        p.error('CUDA requested but unavailable')
print(json.dumps(train(a.root, a.manifest, a.output, device=a.device, resume=a.resume,
                       cfg=ModelConfig(**yaml.safe_load(a.model_config.read_text())),
                       lr=settings.pop('learning_rate'), **settings), indent=2))
