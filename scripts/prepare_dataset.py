import argparse
import json
from pathlib import Path
from consult_audio.data import build_manifest

p = argparse.ArgumentParser()
p.add_argument('--root', type=Path, default=Path('data'))
p.add_argument('--output', type=Path, default=Path('artifacts/manifest.jsonl'))
p.add_argument('--seed', type=int, default=42)
a = p.parse_args()
print(json.dumps(build_manifest(a.root, a.output, a.seed), indent=2))
