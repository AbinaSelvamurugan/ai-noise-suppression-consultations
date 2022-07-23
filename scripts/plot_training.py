"""Plot CSV train/validation loss history."""
import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('--history', type=Path, required=True)
parser.add_argument('--output', type=Path, required=True)
parser.add_argument('--label', default='Training history')
a = parser.parse_args()
with a.history.open() as file:
    rows = list(csv.DictReader(file))
if not rows:
    parser.error('History contains no epochs')
fig, ax = plt.subplots(figsize=(8, 4))
for column, name in [('train_total', 'Train total'), ('val_total', 'Validation total')]:
    ax.plot([int(r['epoch']) for r in rows], [float(r[column]) for r in rows], '-o', label=name)
ax.set_title(a.label)
ax.set_xlabel('Epoch')
ax.set_ylabel('Loss')
ax.legend()
fig.tight_layout()
a.output.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(a.output)
plt.close(fig)
print('Saved training plot:', a.output)
