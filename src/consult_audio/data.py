"""Reproducible, speaker-separated manifests and controlled-SNR mixtures."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from torch.utils.data import Dataset

from .audio import prepare, validate_audio


def mix_at_snr(speech: np.ndarray, noise: np.ndarray, snr_db: float) -> tuple[np.ndarray, np.ndarray]:
    """Return mixture and scaled noise, or unchanged silence if either source is silent."""
    s = np.asarray(speech, np.float32)
    n = np.asarray(noise, np.float32)
    if len(s) != len(n):
        raise ValueError("Mixing operands must have identical lengths")
    sp, npower = np.mean(s.astype(np.float64) ** 2), np.mean(n.astype(np.float64) ** 2)
    if sp <= 1e-10 or npower <= 1e-10:
        return s.copy(), np.zeros_like(n)
    alpha = np.sqrt(sp / (npower * 10 ** (snr_db / 10)))
    scaled = (n * alpha).astype(np.float32)
    return (s + scaled).astype(np.float32), scaled


def load_mono(path: Path) -> tuple[np.ndarray, int]:
    x, sr = sf.read(path, dtype="float32")
    return prepare(validate_audio(x, sr), sr), 16000


def _files(root: Path) -> list[Path]:
    return sorted(p for p in root.rglob('*') if p.suffix.lower() in ('.wav', '.flac', '.ogg'))


def build_manifest(data_root: Path, output: Path, seed: int = 42) -> dict:
    """Expected layout: root/clean/speaker_id/*.wav and root/noise/*.wav.

    Noise is disjoint across splits. Entries store relative paths and sha256 digests.
    """
    clean_root, noise_root = data_root / 'clean', data_root / 'noise'
    if not clean_root.is_dir() or not noise_root.is_dir():
        raise ValueError('Expected data/clean/<speaker>/*.wav and data/noise/*.wav')
    speakers = sorted(p for p in clean_root.iterdir() if p.is_dir())
    sounds = _files(noise_root)
    if len(speakers) < 3 or len(sounds) < 3:
        raise ValueError('Need at least three speaker directories and three noise files')
    for item in (*_files(clean_root), *sounds):
        try:
            with sf.SoundFile(item) as f:
                if f.frames <= 0 or f.channels not in (1, 2) or f.samplerate < 8000:
                    raise ValueError('invalid audio properties')
        except (RuntimeError, OSError, ValueError) as exc:
            raise ValueError(f'Unreadable audio file: {item.relative_to(data_root)}') from exc
    rng = np.random.default_rng(seed)
    rng.shuffle(speakers)
    rng.shuffle(sounds)

    def partition(seq: list) -> dict[str, list]:
        n = len(seq)
        tr = max(1, int(.7 * n))
        va = max(tr + 1, int(.85 * n))
        return {'train': seq[:tr], 'val': seq[tr:va], 'test': seq[va:]}

    speaker_partitions, noise_partitions = partition(speakers), partition(sounds)
    rows = []
    for split in ('train', 'val', 'test'):
        noises = noise_partitions[split]
        for speaker in speaker_partitions[split]:
            for speech in _files(speaker):
                with sf.SoundFile(speech) as f:
                    duration = f.frames / f.samplerate
                chosen = noises[len(rows) % len(noises)]
                rows.append({'split': split, 'speaker': speaker.name,
                             'clean': str(speech.relative_to(data_root)),
                             'noise': str(chosen.relative_to(data_root)),
                             'duration_s': round(duration, 4),
                             'seed': int(rng.integers(0, 2 ** 31)),
                             'clean_sha256': hashlib.sha256(speech.read_bytes()).hexdigest(),
                             'noise_sha256': hashlib.sha256(chosen.read_bytes()).hexdigest()})
    if not all(any(r['split'] == split for r in rows) for split in ('train', 'val', 'test')):
        raise ValueError('Each partition needs at least one clean speech file')
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text('\n'.join(json.dumps(row) for row in rows) + '\n')
    return {'examples': len(rows), 'splits': {s: sum(r['split'] == s for r in rows)
                                            for s in ('train', 'val', 'test')},
            'total_duration_s': sum(r['duration_s'] for r in rows)}


class MixedSpeechDataset(Dataset):
    def __init__(self, root: Path, manifest: Path, split: str = 'train',
                 duration_s: float = 1.0, snr_min: float = -5, snr_max: float = 20,
                 epoch: int = 0):
        if split not in ('train', 'val', 'test') or snr_min > snr_max:
            raise ValueError('Invalid split or SNR range')
        self.root, self.split, self.epoch = root.resolve(), split, epoch
        self.frames = int(duration_s * 16000)
        if self.frames < 512 or self.frames > 16000 * 30:
            raise ValueError('Segment duration must be in [.032, 30] seconds')
        self.snr_min, self.snr_max = snr_min, snr_max
        rows = [json.loads(line) for line in manifest.read_text().splitlines()]
        self.rows = [row for row in rows if row['split'] == split]
        if not self.rows:
            raise ValueError(f'No examples in {split} partition')
        unique_speakers = {s: set() for s in ('train', 'val', 'test')}
        unique_clean = {s: set() for s in unique_speakers}
        unique_noise = {s: set() for s in unique_speakers}
        for row in rows:
            for key in ('clean', 'noise'):
                resolved = (self.root / row[key]).resolve()
                if not resolved.is_relative_to(self.root):
                    raise ValueError('Manifest path escapes dataset root')
                if not resolved.is_file():
                    raise ValueError(f'Missing dataset file: {row[key]}')
            unique_speakers[row['split']].add(row['speaker'])
            unique_clean[row['split']].add(row['clean_sha256'])
            unique_noise[row['split']].add(row['noise_sha256'])
        for i, a in enumerate(('train', 'val', 'test')):
            for b in ('train', 'val', 'test')[i + 1:]:
                if (unique_speakers[a] & unique_speakers[b] or
                        unique_clean[a] & unique_clean[b] or unique_noise[a] & unique_noise[b]):
                    raise ValueError('Manifest leaks speakers or clean/noise files across splits')

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, float]:
        row = self.rows[index]
        rng = np.random.default_rng(row['seed'] + self.epoch * 1000003)
        s, _ = load_mono(self.root / row['clean'])
        n, _ = load_mono(self.root / row['noise'])
        if len(s) < self.frames:
            s = np.pad(s, (0, self.frames - len(s)))
        start = int(rng.integers(0, max(1, len(s) - self.frames + 1)))
        s = s[start:start + self.frames]
        if not len(n):
            raise ValueError('Noise file is empty')
        if len(n) < self.frames:
            n = np.tile(n, int(np.ceil(self.frames / len(n))))
        nstart = int(rng.integers(0, max(1, len(n) - self.frames + 1)))
        snr = float(rng.uniform(self.snr_min, self.snr_max))
        mixed, _ = mix_at_snr(s, n[nstart:nstart + self.frames], snr)
        return torch.from_numpy(mixed), torch.from_numpy(s.copy()), snr
