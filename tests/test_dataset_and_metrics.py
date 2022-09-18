from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from consult_audio.data import MixedSpeechDataset, build_manifest
from consult_audio.metrics import evaluate_pair, si_sdr


def _make_dataset(root: Path):
    rng = np.random.default_rng(42)
    for i in range(9):
        speaker = root / 'clean' / f'speaker_{i:02}'
        speaker.mkdir(parents=True, exist_ok=True)
        sf.write(speaker / 'x.wav', rng.normal(0, .02, 16000), 16000)
        noise = root / 'noise'
        noise.mkdir(parents=True, exist_ok=True)
        sf.write(noise / f'{i}.wav', rng.normal(0, .02, 16000), 16000)


def test_manifest_determinism_and_no_leaks(tmp_path):
    _make_dataset(tmp_path)
    p1, p2 = tmp_path / 'a.jsonl', tmp_path / 'b.jsonl'
    stats = build_manifest(tmp_path, p1)
    build_manifest(tmp_path, p2)
    assert p1.read_bytes() == p2.read_bytes() and stats['examples'] == 9
    partitions = [MixedSpeechDataset(tmp_path, p1, split) for split in ('train', 'val', 'test')]
    assert sum(len(p) for p in partitions) == 9
    for dataset in partitions:
        mixed, clean, snr = dataset[0]
        assert mixed.shape == clean.shape == (16000,) and -5 <= snr <= 20
        assert np.isfinite(mixed.numpy()).all()


def test_metrics_reference_required():
    s = np.sin(np.arange(16000) / 20).astype(np.float32) * .1
    n = s + .02 * np.random.default_rng(0).normal(size=len(s))
    m = evaluate_pair(s, n, s)
    assert m['si_sdr_improvement_db'] > 0
    assert m['pesq'] is None
    assert si_sdr(np.zeros_like(s), s) is None
    with pytest.raises(ValueError):
        evaluate_pair(s, n[:-1], s)
