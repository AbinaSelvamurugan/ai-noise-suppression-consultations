"""Condition-controlled test-set evaluation, raw values and condition summaries."""
from __future__ import annotations

import csv
import json
import time
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from .audio import TARGET_SR
from .checkpoint import load_checkpoint
from .data import MixedSpeechDataset
from .metrics import evaluate_pair
from .spectral import spectral_gate
from .stream import neural_enhance


def benchmark(root: Path, manifest: Path, output: Path, checkpoint: Path | None = None,
              max_examples: int = 100, device: str = 'cpu',
              snrs: tuple[float, ...] = (-5, 0, 5, 10, 20)) -> dict:
    """Each method sees identical mixtures; noisy SNRs are explicitly controlled."""
    if not snrs or max_examples < 1:
        raise ValueError('Need one or more input SNRs and positive max_examples')
    output.mkdir(parents=True, exist_ok=True)
    model = load_checkpoint(checkpoint, device)[0] if checkpoint else None
    records: list[dict] = []
    for snr in snrs:
        dataset = MixedSpeechDataset(root, manifest, 'test', snr_min=snr, snr_max=snr)
        for idx in range(min(len(dataset), max_examples)):
            noisy_t, clean_t, _snr = dataset[idx]
            noisy, clean = noisy_t.numpy(), clean_t.numpy()
            for method in ('spectral', 'neural'):
                if method == 'neural' and model is None:
                    continue
                started = time.perf_counter()
                output_audio = (spectral_gate(noisy, TARGET_SR) if method == 'spectral'
                                else neural_enhance(noisy, model, device))
                latency = time.perf_counter() - started
                values = evaluate_pair(clean, noisy, output_audio)
                records.append({'example': idx, 'method': method, 'snr_db': snr,
                                'duration_s': len(noisy) / TARGET_SR,
                                'latency_ms': round(1000 * latency, 4),
                                'rtf': latency / (len(noisy) / TARGET_SR), **values})
    metric_fields = ('si_sdr_improvement_db', 'latency_ms', 'rtf')
    summary: list[dict] = []
    for method in ('spectral', 'neural'):
        for snr in snrs:
            subset = [r for r in records if r['method'] == method and r['snr_db'] == snr]
            if not subset:
                continue
            row = {'method': method, 'snr_db': snr, 'count': len(subset)}
            for key in metric_fields:
                values = [float(r[key]) for r in subset if r[key] is not None]
                row[f'{key}_mean'] = float(np.mean(values)) if values else None
                row[f'{key}_std'] = float(np.std(values, ddof=1)) if len(values) > 1 else None
                row[f'{key}_ci95_halfwidth'] = (
                    float(1.96 * np.std(values, ddof=1) / np.sqrt(len(values)))
                    if len(values) >= 5 else None)
            summary.append(row)
    report = {'sample_rate': TARGET_SR, 'device': device, 'snrs_db': snrs,
              'checkpoint': checkpoint.name if checkpoint else None,
              'results': records, 'summary': summary,
              'uncertainty': 'Approximate 95% normal CI only when N>=5; not a bootstrap'}
    (output / 'results.json').write_text(json.dumps(report, indent=2))
    with (output / 'results.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    with (output / 'summary.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    for field in metric_fields:
        fig, ax = plt.subplots(figsize=(8, 4))
        for method in ('spectral', 'neural'):
            subset = [r for r in summary if r['method'] == method]
            xs = [r['snr_db'] for r in subset if r[f'{field}_mean'] is not None]
            ys = [r[f'{field}_mean'] for r in subset if r[f'{field}_mean'] is not None]
            if xs:
                ax.plot(xs, ys, '-o', label=method)
        ax.set_xlabel('Controlled input SNR (dB)')
        ax.set_ylabel(field)
        ax.legend()
        fig.tight_layout()
        fig.savefig(output / f'{field}.png')
        plt.close(fig)
    return {'examples_per_condition': min(len(dataset), max_examples),
            'snr_conditions': list(snrs), 'device': device,
            'neural_available': model is not None, 'rows': len(records)}
