"""Resumable deterministic (subject to CUDA kernels) PyTorch training."""
from __future__ import annotations

import csv
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from .checkpoint import load_checkpoint, save_checkpoint
from .data import MixedSpeechDataset
from .model import ConvRecurrentMask, ModelConfig


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def predict_wave(model: ConvRecurrentMask, noisy: torch.Tensor) -> torch.Tensor:
    n_fft, hop = model.cfg.n_fft, model.cfg.hop
    win = torch.sqrt(torch.hann_window(n_fft, device=noisy.device))
    spec = torch.stft(noisy, n_fft=n_fft, hop_length=hop, window=win, return_complex=True)
    mag = spec.abs().transpose(1, 2)
    mask, _ = model(torch.log1p(mag))
    enhanced = spec * mask.transpose(1, 2)
    return torch.istft(enhanced, n_fft=n_fft, hop_length=hop,
                       window=win, length=noisy.shape[-1])


def loss_parts(estimate: torch.Tensor, clean: torch.Tensor,
               waveform_weight: float = 1, spectral_weight: float = .5,
               log_weight: float = .5) -> dict[str, torch.Tensor]:
    parts = {'waveform': nn.functional.l1_loss(estimate, clean)}
    mags, logs = [], []
    for n_fft in (256, 512, 1024):
        win = torch.hann_window(n_fft, device=estimate.device)
        e = torch.stft(estimate, n_fft, hop_length=n_fft // 4,
                       window=win, return_complex=True).abs()
        c = torch.stft(clean, n_fft, hop_length=n_fft // 4,
                       window=win, return_complex=True).abs()
        mags.append(nn.functional.l1_loss(e, c))
        logs.append(nn.functional.l1_loss(torch.log(e + 1e-5), torch.log(c + 1e-5)))
    parts['spectral'] = torch.stack(mags).mean()
    parts['log_spectral'] = torch.stack(logs).mean()
    parts['total'] = (waveform_weight * parts['waveform'] +
                      spectral_weight * parts['spectral'] + log_weight * parts['log_spectral'])
    return parts


def train(root: Path, manifest: Path, output: Path, *, epochs: int = 20,
          batch_size: int = 4, workers: int = 0, seed: int = 42,
          lr: float = 1e-3, patience: int = 7,
          device: str = 'cpu', resume: Path | None = None,
          cfg: ModelConfig | None = None,
          waveform_weight: float = 1, spectral_weight: float = .5,
          log_weight: float = .5, segment_seconds: float = 1.0,
          snr_min: float = -5.0, snr_max: float = 20.0) -> dict:
    if epochs < 1 or batch_size < 1 or workers < 0 or patience < 1 or lr <= 0:
        raise ValueError('Invalid training epochs, batch size, workers, patience or learning rate')
    seed_all(seed)
    output.mkdir(parents=True, exist_ok=True)
    train_ds = MixedSpeechDataset(root, manifest, 'train', segment_seconds, snr_min, snr_max)
    val_ds = MixedSpeechDataset(root, manifest, 'val', segment_seconds, snr_min, snr_max)
    model = ConvRecurrentMask(cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=.5, patience=2)
    start_epoch, best, bad = 0, float('inf'), 0
    settings = {'seed': seed, 'epochs': epochs, 'batch_size': batch_size, 'learning_rate': lr,
                'patience': patience, 'segment_seconds': segment_seconds,
                'waveform_weight': waveform_weight, 'spectral_weight': spectral_weight,
                'log_weight': log_weight, 'snr_min': snr_min, 'snr_max': snr_max,
                'model': model.metadata(), 'dataset': str(manifest)}
    if resume:
        restored, data = load_checkpoint(resume, device)
        if restored.cfg != model.cfg:
            raise ValueError('Resume checkpoint has incompatible architecture')
        model.load_state_dict(restored.state_dict())
        if data.get('optimizer') is None or data.get('scheduler') is None:
            raise ValueError('Checkpoint has no optimizer/scheduler state for resumption')
        optimizer.load_state_dict(data['optimizer'])
        scheduler.load_state_dict(data['scheduler'])
        start_epoch, best = int(data['epoch']) + 1, float(data['best_loss'] or float('inf'))
        if start_epoch >= epochs:
            raise ValueError('Resumed checkpoint has already reached requested epochs')
    (output / 'configuration.json').write_text(json.dumps(settings, indent=2))
    fieldnames = ['epoch', 'train_total', 'train_waveform', 'train_spectral',
                  'train_log_spectral', 'val_total', 'val_waveform', 'val_spectral',
                  'val_log_spectral', 'grad_norm_max', 'seconds', 'learning_rate']
    csv_path = output / 'history.csv'
    if not resume or not csv_path.exists():
        with csv_path.open('w', newline='') as f:
            csv.DictWriter(f, fieldnames).writeheader()
    mixed_precision = device.startswith('cuda') and torch.cuda.is_available()
    scaler = torch.amp.GradScaler('cuda', enabled=mixed_precision)
    total_start = time.perf_counter()
    for epoch in range(start_epoch, epochs):
        tick = time.perf_counter()
        train_ds.epoch = epoch
        train_gen = torch.Generator().manual_seed(seed + epoch)
        loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=workers,
                            generator=train_gen)
        validation = DataLoader(val_ds, batch_size=batch_size, num_workers=workers)
        results = {}
        grad_max = 0.0
        for mode, batches in (('train', loader), ('val', validation)):
            model.train(mode == 'train')
            counters = {k: 0.0 for k in ('total', 'waveform', 'spectral', 'log_spectral')}
            count = 0
            for noisy, clean, _snr in batches:
                noisy, clean = noisy.to(device), clean.to(device)
                optimizer.zero_grad(set_to_none=True)
                with torch.set_grad_enabled(mode == 'train'):
                    with torch.autocast('cuda', enabled=mixed_precision):
                        estimate = predict_wave(model, noisy)
                        loss = loss_parts(estimate, clean, waveform_weight,
                                          spectral_weight, log_weight)
                    if not all(torch.isfinite(v).all() for v in loss.values()):
                        raise FloatingPointError('Nonfinite training or validation loss')
                    if mode == 'train':
                        scaler.scale(loss['total']).backward()
                        scaler.unscale_(optimizer)
                        # With mixed precision the GradScaler routinely meets inf gradients while it
                        # calibrates its scale and skips those steps itself; only fail hard without it.
                        grad_norm = nn.utils.clip_grad_norm_(model.parameters(), 3.0,
                                                             error_if_nonfinite=not mixed_precision)
                        if torch.isfinite(grad_norm):
                            grad_max = max(grad_max, float(grad_norm))
                        scaler.step(optimizer)
                        scaler.update()
                for name in counters:
                    counters[name] += float(loss[name].detach()) * len(noisy)
                count += len(noisy)
            for name, accumulated in counters.items():
                results[f'{mode}_{name}'] = accumulated / count
        scheduler.step(results['val_total'])
        if results['val_total'] < best - 1e-5:
            best, bad = results['val_total'], 0
            save_checkpoint(output / 'best.pt', model, epoch=epoch, optimizer=optimizer,
                            scheduler=scheduler, training=settings, best_loss=best)
        else:
            bad += 1
        save_checkpoint(output / 'last.pt', model, epoch=epoch, optimizer=optimizer,
                        scheduler=scheduler, training=settings, best_loss=best)
        row = {'epoch': epoch, **results, 'grad_norm_max': grad_max,
               'seconds': time.perf_counter() - tick,
               'learning_rate': optimizer.param_groups[0]['lr']}
        with csv_path.open('a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames)
            writer.writerow(row)
        print(json.dumps(row), flush=True)
        if bad >= patience:
            break
    return {'best_validation_loss': best, 'trained_epochs': max(0, epoch - start_epoch + 1),
            'total_seconds': time.perf_counter() - total_start,
            'parameters': model.parameter_count(), 'checkpoint': str(output / 'best.pt')}
