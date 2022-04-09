"""Validated trusted local model checkpoints; never load arbitrary upload bytes."""
from __future__ import annotations

import os
import pickle
from pathlib import Path

import torch

from .model import ConvRecurrentMask, ModelConfig


class CheckpointError(RuntimeError):
    pass


def save_checkpoint(path: Path, model: ConvRecurrentMask, *, epoch: int,
                    optimizer: torch.optim.Optimizer | None = None,
                    scheduler: object | None = None, training: dict | None = None,
                    best_loss: float | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {"format_version": 1, "model": model.state_dict(),
            "metadata": model.metadata(), "epoch": epoch,
            "best_loss": best_loss, "training": training or {},
            "optimizer": optimizer.state_dict() if optimizer else None,
            "scheduler": scheduler.state_dict() if scheduler else None}
    temporary = path.with_suffix(path.suffix + ".partial")
    try:
        torch.save(data, temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def load_checkpoint(path: Path, device: str = "cpu") -> tuple[ConvRecurrentMask, dict]:
    if not path.is_file():
        raise CheckpointError("Trained checkpoint is not installed")
    try:
        # weights_only prevents arbitrary pickled application classes being executed.
        data = torch.load(path, map_location="cpu", weights_only=True)
        meta = data["metadata"]
        if data["format_version"] != 1 or meta["architecture"] != "ConvRecurrentMask-v1":
            raise ValueError("Unsupported checkpoint architecture or format")
        if meta["sample_rate"] != 16000 or meta["features"] != "log1p(abs(rfft(sqrt_hann_windowed_frame)))":
            raise ValueError("Incompatible preprocessing metadata")
        model = ConvRecurrentMask(ModelConfig(**meta["config"]))
        model.load_state_dict(data["model"], strict=True)
        model = model.to(device).eval()
        return model, data
    except (OSError, KeyError, TypeError, RuntimeError, ValueError, EOFError, IndexError, pickle.UnpicklingError) as exc:
        raise CheckpointError("Checkpoint missing, damaged or incompatible") from exc
