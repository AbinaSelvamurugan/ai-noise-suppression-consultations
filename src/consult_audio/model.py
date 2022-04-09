"""Compact convolutional-recurrent causal frequency-mask estimator."""
from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class ModelConfig:
    n_fft: int = 512
    hop: int = 128
    conv_channels: int = 8
    hidden: int = 96
    layers: int = 1

    def validate(self) -> None:
        if self.n_fft != 512 or self.hop != 128:
            raise ValueError("Streaming deployment currently supports n_fft=512, hop=128")
        if not 2 <= self.conv_channels <= 64 or not 16 <= self.hidden <= 512:
            raise ValueError("Unsupported model width")
        if not 1 <= self.layers <= 3:
            raise ValueError("layers must be between 1 and 3")


class ConvRecurrentMask(nn.Module):
    """Input [B,T,257] log1p magnitude; output same mask in [0,1].

    Conv layers operate on *frequency within each frame*, never future time.
    Stateful GRU permits framewise causal inference with a preserved hidden state.
    """

    def __init__(self, cfg: ModelConfig | None = None):
        super().__init__()
        self.cfg = cfg or ModelConfig()
        self.cfg.validate()
        c = self.cfg.conv_channels
        self.encoder = nn.Sequential(
            nn.Conv1d(1, c, 5, stride=2, padding=2), nn.ReLU(),
            nn.Conv1d(c, c, 5, stride=2, padding=2), nn.ReLU(),
        )
        with torch.no_grad():
            encoded_size = self.encoder(torch.zeros(1, 1, self.cfg.n_fft // 2 + 1)).numel()
        self.project = nn.Linear(encoded_size, self.cfg.hidden)
        self.gru = nn.GRU(
            self.cfg.hidden, self.cfg.hidden, num_layers=self.cfg.layers, batch_first=True,
        )
        self.head = nn.Linear(self.cfg.hidden, self.cfg.n_fft // 2 + 1)

    def forward(
        self, features: torch.Tensor, state: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if features.ndim != 3 or features.shape[-1] != self.cfg.n_fft // 2 + 1:
            raise ValueError("Expected log1p-magnitude features [batch, time, 257]")
        batch, frames, frequencies = features.shape
        encoded = self.encoder(features.reshape(batch * frames, 1, frequencies))
        projected = self.project(encoded.reshape(batch, frames, -1))
        recurrent, new_state = self.gru(projected, state)
        return torch.sigmoid(self.head(recurrent)), new_state

    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def metadata(self) -> dict:
        return {"architecture": "ConvRecurrentMask-v1", "config": asdict(self.cfg),
                "parameters": self.parameter_count(), "sample_rate": 16000,
                "features": "log1p(abs(rfft(sqrt_hann_windowed_frame)))",
                "mask": "sigmoid gain [0,1]", "channel_policy": "stereo averaged to mono"}
