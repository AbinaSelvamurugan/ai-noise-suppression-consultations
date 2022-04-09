"""Stateful, window-normalized streaming overlap/add synthesis at 16 kHz."""
from __future__ import annotations

import numpy as np
import torch
from scipy.ndimage import gaussian_filter1d

from .model import ConvRecurrentMask


class StreamingEnhancer:
    """Hop-causal 512-point STFT; first 384 output samples compensate front padding.

    state persists across transport chunks; output length restored by finish().
    """

    n_fft = 512
    hop = 128
    sample_rate = 16000

    def __init__(
        self, *, method: str = "spectral", model: ConvRecurrentMask | None = None,
        device: str = "cpu", strength: float = 1.4, min_gain: float = 0.08,
    ) -> None:
        if method not in ("spectral", "neural"):
            raise ValueError("Unsupported stream method")
        if method == "neural" and model is None:
            raise ValueError("Neural stream requires loaded trained model")
        self.method, self.model, self.device = method, model, device
        self.strength, self.min_gain = strength, min_gain
        self.window = np.sqrt(np.hanning(self.n_fft + 1)[:-1]).astype(np.float32)
        self.in_buffer = np.zeros(self.n_fft - self.hop, dtype=np.float32)
        self.num = np.zeros(self.n_fft, dtype=np.float64)
        self.den = np.zeros(self.n_fft, dtype=np.float64)
        self.drop = self.n_fft - self.hop
        self.input_count = 0
        self.output_count = 0
        self.noise = None
        self.prev_gain = None
        self.frames = 0
        self.hidden = None
        if model is not None:
            self.model.eval()

    def _mask(self, spectrum: np.ndarray) -> np.ndarray:
        magnitude = np.abs(spectrum)
        if self.method == "neural":
            assert self.model is not None
            with torch.inference_mode():
                feat = torch.from_numpy(np.log1p(magnitude).astype(np.float32))
                feat = feat.to(self.device).reshape(1, 1, -1)
                mask, self.hidden = self.model(feat, self.hidden)
                return mask[0, 0].cpu().numpy()
        # Initial 0.35 s is an assumed noise-only calibration region.
        if self.noise is None:
            self.noise = magnitude.copy()
        if self.frames < 44:
            self.noise = 0.97 * self.noise + 0.03 * magnitude
        gain = np.clip(1 - self.strength * self.noise / (magnitude + 1e-8),
                       self.min_gain, 1)
        gain = gaussian_filter1d(gain, 1)
        if self.prev_gain is not None:
            gain = .65 * self.prev_gain + .35 * gain
        self.prev_gain = gain
        return gain

    def _one_frame(self, frame: np.ndarray) -> np.ndarray:
        spec = np.fft.rfft(frame * self.window)
        enhanced = np.fft.irfft(spec * self._mask(spec), n=self.n_fft)
        self.num += enhanced * self.window
        self.den += self.window.astype(np.float64) ** 2
        out = (self.num[:self.hop] / np.maximum(self.den[:self.hop], 1e-10)).astype(np.float32)
        self.num[:-self.hop] = self.num[self.hop:].copy()
        self.num[-self.hop:] = 0
        self.den[:-self.hop] = self.den[self.hop:].copy()
        self.den[-self.hop:] = 0
        self.frames += 1
        if self.drop:
            skipped = min(self.drop, len(out))
            self.drop -= skipped
            out = out[skipped:]
        self.output_count += len(out)
        return out

    def _push(self, samples: np.ndarray) -> np.ndarray:
        self.in_buffer = np.concatenate((self.in_buffer, samples.astype(np.float32)))
        outputs = []
        while len(self.in_buffer) >= self.n_fft:
            outputs.append(self._one_frame(self.in_buffer[:self.n_fft]))
            self.in_buffer = self.in_buffer[self.hop:]
        return np.concatenate(outputs) if outputs else np.zeros(0, np.float32)

    def push(self, samples: np.ndarray) -> np.ndarray:
        x = np.asarray(samples, dtype=np.float32)
        if x.ndim != 1 or not np.all(np.isfinite(x)) or len(x) > 16000:
            raise ValueError("Stream expects up to 16000 finite mono float32 samples")
        self.input_count += len(x)
        return self._push(x)

    def finish(self) -> np.ndarray:
        """Flush filter tails; should be called once at the end of an offline signal."""
        needed = self.input_count - self.output_count
        if needed <= 0:
            return np.zeros(0, np.float32)
        end = self._push(np.zeros(needed + self.n_fft, dtype=np.float32))
        return end[:needed].copy()


def neural_enhance(x: np.ndarray, model: ConvRecurrentMask, device: str = "cpu") -> np.ndarray:
    stream = StreamingEnhancer(method="neural", model=model, device=device)
    results = [stream.push(x[i:i + 4096]) for i in range(0, len(x), 4096)]
    results.append(stream.finish())
    return np.concatenate(results)[:len(x)].astype(np.float32)
