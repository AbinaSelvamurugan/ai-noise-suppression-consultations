from pathlib import Path

import numpy as np
import pytest
import torch

from consult_audio.checkpoint import CheckpointError, load_checkpoint, save_checkpoint
from consult_audio.model import ConvRecurrentMask, ModelConfig
from consult_audio.stream import StreamingEnhancer, neural_enhance
from consult_audio.training import loss_parts, predict_wave


def test_shapes_gradient_and_loss():
    model = ConvRecurrentMask(ModelConfig(conv_channels=4, hidden=16))
    x = torch.randn(2, 1600) * .02
    y = predict_wave(model, x)
    loss = loss_parts(y, x)
    assert y.shape == x.shape and all(torch.isfinite(v) for v in loss.values())
    loss['total'].backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())
    mask, state = model(torch.ones(2, 3, 257))
    assert mask.shape == (2, 3, 257) and state.shape == (1, 2, 16)
    assert model.parameter_count() > 0


def test_checkpoint_round_trip(tmp_path: Path):
    model = ConvRecurrentMask(ModelConfig(hidden=16, conv_channels=4))
    p = tmp_path / 'best.pt'
    save_checkpoint(p, model, epoch=1, best_loss=.4)
    restored, data = load_checkpoint(p)
    assert data['epoch'] == 1 and restored.training is False
    for a, b in zip(model.parameters(), restored.parameters()):
        torch.testing.assert_close(a, b)
    p.write_bytes(b'broken')
    with pytest.raises(CheckpointError):
        load_checkpoint(p)
    with pytest.raises(CheckpointError):
        load_checkpoint(tmp_path / 'missing.pt')


def test_stateful_frames_and_neural_output():
    torch.manual_seed(5)
    model = ConvRecurrentMask(ModelConfig(hidden=16, conv_channels=4))
    inputs = torch.randn(1, 10, 257)
    whole, _ = model(inputs)
    state = None
    pieces = []
    for i in range(10):
        frame, state = model(inputs[:, i:i+1], state)
        pieces.append(frame)
    torch.testing.assert_close(torch.cat(pieces, 1), whole, atol=1e-6, rtol=1e-5)
    x = np.random.default_rng(0).normal(0, .01, 3200).astype(np.float32)
    single = neural_enhance(x, model)
    stream = StreamingEnhancer(method='neural', model=model)
    chunked = np.concatenate([stream.push(x[:333]), stream.push(x[333:]), stream.finish()])
    np.testing.assert_allclose(single, chunked, rtol=1e-5, atol=1e-6)
    assert single.shape == x.shape


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA hardware unavailable')
def test_cuda():
    model = ConvRecurrentMask().cuda()
    y, _ = model(torch.ones(1, 3, 257, device='cuda'))
    assert y.is_cuda
