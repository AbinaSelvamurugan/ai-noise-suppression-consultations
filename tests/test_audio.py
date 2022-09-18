import numpy as np
import pytest

from consult_audio.audio import AudioError, decode, prepare, wav_bytes
from consult_audio.data import mix_at_snr
from consult_audio.spectral import GateConfig, spectral_gate
from consult_audio.stream import StreamingEnhancer


def sine(sr=16000, secs=1.0):
    return (0.12 * np.sin(2 * np.pi * 220 * np.arange(int(sr * secs)) / sr)).astype(np.float32)


@pytest.mark.parametrize('sr', [8000, 16000, 44100, 48000])
def test_spectral_input_rates_and_shapes(sr):
    x = sine(sr, .2)
    y = spectral_gate(x, sr)
    assert y.shape == x.shape and np.isfinite(y).all()
    stereo = np.stack([x, x / 2], axis=1)
    assert spectral_gate(stereo, sr).shape == x.shape
    assert len(prepare(stereo, sr)) == round(len(x) * 16000 / sr)


@pytest.mark.parametrize('length', [1, 5, 127, 255, 511, 512, 1025])
def test_short_and_silent(length):
    x = np.zeros(length, np.float32)
    assert np.array_equal(spectral_gate(x, 16000), x)
    x[0] = .8
    output = spectral_gate(x, 16000)
    assert output.shape == x.shape and np.isfinite(output).all()


def test_colored_white_clipped_and_gating_config():
    rng = np.random.default_rng(0)
    x = rng.normal(0, .1, 16000).astype(np.float32)
    for s in (x, np.cumsum(x).astype(np.float32) / 100, np.full(16000, 2., np.float32)):
        y = spectral_gate(s, 16000, GateConfig(n_fft=1024, hop=256, minimum_gain=.2))
        assert np.isfinite(y).all() and len(y) == len(s)
    with pytest.raises(AudioError):
        spectral_gate(x, 16000, GateConfig(hop=999))
    with pytest.raises(AudioError):
        spectral_gate(x, 16000, GateConfig(noise_start_seconds=5))


def test_roundtrip_and_corrupt_audio():
    data = wav_bytes(sine())
    decoded, info = decode(data)
    assert info.sample_rate == 16000 and info.channels == 1
    np.testing.assert_allclose(decoded, sine(), atol=2e-6)
    with pytest.raises(AudioError):
        decode(b'not a recording')
    with pytest.raises(AudioError):
        decode(data, max_seconds=.1)
    with pytest.raises(AudioError):
        spectral_gate(np.array([np.nan]), 16000)
    with pytest.raises(AudioError):
        spectral_gate(np.ones((10, 3)), 16000)


def test_snr_and_silence():
    s = sine()
    n = np.random.default_rng(1).normal(0, .1, len(s)).astype(np.float32)
    x, scaled = mix_at_snr(s, n, 5)
    snr = 10 * np.log10(np.mean(s ** 2) / np.mean(scaled ** 2))
    assert snr == pytest.approx(5, abs=.02)
    assert x.shape == s.shape
    _, quiet = mix_at_snr(np.zeros_like(s), n, 5)
    assert np.all(quiet == 0)


def test_spectral_streaming_chunks_and_length():
    x = sine(secs=.25)
    a, b = StreamingEnhancer(), StreamingEnhancer()
    fine = [a.push(x[i:i + 128]) for i in range(0, len(x), 128)] + [a.finish()]
    coarse = [b.push(x[i:i + 1000]) for i in range(0, len(x), 1000)] + [b.finish()]
    y, z = np.concatenate(fine), np.concatenate(coarse)
    assert len(y) == len(z) == len(x)
    np.testing.assert_allclose(y, z, atol=1e-5)
    assert np.isfinite(y).all()
