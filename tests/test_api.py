import json
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest
from fastapi.testclient import TestClient

from backend.app.main import app
from consult_audio.audio import decode, wav_bytes


def fixture():
    x = (np.sin(np.arange(3200) / 21) * .08).astype(np.float32)
    return wav_bytes(x)


@pytest.fixture
def client():
    with TestClient(app) as test:
        yield test


def test_health_and_models(client):
    health = client.get('/api/health')
    assert health.status_code == 200 and health.json()['spectral_available']
    info = client.get('/api/models').json()
    assert 'models' in info and '.wav' in info['supported_formats']
    status = client.get('/api/training/status').json()
    assert status['worker_enabled'] is False


def test_real_audio_processing_and_analysis(client):
    response = client.post('/api/enhance', files={'file': ('speech.wav', fixture(), 'audio/wav')})
    assert response.status_code == 200, response.text
    output, info = decode(response.content)
    assert len(output) == 3200 and info.sample_rate == 16000
    assert np.isfinite(output).all()
    metadata = json.loads(response.headers['X-Audio-Metadata'])
    assert metadata['method'] == 'spectral' and metadata['processing_ms'] >= 0
    r = client.post('/api/analyze', files={'file': ('speech.wav', fixture(), 'audio/wav')})
    assert r.status_code == 200 and len(r.json()['spectrogram_db']) <= 129
    compare = client.post('/api/compare', files={'file': ('speech.wav', fixture(), 'audio/wav')})
    assert compare.status_code == 200 and compare.content[:2] == b'PK'


def test_evaluation_and_missing_model(client):
    payload = fixture()
    resp = client.post('/api/evaluate', files={'noisy': ('n.wav', payload, 'audio/wav'),
                                               'reference': ('c.wav', payload, 'audio/wav')})
    assert resp.status_code == 200 and resp.json()['metrics']['si_sdr_noisy_db'] is not None
    missing = client.post('/api/enhance', files={'file': ('a.wav', payload, 'audio/wav')},
                          data={'options': json.dumps({'method': 'neural', 'checkpoint': 'nonexistent'})})
    assert missing.status_code == 503


def test_invalid_inputs(client, monkeypatch):
    payload = fixture()
    bad = client.post('/api/enhance', files={'file': ('bad.wav', b'abc', 'audio/wav')})
    assert bad.status_code == 422
    wrong = client.post('/api/enhance', files={'file': ('bad.exe', payload, 'audio/wav')})
    assert wrong.status_code == 422
    params = client.post('/api/enhance', files={'file': ('x.wav', payload, 'audio/wav')},
                         data={'options': '{"n_fft":17}'})
    assert params.status_code == 422
    import backend.app.main as module
    monkeypatch.setattr(module, 'MAX_MB', 1)
    too_big = client.post('/api/enhance', files={'file': ('big.wav', b'a' * (1024*1024+2), 'audio/wav')})
    assert too_big.status_code == 413


def test_simultaneous_calls(client):
    def run(_):
        return client.post('/api/enhance', files={'file': ('x.wav', fixture(), 'audio/wav')}).status_code
    with ThreadPoolExecutor(max_workers=3) as executor:
        assert list(executor.map(run, range(3))) == [200] * 3


def test_streaming_websocket_packet(client):
    import struct
    with client.websocket_connect('/api/live', headers={'origin': 'http://localhost:5173'}) as ws:
        ws.send_text(json.dumps({'method': 'spectral', 'sample_rate': 16000}))
        ready = ws.receive_json()
        assert ready['status'] == 'ready'
        payload = struct.pack('<I', 12) + np.zeros(1024, dtype='<f4').tobytes()
        ws.send_bytes(payload)
        answer = ws.receive_bytes()
        assert struct.unpack_from('<I', answer)[0] == 12
        assert (len(answer) - 8) % 4 == 0
        assert np.isfinite(np.frombuffer(answer, dtype='<f4', offset=8)).all()

def test_analysis_very_short(client):
    r = client.post('/api/analyze', files={'file': ('short.wav', wav_bytes(np.ones(10, np.float32)), 'audio/wav')})
    assert r.status_code == 200
    assert r.json()['duration_s'] == 10 / 16000
