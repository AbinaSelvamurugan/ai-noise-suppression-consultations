"""FastAPI audio service; stateless uploads, bounded concurrency, retained models only."""
from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from dataclasses import asdict
from io import BytesIO
import json
import logging
import os
from pathlib import Path
import struct
import time
from zipfile import ZipFile, ZIP_DEFLATED

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel, Field, ValidationError
from scipy.signal import stft
import torch

from consult_audio.audio import (AudioError, SUPPORTED_EXTENSIONS, TARGET_SR, decode,
                                 prepare, wav_bytes)
from consult_audio.checkpoint import CheckpointError, load_checkpoint
from consult_audio.metrics import evaluate_pair
from consult_audio.spectral import GateConfig, spectral_gate
from consult_audio.stream import StreamingEnhancer, neural_enhance

VERSION = '0.1.0'
log = logging.getLogger('consult_audio.api')
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(name)s %(message)s')
MAX_MB = max(1, min(100, int(os.getenv('CONSULT_MAX_UPLOAD_MB', '20'))))
MAX_SECONDS = max(1, min(1800, int(os.getenv('CONSULT_MAX_DURATION_S', '120'))))
MAX_CONCURRENT = max(1, min(16, int(os.getenv('CONSULT_MAX_CONCURRENT', '2'))))
CHECKPOINT_DIR = Path(os.getenv('CONSULT_CHECKPOINT_DIR', 'artifacts/checkpoints'))
DEVICE = os.getenv('CONSULT_DEVICE', 'cpu')
if DEVICE == 'cuda' and not torch.cuda.is_available():
    log.warning('CUDA requested but unavailable; switching to CPU')
    DEVICE = 'cpu'


class Settings(BaseModel):
    method: str = 'spectral'
    checkpoint: str = ''
    strength: float = Field(default=1.4, ge=0, le=4)
    n_fft: int = Field(default=512)
    hop: int = Field(default=128, ge=1, le=1024)
    noise_seconds: float = Field(default=.35, gt=0, le=10)
    noise_start_seconds: float = Field(default=0, ge=0, le=1800)
    minimum_gain: float = Field(default=.08, ge=0, le=1)
    smooth_time: float = Field(default=1, ge=0, le=8)
    smooth_frequency: float = Field(default=1, ge=0, le=8)

    def gate(self) -> GateConfig:
        cfg = GateConfig(**{key: getattr(self, key) for key in GateConfig.__dataclass_fields__})
        cfg.check()
        return cfg


class LiveSettings(BaseModel):
    method: str = 'spectral'
    checkpoint: str = ''
    strength: float = Field(default=1.4, ge=0, le=4)
    minimum_gain: float = Field(default=.08, ge=0, le=1)
    sample_rate: int = 16000


class ModelRegistry:
    def __init__(self):
        self.models = {}
        self.errors = {}
        # Checkpoints are placed only by trusted server operators, never users.
        if CHECKPOINT_DIR.exists():
            for path in sorted(CHECKPOINT_DIR.glob('*.pt')):
                if not path.stem.replace('_', '').replace('-', '').isalnum():
                    continue
                try:
                    model, data = load_checkpoint(path, DEVICE)
                    self.models[path.stem] = (model, data['metadata'])
                except CheckpointError:
                    self.errors[path.stem] = 'Missing or incompatible checkpoint'
                    log.warning('Checkpoint rejected: %s', path.name)

    def get(self, name: str):
        if not name:
            name = 'best' if 'best' in self.models else next(iter(self.models), '')
        if name not in self.models:
            raise HTTPException(status_code=503, detail='No compatible trained neural checkpoint available')
        return self.models[name][0]

    def listing(self) -> list[dict]:
        return [{'name': name, **meta} for name, (_, meta) in self.models.items()]


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.registry = ModelRegistry()
    app.state.limit = asyncio.Semaphore(MAX_CONCURRENT)
    yield
    app.state.registry.models.clear()


app = FastAPI(title='Consultation Audio Enhancement API', version=VERSION, lifespan=lifespan)
app.add_middleware(CORSMiddleware,
                   allow_origins=[o.strip() for o in os.getenv('CONSULT_CORS_ORIGINS', 'http://localhost:5173').split(',') if o.strip()],
                   allow_methods=['GET', 'POST'], allow_headers=['Content-Type'],
                   expose_headers=['X-Audio-Metadata'])


@app.exception_handler(AudioError)
async def audio_error(_request, exc: AudioError):
    return JSONResponse(status_code=422, content={'detail': str(exc)})


@app.exception_handler(ValidationError)
async def validation_error(_request, _exc: ValidationError):
    return JSONResponse(status_code=422, content={'detail': 'Invalid enhancement settings'})


def _settings(raw: str) -> Settings:
    try:
        cfg = Settings.model_validate_json(raw)
        if cfg.method not in ('spectral', 'neural'):
            raise AudioError('method must be spectral or neural')
        if cfg.method == 'spectral':
            cfg.gate()
        return cfg
    except ValidationError as exc:
        raise AudioError('Invalid enhancement settings') from exc


async def _read_upload(file: UploadFile) -> tuple[np.ndarray, dict]:
    filename = file.filename or ''
    if Path(filename).suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise AudioError('Supported formats: WAV, FLAC, OGG')
    try:
        payload = await file.read(MAX_MB * 1024 * 1024 + 1)
    finally:
        await file.close()
    if len(payload) > MAX_MB * 1024 * 1024:
        raise HTTPException(status_code=413, detail='Upload exceeds configured size limit')
    audio, info = decode(payload, max_seconds=MAX_SECONDS)
    # Do not persist files or names; payload is scoped to request.
    return prepare(audio, info.sample_rate), asdict(info)


def _run(x: np.ndarray, cfg: Settings, registry: ModelRegistry) -> tuple[np.ndarray, dict]:
    start = time.perf_counter()
    if cfg.method == 'spectral':
        y = spectral_gate(x, TARGET_SR, cfg.gate())
        name = 'spectral-gate'
    else:
        model = registry.get(cfg.checkpoint)
        y = neural_enhance(x, model, DEVICE)
        name = cfg.checkpoint or ('best' if 'best' in registry.models else next(iter(registry.models)))
    seconds = time.perf_counter() - start
    if len(y) != len(x) or not np.isfinite(y).all():
        raise RuntimeError('Enhancer returned an invalid signal')
    return y, {'method': cfg.method, 'model': name, 'device': DEVICE,
               'sample_rate': TARGET_SR, 'channels': 1, 'format': 'WAV IEEE float32',
               'duration_s': len(y) / TARGET_SR, 'processing_ms': round(seconds * 1000, 3),
               'real_time_factor': seconds / max(len(y) / TARGET_SR, 1e-6),
               'peak_memory_mb': None, 'peak_memory_note': 'Unavailable'}


async def _bounded(job, *args):
    try:
        await asyncio.wait_for(app.state.limit.acquire(), timeout=5)
    except asyncio.TimeoutError as exc:
        raise HTTPException(429, 'Server busy; try again later') from exc
    try:
        return await run_in_threadpool(job, *args)
    finally:
        app.state.limit.release()


@app.get('/api/health')
def health():
    return {'status': 'ok', 'version': VERSION, 'sample_rate': TARGET_SR, 'device': DEVICE,
            'neural_available': bool(app.state.registry.models), 'spectral_available': True}


@app.get('/api/models')
def models():
    return {'models': app.state.registry.listing(), 'rejected_checkpoints': len(app.state.registry.errors),
            'device': DEVICE, 'configured_sample_rate': TARGET_SR,
            'supported_formats': sorted(SUPPORTED_EXTENSIONS),
            'max_upload_mb': MAX_MB, 'max_duration_s': MAX_SECONDS,
            'max_concurrent': MAX_CONCURRENT, 'version': VERSION,
            'stream': {'sample_rate': 16000, 'frames_per_packet': 1024,
                       'hop': 128, 'methods': ['spectral', 'neural']}}


@app.get('/api/training/status')
def training_status():
    return {'worker_enabled': False, 'state': 'not_running',
            'message': 'Training runs offline using scripts/train.py; no server-side training worker'}


@app.post('/api/enhance')
async def enhance(file: UploadFile = File(...), options: str = Form('{}')):
    cfg = _settings(options)
    x, source = await _read_upload(file)
    y, stats = await _bounded(_run, x, cfg, app.state.registry)
    stats['source'] = source
    return Response(wav_bytes(y), media_type='audio/wav',
                    headers={'Content-Disposition': 'attachment; filename="enhanced.wav"',
                             'X-Audio-Metadata': json.dumps(stats, separators=(',', ':'))})


def _analysis(x: np.ndarray) -> dict:
    length = len(x)
    blocks = np.array_split(x, min(400, length))
    peaks = [float(np.max(np.abs(b))) for b in blocks]
    averages = [float(np.mean(b)) for b in blocks]
    padded = np.pad(x, (0, max(0, 512 - len(x))))
    _, _, s = stft(padded, fs=TARGET_SR, nperseg=512, noverlap=256)
    # Explicitly downsample axes; maximum 129 x 120 spectrum elements.
    frequencies = np.linspace(0, s.shape[0] - 1, min(129, s.shape[0])).astype(int)
    times = np.linspace(0, s.shape[1] - 1, min(120, s.shape[1])).astype(int)
    db = 20 * np.log10(np.maximum(abs(s[np.ix_(frequencies, times)]), 1e-7))
    return {'duration_s': length / TARGET_SR, 'sample_rate': TARGET_SR,
            'peak': peaks, 'waveform': averages, 'spectrogram_db': np.round(db, 1).tolist(),
            'spectrogram_min_db': -100, 'spectrogram_max_db': 0}


@app.post('/api/analyze')
async def analyze(file: UploadFile = File(...)):
    x, _source = await _read_upload(file)
    return await _bounded(_analysis, x)


@app.post('/api/compare')
async def compare(file: UploadFile = File(...), options: str = Form('{}')):
    cfg = _settings(options)
    x, source = await _read_upload(file)
    def build():
        outcomes = {}
        signals = {}
        for method in ('spectral', 'neural'):
            try:
                method_cfg = cfg.model_copy(update={'method': method})
                if method == 'spectral':
                    method_cfg.gate()
                signals[method], outcomes[method] = _run(x, method_cfg, app.state.registry)
            except HTTPException:
                outcomes[method] = {'available': False, 'reason': 'Trained checkpoint not installed'}
        buf = BytesIO()
        with ZipFile(buf, 'w', ZIP_DEFLATED) as z:
            z.writestr('original.wav', wav_bytes(x))
            for method, signal in signals.items():
                z.writestr(f'{method}.wav', wav_bytes(signal))
            z.writestr('measurements.json', json.dumps({'source': source, 'results': outcomes}, indent=2))
        return buf.getvalue()
    archive = await _bounded(build)
    return Response(archive, media_type='application/zip',
                    headers={'Content-Disposition': 'attachment; filename="comparison.zip"'})


@app.post('/api/evaluate')
async def evaluate(noisy: UploadFile = File(...), reference: UploadFile = File(...),
                   options: str = Form('{}')):
    cfg = _settings(options)
    x, _ = await _read_upload(noisy)
    clean, _ = await _read_upload(reference)
    if len(x) != len(clean):
        raise AudioError('Clean reference and noisy recording must be sample-aligned at 16 kHz')
    y, stats = await _bounded(_run, x, cfg, app.state.registry)
    metrics = await _bounded(evaluate_pair, clean, x, y, TARGET_SR)
    return {'metrics': metrics, 'processing': stats,
            'alignment_assumption': 'Provided clean reference is correctly time-aligned',
            'reference_required': True}


@app.websocket('/api/live')
async def live(ws: WebSocket):
    # Origin restriction for browser websocket connections (CORS middleware does not cover WS).
    origin = ws.headers.get('origin')
    allowed = [x.strip() for x in os.getenv('CONSULT_CORS_ORIGINS', 'http://localhost:5173').split(',')]
    if origin not in allowed:
        await ws.close(code=1008)
        return
    await ws.accept()
    stream = None
    try:
        init = await asyncio.wait_for(ws.receive_text(), timeout=5)
        cfg = LiveSettings.model_validate_json(init)
        if cfg.sample_rate != TARGET_SR or cfg.method not in ('spectral', 'neural'):
            await ws.close(code=1003)
            return
        model = app.state.registry.get(cfg.checkpoint) if cfg.method == 'neural' else None
        stream = StreamingEnhancer(method=cfg.method, model=model, device=DEVICE,
                                   strength=cfg.strength, min_gain=cfg.minimum_gain)
        await ws.send_json({'status': 'ready', 'sample_rate': TARGET_SR,
                            'frame_samples': 1024, 'model': cfg.method})
        received = 0
        while True:
            msg = await asyncio.wait_for(ws.receive_bytes(), timeout=20)
            if len(msg) != 4 + 1024 * 4:
                await ws.close(code=1003)
                return
            seq = struct.unpack_from('<I', msg)[0]
            audio = np.frombuffer(msg, dtype='<f4', count=1024, offset=4)
            if not np.isfinite(audio).all():
                await ws.close(code=1003)
                return
            received += 1024
            if received > 16000 * 60 * 10:
                await ws.close(code=1009)
                return
            started = time.perf_counter()
            out = await _bounded(stream.push, audio.copy())
            processing_ms = (time.perf_counter() - started) * 1000
            # Header: packet sequence + server turnaround ms; client sees only the audio samples.
            await ws.send_bytes(struct.pack('<If', seq, processing_ms) + out.astype('<f4').tobytes())
    except (WebSocketDisconnect, asyncio.TimeoutError):
        pass
    except (ValidationError, ValueError, HTTPException):
        await ws.close(code=1008)
    except Exception:
        log.exception('Live stream failed (without recording payload)')
        await ws.close(code=1011)
    finally:
        del stream
