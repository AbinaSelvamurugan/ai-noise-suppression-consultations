# Limitations and safety

## Implemented and covered by automated tests

- Mono/stereo audio validation and 16 kHz polyphase conversion, spectral gate with configurable calibration, finite reconstruction and 32-bit float WAV output.
- Trainable PyTorch Conv-GRU with batched waveform losses, CPU forward/backward, checkpoint verification, recurrent-state continuity, and saved best/last checkpoints.
- In-memory FastAPI upload / enhance / inspect / compare / paired-reference evaluation and a headless streaming WebSocket frame exchange.
- Reproducible deterministic synthetic manifests with speaker/noise-disjoint partitions and software fixture training.

## Known limitations

- **Speech quality on real audio**: no licensed human-speech training corpus or compatible pretrained speech checkpoint is included, so intelligibility and naturalness on real speech are not characterized; synthetic non-speech test gradients cannot stand in for that.
- **Live round-trip latency**: server packet turnaround is observable, but the microphone driver, browser resampling, JS scheduling, buffering, DAC and acoustic loopback delay are not part of that measurement. Browser support is device-dependent.
- **Regulatory/security scope**: local CORS/WS origins and upload limits are not an internet-ready authentication, encryption or auditing regime.
- **Not implemented**: trained-on-cleaned voice separation guarantees, ASR/WER metrics, extensive nonstationary noise modeling, RIR training augmentation, PESQ, confidence intervals, formal MOS listening study, and clinically validated performance.

Speech can be removed as noise, especially whispered or quiet words. Consultation audio should always be reviewed against the original and never treated as an automatically improved transcript or ground truth.
