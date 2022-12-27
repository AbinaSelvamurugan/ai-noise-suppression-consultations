# Troubleshooting

- **`503 No compatible trained neural checkpoint`**: no local `artifacts/checkpoints/best.pt`; train on real licensed speech/noise with matching model config and restart backend. Never copy the synthetic test fixture checkpoint to a real-serving path.
- **`422 invalid audio`**: use a decodable mono/stereo WAV/FLAC/OGG, at 8–192 kHz and within upload and duration limits. Files mislabeled with `.wav` are rejected after decoder checks.
- **Too little noise reduction**: calibrate on a true noise-only segment and adjust spectral `strength`; stronger values can remove soft speech.
- **Metallic/robotic voice**: lower strength, increase floor and smoothing, choose a cleaner calibration interval. Avoid judging solely by waveform appearance.
- **Mic permission denied**: grant site permission, use localhost or TLS, confirm OS input device and another application is not holding exclusive access.
- **Feedback in live monitor**: use headphones; stop the session, reduce output volume or choose headphones and retry.
- **Crackles, missing audio**: monitor live packet turnaround and overflow indications; close other CPU-intensive apps, check local network and buffering. Packet RTT is NOT end-to-end audio latency.
- **`torch`/`scipy` install failure**: use supported Python 3.11/3.12; confirm pinned wheels are available for architecture and install a matching CUDA wheel separately if needed.
- **Missing npm dependencies**: install Node dependencies with `npm install` to build the frontend from source.
- **Unequal reference/noisy lengths**: use sample-aligned pairs with equal duration after 16 kHz resampling. Do not misinterpret metrics from unrelated recordings.
- **Short audio**: spectral processing zero pads to FFT size and trims output; always audition near boundaries.
