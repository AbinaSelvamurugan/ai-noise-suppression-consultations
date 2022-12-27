# Training protocol

1. License and stage speaker-labeled clean and noise corpora under `data/authorized`.
2. Generate reproducible JSONL manifest: `python scripts/prepare_dataset.py --root data/authorized --output artifacts/authorized-manifest.jsonl`.
3. Inspect speaker split size/count and audio source integrity; retain the manifest for independent reproduction.
4. Edit `configs/model.yaml` and `configs/training.yaml`; run `python scripts/train.py --root data/authorized --manifest artifacts/authorized-manifest.jsonl --output artifacts/checkpoints`.
5. Train on CPU or install matching CUDA PyTorch first and pass `--device cuda`; CUDA kernels can still be nondeterministic on some systems.
6. Check `history.csv` and `configuration.json`, validate no nonfinite losses; `best.pt` is minimum validation total loss; `last.pt` retains optimizer + scheduler for resume. `--resume artifacts/checkpoints/last.pt --epochs 60` resumes epoch count to a larger requested total. Do not change model dimensions between run and resume.
7. Hold out test speakers for benchmark, perform listening reviews, and document noise conditions before deploying neural inference.

Training uses `log1p` magnitude features from 512-point sqrt-Hann STFT; two frequency convolutions, a GRU, and an SNR-aware spectral gain output. Waveform, magnitude and log-magnitude loss terms are separately logged. Optimizer: AdamW. LR: ReduceLROnPlateau. Gradient norm clipping: 3. AMP only for compatible CUDA. Random seeds, DataLoader shuffle generator and worker count configured. Training is done **outside the HTTP server**, and `/api/training/status` reports the trainer state.

**Training reliability limits:** Full optimizer/scheduler/RNG trajectories are not saved in checkpoints; after resume, model and optimizer state are preserved but randomly sampled sequence order may not be bitwise identical to uninterrupted training for all device configurations. Best-checkpoint selection threshold 1e-5; patience defaults to 7. The software test fixture is non-speech and is not a speech dataset.
