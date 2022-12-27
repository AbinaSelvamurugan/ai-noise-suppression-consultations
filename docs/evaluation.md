# Reference-based evaluation

`evaluate_pair(clean, noisy, output)` requires **equal sample counts** in identical 16-kHz aligned waveforms. UI `POST /api/evaluate` processes the noisy input and checks same resampled length; the caller must still assert sample-level **temporal alignment**. If invalid, align externally rather than shifting signals after scoring to maximize a metric.

- **SI-SDR (dB)** projects centered estimate onto centered clean speech, then compares projected target energy with residual energy. SI-SDR improvement = enhanced SI-SDR minus noisy SI-SDR. Cannot infer SI-SDR from a recording alone; silence returns `null`.
- **Segmental SNR (dB)** average of 20 ms active-reference frames with per-frame clipping [−10,+35] dB; defined for aligned waveform and not scale invariant.
- **Log-spectral distance (dB)** time-averaged RMS difference between log STFT magnitudes; smaller generally indicates closer spectral match but does not establish intelligibility.
- **STOI** standard mode via optional `pystoi` library; only for meaningful sufficiently long reference material at declared supported sample rates. Requires the clean waveform and should not be interpreted as a human subjective listening score.
- **PESQ:** left `null` since no qualified, separately licensed PESQ evaluation dependency is redistributed. PESQ has narrowband/wideband and licensing restrictions.
- **Latency (ms):** Python `perf_counter` measurement of the inference path per full example, excluding browser and network.
- **RTF:** inference duration / recording duration; less than 1 on an isolated benchmark *suggests* offline faster-than-real-time but alone does not guarantee live underrun-free operation.

`python scripts/benchmark.py` evaluates the held-out `test` manifest, using the identical generated mixture for each method, records each example/SNR/latency/metric, writes `results.csv` and `results.json`, and plots measured condition means against controlled input SNR to PNG. Report N, input SNR, CPU/GPU hardware, sample rates, seed, checkpoint origin, statistical dispersion, and confidence bounds with N>1. The script exports raw per-example values, per-SNR mean/std and simple 95% normal-approximation intervals for N≥5; it does not perform formal paired bootstrap inference. A single synthetic example is not a statistical comparison.

Experiment report template:

Per method, report N, SI-SDR gain (mean ± std dB), STOI change, RTF (median) and device, together with input SNR, hardware, sample rates, seed, checkpoint origin and confidence bounds. These are the fields to record when measurements are performed, not reported results.
