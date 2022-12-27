# Mathematical methodology

## STFT and spectral gating

For signal \(x[t]\), frame index \(m\), and frequency index \(k\):
\[X[m,k]=\sum_{n=0}^{N-1} x[n+mH]w[n]e^{-j2\pi kn/N}.\]
Median calibration \(\widehat D[k]=\mathrm{median}_{m\in \mathcal N}|X[m,k]|\) assumes \(\mathcal N\) contains **noise-only** frames. Gain \(G[m,k]=\max(g_{\min}, 1-\lambda \widehat D[k]/(|X[m,k]|+\epsilon))\) is smoothed in time/frequency using a Gaussian filter and bounded to [minimum gain, 1]. \(\widehat S=GX\). Inverse STFT with Hann overlap/add reconstructs the signal with output length trimmed to the original. The smooth gain suppresses isolated holes, reducing musical noise. Overlap of speech and noise spectra, nonstationarity, errors in calibration, and severe reverberation can defeat the gate.

The stream uses periodic **sqrt Hann** for both analysis/synthesis plus window-power normalization, and causal noise updates. STFT nFFT=512 and hop=128 at 16 kHz (32 ms frames, 8 ms hops). Offline gate `n_fft` supports 256, 512, 1024, and 2048. The native input rates 8–192 kHz resample to 16 kHz for service comparison. Stereo is averaged (potential phase cancellation is a known trade-off). Quiet or empty windows are treated safely, float output is not peak-limited; floating-point WAV avoids unnecessary clipping.

## Controlled SNR

Given clean \(s\) and noise \(n\), measured mean powers \(P_s=E[s^2]\) and \(P_n=E[n^2]\), target ratio \(r=10^{\mathrm{SNR}_{dB}/10}\):
\[\alpha=\sqrt{P_s/(rP_n)},\qquad x=s+\alpha n.\]
If either power is effectively zero, mixing returns unchanged clean speech and zero added noise; this avoids undefined SNR. During training the SNR is drawn uniformly from −5 to 20 dB. Additive mixing is not reverberation modelling; room impulse response augmentation is not included in this version.

## Loss and stability

The training objective is \(L=w_1\|\widehat s-s\|_1+w_2\frac{1}{3}\sum_{N\in\{256,512,1024\}}\||STFT_N(\widehat s)|-|STFT_N(s)|\|_1+w_3\frac{1}{3}\sum_N\|\log(|STFT_N(\widehat s)|+\epsilon)-\log(|STFT_N(s)|+\epsilon)\|_1\). Defaults: 1.0/0.5/0.5. The time loss favors waveform retention; magnitude/log losses penalize differently scaled spectral errors. The objective combines waveform and spectral losses; no differentiable perceptual metric is included. AdamW, plateau LR scheduler, gradient clipping at norm 3, CUDA AMP when applicable, and nonfinite validation/gradient checks are implemented. Best validation **total loss** selects checkpoint. Early stopping uses configurable patience and threshold 1e-5.

## Fair comparison and voice preservation

Use aligned clean/noisy pairs from the **same** test partition with fixed SNR seeds, and apply both methods to the same mixture. Report noise spectrum, SI-SDR gain and listening examples, not attenuation alone. A neural model with higher SDR can still sound unnatural or lose faint words; objective metrics complement listening, never replace it. Residual \(x-\widehat s\) may contain discarded speech and cannot be called pure noise.
