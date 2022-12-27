# Engineering deliverables

## Suggested written report outline

1. **Abstract** — goal, paired methods, exact trained scope and evidence-based results.
2. **Introduction** — remote consultation audio problems; noise sources, speech privacy and intelligibility.
3. **Literature review** — classical spectral subtraction/gating, STFT masks, CRN/GRU designs, learned filterbank systems and RNNoise; cite the papers from publisher records.
4. **Problem definition** — preserve quiet speech while rejecting ambient noise; target 16 kHz, latency, and meaningful metrics.
5. **Mathematical method** — STFT/ISTFT, spectral mask, source mixing SNR, normalization, losses and streaming OLA.
6. **Requirements/system design** — FastAPI/React/AudioWorklet, CPU device, checkpoint lifecycle, abuse limits, architecture diagram.
7. **Implementation** — source modules, feature tensor contracts, GRU state, data manifest provenance, API/WS formats.
8. **Experiments** — authorized data names, speaker-disjoint split, SNR conditions, training configuration, training duration, checkpoint hashes and device.
9. **Results** — plots and tables from the benchmark; per-example statistics and listening review.
10. **Discussion** — errors on reverberation, broadband overlap, calibrating during speech, latency constraints, cancellation of stereo phase.
11. **Limitations/ethics** — no guarantee of word preservation, data consent, privacy controls, clinical noncertification and evaluation gaps.
12. **Conclusion/future research** — larger multilingual noise corpora, formal human listening tests, ASR WER, confidence intervals, PESQ licensing, RIR augmentation, latency loopback methodology.
13. **References** — cite original DNS/DEMAND/VCTK dataset and architectural papers after checking metadata and licenses.
14. **Appendices** — version lockfiles, commands, configuration files, benchmark outputs, API schemas, test output, consent procedures.

## Presentation outline (12 slides)

1. Context and constraints. 2. Scope and architecture. 3. DSP baseline. 4. Conv-GRU rationale. 5. Mathematical features. 6. Training/data provenance. 7. API and model lifecycle. 8. Before/after UI demonstration. 9. Live AudioWorklet and packet lifecycle. 10. Evaluation setup and measured results. 11. Test evidence and limitations. 12. Conclusion and future work.

## Safe demonstration sequence

Start backend and frontend. Load an authorized nonsensitive demonstration recording, choose a noise-only interval, perform spectral gating, inspect waveform and spectrogram, listen original and processed while keeping playback position, download float WAV and verify duration; compare same-input available methods; show the no-checkpoint state if no trained speech model; show paired-reference evaluation only when an aligned clean demo is available. Stream a non-sensitive microphone phrase through Live with headphones and point out that packet-turnaround time is not the same as full acoustic latency. Show passing tests. Use real speech benchmark evidence rather than toy synthetic-data SI-SDR.

## Example questions and grounded answers

**Q1: Why STFT?** A: It separates time/frequency structure and enables interpretable gain masks; overlapping noise/speech may still be inseparable.

**Q2: What does your GRU retain?** A: A hidden state `[layers,batch,hidden]` from previous spectral frames. It is reset per recording or stream and preserved across chunks, enabling causal recurrent inference.

**Q3: How do you handle overlap/add?** A: Apply periodic sqrt-Hann on analysis and synthesis, accumulate synthesized samples and divide by summed window power. In live mode 512-sample frames advance by 128 samples.

**Q4: How do you create known-SNR pairs?** A: Power-match noise with \(\alpha=\sqrt{P_s/(10^{SNR/10}P_n)}\), with safe handling for negligible power.

**Q5: Why not MSE alone?** A: L1 waveform loss, spectral magnitude and log-magnitude losses jointly capture sample-domain and spectral errors; no single objective guarantees naturalness.

**Q6: Why does the neural option sometimes disappear?** A: The server only loads a compatible locally trained and metadata-validated checkpoint; no randomly initialized inference is used.

**Q7: What metrics need clean speech?** A: SI-SDR, STOI, and log spectral distance all need an aligned clean reference; upload-only noise files cannot yield reference-based quality results.

**Q8: Can the residual represent pure background noise?** A: No; subtraction can include speech components removed by the enhancer and artifacts from alignment/resampling.

**Q9: Is 8 ms the latency?** A: No. Eight milliseconds is an STFT hop interval. Measured server turnaround and real end-to-end acoustic latency are different quantities.

**Q10: Is it ready for medical deployment?** A: No. The prototype does not carry medical certification, legal privacy compliance, word-integrity validation, or clinically supervised performance evidence.
