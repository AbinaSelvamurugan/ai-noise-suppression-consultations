# Dataset acquisition, provenance, partitioning and ethics

The repository contains no real human speech dataset, no consultation recordings, and no third-party checkpoint. Its `scripts/generate_fixture.py` generates mathematically constructed harmonic tones and noise **for unit and integration testing**.

## Public sources and permission checks

1. [DNS Challenge repository, README dataset-license section](https://github.com/microsoft/DNS-Challenge) — Challenge code is MIT; its **components retain separate terms** (LibriVox public-domain speech; VCTK ODC-BY; DEMAND CC BY-SA 3.0; other collections vary). DNS5 fullband is roughly terabyte scale and downloader scripts may run in header-only mode until modified. Confirm every source and restrictions before downloading/redistributing.
2. [Edinburgh VCTK/VoiceBank recordings](https://datashare.ed.ac.uk/handle/10283/2791) — consult the specific data record and included license for speaker IDs, authorized use and required attribution. Do not assume noisy paired VoiceBank+DEMAND derivatives share exactly the same permissions as clean speech.
3. [DEMAND noise recordings](https://zenodo.org/records/1227121) — note CC BY-SA 3.0 terms, including attribution and share-alike obligations where applicable. Verify the current Zenodo record and any third-party components yourself.

The availability of a URL is not permission to redistribute a recording. Confirm dataset access, consent, license compatibility, institutional review rules and attribution before real data use.

## Expected layout and validation

Place distinct speakers in distinct `data/authorized/clean/<speaker>/*.wav` directories; place ambient noise sources in `data/authorized/noise/*.wav`. `prepare_dataset.py` probes files, rejects corrupt sources and validates at least three speakers/noise files; it stores relative filenames, speaker IDs, durations, RNG seeds and SHA-256 hashes in a JSONL manifest. Speaker and noise source partitions are mutually disjoint; a runtime guard also checks SHA-256, speaker names and path boundaries. The split is deterministic for a given seed and input list and approximately 70/15/15 of speakers and noise files, though with small corpora proportions are coarse. All evaluation speech must come from speakers never used during training.

At load, signals are decoded with libsndfile, averaged if stereo, resampled with polyphase filtering to 16 kHz, and segmented at a reproducible offset (validation remains fixed each epoch; train offsets vary reproducibly with epoch). Silent examples are returned safely by the mixer. A real speech corpus should be screened for speech/non-speech tags, sample clipping, poor annotations and duplicated utterances. **Text transcripts, consent handling, VAD-based trimming, optional reverberation/room impulse responses and a listening-clip exporter are future extensions, not part of this repository.**

## Use of external data

Do not use private consultation recordings without explicit informed permission and data governance. Never commit `data/` or arbitrary large checkpoint files to Git. Original corpora may require precise citations; cite them from their primary dataset documentation.
