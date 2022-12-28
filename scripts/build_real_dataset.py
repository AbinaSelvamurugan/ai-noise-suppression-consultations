#!/usr/bin/env python3
"""Download public speech + noise and lay them out for scripts/prepare_dataset.py.

Speech: LibriSpeech dev-clean + test-clean (CC BY 4.0), about 80 speakers, grouped by speaker ID.
Noise : DEMAND (CC BY 4.0, Zenodo 1227121), channel 1 of eight environments, one file each.

Result:  data/authorized/clean/<speaker>/*.flac   data/authorized/noise/<ENV>.wav
The split (train/val/test by speaker AND by noise file) is made later by prepare_dataset.py.

Usage:  python scripts/build_real_dataset.py        (about 1.1 GB of downloads into data/, git-ignored)
"""
from __future__ import annotations

import shutil
import tarfile
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "data"
OUT = ROOT / "authorized"
SPEECH = ["dev-clean", "test-clean"]
SPEECH_URL = "https://www.openslr.org/resources/12/{}.tar.gz"
NOISE = ["STRAFFIC", "PCAFETER", "DKITCHEN", "OOFFICE", "DLIVING", "NPARK", "PSTATION", "TMETRO"]
NOISE_URL = "https://zenodo.org/records/1227121/files/{}_16k.zip"


def fetch(url: str, dest: Path) -> None:
    if not dest.exists():
        dest.parent.mkdir(parents=True, exist_ok=True)
        print(f"downloading {url}", flush=True)
        urllib.request.urlretrieve(url, dest)


def main() -> None:
    raw = ROOT / "raw"
    (OUT / "clean").mkdir(parents=True, exist_ok=True)
    (OUT / "noise").mkdir(parents=True, exist_ok=True)

    for part in SPEECH:
        archive = raw / f"{part}.tar.gz"
        fetch(SPEECH_URL.format(part), archive)
        with tarfile.open(archive) as tar:
            tar.extractall(raw, filter="data")
        for spk in sorted((raw / "LibriSpeech" / part).iterdir()):
            target = OUT / "clean" / f"speaker_{spk.name}"
            target.mkdir(exist_ok=True)
            for flac in spk.rglob("*.flac"):
                shutil.move(str(flac), target / flac.name)
        archive.unlink()

    for name in NOISE:
        zpath = raw / f"{name}_16k.zip"
        fetch(NOISE_URL.format(name), zpath)
        with zipfile.ZipFile(zpath) as z:
            member = next(m for m in z.namelist() if m.endswith("ch01.wav"))
            (OUT / "noise" / f"{name}.wav").write_bytes(z.read(member))
        zpath.unlink()

    speakers = len(list((OUT / "clean").iterdir()))
    files = sum(1 for _ in (OUT / "clean").rglob("*.flac"))
    print(f"speakers: {speakers}, utterances: {files}, noise files: {len(NOISE)} -> {OUT}")


if __name__ == "__main__":
    main()
