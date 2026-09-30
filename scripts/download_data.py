#!/usr/bin/env python3
"""Download and extract the canonical Kaggle NDWS distribution."""

from __future__ import annotations

import argparse
import shutil
import urllib.request
import zipfile
from pathlib import Path

DATASET_PAGE = "https://www.kaggle.com/datasets/fantineh/next-day-wildfire-spread"
DOWNLOAD_URL = "https://www.kaggle.com/api/v1/datasets/download/fantineh/next-day-wildfire-spread"


def download_and_extract(output: Path, archive: Path, url: str = DOWNLOAD_URL) -> list[Path]:
    """Obtain a Kaggle archive and extract only its TFRecord files."""
    output.mkdir(parents=True, exist_ok=True)
    if not archive.is_file():
        archive.parent.mkdir(parents=True, exist_ok=True)
        temporary = archive.with_suffix(archive.suffix + ".part")
        print(f"downloading canonical NDWS release from {url}")
        try:
            with urllib.request.urlopen(url) as response, temporary.open("wb") as destination:
                shutil.copyfileobj(response, destination)
            temporary.replace(archive)
        except Exception:
            temporary.unlink(missing_ok=True)
            raise
    else:
        print(f"using existing archive: {archive}")

    extracted: list[Path] = []
    with zipfile.ZipFile(archive) as bundle:
        members = sorted(name for name in bundle.namelist() if name.endswith(".tfrecord"))
        if not members:
            raise ValueError(f"No .tfrecord files found in {archive}")
        for member in members:
            destination = output / Path(member).name
            if not destination.is_file():
                with bundle.open(member) as source, destination.open("wb") as target:
                    shutil.copyfileobj(source, target)
            extracted.append(destination)
    print(f"ready: {len(extracted)} TFRecord shards in {output}")
    return extracted


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("data/raw"))
    parser.add_argument("--archive", type=Path, default=Path("data/ndws-kaggle.zip"))
    parser.add_argument("--url", default=DOWNLOAD_URL, help=argparse.SUPPRESS)
    args = parser.parse_args()
    try:
        download_and_extract(args.output, args.archive, args.url)
    except Exception as error:
        raise SystemExit(
            f"NDWS download failed: {error}\nDownload the archive from {DATASET_PAGE} "
            f"and rerun with --archive /path/to/archive.zip"
        ) from error


if __name__ == "__main__":
    main()
