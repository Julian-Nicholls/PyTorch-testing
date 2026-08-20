#!/usr/bin/env python3
"""Download original NDWS TFRecord shards without requiring cloud credentials."""

from __future__ import annotations

import argparse
import urllib.request
from pathlib import Path

BASE_URL = "https://storage.googleapis.com/gresearch/next-day-wildfire-spread"
SHARD_COUNTS = {"train": 15, "eval": 2, "test": 2}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("data/raw"))
    parser.add_argument(
        "--full", action="store_true", help="download every shard (default: shard 00 only)"
    )
    parser.add_argument("--base-url", default=BASE_URL)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    for split, count in SHARD_COUNTS.items():
        indices = range(count) if args.full else range(1)
        for index in indices:
            name = f"next_day_wildfire_spread_{split}_{index:02d}.tfrecord"
            destination = args.output / name
            if destination.is_file() and destination.stat().st_size:
                print(f"exists: {destination}")
                continue
            temporary = destination.with_suffix(destination.suffix + ".part")
            url = f"{args.base_url.rstrip('/')}/{name}"
            print(f"downloading {url}")
            try:
                urllib.request.urlretrieve(url, temporary)
                temporary.replace(destination)
            except Exception:
                temporary.unlink(missing_ok=True)
                raise


if __name__ == "__main__":
    main()
