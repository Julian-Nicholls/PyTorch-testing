#!/usr/bin/env python3
"""Convert original NDWS TFRecords to compact, framework-neutral NPZ samples."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from tfrecord.reader import tfrecord_loader

from wildfire_ml.data import CHANNEL_NAMES

TARGET_NAME = "FireMask"
DESCRIPTION = {name: "float" for name in (*CHANNEL_NAMES, TARGET_NAME)}


def prepare_split(raw: Path, output: Path, source_split: str, split: str, limit: int) -> None:
    shards = sorted(raw.glob(f"next_day_wildfire_spread_{source_split}_*.tfrecord"))
    if not shards:
        raise FileNotFoundError(f"No {source_split!r} TFRecords found under {raw}")
    split_dir = output / split
    split_dir.mkdir(parents=True, exist_ok=True)
    entries: list[dict[str, str]] = []

    for shard in shards:
        for record_number, record in enumerate(tfrecord_loader(str(shard), None, DESCRIPTION)):
            sample_id = f"{source_split}/{shard.stem}:{record_number}"
            features = np.stack(
                [np.asarray(record[name], dtype=np.float32).reshape(64, 64) for name in CHANNEL_NAMES]
            )
            raw_target = np.asarray(record[TARGET_NAME], dtype=np.float32).reshape(1, 64, 64)
            # NDWS uses -1 for uncertain/unobserved target cells. Treat any
            # non-finite/non-binary value as invalid rather than as no-fire.
            validity = np.isfinite(raw_target) & np.isin(raw_target, (0.0, 1.0))
            target = np.where(validity, raw_target, 0.0).astype(np.float32)
            filename = f"sample_{len(entries):05d}.npz"
            np.savez_compressed(
                split_dir / filename,
                input=features,
                target=target,
                validity_mask=validity,
                raw_target=raw_target,
            )
            entries.append({"id": sample_id, "file": filename})
            if len(entries) >= limit:
                break
        if len(entries) >= limit:
            break

    manifest = {
        "format_version": 1,
        "source_split": source_split,
        "channel_names": list(CHANNEL_NAMES),
        "samples": entries,
    }
    (split_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"prepared {len(entries)} {split} samples in {split_dir}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", type=Path, default=Path("data/raw"))
    parser.add_argument("--output", type=Path, default=Path("data/prepared"))
    parser.add_argument("--samples-per-split", type=int, default=64)
    args = parser.parse_args()
    if args.samples_per_split < 1:
        parser.error("--samples-per-split must be positive")
    for source, destination in (("train", "train"), ("eval", "validation"), ("test", "test")):
        prepare_split(args.raw, args.output, source, destination, args.samples_per_split)


if __name__ == "__main__":
    main()
