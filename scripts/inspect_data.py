#!/usr/bin/env python3
"""Print and plot one prepared NDWS sample."""

from __future__ import annotations

import argparse
import numpy as np

from wildfire_ml.data import CHANNEL_NAMES, NDWSDataset
from wildfire_ml.visualize import plot_sample


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepared", default="data/prepared")
    parser.add_argument("--split", choices=("train", "validation", "test"), default="train")
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--output", help="write a PNG instead of opening a window")
    args = parser.parse_args()
    sample = NDWSDataset(args.prepared, args.split)[args.index]
    features = sample["input"].numpy()
    target = sample["target"].numpy()
    valid = sample["validity_mask"].numpy()
    print(f"sample_id: {sample['sample_id']}")
    print(f"channels: {list(CHANNEL_NAMES)}")
    print(f"input: {features.shape}; target: {target.shape}; validity_mask: {valid.shape}")
    for index, name in enumerate(CHANNEL_NAMES):
        values = features[index]
        print(f"{name:>12}: min={np.nanmin(values):.4g}, max={np.nanmax(values):.4g}, nonfinite={np.count_nonzero(~np.isfinite(values))}")
    print(f"valid target pixels: {valid.sum()}/{valid.size}; fire pixels: {target[valid].sum():.0f}")
    plot_sample(sample, args.output)


if __name__ == "__main__":
    main()
