"""PyTorch boundary for prepared Next Day Wildfire Spread samples."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

# Original feature spelling and order from the NDWS publication/code.
CHANNEL_NAMES = (
    "elevation",
    "pdsi",
    "NDVI",
    "pr",
    "sph",
    "th",
    "tmmn",
    "tmmx",
    "vs",
    "erc",
    "population",
    "PrevFireMask",
)


class NDWSDataset(Dataset[dict[str, object]]):
    """Load deterministic, prepared ``.npz`` samples as named tensors.

    No normalization or augmentation is hidden here. Invalid target cells have
    a placeholder zero in ``target`` and are *only* identified by
    ``validity_mask``; future losses and metrics must apply that mask.
    """

    def __init__(self, prepared_root: str | Path, split: str = "train") -> None:
        self.split = split
        manifest_path = Path(prepared_root) / split / "manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(f"Prepared split manifest not found: {manifest_path}")
        manifest = json.loads(manifest_path.read_text())
        if tuple(manifest["channel_names"]) != CHANNEL_NAMES:
            raise ValueError("Prepared channel names/order do not match this package")
        self._directory = manifest_path.parent
        self._samples = manifest["samples"]

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, index: int) -> dict[str, object]:
        entry = self._samples[index]
        with np.load(self._directory / entry["file"]) as sample:
            features = np.array(sample["input"], dtype=np.float32, copy=True)
            target = np.array(sample["target"], dtype=np.float32, copy=True)
            valid = np.array(sample["validity_mask"], dtype=np.bool_, copy=True)
        if features.shape != (12, 64, 64) or target.shape != (1, 64, 64):
            raise ValueError(f"Unexpected sample shapes: {features.shape}, {target.shape}")
        if valid.shape != target.shape:
            raise ValueError("Validity mask and target shapes differ")
        if not np.all(np.isin(target[valid], (0.0, 1.0))):
            raise ValueError("Valid targets must be binary")
        return {
            "input": torch.from_numpy(features),
            "target": torch.from_numpy(target),
            "validity_mask": torch.from_numpy(valid),
            "sample_id": entry["id"],
        }
