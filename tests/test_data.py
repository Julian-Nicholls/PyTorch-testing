from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import zipfile

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from wildfire_ml.data import CHANNEL_NAMES, NDWSDataset
from wildfire_ml.visualize import plot_sample


def load_script(name: str):
    script_path = Path(__file__).parents[1] / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, script_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture()
def prepared(tmp_path: Path) -> Path:
    for split, offset in (("train", 0), ("validation", 10), ("test", 20)):
        directory = tmp_path / split
        directory.mkdir()
        features = np.full((12, 64, 64), offset, dtype=np.float32)
        target = np.zeros((1, 64, 64), dtype=np.float32)
        target[:, 2, 3] = 1
        validity = np.ones((1, 64, 64), dtype=np.bool_)
        validity[:, 4, 5] = False
        np.savez_compressed(
            directory / "sample_00000.npz",
            input=features,
            target=target,
            validity_mask=validity,
        )
        manifest = {
            "format_version": 1,
            "channel_names": list(CHANNEL_NAMES),
            "samples": [{"id": f"{split}/unique-0", "file": "sample_00000.npz"}],
        }
        (directory / "manifest.json").write_text(json.dumps(manifest))
    return tmp_path


def test_prepared_sample_shapes_channels_and_target_semantics(prepared: Path) -> None:
    dataset = NDWSDataset(prepared, "train")
    sample = dataset[0]

    assert CHANNEL_NAMES == (
        "elevation", "pdsi", "NDVI", "pr", "sph", "th", "tmmn", "tmmx",
        "vs", "erc", "population", "PrevFireMask",
    )
    assert sample["input"].shape == (12, 64, 64)
    assert sample["target"].shape == (1, 64, 64)
    assert sample["validity_mask"].shape == (1, 64, 64)
    assert set(torch.unique(sample["target"]).tolist()) == {0.0, 1.0}
    assert sample["validity_mask"].dtype is torch.bool
    assert not sample["validity_mask"][0, 4, 5]


def test_split_sample_identities_are_disjoint(prepared: Path) -> None:
    identities = [
        {NDWSDataset(prepared, split)[i]["sample_id"] for i in range(1)}
        for split in ("train", "validation", "test")
    ]
    assert identities[0].isdisjoint(identities[1])
    assert identities[0].isdisjoint(identities[2])
    assert identities[1].isdisjoint(identities[2])


def test_dataloader_batches_named_tensors(prepared: Path) -> None:
    batch = next(iter(DataLoader(NDWSDataset(prepared), batch_size=1)))
    assert batch["input"].shape == (1, 12, 64, 64)
    assert batch["target"].shape == (1, 1, 64, 64)
    assert batch["validity_mask"].shape == (1, 1, 64, 64)
    assert batch["sample_id"] == ["train/unique-0"]


def test_plot_creates_output_parent(prepared: Path, tmp_path: Path) -> None:
    output = tmp_path / "new-parent" / "sample.png"
    plot_sample(NDWSDataset(prepared)[0], str(output))
    assert output.is_file()


def test_download_and_extract_happy_path(tmp_path: Path) -> None:
    source_archive = tmp_path / "source.zip"
    filename = "next_day_wildfire_spread_train_00.tfrecord"
    with zipfile.ZipFile(source_archive, "w") as bundle:
        bundle.writestr(f"nested/{filename}", b"representative tfrecord bytes")
    module = load_script("download_data")

    extracted = module.download_and_extract(
        tmp_path / "raw", tmp_path / "download.zip", source_archive.as_uri()
    )

    assert [path.name for path in extracted] == [filename]
    assert extracted[0].read_bytes() == b"representative tfrecord bytes"


def test_preparation_preserves_invalid_raw_target(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    module = load_script("prepare_data")
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "next_day_wildfire_spread_train_00.tfrecord").touch()
    record = {name: np.zeros(4096, dtype=np.float32) for name in CHANNEL_NAMES}
    record["FireMask"] = np.zeros(4096, dtype=np.float32)
    record["FireMask"][7] = -1
    monkeypatch.setattr(module, "tfrecord_loader", lambda *args: iter([record]))

    module.prepare_split(raw, tmp_path / "prepared", "train", "train", 1)
    with np.load(tmp_path / "prepared/train/sample_00000.npz") as sample:
        assert sample["raw_target"].reshape(-1)[7] == -1
        assert not sample["validity_mask"].reshape(-1)[7]
        assert sample["target"].reshape(-1)[7] == 0
