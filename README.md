# PyTorch Next Day Wildfire Spread learning project

This repository removes the data-plumbing friction from a small educational
PyTorch experiment: **given present-day wildfire and environmental raster
features, predict the observed next-day fire footprint**. It intentionally has
no model, loss, optimizer, training/validation loop, metrics, checkpointing, or
inference pipeline. Those are the learning exercise.

## Dataset and provenance

[Next Day Wildfire Spread (NDWS)](https://arxiv.org/abs/2112.02447) was released
with the paper *Next Day Wildfire Spread: A Machine Learning Dataset to Predict
Wildfire Spreading from Remote-Sensing Data*. This project downloads the
authors' **canonical Kaggle TFRecord distribution** from the
[NDWS Kaggle release](https://www.kaggle.com/datasets/fantineh/next-day-wildfire-spread),
not a third-party array conversion. This is the distribution linked by the
official Google Research README. The corresponding reference implementation is in
[Google Research](https://github.com/google-research/google-research/tree/master/simulation_research/next_day_wildfire_spread).

Preparation converts records to compressed NumPy NPZ files so TensorFlow is not
needed. The lightweight `tfrecord` reader is used only for decoding. Every local
sample contains:

| key | shape | meaning |
| --- | --- | --- |
| `input` | `[12, 64, 64]` float32 | ordered present-day feature planes |
| `target` | `[1, 64, 64]` float32 | next-day `FireMask`, binary where valid |
| `validity_mask` | `[1, 64, 64]` bool | whether target may enter loss/metrics |
| `raw_target` | `[1, 64, 64]` float32 | untouched source target, for auditing |

The feature order exactly follows Google Research's canonical `INPUT_FEATURES`:
`elevation`, `pdsi`, `NDVI`, `pr`, `sph`, `th` (wind direction), `tmmn`, `tmmx`,
`vs` (wind speed), `erc`, `population`, and `PrevFireMask`. NDWS encodes uncertain/unobserved fire
labels as `-1`, rather than definite no-fire (`0`). Preparation retains that in
`raw_target`, places a harmless `0` placeholder in `target`, and sets
`validity_mask=False`. Any future loss and metric **must** mask those cells.

No normalization is performed, so there is no risk of leaking validation or
test statistics. The original train/eval/test membership is retained (`eval`
is named `validation` locally), and records are selected in sorted-shard,
in-file order rather than randomly.

## Setup

Python 3.11 or newer is supported. From a clean checkout:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

PyTorch wheels vary by operating system and accelerator. The command above is
appropriate for a CPU learning run; use the
[official PyTorch installer](https://pytorch.org/get-started/locally/) first if
you require a platform-specific build.

## Acquire and prepare data

The workflow downloads and extracts the canonical Kaggle archive, then writes
the first 64 records from each original split—a manageable repeatable subset
for CPU experiments:

```bash
python scripts/download_data.py
python scripts/prepare_data.py --samples-per-split 64
```

The archive retains every original shard; choose any positive subset size during
preparation. Raw, archived, and prepared data are ignored by Git.

The Kaggle release is public, but automated access can still be affected by
Kaggle authentication, rate limits, or network policy. If it is blocked, use
the Kaggle page above to download the dataset archive and run:

```bash
python scripts/download_data.py --archive /path/to/archive.zip
python scripts/prepare_data.py --samples-per-split 64
```

The downloader extracts TFRecords from the archive and checks that at least one
exists. The download/extraction happy path is covered by an offline test using a
representative archive, while the URL is the canonical Kaggle API endpoint.

## Inspect a real sample

Print channel names, tensor shapes, per-channel ranges/non-finite counts, target
validity, and create a small plot:

```bash
python scripts/inspect_data.py --output artifacts/sample.png
```

Omit `--output` for an interactive matplotlib window. The plot shows selected
environmental channels, current-day fire (`PrevFireMask`), observed next-day
fire, and valid target coverage.

The PyTorch boundary is deliberately ordinary and readable:

```python
from torch.utils.data import DataLoader
from wildfire_ml.data import NDWSDataset

dataset = NDWSDataset("data/prepared", split="train")
sample = dataset[0]
batch = next(iter(DataLoader(dataset, batch_size=4, shuffle=True)))
print(sample["input"].shape)  # torch.Size([12, 64, 64])
print(batch["target"].shape)  # torch.Size([4, 1, 64, 64])
```

## Tests

Tests use generated tiny fixtures and never download NDWS:

```bash
pytest
```

They cover channel ordering, shapes, binary/invalid target semantics, disjoint
split identities, preparation of `-1` observations, and DataLoader batching.

## Scientific limitations

- This is an educational experiment, not an operational forecasting system.
- Satellite-derived NDWS target observations have uncertainty and missingness;
  they are not perfect ground truth.
- The published split is preserved, but it should not be assumed to establish
  generalization to unseen geographies, fires, climates, or future conditions.
- A successful classroom metric does not demonstrate operational wildfire
  forecasting ability or justify safety-critical decisions.
- Spatial augmentation is intentionally absent. Flipping or rotating rasters
  without physically consistent changes to directional variables such as wind
  would alter their meaning.
