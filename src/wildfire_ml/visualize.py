"""One deliberately small NDWS inspection plot."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from .data import CHANNEL_NAMES


def plot_sample(sample: dict[str, object], output: str | None = None) -> None:
    x = sample["input"].numpy()  # type: ignore[union-attr]
    target = sample["target"].numpy()[0]  # type: ignore[union-attr]
    valid = sample["validity_mask"].numpy()[0]  # type: ignore[union-attr]
    selections = ("elevation", "NDVI", "erc", "PrevFireMask")
    fig, axes = plt.subplots(2, 3, figsize=(11, 7))
    for axis, name in zip(axes.flat, selections):
        image = axis.imshow(x[CHANNEL_NAMES.index(name)], cmap="viridis")
        axis.set_title(name)
        fig.colorbar(image, ax=axis, shrink=0.7)
    axes.flat[4].imshow(np.ma.masked_where(~valid, target), vmin=0, vmax=1, cmap="Reds")
    axes.flat[4].set_title("next-day FireMask")
    axes.flat[5].imshow(valid, vmin=0, vmax=1, cmap="gray")
    axes.flat[5].set_title("valid target pixels")
    for axis in axes.flat:
        axis.set_axis_off()
    fig.tight_layout()
    if output:
        fig.savefig(output, dpi=150)
    else:
        plt.show()
