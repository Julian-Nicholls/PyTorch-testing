"""One deliberately small NDWS inspection plot."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import Patch

from .data import CHANNEL_NAMES

def plot_prediction(
    model,
    sample,
    means,
    stds,
    threshold=0.9,
    output=None,
):
    """Compare persistence, CNN probability, and next-day fire truth."""

    import torch

    inputs = sample["input"]
    target = sample["target"][0]
    valid = sample["validity_mask"][0]

    # [12, 64, 64] -> [1, 12, 64, 64]
    model_input = inputs.unsqueeze(0)

    # Use training-set normalization statistics.
    model_input = (model_input - means) / stds

    model.eval()

    with torch.no_grad():
        logits = model(model_input)
        probabilities = torch.sigmoid(logits)[0, 0]

    current_fire = inputs[CHANNEL_NAMES.index("PrevFireMask")]
    prediction = probabilities >= threshold

    # Convert to NumPy for matplotlib.
    current_fire = current_fire.numpy()
    target = target.numpy()
    valid = valid.numpy()
    probabilities = probabilities.numpy()
    prediction = prediction.numpy()

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(9, 9),
    )

    axes[0, 0].imshow(
        current_fire,
        vmin=0,
        vmax=1,
        cmap="Reds",
    )
    axes[0, 0].set_title(
        "Current fire / persistence baseline"
    )

    axes[0, 1].imshow(
        np.ma.masked_where(~valid, target),
        vmin=0,
        vmax=1,
        cmap="Reds",
    )
    axes[0, 1].set_title(
        "Actual next-day fire"
    )

    probability_image = axes[1, 0].imshow(
        probabilities,
        vmin=0,
        vmax=1,
        cmap="magma",
    )
    axes[1, 0].set_title(
        "CNN fire probability"
    )

    fig.colorbar(
        probability_image,
        ax=axes[1, 0],
        shrink=0.7,
    )

    axes[1, 1].imshow(
        prediction,
        vmin=0,
        vmax=1,
        cmap="Reds",
    )
    axes[1, 1].set_title(
        f"CNN prediction (threshold={threshold:.2f})"
    )

    for axis in axes.flat:
        axis.set_axis_off()

    fig.tight_layout()

    if output:
        Path(output).parent.mkdir(
            parents=True,
            exist_ok=True,
        )
        fig.savefig(
            output,
            dpi=150,
        )
        plt.close(fig)
    else:
        plt.show()

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
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, dpi=150)
    else:
        plt.show()

def plot_error_map(
    model,
    sample,
    means,
    stds,
    threshold=0.9,
    output=None,
):
    """Show where a prediction is right and wrong spatially."""

    import torch

    inputs = sample["input"]
    target = sample["target"][0]
    valid = sample["validity_mask"][0]

    model_input = inputs.unsqueeze(0)
    model_input = (model_input - means) / stds

    model.eval()

    with torch.no_grad():
        logits = model(model_input)
        probabilities = torch.sigmoid(logits)[0, 0]

    prediction = probabilities >= threshold
    actual = target >= 0.5

    # Only evaluate pixels with trustworthy target data.
    prediction = prediction.numpy()
    actual = actual.numpy()
    valid = valid.numpy()

    # Integer code for each kind of outcome:
    #
    # 0 = true negative
    # 1 = true positive
    # 2 = false positive
    # 3 = false negative
    # 4 = invalid / unknown target
    errors = np.full(
        actual.shape,
        4,
        dtype=np.uint8,
    )

    errors[
        valid & ~prediction & ~actual
    ] = 0

    errors[
        valid & prediction & actual
    ] = 1

    errors[
        valid & prediction & ~actual
    ] = 2

    errors[
        valid & ~prediction & actual
    ] = 3

    # Deliberately categorical rather than continuous.
    colors = [
        "#eeeeee",  # true negative
        "#2ca02c",  # true positive
        "#ff9900",  # false positive
        "#d62728",  # false negative
        "#777777",  # invalid
    ]

    cmap = ListedColormap(colors)

    norm = BoundaryNorm(
        boundaries=[-0.5, 0.5, 1.5, 2.5, 3.5, 4.5],
        ncolors=len(colors),
    )

    fig, axis = plt.subplots(
        figsize=(8, 8),
    )

    axis.imshow(
        errors,
        cmap=cmap,
        norm=norm,
        interpolation="nearest",
    )

    axis.set_title(
        f"CNN error map (threshold={threshold:.2f})"
    )

    axis.set_axis_off()

    legend = [
        Patch(
            facecolor=colors[1],
            label="True positive",
        ),
        Patch(
            facecolor=colors[2],
            label="False positive",
        ),
        Patch(
            facecolor=colors[3],
            label="False negative",
        ),
        Patch(
            facecolor=colors[0],
            label="True negative",
        ),
        Patch(
            facecolor=colors[4],
            label="Invalid target",
        ),
    ]

    axis.legend(
        handles=legend,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.12),
        ncol=3,
    )

    fig.tight_layout()

    if output:
        Path(output).parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        fig.savefig(
            output,
            dpi=150,
            bbox_inches="tight",
        )

        plt.close(fig)
    else:
        plt.show()