import torch
from torch import nn
from torch.utils.data import DataLoader

from wildfire_ml.data import CHANNEL_NAMES, NDWSDataset
from wildfire_ml.model import TinyWildfireNet
from wildfire_ml.visualize import (
    plot_error_map,
    plot_prediction,
)


BATCH_SIZE = 4
EPOCHS = 10
LEARNING_RATE = 0.001


def calculate_normalization(dataset):
    """Calculate per-channel statistics using training data only."""

    inputs = torch.stack(
        [dataset[i]["input"] for i in range(len(dataset))]
    )

    means = inputs.mean(dim=(0, 2, 3))
    stds = inputs.std(dim=(0, 2, 3))

    # PrevFireMask has meaningful values -1, 0, 1.
    # Leave it unchanged.
    means[11] = 0.0
    stds[11] = 1.0

    # Reshape for broadcasting across [N, C, H, W].
    means = means.view(1, 12, 1, 1)
    stds = stds.view(1, 12, 1, 1)

    return means, stds


def normalize(inputs, means, stds):
    return (inputs - means) / stds

def calculate_pos_weight(dataset):
    positive_pixels = 0
    negative_pixels = 0

    for i in range(len(dataset)):
        sample = dataset[i]

        target = sample["target"]
        valid = sample["validity_mask"]

        fire = target >= 0.5

        positive_pixels += (fire & valid).sum().item()
        negative_pixels += (~fire & valid).sum().item()

    pos_weight = negative_pixels / positive_pixels

    print("\nClass balance:")
    print("positive fire pixels:", positive_pixels)
    print("negative pixels:     ", negative_pixels)
    print(f"positive weight:      {pos_weight:.2f}")

    return pos_weight

def main():
    # Reproducible random initialization / shuffling.
    torch.manual_seed(42)

    # ---------------------------------------
    # DATA
    # ---------------------------------------

    train_dataset = NDWSDataset(
        "data/prepared",
        split="train",
    )

    validation_dataset = NDWSDataset(
        "data/prepared",
        split="validation",
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    means, stds = calculate_normalization(train_dataset)

    pos_weight = calculate_pos_weight(train_dataset)

    print(f"training samples:   {len(train_dataset)}")
    print(f"validation samples: {len(validation_dataset)}")
    print(f"batches per epoch:  {len(train_loader)}")

    print("\nNormalization statistics:")

    for i, name in enumerate(CHANNEL_NAMES):
        print(
            f"{name:15s}",
            f"mean={means[0, i, 0, 0].item():9.3f}",
            f"std={stds[0, i, 0, 0].item():9.3f}",
        )

    # ---------------------------------------
    # MODEL
    # ---------------------------------------

    pos_weight = calculate_pos_weight(train_dataset)

    model = TinyWildfireNet()

    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(pos_weight)
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
    )

    # ---------------------------------------
    # TRAINING
    # ---------------------------------------

    print("\nTraining:")

    for epoch in range(EPOCHS):
        total_loss = 0.0

        for batch in train_loader:
            inputs = batch["input"]
            targets = batch["target"]
            valid = batch["validity_mask"]

            inputs = normalize(
                inputs,
                means,
                stds,
            )

            # 1. Remove gradients left over from previous batch.
            optimizer.zero_grad()

            # 2. Forward pass.
            logits = model(inputs)

            # 3. Compare prediction with reality.
            loss = criterion(
                logits[valid],
                targets[valid],
            )

            # 4. Calculate gradients.
            loss.backward()

            # 5. Move the weights.
            optimizer.step()

            total_loss += loss.item()

        average_loss = total_loss / len(train_loader)

        print(
            f"epoch {epoch + 1:2d}/{EPOCHS}",
            f"loss={average_loss:.6f}",
        )



    print("\n--- TRAINING SET ---")

    evaluate(
        model,
        train_loader,
        means,
        stds,
        threshold=0.9,
    )

    print("\n--- VALIDATION SET ---")

    evaluate(
        model,
        validation_loader,
        means,
        stds,
        threshold=0.9,
    )

    evaluate_persistence(validation_loader)

    print("\nSaving validation prediction plots...")

    plots_saved = 0

    for i in range(len(validation_dataset)):
        sample = validation_dataset[i]

        actual_fire_pixels = (
            (
                sample["target"] >= 0.5
            )
            & sample["validity_mask"]
        ).sum().item()

        # Skip boring scenes with almost no observed fire.
        if actual_fire_pixels < 10:
            continue

        output = (
            f"outputs/validation_prediction_{i:04d}.png"
        )

        plot_prediction(
            model,
            sample,
            means,
            stds,
            threshold=0.9,
            output=output,
        )

        error_output = (
            f"outputs/validation_error_{i:04d}.png"
        )

        plot_error_map(
            model,
            sample,
            means,
            stds,
            threshold=0.9,
            output=error_output,
        )

        print(
            output,
            f"({actual_fire_pixels} actual fire pixels)",
        )

        plots_saved += 1

        if plots_saved == 3:
            break

def evaluate(model, loader, means, stds, threshold=0.5):
    model.eval()

    true_positive = 0
    false_positive = 0
    false_negative = 0
    true_negative = 0

    predicted_fire = 0
    actual_fire = 0
    valid_pixels = 0

    with torch.no_grad():
        for batch in loader:
            inputs = normalize(
                batch["input"],
                means,
                stds,
            )

            targets = batch["target"]
            valid = batch["validity_mask"]

            logits = model(inputs)
            probabilities = torch.sigmoid(logits)

            predictions = probabilities >= threshold
            actual = targets >= threshold

            predictions = predictions[valid]
            actual = actual[valid]

            true_positive += (predictions & actual).sum().item()
            false_positive += (predictions & ~actual).sum().item()
            false_negative += (~predictions & actual).sum().item()
            true_negative += (~predictions & ~actual).sum().item()

            predicted_fire += predictions.sum().item()
            actual_fire += actual.sum().item()
            valid_pixels += actual.numel()

    precision = (
        true_positive / (true_positive + false_positive)
        if true_positive + false_positive > 0
        else 0.0
    )

    recall = (
        true_positive / (true_positive + false_negative)
        if true_positive + false_negative > 0
        else 0.0
    )

    iou = (
        true_positive
        / (true_positive + false_positive + false_negative)
        if true_positive + false_positive + false_negative > 0
        else 0.0
    )

    accuracy = (
        (true_positive + true_negative) / valid_pixels
        if valid_pixels > 0
        else 0.0
    )

    print(f"\nEvaluation at threshold {threshold:.2f}:")
    print("valid pixels:       ", valid_pixels)
    print("actual fire pixels: ", actual_fire)
    print("predicted fire:     ", predicted_fire)

    print()
    print("true positive:      ", true_positive)
    print("false positive:     ", false_positive)
    print("false negative:     ", false_negative)
    print("true negative:      ", true_negative)

    print()
    print(f"precision:           {precision:.4f}")
    print(f"recall:              {recall:.4f}")
    print(f"IoU:                 {iou:.4f}")
    print(f"pixel accuracy:      {accuracy:.4f}")

def evaluate_persistence(loader):
    true_positive = 0
    false_positive = 0
    false_negative = 0
    true_negative = 0

    with torch.no_grad():
        for batch in loader:
            # Channel 11 is PrevFireMask.
            predictions = batch["input"][:, 11:12] >= 0.5
            actual = batch["target"] >= 0.5
            valid = batch["validity_mask"]

            predictions = predictions[valid]
            actual = actual[valid]

            true_positive += (predictions & actual).sum().item()
            false_positive += (predictions & ~actual).sum().item()
            false_negative += (~predictions & actual).sum().item()
            true_negative += (~predictions & ~actual).sum().item()

    precision = (
        true_positive / (true_positive + false_positive)
        if true_positive + false_positive > 0
        else 0.0
    )

    recall = (
        true_positive / (true_positive + false_negative)
        if true_positive + false_negative > 0
        else 0.0
    )

    iou = (
        true_positive /
        (true_positive + false_positive + false_negative)
        if true_positive + false_positive + false_negative > 0
        else 0.0
    )

    print("\n--- PERSISTENCE BASELINE ---")
    print("true positive: ", true_positive)
    print("false positive:", false_positive)
    print("false negative:", false_negative)
    print(f"precision:      {precision:.4f}")
    print(f"recall:         {recall:.4f}")
    print(f"IoU:            {iou:.4f}")

if __name__ == "__main__":
    main()