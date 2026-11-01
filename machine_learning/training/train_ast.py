
import argparse
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import ASTForAudioClassification

from machine_learning.training.ast_dataset import (
    create_ast_datasets,
    MODEL_NAME,
)


# -----------------------------
# Configuration
# -----------------------------

BATCH_SIZE = 4
EPOCHS = 3
LEARNING_RATE = 1e-5
GRADIENT_ACCUMULATION_STEPS = 4

TRIAL_TRAIN_PER_CLASS = 64
TRIAL_DEV_PER_CLASS = 32

CHECKPOINT_DIR = Path("machine_learning/checkpoints")
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

TRIAL_MODEL_PATH = CHECKPOINT_DIR / "ast_trial_best.pt"
BEST_MODEL_PATH = CHECKPOINT_DIR / "ast_best.pt"

LABEL2ID = {
    "bonafide": 0,
    "spoof": 1,
}

ID2LABEL = {
    0: "bonafide",
    1: "spoof",
}


# -----------------------------
# Reproducibility
# -----------------------------

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# -----------------------------
# Class weights
# -----------------------------

def calculate_class_weights(train_dataset, device):

    class_counts = (
        train_dataset.dataframe["label"]
        .map(LABEL2ID)
        .value_counts()
        .reindex(range(len(LABEL2ID)), fill_value=0)
    )

    if (class_counts == 0).any():
        raise ValueError(
            "Training dataset must contain both bonafide and spoof samples."
        )

    total_samples = len(train_dataset)
    number_of_classes = len(LABEL2ID)

    weights = [
        total_samples / (number_of_classes * class_counts[class_id])
        for class_id in range(number_of_classes)
    ]

    return torch.tensor(
        weights,
        dtype=torch.float32,
        device=device,
    )


# -----------------------------
# Evaluation
# -----------------------------

def evaluate(model, dataloader, device, loss_function):

    model.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():

        for batch in dataloader:

            input_values = batch["input_values"].to(device)
            labels = batch["labels"].to(device)

            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=(device.type == "cuda"),
            ):
                outputs = model(input_values=input_values)
                loss = loss_function(outputs.logits, labels)

            total_loss += loss.item() * labels.size(0)

            predictions = torch.argmax(outputs.logits, dim=1)

            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    average_loss = total_loss / total
    accuracy = correct / total

    return average_loss, accuracy


# -----------------------------
# Main
# -----------------------------

def main():

    parser = argparse.ArgumentParser(
        description="Train and evaluate the AST deepfake audio classifier."
    )

    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Test AST batching and GPU forward/backward passes only.",
    )

    parser.add_argument(
        "--trial",
        action="store_true",
        help="Train on a small balanced subset.",
    )

    parser.add_argument(
        "--train",
        action="store_true",
        help="Start full AST training.",
    )

    args = parser.parse_args()

    selected_modes = [
        args.smoke_test,
        args.trial,
        args.train,
    ]

    if sum(selected_modes) != 1:
        parser.error(
            "Choose exactly one: --smoke-test, --trial, or --train."
        )

    set_seed()

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("Using device:", device)

    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))

    # Select checkpoint destination
    best_model_path = (
        TRIAL_MODEL_PATH if args.trial else BEST_MODEL_PATH
    )

    # -----------------------------
    # Create datasets
    # -----------------------------

    print("\nCreating AST datasets...")

    train_dataset, dev_dataset = create_ast_datasets()

    if args.trial:

        train_dataset.dataframe = (
            train_dataset.dataframe
            .groupby("label", group_keys=False)
            .sample(
                n=TRIAL_TRAIN_PER_CLASS,
                random_state=42,
            )
            .reset_index(drop=True)
        )

        dev_dataset.dataframe = (
            dev_dataset.dataframe
            .groupby("label", group_keys=False)
            .sample(
                n=TRIAL_DEV_PER_CLASS,
                random_state=42,
            )
            .reset_index(drop=True)
        )

        print(
            "\nTRIAL MODE: Using "
            f"{TRIAL_TRAIN_PER_CLASS} samples per class for training."
        )

        print(
            "Using "
            f"{TRIAL_DEV_PER_CLASS} samples per class for validation."
        )

    else:
        print("\nFULL DATASET MODE")

    print("\nTraining label distribution:")
    print(train_dataset.dataframe["label"].value_counts())

    print("\nValidation label distribution:")
    print(dev_dataset.dataframe["label"].value_counts())

    print("\nTraining samples:", len(train_dataset))
    print("Validation samples:", len(dev_dataset))

    # -----------------------------
    # Class weights
    # -----------------------------

    class_weights = calculate_class_weights(
        train_dataset,
        device,
    )

    print("\nClass weights:")
    print("Bonafide:", class_weights[0].item())
    print("Spoof:", class_weights[1].item())

    # -----------------------------
    # DataLoaders
    # -----------------------------

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )

    dev_loader = DataLoader(
        dev_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )

    # -----------------------------
    # Load AST model
    # -----------------------------

    print("\nLoading AST model...")

    model = ASTForAudioClassification.from_pretrained(
        MODEL_NAME,
        num_labels=2,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
        ignore_mismatched_sizes=True,
    )

    model.to(device)

    loss_function = nn.CrossEntropyLoss(
        weight=class_weights
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
    )

    # -----------------------------
    # Smoke test
    # -----------------------------

    if args.smoke_test:

        print("\nRunning AST smoke test...")

        model.train()
        optimizer.zero_grad()

        for batch_index, batch in enumerate(train_loader):

            input_values = batch["input_values"].to(device)
            labels = batch["labels"].to(device)

            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=(device.type == "cuda"),
            ):
                outputs = model(input_values=input_values)
                loss = loss_function(outputs.logits, labels)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            print(
                f"Batch {batch_index + 1}: "
                f"input shape = {tuple(input_values.shape)}, "
                f"loss = {loss.item():.4f}"
            )

            if batch_index == 1:
                break

        print("\nAST smoke test passed.")
        return

    # -----------------------------
    # Training
    # -----------------------------

    best_validation_loss = float("inf")

    for epoch in range(EPOCHS):

        model.train()

        running_loss = 0.0
        total = 0

        optimizer.zero_grad()

        print(f"\nEpoch {epoch + 1}/{EPOCHS}")

        for batch_index, batch in enumerate(train_loader):

            input_values = batch["input_values"].to(device)
            labels = batch["labels"].to(device)

            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=(device.type == "cuda"),
            ):
                outputs = model(input_values=input_values)

                loss = loss_function(
                    outputs.logits,
                    labels,
                )

                scaled_loss = (
                    loss / GRADIENT_ACCUMULATION_STEPS
                )

            scaled_loss.backward()

            # Update parameters after accumulating gradients
            if (
                (batch_index + 1) % GRADIENT_ACCUMULATION_STEPS == 0
                or (batch_index + 1) == len(train_loader)
            ):
                optimizer.step()
                optimizer.zero_grad()

            running_loss += loss.item() * labels.size(0)
            total += labels.size(0)

            if (batch_index + 1) % 100 == 0:
                print(
                    f"Batch {batch_index + 1}/{len(train_loader)} "
                    f"| Loss: {loss.item():.4f}"
                )

        train_loss = running_loss / total

        # -----------------------------
        # Validation
        # -----------------------------

        validation_loss, validation_accuracy = evaluate(
            model,
            dev_loader,
            device,
            loss_function,
        )

        print(f"\nTraining loss: {train_loss:.4f}")
        print(f"Validation loss: {validation_loss:.4f}")
        print(f"Validation accuracy: {validation_accuracy:.4f}")

        # -----------------------------
        # Save best checkpoint
        # -----------------------------

        if validation_loss < best_validation_loss:

            best_validation_loss = validation_loss

            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "validation_loss": validation_loss,
                    "validation_accuracy": validation_accuracy,
                    "class_weights": class_weights.cpu(),
                    "label2id": LABEL2ID,
                    "id2label": ID2LABEL,
                },
                best_model_path,
            )

            print("Best AST checkpoint saved.")

    print("\nAST training completed.")
    print("Best checkpoint:", best_model_path)

    if device.type == "cuda":
        print(
            "Peak GPU memory:",
            round(torch.cuda.max_memory_allocated() / (1024 ** 3), 2),
            "GB",
        )


if __name__ == "__main__":
    main()
