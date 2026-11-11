
import argparse
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import ASTForAudioClassification

from machine_learning.training.ast_dataset import MODEL_NAME
from machine_learning.training.ast_rir_fusion import ASTRIRFusionModel
from machine_learning.training.fusion_dataset import ASTRIRFusionDataset
from machine_learning.inference.predict import (
    CHECKPOINT_PATH,
    ID2LABEL,
    LABEL2ID,
)


# -----------------------------
# Configuration
# -----------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

TRAIN_CSV = PROJECT_ROOT / "data" / "processed" / "fusion_train_pairs.csv"
DEV_CSV = PROJECT_ROOT / "data" / "processed" / "fusion_dev_pairs.csv"

CHECKPOINT_DIR = PROJECT_ROOT / "machine_learning" / "checkpoints"
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

BEST_MODEL_PATH = CHECKPOINT_DIR / "ast_rir_fusion_best.pt"
TRIAL_MODEL_PATH = CHECKPOINT_DIR / "ast_rir_fusion_trial_best.pt"

BATCH_SIZE = 4
EPOCHS = 3
LEARNING_RATE = 1e-4
GRADIENT_ACCUMULATION_STEPS = 4

TRIAL_TRAIN_PER_CLASS = 64
TRIAL_DEV_PER_CLASS = 32


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
# Dataset selection
# -----------------------------

def select_balanced_records(dataset, samples_per_class):
    selected = []

    for class_id in range(len(LABEL2ID)):
        class_records = [
            record for record in dataset.records
            if int(record["label"]) == class_id
        ]

        if len(class_records) < samples_per_class:
            raise ValueError(
                f"Not enough samples for class {class_id}."
            )

        selected.extend(
            random.sample(class_records, samples_per_class)
        )

    random.shuffle(selected)
    dataset.records = selected


# -----------------------------
# Class weights
# -----------------------------

def calculate_class_weights(dataset, device):
    labels = [
        int(record["label"])
        for record in dataset.records
    ]

    class_counts = np.bincount(
        labels,
        minlength=len(LABEL2ID),
    )

    if (class_counts == 0).any():
        raise ValueError(
            "Training dataset must contain both bonafide and spoof samples."
        )

    total_samples = len(labels)
    number_of_classes = len(LABEL2ID)

    weights = total_samples / (
        number_of_classes * class_counts
    )

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

    with torch.inference_mode():

        for batch in dataloader:

            input_values = batch["input_values"].to(device)
            acoustic_features = batch["acoustic_features"].to(device)
            labels = batch["label"].to(device)

            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=(device.type == "cuda"),
            ):
                logits = model(
                    input_values,
                    acoustic_features,
                )

                loss = loss_function(logits, labels)

            total_loss += loss.item() * labels.size(0)

            predictions = torch.argmax(logits, dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    return total_loss / total, correct / total


# -----------------------------
# Main
# -----------------------------

def main():

    parser = argparse.ArgumentParser(
        description="Train the AST-RIR fusion model."
    )

    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run a small fusion training test.",
    )

    parser.add_argument(
        "--trial",
        action="store_true",
        help="Train using a small balanced subset.",
    )

    parser.add_argument(
        "--train",
        action="store_true",
        help="Train using the full fusion dataset.",
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

    best_model_path = (
        TRIAL_MODEL_PATH if args.trial else BEST_MODEL_PATH
    )

    # -----------------------------
    # Create datasets
    # -----------------------------

    print("\nLoading fusion datasets...")

    train_dataset = ASTRIRFusionDataset(TRAIN_CSV)
    dev_dataset = ASTRIRFusionDataset(DEV_CSV)

    if args.smoke_test:
        select_balanced_records(train_dataset, 2)
        select_balanced_records(dev_dataset, 2)

        print("\nSMOKE TEST MODE")

    elif args.trial:
        select_balanced_records(
            train_dataset,
            TRIAL_TRAIN_PER_CLASS,
        )

        select_balanced_records(
            dev_dataset,
            TRIAL_DEV_PER_CLASS,
        )

        print("\nTRIAL MODE")

    else:
        print("\nFULL DATASET MODE")

    print("Training samples:", len(train_dataset))
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
    # Load fine-tuned AST checkpoint
    # -----------------------------

    print("\nLoading fine-tuned AST model...")

    if not CHECKPOINT_PATH.is_file():
        raise FileNotFoundError(
            f"AST checkpoint not found: {CHECKPOINT_PATH}"
        )

    ast_model = ASTForAudioClassification.from_pretrained(
        MODEL_NAME,
        num_labels=2,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
        ignore_mismatched_sizes=True,
    )

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=device,
        weights_only=True,
    )

    ast_model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    # -----------------------------
    # Create fusion model
    # -----------------------------

    print("\nCreating AST-RIR fusion model...")

    model = ASTRIRFusionModel(ast_model).to(device)

    # Freeze AST backbone.
    for parameter in model.ast_backbone.parameters():
        parameter.requires_grad = False

    # Only train the fusion classifier.
    optimizer = torch.optim.AdamW(
        model.fusion_classifier.parameters(),
        lr=LEARNING_RATE,
    )

    loss_function = nn.CrossEntropyLoss(
        weight=class_weights
    )

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=(device.type == "cuda"),
    )

    print("AST backbone: Frozen")
    print("Fusion classifier: Trainable")

    # -----------------------------
    # Smoke test
    # -----------------------------

    if args.smoke_test:

        model.train()
        model.ast_backbone.eval()

        batch = next(iter(train_loader))

        input_values = batch["input_values"].to(device)
        acoustic_features = batch["acoustic_features"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=(device.type == "cuda"),
        ):
            logits = model(input_values, acoustic_features)
            loss = loss_function(logits, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        assert logits.shape == (len(labels), 2)
        assert torch.isfinite(logits).all()

        print("\nFusion smoke test passed.")
        print("Input shape:", tuple(input_values.shape))
        print("Acoustic shape:", tuple(acoustic_features.shape))
        print("Output shape:", tuple(logits.shape))
        print("Loss:", round(loss.item(), 4))

        return

    # -----------------------------
    # Training
    # -----------------------------

    best_validation_loss = float("inf")

    for epoch in range(EPOCHS):

        model.train()

        # Keep the frozen AST backbone in evaluation mode.
        model.ast_backbone.eval()

        running_loss = 0.0
        total = 0

        optimizer.zero_grad(set_to_none=True)

        print(f"\nEpoch {epoch + 1}/{EPOCHS}")

        for batch_index, batch in enumerate(train_loader):

            input_values = batch["input_values"].to(device)
            acoustic_features = batch["acoustic_features"].to(device)
            labels = batch["label"].to(device)

            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=(device.type == "cuda"),
            ):
                logits = model(
                    input_values,
                    acoustic_features,
                )

                loss = loss_function(logits, labels)

                scaled_loss = (
                    loss / GRADIENT_ACCUMULATION_STEPS
                )

            scaler.scale(scaled_loss).backward()

            if (
                (batch_index + 1) % GRADIENT_ACCUMULATION_STEPS == 0
                or (batch_index + 1) == len(train_loader)
            ):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

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
        # Save best fusion classifier
        # -----------------------------

        if validation_loss < best_validation_loss:

            best_validation_loss = validation_loss

            torch.save(
                {
                    "epoch": epoch + 1,
                    "fusion_classifier_state_dict": (
                        model.fusion_classifier.state_dict()
                    ),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "validation_loss": validation_loss,
                    "validation_accuracy": validation_accuracy,
                    "class_weights": class_weights.cpu(),
                    "label2id": LABEL2ID,
                    "id2label": ID2LABEL,
                    "ast_checkpoint": str(CHECKPOINT_PATH),
                    "ast_backbone_frozen": True,
                },
                best_model_path,
            )

            print("Best fusion checkpoint saved.")

    print("\nFusion training completed.")
    print("Best checkpoint:", best_model_path)

    if device.type == "cuda":
        print(
            "Peak GPU memory:",
            round(
                torch.cuda.max_memory_allocated() / (1024 ** 3),
                2,
            ),
            "GB",
        )


if __name__ == "__main__":
    main()
