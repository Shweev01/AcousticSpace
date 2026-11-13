
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

OUTPUT_DIR = PROJECT_ROOT / "machine_learning" / "checkpoints"

BATCH_SIZE = 4
EPOCHS = 3
LEARNING_RATE = 1e-4
GRADIENT_ACCUMULATION_STEPS = 4


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

def evaluate(model, loader, loss_function, device):
    model.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    with torch.inference_mode():

        for batch in loader:
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
# Ablation training
# -----------------------------

def train_ablation(feature_name, feature_index):

    print("\n" + "=" * 55)
    print(f"Training AST + {feature_name}")
    print("=" * 55)

    set_seed(42)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("Device:", device)

    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))

    # -----------------------------
    # Create datasets
    # -----------------------------

    print("\nLoading datasets...")

    train_dataset = ASTRIRFusionDataset(
        TRAIN_CSV,
        feature_indices=(feature_index,),
    )

    dev_dataset = ASTRIRFusionDataset(
        DEV_CSV,
        feature_indices=(feature_index,),
    )

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
    # Load fine-tuned AST model
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

    print("\nCreating ablation model...")

    model = ASTRIRFusionModel(
        ast_model,
        acoustic_feature_size=1,
    ).to(device)

    # Freeze AST backbone.
    for parameter in model.ast_backbone.parameters():
        parameter.requires_grad = False

    # Train only fusion classifier.
    optimizer = torch.optim.AdamW(
        model.fusion_classifier.parameters(),
        lr=LEARNING_RATE,
    )

    loss_function = nn.CrossEntropyLoss(
        weight=class_weights,
    )

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=(device.type == "cuda"),
    )

    print("AST backbone: Frozen")
    print("Ablation classifier: Trainable")

    # -----------------------------
    # Checkpoint configuration
    # -----------------------------

    best_validation_loss = float("inf")

    checkpoint_path = (
        OUTPUT_DIR / f"ast_rir_{feature_name.lower()}_best.pt"
    )

    # -----------------------------
    # Training
    # -----------------------------

    for epoch in range(EPOCHS):

        model.train()

        # Keep frozen AST backbone in evaluation mode.
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
            loss_function,
            device,
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
                    "fusion_classifier_state_dict": (
                        model.fusion_classifier.state_dict()
                    ),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "validation_loss": validation_loss,
                    "validation_accuracy": validation_accuracy,
                    "class_weights": class_weights.cpu(),
                    "feature_name": feature_name,
                    "feature_index": feature_index,
                    "acoustic_feature_size": 1,
                    "ast_checkpoint": str(CHECKPOINT_PATH),
                    "frozen_ast_backbone": True,
                    "label2id": LABEL2ID,
                    "id2label": ID2LABEL,
                },
                checkpoint_path,
            )

            print("Best checkpoint saved:", checkpoint_path)

    print("\n" + "=" * 55)
    print(f"{feature_name} ABLATION TRAINING COMPLETED")
    print("=" * 55)

    print("Best validation loss:", round(best_validation_loss, 4))
    print("Checkpoint:", checkpoint_path)

    if device.type == "cuda":
        print(
            "Peak GPU memory:",
            round(
                torch.cuda.max_memory_allocated() / (1024 ** 3),
                2,
            ),
            "GB",
        )


# -----------------------------
# Main
# -----------------------------

def main():

    parser = argparse.ArgumentParser(
        description="Train AST acoustic-feature ablation models."
    )

    parser.add_argument(
        "--feature",
        choices=["rt60", "drr", "both"],
        default="both",
        help="Which acoustic feature to train with.",
    )

    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if args.feature == "rt60":
        train_ablation("RT60", 0)

    elif args.feature == "drr":
        train_ablation("DRR", 1)

    else:
        train_ablation("RT60", 0)
        train_ablation("DRR", 1)


if __name__ == "__main__":
    main()
