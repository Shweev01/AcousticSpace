
from pathlib import Path

import torch
import torch.nn as nn

from machine_learning.models.cnn import AudioCNN
from machine_learning.training.data_loader import create_data_loaders


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CHECKPOINT_DIR = PROJECT_ROOT / "machine_learning" / "checkpoints"


def run_epoch(model, data_loader, criterion, device, optimizer=None):
    """Run one training or validation epoch."""

    is_training = optimizer is not None
    model.train() if is_training else model.eval()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    context = torch.enable_grad() if is_training else torch.no_grad()

    with context:
        for features, labels in data_loader:
            features = features.to(device)
            labels = labels.to(device)

            if is_training:
                optimizer.zero_grad(set_to_none=True)

            logits = model(features)
            loss = criterion(logits, labels)

            if is_training:
                loss.backward()
                optimizer.step()

            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size
            total_correct += (logits.argmax(dim=1) == labels).sum().item()
            total_samples += batch_size

    return {
        "loss": total_loss / total_samples,
        "accuracy": total_correct / total_samples,
    }


def train_model(epochs=3, batch_size=32, learning_rate=1e-3):
    """Train the baseline CNN and save its best validation checkpoint."""

    if epochs < 1:
        raise ValueError("epochs must be at least 1.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    train_loader, val_loader = create_data_loaders(batch_size=batch_size)

    model = AudioCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    checkpoint_path = CHECKPOINT_DIR / "baseline_cnn_best.pt"
    best_val_loss = float("inf")

    for epoch in range(1, epochs + 1):
        train_metrics = run_epoch(
            model, train_loader, criterion, device, optimizer
        )
        val_metrics = run_epoch(
            model, val_loader, criterion, device
        )

        print(
            f"Epoch {epoch}/{epochs} | "
            f"Train loss: {train_metrics['loss']:.4f} | "
            f"Train accuracy: {train_metrics['accuracy']:.4f} | "
            f"Val loss: {val_metrics['loss']:.4f} | "
            f"Val accuracy: {val_metrics['accuracy']:.4f}"
        )

        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "validation_loss": best_val_loss,
                },
                checkpoint_path,
            )
            print(f"Saved best checkpoint: {checkpoint_path}")

    return model


if __name__ == "__main__":
    train_model()
