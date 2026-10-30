import torch
from torch.utils.data import DataLoader

from audio_analysis.dataset_loader import (
    load_training_dataset,
    load_validation_dataset,
)
from machine_learning.models.cnn import AudioCNN
from machine_learning.training.dataset import AudioDataset


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    train_df = load_training_dataset().head(8)
    val_df = load_validation_dataset().head(8)

    train_loader = DataLoader(
        AudioDataset(train_df), batch_size=4
    )
    val_loader = DataLoader(
        AudioDataset(val_df), batch_size=4
    )

    model = AudioCNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.CrossEntropyLoss()

    # Train on 8 recordings.
    model.train()
    for features, labels in train_loader:
        features = features.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        loss = criterion(model(features), labels)
        loss.backward()
        optimizer.step()

    print("Training batches: PASS")

    # Validate on 8 recordings.
    model.eval()
    with torch.no_grad():
        for features, labels in val_loader:
            features = features.to(device)
            outputs = model(features)
            print("Validation output shape:", tuple(outputs.shape))

    print("SMALL-SCALE TRAINING AND VALIDATION: PASS")


if __name__ == "__main__":
    main()