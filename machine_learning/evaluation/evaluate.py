
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    recall_score,
)

from machine_learning.models.cnn import AudioCNN
from machine_learning.training.dataset import AudioDataset
from audio_analysis.dataset_loader import load_validation_dataset


PROJECT_ROOT = Path(__file__).resolve().parents[2]

CHECKPOINT_PATH = (
    PROJECT_ROOT / "machine_learning" / "checkpoints" / "baseline_cnn_best.pt"
)

RESULTS_DIR = (
    PROJECT_ROOT / "machine_learning" / "evaluation" / "results"
)

THRESHOLD = 0.30


def evaluate_model(batch_size=32):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Evaluation device: {device}")

    if not CHECKPOINT_PATH.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {CHECKPOINT_PATH}")

    # Load the LA development dataset.
    validation_df = load_validation_dataset()
    dataset = AudioDataset(validation_df)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )

    # Load the same best CNN checkpoint.
    model = AudioCNN().to(device)

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=device,
        weights_only=True,
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print(f"Checkpoint epoch: {checkpoint['epoch']}")
    print(f"Checkpoint validation loss: {checkpoint['validation_loss']:.4f}")
    print(f"Validation recordings: {len(dataset)}")

    all_labels = []
    all_probabilities = []

    # Run inference on the complete validation dataset.
    with torch.no_grad():

        for features, labels in loader:

            features = features.to(device)

            logits = model(features)
            probabilities = torch.softmax(logits, dim=1).cpu()

            all_labels.extend(labels.tolist())
            all_probabilities.extend(probabilities.tolist())

    # Extract bonafide probabilities.
    bonafide_probabilities = [
        probabilities[0] for probabilities in all_probabilities
    ]

    # Apply the selected threshold.
    # Class 0 = bonafide, Class 1 = spoof.
    all_predictions = [
        0 if probability >= THRESHOLD else 1
        for probability in bonafide_probabilities
    ]

    # Calculate evaluation metrics.
    accuracy = accuracy_score(all_labels, all_predictions)

    macro_f1 = f1_score(
        all_labels,
        all_predictions,
        average="macro",
        zero_division=0,
    )

    bonafide_recall = recall_score(
        all_labels,
        all_predictions,
        pos_label=0,
        zero_division=0,
    )

    spoof_recall = recall_score(
        all_labels,
        all_predictions,
        pos_label=1,
        zero_division=0,
    )

    matrix = confusion_matrix(
        all_labels,
        all_predictions,
        labels=[0, 1],
    )

    print(f"\nSelected threshold: {THRESHOLD}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Macro F1: {macro_f1:.4f}")
    print(f"Bonafide recall: {bonafide_recall:.4f}")
    print(f"Spoof recall: {spoof_recall:.4f}")

    print("\nClassification Report:")
    print(
        classification_report(
            all_labels,
            all_predictions,
            labels=[0, 1],
            target_names=["bonafide", "spoof"],
            zero_division=0,
        )
    )

    print("Confusion Matrix (rows=actual, columns=predicted):")
    print(matrix)

    # Save final predictions.
    results_df = validation_df[["audio_id", "label"]].copy()

    results_df["actual"] = all_labels
    results_df["predicted"] = all_predictions
    results_df["bonafide_probability"] = bonafide_probabilities

    results_df["actual_class"] = results_df["actual"].map(
        {0: "bonafide", 1: "spoof"}
    )

    results_df["predicted_class"] = results_df["predicted"].map(
        {0: "bonafide", 1: "spoof"}
    )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    results_df.to_csv(
        RESULTS_DIR / "final_predictions_threshold_030.csv",
        index=False,
    )

    # Save final metrics.
    metrics_df = pd.DataFrame([{
        "threshold": THRESHOLD,
        "recordings": len(dataset),
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "bonafide_recall": bonafide_recall,
        "spoof_recall": spoof_recall,
    }])

    metrics_df.to_csv(
        RESULTS_DIR / "final_metrics_threshold_030.csv",
        index=False,
    )

    print("\nFinal evaluation files saved in:")
    print(RESULTS_DIR)


if __name__ == "__main__":
    evaluate_model()
