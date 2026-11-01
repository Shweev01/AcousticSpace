
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from transformers import ASTForAudioClassification

from machine_learning.training.ast_dataset import (
    ASTAudioDataset,
    MODEL_NAME,
)
from audio_analysis.dataset_loader import load_validation_dataset


PROJECT_ROOT = Path(__file__).resolve().parents[2]

CHECKPOINT_PATH = (
    PROJECT_ROOT / "machine_learning" / "checkpoints" / "ast_best.pt"
)

RESULTS_DIR = (
    PROJECT_ROOT / "machine_learning" / "evaluation" / "results"
)

LABEL2ID = {
    "bonafide": 0,
    "spoof": 1,
}

ID2LABEL = {
    0: "bonafide",
    1: "spoof",
}


def evaluate_ast(batch_size=4):

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"Evaluation device: {device}")

    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))

    if not CHECKPOINT_PATH.is_file():
        raise FileNotFoundError(
            f"AST checkpoint not found: {CHECKPOINT_PATH}"
        )

    # Load the complete LA development dataset.
    print("\nLoading AST validation dataset...")

    validation_df = load_validation_dataset()

    from transformers import ASTFeatureExtractor

    feature_extractor = ASTFeatureExtractor.from_pretrained(
        MODEL_NAME
    )

    dataset = ASTAudioDataset(
        validation_df,
        feature_extractor,
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )

    # Load the AST model architecture.
    print("\nLoading AST model...")

    model = ASTForAudioClassification.from_pretrained(
        MODEL_NAME,
        num_labels=2,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
        ignore_mismatched_sizes=True,
    )

    # Load the best fine-tuned checkpoint.
    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=device,
        weights_only=True,
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.to(device)
    model.eval()

    print(f"Checkpoint epoch: {checkpoint['epoch']}")
    print(
        "Checkpoint validation loss: "
        f"{checkpoint['validation_loss']:.4f}"
    )
    print(
        "Checkpoint validation accuracy: "
        f"{checkpoint['validation_accuracy']:.4f}"
    )
    print(f"Validation recordings: {len(dataset)}")

    all_labels = []
    all_predictions = []
    all_probabilities = []

    # Run inference on the complete development dataset.
    print("\nRunning AST evaluation...")

    with torch.no_grad():

        for batch_index, batch in enumerate(loader):

            input_values = batch["input_values"].to(device)
            labels = batch["labels"].to(device)

            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=(device.type == "cuda"),
            ):
                outputs = model(
                    input_values=input_values
                )

            probabilities = torch.softmax(
                outputs.logits.float(),
                dim=1,
            )

            predictions = torch.argmax(
                probabilities,
                dim=1,
            )

            all_labels.extend(labels.cpu().tolist())
            all_predictions.extend(predictions.cpu().tolist())
            all_probabilities.extend(probabilities.cpu().tolist())

            if (batch_index + 1) % 500 == 0:
                print(
                    f"Processed batches: "
                    f"{batch_index + 1}/{len(loader)}"
                )

    # Extract bonafide probabilities.
    bonafide_probabilities = [
        probabilities[0]
        for probabilities in all_probabilities
    ]

    # Calculate evaluation metrics.
    accuracy = accuracy_score(
        all_labels,
        all_predictions,
    )

    macro_f1 = f1_score(
        all_labels,
        all_predictions,
        average="macro",
        zero_division=0,
    )

    bonafide_precision = precision_score(
        all_labels,
        all_predictions,
        pos_label=0,
        zero_division=0,
    )

    spoof_precision = precision_score(
        all_labels,
        all_predictions,
        pos_label=1,
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

    print("\nAST Evaluation Results")
    print("---------------------")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Macro F1: {macro_f1:.4f}")
    print(f"Bonafide precision: {bonafide_precision:.4f}")
    print(f"Bonafide recall: {bonafide_recall:.4f}")
    print(f"Spoof precision: {spoof_precision:.4f}")
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

    print(
        "Confusion Matrix "
        "(rows=actual, columns=predicted):"
    )
    print(matrix)

    # Save predictions.
    results_df = validation_df[
        ["audio_id", "label"]
    ].copy()

    results_df["actual"] = all_labels
    results_df["predicted"] = all_predictions
    results_df["bonafide_probability"] = bonafide_probabilities

    results_df["actual_class"] = results_df["actual"].map(
        ID2LABEL
    )

    results_df["predicted_class"] = results_df["predicted"].map(
        ID2LABEL
    )

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    results_df.to_csv(
        RESULTS_DIR / "ast_final_predictions.csv",
        index=False,
    )

    # Save evaluation metrics.
    metrics_df = pd.DataFrame([{
        "checkpoint_epoch": checkpoint["epoch"],
        "checkpoint_validation_loss": checkpoint["validation_loss"],
        "checkpoint_validation_accuracy": checkpoint["validation_accuracy"],
        "recordings": len(dataset),
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "bonafide_precision": bonafide_precision,
        "bonafide_recall": bonafide_recall,
        "spoof_precision": spoof_precision,
        "spoof_recall": spoof_recall,
    }])

    metrics_df.to_csv(
        RESULTS_DIR / "ast_final_metrics.csv",
        index=False,
    )

    # Save confusion matrix.
    matrix_df = pd.DataFrame(
        matrix,
        index=["actual_bonafide", "actual_spoof"],
        columns=["predicted_bonafide", "predicted_spoof"],
    )

    matrix_df.to_csv(
        RESULTS_DIR / "ast_confusion_matrix.csv"
    )

    print("\nAST evaluation files saved in:")
    print(RESULTS_DIR)


if __name__ == "__main__":
    evaluate_ast()
