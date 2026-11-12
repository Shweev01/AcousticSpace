
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import ASTForAudioClassification
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
)

from machine_learning.training.ast_dataset import MODEL_NAME
from machine_learning.training.ast_rir_fusion import ASTRIRFusionModel
from machine_learning.training.fusion_dataset import ASTRIRFusionDataset
from machine_learning.inference.predict import (
    CHECKPOINT_PATH,
    ID2LABEL,
    LABEL2ID,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEV_CSV = PROJECT_ROOT / "data" / "processed" / "fusion_dev_pairs.csv"

FUSION_CHECKPOINT = (
    PROJECT_ROOT
    / "machine_learning"
    / "checkpoints"
    / "ast_rir_fusion_best.pt"
)

RESULTS_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "model_comparison_results.json"
)

BATCH_SIZE = 4


def calculate_metrics(labels, probabilities):

    predictions = (probabilities >= 0.5).astype(int)

    fpr, tpr, _ = roc_curve(labels, probabilities)
    fnr = 1 - tpr

    eer = np.interp(0, fpr - fnr, fpr)

    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "precision": float(
            precision_score(labels, predictions, zero_division=0)
        ),
        "recall": float(
            recall_score(labels, predictions, zero_division=0)
        ),
        "f1_score": float(
            f1_score(labels, predictions, zero_division=0)
        ),
        "roc_auc": float(roc_auc_score(labels, probabilities)),
        "eer": float(eer),
    }


def main():

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("Using device:", device)

    if not CHECKPOINT_PATH.is_file():
        raise FileNotFoundError(
            f"AST checkpoint not found: {CHECKPOINT_PATH}"
        )

    if not FUSION_CHECKPOINT.is_file():
        raise FileNotFoundError(
            f"Fusion checkpoint not found: {FUSION_CHECKPOINT}"
        )

    # Load validation dataset.
    print("\nLoading validation dataset...")

    dev_dataset = ASTRIRFusionDataset(DEV_CSV)

    dev_loader = DataLoader(
        dev_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )

    print("Validation samples:", len(dev_dataset))

    # Load fine-tuned AST model.
    print("\nLoading trained AST model...")

    ast_model = ASTForAudioClassification.from_pretrained(
        MODEL_NAME,
        num_labels=2,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
        ignore_mismatched_sizes=True,
    )

    ast_checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=device,
        weights_only=True,
    )

    ast_model.load_state_dict(
        ast_checkpoint["model_state_dict"]
    )

    ast_model.to(device).eval()

    # Create fusion model using the same AST backbone.
    print("\nLoading trained fusion model...")

    fusion_model = ASTRIRFusionModel(ast_model).to(device)

    fusion_checkpoint = torch.load(
        FUSION_CHECKPOINT,
        map_location=device,
        weights_only=True,
    )

    fusion_model.fusion_classifier.load_state_dict(
        fusion_checkpoint["fusion_classifier_state_dict"]
    )

    fusion_model.eval()

    print("AST checkpoint loaded.")
    print("Fusion checkpoint loaded.")

    # Collect predictions from both models.
    all_labels = []
    ast_probabilities = []
    fusion_probabilities = []

    print("\nEvaluating both models...")

    with torch.inference_mode():

        for batch_index, batch in enumerate(dev_loader):

            input_values = batch["input_values"].to(device)
            acoustic_features = batch["acoustic_features"].to(device)
            labels = batch["label"].to(device)

            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=(device.type == "cuda"),
            ):

                # AST-only prediction.
                ast_logits = ast_model(
                    input_values=input_values
                ).logits

                # AST + RIR fusion prediction.
                fusion_logits = fusion_model(
                    input_values,
                    acoustic_features,
                )

            ast_probs = torch.softmax(
                ast_logits.float(), dim=1
            )[:, 1]

            fusion_probs = torch.softmax(
                fusion_logits.float(), dim=1
            )[:, 1]

            all_labels.extend(labels.cpu().numpy().tolist())
            ast_probabilities.extend(
                ast_probs.cpu().numpy().tolist()
            )
            fusion_probabilities.extend(
                fusion_probs.cpu().numpy().tolist()
            )

            if (batch_index + 1) % 500 == 0:
                print(
                    f"Processed batches: "
                    f"{batch_index + 1}/{len(dev_loader)}"
                )

    labels = np.array(all_labels)
    ast_probabilities = np.array(ast_probabilities)
    fusion_probabilities = np.array(fusion_probabilities)

    # Calculate metrics.
    ast_metrics = calculate_metrics(
        labels,
        ast_probabilities,
    )

    fusion_metrics = calculate_metrics(
        labels,
        fusion_probabilities,
    )

    # Display comparison.
    print("\n========== MODEL COMPARISON ==========")

    print(
        f"{'Metric':<15}"
        f"{'AST-only':>15}"
        f"{'AST + RIR':>15}"
    )

    for metric in ast_metrics:

        print(
            f"{metric:<15}"
            f"{ast_metrics[metric]:>15.4f}"
            f"{fusion_metrics[metric]:>15.4f}"
        )

    results = {
        "validation_samples": len(labels),
        "positive_class": "spoof",
        "ast_only": ast_metrics,
        "ast_rir_fusion": fusion_metrics,
    }

    RESULTS_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with RESULTS_PATH.open("w", encoding="utf-8") as file:
        json.dump(results, file, indent=4)

    print("\nComparison results saved to:")
    print(RESULTS_PATH)


if __name__ == "__main__":
    main()
