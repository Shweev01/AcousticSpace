
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
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


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEV_CSV = PROJECT_ROOT / "data" / "processed" / "fusion_dev_pairs.csv"
CHECKPOINT_DIR = PROJECT_ROOT / "machine_learning" / "checkpoints"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"

AST_CHECKPOINT = CHECKPOINT_PATH
FULL_CHECKPOINT = CHECKPOINT_DIR / "ast_rir_fusion_best.pt"
RT60_CHECKPOINT = CHECKPOINT_DIR / "ast_rir_rt60_best.pt"
DRR_CHECKPOINT = CHECKPOINT_DIR / "ast_rir_drr_best.pt"

PREDICTIONS_PATH = OUTPUT_DIR / "error_analysis_predictions.csv"
ERRORS_PATH = OUTPUT_DIR / "error_analysis_misclassified.csv"
SUMMARY_PATH = OUTPUT_DIR / "error_analysis_summary.json"

BATCH_SIZE = 4

MODEL_NAMES = {
    "ast_only": "AST-only",
    "ast_rt60": "AST + RT60",
    "ast_drr": "AST + DRR",
    "ast_rt60_drr": "AST + RT60 + DRR",
}


def load_fusion_checkpoint(model, path, device):
    checkpoint = torch.load(
        path,
        map_location=device,
        weights_only=True,
    )

    model.fusion_classifier.load_state_dict(
        checkpoint["fusion_classifier_state_dict"]
    )
    model.eval()
    return model


def classify_errors(labels, predictions):
    return np.where(
        (labels == 0) & (predictions == 1),
        "false_positive",
        np.where(
            (labels == 1) & (predictions == 0),
            "false_negative",
            "correct",
        ),
    )


def main():

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    print("Using device:", device)

    required_files = [
        DEV_CSV,
        AST_CHECKPOINT,
        FULL_CHECKPOINT,
        RT60_CHECKPOINT,
        DRR_CHECKPOINT,
    ]

    for path in required_files:
        if not path.is_file():
            raise FileNotFoundError(f"Missing file: {path}")

    # Load source records in their original order.
    source_df = pd.read_csv(DEV_CSV)

    dev_dataset = ASTRIRFusionDataset(DEV_CSV)

    if len(source_df) != len(dev_dataset):
        raise ValueError("CSV and dataset lengths do not match.")

    dev_loader = DataLoader(
        dev_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )

    print("Validation samples:", len(dev_dataset))

    # Load trained AST model.
    print("\nLoading AST model...")

    ast_model = ASTForAudioClassification.from_pretrained(
        MODEL_NAME,
        num_labels=2,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
        ignore_mismatched_sizes=True,
    )

    checkpoint = torch.load(
        AST_CHECKPOINT,
        map_location=device,
        weights_only=True,
    )

    ast_model.load_state_dict(checkpoint["model_state_dict"])
    ast_model.to(device).eval()

    # Load fusion models.
    print("Loading fusion models...")

    rt60_model = ASTRIRFusionModel(
        ast_model, acoustic_feature_size=1
    ).to(device)

    drr_model = ASTRIRFusionModel(
        ast_model, acoustic_feature_size=1
    ).to(device)

    full_model = ASTRIRFusionModel(
        ast_model, acoustic_feature_size=2
    ).to(device)

    load_fusion_checkpoint(rt60_model, RT60_CHECKPOINT, device)
    load_fusion_checkpoint(drr_model, DRR_CHECKPOINT, device)
    load_fusion_checkpoint(full_model, FULL_CHECKPOINT, device)

    probabilities = {
        name: [] for name in MODEL_NAMES
    }
    all_labels = []

    print("\nRunning error analysis...")

    with torch.inference_mode():

        for batch_index, batch in enumerate(dev_loader):

            inputs = batch["input_values"].to(device)
            features = batch["acoustic_features"].to(device)
            labels = batch["label"].to(device)

            rt60_features = features[:, [0]]
            drr_features = features[:, [1]]

            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=(device.type == "cuda"),
            ):

                logits_dict = {
                    "ast_only": ast_model(
                        input_values=inputs
                    ).logits,

                    "ast_rt60": rt60_model(
                        inputs, rt60_features
                    ),

                    "ast_drr": drr_model(
                        inputs, drr_features
                    ),

                    "ast_rt60_drr": full_model(
                        inputs, features
                    ),
                }

            for name, logits in logits_dict.items():

                spoof_probs = torch.softmax(
                    logits.float(), dim=1
                )[:, 1]

                probabilities[name].extend(
                    spoof_probs.cpu().numpy().tolist()
                )

            all_labels.extend(labels.cpu().numpy().tolist())

            if (batch_index + 1) % 100 == 0:
                print(
                    f"Processed {batch_index + 1}/"
                    f"{len(dev_loader)} batches"
                )

    # Verify row alignment.
    labels = np.asarray(all_labels)

    if not np.array_equal(
        labels,
        source_df["label"].to_numpy(),
    ):
        raise ValueError(
            "Prediction labels do not match source CSV order."
        )

    # Build detailed prediction report.
    report = source_df.copy()

    report["true_label"] = labels
    report["true_class"] = np.where(
        labels == 0, "bonafide", "spoof"
    )

    for name in MODEL_NAMES:

        probs = np.asarray(probabilities[name])
        predictions = (probs >= 0.5).astype(int)

        confidence = np.where(
            predictions == 1,
            probs,
            1 - probs,
        )

        report[f"{name}_spoof_probability"] = probs
        report[f"{name}_prediction"] = predictions
        report[f"{name}_predicted_class"] = np.where(
            predictions == 0, "bonafide", "spoof"
        )
        report[f"{name}_confidence"] = confidence
        report[f"{name}_error_type"] = classify_errors(
            labels, predictions
        )

    # Compare all four model predictions.
    prediction_columns = [
        f"{name}_prediction" for name in MODEL_NAMES
    ]

    predictions_matrix = report[
        prediction_columns
    ].to_numpy()

    report["model_disagreement"] = (
        np.any(
            predictions_matrix != predictions_matrix[:, [0]],
            axis=1,
        )
    )

    # Save all predictions and incorrect predictions.
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    report.to_csv(PREDICTIONS_PATH, index=False)

    errors = report[
        report[
            [f"{name}_error_type" for name in MODEL_NAMES]
        ].ne("correct").any(axis=1)
    ].copy()

    errors.to_csv(ERRORS_PATH, index=False)

    # Summarize errors and disagreements.
    summary = {
        "validation_samples": len(report),
        "models": {},
        "model_disagreement_count": int(
            report["model_disagreement"].sum()
        ),
        "samples_with_any_model_error": len(errors),
    }

    for name in MODEL_NAMES:

        error_types = report[f"{name}_error_type"]
        confidence = report[f"{name}_confidence"]

        summary["models"][name] = {
            "false_positives": int(
                (error_types == "false_positive").sum()
            ),
            "false_negatives": int(
                (error_types == "false_negative").sum()
            ),
            "correct": int(
                (error_types == "correct").sum()
            ),
            "high_confidence_errors": int(
                (
                    (error_types != "correct")
                    & (confidence >= 0.90)
                ).sum()
            ),
        }

    with SUMMARY_PATH.open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=4)

    print("\n========== DAY 7 ERROR ANALYSIS ==========")

    for name, display_name in MODEL_NAMES.items():

        values = summary["models"][name]

        print(f"\n{display_name}")
        print("False positives:", values["false_positives"])
        print("False negatives:", values["false_negatives"])
        print("Correct:", values["correct"])
        print(
            "High-confidence errors:",
            values["high_confidence_errors"],
        )

    print(
        "\nModel disagreements:",
        summary["model_disagreement_count"],
    )

    print("\nSaved files:")
    print(PREDICTIONS_PATH)
    print(ERRORS_PATH)
    print(SUMMARY_PATH)


if __name__ == "__main__":
    main()
