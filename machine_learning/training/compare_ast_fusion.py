
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

DEV_CSV = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "fusion_dev_pairs.csv"
)

CHECKPOINT_DIR = (
    PROJECT_ROOT
    / "machine_learning"
    / "checkpoints"
)

AST_CHECKPOINT = CHECKPOINT_PATH

FULL_FUSION_CHECKPOINT = (
    CHECKPOINT_DIR / "ast_rir_fusion_best.pt"
)

RT60_CHECKPOINT = (
    CHECKPOINT_DIR / "ast_rir_rt60_best.pt"
)

DRR_CHECKPOINT = (
    CHECKPOINT_DIR / "ast_rir_drr_best.pt"
)

RESULTS_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "ablation_comparison_results.json"
)

BATCH_SIZE = 4


def calculate_metrics(labels, probabilities):

    predictions = (probabilities >= 0.5).astype(int)

    fpr, tpr, _ = roc_curve(labels, probabilities)
    fnr = 1 - tpr

    eer = np.interp(0, fpr - fnr, fpr)

    return {
        "accuracy": float(
            accuracy_score(labels, predictions)
        ),
        "precision": float(
            precision_score(labels, predictions, zero_division=0)
        ),
        "recall": float(
            recall_score(labels, predictions, zero_division=0)
        ),
        "f1_score": float(
            f1_score(labels, predictions, zero_division=0)
        ),
        "roc_auc": float(
            roc_auc_score(labels, probabilities)
        ),
        "eer": float(eer),
    }


def load_fusion_checkpoint(model, checkpoint_path, device):

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=True,
    )

    model.fusion_classifier.load_state_dict(
        checkpoint["fusion_classifier_state_dict"]
    )

    model.eval()

    return model


def main():

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("Using device:", device)

    required_files = [
        DEV_CSV,
        AST_CHECKPOINT,
        FULL_FUSION_CHECKPOINT,
        RT60_CHECKPOINT,
        DRR_CHECKPOINT,
    ]

    for path in required_files:
        if not path.is_file():
            raise FileNotFoundError(
                f"Required file not found: {path}"
            )

    # --------------------------------------------------
    # Load validation dataset ONCE
    # --------------------------------------------------

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

    # --------------------------------------------------
    # Load trained AST model
    # --------------------------------------------------

    print("\nLoading trained AST model...")

    ast_model = ASTForAudioClassification.from_pretrained(
        MODEL_NAME,
        num_labels=2,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
        ignore_mismatched_sizes=True,
    )

    ast_checkpoint = torch.load(
        AST_CHECKPOINT,
        map_location=device,
        weights_only=True,
    )

    ast_model.load_state_dict(
        ast_checkpoint["model_state_dict"]
    )

    ast_model.to(device).eval()

    # --------------------------------------------------
    # Load four model configurations
    # --------------------------------------------------

    print("\nLoading fusion models...")

    rt60_model = ASTRIRFusionModel(
        ast_model,
        acoustic_feature_size=1,
    ).to(device)

    drr_model = ASTRIRFusionModel(
        ast_model,
        acoustic_feature_size=1,
    ).to(device)

    full_fusion_model = ASTRIRFusionModel(
        ast_model,
        acoustic_feature_size=2,
    ).to(device)

    rt60_model = load_fusion_checkpoint(
        rt60_model,
        RT60_CHECKPOINT,
        device,
    )

    drr_model = load_fusion_checkpoint(
        drr_model,
        DRR_CHECKPOINT,
        device,
    )

    full_fusion_model = load_fusion_checkpoint(
        full_fusion_model,
        FULL_FUSION_CHECKPOINT,
        device,
    )

    print("AST-only checkpoint loaded.")
    print("AST + RT60 checkpoint loaded.")
    print("AST + DRR checkpoint loaded.")
    print("AST + RT60 + DRR checkpoint loaded.")

    # --------------------------------------------------
    # Collect predictions
    # --------------------------------------------------

    all_labels = []

    probabilities = {
        "ast_only": [],
        "ast_rt60": [],
        "ast_drr": [],
        "ast_rt60_drr": [],
    }

    print("\nEvaluating all four configurations...")

    with torch.inference_mode():

        for batch_index, batch in enumerate(dev_loader):

            input_values = batch["input_values"].to(device)

            # Dataset contains both normalized acoustic features.
            acoustic_features = batch[
                "acoustic_features"
            ].to(device)

            labels = batch["label"].to(device)

            rt60_features = acoustic_features[:, [0]]
            drr_features = acoustic_features[:, [1]]

            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=(device.type == "cuda"),
            ):

                # 1. AST only
                ast_logits = ast_model(
                    input_values=input_values
                ).logits

                # 2. AST + RT60
                rt60_logits = rt60_model(
                    input_values,
                    rt60_features,
                )

                # 3. AST + DRR
                drr_logits = drr_model(
                    input_values,
                    drr_features,
                )

                # 4. AST + RT60 + DRR
                full_logits = full_fusion_model(
                    input_values,
                    acoustic_features,
                )

            logits_dict = {
                "ast_only": ast_logits,
                "ast_rt60": rt60_logits,
                "ast_drr": drr_logits,
                "ast_rt60_drr": full_logits,
            }

            for name, logits in logits_dict.items():

                spoof_probs = torch.softmax(
                    logits.float(),
                    dim=1,
                )[:, 1]

                probabilities[name].extend(
                    spoof_probs.cpu().numpy().tolist()
                )

            all_labels.extend(
                labels.cpu().numpy().tolist()
            )

            if (batch_index + 1) % 100 == 0:
                print(
                    f"Processed batches: "
                    f"{batch_index + 1}/{len(dev_loader)}"
                )

    # --------------------------------------------------
    # Calculate metrics
    # --------------------------------------------------

    labels = np.array(all_labels)

    results = {
        "validation_samples": len(labels),
        "positive_class": "spoof",
    }

    for name in probabilities:

        probs = np.array(probabilities[name])

        results[name] = calculate_metrics(
            labels,
            probs,
        )

    # --------------------------------------------------
    # Display results
    # --------------------------------------------------

    print("\n========== DAY 6 ABLATION RESULTS ==========")

    model_names = {
        "ast_only": "AST-only",
        "ast_rt60": "AST + RT60",
        "ast_drr": "AST + DRR",
        "ast_rt60_drr": "AST + RT60 + DRR",
    }

    metrics = [
        "accuracy",
        "precision",
        "recall",
        "f1_score",
        "roc_auc",
        "eer",
    ]

    print(f"{'Metric':<15}", end="")

    for name in model_names:
        print(f"{model_names[name]:>20}", end="")

    print()

    for metric in metrics:

        print(f"{metric:<15}", end="")

        for name in model_names:

            value = results[name][metric]

            print(f"{value:>20.4f}", end="")

        print()

    # --------------------------------------------------
    # Save results
    # --------------------------------------------------

    RESULTS_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with RESULTS_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            results,
            file,
            indent=4,
        )

    print("\nAblation results saved to:")
    print(RESULTS_PATH)


if __name__ == "__main__":
    main()
