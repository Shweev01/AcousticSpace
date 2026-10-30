
from pathlib import Path

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    recall_score,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]

RESULTS_DIR = PROJECT_ROOT / "machine_learning" / "evaluation" / "results"

PREDICTIONS_FILE = RESULTS_DIR / "validation_predictions.csv"

THRESHOLDS = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]

SELECTED_THRESHOLD = 0.30


def analyze_thresholds():

    if not PREDICTIONS_FILE.is_file():
        raise FileNotFoundError(
            f"Validation predictions not found: {PREDICTIONS_FILE}"
        )

    df = pd.read_csv(PREDICTIONS_FILE)

    required_columns = {
        "actual",
        "bonafide_probability",
    }

    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {sorted(missing_columns)}"
        )

    y_true = df["actual"].to_numpy()
    bonafide_probabilities = df["bonafide_probability"].to_numpy()

    results = []

    for threshold in THRESHOLDS:

        # Class 0 = bonafide, Class 1 = spoof.
        y_pred = (bonafide_probabilities < threshold).astype(int)

        results.append({
            "threshold": threshold,
            "accuracy": accuracy_score(y_true, y_pred),
            "bonafide_recall": recall_score(
                y_true, y_pred, pos_label=0, zero_division=0
            ),
            "spoof_recall": recall_score(
                y_true, y_pred, pos_label=1, zero_division=0
            ),
            "macro_f1": f1_score(
                y_true, y_pred, average="macro", zero_division=0
            ),
        })

    comparison_df = pd.DataFrame(results)

    comparison_path = RESULTS_DIR / "threshold_comparison.csv"
    comparison_df.to_csv(comparison_path, index=False)

    # Generate confusion matrix for the selected threshold.
    selected_predictions = (
        bonafide_probabilities < SELECTED_THRESHOLD
    ).astype(int)

    matrix = confusion_matrix(
        y_true,
        selected_predictions,
        labels=[0, 1],
    )

    matrix_df = pd.DataFrame(
        matrix,
        index=["actual_bonafide", "actual_spoof"],
        columns=["predicted_bonafide", "predicted_spoof"],
    )

    matrix_path = RESULTS_DIR / "confusion_matrix_threshold_030.csv"
    matrix_df.to_csv(matrix_path)

    print("\nTHRESHOLD COMPARISON")
    print(comparison_df.round(4).to_string(index=False))

    print(f"\nSelected threshold: {SELECTED_THRESHOLD}")
    print("\nConfusion Matrix:")
    print(matrix)

    print("\nSaved files:")
    print(comparison_path)
    print(matrix_path)


if __name__ == "__main__":
    analyze_thresholds()
