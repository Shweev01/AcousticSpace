
import csv
import json
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]

TRAIN_CSV = PROJECT_ROOT / "data" / "processed" / "fusion_train_pairs.csv"

STATS_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "acoustic_feature_stats.json"
)


def calculate_acoustic_stats(csv_path):
    rt60_values = []
    drr_values = []

    with open(csv_path, "r", newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)

        for row in reader:
            rt60_values.append(float(row["rt60_seconds"]))
            drr_values.append(float(row["drr_db"]))

    features = np.array(
        list(zip(rt60_values, drr_values)),
        dtype=np.float64,
    )

    if len(features) == 0 or not np.isfinite(features).all():
        raise ValueError("Training acoustic features are empty or invalid.")

    means = features.mean(axis=0)
    stds = features.std(axis=0)

    # Prevent division by zero.
    stds[stds == 0] = 1.0

    return {
        "mean": means.tolist(),
        "std": stds.tolist(),
        "feature_names": ["rt60_seconds", "drr_db"],
    }


def save_acoustic_stats(stats, output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(stats, file, indent=4)


def load_acoustic_stats(path=STATS_PATH):
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


if __name__ == "__main__":
    stats = calculate_acoustic_stats(TRAIN_CSV)
    save_acoustic_stats(stats, STATS_PATH)

    print("Acoustic feature normalization statistics saved.")
    print("Mean:", stats["mean"])
    print("Standard deviation:", stats["std"])
    print("Saved to:", STATS_PATH)
