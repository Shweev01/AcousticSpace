
import csv
import random
from collections import defaultdict
from itertools import cycle
from pathlib import Path

import numpy as np
import soundfile as sf


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATASET_ROOT = Path(r"C:\AcousticSpace_Dataset")

TRAIN_AUDIO_DIR = (
    DATASET_ROOT / "LA" / "LA" /
    "ASVspoof2019_LA_train" / "flac"
)

DEV_AUDIO_DIR = (
    DATASET_ROOT / "LA" / "LA" /
    "ASVspoof2019_LA_dev" / "flac"
)

TRAIN_PROTOCOL = (
    DATASET_ROOT / "LA" / "LA" /
    "ASVspoof2019_LA_cm_protocols" /
    "ASVspoof2019.LA.cm.train.trn.txt"
)

DEV_PROTOCOL = (
    DATASET_ROOT / "LA" / "LA" /
    "ASVspoof2019_LA_cm_protocols" /
    "ASVspoof2019.LA.cm.dev.trl.txt"
)

RIR_ROOT = (
    DATASET_ROOT / "real_rirs_isotropic_noises"
)

RIR_MANIFEST = RIR_ROOT / "rir_list"

RIR_FEATURES_CSV = (
    PROJECT_ROOT / "data" / "processed" /
    "rir_feature_validation.csv"
)

OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"

RANDOM_SEED = 42

OUTPUT_FIELDS = [
    "audio_path",
    "label",
    "rir_path",
    "rir_channel",
    "rt60_seconds",
    "drr_db",
    "rt60_r_squared",
]


def load_protocol(protocol_path, audio_dir):
    """Read ASVspoof protocol and locate audio files."""

    if not protocol_path.is_file():
        raise FileNotFoundError(protocol_path)

    records = []

    with protocol_path.open("r") as file:
        for line in file:
            parts = line.strip().split()

            if len(parts) < 4:
                continue

            audio_id = parts[1]
            label = parts[-1].lower()

            if label not in {"bonafide", "spoof"}:
                continue

            audio_path = audio_dir / f"{audio_id}.flac"

            if not audio_path.is_file():
                continue

            records.append({
                "audio_path": str(audio_path),
                "label": 0 if label == "bonafide" else 1,
            })

    if not records:
        raise RuntimeError(
            f"No valid audio records found: {protocol_path}"
        )

    return records


def load_validated_rirs():
    """Load RIR recordings and select their best valid channel."""

    if not RIR_FEATURES_CSV.is_file():
        raise FileNotFoundError(RIR_FEATURES_CSV)

    if not RIR_MANIFEST.is_file():
        raise FileNotFoundError(RIR_MANIFEST)

    manifest_names = {
        Path(line.strip()).name
        for line in RIR_MANIFEST.read_text().splitlines()
        if line.strip()
    }

    grouped = defaultdict(list)

    with RIR_FEATURES_CSV.open(
        "r", newline=""
    ) as file:

        reader = csv.DictReader(file)

        for row in reader:
            filename = row["filename"]

            if filename not in manifest_names:
                continue

            try:
                rt60 = float(row["rt60_seconds"])
                drr = float(row["drr_db"])
                r_squared = float(row["rt60_r_squared"])
                channel = int(row["channel_index"])

                if (
                    not np.isfinite(rt60)
                    or not np.isfinite(drr)
                    or not np.isfinite(r_squared)
                    or rt60 <= 0
                ):
                    continue

            except (ValueError, TypeError):
                continue

            grouped[filename].append({
                "filename": filename,
                "channel": channel,
                "rt60": rt60,
                "drr": drr,
                "r_squared": r_squared,
            })

    rir_records = []

    for filename, channels in grouped.items():

        # Keep the channel with the strongest RT60 fit.
        best_channel = max(
            channels,
            key=lambda item: item["r_squared"],
        )

        matches = list(RIR_ROOT.rglob(filename))

        if not matches:
            continue

        rir_path = matches[0]

        try:
            info = sf.info(rir_path)

            if best_channel["channel"] >= info.channels:
                continue

        except (RuntimeError, OSError):
            continue

        rir_records.append({
            **best_channel,
            "rir_path": str(rir_path),
        })

    if len(rir_records) < 2:
        raise RuntimeError(
            "Not enough valid RIR recordings were found."
        )

    return rir_records


def split_rirs(rir_records):
    """Create separate deterministic RIR pools."""

    rng = random.Random(RANDOM_SEED)

    rir_records = list(rir_records)
    rng.shuffle(rir_records)

    split_index = max(
        1,
        int(len(rir_records) * 0.8),
    )

    train_rirs = rir_records[:split_index]
    dev_rirs = rir_records[split_index:]

    if not dev_rirs:
        raise RuntimeError(
            "Not enough RIR recordings for a separate dev pool."
        )

    return train_rirs, dev_rirs


def create_pairs(audio_records, rir_records, output_path):
    """Pair each speech recording with a validated RIR."""

    rir_cycle = cycle(rir_records)
    paired_records = []

    for audio in audio_records:

        rir = next(rir_cycle)

        paired_records.append({
            "audio_path": audio["audio_path"],
            "label": audio["label"],
            "rir_path": rir["rir_path"],
            "rir_channel": rir["channel"],
            "rt60_seconds": rir["rt60"],
            "drr_db": rir["drr"],
            "rt60_r_squared": rir["r_squared"],
        })

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_path.open(
        "w", newline=""
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=OUTPUT_FIELDS,
        )

        writer.writeheader()
        writer.writerows(paired_records)

    return paired_records


def main():

    print("Loading ASVspoof recordings...")

    train_audio = load_protocol(
        TRAIN_PROTOCOL,
        TRAIN_AUDIO_DIR,
    )

    dev_audio = load_protocol(
        DEV_PROTOCOL,
        DEV_AUDIO_DIR,
    )

    print("Loading validated RIR features...")

    rir_records = load_validated_rirs()

    train_rirs, dev_rirs = split_rirs(
        rir_records
    )

    print("\nCreating training pairs...")

    train_pairs = create_pairs(
        train_audio,
        train_rirs,
        OUTPUT_DIR / "fusion_train_pairs.csv",
    )

    print("Creating validation pairs...")

    dev_pairs = create_pairs(
        dev_audio,
        dev_rirs,
        OUTPUT_DIR / "fusion_dev_pairs.csv",
    )

    print("\nFusion dataset preparation complete.")

    print("Training audio:", len(train_audio))
    print("Validation audio:", len(dev_audio))

    print("Training RIR recordings:", len(train_rirs))
    print("Validation RIR recordings:", len(dev_rirs))

    print("Training pairs:", len(train_pairs))
    print("Validation pairs:", len(dev_pairs))

    print(
        "Training bonafide:",
        sum(row["label"] == 0 for row in train_pairs),
    )

    print(
        "Training spoof:",
        sum(row["label"] == 1 for row in train_pairs),
    )

    print(
        "\nSaved:",
        OUTPUT_DIR / "fusion_train_pairs.csv",
    )

    print(
        "Saved:",
        OUTPUT_DIR / "fusion_dev_pairs.csv",
    )


if __name__ == "__main__":
    main()
