
from pathlib import Path

import pandas as pd


DATASET_ROOT = Path(r"C:\AcousticSpace_Dataset\LA\LA")

TRAIN_AUDIO_DIR = DATASET_ROOT / "ASVspoof2019_LA_train" / "flac"

TRAIN_PROTOCOL_FILE = (
    DATASET_ROOT
    / "ASVspoof2019_LA_cm_protocols"
    / "ASVspoof2019.LA.cm.train.trn.txt"
)



def load_protocol(protocol_path: Path = TRAIN_PROTOCOL_FILE) -> pd.DataFrame:
    """Load recording IDs and labels from an ASVspoof protocol."""

    if not protocol_path.is_file():
        raise FileNotFoundError(f"Protocol file not found: {protocol_path}")

    protocol = pd.read_csv(
        protocol_path,
        sep=r"\s+",
        header=None,
        names=["speaker_id", "audio_id", "placeholder", "attack_type", "label"],
    )

    if protocol.empty:
        raise ValueError("The protocol file contains no records.")

    if protocol.isna().any().any():
        raise ValueError("The protocol contains missing values.")

    if not protocol["label"].isin(["bonafide", "spoof"]).all():
        raise ValueError("The protocol contains unexpected labels.")

    if protocol["audio_id"].duplicated().any():
        raise ValueError("The protocol contains duplicate audio IDs.")

    return protocol[["audio_id", "label"]].copy()



def load_training_dataset(
    audio_dir: Path = TRAIN_AUDIO_DIR,
    protocol_path: Path = TRAIN_PROTOCOL_FILE,
) -> pd.DataFrame:
    """Match protocol records to their FLAC audio files."""

    if not audio_dir.is_dir():
        raise FileNotFoundError(f"Audio directory not found: {audio_dir}")

    dataset = load_protocol(protocol_path)

    dataset["file_path"] = dataset["audio_id"].map(
        lambda audio_id: str(audio_dir / f"{audio_id}.flac")
    )

    dataset["file_exists"] = dataset["file_path"].map(
        lambda path: Path(path).is_file()
    )

    missing_count = int((~dataset["file_exists"]).sum())

    if missing_count:
        missing_ids = (
            dataset.loc[~dataset["file_exists"], "audio_id"].head(5).tolist()
        )
        raise FileNotFoundError(
            f"{missing_count} audio files are missing. Examples: {missing_ids}"
        )

    return dataset



if __name__ == "__main__":
    dataset = load_training_dataset()

    print("Total protocol records:", len(dataset))
    print("\nLabel counts:")
    print(dataset["label"].value_counts())
    print("\nMissing audio files:", (~dataset["file_exists"]).sum())
    print("\nFirst five records:")
    print(dataset.head().to_string(index=False))
