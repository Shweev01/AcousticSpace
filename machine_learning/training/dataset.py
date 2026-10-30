
import hashlib
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from audio_analysis.dataset_loader import (
    load_training_dataset,
    load_validation_dataset,
)
from audio_analysis.feature_extraction import extract_audio_features


LABEL_MAP = {
    "bonafide": 0,
    "spoof": 1,
}

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CACHE_DIR = PROJECT_ROOT / "data" / "processed" / "mel_cache"


class AudioDataset(Dataset):
    """Load audio spectrograms and cache them for reuse."""

    def __init__(self, dataframe=None, max_frames=256):
        self.dataframe = (
            load_training_dataset() if dataframe is None else dataframe
        ).reset_index(drop=True)

        if max_frames <= 0:
            raise ValueError("max_frames must be positive.")

        if not self.dataframe["label"].isin(LABEL_MAP).all():
            raise ValueError("Dataset contains unknown labels.")

        self.max_frames = max_frames
        self.cache_settings = f"mel80-fft1024-hop256-frames{max_frames}"
        CACHE_DIR.mkdir(parents=True, exist_ok=True)

    def __len__(self):
        return len(self.dataframe)

    def _cache_path(self, file_path):
        key = hashlib.sha256(
            f"{self.cache_settings}|{file_path}".encode("utf-8")
        ).hexdigest()
        return CACHE_DIR / f"{key}.npy"

    def _load_mel(self, file_path):
        cache_path = self._cache_path(file_path)

        if cache_path.exists():
            return np.load(cache_path, allow_pickle=False)

        features = extract_audio_features(file_path)
        mel = features["mel_spectrogram"]

        if mel.shape[1] > self.max_frames:
            mel = mel[:, :self.max_frames]
        elif mel.shape[1] < self.max_frames:
            mel = np.pad(
                mel,
                ((0, 0), (0, self.max_frames - mel.shape[1])),
                mode="constant",
            )

        mel = np.asarray(mel, dtype=np.float32)

        # Save directly to the cache file to avoid Windows rename errors.
        with cache_path.open("wb") as file:
            np.save(file, mel, allow_pickle=False)

        return mel

    def __getitem__(self, index):
        row = self.dataframe.iloc[index]
        file_path = row["file_path"]

        mel = self._load_mel(file_path)

        spectrogram = torch.from_numpy(mel.copy()).unsqueeze(0)
        label = torch.tensor(
            LABEL_MAP[row["label"]], dtype=torch.long
        )

        return spectrogram, label


def create_datasets():
    """Create separate training and validation datasets."""
    train_df = load_training_dataset()
    val_df = load_validation_dataset()

    return AudioDataset(train_df), AudioDataset(val_df)
