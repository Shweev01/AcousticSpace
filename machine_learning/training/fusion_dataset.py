
import csv
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
import torch
from torch.utils.data import Dataset
from transformers import ASTFeatureExtractor

from machine_learning.training.ast_dataset import MODEL_NAME
from audio_analysis.rir_analysis.reverberation import apply_reverberation
from machine_learning.training.fusion_normalization import load_acoustic_stats


SAMPLE_RATE = 16000


class ASTRIRFusionDataset(Dataset):
    """
    Loads speech, applies its paired RIR, and prepares
    AST audio features with RT60 and DRR values.
    """

    def __init__(
        self,
        csv_path,
        feature_extractor=None,
        feature_indices=(0, 1),
    ):
        self.csv_path = Path(csv_path)
        self.feature_indices = tuple(feature_indices)

        if not self.feature_indices or any(
            index not in (0, 1) for index in self.feature_indices
        ):
            raise ValueError(
                "feature_indices must contain 0 (RT60), 1 (DRR), or both."
        )
        self.acoustic_stats = load_acoustic_stats()
        self.acoustic_mean = torch.tensor(
            self.acoustic_stats["mean"], dtype=torch.float32
        )
        self.acoustic_std = torch.tensor(
            self.acoustic_stats["std"], dtype=torch.float32
        )
        
        with self.csv_path.open("r", newline="", encoding="utf-8") as file:
            self.records = list(csv.DictReader(file))

        if not self.records:
            raise ValueError(f"No records found in {self.csv_path}")

        self.feature_extractor = (
            feature_extractor
            or ASTFeatureExtractor.from_pretrained(MODEL_NAME)
        )

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]

        # Load original speech as 16 kHz mono.
        audio, _ = librosa.load(
            record["audio_path"],
            sr=SAMPLE_RATE,
            mono=True,
        )

        # Load the selected RIR channel.
        rir, rir_sample_rate = sf.read(
            record["rir_path"],
            dtype="float32",
            always_2d=True,
        )

        channel = int(record["rir_channel"])

        if channel >= rir.shape[1]:
            raise ValueError(
                f"RIR channel {channel} unavailable: {record['rir_path']}"
            )

        rir = rir[:, channel]

        # Resample RIR to match the speech sample rate.
        if rir_sample_rate != SAMPLE_RATE:
            rir = librosa.resample(
                y=rir,
                orig_sr=rir_sample_rate,
                target_sr=SAMPLE_RATE,
            )

        # Apply room reverberation.
        reverberated_audio = apply_reverberation(
            audio,
            rir,
            normalize=True,
        )

        # Extract AST input features.
        features = self.feature_extractor(
            reverberated_audio,
            sampling_rate=SAMPLE_RATE,
            return_tensors="pt",
        )

        input_values = features["input_values"].squeeze(0)

        all_acoustic_features = torch.tensor(
            [
                float(record["rt60_seconds"]),
                float(record["drr_db"]),
            ],
            dtype=torch.float32,
        )

        # Select RT60, DRR, or both.
        acoustic_features = all_acoustic_features[
            list(self.feature_indices)
        ]

        # Normalize using training-set statistics.
        acoustic_features = (
            acoustic_features - self.acoustic_mean[list(self.feature_indices)]
        ) / self.acoustic_std[list(self.feature_indices)]

        label = torch.tensor(
            int(record["label"]),
            dtype=torch.long,
        )

        return {
            "input_values": input_values,
            "acoustic_features": acoustic_features,
            "label": label,
        }