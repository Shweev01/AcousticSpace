from pathlib import Path

import librosa
import torch
from torch.utils.data import Dataset
from transformers import ASTFeatureExtractor

from audio_analysis.dataset_loader import (
    load_training_dataset,
    load_validation_dataset,
)


MODEL_NAME = "MIT/ast-finetuned-audioset-10-10-0.4593"

LABEL_MAP = {
    "bonafide": 0,
    "spoof": 1,
}


class ASTAudioDataset(Dataset):

    def __init__(self, dataframe, feature_extractor):
        self.dataframe = dataframe.reset_index(drop=True)
        self.feature_extractor = feature_extractor

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index):
        row = self.dataframe.iloc[index]

        waveform, sample_rate = librosa.load(
            row["file_path"],
            sr=16000,
            mono=True,
        )

        inputs = self.feature_extractor(
            waveform,
            sampling_rate=sample_rate,
            return_tensors="pt",
        )

        input_values = inputs["input_values"].squeeze(0)

        label = LABEL_MAP[row["label"]]

        return {
            "input_values": input_values,
            "labels": torch.tensor(label, dtype=torch.long),
        }


def create_ast_datasets():
    feature_extractor = ASTFeatureExtractor.from_pretrained(
        MODEL_NAME
    )

    train_df = load_training_dataset()
    dev_df = load_validation_dataset()

    train_dataset = ASTAudioDataset(
        train_df,
        feature_extractor,
    )

    dev_dataset = ASTAudioDataset(
        dev_df,
        feature_extractor,
    )

    return train_dataset, dev_dataset