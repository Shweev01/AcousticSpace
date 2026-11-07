
from pathlib import Path

import librosa
import numpy as np
import torch
from transformers import ASTFeatureExtractor


MODEL_NAME = "MIT/ast-finetuned-audioset-10-10-0.4593"

AUDIO_PATH = Path(
    r"C:\AcousticSpace_Dataset\LA\LA\ASVspoof2019_LA_dev\flac"
)


def test_ast_preprocessing():

    audio_files = list(AUDIO_PATH.glob("*.flac"))

    if not audio_files:
        raise FileNotFoundError("No FLAC audio files found.")

    sample_path = audio_files[0]

    print("Sample audio:", sample_path.name)

    waveform, sample_rate = librosa.load(
        sample_path,
        sr=16000,
        mono=True,
    )

    print("\nAUDIO INFORMATION")
    print("Sampling rate:", sample_rate)
    print("Duration:", round(len(waveform) / sample_rate, 2), "seconds")
    print("Waveform shape:", waveform.shape)

    feature_extractor = ASTFeatureExtractor.from_pretrained(MODEL_NAME)

    inputs = feature_extractor(
        waveform,
        sampling_rate=sample_rate,
        return_tensors="pt",
    )

    input_values = inputs["input_values"]

    print("\nAST FEATURE INFORMATION")
    print("Input shape:", input_values.shape)
    print("Data type:", input_values.dtype)
    print("Minimum:", input_values.min().item())
    print("Maximum:", input_values.max().item())
    print("Contains NaN:", torch.isnan(input_values).any().item())
    print("Contains Inf:", torch.isinf(input_values).any().item())

    assert input_values.shape[1] == 1024
    assert input_values.shape[2] == 128
    assert torch.isfinite(input_values).all()

    print("\nAST preprocessing test passed.")


if __name__ == "__main__":
    test_ast_preprocessing()
