from pathlib import Path

import librosa
import torch
from transformers import ASTForAudioClassification, ASTFeatureExtractor


MODEL_NAME = "MIT/ast-finetuned-audioset-10-10-0.4593"

AUDIO_PATH = Path(
    r"C:\AcousticSpace_Dataset\LA\LA\ASVspoof2019_LA_dev\flac"
)

ID2LABEL = {
    0: "bonafide",
    1: "spoof",
}

LABEL2ID = {
    "bonafide": 0,
    "spoof": 1,
}


def test_ast_model():

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("Using device:", device)

    audio_files = list(AUDIO_PATH.glob("*.flac"))

    if not audio_files:
        raise FileNotFoundError("No audio files found.")

    sample_path = audio_files[0]

    waveform, sample_rate = librosa.load(
        sample_path,
        sr=16000,
        mono=True,
    )

    feature_extractor = ASTFeatureExtractor.from_pretrained(
        MODEL_NAME
    )

    print("\nLoading pretrained AST model...")

    model = ASTForAudioClassification.from_pretrained(
        MODEL_NAME,
        num_labels=2,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
        ignore_mismatched_sizes=True,
    )

    model.to(device)
    model.eval()

    inputs = feature_extractor(
        waveform,
        sampling_rate=sample_rate,
        return_tensors="pt",
    )

    inputs = {
        key: value.to(device)
        for key, value in inputs.items()
    }

    print("\nRunning AST inference...")

    with torch.no_grad():
        outputs = model(**inputs)

    probabilities = torch.softmax(outputs.logits, dim=-1)

    predicted_class = torch.argmax(
        probabilities,
        dim=-1,
    ).item()

    print("\nAST MODEL RESULTS")
    print("Audio:", sample_path.name)
    print("Logits shape:", outputs.logits.shape)
    print("Bonafide probability:", probabilities[0][0].item())
    print("Spoof probability:", probabilities[0][1].item())
    print("Predicted class:", ID2LABEL[predicted_class])

    assert outputs.logits.shape == (1, 2)
    assert torch.isfinite(outputs.logits).all()

    print("\nAST model test passed.")


if __name__ == "__main__":
    test_ast_model()