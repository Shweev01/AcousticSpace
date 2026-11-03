
from pathlib import Path

import librosa
import numpy as np
import torch
from transformers import ASTFeatureExtractor, ASTForAudioClassification

from machine_learning.training.ast_dataset import MODEL_NAME


PROJECT_ROOT = Path(__file__).resolve().parents[2]

CHECKPOINT_PATH = (
    PROJECT_ROOT / "machine_learning" / "checkpoints" / "ast_best.pt"
)

LABEL2ID = {
    "bonafide": 0,
    "spoof": 1,
}

ID2LABEL = {
    0: "bonafide",
    1: "spoof",
}


class ASTPredictor:

    def __init__(self):

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        if not CHECKPOINT_PATH.is_file():
            raise FileNotFoundError(
                f"AST checkpoint not found: {CHECKPOINT_PATH}"
            )

        self.feature_extractor = ASTFeatureExtractor.from_pretrained(
            MODEL_NAME
        )

        self.model = ASTForAudioClassification.from_pretrained(
            MODEL_NAME,
            num_labels=2,
            id2label=ID2LABEL,
            label2id=LABEL2ID,
            ignore_mismatched_sizes=True,
        )

        checkpoint = torch.load(
            CHECKPOINT_PATH,
            map_location=self.device,
            weights_only=True,
        )

        self.model.load_state_dict(
            checkpoint["model_state_dict"]
        )

        self.model.to(self.device)
        self.model.eval()

        print(f"AST model loaded on: {self.device}")

    def predict(self, audio_path):

        audio_path = Path(audio_path)

        if not audio_path.is_file():
            raise FileNotFoundError(
                f"Audio file not found: {audio_path}"
            )

        waveform, sample_rate = librosa.load(
            audio_path,
            sr=16000,
            mono=True,
        )

        if waveform.size == 0:
            raise ValueError("Audio file is empty.")

        if len(waveform) / sample_rate < 1.0:
            raise ValueError("Audio must be at least 1 second long.")

        if not np.isfinite(waveform).all():
            raise ValueError("Audio contains invalid values.")

        inputs = self.feature_extractor(
            waveform,
            sampling_rate=sample_rate,
            return_tensors="pt",
        )

        input_values = inputs["input_values"].to(self.device)

        with torch.inference_mode():

            with torch.autocast(
                device_type=self.device.type,
                dtype=torch.float16,
                enabled=(self.device.type == "cuda"),
            ):
                outputs = self.model(
                    input_values=input_values
                )

            probabilities = torch.softmax(
                outputs.logits.float(),
                dim=1,
            )[0]

        predicted_id = int(torch.argmax(probabilities).item())

        return {
            "prediction": ID2LABEL[predicted_id],
            "confidence": float(probabilities[predicted_id].item()),
            "probabilities": {
                ID2LABEL[index]: float(probabilities[index].item())
                for index in range(2)
            },
        }
