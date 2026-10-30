
import librosa
import numpy as np


def load_audio(file_path: str, target_sr: int = 16000):
    """Load audio as mono, resample it, and validate the waveform."""

    if target_sr <= 0:
        raise ValueError("Target sample rate must be positive.")

    audio, sample_rate = librosa.load(
        file_path,
        sr=target_sr,
        mono=True,
    )

    if audio.size == 0:
        raise ValueError("The audio file is empty.")

    if not np.isfinite(audio).all():
        raise ValueError("Audio contains invalid values.")

    return audio, sample_rate
