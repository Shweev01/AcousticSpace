
import librosa
import numpy as np


def load_audio(file_path: str, target_sr: int = 16000):
    """
    Load an audio file as mono and resample it to the target sample rate.

    Args:
        file_path: Path to a WAV or FLAC audio file.
        target_sr: Desired sample rate in Hz (default: 16000).

    Returns:
        audio: Mono audio waveform as a NumPy array.
        sample_rate: The target sample rate.
    """
    audio, sample_rate = librosa.load(
        file_path,
        sr=target_sr,
        mono=True
    )

    if audio.size == 0:
        raise ValueError("The audio file is empty.")

    return audio, sample_rate
