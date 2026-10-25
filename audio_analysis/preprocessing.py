
import librosa
import numpy as np


def load_audio(file_path: str, target_sr: int = 16000):
    """
    Load audio as mono, resample it, and validate its duration.

    Args:
        file_path: Path to a WAV or FLAC file.
        target_sr: Desired sample rate in Hz.

    Returns:
        audio: Mono audio waveform as a NumPy array.
        sample_rate: Target sample rate.
    """
    audio, sample_rate = librosa.load(
        file_path,
        sr=target_sr,
        mono=True,
    )

    if audio.size == 0:
        raise ValueError("The audio file is empty.")

    duration = len(audio) / sample_rate

    if duration < 1.0:
        raise ValueError(
            f"Audio is too short: {duration:.2f} seconds. "
            "Minimum duration is 1.0 second."
        )

    return audio, sample_rate
