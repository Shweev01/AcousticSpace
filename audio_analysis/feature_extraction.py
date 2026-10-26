
import librosa
import numpy as np

from audio_analysis.preprocessing import load_audio


def extract_mel_spectrogram(
    audio: np.ndarray,
    sample_rate: int = 16000,
    n_mels: int = 80,
    n_fft: int = 1024,
    hop_length: int = 256,
) -> np.ndarray:
    """Convert a waveform into a log-scaled Mel spectrogram."""

    if audio.size == 0:
        raise ValueError("Audio waveform is empty.")

    if sample_rate <= 0:
        raise ValueError("Sample rate must be positive.")

    if n_mels <= 0 or n_fft <= 0 or hop_length <= 0:
        raise ValueError("Spectrogram parameters must be positive.")

    mel_power = librosa.feature.melspectrogram(
        y=audio,
        sr=sample_rate,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        power=2.0,
    )

    mel_db = librosa.power_to_db(mel_power, ref=np.max)

    if not np.isfinite(mel_db).all():
        raise ValueError("Mel spectrogram contains invalid values.")

    return mel_db


def extract_audio_features(file_path: str) -> dict:
    """Load an audio file and extract its waveform and Mel spectrogram."""

    waveform, sample_rate = load_audio(file_path)

    mel_spectrogram = extract_mel_spectrogram(
        audio=waveform,
        sample_rate=sample_rate,
    )

    return {
        "waveform": waveform,
        "sample_rate": sample_rate,
        "mel_spectrogram": mel_spectrogram,
        "duration": len(waveform) / sample_rate,
    }
