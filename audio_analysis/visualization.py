
import numpy as np
import matplotlib.pyplot as plt

from audio_analysis.feature_extraction import extract_audio_features


def plot_audio_features(file_path: str) -> None:
    """Plot the waveform and Mel spectrogram of an audio file."""

    features = extract_audio_features(file_path)

    waveform = features["waveform"]
    sample_rate = features["sample_rate"]
    mel_spectrogram = features["mel_spectrogram"]
    duration = features["duration"]

    time_axis = np.arange(len(waveform)) / sample_rate

    fig, axes = plt.subplots(2, 1, figsize=(12, 8))

    # Waveform
    axes[0].plot(time_axis, waveform)
    axes[0].set_title("Audio Waveform")
    axes[0].set_xlabel("Time (seconds)")
    axes[0].set_ylabel("Amplitude")

    # Mel spectrogram
    image = axes[1].imshow(
        mel_spectrogram,
        origin="lower",
        aspect="auto",
        extent=[0, duration, 0, mel_spectrogram.shape[0]],
        cmap="magma",
    )
    axes[1].set_title("Mel Spectrogram")
    axes[1].set_xlabel("Time (seconds)")
    axes[1].set_ylabel("Mel Frequency Band")

    fig.colorbar(image, ax=axes[1], format="%+2.0f dB")
    fig.tight_layout()
    plt.show()
