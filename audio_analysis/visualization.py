
import numpy as np
import matplotlib.pyplot as plt

from audio_analysis.feature_extraction import extract_audio_features
from audio_analysis.rir_analysis.rt60 import estimate_rt60
from audio_analysis.rir_analysis.drr import estimate_drr

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


def plot_rir_analysis(rir: np.ndarray, sample_rate: int = 16000) -> None:
    """Plot RIR waveform and Schroeder energy decay with RT60 and DRR."""

    rir = np.asarray(rir, dtype=np.float64)

    if rir.ndim != 1 or rir.size == 0:
        raise ValueError("RIR must be a non-empty 1D array.")

    if not np.isfinite(rir).all():
        raise ValueError("RIR contains invalid values.")

    if np.max(np.abs(rir)) == 0:
        raise ValueError("RIR cannot contain only zeros.")

    if sample_rate <= 0:
        raise ValueError("Sample rate must be positive.")

    time_axis = np.arange(len(rir)) / sample_rate

    # Calculate Schroeder energy decay curve.
    energy = rir ** 2
    decay_curve = np.cumsum(energy[::-1])[::-1]

    decay_db = 10 * np.log10(
        np.maximum(decay_curve / decay_curve[0], 1e-12)
    )

    # Estimate RT60 and DRR.
    try:
        rt60 = estimate_rt60(rir, sample_rate)
        rt60_text = f"{rt60:.3f} s"
    except ValueError:
        rt60 = None
        rt60_text = "Unavailable"

    try:
        drr = estimate_drr(rir, sample_rate)
        drr_text = f"{drr:.3f} dB"
    except ValueError:
        drr = None
        drr_text = "Unavailable"

    print(f"Estimated RT60: {rt60_text}")
    print(f"Estimated DRR: {drr_text}")

    fig, axes = plt.subplots(2, 1, figsize=(12, 8))

    # RIR waveform.
    axes[0].plot(time_axis, rir)
    axes[0].set_title("Room Impulse Response (RIR)")
    axes[0].set_xlabel("Time (seconds)")
    axes[0].set_ylabel("Amplitude")
    axes[0].grid(True)

    # Schroeder energy decay curve.
    axes[1].plot(time_axis, decay_db, label="Energy decay")

    axes[1].axhline(-5, linestyle="--", label="-5 dB")
    axes[1].axhline(-35, linestyle="--", label="-35 dB")

    axes[1].set_title(
        f"Schroeder Energy Decay Curve | RT60: {rt60_text} | DRR: {drr_text}"
    )
    axes[1].set_xlabel("Time (seconds)")
    axes[1].set_ylabel("Relative Energy (dB)")
    axes[1].set_ylim(-60, 5)
    axes[1].legend()
    axes[1].grid(True)

    fig.tight_layout()
    plt.show()

   