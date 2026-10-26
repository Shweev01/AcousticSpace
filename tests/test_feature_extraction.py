from audio_analysis.feature_extraction import extract_audio_features
import numpy as np

from audio_analysis.feature_extraction import (
    extract_mel_spectrogram,
)


def test_mel_spectrogram_shape():
    """The spectrogram should contain 80 Mel bands."""
    sample_rate = 16000
    duration = 3
    audio = np.zeros(sample_rate * duration, dtype=np.float32)

    # Add a signal so the test isn't using silence alone.
    time = np.arange(len(audio)) / sample_rate
    audio = 0.5 * np.sin(2 * np.pi * 440 * time)

    mel = extract_mel_spectrogram(audio, sample_rate)

    assert mel.shape[0] == 80
    assert mel.shape[1] > 0
    assert np.isfinite(mel).all()

    print("PASS: Mel spectrogram has 80 bands and valid values.")


def test_mel_spectrogram_rejects_empty_audio():
    """Empty audio should be rejected."""
    try:
        extract_mel_spectrogram(
            np.array([], dtype=np.float32),
            sample_rate=16000,
        )
    except ValueError:
        print("PASS: Empty audio is rejected.")
    else:
        raise AssertionError("Expected ValueError for empty audio.")


def test_extract_features_from_real_audio():
    """Extract waveform and Mel spectrogram from a real ASVspoof file."""
    from pathlib import Path

    audio_path = Path(
        r"C:\AcousticSpace_Dataset\LA\LA\ASVspoof2019_LA_train"
        r"\flac\LA_T_1138215.flac"
    )

    assert audio_path.is_file(), f"Audio file not found: {audio_path}"

    features = extract_audio_features(str(audio_path))

    waveform = features["waveform"]
    mel = features["mel_spectrogram"]

    assert features["sample_rate"] == 16000
    assert waveform.ndim == 1
    assert waveform.size > 0
    assert mel.shape[0] == 80
    assert mel.shape[1] > 0
    assert np.isfinite(waveform).all()
    assert np.isfinite(mel).all()

    print("PASS: Real ASVspoof audio processed successfully.")
    print("Waveform shape:", waveform.shape)
    print("Mel spectrogram shape:", mel.shape)
    print("Duration:", round(features["duration"], 2), "seconds")


if __name__ == "__main__":
    test_mel_spectrogram_shape()
    test_mel_spectrogram_rejects_empty_audio()
    test_extract_features_from_real_audio()
    print("All feature extraction tests passed.")
