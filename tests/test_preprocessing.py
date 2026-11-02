
import numpy as np
import soundfile as sf
import pytest

from audio_analysis.preprocessing import load_audio


@pytest.fixture
def audio_signal():
    sample_rate = 22050
    duration = 3.0
    t = np.linspace(
        0,
        duration,
        int(sample_rate * duration),
        endpoint=False,
    )
    return 0.5 * np.sin(2 * np.pi * 440 * t), sample_rate


def test_load_audio_resamples_and_converts_to_mono(tmp_path, audio_signal):
    audio, sample_rate = audio_signal
    file_path = tmp_path / "test_audio.wav"
    sf.write(file_path, audio, sample_rate)

    loaded_audio, sr = load_audio(str(file_path))

    assert sr == 16000
    assert loaded_audio.ndim == 1
    assert abs(len(loaded_audio) / sr - 3.0) < 0.02


def test_load_audio_rejects_short_audio(tmp_path, audio_signal):
    audio, sample_rate = audio_signal
    short_audio = audio[:int(0.5 * sample_rate)]

    file_path = tmp_path / "test_audio_short.wav"
    sf.write(file_path, short_audio, sample_rate)

    with pytest.raises(ValueError, match="too short"):
        load_audio(str(file_path))
