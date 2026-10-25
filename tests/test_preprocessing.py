
from pathlib import Path

import numpy as np
import soundfile as sf

from audio_analysis.preprocessing import load_audio


test_dir = Path(__file__).parent
sample_rate = 22050

duration = 3
t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
audio = 0.5 * np.sin(2 * np.pi * 440 * t)

valid_path = test_dir / "test_audio.wav"
sf.write(valid_path, audio, sample_rate)

loaded_audio, sr = load_audio(str(valid_path))

assert sr == 16000
assert loaded_audio.ndim == 1
assert abs(len(loaded_audio) / sr - 3.0) < 0.02

print("PASS: Valid audio loads at 16 kHz in mono.")


short_path = test_dir / "test_audio_short.wav"
sf.write(short_path, audio[:int(0.5 * sample_rate)], sample_rate)

try:
    load_audio(str(short_path))
except ValueError as error:
    assert "too short" in str(error).lower()
    print("PASS: Audio shorter than one second is rejected.")
else:
    raise AssertionError("Short audio should have been rejected.")
