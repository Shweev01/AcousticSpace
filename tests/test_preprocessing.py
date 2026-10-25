
from audio_analysis.preprocessing import load_audio

audio, sr = load_audio(
    r"C:\AcousticSpace\tests\test_audio.wav"
)

print("Sample rate:", sr)
print("Audio shape:", audio.shape)
print("Duration:", len(audio) / sr, "seconds")
print("Audio loaded successfully!")
