
import numpy as np
import soundfile as sf

from audio_analysis.rir_analysis.reverberation import apply_reverberation
from audio_analysis.rir_analysis.rt60 import estimate_rt60


AUDIO_PATH = (
    r"C:\AcousticSpace_Dataset\LA\LA\ASVspoof2019_LA_train"
    r"\flac\LA_T_1138215.flac"
)


def create_synthetic_rir(
    rt60_seconds: float = 0.4,
    sample_rate: int = 16000,
    duration: float = 1.2,
) -> np.ndarray:
    rng = np.random.default_rng(42)
    time = np.arange(int(sample_rate * duration)) / sample_rate

    # Exponential amplitude decay with a diffuse noise tail.
    envelope = np.exp(-np.log(1000) * time / rt60_seconds)
    rir = rng.normal(0, 1, len(time)) * envelope

    # Add a strong direct arrival.
    rir[0] += 1.0
    return rir.astype(np.float32)


def main():
    audio, sample_rate = sf.read(AUDIO_PATH)

    if audio.ndim == 2:
        audio = audio.mean(axis=1)

    if sample_rate != 16000:
        raise ValueError(
            f"Expected 16000 Hz audio, received {sample_rate} Hz."
        )

    rir = create_synthetic_rir(sample_rate=sample_rate)
    reverberated = apply_reverberation(
    audio,
    rir,
    normalize=True,
    target_peak=0.99,
)

    assert np.isfinite(reverberated).all()
    assert len(reverberated) == len(audio) + len(rir) - 1

    print(f"Input duration: {len(audio) / sample_rate:.2f} s")
    print(f"RIR duration: {len(rir) / sample_rate:.2f} s")
    print(f"Output duration: {len(reverberated) / sample_rate:.2f} s")
    print(f"Input peak: {np.max(np.abs(audio)):.4f}")
    print(f"Output peak: {np.max(np.abs(reverberated)):.4f}")
    print("PASS: Real speech was reverberated successfully.")

def test_real_rir_augmentation():
    import numpy as np
    import soundfile as sf

    from audio_analysis.preprocessing import load_audio
    from audio_analysis.augmentation.rir_augmentation import augment_with_rir

    speech_path = AUDIO_PATH
    
    rir_path = (
        r"C:\AcousticSpace_Dataset\real_rirs_isotropic_noises"
        r"\RWCP_type1_rir_circle_e1b_imp110.wav"
    )

    speech, sample_rate = load_audio(speech_path)
    rir, rir_sample_rate = sf.read(rir_path)

    assert sample_rate == rir_sample_rate
    assert rir.ndim == 2

    augmented = augment_with_rir(speech, rir[:, 0])

    assert augmented.ndim == 1
    assert len(augmented) == len(speech) + len(rir) - 1
    assert np.isfinite(augmented).all()
    assert np.max(np.abs(augmented)) <= 1.0
    assert len(augmented) > len(speech)


if __name__ == "__main__":
    main()
