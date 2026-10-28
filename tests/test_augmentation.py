
import numpy as np

from audio_analysis.augmentation.rir_augmentation import augment_with_rir


def test_augmentation_output_length():
    audio = np.zeros(1000, dtype=np.float32)
    audio[0] = 1.0

    rir = np.zeros(200, dtype=np.float32)
    rir[0] = 1.0
    rir[100] = 0.5

    output = augment_with_rir(audio, rir)

    expected_length = len(audio) + len(rir) - 1
    assert len(output) == expected_length
    print("PASS: Augmented audio has the expected length.")


def test_augmentation_normalization():
    audio = np.ones(1000, dtype=np.float32)
    rir = np.ones(100, dtype=np.float32)

    output = augment_with_rir(
        audio,
        rir,
        normalize=True,
        target_peak=0.8,
    )

    assert np.isclose(np.max(np.abs(output)), 0.8, atol=1e-6)
    print("PASS: Augmented audio peak is normalized.")


def test_augmentation_rejects_invalid_inputs():
    invalid_cases = [
        (np.array([]), np.ones(10)),
        (np.ones(10), np.array([])),
        (np.zeros(10), np.ones(10)),
        (np.ones(10), np.zeros(10)),
        (np.array([1.0, np.nan]), np.ones(10)),
    ]

    for audio, rir in invalid_cases:
        try:
            augment_with_rir(audio, rir)
        except ValueError:
            continue

        raise AssertionError("Invalid input should raise ValueError.")

    print("PASS: Invalid and silent inputs are rejected.")


def test_augmentation_output_is_finite():
    audio = np.ones(1000, dtype=np.float32)
    rir = np.zeros(100, dtype=np.float32)
    rir[0] = 1.0
    rir[10] = 0.2

    output = augment_with_rir(audio, rir)

    assert np.isfinite(output).all()
    print("PASS: Augmented output contains only finite values.")


if __name__ == "__main__":
    test_augmentation_output_length()
    test_augmentation_normalization()
    test_augmentation_rejects_invalid_inputs()
    test_augmentation_output_is_finite()
    print("\nAll augmentation tests passed.")
