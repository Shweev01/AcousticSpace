
import numpy as np

from audio_analysis.rir_analysis.reverberation import apply_reverberation


def test_reverberation_output_length():
    audio = np.zeros(1000, dtype=np.float32)
    audio[0] = 1.0

    rir = np.zeros(200, dtype=np.float32)
    rir[0] = 1.0
    rir[100] = 0.5

    output = apply_reverberation(audio, rir)

    assert len(output) == len(audio) + len(rir) - 1
    print("PASS: Full convolution preserves the expected output length.")


def test_reverberation_preserves_tail():
    audio = np.zeros(1000, dtype=np.float32)
    audio[0] = 1.0

    rir = np.zeros(200, dtype=np.float32)
    rir[0] = 1.0
    rir[150] = 0.5

    output = apply_reverberation(audio, rir)

    assert np.abs(output[150]) > 0
    print("PASS: Reverberant tail is preserved.")


def test_reverberation_rejects_invalid_inputs():
    for audio, rir in [
        (np.array([]), np.ones(10)),
        (np.ones(10), np.array([])),
        (np.zeros(10), np.ones(10)),
        (np.ones(10), np.zeros(10)),
    ]:
        try:
            apply_reverberation(audio, rir)
        except ValueError:
            continue
        raise AssertionError("Invalid input should raise ValueError.")

    print("PASS: Invalid and silent inputs are rejected.")


def test_reverberation_output_is_finite():
    audio = np.ones(1000, dtype=np.float32)
    rir = np.zeros(100, dtype=np.float32)
    rir[0] = 1.0
    rir[10] = 0.2

    output = apply_reverberation(audio, rir)

    assert np.isfinite(output).all()
    print("PASS: Output contains only finite values.")


def test_normalization_controls_peak():
    audio = np.ones(1000, dtype=np.float32)
    rir = np.ones(100, dtype=np.float32)

    output = apply_reverberation(
        audio,
        rir,
        normalize=True,
        target_peak=0.8,
    )

    assert np.max(np.abs(output)) <= 0.800001
    assert np.isclose(np.max(np.abs(output)), 0.8, atol=1e-6)
    print("PASS: Normalization controls the output peak.")


def test_normalization_is_optional():
    audio = np.ones(100, dtype=np.float32)
    rir = np.ones(20, dtype=np.float32)

    raw_output = apply_reverberation(audio, rir)
    normalized_output = apply_reverberation(
        audio, rir, normalize=True
    )

    assert np.max(np.abs(raw_output)) > 1.0
    assert np.max(np.abs(normalized_output)) <= 1.0
    print("PASS: Normalization is optional.")


def test_invalid_target_peak_is_rejected():
    audio = np.ones(10, dtype=np.float32)
    rir = np.ones(5, dtype=np.float32)

    for target_peak in [0, -0.1, 1.1, np.inf, np.nan]:
        try:
            apply_reverberation(
                audio, rir, normalize=True, target_peak=target_peak
            )
        except ValueError:
            continue
        raise AssertionError("Invalid target_peak should be rejected.")

    print("PASS: Invalid normalization targets are rejected.")


if __name__ == "__main__":
    test_reverberation_output_length()
    test_reverberation_preserves_tail()
    test_reverberation_rejects_invalid_inputs()
    test_reverberation_output_is_finite()
    test_normalization_controls_peak()
    test_normalization_is_optional()
    test_invalid_target_peak_is_rejected()
