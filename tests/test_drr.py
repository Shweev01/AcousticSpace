
import numpy as np

from audio_analysis.rir_analysis.drr import estimate_drr


def test_drr_known_energy_ratio():
    sample_rate = 16000
    rir = np.zeros(sample_rate, dtype=float)

    # Direct sound: energy = 10.
    rir[0] = np.sqrt(10.0)

    # Reverberant sound: total energy = 1.
    rir[100] = 1.0

    estimated_drr = estimate_drr(
        rir,
        sample_rate=sample_rate,
        direct_window_ms=2.5,
    )

    expected_drr = 10 * np.log10(10.0 / 1.0)

    print(f"Expected DRR:  {expected_drr:.3f} dB")
    print(f"Estimated DRR: {estimated_drr:.3f} dB")

    assert abs(estimated_drr - expected_drr) < 0.01
    print("PASS: DRR matches the known energy ratio.")


def test_drr_rejects_empty_rir():
    try:
        estimate_drr(np.array([]))
    except ValueError:
        print("PASS: Empty RIR is rejected.")
    else:
        raise AssertionError("Empty RIR should be rejected.")


def test_drr_rejects_silent_rir():
    try:
        estimate_drr(np.zeros(16000))
    except ValueError:
        print("PASS: Silent RIR is rejected.")
    else:
        raise AssertionError("Silent RIR should be rejected.")


def test_drr_rejects_invalid_sample_rate():
    try:
        estimate_drr(np.array([1.0, 0.5]), sample_rate=0)
    except ValueError:
        print("PASS: Invalid sample rate is rejected.")
    else:
        raise AssertionError("Invalid sample rate should be rejected.")


if __name__ == "__main__":
    test_drr_known_energy_ratio()
    test_drr_rejects_empty_rir()
    test_drr_rejects_silent_rir()
    test_drr_rejects_invalid_sample_rate()
    print("All DRR tests passed.")
