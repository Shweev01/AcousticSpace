
import numpy as np

from audio_analysis.rir_analysis.rt60 import estimate_rt60


def create_synthetic_rir(
    rt60_seconds: float = 0.5,
    sample_rate: int = 16000,
    duration: float = 1.0,
) -> np.ndarray:
    """Generate a synthetic RIR with an exponentially decaying tail."""

    rng = np.random.default_rng(42)
    time = np.arange(int(sample_rate * duration)) / sample_rate

    # Amplitude envelope designed to decay by 60 dB at the target RT60.
    envelope = np.exp(-np.log(1000) * time / rt60_seconds)
    rir = rng.normal(0, 1, len(time)) * envelope

    # Add a strong direct sound at the beginning.
    rir[0] += 1.0

    return rir


def test_rt60_estimation():
    known_rt60 = 0.5
    sample_rate = 16000

    rir = create_synthetic_rir(
        rt60_seconds=known_rt60,
        sample_rate=sample_rate,
        duration=1.0,
    )

    estimated_rt60 = estimate_rt60(rir, sample_rate)

    print(f"Known RT60:     {known_rt60:.3f} seconds")
    print(f"Estimated RT60: {estimated_rt60:.3f} seconds")

    # Allow some error because this is a simplified synthetic RIR.
    assert abs(estimated_rt60 - known_rt60) < 0.15, (
        "RT60 estimate is outside the expected tolerance."
    )

    print("PASS: RT60 estimation is within tolerance.")


def test_rt60_rejects_empty_rir():
    try:
        estimate_rt60(np.array([]))
    except ValueError:
        print("PASS: Empty RIR is rejected.")
    else:
        raise AssertionError("Empty RIR should be rejected.")


def test_rt60_rejects_silent_rir():
    try:
        estimate_rt60(np.zeros(16000))
    except ValueError:
        print("PASS: Silent RIR is rejected.")
    else:
        raise AssertionError("Silent RIR should be rejected.")


def test_rt60_rejects_invalid_sample_rate():
    try:
        estimate_rt60(np.ones(16000), sample_rate=0)
    except ValueError:
        print("PASS: Invalid sample rate is rejected.")
    else:
        raise AssertionError("Invalid sample rate should be rejected.")


def test_rt60_multiple_values():
    sample_rate = 16000

    for known_rt60 in [0.3, 0.5, 0.8]:
        rir = create_synthetic_rir(
            rt60_seconds=known_rt60,
            sample_rate=sample_rate,
            duration=1.5,
        )

        estimated_rt60 = estimate_rt60(rir, sample_rate)

        print(
            f"Known: {known_rt60:.3f}s | "
            f"Estimated: {estimated_rt60:.3f}s"
        )

if __name__ == "__main__":
    test_rt60_estimation()
    test_rt60_rejects_empty_rir()
    test_rt60_rejects_silent_rir()
    test_rt60_rejects_invalid_sample_rate()
    test_rt60_multiple_values()
