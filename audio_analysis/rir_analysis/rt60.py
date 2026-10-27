
import numpy as np


def estimate_rt60(rir: np.ndarray, sample_rate: int = 16000) -> float:
    """
    Estimate RT60 from a Room Impulse Response (RIR).

    Uses the Schroeder Energy Decay Curve and fits the
    decay between -5 dB and -35 dB to estimate RT60.

    Args:
        rir: Room impulse response as a NumPy array.
        sample_rate: Sampling rate of the RIR in Hz.

    Returns:
        Estimated RT60 in seconds.
    """

    if rir.size == 0:
        raise ValueError("RIR cannot be empty.")

    if sample_rate <= 0:
        raise ValueError("Sample rate must be positive.")

    if not np.isfinite(rir).all():
        raise ValueError("RIR contains invalid values.")

    if np.max(np.abs(rir)) == 0:
        raise ValueError("RIR cannot contain only zeros.")

    # Calculate the reverse cumulative energy (Schroeder method).
    energy = rir**2
    decay_curve = np.cumsum(energy[::-1])[::-1]

    # Convert energy decay to decibels.
    decay_db = 10 * np.log10(
        np.maximum(decay_curve / decay_curve[0], 1e-12)
    )

    # Fit the decay between -5 dB and -35 dB.
    time = np.arange(len(rir)) / sample_rate
    mask = (decay_db <= -5) & (decay_db >= -35)

    if np.count_nonzero(mask) < 2:
        raise ValueError(
            "Insufficient decay range to estimate RT60 reliably."
        )

    slope, intercept = np.polyfit(time[mask], decay_db[mask], 1)

    if slope >= 0:
        raise ValueError("Unable to estimate a decreasing decay curve.")

    # Extrapolate the fitted decay to a 60 dB drop.
    rt60 = -60 / slope

    return float(rt60)
