
import numpy as np


def estimate_drr(
    rir: np.ndarray,
    sample_rate: int = 16000,
    direct_window_ms: float = 2.5,
) -> float:
    """
    Estimate the Direct-to-Reverberant Ratio (DRR) in decibels.

    The direct region is defined as a short window beginning at the
    first significant RIR sample. Energy after that window is treated
    as reverberant energy.

    Note:
        This is a simplified estimator. The direct-window length and
        arrival detection can affect the result.

    Returns:
        DRR in dB.
    """

    rir = np.asarray(rir, dtype=float)

    if rir.ndim != 1 or rir.size == 0:
        raise ValueError("RIR must be a non-empty one-dimensional array.")

    if not np.isfinite(rir).all():
        raise ValueError("RIR contains invalid values.")

    if sample_rate <= 0:
        raise ValueError("Sample rate must be positive.")

    if direct_window_ms <= 0:
        raise ValueError("Direct window must be positive.")

    if np.max(np.abs(rir)) == 0:
        raise ValueError("RIR cannot contain only zeros.")

    # Find the first sample above 1% of the peak amplitude.
    threshold = 0.01 * np.max(np.abs(rir))
    arrivals = np.flatnonzero(np.abs(rir) >= threshold)

    if arrivals.size == 0:
        raise ValueError("Could not detect the direct sound arrival.")

    arrival_index = int(arrivals[0])
    window_samples = max(
        1, int(round(sample_rate * direct_window_ms / 1000))
    )
    direct_end = min(arrival_index + window_samples, len(rir))

    direct_energy = float(np.sum(rir[arrival_index:direct_end] ** 2))

    # Count energy outside the direct window as reverberant energy.
    reverberant_energy = float(
        np.sum(rir**2) - direct_energy
    )

    if direct_energy <= 0:
        raise ValueError("Direct sound energy must be positive.")

    if reverberant_energy <= 0:
        raise ValueError("Reverberant energy must be positive.")

    return float(10 * np.log10(direct_energy / reverberant_energy))
