
import numpy as np
from scipy.signal import fftconvolve


def apply_reverberation(
    audio: np.ndarray,
    rir: np.ndarray,
    normalize: bool = False,
    target_peak: float = 0.99,
) -> np.ndarray:
    """
    Apply a room impulse response to an audio signal.

    Full convolution preserves the reverberant tail.

    Args:
        audio: One-dimensional input audio.
        rir: One-dimensional room impulse response.
        normalize: If True, scale the output to target_peak.
        target_peak: Desired maximum absolute output amplitude.

    Returns:
        Reverberated audio as a NumPy array.
    """
    audio = np.asarray(audio, dtype=np.float64)
    rir = np.asarray(rir, dtype=np.float64)

    if audio.ndim != 1 or audio.size == 0:
        raise ValueError("Audio must be a non-empty 1D array.")

    if rir.ndim != 1 or rir.size == 0:
        raise ValueError("RIR must be a non-empty 1D array.")

    if not np.isfinite(audio).all():
        raise ValueError("Audio contains invalid values.")

    if not np.isfinite(rir).all():
        raise ValueError("RIR contains invalid values.")

    if not np.any(audio):
        raise ValueError("Audio cannot contain only zeros.")

    if not np.any(rir):
        raise ValueError("RIR cannot contain only zeros.")

    if not isinstance(normalize, (bool, np.bool_)):
        raise ValueError("normalize must be a boolean.")

    if not np.isfinite(target_peak) or not 0 < target_peak <= 1:
        raise ValueError("target_peak must be greater than 0 and at most 1.")

    reverberated = fftconvolve(audio, rir, mode="full")

    if normalize:
        peak = np.max(np.abs(reverberated))
        if peak > 0:
            reverberated = reverberated * (target_peak / peak)

    return reverberated.astype(np.float32)
