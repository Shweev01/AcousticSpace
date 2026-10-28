
import numpy as np

from audio_analysis.rir_analysis.reverberation import apply_reverberation


def augment_with_rir(
    audio: np.ndarray,
    rir: np.ndarray,
    normalize: bool = True,
    target_peak: float = 0.99,
) -> np.ndarray:
    """
    Create an augmented audio sample by applying a room impulse response.

    Args:
        audio: One-dimensional speech waveform.
        rir: One-dimensional room impulse response.
        normalize: Whether to normalize the output peak.
        target_peak: Desired peak amplitude when normalization is enabled.

    Returns:
        Reverberated audio as a NumPy array.
    """
    audio = np.asarray(audio, dtype=np.float32)
    rir = np.asarray(rir, dtype=np.float32)

    if audio.ndim != 1 or audio.size == 0:
        raise ValueError("Audio must be a non-empty 1D array.")

    if rir.ndim != 1 or rir.size == 0:
        raise ValueError("RIR must be a non-empty 1D array.")

    if not np.isfinite(audio).all():
        raise ValueError("Audio contains invalid values.")

    if not np.isfinite(rir).all():
        raise ValueError("RIR contains invalid values.")

    return apply_reverberation(
        audio,
        rir,
        normalize=normalize,
        target_peak=target_peak,
    )
