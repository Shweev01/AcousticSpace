
import numpy as np

from audio_analysis.rir_analysis.reverberation import apply_reverberation


def augment_with_rir(
    audio: np.ndarray,
    rir: np.ndarray,
    normalize: bool = True,
    target_peak: float = 0.99,
) -> np.ndarray:
    """
    Apply a Room Impulse Response to speech audio.

    Returns a reverberated waveform.
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

    if normalize:
        if not np.isfinite(target_peak) or target_peak <= 0:
            raise ValueError("Target peak must be positive and finite.")

    augmented_audio = apply_reverberation(
        audio,
        rir,
        normalize=normalize,
        target_peak=target_peak,
    )

    if not np.isfinite(augmented_audio).all():
        raise ValueError("Augmented audio contains invalid values.")

    return augmented_audio
