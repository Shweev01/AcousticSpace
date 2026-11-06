
from pathlib import Path

import numpy as np
import soundfile as sf

from audio_analysis.visualization import plot_rir_analysis


# Path to the real RIR recording.
rir_path = Path(
    r"C:\AcousticSpace_Dataset\real_rirs_isotropic_noises"
    r"\RWCP_type1_rir_circle_e1b_imp110.wav"
)


def main() -> None:
    # Check whether the RIR file exists.
    if not rir_path.exists():
        raise FileNotFoundError(f"RIR file not found: {rir_path}")

    # Load the real RIR.
    rir, sample_rate = sf.read(rir_path)

    # Select channel 0 if the recording is multichannel.
    if rir.ndim == 2:
        rir = rir[:, 0]

    rir = np.asarray(rir, dtype=np.float64)

    if rir.size == 0:
        raise ValueError("Loaded RIR is empty.")

    if not np.isfinite(rir).all():
        raise ValueError("RIR contains invalid values.")

    # Basic recording information.
    print(f"Sample rate: {sample_rate} Hz")
    print(f"RIR duration: {len(rir) / sample_rate:.3f} seconds")
    print(
        f"Direct peak time: "
        f"{np.argmax(np.abs(rir)) / sample_rate:.5f} seconds"
    )

    # Plot RIR waveform and Schroeder decay curve.
    # RT60 and DRR are calculated inside this reusable function.
    plot_rir_analysis(rir, sample_rate)


if __name__ == "__main__":
    main()
