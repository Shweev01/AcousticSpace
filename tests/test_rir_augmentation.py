
from pathlib import Path

import numpy as np
import soundfile as sf

from audio_analysis.augmentation.rir_augmentation import augment_with_rir


audio_path = Path(
    r"C:\AcousticSpace_Dataset\LA\LA\ASVspoof2019_LA_dev"
    r"\flac\LA_D_1047731.flac"
)

rir_path = Path(
    r"C:\AcousticSpace_Dataset\real_rirs_isotropic_noises"
    r"\RWCP_type1_rir_circle_e1b_imp110.wav"
)

output_path = Path("data/processed/rir_augmented_demo.wav")


def main():
    audio, audio_sr = sf.read(audio_path)
    rir, rir_sr = sf.read(rir_path)

    if audio.ndim == 2:
        audio = audio[:, 0]

    if rir.ndim == 2:
        rir = rir[:, 0]

    if audio_sr != rir_sr:
        raise ValueError(
            f"Sample-rate mismatch: audio={audio_sr}, RIR={rir_sr}"
        )

    augmented = augment_with_rir(
        audio,
        rir,
        normalize=True,
        target_peak=0.99,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(output_path, augmented, audio_sr)

    print(f"Original duration: {len(audio) / audio_sr:.3f} seconds")
    print(f"Augmented duration: {len(augmented) / audio_sr:.3f} seconds")
    print(f"Sample rate: {audio_sr} Hz")
    print(f"Output peak: {np.max(np.abs(augmented)):.4f}")
    print(f"Finite output: {np.isfinite(augmented).all()}")
    print(f"Saved to: {output_path}")


if __name__ == "__main__":
    main()
