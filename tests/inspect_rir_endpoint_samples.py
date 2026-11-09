
from pathlib import Path

import numpy as np
import soundfile as sf


DATASET_ROOT = Path(
    r"C:\AcousticSpace_Dataset\real_rirs_isotropic_noises"
)

FILES = [
    "air_type1_air_phone_bt_corridor_hhp.wav",
    "air_type1_air_phone_bt_stairway_hhp.wav",
    "air_type1_air_phone_bt_office_hhp.wav",
]

SAMPLES_TO_DISPLAY = 30


def inspect_file(filename):
    path = DATASET_ROOT / filename

    audio, sample_rate = sf.read(str(path))

    if audio.ndim == 1:
        audio = audio[:, np.newaxis]

    print(f"\n{'=' * 70}")
    print(f"File: {filename}")
    print(f"Sample rate: {sample_rate}")
    print(f"Shape: {audio.shape}")

    for channel_index in range(audio.shape[1]):

        signal = audio[:, channel_index]
        final_samples = signal[-SAMPLES_TO_DISPLAY:]

        start_index = len(signal) - SAMPLES_TO_DISPLAY

        print(f"\nChannel {channel_index}")
        print("Last 30 samples:")

        for i, value in enumerate(final_samples):
            sample_index = start_index + i
            time = sample_index / sample_rate

            print(
                f"Sample {sample_index:7d} | "
                f"Time {time:.7f}s | "
                f"Amplitude {value:+.8f}"
            )


def main():
    for filename in FILES:
        inspect_file(filename)


if __name__ == "__main__":
    main()
