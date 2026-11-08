
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

REGIONS = {
    "First 290 ms": (0.0, 0.290),
    "Final 10 ms": (0.290, 0.300),
}


def load_region(filename, channel, start, end):
    audio, sr = sf.read(str(DATASET_ROOT / filename))

    if audio.ndim == 1:
        audio = audio[:, np.newaxis]

    start_sample = int(start * sr)
    end_sample = int(end * sr)

    return audio[start_sample:end_sample, channel].astype(np.float64)


def correlation(a, b):
    if len(a) != len(b):
        raise ValueError("Signal lengths do not match.")

    if np.std(a) == 0 or np.std(b) == 0:
        return float("nan")

    return float(np.corrcoef(a, b)[0, 1])


def main():
    for channel in (0, 1):
        print(f"\n{'=' * 55}")
        print(f"CHANNEL {channel}")
        print(f"{'=' * 55}")

        for region_name, (start, end) in REGIONS.items():
            print(f"\n{region_name}")

            signals = {
                filename: load_region(filename, channel, start, end)
                for filename in FILES
            }

            for i in range(len(FILES)):
                for j in range(i + 1, len(FILES)):
                    file_a = FILES[i]
                    file_b = FILES[j]

                    value = correlation(
                        signals[file_a],
                        signals[file_b]
                    )

                    print(
                        f"{file_a.split('_phone_')[1]} vs "
                        f"{file_b.split('_phone_')[1]}: {value:.6f}"
                    )


if __name__ == "__main__":
    main()
