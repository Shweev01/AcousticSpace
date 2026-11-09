
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

ENDPOINT_SECONDS = 0.3
MAX_LAG_MS = 5


def load_endpoint(filename, channel_index):
    path = DATASET_ROOT / filename
    audio, sample_rate = sf.read(str(path))

    if audio.ndim == 1:
        audio = audio[:, np.newaxis]

    if channel_index >= audio.shape[1]:
        raise ValueError(f"Channel {channel_index} unavailable in {filename}")

    endpoint_samples = int(ENDPOINT_SECONDS * sample_rate)

    signal = audio[-endpoint_samples:, channel_index].astype(np.float64)

    return signal, sample_rate


def pearson_correlation(signal_a, signal_b):
    """Calculate Pearson correlation between two equal-length signals."""

    if len(signal_a) != len(signal_b):
        raise ValueError("Signals must have equal lengths.")

    if np.std(signal_a) == 0 or np.std(signal_b) == 0:
        return float("nan")

    return float(np.corrcoef(signal_a, signal_b)[0, 1])


def best_lag_correlation(signal_a, signal_b, max_lag):
    """
    Find the highest Pearson correlation within a limited lag range.
    Positive lag means signal_b is shifted relative to signal_a.
    """

    best_correlation = -np.inf
    best_lag = 0

    for lag in range(-max_lag, max_lag + 1):

        if lag > 0:
            a = signal_a[:-lag]
            b = signal_b[lag:]
        elif lag < 0:
            a = signal_a[-lag:]
            b = signal_b[:lag]
        else:
            a = signal_a
            b = signal_b

        correlation = pearson_correlation(a, b)

        if np.isfinite(correlation) and correlation > best_correlation:
            best_correlation = correlation
            best_lag = lag

    return float(best_correlation), best_lag


def compare_channels(channel_index):

    signals = {}

    print(f"\n{'=' * 65}")
    print(f"CHANNEL {channel_index}")
    print(f"{'=' * 65}")

    for filename in FILES:
        signal, sample_rate = load_endpoint(filename, channel_index)
        signals[filename] = signal

    sample_lengths = {len(signal) for signal in signals.values()}

    if len(sample_lengths) != 1:
        raise ValueError("Endpoint lengths differ between recordings.")

    sample_rate = 16000
    max_lag = int(MAX_LAG_MS / 1000 * sample_rate)

    for i in range(len(FILES)):
        for j in range(i + 1, len(FILES)):

            file_a = FILES[i]
            file_b = FILES[j]

            signal_a = signals[file_a]
            signal_b = signals[file_b]

            zero_lag = pearson_correlation(signal_a, signal_b)

            best_corr, best_lag = best_lag_correlation(
                signal_a,
                signal_b,
                max_lag
            )

            print(f"\n{file_a}")
            print(f"vs {file_b}")
            print(f"Zero-lag correlation: {zero_lag:.6f}")
            print(f"Best correlation: {best_corr:.6f}")
            print(
                f"Best lag: {best_lag} samples "
                f"({best_lag / sample_rate * 1000:.3f} ms)"
            )


def main():

    for channel_index in (0, 1):
        compare_channels(channel_index)


if __name__ == "__main__":
    main()
