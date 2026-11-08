
"""
Analyze endpoint energy in AIR phone RIR recordings.

Run from the AcousticSpace repository root:
    python tests/analyze_rir_endpoint.py

Output:
    data/processed/air_phone_endpoint_analysis.csv
"""

from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf


REPO_ROOT = Path(__file__).resolve().parents[1]

DATASET_ROOT = Path(
    r"C:\AcousticSpace_Dataset\real_rirs_isotropic_noises"
)

VALIDATION_CSV = (
    REPO_ROOT / "data" / "processed" / "rir_feature_validation.csv"
)

OUTPUT_CSV = (
    REPO_ROOT / "data" / "processed" / "air_phone_endpoint_analysis.csv"
)

WINDOW_SECONDS = 0.1
ENDPOINT_SECONDS = 0.3
BIN_SECONDS = 0.01
CLIPPING_THRESHOLD = 0.999


def calculate_rms(signal):
    """Calculate RMS amplitude of a non-empty signal."""
    if signal.size == 0:
        raise ValueError("Cannot calculate RMS of an empty signal.")

    return float(np.sqrt(np.mean(signal.astype(np.float64) ** 2)))


def analyze_recording(path, filename):
    """Compare the final 100 ms with the preceding 100 ms."""

    audio, sample_rate = sf.read(str(path))

    if audio.ndim == 1:
        audio = audio[:, np.newaxis]

    if audio.ndim != 2:
        raise ValueError(f"Unsupported audio shape: {audio.shape}")

    window_samples = int(round(WINDOW_SECONDS * sample_rate))

    if audio.shape[0] < 2 * window_samples:
        raise ValueError("Recording is too short for endpoint analysis.")

    results = []

    for channel_index in range(audio.shape[1]):

        signal = np.asarray(
            audio[:, channel_index], dtype=np.float64
        )

        previous_window = signal[
            -2 * window_samples:-window_samples
        ]

        final_window = signal[-window_samples:]

        previous_rms = calculate_rms(previous_window)
        final_rms = calculate_rms(final_window)

        if previous_rms > 0:
            ratio = final_rms / previous_rms
        elif final_rms > 0:
            ratio = float("inf")
        else:
            ratio = 1.0

        results.append({
            "filename": filename,
            "channel_index": channel_index,
            "sample_rate": sample_rate,
            "previous_window_rms": previous_rms,
            "final_window_rms": final_rms,
            "final_to_previous_rms_ratio": ratio
        })

    return results


def inspect_final_endpoint(path, filename):
    """
    Inspect the final 300 ms in 10 ms bins, independently per channel.
    Also report the largest peak and possible clipping in that region.
    """

    audio, sample_rate = sf.read(str(path))

    if audio.ndim == 1:
        audio = audio[:, np.newaxis]

    endpoint_samples = int(round(ENDPOINT_SECONDS * sample_rate))
    bin_samples = int(round(BIN_SECONDS * sample_rate))

    if audio.shape[0] < endpoint_samples:
        raise ValueError("Recording is too short for endpoint inspection.")

    print(f"\nRecording: {filename}")

    for channel_index in range(audio.shape[1]):

        signal = np.asarray(
            audio[:, channel_index], dtype=np.float64
        )

        endpoint = signal[-endpoint_samples:]

        peak_index = int(np.argmax(np.abs(endpoint)))
        peak_amplitude = float(endpoint[peak_index])

        peak_time = (
            len(signal) - endpoint_samples + peak_index
        ) / sample_rate

        clipping_count = int(
            np.sum(np.abs(endpoint) >= CLIPPING_THRESHOLD)
        )

        print(f"\nChannel: {channel_index}")
        print(f"Peak time: {peak_time:.6f} seconds")
        print(f"Peak amplitude: {peak_amplitude:.6f}")
        print(f"Samples near full scale: {clipping_count}")

        print("RMS energy in final 300 ms:")

        for start in range(0, endpoint_samples, bin_samples):

            end = min(start + bin_samples, endpoint_samples)
            segment = endpoint[start:end]

            if segment.size == 0:
                continue

            rms = calculate_rms(segment)

            bin_start_time = (
                len(signal) - endpoint_samples + start
            ) / sample_rate

            bin_end_time = (
                len(signal) - endpoint_samples + end
            ) / sample_rate

            print(
                f"  {bin_start_time:.3f}s–{bin_end_time:.3f}s: "
                f"RMS = {rms:.8f}"
            )


def main():

    if not VALIDATION_CSV.exists():
        raise FileNotFoundError(
            f"Validation CSV not found: {VALIDATION_CSV}"
        )

    if not DATASET_ROOT.exists():
        raise FileNotFoundError(
            f"Dataset directory not found: {DATASET_ROOT}"
        )

    validation = pd.read_csv(VALIDATION_CSV)

    filenames = sorted(
        filename
        for filename in validation["filename"].dropna().unique()
        if "air_phone" in str(filename).lower()
    )

    results = []
    failures = []

    for filename in filenames:

        recording_path = DATASET_ROOT / filename

        if not recording_path.exists():
            failures.append((filename, "File not found"))
            continue

        try:
            results.extend(
                analyze_recording(recording_path, filename)
            )
        except (OSError, RuntimeError, ValueError) as exc:
            failures.append((filename, str(exc)))

    if results:

        result_df = pd.DataFrame(results)

        result_df = result_df.sort_values(
            "final_to_previous_rms_ratio",
            ascending=False
        )

        OUTPUT_CSV.parent.mkdir(
            parents=True, exist_ok=True
        )

        result_df.to_csv(OUTPUT_CSV, index=False)

        print(f"AIR phone recordings: {len(filenames)}")
        print(f"Channels analyzed: {len(result_df)}")

        print("\nHighest endpoint RMS ratios:")
        print(result_df.head(15).to_string(index=False))

        top_files = result_df.nlargest(
            3, "final_to_previous_rms_ratio"
        )["filename"].unique()

        print("\nDetailed final 300 ms inspection:")

        for filename in top_files:
            try:
                inspect_final_endpoint(
                    DATASET_ROOT / filename,
                    filename
                )
            except (OSError, RuntimeError, ValueError) as exc:
                failures.append((filename, str(exc)))

        print(f"\nResults saved to: {OUTPUT_CSV}")

    if failures:
        print(f"\nFailed inspections: {len(failures)}")

        for filename, error in failures:
            print(f"{filename}: {error}")


if __name__ == "__main__":
    main()
