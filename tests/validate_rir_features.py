
from pathlib import Path
import csv
import re

import numpy as np
import soundfile as sf

from audio_analysis.rir_analysis.rt60 import estimate_rt60
from audio_analysis.rir_analysis.drr import estimate_drr


RIR_DIR = Path(
    r"C:\AcousticSpace_Dataset\real_rirs_isotropic_noises"
)
RIR_MANIFEST = RIR_DIR / "rir_list"

OUTPUT_FILE = Path(
    "data/processed/rir_feature_validation.csv"
)

RUN_RT60_ENDPOINT_DIAGNOSTIC = True
DIAGNOSTIC_FILE = "air_type1_air_phone_bt_corridor_hhp.wav"
DIAGNOSTIC_CHANNEL = 0
EXCLUDE_FINAL_MS = 10


def load_rir_manifest():
    """Read RIR filenames and room IDs from the official manifest."""

    entries = []

    with RIR_MANIFEST.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()

            if not line:
                continue

            match = re.search(
                r"--room-id\s+(\S+)\s+(.+)$",
                line
            )

            if not match:
                continue

            entries.append({
                "room_id": match.group(1),
                "filename": match.group(2).split("/")[-1],
            })

    return entries


def calculate_rt60_fit_diagnostics(rir, sample_rate):
    """
    Reproduce the RT60 estimator's exact fitting procedure
    and calculate R².
    """

    energy = rir ** 2
    decay_curve = np.cumsum(energy[::-1])[::-1]

    decay_db = 10 * np.log10(
        np.maximum(decay_curve / decay_curve[0], 1e-12)
    )

    time = np.arange(len(rir)) / sample_rate

    mask = (decay_db <= -5) & (decay_db >= -35)

    if np.count_nonzero(mask) < 2:
        raise ValueError("Insufficient decay points for R² calculation.")

    x = time[mask]
    y = decay_db[mask]

    slope, intercept = np.polyfit(x, y, 1)
    predicted = slope * x + intercept

    ss_res = np.sum((y - predicted) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)

    if ss_tot == 0:
        raise ValueError("Cannot calculate R² for constant decay values.")

    r_squared = 1 - (ss_res / ss_tot)

    return {
        "r_squared": float(r_squared),
        "fit_points": int(len(y)),
    }


def run_rt60_endpoint_diagnostic():
    """Compare RT60 and R² before and after endpoint exclusion."""

    rir_path = RIR_DIR / DIAGNOSTIC_FILE

    print("\nRT60 ENDPOINT SENSITIVITY DIAGNOSTIC")
    print(f"File: {DIAGNOSTIC_FILE}")
    print(f"Channel: {DIAGNOSTIC_CHANNEL}")

    audio, sample_rate = sf.read(rir_path)
    audio = np.asarray(audio, dtype=np.float64)

    if audio.ndim == 1:
        audio = audio[:, np.newaxis]

    if DIAGNOSTIC_CHANNEL >= audio.shape[1]:
        raise ValueError("Requested diagnostic channel is unavailable.")

    rir = audio[:, DIAGNOSTIC_CHANNEL]

    samples_to_exclude = int(
        EXCLUDE_FINAL_MS / 1000 * sample_rate
    )

    if samples_to_exclude <= 0 or samples_to_exclude >= len(rir):
        raise ValueError("Invalid endpoint exclusion length.")

    signals = {
        "Original": rir,
        f"Exclude final {EXCLUDE_FINAL_MS} ms": rir[:-samples_to_exclude],
    }

    for label, signal in signals.items():
        print(f"\n{label}")
        print(f"Duration: {len(signal) / sample_rate:.6f} s")

        try:
            rt60 = estimate_rt60(signal, sample_rate)
            diagnostics = calculate_rt60_fit_diagnostics(
                signal, sample_rate
            )

            print(f"RT60: {rt60:.6f} s")
            print(f"R²: {diagnostics['r_squared']:.6f}")
            print(f"Fit points: {diagnostics['fit_points']}")

        except (ValueError, np.linalg.LinAlgError) as error:
            print(f"Diagnostic failed: {error}")


def main():

    entries = load_rir_manifest()

    if not entries:
        raise FileNotFoundError(
            f"No RIR entries found in {RIR_MANIFEST}"
        )

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    rows = []

    rt60_success = 0
    rt60_r2_success = 0
    drr_success = 0
    channel_count = 0

    for file_index, entry in enumerate(entries, start=1):

        rir_path = RIR_DIR / entry["filename"]

        try:
            audio, sample_rate = sf.read(rir_path)
            audio = np.asarray(audio, dtype=np.float64)

            if audio.ndim == 1:
                audio = audio[:, np.newaxis]

            for channel_index in range(audio.shape[1]):

                rir = audio[:, channel_index]
                duration = len(rir) / sample_rate
                channel_count += 1

                # RT60
                try:
                    rt60 = estimate_rt60(rir, sample_rate)
                    rt60_status = "success"
                    rt60_success += 1

                except (ValueError, np.linalg.LinAlgError) as error:
                    rt60 = ""
                    rt60_status = str(error)

                # RT60 fit quality
                try:
                    diagnostics = calculate_rt60_fit_diagnostics(
                        rir, sample_rate
                    )

                    rt60_r_squared = diagnostics["r_squared"]
                    rt60_fit_points = diagnostics["fit_points"]
                    rt60_r2_status = "success"
                    rt60_r2_success += 1

                except (ValueError, np.linalg.LinAlgError) as error:
                    rt60_r_squared = ""
                    rt60_fit_points = ""
                    rt60_r2_status = str(error)

                # DRR
                try:
                    drr = estimate_drr(rir, sample_rate)
                    drr_status = "success"
                    drr_success += 1

                except ValueError as error:
                    drr = ""
                    drr_status = str(error)

                rows.append({
                    "filename": entry["filename"],
                    "room_id": entry["room_id"],
                    "channel_index": channel_index,
                    "sample_rate": sample_rate,
                    "duration_seconds": round(duration, 4),
                    "rt60_seconds": rt60,
                    "rt60_status": rt60_status,
                    "rt60_r_squared": rt60_r_squared,
                    "rt60_fit_points": rt60_fit_points,
                    "rt60_r2_status": rt60_r2_status,
                    "drr_db": drr,
                    "drr_status": drr_status,
                })

        except Exception as error:

            rows.append({
                "filename": entry["filename"],
                "room_id": entry["room_id"],
                "channel_index": "",
                "sample_rate": "",
                "duration_seconds": "",
                "rt60_seconds": "",
                "rt60_status": f"file_error: {error}",
                "rt60_r_squared": "",
                "rt60_fit_points": "",
                "rt60_r2_status": "not_processed",
                "drr_db": "",
                "drr_status": "not_processed",
            })

        if file_index % 50 == 0 or file_index == len(entries):
            print(
                f"Processed {file_index}/{len(entries)} RIR files"
            )

    fieldnames = [
        "filename",
        "room_id",
        "channel_index",
        "sample_rate",
        "duration_seconds",
        "rt60_seconds",
        "rt60_status",
        "rt60_r_squared",
        "rt60_fit_points",
        "rt60_r2_status",
        "drr_db",
        "drr_status",
    ]

    with OUTPUT_FILE.open(
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames
        )

        writer.writeheader()
        writer.writerows(rows)

    print("\nCHANNEL-WISE VALIDATION SUMMARY")
    print(f"RIR files: {len(entries)}")
    print(f"Channels processed: {channel_count}")
    print(f"RT60 successful: {rt60_success}")
    print(f"RT60 failed: {channel_count - rt60_success}")
    print(f"RT60 R² successful: {rt60_r2_success}")
    print(f"RT60 R² failed: {channel_count - rt60_r2_success}")
    print(f"DRR successful: {drr_success}")
    print(f"DRR failed: {channel_count - drr_success}")
    print(f"\nCSV saved to: {OUTPUT_FILE}")

    if RUN_RT60_ENDPOINT_DIAGNOSTIC:
        run_rt60_endpoint_diagnostic()


if __name__ == "__main__":
    main()
