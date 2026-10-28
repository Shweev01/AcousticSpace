
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf

from audio_analysis.rir_analysis.rt60 import estimate_rt60

rir_path = Path(
    r"C:\AcousticSpace_Dataset\real_rirs_isotropic_noises"
    r"\RWCP_type1_rir_circle_e1b_imp110.wav"
)

# Load the real RIR and select channel 0.
rir, sample_rate = sf.read(rir_path)
rir = rir[:, 0].astype(np.float64)

# Time axis for the waveform.
time = np.arange(len(rir)) / sample_rate

# Schroeder energy-decay curve.
# Reverse cumulative energy estimates the remaining energy
# from each point to the end of the RIR.
energy = rir**2
decay_energy = np.cumsum(energy[::-1])[::-1]

decay_db = 10 * np.log10(
    np.maximum(decay_energy / decay_energy[0], 1e-12)
)

# Estimate RT60 using our existing implementation.
try:
    rt60 = estimate_rt60(rir, sample_rate)
    print(f"Estimated RT60: {rt60:.3f} seconds")
except ValueError as error:
    rt60 = None
    print(f"RT60 estimation warning: {error}")

print(f"Sample rate: {sample_rate} Hz")
print(f"RIR duration: {len(rir) / sample_rate:.3f} seconds")
print(f"Direct peak time: {np.argmax(np.abs(rir)) / sample_rate:.5f} seconds")

# Plot waveform and decay curve.
fig, axes = plt.subplots(2, 1, figsize=(11, 8))

axes[0].plot(time, rir)
axes[0].set_title("Real RIR — Channel 0 Waveform")
axes[0].set_xlabel("Time (seconds)")
axes[0].set_ylabel("Amplitude")
axes[0].grid(True, alpha=0.3)

axes[1].plot(time, decay_db)
axes[1].axhline(-5, linestyle="--", label="-5 dB")
axes[1].axhline(-35, linestyle="--", label="-35 dB")
axes[1].set_title("Schroeder Energy-Decay Curve")
axes[1].set_xlabel("Time (seconds)")
axes[1].set_ylabel("Relative energy (dB)")
axes[1].set_ylim(-65, 3)
axes[1].grid(True, alpha=0.3)
axes[1].legend()

plt.tight_layout()
plt.show()
