# Audio Analysis Module

## 1. Overview

The `audio_analysis` module handles audio data loading, preprocessing, feature extraction, visualisation, Room Impulse Response (RIR) analysis, and RIR-based audio augmentation in the AcousticSpace project.

Its purpose is to prepare speech recordings for deepfake audio detection and provide tools to investigate how room acoustics affect recorded audio.

## 2. Module Structure

```text
audio_analysis/
│
├── dataset_loader.py
├── preprocessing.py
├── feature_extraction.py
├── visualization.py
│
├── augmentation/
│   └── rir_augmentation.py
│
└── rir_analysis/
    ├── drr.py
    ├── reverberation.py
    └── rt60.py
```

## 3. Dataset Loading

**File:** `dataset_loader.py`

This module loads and validates the ASVspoof 2019 Logical Access (LA) dataset.

Available functions:

- `load_protocol()` – reads recording IDs and labels from an ASVspoof protocol file.
- `load_training_dataset()` – matches training protocol entries with their corresponding FLAC audio files.
- `load_validation_dataset()` – loads and validates the development dataset.

The loader checks for:
- Missing protocol files.
- Empty protocol files and missing values.
- Unexpected labels outside `bonafide` and `spoof`.
- Duplicate audio IDs.
- Missing audio recordings.

The resulting DataFrame contains the recording ID, class label, audio file path, and file-existence information.

## 4. Audio Preprocessing

**File:** `preprocessing.py`

The `load_audio()` function loads and validates an audio recording.

```python
from audio_analysis.preprocessing import load_audio

audio, sample_rate = load_audio(
    file_path,
    target_sr=16000
)
```

The function:
- Loads audio using Librosa.
- Converts the recording to mono.
- Resamples it to 16 kHz by default.
- Rejects empty recordings.
- Rejects recordings shorter than one second.
- Checks for invalid numerical values.

It returns the processed waveform and its sample rate.

## 5. Feature Extraction

**File:** `feature_extraction.py`

This module converts audio waveforms into log-scaled Mel spectrograms.

### Mel spectrogram extraction

```python
from audio_analysis.feature_extraction import extract_mel_spectrogram

mel_spectrogram = extract_mel_spectrogram(
    audio,
    sample_rate=16000,
    n_mels=80,
    n_fft=1024,
    hop_length=256
)
```

Default parameters:

| Parameter | Value |
|---|---:|
| Sample rate | 16,000 Hz |
| Mel bands | 80 |
| FFT size | 1,024 |
| Hop length | 256 samples |
| Power | 2.0 |

The extracted Mel power spectrogram is converted into decibels using Librosa's `power_to_db()` function.

The module also provides `extract_audio_features()`, which returns:
- Waveform
- Sample rate
- Mel spectrogram
- Audio duration

## 6. Audio and RIR Visualisation

**File:** `visualization.py`

This module provides visual representations of audio signals and room impulse responses.

### 6.1 Audio waveform and Mel spectrogram

The `plot_audio_features()` function extracts audio features and displays the waveform and Mel spectrogram in separate plots.

```python
from audio_analysis.visualization import plot_audio_features

plot_audio_features("path/to/audio.wav")
```

### 6.2 RIR waveform and energy decay curve

The `plot_rir_analysis()` function displays:
- The RIR waveform over time.
- The Schroeder energy decay curve.
- The estimated RT60 value in the plot title.

```python
from audio_analysis.visualization import plot_rir_analysis

plot_rir_analysis(rir, sample_rate=16000)
```

These visualisations help inspect speech characteristics and understand the decay of sound energy in a room.

## 7. RIR Analysis

The `rir_analysis/` directory contains tools for analysing acoustic properties of Room Impulse Responses.

### 7.1 Reverberation Time (RT60)

**File:** `rir_analysis/rt60.py`

The `estimate_rt60()` function estimates the time required for sound energy to decay by 60 dB.

Methodology:
1. Calculate the squared RIR waveform to obtain its energy.
2. Apply the Schroeder reverse cumulative energy method.
3. Convert the normalised energy decay curve into decibels.
4. Fit a linear regression over the decay region between −5 dB and −35 dB.
5. Extrapolate the fitted decay slope to estimate RT60.

```python
from audio_analysis.rir_analysis.rt60 import estimate_rt60

rt60 = estimate_rt60(rir, sample_rate=16000)
```

The result is returned in seconds.

### 7.2 Direct-to-Reverberant Ratio (DRR)

**File:** `rir_analysis/drr.py`

The `estimate_drr()` function estimates the ratio between direct sound energy and reverberant sound energy in an RIR.

Methodology:
1. Detect the first significant RIR sample using a threshold of 1% of the peak amplitude.
2. Define a direct-sound window beginning at the detected arrival.
3. Use a default direct window of 2.5 milliseconds.
4. Calculate direct and reverberant energy.
5. Express their ratio in decibels.

```python
from audio_analysis.rir_analysis.drr import estimate_drr

drr = estimate_drr(rir, sample_rate=16000)
```

The result is returned in decibels (dB).

**Limitation:** This is a simplified estimator. Energy outside the direct window is classified as reverberant energy, so early reflections may also be included. The arrival threshold and direct-window duration can affect the estimate.

### 7.3 Applying Reverberation

**File:** `rir_analysis/reverberation.py`

The `apply_reverberation()` function simulates the effect of a room on an audio signal by convolving the input waveform with an RIR.

It uses SciPy's `fftconvolve()` with full convolution mode, preserving the reverberant tail.

```python
from audio_analysis.rir_analysis.reverberation import apply_reverberation

reverberated_audio = apply_reverberation(
    audio,
    rir,
    normalize=True,
    target_peak=0.99
)
```

The function validates both input signals and supports optional peak normalisation.

## 8. RIR-Based Audio Augmentation

**File:** `augmentation/rir_augmentation.py`

The `augment_with_rir()` function creates an augmented speech sample by applying an RIR to an input audio waveform.

It:
- Converts the audio and RIR inputs into NumPy arrays.
- Validates that both are non-empty, one-dimensional signals.
- Checks for invalid numerical values.
- Calls `apply_reverberation()` to perform convolution.
- Supports optional output peak normalisation.

```python
from audio_analysis.augmentation.rir_augmentation import augment_with_rir

augmented_audio = augment_with_rir(
    audio,
    rir,
    normalize=True,
    target_peak=0.99
)
```

This augmentation approach introduces simulated room reverberation into speech recordings. It helps investigate how acoustic environments influence genuine and spoofed speech recordings.

Both synthetic and real RIRs are used in the project's reverberation experiments.

## 9. Testing

The audio analysis module is supported by unit and integration tests covering:

- Dataset protocol loading and validation.
- Audio preprocessing.
- Mel spectrogram extraction.
- RT60 estimation.
- DRR estimation.
- Reverberation and RIR augmentation.
- Real speech reverberation validation.

Run the complete test suite from the project root:

```powershell
python -m pytest tests/ -v
```

**Latest validation:** 34 tests passed.

## 10. Dependencies and Environment Setup

All project dependencies are maintained in the central `requirements.txt` file located at the root of the AcousticSpace repository.

From the project root, install dependencies using:

```powershell
pip install -r requirements.txt
```

The audio analysis implementation uses libraries including NumPy, Pandas, Librosa, SciPy, Matplotlib, and SoundFile.

---

*This README documents the audio loading, preprocessing, feature extraction, visualisation, RIR analysis, and augmentation components of AcousticSpace.*