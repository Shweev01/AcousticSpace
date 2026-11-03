# Machine Learning Module

## 1. Overview

The `machine_learning` module implements deep learning models for **deepfake audio detection**, classifying audio recordings into two categories:

- **Bonafide (0):** Genuine human speech.
- **Spoof (1):** Artificially generated or manipulated speech.

The module includes a baseline Convolutional Neural Network (CNN), an Audio Spectrogram Transformer (AST), training pipelines, dataset preparation, model evaluation and threshold analysis.

The models are trained and evaluated using the **ASVspoof 2019 Logical Access (LA) dataset**.

## 2. Module Structure

```text
machine_learning/
│
├── checkpoints/
│   ├── baseline_cnn_best.pt
│   ├── ast_best.pt
│   └── ast_trial_best.pt
│
├── models/
│   ├── cnn.py
│   ├── inspect_ast.py
│   ├── test_ast_model.py
│   └── test_ast_preprocessing.py
│
├── training/
│   ├── dataset.py
│   ├── data_loader.py
│   ├── train.py
│   ├── ast_dataset.py
│   └── train_ast.py
│
├── evaluation/
│   ├── evaluate.py
│   ├── evaluate_ast.py
│   ├── threshold_analysis.py
│   └── results/
│
└── README.md
```

## 3. Dataset Preparation

The module uses the ASVspoof 2019 LA training and development datasets.

Dataset preparation is handled through the audio analysis dataset loader and the machine learning dataset classes.

### Baseline CNN Dataset

The `training/dataset.py` file implements the `AudioDataset` class.

Its main responsibilities include:

- Loading training and validation dataset information.
- Extracting Mel spectrogram features from audio recordings.
- Using 80 Mel frequency bands.
- Cropping or zero-padding spectrograms to a fixed length of 256 frames.
- Caching processed spectrograms to reduce repeated feature extraction.
- Converting labels into numerical classes:
  - Bonafide: `0`
  - Spoof: `1`

Each sample is returned as a spectrogram tensor with shape:

`(1, 80, 256)`

### AST Dataset

The `training/ast_dataset.py` file prepares audio recordings for the Audio Spectrogram Transformer.

It uses the pretrained AST feature extractor:

`MIT/ast-finetuned-audioset-10-10-0.4593`

Audio is loaded as mono at 16 kHz and converted into the input representation expected by the transformer model.

## 4. Baseline CNN Model

The baseline model is implemented in `models/cnn.py` using PyTorch.

The `AudioCNN` architecture contains:

- Three convolutional layers with 16, 32 and 64 filters.
- Batch normalization and ReLU activation.
- Max pooling layers for reducing feature-map dimensions.
- Adaptive average pooling.
- A fully connected classification layer.

The model accepts single-channel Mel spectrograms and produces two-class classification logits.

## 5. CNN Training Pipeline

The baseline CNN training process is implemented in:

- `training/train.py`
- `training/data_loader.py`
- `training/dataset.py`

The data loader creates separate training and validation batches.

Default training configuration:

| Parameter | Value |
|---|---:|
| Epochs | 3 |
| Batch size | 32 |
| Learning rate | 0.001 |
| Optimizer | Adam |
| Loss function | Cross Entropy Loss |
| Device | CUDA when available, otherwise CPU |

The training process tracks training and validation loss and accuracy.

The checkpoint with the lowest validation loss is saved as:

`checkpoints/baseline_cnn_best.pt`

## 6. Audio Spectrogram Transformer (AST)

The project also implements an Audio Spectrogram Transformer for deepfake audio classification.

The AST model is based on the pretrained Hugging Face model:

`MIT/ast-finetuned-audioset-10-10-0.4593`

Its classification head is adapted for two classes:

- Bonafide
- Spoof

The AST implementation uses the pretrained AST feature extractor and fine-tunes the model on the ASVspoof 2019 LA dataset.

## 7. AST Training Pipeline

AST training is implemented in:

- `training/ast_dataset.py`
- `training/train_ast.py`

The training pipeline includes:

- Reproducible random seed configuration.
- Class-weighted cross-entropy loss to account for class imbalance.
- AdamW optimizer.
- Gradient accumulation to manage GPU memory.
- Mixed-precision training when CUDA is available.
- Validation after each training epoch.
- Best-checkpoint selection based on validation loss.

Default training configuration:

| Parameter | Value |
|---|---:|
| Epochs | 3 |
| Batch size | 4 |
| Learning rate | 0.00001 |
| Gradient accumulation steps | 4 |
| Optimizer | AdamW |
| Random seed | 42 |
| Device | CUDA when available, otherwise CPU |

The script supports three execution modes:

- `--smoke-test`: Performs a small forward and backward pass to verify the training pipeline.
- `--trial`: Runs a smaller, balanced training experiment.
- `--train`: Runs the full training process.

The best AST checkpoint is saved as:

`checkpoints/ast_best.pt`

The trial checkpoint is saved as:

`checkpoints/ast_trial_best.pt`

## 8. Model Evaluation

The evaluation module contains separate scripts for the CNN and AST models.

### CNN Evaluation

Implemented in `evaluation/evaluate.py`.

The script:

1. Loads the best CNN checkpoint.
2. Loads the ASVspoof 2019 LA development dataset.
3. Generates bonafide probabilities for each recording.
4. Applies a bonafide classification threshold of `0.30`.
5. Calculates evaluation metrics.
6. Saves predictions and metrics as CSV files.

### AST Evaluation

Implemented in `evaluation/evaluate_ast.py`.

The script:

1. Loads the best fine-tuned AST checkpoint.
2. Prepares the development dataset using the AST feature extractor.
3. Runs inference in batches.
4. Uses the highest-probability class as the final prediction.
5. Calculates accuracy, macro F1, precision and recall for both classes.
6. Generates a confusion matrix.
7. Saves evaluation outputs as CSV files.

## 9. Threshold Analysis

The `evaluation/threshold_analysis.py` script analyses different classification thresholds for the baseline CNN.

It compares the following bonafide probability thresholds:

`0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60`

For each threshold, it calculates:

- Accuracy
- Bonafide recall
- Spoof recall
- Macro F1 score

The selected threshold is `0.30`.

Threshold analysis helps understand how changing the classification threshold affects the balance between detecting spoof recordings and correctly identifying genuine recordings.

## 10. Evaluation Metrics

The module uses the following metrics:

| Metric | Purpose |
|---|---|
| Accuracy | Measures the proportion of correctly classified recordings. |
| Precision | Measures how many predictions for a class are correct. |
| Recall | Measures how many actual recordings of a class are detected. |
| Macro F1 | Calculates the average F1 score across both classes. |
| Confusion Matrix | Shows correct and incorrect predictions for each class. |

Both bonafide and spoof class metrics are considered to provide a more complete evaluation.

**Evaluation note:** The current evaluation scripts use the ASVspoof 2019 LA development dataset. Their results should therefore be interpreted as development-set performance, not as official test-set performance.

## 11. Saved Results

Evaluation outputs are stored in `evaluation/results/`.

The generated files include:

| File | Description |
|---|---|
| `final_predictions_threshold_030.csv` | CNN predictions using the selected threshold. |
| `final_metrics_threshold_030.csv` | CNN evaluation metrics. |
| `ast_final_predictions.csv` | AST predictions for the development dataset. |
| `ast_final_metrics.csv` | AST evaluation metrics. |
| `ast_confusion_matrix.csv` | AST confusion matrix. |
| `threshold_comparison.csv` | CNN metrics across different thresholds. |
| `confusion_matrix_threshold_030.csv` | CNN confusion matrix at threshold 0.30. |
| `validation_predictions.csv` | Predictions used for threshold analysis. |
| `misclassified_recordings.csv` | Recorded misclassified samples. |

## 12. Dependencies and Setup

The machine learning module uses the project's central `requirements.txt` file.

Main libraries include:

- PyTorch
- Torchvision
- Transformers
- Librosa
- NumPy
- Pandas
- Scikit-learn
- Matplotlib

Install the project dependencies from the repository root:

```bash
pip install -r requirements.txt
```

Ensure that the ASVspoof 2019 LA dataset is downloaded and that the dataset paths configured in the audio analysis dataset loader are correct.

## 13. Summary

The machine learning module provides two deep learning approaches for deepfake audio classification:

1. **Baseline CNN:** Uses fixed-length Mel spectrogram features for binary classification.
2. **Audio Spectrogram Transformer:** Fine-tunes a pretrained transformer model for bonafide and spoof audio classification.

It also provides reusable dataset preparation, training, evaluation and threshold-analysis pipelines to support model development and performance comparison.