
# AST + RIR Feature Fusion Architecture

## 1. Project Objective

AcousticSpace aims to detect deepfake audio by combining audio representations learned by an Audio Spectrogram Transformer (AST) with room-related acoustic information.

The existing AST model serves as the spectral baseline. The proposed fusion model will investigate whether additional acoustic features improve bonafide and spoof classification.

## 2. Existing Components

The project currently contains:

- AST-based audio classification.
- Audio preprocessing at 16 kHz and mono.
- Log-Mel spectrogram extraction.
- RT60 estimation using the Schroeder energy decay curve.
- DRR estimation using a direct-arrival window.
- RIR-based reverberation augmentation using convolution.
- AST inference through `ASTPredictor`.

The existing AST baseline does not currently consume RT60 or DRR as additional model inputs.

## 3. Proposed Fusion Architecture

The proposed architecture contains two branches.

### Branch A: AST Audio Representation

1. Load the input audio.
2. Convert it to 16 kHz mono.
3. Apply the existing AST feature extractor.
4. Pass the extracted representation through the AST backbone.
5. Obtain a hidden audio embedding.

The existing AST-only classifier will remain available as the baseline.

### Branch B: Acoustic Feature Representation

The acoustic branch will investigate two features:

| Feature | Description | Unit |
|---|---|---|
| RT60 | Estimated reverberation decay time | Seconds |
| DRR | Direct-to-reverberant energy ratio | Decibels (dB) |

These features must be derived consistently from the audio available at inference time.

Speech-derived estimates will initially be treated as acoustic proxies, not guaranteed measurements of the actual room impulse response.

### Fusion and Classification

The AST embedding and acoustic feature vector will be concatenated and passed through a classification head.

Conceptual pipeline:

Audio
  ├── AST backbone → AST embedding
  └── Acoustic analysis → [RT60, DRR]
                  ↓
             Concatenation
                  ↓
             MLP classifier
                  ↓
          Bonafide / Spoof

## 4. Proposed Feature Contract

| Component | Proposed shape |
|---|---|
| AST embedding | `[batch_size, hidden_size]` |
| Acoustic feature vector | `[batch_size, 2]` |
| Fused representation | `[batch_size, hidden_size + 2]` |
| Classification output | `[batch_size, 2]` |

The exact AST embedding dimension and extraction method must be confirmed from the installed Transformers model before implementation.

## 5. Feature Normalization

RT60 and DRR will be normalized using statistics calculated exclusively from the training split.

The same training-set mean and standard deviation must be reused during validation, testing, and inference.

Missing or invalid acoustic estimates must be handled explicitly rather than silently replaced with arbitrary values.

## 6. Training and Inference Consistency

A central requirement is that the fusion model must receive compatible features during both training and inference.

The training pipeline must not depend on known RIR measurements from augmentation experiments if those measurements will not be available for ordinary uploaded audio at inference time.

Real RWCP RIR recordings will be used for controlled experiments and validation of acoustic estimation.

If reliable speech-based acoustic estimates cannot be obtained, the feature strategy must be revised before fusion training.

## 7. Evaluation Strategy

The fusion model will be compared against the existing AST-only baseline using the same data split.

Evaluation metrics:

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- Equal Error Rate (EER)

The comparison will focus particularly on spoof detection performance.

## 8. Ablation Study

The following configurations are planned:

1. AST-only baseline.
2. AST + RT60 proxy.
3. AST + DRR proxy.
4. AST + RT60 + DRR proxies.

The purpose is to investigate whether acoustic information contributes measurable improvements over the spectral baseline.

## 9. Known Limitations and Open Questions

- Speech-based RT60 and DRR estimation may be unreliable.
- The current RT60 and DRR implementations operate on known RIR signals.
- The current augmentation demonstration uses a real RIR but does not establish a dataset-wide RIR pairing pipeline.
- AST embedding extraction must be verified before fusion implementation.
- Feature distributions and normalization statistics must be validated.
- Improvements must be demonstrated experimentally rather than assumed.

## 10. Implementation Sequence

1. Verify AST embedding extraction.
2. Validate acoustic feature estimation.
3. Establish the training and inference feature contract.
4. Implement the fusion model.
5. Train and evaluate the fusion model.
6. Compare against the AST-only baseline.
7. Perform ablation and error analysis.
