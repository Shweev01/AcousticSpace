
import torch
import pytest
from transformers import ASTForAudioClassification

from machine_learning.training.ast_dataset import MODEL_NAME
from machine_learning.training.ast_rir_fusion import ASTRIRFusionModel
from machine_learning.inference.predict import (
    CHECKPOINT_PATH,
    ID2LABEL,
    LABEL2ID,
)


def test_trained_ast_checkpoint_with_fusion_model():

    # Select device
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    # Check checkpoint exists
    assert CHECKPOINT_PATH.is_file(), (
        f"AST checkpoint not found: {CHECKPOINT_PATH}"
    )

    # Load AST architecture
    ast_model = ASTForAudioClassification.from_pretrained(
        MODEL_NAME,
        num_labels=2,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
        ignore_mismatched_sizes=True,
    )

    # Load trained AST weights
    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=device,
        weights_only=True,
    )

    ast_model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    # Create fusion model
    fusion_model = ASTRIRFusionModel(ast_model).to(device)
    fusion_model.eval()

    # Dummy inputs for architecture testing only
    input_values = torch.randn(
        1, 1024, 128, device=device
    )

    acoustic_features = torch.randn(
        1, 2, device=device
    )

    # Run inference
    with torch.inference_mode():
        outputs = fusion_model(
            input_values,
            acoustic_features,
        )

    # Verify output
    assert outputs.shape == (1, 2)
    assert torch.isfinite(outputs).all()

    print("\nTrained AST + RIR Fusion Test Passed")
    print("Device:", device)
    print("Output shape:", tuple(outputs.shape))
