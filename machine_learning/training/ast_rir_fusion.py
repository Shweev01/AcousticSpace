
import torch
import torch.nn as nn


class ASTRIRFusionModel(nn.Module):
    """
    Combines AST audio embeddings with RT60 and DRR features.
    """

    def __init__(self, ast_model, acoustic_feature_size=2, dropout=0.3):
        super().__init__()

        self.ast_backbone = ast_model.audio_spectrogram_transformer

        hidden_size = ast_model.config.hidden_size

        self.fusion_classifier = nn.Sequential(
            nn.Linear(hidden_size + acoustic_feature_size, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 2),
        )

    def forward(self, input_values, acoustic_features):

        if acoustic_features.ndim != 2:
            raise ValueError(
                "Acoustic features must have shape [batch_size, 2]."
            )

        if acoustic_features.shape[1] != 2:
            raise ValueError(
                "Expected exactly two acoustic features: RT60 and DRR."
            )

        ast_outputs = self.ast_backbone(input_values)

        ast_embedding = ast_outputs.pooler_output

        if ast_embedding is None:
            raise RuntimeError("AST did not return a pooled embedding.")

        acoustic_features = acoustic_features.to(
            device=ast_embedding.device,
            dtype=ast_embedding.dtype,
        )

        fused_features = torch.cat(
            (ast_embedding, acoustic_features),
            dim=1,
        )

        logits = self.fusion_classifier(fused_features)

        return logits
