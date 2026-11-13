import torch
import torch.nn as nn


class ASTRIRFusionModel(nn.Module):
    """
    Combines AST audio embeddings with acoustic features.
    """

    def __init__(self, ast_model, acoustic_feature_size=2, dropout=0.3):
        super().__init__()

        if acoustic_feature_size not in (1, 2):
            raise ValueError(
                "Acoustic feature size must be 1 or 2."
            )

        self.ast_backbone = ast_model.audio_spectrogram_transformer
        self.acoustic_feature_size = acoustic_feature_size

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
                "Acoustic features must be a 2D tensor."
            )

        if acoustic_features.shape[1] != self.acoustic_feature_size:
            raise ValueError(
                f"Expected {self.acoustic_feature_size} acoustic "
                f"features, got {acoustic_features.shape[1]}."
            )

        ast_outputs = self.ast_backbone(input_values)
        ast_embedding = ast_outputs.pooler_output

        if ast_embedding is None:
            raise RuntimeError(
                "AST did not return a pooled embedding."
            )

        acoustic_features = acoustic_features.to(
            device=ast_embedding.device,
            dtype=ast_embedding.dtype
        )

        fused_features = torch.cat(
            (ast_embedding, acoustic_features), dim=1
        )

        return self.fusion_classifier(fused_features)