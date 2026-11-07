
from transformers import ASTConfig, ASTFeatureExtractor

MODEL_NAME = "MIT/ast-finetuned-audioset-10-10-0.4593"


def inspect_ast():

    print("Loading AST configuration and feature extractor...")

    config = ASTConfig.from_pretrained(MODEL_NAME)
    feature_extractor = ASTFeatureExtractor.from_pretrained(MODEL_NAME)

    print("\nAST MODEL CONFIGURATION")
    print("Model:", MODEL_NAME)
    print("Hidden size:", config.hidden_size)
    print("Transformer layers:", config.num_hidden_layers)
    print("Attention heads:", config.num_attention_heads)
    print("Original classes:", config.num_labels)

    print("\nAST FEATURE EXTRACTOR")
    print("Sampling rate:", feature_extractor.sampling_rate)
    print("Mel bins:", feature_extractor.num_mel_bins)
    print("Maximum length:", feature_extractor.max_length)
    print("Normalization enabled:", feature_extractor.do_normalize)
    print("Mean:", feature_extractor.mean)
    print("Standard deviation:", feature_extractor.std)

    print("\nAST inspection completed successfully.")


if __name__ == "__main__":
    inspect_ast()
