from machine_learning.inference.predict import ASTPredictor


def test_ast_predictions():

    test_cases = [
        (
            r"C:\AcousticSpace_Dataset\LA\LA\ASVspoof2019_LA_dev\flac\LA_D_1047731.flac",
            "bonafide",
        ),
        (
            r"C:\AcousticSpace_Dataset\LA\LA\ASVspoof2019_LA_dev\flac\LA_D_1000265.flac",
            "spoof",
        ),
    ]

    # Load the model only once
    predictor = ASTPredictor()

    for audio_path, expected_label in test_cases:

        result = predictor.predict(audio_path)

        print(f"\nFile: {audio_path}")
        print(f"Expected: {expected_label}")
        print(f"Predicted: {result['prediction']}")
        print(f"Confidence: {result['confidence']:.4f}")

        assert result["prediction"] == expected_label
        assert 0.0 <= result["confidence"] <= 1.0

    print("\nAll AST inference tests passed!")

def test_invalid_audio_inputs():

    predictor = ASTPredictor()

    # Test 1: Missing file
    try:
        predictor.predict("missing_audio.flac")
        raise AssertionError("Missing file was not rejected.")
    except FileNotFoundError:
        print("PASS: Missing file rejected.")

    # Test 2: Audio shorter than 1 second
    import numpy as np
    import soundfile as sf
    import tempfile

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp:
        short_audio_path = temp.name

    sf.write(short_audio_path, np.zeros(4000), 16000)

    try:
        predictor.predict(short_audio_path)
        raise AssertionError("Short audio was not rejected.")
    except ValueError as error:
        print(f"PASS: Short audio rejected: {error}")

    import os
    os.remove(short_audio_path)

        # Test 3: Empty audio
    empty_audio_path = "tests/empty_audio.wav"
    sf.write(empty_audio_path, np.array([], dtype=np.float32), 16000)

    try:
        predictor.predict(empty_audio_path)
        raise AssertionError("Empty audio was not rejected.")
    except ValueError as error:
        print(f"PASS: Empty audio rejected: {error}")
    finally:
        if os.path.exists(empty_audio_path):
            os.remove(empty_audio_path)


if __name__ == "__main__":
    test_ast_predictions()
    test_invalid_audio_inputs()