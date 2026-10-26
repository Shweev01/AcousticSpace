
import tempfile
from pathlib import Path


from audio_analysis.dataset_loader import load_protocol


def test_load_protocol_reads_labels():
    """The protocol should return recording IDs and valid labels."""

    with tempfile.TemporaryDirectory() as temp_dir:
        protocol_path = Path(temp_dir) / "sample_protocol.txt"

        protocol_path.write_text(
            "LA_0098 LA_T_9557645 - - bonafide\n"
            "LA_0079 LA_T_1004644 - A01 spoof\n",
            encoding="utf-8",
        )

        result = load_protocol(protocol_path)

        assert list(result.columns) == ["audio_id", "label"]
        assert len(result) == 2
        assert result.iloc[0]["audio_id"] == "LA_T_9557645"
        assert result.iloc[0]["label"] == "bonafide"
        assert result.iloc[1]["label"] == "spoof"

    print("PASS: Protocol IDs and labels are loaded correctly.")


def test_load_protocol_rejects_missing_file():
    """A missing protocol file should raise a clear error."""

    missing_path = Path("this_protocol_file_does_not_exist.txt")

    try:
        load_protocol(missing_path)
    except FileNotFoundError:
        print("PASS: Missing protocol file is rejected.")
    else:
        raise AssertionError("Expected FileNotFoundError for a missing file.")


def test_load_protocol_rejects_duplicate_audio_ids():
    """Duplicate recording IDs should be rejected."""
    with tempfile.TemporaryDirectory() as temp_dir:
        protocol_path = Path(temp_dir) / "duplicate_protocol.txt"
        protocol_path.write_text(
            "LA_0098 LA_T_9557645 - - bonafide\n"
            "LA_0079 LA_T_9557645 - A01 spoof\n",
            encoding="utf-8",
        )

        try:
            load_protocol(protocol_path)
        except ValueError:
            print("PASS: Duplicate audio IDs are rejected.")
        else:
            raise AssertionError(
                "Expected ValueError for duplicate audio IDs."
            )


def test_load_training_dataset_rejects_missing_audio():
    """A protocol entry without a matching FLAC should be rejected."""
    from audio_analysis.dataset_loader import load_training_dataset

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        audio_dir = temp_path / "flac"
        audio_dir.mkdir()

        protocol_path = temp_path / "sample_protocol.txt"
        protocol_path.write_text(
            "LA_0098 LA_T_9557645 - - bonafide\n",
            encoding="utf-8",
        )

        try:
            load_training_dataset(
                audio_dir=audio_dir,
                protocol_path=protocol_path,
            )
        except FileNotFoundError:
            print("PASS: Missing audio files are rejected.")
        else:
            raise AssertionError(
                "Expected FileNotFoundError for missing audio."
            )

def test_load_protocol_rejects_invalid_label():
    """Unexpected class labels should be rejected."""
    with tempfile.TemporaryDirectory() as temp_dir:
        protocol_path = Path(temp_dir) / "invalid_protocol.txt"
        protocol_path.write_text(
            "LA_0098 LA_T_9557645 - - unknown\n",
            encoding="utf-8",
        )

        try:
            load_protocol(protocol_path)
        except ValueError:
            print("PASS: Invalid labels are rejected.")
        else:
            raise AssertionError(
                "Expected ValueError for an invalid label."
            )        

if __name__ == "__main__":
    test_load_protocol_reads_labels()
    test_load_protocol_rejects_missing_file()
    test_load_protocol_rejects_duplicate_audio_ids()
    test_load_training_dataset_rejects_missing_audio()
    test_load_protocol_rejects_invalid_label()
    print("All dataset loader tests passed.")