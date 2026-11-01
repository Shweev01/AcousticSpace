import torch

from machine_learning.training.ast_dataset import create_ast_datasets


def test_ast_dataset():

    print("Creating AST datasets...")

    train_dataset, dev_dataset = create_ast_datasets()
    print("\nTraining label distribution:")
    print(train_dataset.dataframe["label"].value_counts())

    print("\nValidation label distribution:")
    print(dev_dataset.dataframe["label"].value_counts())
    print("\nDataset sizes:")
    print("Training:", len(train_dataset))
    print("Validation:", len(dev_dataset))

    # Check dataset sizes
    assert len(train_dataset) > 0
    assert len(dev_dataset) > 0

    # Check one training sample
    train_sample = train_dataset[0]

    print("\nTraining sample:")
    print("Input shape:", train_sample["input_values"].shape)
    print("Label:", train_sample["labels"].item())

    assert train_sample["input_values"].shape == (1024, 128)
    assert train_sample["labels"].item() in [0, 1]
    assert train_sample["labels"].dtype == torch.long
    assert torch.isfinite(train_sample["input_values"]).all()

    # Check one validation sample
    dev_sample = dev_dataset[0]

    print("\nValidation sample:")
    print("Input shape:", dev_sample["input_values"].shape)
    print("Label:", dev_sample["labels"].item())

    assert dev_sample["input_values"].shape == (1024, 128)
    assert dev_sample["labels"].item() in [0, 1]
    assert dev_sample["labels"].dtype == torch.long
    assert torch.isfinite(dev_sample["input_values"]).all()

    print("\nAST dataset test passed.")


if __name__ == "__main__":
    test_ast_dataset()