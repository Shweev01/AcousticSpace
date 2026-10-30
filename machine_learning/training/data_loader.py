from torch.utils.data import DataLoader

from machine_learning.training.dataset import create_datasets


def create_data_loaders(batch_size=32):
    """Create batched loaders for training and validation."""

    train_dataset, val_dataset = create_datasets()

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )

    return train_loader, val_loader