"""Dataset loading and preprocessing utilities for project experiments."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader
from torch.utils.data import Subset
from torch.utils.data import TensorDataset
from torchvision import datasets
from torchvision import transforms


def get_california_housing(
    batch_size: int = 64,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
    num_workers: int = 0,
) -> tuple[DataLoader, DataLoader, DataLoader, dict[str, Any]]:
    """Load California Housing and return train, validation, and test loaders.

    The feature scaler is fit on the training split only to prevent leakage.
    On Windows, using ``num_workers=0`` is often the most reliable option for
    DataLoader multiprocessing.

    Args:
        batch_size: Batch size for all loaders.
        val_ratio: Validation split fraction relative to the full dataset.
        test_ratio: Test split fraction relative to the full dataset.
        seed: Random seed used for deterministic splitting.
        num_workers: Number of DataLoader worker processes.

    Returns:
        Train, validation, and test DataLoaders plus an info dictionary.
    """
    if not 0.0 < val_ratio < 1.0:
        raise ValueError("val_ratio must be between 0 and 1.")
    if not 0.0 < test_ratio < 1.0:
        raise ValueError("test_ratio must be between 0 and 1.")
    if val_ratio + test_ratio >= 1.0:
        raise ValueError("val_ratio + test_ratio must be less than 1.")

    dataset = fetch_california_housing(data_home="./data")
    features = dataset.data.astype(np.float32)
    targets = dataset.target.astype(np.float32)

    x_train_val, x_test, y_train_val, y_test = train_test_split(
        features,
        targets,
        test_size=test_ratio,
        random_state=seed,
        shuffle=True,
    )

    adjusted_val_ratio = val_ratio / (1.0 - test_ratio)
    x_train, x_val, y_train, y_val = train_test_split(
        x_train_val,
        y_train_val,
        test_size=adjusted_val_ratio,
        random_state=seed,
        shuffle=True,
    )

    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train).astype(np.float32)
    x_val = scaler.transform(x_val).astype(np.float32)
    x_test = scaler.transform(x_test).astype(np.float32)

    y_train = y_train.reshape(-1, 1).astype(np.float32)
    y_val = y_val.reshape(-1, 1).astype(np.float32)
    y_test = y_test.reshape(-1, 1).astype(np.float32)

    train_dataset = TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train))
    val_dataset = TensorDataset(torch.from_numpy(x_val), torch.from_numpy(y_val))
    test_dataset = TensorDataset(torch.from_numpy(x_test), torch.from_numpy(y_test))

    pin_memory = torch.cuda.is_available()
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    info = {
        "feature_names": list(dataset.feature_names),
        "n_features": x_train.shape[1],
        "n_train": len(train_dataset),
        "n_val": len(val_dataset),
        "n_test": len(test_dataset),
        "scaler": scaler,
        "target_name": "MedianHouseValue",
    }

    return train_loader, val_loader, test_loader, info


def get_cifar10(
    batch_size: int = 128,
    val_ratio: float = 0.1,
    seed: int = 42,
    data_dir: str = "./data",
    num_workers: int = 2,
) -> tuple[DataLoader, DataLoader, DataLoader, dict[str, Any]]:
    """Load CIFAR-10 with an augmentation-free validation split.

    The training set is loaded twice so the train split can use augmentation
    while the validation split uses evaluation transforms on the same indices.
    On Windows, ``num_workers=0`` is often the safest choice if worker startup
    causes issues.

    Args:
        batch_size: Batch size for all loaders.
        val_ratio: Fraction of the 50,000-image training set reserved for validation.
        seed: Random seed used for deterministic splitting.
        data_dir: Directory used by torchvision to download and cache CIFAR-10.
        num_workers: Number of DataLoader worker processes.

    Returns:
        Train, validation, and test DataLoaders plus an info dictionary.
    """
    if not 0.0 < val_ratio < 1.0:
        raise ValueError("val_ratio must be between 0 and 1.")

    mean = [0.4914, 0.4822, 0.4465]
    std = [0.2470, 0.2435, 0.2616]

    train_transform = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )
    test_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )

    full_train_aug = datasets.CIFAR10(
        root=data_dir,
        train=True,
        transform=train_transform,
        download=True,
    )
    full_train_eval = datasets.CIFAR10(
        root=data_dir,
        train=True,
        transform=test_transform,
        download=True,
    )
    test_dataset = datasets.CIFAR10(
        root=data_dir,
        train=False,
        transform=test_transform,
        download=True,
    )

    n_total = len(full_train_aug)
    val_size = int(n_total * val_ratio)
    train_size = n_total - val_size

    split_generator_aug = torch.Generator().manual_seed(seed)
    split_generator_eval = torch.Generator().manual_seed(seed)
    train_subset, _ = torch.utils.data.random_split(
        full_train_aug,
        [train_size, val_size],
        generator=split_generator_aug,
    )
    _, val_subset = torch.utils.data.random_split(
        full_train_eval,
        [train_size, val_size],
        generator=split_generator_eval,
    )

    if not isinstance(train_subset, Subset) or not isinstance(val_subset, Subset):
        raise RuntimeError("Expected random_split to return torch.utils.data.Subset instances.")

    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_subset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    info = {
        "n_classes": 10,
        "class_names": [
            "airplane",
            "automobile",
            "bird",
            "cat",
            "deer",
            "dog",
            "frog",
            "horse",
            "ship",
            "truck",
        ],
        "n_train": len(train_subset),
        "n_val": len(val_subset),
        "n_test": len(test_dataset),
        "image_shape": (3, 32, 32),
    }

    return train_loader, val_loader, test_loader, info


def get_dataset(name: str, **kwargs: Any) -> tuple[DataLoader, DataLoader, DataLoader, dict[str, Any]]:
    """Dispatch to a supported dataset loader by name.

    Args:
        name: Dataset identifier.
        **kwargs: Additional keyword arguments forwarded to the loader.

    Returns:
        Train, validation, and test DataLoaders plus an info dictionary.

    Raises:
        ValueError: If the dataset name is unsupported.
    """
    if name == "california_housing":
        return get_california_housing(**kwargs)
    if name == "cifar10":
        return get_cifar10(**kwargs)
    raise ValueError(f"Unknown dataset: {name}")


def _run_california_smoke_test() -> None:
    """Run the California Housing smoke test."""
    train_loader, _, _, info = get_california_housing(batch_size=64)
    print(
        "California Housing sizes:",
        info["n_train"],
        info["n_val"],
        info["n_test"],
    )

    features, targets = next(iter(train_loader))
    print("California batch shapes:", tuple(features.shape), tuple(targets.shape))
    assert features.shape == (64, 8)
    assert targets.shape == (64, 1)

    batch_mean = features.mean(dim=0)
    batch_std = features.std(dim=0, unbiased=False)
    print("California batch feature means:", batch_mean)
    print("California batch feature stds:", batch_std)
    assert torch.isfinite(batch_mean).all()
    assert torch.isfinite(batch_std).all()
    assert batch_mean.abs().mean().item() < 0.25
    assert 0.4 < batch_std.mean().item() < 1.6

    print("California Housing smoke test passed!")


def _run_cifar10_smoke_test() -> None:
    """Run the CIFAR-10 smoke test."""
    train_loader, val_loader, _, info = get_cifar10(batch_size=128, data_dir="./data")
    print("CIFAR-10 sizes:", info["n_train"], info["n_val"], info["n_test"])

    images, labels = next(iter(train_loader))
    print("CIFAR-10 train batch shapes:", tuple(images.shape), tuple(labels.shape))
    assert images.shape == (128, 3, 32, 32)
    assert labels.shape == (128,)
    assert images.min().item() < 0.0

    val_images, val_labels = next(iter(val_loader))
    assert val_images.shape[1:] == (3, 32, 32)
    assert val_labels.ndim == 1

    print("CIFAR-10 smoke test passed!")


def _run_dispatcher_smoke_test() -> None:
    """Run the dataset dispatcher smoke test."""
    get_dataset("california_housing", batch_size=32)
    get_dataset("cifar10", batch_size=64, data_dir="./data")

    try:
        get_dataset("imagenet")
    except ValueError:
        print("Dispatcher smoke test passed!")
        return

    raise AssertionError("Expected get_dataset('imagenet') to raise ValueError.")


if __name__ == "__main__":
    _run_california_smoke_test()
    _run_cifar10_smoke_test()
    _run_dispatcher_smoke_test()
