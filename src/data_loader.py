"""
data_loader.py
--------------
Factory functions for creating PyTorch DataLoaders.
Also includes a WeightedSampler helper for class imbalance.
"""

import torch
import numpy as np
from torch.utils.data import DataLoader, WeightedRandomSampler
from collections import Counter

from dataset import StanfordDogsDataset
from transforms import get_transforms


def get_dataloader(
    data_dir: str,
    split: str,
    batch_size: int = 32,
    num_workers: int = 4,
    balance_classes: bool = False,
) -> DataLoader:
    """
    Build a DataLoader for the given split.

    Args:
        data_dir:        path to data/ folder (contains train/, val/, test/)
        split:           'train', 'val', or 'test'
        batch_size:      number of images per batch
        num_workers:     parallel workers for loading (set 0 on Windows)
        balance_classes: if True, use WeightedRandomSampler to oversample
                         rare breeds (train split only)

    Returns:
        torch.utils.data.DataLoader
    """
    transform = get_transforms(split)
    dataset = StanfordDogsDataset(
        root_dir=data_dir,
        split=split,
        transform=transform,
    )

    sampler = None
    shuffle = split == "train"

    if balance_classes and split == "train":
        sampler = _make_weighted_sampler(dataset)
        shuffle = False  # mutually exclusive with sampler

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=(split == "train"),  # avoids tiny final batches during training
    )

    return loader


def get_all_loaders(
    data_dir: str,
    batch_size: int = 32,
    num_workers: int = 4,
    balance_classes: bool = False,
) -> dict:
    """
    Convenience function — returns all three loaders at once.

    Usage:
        loaders = get_all_loaders('data/')
        train_loader = loaders['train']
        val_loader   = loaders['val']
        test_loader  = loaders['test']
    """
    return {
        split: get_dataloader(
            data_dir=data_dir,
            split=split,
            batch_size=batch_size,
            num_workers=num_workers,
            balance_classes=balance_classes,
        )
        for split in ("train", "val", "test")
    }


def get_class_distribution(dataset: StanfordDogsDataset) -> dict:
    """
    Count images per class.

    Returns:
        dict mapping class_name -> count, sorted by count ascending
    """
    labels = [label for _, label in dataset.samples]
    counts = Counter(labels)
    return {
        dataset.get_class_name(idx): count
        for idx, count in sorted(counts.items(), key=lambda x: x[1])
    }


def print_class_distribution(dataset: StanfordDogsDataset):
    """Print a compact class distribution summary to stdout."""
    dist = get_class_distribution(dataset)
    counts = list(dist.values())

    print(f"\nClass distribution ({len(dist)} breeds):")
    print(f"  Min images per breed : {min(counts)}")
    print(f"  Max images per breed : {max(counts)}")
    print(f"  Mean                 : {np.mean(counts):.1f}")
    print(f"  Std                  : {np.std(counts):.1f}")

    # Flag the most and least represented breeds
    items = list(dist.items())
    print(f"\n  Least represented:")
    for name, cnt in items[:5]:
        print(f"    {name:<40} {cnt} images")
    print(f"\n  Most represented:")
    for name, cnt in items[-5:]:
        print(f"    {name:<40} {cnt} images")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _make_weighted_sampler(dataset: StanfordDogsDataset) -> WeightedRandomSampler:
    """
    Build a WeightedRandomSampler so each class is sampled equally.
    Rare breeds are oversampled; common breeds are undersampled.
    """
    labels = [label for _, label in dataset.samples]
    class_counts = Counter(labels)
    total = len(labels)

    # Weight for each sample = 1 / frequency of its class
    weights = [1.0 / class_counts[label] for label in labels]
    weights = torch.DoubleTensor(weights)

    sampler = WeightedRandomSampler(
        weights=weights,
        num_samples=total,
        replacement=True,
    )
    return sampler