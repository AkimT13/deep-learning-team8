"""
transforms.py
-------------
Preprocessing and augmentation pipelines.
Import get_transforms() and pass the split name to get the right pipeline.
"""

from torchvision import transforms

# ImageNet mean/std — used because our pretrained models were trained on ImageNet
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

IMAGE_SIZE = 224  # ViT and most CNNs expect 224x224


def get_transforms(split: str) -> transforms.Compose:
    """
    Returns the appropriate transform pipeline for a given split.

    Train:  augmentation (random crop, flip, color jitter) + normalize
    Val:    deterministic resize + center crop + normalize
    Test:   same as val (no randomness during evaluation)

    Args:
        split: 'train', 'val', or 'test'

    Returns:
        torchvision.transforms.Compose object
    """
    if split == "train":
        return transforms.Compose([
            # Resize slightly larger than target, then random crop
            transforms.Resize(256),
            transforms.RandomCrop(IMAGE_SIZE),

            # Basic augmentations — helps with small dataset like Stanford Dogs
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=0.2,
                hue=0.05,
            ),
            transforms.RandomRotation(degrees=15),

            # Convert to tensor and normalize
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])

    elif split in ("val", "test"):
        return transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])

    else:
        raise ValueError(f"Unknown split: '{split}'. Expected 'train', 'val', or 'test'.")


def denormalize(tensor):
    """
    Reverse ImageNet normalization for visualization.
    Useful when displaying a batch with matplotlib.

    Args:
        tensor: normalized image tensor (C, H, W)
    Returns:
        denormalized tensor clamped to [0, 1]
    """
    import torch
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std  = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    return (tensor * std + mean).clamp(0, 1)