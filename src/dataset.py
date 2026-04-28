"""
dataset.py
----------
PyTorch Dataset class for Stanford Dogs.
Handles loading images and labels from the split directories.
"""

import os
from pathlib import Path
from PIL import Image, UnidentifiedImageError
from torch.utils.data import Dataset


class StanfordDogsDataset(Dataset):
    """
    Expects data laid out as:
        data/
          train/
            n02085620-Chihuahua/
              image1.jpg
              ...
            n02085782-Japanese_spaniel/
              ...
          val/
            ...
          test/
            ...
    """

    def __init__(self, root_dir: str, split: str = "train", transform=None):
        """
        Args:
            root_dir: path to the data/ folder
            split:    one of 'train', 'val', 'test'
            transform: torchvision transforms to apply
        """
        self.root = Path(root_dir) / split
        self.transform = transform

        if not self.root.exists():
            raise FileNotFoundError(
                f"Split directory not found: {self.root}\n"
                f"Run scripts/download_and_split.py first."
            )

        # Build sorted list of class names (strip the nXXXXXXXX- prefix for display)
        self.classes = sorted([d.name for d in self.root.iterdir() if d.is_dir()])
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}

        # Collect all valid image paths
        self.samples = self._load_samples()

        print(
            f"[{split}] {len(self.samples)} images across {len(self.classes)} classes"
        )

    def _load_samples(self):
        samples = []
        skipped = 0
        valid_exts = {".jpg", ".jpeg", ".png"}

        for cls_name in self.classes:
            cls_dir = self.root / cls_name
            label = self.class_to_idx[cls_name]

            for img_path in cls_dir.iterdir():
                if img_path.suffix.lower() not in valid_exts:
                    continue
                # Quick corruption check — try opening header only
                try:
                    with Image.open(img_path) as img:
                        img.verify()
                    samples.append((str(img_path), label))
                except (UnidentifiedImageError, Exception):
                    skipped += 1

        if skipped:
            print(f"  Skipped {skipped} corrupted/unreadable images")

        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]

        # Re-open after verify() (verify() closes the file)
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label

    def get_class_name(self, idx: int) -> str:
        """Return human-readable breed name for a label index."""
        raw = self.classes[idx]
        # Strip the WordNet ID prefix e.g. 'n02085620-Chihuahua' -> 'Chihuahua'
        return raw.split("-", 1)[-1].replace("_", " ")