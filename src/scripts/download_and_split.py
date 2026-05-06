"""
scripts/download_and_split.py
-----------------------------
Create data/train, data/val, and data/test from breed folders.
"""

import argparse
import random
import shutil
from collections import defaultdict
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Split dog breed folders into train/val/test.")
    parser.add_argument("--source-dir", default="Images")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def build_splits(source_dir, data_dir, train_ratio, val_ratio, seed):
    random.seed(seed)
    source_dir = Path(source_dir)
    data_dir = Path(data_dir)
    valid_exts = {".jpg", ".jpeg", ".png"}

    class_dirs = sorted(d for d in source_dir.iterdir() if d.is_dir())
    if not class_dirs:
        raise RuntimeError(f"No breed folders found in {source_dir}")

    print(f"Found {len(class_dirs)} breed directories.")
    split_counts = defaultdict(int)

    for class_dir in class_dirs:
        images = sorted(p for p in class_dir.iterdir() if p.suffix.lower() in valid_exts)
        random.shuffle(images)

        n_total = len(images)
        n_train = int(n_total * train_ratio)
        n_val = int(n_total * val_ratio)

        splits = {
            "train": images[:n_train],
            "val": images[n_train:n_train + n_val],
            "test": images[n_train + n_val:],
        }

        for split_name, split_images in splits.items():
            target_dir = data_dir / split_name / class_dir.name
            target_dir.mkdir(parents=True, exist_ok=True)
            for image_path in split_images:
                target_path = target_dir / image_path.name
                if not target_path.exists():
                    shutil.copy2(image_path, target_path)
                split_counts[split_name] += 1

    print("\nSplit summary:")
    for split_name in ("train", "val", "test"):
        print(f"  {split_name:<8} {split_counts[split_name]} images")


def main():
    args = parse_args()
    total = args.train_ratio + args.val_ratio + args.test_ratio
    if abs(total - 1.0) > 1e-6:
        raise ValueError("train/val/test ratios must add to 1.0")

    data_dir = Path(args.data_dir)
    if args.force and data_dir.exists():
        for split in ("train", "val", "test"):
            split_dir = data_dir / split
            if split_dir.exists():
                shutil.rmtree(split_dir)

    build_splits(
        source_dir=args.source_dir,
        data_dir=args.data_dir,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )
    print("\nDone. Next run: python src/verify_pipeline.py")


if __name__ == "__main__":
    main()
