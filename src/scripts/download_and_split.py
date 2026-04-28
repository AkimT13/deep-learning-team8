"""
scripts/download_and_split.py
------------------------------
Downloads the Stanford Dogs dataset and organises it into
    data/train/   data/val/   data/test/

Run once before anything else:
    python scripts/download_and_split.py
"""

import os
import shutil
import tarfile
import urllib.request
from pathlib import Path
from collections import defaultdict
import random

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DATA_DIR   = Path("data")
SPLITS     = {"train": 0.70, "val": 0.15, "test": 0.15}
RANDOM_SEED = 42

DATASET_URL = "http://vision.stanford.edu/aditya86/ImageNetDogs/images.tar"
ANNOT_URL   = "http://vision.stanford.edu/aditya86/ImageNetDogs/annotation.tar"


# ---------------------------------------------------------------------------
# Download helpers
# ---------------------------------------------------------------------------

def download_file(url: str, dest: Path):
    if dest.exists():
        print(f"  Already downloaded: {dest.name}")
        return
    print(f"  Downloading {dest.name} ...")
    urllib.request.urlretrieve(url, dest, reporthook=_progress)
    print()


def _progress(block_num, block_size, total_size):
    downloaded = block_num * block_size
    pct = min(downloaded / total_size * 100, 100) if total_size > 0 else 0
    print(f"\r  {pct:.1f}%", end="", flush=True)


def extract_tar(tar_path: Path, dest: Path, check_dir: str = None):
    # If a specific subfolder is given, check for that instead of dest itself
    already_done = (dest / check_dir).exists() if check_dir else dest.exists()
    if already_done:
        print(f"  Already extracted: {check_dir or dest.name}")
        return
    print(f"  Extracting {tar_path.name} ...")
    dest.mkdir(parents=True, exist_ok=True)
    with tarfile.open(tar_path) as tar:
        tar.extractall(dest)
    print(f"  Done.")


# ---------------------------------------------------------------------------
# Split logic
# ---------------------------------------------------------------------------

def build_splits(images_dir: Path, output_dir: Path):
    """
    Walk images_dir, group by class, then split each class's images
    into train/val/test at the configured ratios.
    """
    random.seed(RANDOM_SEED)

    class_dirs = sorted([d for d in images_dir.iterdir() if d.is_dir()])
    print(f"\nFound {len(class_dirs)} breed directories.")

    split_counts = defaultdict(int)

    for cls_dir in class_dirs:
        images = sorted(cls_dir.glob("*.jpg")) + sorted(cls_dir.glob("*.JPEG"))
        if not images:
            continue

        random.shuffle(images)
        n = len(images)
        n_train = int(n * SPLITS["train"])
        n_val   = int(n * SPLITS["val"])

        assignment = (
            [("train", img) for img in images[:n_train]] +
            [("val",   img) for img in images[n_train : n_train + n_val]] +
            [("test",  img) for img in images[n_train + n_val :]]
        )

        for split_name, img_path in assignment:
            dest_dir = output_dir / split_name / cls_dir.name
            dest_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(img_path, dest_dir / img_path.name)
            split_counts[split_name] += 1

    print("\nSplit summary:")
    for split_name, count in sorted(split_counts.items()):
        print(f"  {split_name:<8} {count} images")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    DATA_DIR.mkdir(exist_ok=True)
    raw_dir = DATA_DIR / "raw"
    raw_dir.mkdir(exist_ok=True)

    images_tar = raw_dir / "images.tar"
    download_file(DATASET_URL, images_tar)
    extract_tar(images_tar, raw_dir, check_dir="Images")

    images_dir = raw_dir / "Images"
    if not images_dir.exists():
        # Some versions extract to a different folder name
        candidates = list(raw_dir.glob("*Images*")) + list(raw_dir.glob("*images*"))
        if candidates:
            images_dir = candidates[0]
        else:
            raise RuntimeError(
                f"Could not find images directory inside {raw_dir}. "
                f"Contents: {list(raw_dir.iterdir())}"
            )

    print(f"\nBuilding train/val/test splits in {DATA_DIR}/...")
    build_splits(images_dir, DATA_DIR)

    print("\nDone! You can now run: python src/verify_pipeline.py")


if __name__ == "__main__":
    main()