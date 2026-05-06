"""
verify_pipeline.py
------------------
Run this to confirm the full data pipeline is working.
"""

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(__file__))

import matplotlib
matplotlib.use("Agg")
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt

from data_loader import get_all_loaders, print_class_distribution
from transforms import denormalize


def parse_args():
    parser = argparse.ArgumentParser(description="Verify the dog breed data pipeline.")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    return parser.parse_args()


def verify_batch_shape(loaders):
    print("\n" + "=" * 50)
    print("BATCH SHAPE CHECK")
    print("=" * 50)

    for split, loader in loaders.items():
        images, labels = next(iter(loader))
        num_classes = len(loader.dataset.classes)

        print(f"\n[{split}]")
        print(f"  images shape : {images.shape}")
        print(f"  labels shape : {labels.shape}")
        print(f"  dtype        : {images.dtype}")
        print(f"  pixel range  : [{images.min():.2f}, {images.max():.2f}]")
        print(f"  classes      : {num_classes}")

        assert images.shape[1:] == (3, 224, 224), (
            f"Expected (3, 224, 224), got {images.shape[1:]}"
        )
        assert labels.max() < num_classes, f"Label out of range: {labels.max()}"

    print("\nAll shape checks passed.")


def verify_labels(loaders):
    print("\n" + "=" * 50)
    print("LABEL CHECK")
    print("=" * 50)

    train_loader = loaders["train"]
    dataset = train_loader.dataset
    _, labels = next(iter(train_loader))
    names = [dataset.get_class_name(l.item()) for l in labels[:8]]

    print(f"\nFirst 8 label indices : {labels[:8].tolist()}")
    print(f"First 8 breed names   : {names}")


def verify_class_distribution(loaders):
    print("\n" + "=" * 50)
    print("CLASS DISTRIBUTION")
    print("=" * 50)
    print_class_distribution(loaders["train"].dataset)


def save_sample_grid(loaders, output_dir: str):
    print("\n" + "=" * 50)
    print("SAVING SAMPLE IMAGE GRID")
    print("=" * 50)

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    train_loader = loaders["train"]
    dataset = train_loader.dataset
    images, labels = next(iter(train_loader))

    n_show = min(16, images.size(0))
    cols = min(8, n_show)
    rows = (n_show + cols - 1) // cols

    fig = plt.figure(figsize=(cols * 2, rows * 2.5))
    gs = gridspec.GridSpec(rows, cols, figure=fig, hspace=0.4, wspace=0.2)

    for i in range(n_show):
        ax = fig.add_subplot(gs[i // cols, i % cols])
        img = denormalize(images[i]).permute(1, 2, 0).numpy()
        ax.imshow(img)
        ax.set_title(dataset.get_class_name(labels[i].item()), fontsize=7, wrap=True)
        ax.axis("off")

    out_path = Path(output_dir) / "sample_batch.png"
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved sample grid to: {out_path}")


def main():
    args = parse_args()
    print("Loading data loaders...")
    loaders = get_all_loaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    verify_batch_shape(loaders)
    verify_labels(loaders)
    verify_class_distribution(loaders)
    save_sample_grid(loaders, args.output_dir)

    print("\n" + "=" * 50)
    print("PIPELINE VERIFIED")
    print("=" * 50)
    print("\nThe pretrained ResNet training script can now read batches.")


if __name__ == "__main__":
    main()
