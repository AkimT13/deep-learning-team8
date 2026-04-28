"""
verify_pipeline.py
-------------------
Milestone 1 verification script.
Run this to confirm the full data pipeline is working.

Expected output:
    [train] 14,580 images across 120 classes
    [val]     3,118 images across 120 classes
    [test]    3,112 images across 120 classes

    Batch shape : torch.Size([32, 3, 224, 224])
    Labels      : tensor([42, 7, 119, ...])
    Label names : ['Maltese dog', 'Beagle', 'Welsh springer spaniel', ...]

    Class distribution summary printed...
    Sample image grid saved to outputs/sample_batch.png

Usage:
    cd dog_classifier/
    python src/verify_pipeline.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import torch
import matplotlib
matplotlib.use("Agg")  # headless — saves to file instead of showing window
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from data_loader import get_all_loaders, print_class_distribution
from transforms import denormalize

DATA_DIR    = "data"
OUTPUT_DIR  = "outputs"
BATCH_SIZE  = 32


def verify_batch_shape(loaders):
    print("\n" + "="*50)
    print("BATCH SHAPE CHECK")
    print("="*50)

    for split, loader in loaders.items():
        images, labels = next(iter(loader))
        print(f"\n[{split}]")
        print(f"  images shape : {images.shape}")
        print(f"  labels shape : {labels.shape}")
        print(f"  dtype        : {images.dtype}")
        print(f"  pixel range  : [{images.min():.2f}, {images.max():.2f}]")

        # Sanity check
        assert images.shape[1:] == (3, 224, 224), \
            f"Expected (3, 224, 224), got {images.shape[1:]}"
        assert labels.max() < 120, \
            f"Label out of range: {labels.max()}"

    print("\nAll shape checks passed.")


def verify_labels(loaders):
    print("\n" + "="*50)
    print("LABEL CHECK")
    print("="*50)

    train_loader = loaders["train"]
    dataset = train_loader.dataset

    images, labels = next(iter(train_loader))
    names = [dataset.get_class_name(l.item()) for l in labels[:8]]

    print(f"\nFirst 8 label indices : {labels[:8].tolist()}")
    print(f"First 8 breed names   : {names}")


def verify_class_distribution(loaders):
    print("\n" + "="*50)
    print("CLASS DISTRIBUTION")
    print("="*50)

    train_dataset = loaders["train"].dataset
    print_class_distribution(train_dataset)


def save_sample_grid(loaders, output_dir: str):
    print("\n" + "="*50)
    print("SAVING SAMPLE IMAGE GRID")
    print("="*50)

    os.makedirs(output_dir, exist_ok=True)
    train_loader = loaders["train"]
    dataset = train_loader.dataset

    images, labels = next(iter(train_loader))

    n_show = 16
    cols = 8
    rows = n_show // cols

    fig = plt.figure(figsize=(cols * 2, rows * 2.5))
    gs = gridspec.GridSpec(rows, cols, figure=fig, hspace=0.4, wspace=0.2)

    for i in range(n_show):
        ax = fig.add_subplot(gs[i // cols, i % cols])
        img = denormalize(images[i]).permute(1, 2, 0).numpy()
        ax.imshow(img)
        ax.set_title(dataset.get_class_name(labels[i].item()), fontsize=7, wrap=True)
        ax.axis("off")

    out_path = os.path.join(output_dir, "sample_batch.png")
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved sample grid to: {out_path}")


def main():
    print("Loading data loaders...")
    loaders = get_all_loaders(
        data_dir=DATA_DIR,
        batch_size=BATCH_SIZE,
        num_workers=0,  # set 0 for initial debugging
    )

    verify_batch_shape(loaders)
    verify_labels(loaders)
    verify_class_distribution(loaders)
    save_sample_grid(loaders, OUTPUT_DIR)

    print("\n" + "="*50)
    print("MILESTONE 1 PIPELINE VERIFIED")
    print("="*50)
    print("\nThe training loop can now read batches without crashing.")
    print("Hand off to Person 2 (model) and Person 3 (training loop).")


if __name__ == "__main__":
    main()