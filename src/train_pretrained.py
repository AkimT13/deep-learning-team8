"""
train_pretrained.py
-------------------
Train a pretrained ResNet-18 model for dog breed classification.
"""

import argparse
import csv
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(__file__))

from data_loader import get_all_loaders
from model import build_resnet18, count_trainable_parameters


def parse_args():
    parser = argparse.ArgumentParser(description="Train pretrained ResNet-18.")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--freeze-backbone", action="store_true")
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--max-batches", type=int, default=None)
    return parser.parse_args()


def accuracy(logits, labels, k=1):
    k = min(k, logits.shape[1])
    _, pred = logits.topk(k, dim=1)
    correct = pred.eq(labels.view(-1, 1)).any(dim=1).sum().item()
    return correct


def run_epoch(model, loader, criterion, device, optimizer=None, max_batches=None):
    training = optimizer is not None
    model.train(training)

    total_loss = 0.0
    total_samples = 0
    top1_correct = 0
    top5_correct = 0

    loop = tqdm(loader, desc="train" if training else "val", leave=False)
    for batch_idx, (images, labels) in enumerate(loop):
        if max_batches is not None and batch_idx >= max_batches:
            break

        images = images.to(device)
        labels = labels.to(device)

        if training:
            optimizer.zero_grad()

        with torch.set_grad_enabled(training):
            logits = model(images)
            loss = criterion(logits, labels)

            if training:
                loss.backward()
                optimizer.step()

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        total_samples += batch_size
        top1_correct += accuracy(logits, labels, k=1)
        top5_correct += accuracy(logits, labels, k=5)
        loop.set_postfix(loss=total_loss / max(total_samples, 1))

    return {
        "loss": total_loss / total_samples,
        "top1": 100.0 * top1_correct / total_samples,
        "top5": 100.0 * top5_correct / total_samples,
    }


def save_checkpoint(path, model, class_names, epoch, best_val_top1, args):
    checkpoint = {
        "model_name": "resnet18",
        "state_dict": model.state_dict(),
        "class_names": class_names,
        "num_classes": len(class_names),
        "epoch": epoch,
        "best_val_top1": best_val_top1,
        "args": vars(args),
    }
    torch.save(checkpoint, path)


def save_history(history, output_dir):
    reports_dir = Path(output_dir) / "reports"
    plots_dir = Path(output_dir) / "plots"
    reports_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    csv_path = reports_dir / "training_history.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=history[0].keys())
        writer.writeheader()
        writer.writerows(history)

    epochs = [row["epoch"] for row in history]
    plt.figure(figsize=(9, 4))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, [row["train_loss"] for row in history], label="train")
    plt.plot(epochs, [row["val_loss"] for row in history], label="val")
    plt.title("Loss")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, [row["val_top1"] for row in history], label="val top-1")
    plt.plot(epochs, [row["val_top5"] for row in history], label="val top-5")
    plt.title("Validation Accuracy")
    plt.legend()

    plt.tight_layout()
    plt.savefig(plots_dir / "training_history.png", dpi=150)
    plt.close()


def main():
    args = parse_args()
    device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
    )
    print(f"Device: {device}")

    loaders = get_all_loaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        balance_classes=True,
    )

    class_names = loaders["train"].dataset.classes
    num_classes = len(class_names)
    print(f"Dog breed classes: {num_classes}")

    model = build_resnet18(
        num_classes=num_classes,
        pretrained=not args.no_pretrained,
        freeze_backbone=args.freeze_backbone,
    ).to(device)
    print(f"Trainable parameters: {count_trainable_parameters(model):,}")

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    models_dir = Path(args.output_dir) / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    best_val_top1 = 0.0
    history = []

    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        train_metrics = run_epoch(
            model, loaders["train"], criterion, device, optimizer, args.max_batches
        )
        val_metrics = run_epoch(
            model, loaders["val"], criterion, device, None, args.max_batches
        )

        row = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_top1": train_metrics["top1"],
            "train_top5": train_metrics["top5"],
            "val_loss": val_metrics["loss"],
            "val_top1": val_metrics["top1"],
            "val_top5": val_metrics["top5"],
        }
        history.append(row)

        print(
            f"train loss {row['train_loss']:.4f} | "
            f"val loss {row['val_loss']:.4f} | "
            f"val top-1 {row['val_top1']:.2f}% | "
            f"val top-5 {row['val_top5']:.2f}%"
        )

        save_checkpoint(models_dir / "last_resnet18.pth", model, class_names, epoch, best_val_top1, args)
        if val_metrics["top1"] >= best_val_top1:
            best_val_top1 = val_metrics["top1"]
            save_checkpoint(models_dir / "best_resnet18.pth", model, class_names, epoch, best_val_top1, args)
            print(f"Saved best model with val top-1: {best_val_top1:.2f}%")

        save_history(history, args.output_dir)

    print(f"\nDone. Best model: {models_dir / 'best_resnet18.pth'}")


if __name__ == "__main__":
    main()
