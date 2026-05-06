"""
evaluate.py
-----------
Evaluate the trained ResNet-18 dog breed model.
"""

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(__file__))

from data_loader import get_dataloader
from model import build_resnet18


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a dog breed checkpoint.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--max-batches", type=int, default=None)
    return parser.parse_args()


def pretty_name(raw):
    return raw.split("-", 1)[-1].replace("_", " ")


def topk_correct(logits, labels, k):
    k = min(k, logits.shape[1])
    _, pred = logits.topk(k, dim=1)
    return pred.eq(labels.view(-1, 1)).any(dim=1).sum().item()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(args.checkpoint, map_location=device)
    class_names = checkpoint["class_names"]

    model = build_resnet18(num_classes=len(class_names), pretrained=False)
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    model.eval()

    loader = get_dataloader(
        data_dir=args.data_dir,
        split=args.split,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    total_samples = 0
    top1 = 0
    top5 = 0
    y_true = []
    y_pred = []

    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(tqdm(loader, desc="evaluate")):
            if args.max_batches is not None and batch_idx >= args.max_batches:
                break

            images = images.to(device)
            labels = labels.to(device)
            logits = model(images)
            loss = criterion(logits, labels)

            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size
            top1 += topk_correct(logits, labels, 1)
            top5 += topk_correct(logits, labels, 5)
            y_true.extend(labels.cpu().tolist())
            y_pred.extend(logits.argmax(dim=1).cpu().tolist())

    metrics = {
        "split": args.split,
        "loss": total_loss / total_samples,
        "top1_accuracy": 100.0 * top1 / total_samples,
        "top5_accuracy": 100.0 * top5 / total_samples,
        "num_samples": total_samples,
    }

    reports_dir = Path(args.output_dir) / "reports"
    plots_dir = Path(args.output_dir) / "plots"
    reports_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    with (reports_dir / f"{args.split}_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    labels_range = list(range(len(class_names)))
    display_names = [pretty_name(name) for name in class_names]
    report = classification_report(
        y_true,
        y_pred,
        labels=labels_range,
        target_names=display_names,
        output_dict=True,
        zero_division=0,
    )
    with (reports_dir / f"{args.split}_classification_report.json").open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    cm = confusion_matrix(y_true, y_pred, labels=labels_range)
    per_class_path = reports_dir / f"{args.split}_per_class_accuracy.csv"
    with per_class_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["breed", "accuracy", "correct", "total"])
        totals = cm.sum(axis=1)
        correct = np.diag(cm)
        for idx, name in enumerate(display_names):
            acc = 0.0 if totals[idx] == 0 else 100.0 * correct[idx] / totals[idx]
            writer.writerow([name, f"{acc:.2f}", int(correct[idx]), int(totals[idx])])

    plt.figure(figsize=(10, 8))
    plt.imshow(cm, cmap="Blues")
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(plots_dir / f"{args.split}_confusion_matrix.png", dpi=150)
    plt.close()

    print(f"\n{args.split} loss: {metrics['loss']:.4f}")
    print(f"{args.split} top-1 accuracy: {metrics['top1_accuracy']:.2f}%")
    print(f"{args.split} top-5 accuracy: {metrics['top5_accuracy']:.2f}%")
    print(f"Reports saved to: {reports_dir}")


if __name__ == "__main__":
    main()
