"""
predict.py
----------
Predict dog breed for one image using a trained checkpoint.
"""

import argparse
import os
import sys

import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(__file__))

from model import build_resnet18
from transforms import get_transforms


def parse_args():
    parser = argparse.ArgumentParser(description="Predict dog breed for one image.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--top-k", type=int, default=5)
    return parser.parse_args()


def pretty_name(raw):
    return raw.split("-", 1)[-1].replace("_", " ")


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

    transform = get_transforms("test")
    image = Image.open(args.image).convert("RGB")
    tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1).squeeze(0)

    k = min(args.top_k, len(class_names))
    values, indices = torch.topk(probs, k)

    print(f"\nImage: {args.image}")
    for prob, idx in zip(values, indices):
        print(f"{pretty_name(class_names[idx.item()]):<35} {prob.item() * 100:6.2f}%")


if __name__ == "__main__":
    main()
