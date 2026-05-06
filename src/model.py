"""
model.py
--------
Pretrained ResNet-18 model for dog breed classification.
"""

import torch.nn as nn
from torchvision import models


def build_resnet18(num_classes: int, pretrained: bool = True, freeze_backbone: bool = False):
    """
    Build a ResNet-18 transfer-learning model.

    Args:
        num_classes: number of dog breeds.
        pretrained: load ImageNet pretrained weights.
        freeze_backbone: train only the final classifier layer.
    """
    weights = models.ResNet18_Weights.DEFAULT if pretrained else None
    model = models.resnet18(weights=weights)

    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False

    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model


def count_trainable_parameters(model) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
