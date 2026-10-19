"""Model zoo: a from-scratch baseline CNN plus pretrained torchvision backbones.

Every model exposes a "head" (the final classifier) so we can do two-phase transfer
learning: train only the head first, then fine-tune everything with a lower backbone LR.
"""
from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
import torchvision

PRETRAINED_ARCHS = ["resnet18", "resnet50", "efficientnet_b0", "mobilenet_v3_large", "convnext_tiny"]
ARCHS = ["baseline_cnn"] + PRETRAINED_ARCHS


class BaselineCNN(nn.Module):
    """VGG-style CNN trained from scratch - the reference point for transfer learning."""

    def __init__(self, num_classes: int, width: int = 32, dropout: float = 0.3):
        super().__init__()

        def block(cin, cout):
            return nn.Sequential(
                nn.Conv2d(cin, cout, 3, padding=1, bias=False),
                nn.BatchNorm2d(cout),
                nn.ReLU(inplace=False),
                nn.Conv2d(cout, cout, 3, padding=1, bias=False),
                nn.BatchNorm2d(cout),
                nn.ReLU(inplace=False),
                nn.MaxPool2d(2),
            )

        self.features = nn.Sequential(
            block(3, width),
            block(width, width * 2),
            block(width * 2, width * 4),
            block(width * 4, width * 8),
            block(width * 8, width * 8),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Dropout(dropout), nn.Linear(width * 8, num_classes)
        )

    def forward(self, x):
        return self.classifier(self.pool(self.features(x)))


def build_model(arch: str, num_classes: int, pretrained: bool = True, dropout: float = 0.3) -> nn.Module:
    if arch not in ARCHS:
        raise ValueError(f"Unknown architecture '{arch}'. Choose from {ARCHS}")
    if arch == "baseline_cnn":
        return BaselineCNN(num_classes, dropout=dropout)

    model = torchvision.models.get_model(arch, weights="DEFAULT" if pretrained else None)
    if arch.startswith("resnet"):
        model.fc = nn.Sequential(nn.Dropout(dropout), nn.Linear(model.fc.in_features, num_classes))
    elif arch.startswith("efficientnet"):
        model.classifier = nn.Sequential(
            nn.Dropout(dropout, inplace=False), nn.Linear(model.classifier[1].in_features, num_classes)
        )
    elif arch.startswith("mobilenet_v3"):
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
    elif arch.startswith("convnext"):
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
    return model


def get_head(model: nn.Module) -> nn.Module:
    return model.fc if hasattr(model, "fc") else model.classifier


def set_backbone_trainable(model: nn.Module, trainable: bool) -> None:
    head_ids = {id(p) for p in get_head(model).parameters()}
    for p in model.parameters():
        if id(p) not in head_ids:
            p.requires_grad = trainable


def gradcam_layer(model: nn.Module, arch: str) -> nn.Module:
    """The last convolutional feature map - what Grad-CAM visualises."""
    if arch == "baseline_cnn":
        return model.features[-1][-2]  # last ReLU, before the final max-pool
    if arch.startswith("resnet"):
        return model.layer4[-1]
    return model.features[-1]  # efficientnet / mobilenet_v3 / convnext


def save_checkpoint(path, model, arch, classes, img_size, dropout, extra=None):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    meta = {
        "arch": arch,
        "classes": list(classes),
        "img_size": int(img_size),
        "dropout": float(dropout),
        "temperature": 1.0,
        "threshold": 0.5,
        "test_metrics": {},
    }
    if extra:
        meta.update(extra)
    torch.save({"state_dict": model.state_dict(), "meta": meta}, path)


def load_checkpoint(path, device="cpu"):
    ckpt = torch.load(path, map_location=device, weights_only=True)
    meta = ckpt["meta"]
    model = build_model(meta["arch"], len(meta["classes"]), pretrained=False, dropout=meta["dropout"])
    model.load_state_dict(ckpt["state_dict"])
    model.to(device).eval()
    return model, meta


def update_checkpoint_meta(path, **updates) -> None:
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    ckpt["meta"].update(updates)
    torch.save(ckpt, path)
