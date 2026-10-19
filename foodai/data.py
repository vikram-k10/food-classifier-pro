"""Dataset discovery, auditing, stratified splitting, transforms and MixUp/CutMix.

The loader is deliberately forgiving about how the Kaggle archive is laid out: point
``--data-dir`` at the extracted folder (or any parent of it) and the folder that holds
one sub-folder per class is detected automatically.
"""
from __future__ import annotations

import hashlib
import math
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T

from .utils import load_json, save_json

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


# --------------------------------------------------------------------------- discovery
def _is_image(p: Path) -> bool:
    return p.is_file() and p.suffix.lower() in IMG_EXT


def _class_dirs(folder: Path) -> list[Path]:
    """Sub-folders of ``folder`` that directly contain at least one image."""
    out = []
    for sub in folder.iterdir():
        if sub.is_dir() and any(_is_image(f) for f in sub.iterdir()):
            out.append(sub)
    return out


def find_dataset_root(path: str | Path) -> Path:
    """Return the folder whose children are the class folders."""
    path = Path(path)
    if not path.is_dir():
        raise FileNotFoundError(f"Dataset directory not found: {path}")
    best, best_n = None, 0
    candidates = [path] + [p for p in path.rglob("*") if p.is_dir()]
    for folder in candidates:
        n = len(_class_dirs(folder))
        if n > best_n:
            best, best_n = folder, n
    if best is None or best_n < 2:
        raise ValueError(
            f"Could not find an image-folder dataset under {path} "
            "(expected a folder with one sub-folder of images per class)."
        )
    return best


def scan_images(root: Path) -> tuple[list[str], list[tuple[str, int]]]:
    """Return (class names, [(relative_path, label), ...]) sorted deterministically."""
    classes = sorted(d.name for d in _class_dirs(root))
    samples: list[tuple[str, int]] = []
    for idx, name in enumerate(classes):
        for f in sorted((root / name).rglob("*")):
            if _is_image(f):
                samples.append((f.relative_to(root).as_posix(), idx))
    return classes, samples


# --------------------------------------------------------------------------- auditing
def audit_samples(root: Path, samples: list[tuple[str, int]]):
    """Drop unreadable files and exact byte-level duplicates (prevents train/test leakage)."""
    seen: dict[str, str] = {}
    clean: list[tuple[str, int]] = []
    corrupt, duplicates = [], []
    for rel, label in samples:
        fp = root / rel
        try:
            with Image.open(fp) as im:
                im.verify()
        except Exception:
            corrupt.append(rel)
            continue
        digest = hashlib.md5(fp.read_bytes()).hexdigest()
        if digest in seen:
            duplicates.append(rel)
            continue
        seen[digest] = rel
        clean.append((rel, label))
    return clean, corrupt, duplicates


# --------------------------------------------------------------------------- splits
def make_splits(samples, val_frac: float = 0.15, test_frac: float = 0.15, seed: int = 42):
    """Stratified train/val/test split (same class balance in every split)."""
    paths = [s[0] for s in samples]
    labels = [s[1] for s in samples]
    p_tv, p_te, y_tv, y_te = train_test_split(
        paths, labels, test_size=test_frac, stratify=labels, random_state=seed
    )
    rel_val = val_frac / (1.0 - test_frac)
    p_tr, p_va, y_tr, y_va = train_test_split(
        p_tv, y_tv, test_size=rel_val, stratify=y_tv, random_state=seed
    )
    return {
        "train": [[p, int(y)] for p, y in zip(p_tr, y_tr)],
        "val": [[p, int(y)] for p, y in zip(p_va, y_va)],
        "test": [[p, int(y)] for p, y in zip(p_te, y_te)],
    }


def prepare_data(data_dir, out_dir, val_frac=0.15, test_frac=0.15, seed=42, reuse=True):
    """Detect, audit and split the dataset; persist ``splits.json`` + ``data_report.json``."""
    out_dir = Path(out_dir)
    splits_path = out_dir / "splits.json"
    if reuse and splits_path.exists():
        info = load_json(splits_path)
        return Path(info["root"]), info["classes"], info["splits"]

    root = find_dataset_root(data_dir)
    classes, samples = scan_images(root)
    n_raw = len(samples)
    samples, corrupt, duplicates = audit_samples(root, samples)
    splits = make_splits(samples, val_frac, test_frac, seed)

    counts = {c: 0 for c in classes}
    for _, y in samples:
        counts[classes[y]] += 1
    report = {
        "root": str(root),
        "num_classes": len(classes),
        "images_found": n_raw,
        "images_used": len(samples),
        "corrupt_removed": corrupt,
        "duplicates_removed": duplicates,
        "class_counts": counts,
        "imbalance_ratio": float(max(counts.values()) / max(1, min(counts.values()))),
        "split_sizes": {k: len(v) for k, v in splits.items()},
        "seed": seed,
    }
    save_json({"root": str(root), "classes": classes, "splits": splits}, splits_path)
    save_json(report, out_dir / "data_report.json")
    return root, classes, splits


# --------------------------------------------------------------------------- transforms
def build_transforms(img_size: int, train: bool, augment: bool = True):
    normalize = T.Normalize(IMAGENET_MEAN, IMAGENET_STD)
    if train and augment:
        return T.Compose(
            [
                T.RandomResizedCrop(img_size, scale=(0.55, 1.0), ratio=(0.8, 1.25)),
                T.RandomHorizontalFlip(),
                T.TrivialAugmentWide(),
                T.ToTensor(),
                normalize,
                T.RandomErasing(p=0.25, scale=(0.02, 0.15)),
            ]
        )
    if train:  # "no augmentation" ablation: only the bare minimum to get a fixed-size tensor
        return T.Compose([T.Resize((img_size, img_size)), T.ToTensor(), normalize])
    return T.Compose(
        [
            T.Resize(int(round(img_size * 1.14))),
            T.CenterCrop(img_size),
            T.ToTensor(),
            normalize,
        ]
    )


def denormalize(x: torch.Tensor) -> torch.Tensor:
    """Undo ImageNet normalisation. ``x`` is (C,H,W) or (B,C,H,W)."""
    mean = torch.tensor(IMAGENET_MEAN, device=x.device).view(-1, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=x.device).view(-1, 1, 1)
    return (x * std + mean).clamp(0, 1)


def load_rgb(path_or_file) -> Image.Image:
    """Open an image, honour EXIF rotation (phone photos), convert to RGB."""
    img = Image.open(path_or_file)
    img = ImageOps.exif_transpose(img)
    return img.convert("RGB")


class FoodDataset(Dataset):
    def __init__(self, root: Path, items, transform):
        self.root = Path(root)
        self.items = [(p, int(y)) for p, y in items]
        self.transform = transform

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        rel, y = self.items[i]
        with Image.open(self.root / rel) as im:
            img = ImageOps.exif_transpose(im).convert("RGB")
        return self.transform(img), y


def build_loaders(root, splits, img_size, batch_size, num_workers=2, augment=True):
    ds = {
        "train": FoodDataset(root, splits["train"], build_transforms(img_size, True, augment)),
        "val": FoodDataset(root, splits["val"], build_transforms(img_size, False)),
        "test": FoodDataset(root, splits["test"], build_transforms(img_size, False)),
    }
    pin = torch.cuda.is_available()
    persistent = num_workers > 0
    loaders = {
        "train": DataLoader(
            ds["train"], batch_size=batch_size, shuffle=True, num_workers=num_workers,
            pin_memory=pin, drop_last=len(ds["train"]) > batch_size, persistent_workers=persistent,
        ),
        "val": DataLoader(
            ds["val"], batch_size=batch_size, shuffle=False, num_workers=num_workers,
            pin_memory=pin, persistent_workers=persistent,
        ),
        "test": DataLoader(
            ds["test"], batch_size=batch_size, shuffle=False, num_workers=num_workers,
            pin_memory=pin, persistent_workers=persistent,
        ),
    }
    return loaders


def class_weights_from_items(items, num_classes: int) -> torch.Tensor:
    """Inverse-frequency weights, normalised to mean 1 (handles class imbalance)."""
    counts = np.bincount([y for _, y in items], minlength=num_classes).astype(np.float64)
    counts = np.maximum(counts, 1.0)
    w = counts.sum() / (num_classes * counts)
    return torch.tensor(w / w.mean(), dtype=torch.float32)


# --------------------------------------------------------------------------- MixUp / CutMix
def mix_batch(x, y, mixup_alpha=0.2, cutmix_alpha=1.0, prob=0.5):
    """Apply MixUp or CutMix to a batch. Returns (x, y_a, y_b, lam).

    Loss is then ``lam * CE(out, y_a) + (1 - lam) * CE(out, y_b)``.
    """
    if prob <= 0 or np.random.rand() > prob or x.size(0) < 2:
        return x, y, y, 1.0
    use_cutmix = cutmix_alpha > 0 and (mixup_alpha <= 0 or np.random.rand() < 0.5)
    alpha = cutmix_alpha if use_cutmix else mixup_alpha
    if alpha <= 0:
        return x, y, y, 1.0
    lam = float(np.random.beta(alpha, alpha))
    idx = torch.randperm(x.size(0), device=x.device)
    if use_cutmix:
        h, w = x.shape[2], x.shape[3]
        cut = math.sqrt(1.0 - lam)
        ch, cw = int(h * cut), int(w * cut)
        cy, cx = np.random.randint(h), np.random.randint(w)
        y1, y2 = max(cy - ch // 2, 0), min(cy + ch // 2, h)
        x1, x2 = max(cx - cw // 2, 0), min(cx + cw // 2, w)
        x = x.clone()
        x[:, :, y1:y2, x1:x2] = x[idx, :, y1:y2, x1:x2]
        lam = 1.0 - ((y2 - y1) * (x2 - x1) / float(h * w))
    else:
        x = lam * x + (1.0 - lam) * x[idx]
    return x, y, y[idx], lam
