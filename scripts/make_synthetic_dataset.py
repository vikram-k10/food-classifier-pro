"""Create a tiny synthetic image-folder dataset (for smoke tests / trying the pipeline in seconds).

    python scripts/make_synthetic_dataset.py --out data/synthetic --classes 4 --per-class 30

Each class is a distinct colour + stripe pattern with noise, so a model can learn it quickly.
This is NOT food data - it only proves the code runs end to end.
"""
import argparse
from pathlib import Path

import numpy as np
from PIL import Image


def make_dataset(out, classes=4, per_class=30, size=64, seed=0):
    rng = np.random.RandomState(seed)
    out = Path(out)
    names = [f"class_{i}" for i in range(classes)]
    for ci, name in enumerate(names):
        folder = out / name
        folder.mkdir(parents=True, exist_ok=True)
        base = rng.randint(40, 215, size=3)
        period = 4 + 3 * ci
        for k in range(per_class):
            img = np.zeros((size, size, 3), dtype=np.float32) + base
            stripes = (np.arange(size) // period) % 2
            img[:, :, ci % 3] += 40 * stripes[None, :] if ci % 2 == 0 else 40 * stripes[:, None]
            img += rng.normal(0, 18, img.shape)
            Image.fromarray(img.clip(0, 255).astype(np.uint8)).save(folder / f"{name}_{k:03d}.jpg", quality=92)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/synthetic")
    ap.add_argument("--classes", type=int, default=4)
    ap.add_argument("--per-class", type=int, default=30)
    ap.add_argument("--size", type=int, default=64)
    a = ap.parse_args()
    print("Created", make_dataset(a.out, a.classes, a.per_class, a.size))
