"""Shrink a large image-folder dataset once, so training is not bottlenecked by JPEG decoding.

    python scripts/resize_dataset.py --src <kaggle-dataset-path> --out data/indian-food-256 --size 256

Keeps the class-folder structure, shortens the *shorter* side to --size (aspect ratio preserved),
honours EXIF rotation, skips unreadable files and runs on all CPU cores. A 1.6 GB dataset typically
drops to a few hundred MB. Then train with:  --data-dir data/indian-food-256 --num-workers 4
(256 px is enough for the default 224 px training crops.)
"""
import argparse
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from PIL import Image, ImageOps

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _find_root(src: Path) -> Path:
    """Folder whose sub-folders hold the class images (same rule as foodai.data)."""
    best, best_n = None, 0
    for folder in [src] + [p for p in src.rglob("*") if p.is_dir()]:
        n = sum(1 for s in folder.iterdir()
                if s.is_dir() and any(f.is_file() and f.suffix.lower() in IMG_EXT for f in s.iterdir()))
        if n > best_n:
            best, best_n = folder, n
    if best is None or best_n < 2:
        sys.exit(f"No image-folder dataset found under {src}")
    return best


def _resize_one(args):
    src, dst, size, quality = args
    try:
        with Image.open(src) as im:
            im = ImageOps.exif_transpose(im).convert("RGB")
            w, h = im.size
            scale = size / min(w, h)
            if scale < 1.0:  # never upscale
                im = im.resize((max(1, round(w * scale)), max(1, round(h * scale))), Image.LANCZOS)
            dst.parent.mkdir(parents=True, exist_ok=True)
            im.save(dst.with_suffix(".jpg"), quality=quality, optimize=True)
        return True
    except Exception:
        return False


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--size", type=int, default=256, help="target length of the shorter side")
    ap.add_argument("--quality", type=int, default=90)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2)))
    a = ap.parse_args()

    root = _find_root(Path(a.src))
    out = Path(a.out)
    jobs = [(f, out / f.relative_to(root), a.size, a.quality)
            for f in sorted(root.rglob("*")) if f.is_file() and f.suffix.lower() in IMG_EXT]
    print(f"{len(jobs)} images found in {root}")
    with ProcessPoolExecutor(a.workers) as pool:
        ok = sum(pool.map(_resize_one, jobs, chunksize=32))
    print(f"resized {ok}/{len(jobs)} images ({len(jobs) - ok} unreadable skipped) -> {out}")

    def du(p):
        return sum(f.stat().st_size for f in Path(p).rglob("*") if f.is_file()) / 1e6
    print(f"size: {du(root):.0f} MB -> {du(out):.0f} MB")


if __name__ == "__main__":
    main()
