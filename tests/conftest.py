import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from make_synthetic_dataset import make_dataset  # noqa: E402


@pytest.fixture(scope="session")
def synthetic_root(tmp_path_factory):
    """4 classes x 30 tiny images, nested one level deeper to exercise root auto-detection."""
    base = tmp_path_factory.mktemp("data")
    make_dataset(base / "archive" / "Images", classes=4, per_class=30, size=64)
    return base


@pytest.fixture(scope="session")
def trained_run(synthetic_root, tmp_path_factory):
    """One real (tiny) training run shared by the tests that need a checkpoint."""
    from foodai import train

    out = tmp_path_factory.mktemp("runs")
    # --no-augment: the toy classes are separated by colour, which colour-jitter style
    # augmentation would blur, so the learnability check runs on clean inputs.
    summary = train.main([
        "--data-dir", str(synthetic_root), "--arch", "baseline_cnn", "--img-size", "64",
        "--epochs", "12", "--batch-size", "16", "--num-workers", "0", "--device", "cpu",
        "--out-dir", str(out), "--no-pretrained", "--no-augment", "--patience", "20",
    ])
    return Path(summary["run_dir"]), summary


@pytest.fixture(scope="session")
def augmented_run(synthetic_root, tmp_path_factory):
    """Short run with the full augmentation + MixUp/CutMix + class-weight path switched on."""
    from foodai import train

    out = tmp_path_factory.mktemp("runs_aug")
    summary = train.main([
        "--data-dir", str(synthetic_root), "--arch", "resnet18", "--img-size", "64",
        "--epochs", "2", "--batch-size", "16", "--num-workers", "0", "--device", "cpu",
        "--out-dir", str(out), "--no-pretrained", "--mix-prob", "1.0", "--class-weights",
    ])
    return Path(summary["run_dir"]), summary
