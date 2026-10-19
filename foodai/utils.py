"""Small shared helpers: seeding, device selection, plotting, timing."""
from __future__ import annotations

import copy
import json
import random
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless-safe backend (servers, Colab, CI)
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402


def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device(preference: str = "auto") -> torch.device:
    if preference != "auto":
        return torch.device(preference)
    if torch.cuda.is_available():
        return torch.device("cuda")
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def count_parameters(model: torch.nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters()))


def pretty_name(class_name: str) -> str:
    """'butter_naan' -> 'Butter Naan'."""
    return class_name.replace("_", " ").replace("-", " ").strip().title()


def save_json(obj, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def load_json(path: str | Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


@torch.no_grad()
def benchmark_latency(model: torch.nn.Module, img_size: int, runs: int = 20, warmup: int = 3) -> float:
    """Average single-image CPU inference latency in milliseconds (fair across models)."""
    m = copy.deepcopy(model).cpu().eval()
    x = torch.randn(1, 3, img_size, img_size)
    for _ in range(warmup):
        m(x)
    start = time.perf_counter()
    for _ in range(runs):
        m(x)
    return float((time.perf_counter() - start) / runs * 1000.0)


def plot_history(history: list[dict], path: str | Path) -> None:
    """Loss / accuracy / learning-rate curves for a training run."""
    if not history:
        return
    epochs = [h["epoch"] for h in history]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].plot(epochs, [h["train_loss"] for h in history], label="train")
    axes[0].plot(epochs, [h["val_loss"] for h in history], label="val")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("epoch")
    axes[0].legend()

    axes[1].plot(epochs, [h["train_acc"] for h in history], label="train")
    axes[1].plot(epochs, [h["val_acc"] for h in history], label="val")
    axes[1].plot(epochs, [h["val_f1"] for h in history], label="val macro-F1", linestyle="--")
    axes[1].set_title("Accuracy / F1")
    axes[1].set_xlabel("epoch")
    axes[1].legend()

    axes[2].plot(epochs, [h["lr"] for h in history], color="tab:green")
    axes[2].set_title("Learning rate")
    axes[2].set_xlabel("epoch")

    for ax in axes:
        ax.grid(alpha=0.3)
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130)
    plt.close(fig)
