"""Grad-CAM: highlight the image regions that drove a prediction (model explainability)."""
from __future__ import annotations

import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from .data import denormalize  # noqa: E402
from .models import gradcam_layer  # noqa: E402
from .utils import pretty_name  # noqa: E402


class GradCAM:
    """Gradient-weighted Class Activation Mapping (Selvaraju et al., 2017).

    cam = ReLU( sum_k  mean_xy(d score / d A_k)  *  A_k )
    where A is the activation map of the last convolutional stage.
    """

    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module):
        self.model = model
        self.target_layer = target_layer

    def __call__(self, x: torch.Tensor, class_idx: int | None = None):
        """x: (1,C,H,W) normalised tensor. Returns (cam[H,W] in 0..1, class_idx)."""
        self.model.eval()
        store: dict[str, torch.Tensor] = {}
        handle = self.target_layer.register_forward_hook(lambda m, i, o: store.__setitem__("a", o))
        try:
            x = x.detach().clone().requires_grad_(True)  # guarantees a graph even if params are frozen
            with torch.enable_grad():
                logits = self.model(x)
                if class_idx is None:
                    class_idx = int(logits[0].argmax())
                score = logits[0, class_idx]
                grads = torch.autograd.grad(score, store["a"])[0]
        finally:
            handle.remove()
        acts = store["a"].detach()
        weights = grads.mean(dim=(2, 3), keepdim=True)
        cam = F.relu((weights * acts).sum(dim=1, keepdim=True))
        cam = F.interpolate(cam, size=x.shape[2:], mode="bilinear", align_corners=False)[0, 0]
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        return cam.cpu().numpy(), class_idx


def overlay_cam(rgb: np.ndarray, cam: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    """Blend a jet heat-map onto an RGB image (float 0..1 or uint8) -> uint8 image."""
    rgb = rgb.astype(np.float32)
    if rgb.max() > 1.0:
        rgb = rgb / 255.0
    heat = matplotlib.colormaps["jet"](cam)[..., :3]
    out = (1.0 - alpha) * rgb + alpha * heat
    return (out.clip(0, 1) * 255).astype(np.uint8)


def explain_grid(model, arch, dataset, classes, preds, labels, path, n_correct=4, n_wrong=4):
    """Save a figure with Grad-CAM overlays for some correct and some wrong test predictions."""
    preds, labels = np.asarray(preds), np.asarray(labels)
    chosen = [(int(i), True) for i in np.where(preds == labels)[0][:n_correct]]
    chosen += [(int(i), False) for i in np.where(preds != labels)[0][:n_wrong]]
    if not chosen:
        return
    device = next(model.parameters()).device
    engine = GradCAM(model, gradcam_layer(model, arch))
    cols = min(4, len(chosen))
    rows = math.ceil(len(chosen) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(3.2 * cols, 3.4 * rows), squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")
    for ax, (i, ok) in zip(axes.ravel(), chosen):
        x, y = dataset[i]
        cam, cls = engine(x.unsqueeze(0).to(device))
        img = denormalize(x).permute(1, 2, 0).numpy()
        ax.imshow(overlay_cam(img, cam))
        ax.set_title(
            f"true: {pretty_name(classes[int(y)])}\npred: {pretty_name(classes[cls])}",
            fontsize=8, color="green" if ok else "crimson",
        )
    fig.suptitle("Grad-CAM: where the model looks (green = correct, red = wrong)", fontsize=10)
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130)
    plt.close(fig)
