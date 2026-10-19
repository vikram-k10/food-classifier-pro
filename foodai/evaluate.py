"""Rigorous evaluation: metrics, confusion matrix, calibration, abstention, Grad-CAM gallery.

Run on a finished training run::

    python -m foodai.evaluate --run-dir runs/efficientnet_b0
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

from .data import IMAGENET_MEAN, IMAGENET_STD, FoodDataset, build_transforms, find_dataset_root  # noqa: E402
from .gradcam import explain_grid  # noqa: E402
from .models import load_checkpoint, update_checkpoint_meta  # noqa: E402
from .utils import benchmark_latency, count_parameters, get_device, load_json, pretty_name, save_json  # noqa: E402


# --------------------------------------------------------------------------- core helpers
@torch.no_grad()
def collect_logits(model, loader, device):
    model.eval()
    all_logits, all_labels = [], []
    for x, y in loader:
        all_logits.append(model(x.to(device)).float().cpu())
        all_labels.append(y)
    return torch.cat(all_logits), torch.cat(all_labels)


def expected_calibration_error(probs: torch.Tensor, labels: torch.Tensor, n_bins: int = 15) -> float:
    """ECE: average |accuracy - confidence| over confidence bins (lower = better calibrated)."""
    conf, pred = probs.max(dim=1)
    correct = (pred == labels).float()
    edges = torch.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            ece += float(m.float().mean()) * abs(float(correct[m].mean()) - float(conf[m].mean()))
    return float(ece)


def fit_temperature(logits: torch.Tensor, labels: torch.Tensor) -> float:
    """Temperature scaling (Guo et al., 2017): learn one scalar T minimising validation NLL."""
    log_t = torch.zeros(1, requires_grad=True)
    opt = torch.optim.LBFGS([log_t], lr=0.1, max_iter=200)

    def closure():
        opt.zero_grad()
        loss = F.cross_entropy(logits / log_t.exp(), labels)
        loss.backward()
        return loss

    opt.step(closure)
    return float(log_t.exp().detach().clamp(0.05, 10.0))


def select_threshold(conf: np.ndarray, correct: np.ndarray, target_acc=0.90, min_coverage=0.5) -> float:
    """Lowest confidence threshold whose accepted predictions reach ``target_acc`` accuracy."""
    for t in np.linspace(0.05, 0.95, 91):
        accepted = conf >= t
        if accepted.mean() < min_coverage:
            break
        if correct[accepted].mean() >= target_acc:
            return float(round(t, 2))
    return 0.5


def selective_curve(conf: np.ndarray, correct: np.ndarray):
    """Risk-coverage curve: accuracy on the accepted subset as the threshold rises."""
    cov, acc, thr = [], [], []
    for t in np.linspace(0.0, 0.99, 100):
        accepted = conf >= t
        if accepted.sum() == 0:
            break
        thr.append(float(t))
        cov.append(float(accepted.mean()))
        acc.append(float(correct[accepted].mean()))
    return thr, cov, acc


@torch.no_grad()
def ood_sanity(model, temperature, threshold, device, img_size, n=64):
    """Cheap sanity check: synthetic non-food inputs should mostly be rejected.

    This is NOT a rigorous out-of-distribution benchmark - it only catches the worst case
    of a model that is confidently wrong on noise / flat colours / gradients.
    """
    g = torch.Generator().manual_seed(0)
    shape = (n, 3, img_size, img_size)
    noise = torch.rand(shape, generator=g)
    solid = torch.rand(n, 3, 1, 1, generator=g).expand(shape)
    ramp = torch.linspace(0, 1, img_size).view(1, 1, 1, -1).expand(shape) * torch.rand(n, 3, 1, 1, generator=g)
    mean = torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(1, 3, 1, 1)
    out = {}
    for name, imgs in [("uniform_noise", noise), ("solid_color", solid), ("gradient", ramp)]:
        logits = model(((imgs - mean) / std).to(device)).float().cpu()
        conf = F.softmax(logits / temperature, dim=1).max(dim=1).values
        out[name] = float((conf < threshold).float().mean())
    return out


# --------------------------------------------------------------------------- plots
def plot_confusion(cm: np.ndarray, classes, path):
    norm = cm.astype(np.float64) / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    n = len(classes)
    size = max(6, 0.55 * n)
    fig, ax = plt.subplots(figsize=(size, size))
    im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    labels = [pretty_name(c) for c in classes]
    ax.set_xticklabels(labels, rotation=60, ha="right", fontsize=8)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Normalised confusion matrix (test set)")
    if n <= 30:
        for i in range(n):
            for j in range(n):
                if norm[i, j] >= 0.01:
                    ax.text(j, i, f"{norm[i, j]:.2f}", ha="center", va="center", fontsize=6,
                            color="white" if norm[i, j] > 0.5 else "black")
    fig.colorbar(im, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def plot_per_class_f1(report: dict, classes, path):
    f1s = [report[c]["f1-score"] for c in classes]
    order = np.argsort(f1s)
    fig, ax = plt.subplots(figsize=(7, max(4, 0.32 * len(classes))))
    ax.barh([pretty_name(classes[i]) for i in order], [f1s[i] for i in order], color="tab:blue")
    ax.set_xlim(0, 1)
    ax.set_xlabel("F1-score")
    ax.set_title("Per-class F1 (test set, weakest first)")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_reliability(probs_raw, probs_cal, labels, path, n_bins=10):
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], "k--", label="perfect")
    edges = np.linspace(0, 1, n_bins + 1)
    for probs, name in [(probs_raw, "before temperature scaling"), (probs_cal, "after temperature scaling")]:
        conf, pred = probs.max(dim=1)
        conf, correct = conf.numpy(), (pred == labels).float().numpy()
        xs, ys = [], []
        for lo, hi in zip(edges[:-1], edges[1:]):
            m = (conf > lo) & (conf <= hi)
            if m.sum() > 0:
                xs.append(conf[m].mean())
                ys.append(correct[m].mean())
        ax.plot(xs, ys, marker="o", label=name)
    ax.set_xlabel("Confidence")
    ax.set_ylabel("Accuracy")
    ax.set_title("Reliability diagram (test set)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_selective(thr, cov, acc, chosen_threshold, path):
    fig, ax = plt.subplots(figsize=(5.5, 4))
    ax.plot(cov, acc, color="tab:purple")
    if thr:
        idx = int(np.argmin(np.abs(np.array(thr) - chosen_threshold)))
        ax.scatter([cov[idx]], [acc[idx]], color="crimson", zorder=3,
                   label=f"chosen threshold = {chosen_threshold:.2f}")
    ax.set_xlabel("Coverage (fraction of images answered)")
    ax.set_ylabel("Accuracy on answered images")
    ax.set_title("Accuracy vs coverage when rejecting low confidence")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


# --------------------------------------------------------------------------- main routine
def run_evaluation(run_dir, data_dir=None, device="auto", batch_size=64, num_workers=2, ckpt_name="best.pt"):
    run_dir = Path(run_dir)
    dev = get_device(device)
    model, meta = load_checkpoint(run_dir / ckpt_name, dev)
    classes, img_size, arch = meta["classes"], meta["img_size"], meta["arch"]

    info = load_json(run_dir / "splits.json")
    root = find_dataset_root(data_dir) if data_dir else Path(info["root"])
    tf = build_transforms(img_size, train=False)
    val_ds = FoodDataset(root, info["splits"]["val"], tf)
    test_ds = FoodDataset(root, info["splits"]["test"], tf)
    kw = dict(batch_size=batch_size, shuffle=False, num_workers=num_workers)
    val_logits, val_labels = collect_logits(model, torch.utils.data.DataLoader(val_ds, **kw), dev)
    test_logits, test_labels = collect_logits(model, torch.utils.data.DataLoader(test_ds, **kw), dev)

    # --- calibration + abstention threshold are fitted on VALIDATION only (no test leakage)
    temperature = fit_temperature(val_logits, val_labels)
    val_probs_cal = F.softmax(val_logits / temperature, dim=1)
    val_conf, val_pred = val_probs_cal.max(dim=1)
    threshold = select_threshold(val_conf.numpy(), (val_pred == val_labels).numpy())

    # --- test metrics
    y_true = test_labels.numpy()
    probs_raw = F.softmax(test_logits, dim=1)
    probs_cal = F.softmax(test_logits / temperature, dim=1)
    y_pred = probs_raw.argmax(dim=1).numpy()
    n_cls = len(classes)
    k = min(3, n_cls)
    top3 = float((test_logits.topk(k, dim=1).indices == test_labels[:, None]).any(dim=1).float().mean())
    report = classification_report(
        y_true, y_pred, labels=list(range(n_cls)), target_names=classes, output_dict=True, zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred, labels=list(range(n_cls)))

    conf_cal = probs_cal.max(dim=1).values.numpy()
    correct = (y_pred == y_true)
    accepted = conf_cal >= threshold
    thr_curve, cov_curve, acc_curve = selective_curve(conf_cal, correct)

    off = [(int(cm[i, j]), i, j) for i in range(n_cls) for j in range(n_cls) if i != j and cm[i, j] > 0]
    off.sort(reverse=True)
    top_confusions = [{"true": classes[i], "predicted": classes[j], "count": c} for c, i, j in off[:10]]

    metrics = {
        "arch": arch,
        "num_test_images": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "top3_accuracy": top3,
        "macro_precision": float(report["macro avg"]["precision"]),
        "macro_recall": float(report["macro avg"]["recall"]),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", labels=list(range(n_cls)), zero_division=0)),
        "weighted_f1": float(report["weighted avg"]["f1-score"]),
        "nll_before_calibration": float(F.cross_entropy(test_logits, test_labels)),
        "nll_after_calibration": float(F.nll_loss(torch.log(probs_cal.clamp_min(1e-12)), test_labels)),
        "ece_before_calibration": expected_calibration_error(probs_raw, test_labels),
        "ece_after_calibration": expected_calibration_error(probs_cal, test_labels),
        "temperature": temperature,
        "reject_threshold": threshold,
        "coverage_at_threshold": float(accepted.mean()),
        "accuracy_when_answered": float(correct[accepted].mean()) if accepted.any() else None,
        "synthetic_input_rejection_rate": ood_sanity(model, temperature, threshold, dev, img_size),
        "top_confusions": top_confusions,
        "params": count_parameters(model),
        "latency_cpu_ms": benchmark_latency(model, img_size),
        "per_class": {c: {k_: float(v) for k_, v in report[c].items()} for c in classes},
    }
    save_json(metrics, run_dir / "metrics.json")

    with open(run_dir / "per_class_metrics.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["class", "precision", "recall", "f1", "support"])
        for c in classes:
            r = report[c]
            w.writerow([c, f"{r['precision']:.4f}", f"{r['recall']:.4f}", f"{r['f1-score']:.4f}", int(r["support"])])

    plot_confusion(cm, classes, run_dir / "confusion_matrix.png")
    plot_per_class_f1(report, classes, run_dir / "per_class_f1.png")
    plot_reliability(probs_raw, probs_cal, test_labels, run_dir / "reliability_diagram.png")
    plot_selective(thr_curve, cov_curve, acc_curve, threshold, run_dir / "selective_prediction.png")
    explain_grid(model, arch, test_ds, classes, y_pred, y_true, run_dir / "gradcam_examples.png")

    # --- bake calibration + headline metrics into the checkpoint so the web app is self-contained
    update_checkpoint_meta(
        run_dir / ckpt_name,
        temperature=float(temperature),
        threshold=float(threshold),
        test_metrics={
            "accuracy": metrics["accuracy"],
            "top3_accuracy": metrics["top3_accuracy"],
            "macro_f1": metrics["macro_f1"],
            "ece": metrics["ece_after_calibration"],
        },
    )
    return metrics


def main(argv=None):
    p = argparse.ArgumentParser(description="Evaluate a finished training run on its held-out test split.")
    p.add_argument("--run-dir", required=True)
    p.add_argument("--data-dir", default=None, help="override dataset location if you moved it")
    p.add_argument("--device", default="auto")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--num-workers", type=int, default=2)
    args = p.parse_args(argv)
    m = run_evaluation(args.run_dir, args.data_dir, args.device, args.batch_size, args.num_workers)
    print(f"accuracy={m['accuracy']:.4f}  top3={m['top3_accuracy']:.4f}  macro_f1={m['macro_f1']:.4f}  "
          f"ECE {m['ece_before_calibration']:.4f} -> {m['ece_after_calibration']:.4f}")
    return m


if __name__ == "__main__":
    main()
