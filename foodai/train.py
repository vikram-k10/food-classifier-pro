"""Train a food classifier end to end (data audit -> training -> evaluation).

Examples::

    # best-quality default: pretrained EfficientNet-B0, two-phase fine-tuning
    python -m foodai.train --data-dir data/indian-food --arch efficientnet_b0

    # from-scratch baseline for the comparison table
    python -m foodai.train --data-dir data/indian-food --arch baseline_cnn --no-pretrained
"""
from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score

from .data import build_loaders, class_weights_from_items, mix_batch, prepare_data
from .evaluate import collect_logits, run_evaluation
from .models import ARCHS, build_model, get_head, save_checkpoint, set_backbone_trainable
from .utils import count_parameters, get_device, load_json, plot_history, save_json, seed_everything


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", required=True, help="folder containing one sub-folder of images per class")
    p.add_argument("--arch", default="efficientnet_b0", choices=ARCHS)
    p.add_argument("--name", default=None, help="run name (default: the architecture name)")
    p.add_argument("--out-dir", default="runs")
    p.add_argument("--img-size", type=int, default=224)
    p.add_argument("--epochs", type=int, default=25)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3, help="learning rate of the classifier head (or whole net if scratch)")
    p.add_argument("--backbone-lr-mult", type=float, default=0.1, help="backbone LR = lr * this (pretrained only)")
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--freeze-epochs", type=int, default=2, help="epochs with the backbone frozen (pretrained only)")
    p.add_argument("--warmup-epochs", type=int, default=1)
    p.add_argument("--dropout", type=float, default=0.3)
    p.add_argument("--label-smoothing", type=float, default=0.1)
    p.add_argument("--mixup", type=float, default=0.2, help="MixUp alpha (0 disables)")
    p.add_argument("--cutmix", type=float, default=1.0, help="CutMix alpha (0 disables)")
    p.add_argument("--mix-prob", type=float, default=0.5, help="probability a batch is mixed (0 disables)")
    p.add_argument("--class-weights", action="store_true", help="inverse-frequency loss weights for imbalanced data")
    p.add_argument("--no-augment", action="store_true", help="ablation: disable all augmentation and mixing")
    p.add_argument("--no-pretrained", action="store_true", help="train from random initialisation")
    p.add_argument("--patience", type=int, default=7, help="early-stopping patience in epochs")
    p.add_argument("--val-frac", type=float, default=0.15)
    p.add_argument("--test-frac", type=float, default=0.15)
    p.add_argument("--resplit", action="store_true", help="ignore an existing splits.json and split again")
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="auto")
    p.add_argument("--no-amp", action="store_true", help="disable mixed precision on CUDA")
    p.add_argument("--skip-eval", action="store_true", help="train only; run foodai.evaluate later")
    return p.parse_args(argv)


def make_param_groups(model, lr, backbone_mult, weight_decay):
    """Head vs backbone learning rates; no weight decay on biases / norm parameters."""
    head_ids = {id(p) for p in get_head(model).parameters()}
    buckets: dict[tuple[bool, bool], list] = {}
    for p in model.parameters():
        buckets.setdefault((id(p) in head_ids, p.ndim <= 1), []).append(p)
    return [
        {
            "params": params,
            "lr": lr * (1.0 if is_head else backbone_mult),
            "weight_decay": 0.0 if no_decay else weight_decay,
        }
        for (is_head, no_decay), params in buckets.items()
    ]


def cosine_with_warmup(total_steps: int, warmup_steps: int, floor: float = 0.01):
    def fn(step):
        if step < warmup_steps:
            return (step + 1) / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return floor + (1.0 - floor) * 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

    return fn


def train_one_epoch(model, loader, criterion, optimizer, scheduler, scaler, device, args, frozen, use_amp):
    model.train()
    if frozen:  # frozen backbone: keep BatchNorm statistics / stochastic depth fixed, train only the head
        model.eval()
        get_head(model).train()
    total_loss, total_correct, total_n = 0.0, 0.0, 0
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        x, y_a, y_b, lam = mix_batch(x, y, args.mixup, args.cutmix, args.mix_prob)
        with torch.autocast(device_type="cuda" if device.type == "cuda" else "cpu",
                            dtype=torch.float16 if device.type == "cuda" else torch.bfloat16,
                            enabled=use_amp):
            out = model(x)
            loss = lam * criterion(out, y_a) + (1.0 - lam) * criterion(out, y_b)
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        pred = out.argmax(dim=1)
        n = y.size(0)
        total_loss += float(loss) * n
        total_correct += lam * float((pred == y_a).sum()) + (1.0 - lam) * float((pred == y_b).sum())
        total_n += n
    return total_loss / total_n, total_correct / total_n


def validate(model, loader, device):
    logits, labels = collect_logits(model, loader, device)
    preds = logits.argmax(dim=1)
    return (
        float(F.cross_entropy(logits, labels)),
        float((preds == labels).float().mean()),
        float(f1_score(labels.numpy(), preds.numpy(), average="macro", zero_division=0)),
    )


def main(argv=None):
    args = parse_args(argv)
    seed_everything(args.seed)
    device = get_device(args.device)
    pretrained = not args.no_pretrained and args.arch != "baseline_cnn"
    if args.no_augment:
        args.mix_prob = 0.0
    run_dir = Path(args.out_dir) / (args.name or args.arch)
    run_dir.mkdir(parents=True, exist_ok=True)

    root, classes, splits = prepare_data(
        args.data_dir, run_dir, args.val_frac, args.test_frac, args.seed, reuse=not args.resplit
    )
    n_cls = len(classes)
    report = load_json(run_dir / "data_report.json") if (run_dir / "data_report.json").exists() else {}
    print(f"[data] {n_cls} classes | train {len(splits['train'])} / val {len(splits['val'])} / "
          f"test {len(splits['test'])} | root={root}")
    if report.get("corrupt_removed") or report.get("duplicates_removed"):
        print(f"[data] removed {len(report.get('corrupt_removed', []))} corrupt and "
              f"{len(report.get('duplicates_removed', []))} duplicate images")

    loaders = build_loaders(root, splits, args.img_size, args.batch_size, args.num_workers,
                            augment=not args.no_augment)
    model = build_model(args.arch, n_cls, pretrained=pretrained, dropout=args.dropout).to(device)
    n_params = count_parameters(model)
    print(f"[model] {args.arch} | pretrained={pretrained} | {n_params / 1e6:.2f}M params | device={device}")

    freeze_epochs = min(args.freeze_epochs, max(args.epochs - 1, 0)) if pretrained else 0
    backbone_mult = args.backbone_lr_mult if pretrained else 1.0
    weight = class_weights_from_items(splits["train"], n_cls).to(device) if args.class_weights else None
    criterion = nn.CrossEntropyLoss(weight=weight, label_smoothing=args.label_smoothing)

    optimizer = torch.optim.AdamW(make_param_groups(model, args.lr, backbone_mult, args.weight_decay))
    steps_per_epoch = len(loaders["train"])
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        cosine_with_warmup(args.epochs * steps_per_epoch, args.warmup_epochs * steps_per_epoch),
    )
    use_amp = device.type == "cuda" and not args.no_amp
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    if freeze_epochs > 0:
        set_backbone_trainable(model, False)
        print(f"[train] phase 1: backbone frozen for {freeze_epochs} epoch(s)")

    history, best_acc, best_loss, best_epoch, bad_epochs = [], -1.0, float("inf"), 0, 0
    ckpt_path = run_dir / "best.pt"
    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        frozen = epoch <= freeze_epochs
        if freeze_epochs > 0 and epoch == freeze_epochs + 1:
            set_backbone_trainable(model, True)
            print("[train] phase 2: fine-tuning the full network")
        t_ep = time.time()
        tr_loss, tr_acc = train_one_epoch(
            model, loaders["train"], criterion, optimizer, scheduler, scaler, device, args, frozen, use_amp
        )
        va_loss, va_acc, va_f1 = validate(model, loaders["val"], device)
        lr_now = max(g["lr"] for g in optimizer.param_groups)
        history.append({
            "epoch": epoch, "train_loss": tr_loss, "train_acc": tr_acc, "val_loss": va_loss,
            "val_acc": va_acc, "val_f1": va_f1, "lr": lr_now, "seconds": time.time() - t_ep,
        })
        improved = va_acc > best_acc or (va_acc == best_acc and va_loss < best_loss)
        if improved:
            best_acc, best_loss, best_epoch, bad_epochs = va_acc, va_loss, epoch, 0
            save_checkpoint(ckpt_path, model, args.arch, classes, args.img_size, args.dropout)
        else:
            bad_epochs += 1
        print(f"epoch {epoch:03d}/{args.epochs} | train loss {tr_loss:.4f} acc {tr_acc:.4f} | "
              f"val loss {va_loss:.4f} acc {va_acc:.4f} f1 {va_f1:.4f} | lr {lr_now:.2e}"
              f"{'  *best*' if improved else ''}")
        if not frozen and bad_epochs >= args.patience:
            print(f"[train] early stopping (no val improvement for {args.patience} epochs)")
            break
    train_seconds = time.time() - t0

    save_json(history, run_dir / "history.json")
    plot_history(history, run_dir / "history.png")
    config = {k: v for k, v in vars(args).items()}
    config.update({"pretrained": pretrained, "classes": classes, "params": n_params,
                   "dataset_root": str(root), "device": str(device)})
    save_json(config, run_dir / "config.json")

    summary = {"run_dir": str(run_dir), "best_epoch": best_epoch, "best_val_acc": best_acc}
    if not args.skip_eval:
        print("[eval] evaluating best checkpoint on the held-out test split ...")
        metrics = run_evaluation(run_dir, device=args.device, batch_size=args.batch_size,
                                 num_workers=args.num_workers)
        metrics.update({
            "run_name": run_dir.name, "pretrained": pretrained, "augmentation": not args.no_augment,
            "epochs_run": len(history), "best_epoch": best_epoch, "best_val_acc": best_acc,
            "train_seconds": train_seconds,
        })
        save_json(metrics, run_dir / "metrics.json")
        print(f"[eval] test accuracy {metrics['accuracy']:.4f} | top-3 {metrics['top3_accuracy']:.4f} | "
              f"macro-F1 {metrics['macro_f1']:.4f} | ECE {metrics['ece_before_calibration']:.4f} -> "
              f"{metrics['ece_after_calibration']:.4f}")
        summary["metrics"] = metrics
    print(f"[done] artifacts in {run_dir}/  (best.pt, metrics.json, confusion_matrix.png, history.png, ...)")
    return summary


if __name__ == "__main__":
    main()
