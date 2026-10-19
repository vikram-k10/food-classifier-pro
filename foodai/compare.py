"""Run several experiments on identical splits and produce the comparison + ablation tables.

    python -m foodai.compare --data-dir data/indian-food --epochs 20 --ablation --export-best

Outputs (in ``results/``): comparison.csv, comparison.md, comparison.png
"""
from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from . import train as train_mod  # noqa: E402
from .models import ARCHS  # noqa: E402

DEFAULT_ARCHS = ["baseline_cnn", "resnet18", "efficientnet_b0", "mobilenet_v3_large"]


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", required=True)
    p.add_argument("--archs", nargs="+", default=DEFAULT_ARCHS, choices=ARCHS)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--img-size", type=int, default=224)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--device", default="auto")
    p.add_argument("--out-dir", default="runs")
    p.add_argument("--results-dir", default="results")
    p.add_argument("--ablation", action="store_true",
                   help="also run the best pretrained arch without pretraining and without augmentation")
    p.add_argument("--export-best", action="store_true", help="copy the best checkpoint to models/best.pt")
    args = p.parse_args(argv)

    common = ["--data-dir", args.data_dir, "--epochs", str(args.epochs), "--img-size", str(args.img_size),
              "--batch-size", str(args.batch_size), "--num-workers", str(args.num_workers),
              "--device", args.device, "--out-dir", args.out_dir]
    jobs = [(a, ["--arch", a] + (["--no-pretrained"] if a == "baseline_cnn" else [])) for a in args.archs]

    rows, first_run = {}, None

    def run(name, extra):
        nonlocal first_run
        run_dir = Path(args.out_dir) / name
        if first_run is not None and not (run_dir / "splits.json").exists():
            run_dir.mkdir(parents=True, exist_ok=True)  # identical splits => fair comparison
            for f in ("splits.json", "data_report.json"):
                shutil.copy(first_run / f, run_dir / f)
        print(f"\n===== experiment: {name} =====")
        summary = train_mod.main(common + ["--name", name] + extra)
        if first_run is None:
            first_run = run_dir
        rows[name] = summary["metrics"]

    for name, extra in jobs:
        run(name, extra)

    if args.ablation:
        pre = {n: m for n, m in rows.items() if n != "baseline_cnn"}
        base_arch = max(pre, key=lambda n: pre[n]["accuracy"]) if pre else "efficientnet_b0"
        run(f"{base_arch}_no_pretraining", ["--arch", base_arch, "--no-pretrained"])
        run(f"{base_arch}_no_augmentation", ["--arch", base_arch, "--no-augment"])

    # ----------------------------------------------------------------- tables / chart
    results = Path(args.results_dir)
    results.mkdir(parents=True, exist_ok=True)
    cols = ["experiment", "accuracy", "top3_accuracy", "macro_f1", "ece_after_calibration",
            "params_M", "latency_cpu_ms", "train_minutes", "epochs_run"]
    table = []
    for name, m in rows.items():
        table.append({
            "experiment": name,
            "accuracy": m["accuracy"], "top3_accuracy": m["top3_accuracy"], "macro_f1": m["macro_f1"],
            "ece_after_calibration": m["ece_after_calibration"],
            "params_M": m["params"] / 1e6, "latency_cpu_ms": m["latency_cpu_ms"],
            "train_minutes": m["train_seconds"] / 60.0, "epochs_run": m["epochs_run"],
        })
    table.sort(key=lambda r: r["accuracy"], reverse=True)

    with open(results / "comparison.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(table)

    md = ["| Experiment | Acc | Top-3 | Macro-F1 | ECE | Params (M) | CPU ms/img | Train min | Epochs |",
          "|---|---|---|---|---|---|---|---|---|"]
    for r in table:
        md.append(f"| {r['experiment']} | {r['accuracy']:.4f} | {r['top3_accuracy']:.4f} | {r['macro_f1']:.4f} | "
                  f"{r['ece_after_calibration']:.4f} | {r['params_M']:.2f} | {r['latency_cpu_ms']:.1f} | "
                  f"{r['train_minutes']:.1f} | {r['epochs_run']} |")
    (results / "comparison.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    fig, ax = plt.subplots(figsize=(max(6, 1.4 * len(table)), 4.5))
    names = [r["experiment"] for r in table]
    xs = range(len(table))
    ax.bar([x - 0.2 for x in xs], [r["accuracy"] for r in table], 0.4, label="accuracy")
    ax.bar([x + 0.2 for x in xs], [r["macro_f1"] for r in table], 0.4, label="macro-F1")
    ax.set_xticks(list(xs))
    ax.set_xticklabels(names, rotation=25, ha="right", fontsize=8)
    ax.set_ylim(0, 1)
    ax.set_title("Experiment comparison (test set, identical splits)")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(results / "comparison.png", dpi=140)
    plt.close(fig)

    print("\n" + "\n".join(md))
    if args.export_best:
        best = table[0]["experiment"]
        Path("models").mkdir(exist_ok=True)
        shutil.copy(Path(args.out_dir) / best / "best.pt", "models/best.pt")
        print(f"\nBest model '{best}' copied to models/best.pt")
    return table


if __name__ == "__main__":
    main()
