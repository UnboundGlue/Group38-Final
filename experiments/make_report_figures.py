"""Render report figures from saved metrics.

Reads ``results/metrics.json`` (CNN-LSTM confusion matrix + per-class F1) and a
member ``training.json`` (per-epoch trace) and writes three PDFs used by the
ACL report: confusion matrix heatmap, per-author F1 bar chart, and training
curves.

    python -m experiments.make_report_figures
    python -m experiments.make_report_figures --metrics results/metrics.json \
        --training artifacts/runs/<run>/member_train5/training.json \
        --out-dir Docs/Final-Project/acl-style-files-master/figures
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def confusion_matrix_fig(metrics: dict, out_path: Path) -> None:
    cm = np.asarray(metrics["cnn_lstm"]["confusion_matrix"], dtype=float)
    row_sums = cm.sum(axis=1, keepdims=True)
    norm = np.divide(cm, row_sums, out=np.zeros_like(cm), where=row_sums > 0)

    fig, ax = plt.subplots(figsize=(5.4, 4.6))
    im = ax.imshow(norm, cmap="viridis", vmin=0.0, vmax=1.0, aspect="equal")
    ax.set_xlabel("Predicted author index")
    ax.set_ylabel("True author index")
    n = cm.shape[0]
    ticks = list(range(0, n, 5))
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.tick_params(labelsize=7)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Row-normalised proportion", fontsize=8)
    cbar.ax.tick_params(labelsize=7)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def per_class_f1_fig(metrics: dict, out_path: Path) -> None:
    f1_map = metrics["cnn_lstm"]["f1_per_class"]
    items = sorted(f1_map.items(), key=lambda kv: int(kv[0]))
    idx = [int(k) for k, _ in items]
    vals = [float(v) for _, v in items]
    mean_f1 = float(np.mean(vals))

    colors = ["#2a788e" if v >= mean_f1 else "#cc5b45" for v in vals]
    fig, ax = plt.subplots(figsize=(7.0, 3.0))
    ax.bar(idx, vals, color=colors, width=0.8)
    ax.axhline(mean_f1, color="black", linestyle="--", linewidth=1.0,
               label=f"macro-F$_1$ = {mean_f1:.3f}")
    ax.set_xlabel("Author index")
    ax.set_ylabel("Test F$_1$")
    ax.set_ylim(0.0, 1.05)
    ax.set_xlim(-1, len(idx))
    ax.tick_params(labelsize=7)
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def training_curves_fig(training: dict, out_path: Path) -> None:
    per_epoch = training["per_epoch"]
    epochs = [e["epoch"] for e in per_epoch]
    loss = [e.get("train_loss") for e in per_epoch]
    val_f1 = [e.get("val_f1_macro") for e in per_epoch]

    fig, ax1 = plt.subplots(figsize=(5.6, 3.4))
    line1 = ax1.plot(epochs, loss, color="#cc5b45", label="train loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Training loss", color="#cc5b45")
    ax1.tick_params(axis="y", labelcolor="#cc5b45")

    ax2 = ax1.twinx()
    line2 = ax2.plot(epochs, val_f1, color="#2a788e", label="val macro-F$_1$")
    ax2.set_ylabel("Validation macro-F$_1$", color="#2a788e")
    ax2.tick_params(axis="y", labelcolor="#2a788e")
    ax2.set_ylim(0.0, 1.0)

    best_i = int(np.argmax([v if v is not None else -1 for v in val_f1]))
    ax2.axvline(epochs[best_i], color="gray", linestyle=":", linewidth=1.0)
    ax2.annotate(
        f"best val F$_1$ = {val_f1[best_i]:.3f}\n(epoch {epochs[best_i]})",
        xy=(epochs[best_i], val_f1[best_i]),
        xytext=(0.45, 0.25), textcoords="axes fraction", fontsize=8,
        arrowprops=dict(arrowstyle="->", color="gray", lw=0.8),
    )

    lines = line1 + line2
    ax1.legend(lines, [l.get_label() for l in lines], fontsize=8, loc="center right")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    repo = Path(__file__).resolve().parent.parent
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics", default=str(repo / "results" / "metrics.json"))
    ap.add_argument(
        "--training",
        default=str(repo / "artifacts" / "runs" / "run_20260515_113714"
                    / "member_train5" / "training.json"),
    )
    ap.add_argument(
        "--out-dir",
        default=str(repo / "Docs" / "Final-Project" / "acl-style-files-master" / "figures"),
    )
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics = _load(Path(args.metrics))

    confusion_matrix_fig(metrics, out_dir / "confusion_matrix.pdf")
    per_class_f1_fig(metrics, out_dir / "per_class_f1.pdf")
    print(f"Wrote confusion_matrix.pdf and per_class_f1.pdf to {out_dir}")

    training_path = Path(args.training)
    if training_path.is_file():
        training_curves_fig(_load(training_path), out_dir / "training_curves.pdf")
        print(f"Wrote training_curves.pdf (from {training_path.name}) to {out_dir}")
    else:
        print(f"Skipped training_curves.pdf (no training.json at {training_path})")


if __name__ == "__main__":
    main()
