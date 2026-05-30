"""Render a LIME token-attribution figure from the shipped CNN-LSTM bundle.

Loads ``artifacts/best_model_bundle`` (model + tokeniser), reproduces the
stratified split recorded in ``training.json``, finds a confidently and
correctly attributed test tweet, runs :meth:`ExplainabilityModule.explain_lime`,
and plots the signed per-token contributions toward the predicted author.

    python -m experiments.make_explainability_figure
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from experiments.load_cnn_checkpoint import load_bundle, load_tokeniser
from src.dataset import DatasetLoader
from src.explainability import ExplainabilityModule
from src.preprocessing import Preprocessor


def _pick_example(model, tokeniser, texts, labels, max_len, device,
                  min_words=6, max_words=26, min_prob=0.55):
    best = None
    for i, (t, y) in enumerate(zip(texts, labels)):
        n_words = len(t.split())
        if not (min_words <= n_words <= max_words):
            continue
        ids = tokeniser.encode(t, max_length=max_len)
        with torch.no_grad():
            probs = torch.softmax(
                model(torch.tensor([ids], dtype=torch.long).to(device)), dim=-1
            )[0]
        pred = int(probs.argmax())
        p = float(probs[pred])
        if pred == y and p >= min_prob:
            if best is None or p > best[3]:
                best = (i, t, y, p, pred)
    return best


def main() -> None:
    repo = Path(__file__).resolve().parent.parent
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", default=str(repo / "artifacts" / "best_model_bundle"))
    ap.add_argument("--out-dir", default=str(repo / "Docs" / "Final-Project"
                                             / "acl-style-files-master" / "figures"))
    ap.add_argument("--png-dir", default=str(repo / "submitted clean repo" / "docs" / "figures"))
    ap.add_argument("--num-samples", type=int, default=600)
    args = ap.parse_args()

    bundle = Path(args.bundle)
    meta, model, device = load_bundle(bundle)
    tokeniser = load_tokeniser(bundle)
    max_len = int(meta["model_config"]["max_seq_len"])
    split_seed = int(meta.get("split_seed", 42))
    dataset_path = meta.get("cli_args", {}).get("dataset")

    loader = DatasetLoader()
    texts, labels = loader.load(dataset_path, fetch_if_missing=True)
    texts = Preprocessor().batch_clean(texts)
    paired = [(t, l) for t, l in zip(texts, labels) if t]
    texts, labels = map(list, zip(*paired))
    _, _, test = loader.split(texts, labels, seed=split_seed)

    picked = _pick_example(model, tokeniser, test.texts, test.labels, max_len, device)
    if picked is None:
        raise SystemExit("No confidently correct example found in test split.")
    idx, text, true_y, prob, pred = picked
    print(f"Example test idx={idx} author={true_y} pred={pred} p={prob:.3f}")
    print(f"Text: {text}")

    lime_exp = ExplainabilityModule().explain_lime(
        model, tokeniser, text, num_samples=args.num_samples, max_length=max_len
    )

    items = sorted(lime_exp.explanation.items(), key=lambda kv: kv[1])
    items = items[:6] + items[-8:] if len(items) > 14 else items
    toks = [k for k, _ in items]
    weights = [v for _, v in items]
    colors = ["#2a788e" if w >= 0 else "#cc5b45" for w in weights]

    fig, ax = plt.subplots(figsize=(5.6, 3.6))
    ax.barh(range(len(weights)), weights, color=colors)
    ax.set_yticks(range(len(toks)))
    ax.set_yticklabels(toks, fontsize=8)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel(f"LIME weight toward author {pred} (p={prob:.2f})")
    ax.set_title(f"Token contributions for a correctly attributed tweet",
                 fontsize=9)
    fig.tight_layout()

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "explainability_lime.pdf", bbox_inches="tight")
    png_dir = Path(args.png_dir); png_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(png_dir / "explainability_lime.png", bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote explainability_lime.pdf to {out_dir} and .png to {png_dir}")


if __name__ == "__main__":
    main()
