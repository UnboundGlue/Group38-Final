"""Load real text, preprocess, split; forward pass with stub token ids (not BPE).

    python -m experiments.dry_run_cnn_lstm_real_text_stub --fetch-dataset
"""

from __future__ import annotations

import argparse
import os
import random
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.dataset import DEFAULT_CHANCHAL_CSV, DatasetLoader
from src.model import CNNLSTMModel
from src.models import ModelConfig
from src.preprocessing import Preprocessor
from src.stub_tokenisation import stub_char_token_ids


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--dataset",
        default=DEFAULT_CHANCHAL_CSV,
        help="CSV/JSON path (default: Chanchal 50_tweets... relative to project root).",
    )
    p.add_argument(
        "--fetch-dataset",
        action="store_true",
        help="Clone AuthorIdentification if the file is missing.",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--batch-size", type=int, default=8, help="Number of training texts in one forward.")
    p.add_argument("--vocab-size", type=int, default=10_000, dest="vocab_size")
    p.add_argument("--max-seq-len", type=int, default=256, dest="max_seq_len")
    p.add_argument("--embed-dim", type=int, default=128, dest="embed_dim")
    p.add_argument("--num-filters", type=int, default=64, dest="num_filters")
    p.add_argument("--lstm-hidden", type=int, default=128, dest="lstm_hidden")
    p.add_argument("--lstm-layers", type=int, default=2, dest="lstm_layers")
    p.add_argument("--dropout", type=float, default=0.3)
    args = p.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    loader = DatasetLoader()
    texts, labels = loader.load(args.dataset, fetch_if_missing=args.fetch_dataset)
    pre = Preprocessor()
    texts = pre.batch_clean(texts)
    paired = [(t, l) for t, l in zip(texts, labels) if t]
    if not paired:
        raise SystemExit("All texts empty after preprocessing.")
    texts, labels = zip(*paired)
    texts, labels = list(texts), list(labels)

    train, _, _ = loader.split(list(texts), list(labels), seed=args.seed)
    num_classes = loader.num_authors

    n = min(args.batch_size, len(train.texts))
    batch_texts = train.texts[:n]
    # Labels not needed for forward-only smoke; model needs correct num_classes

    token_ids = stub_char_token_ids(
        batch_texts, vocab_size=args.vocab_size, max_len=args.max_seq_len
    )

    config = ModelConfig(
        vocab_size=args.vocab_size,
        embed_dim=args.embed_dim,
        num_filters=args.num_filters,
        kernel_sizes=[2, 3, 4],
        lstm_hidden=args.lstm_hidden,
        lstm_layers=args.lstm_layers,
        dropout=args.dropout,
        max_seq_len=args.max_seq_len,
        num_classes=num_classes,
    )
    model = CNNLSTMModel(config)
    model.eval()

    with torch.no_grad():
        logits = model(token_ids)

    assert torch.isfinite(logits).all()
    assert logits.shape == (n, num_classes)

    print("Forward on preprocessed text OK (stub token ids)")
    print(f"  dataset:     {args.dataset}")
    print(f"  train split: {len(train.texts)} rows; using batch: {n}")
    print(f"  authors:     {num_classes}  (num_classes for head)")
    print(f"  token_ids:   {tuple(token_ids.shape)}  (char-hash stub, pad=0)")
    print(f"  logits:      {tuple(logits.shape)}")
    print(f"  logit min/max: {logits.min().item():.4f} / {logits.max().item():.4f}")


if __name__ == "__main__":
    main()
