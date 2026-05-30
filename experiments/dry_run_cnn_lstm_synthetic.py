"""Forward-pass smoke test: CNNLSTMModel on random token ids.

    python -m experiments.dry_run_cnn_lstm_synthetic
"""

from __future__ import annotations

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.model import CNNLSTMModel
from src.models import ModelConfig


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--batch", type=int, default=4, help="Batch size B")
    p.add_argument("--seq-len", type=int, default=64, help="Sequence length T (>= max kernel size)")
    p.add_argument("--num-classes", type=int, default=10, help="Number of author classes C")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    config = ModelConfig(
        vocab_size=1000,
        embed_dim=64,
        num_filters=32,
        kernel_sizes=[2, 3, 4],
        lstm_hidden=64,
        lstm_layers=2,
        dropout=0.3,
        max_seq_len=256,
        num_classes=args.num_classes,
    )
    model = CNNLSTMModel(config)
    model.eval()

    if args.seq_len < max(config.kernel_sizes):
        print(
            f"seq-len {args.seq_len} must be >= max(kernel_sizes)={max(config.kernel_sizes)}",
            file=sys.stderr,
        )
        sys.exit(1)

    token_ids = torch.randint(0, config.vocab_size, (args.batch, args.seq_len))

    with torch.no_grad():
        logits = model(token_ids)

    assert logits.shape == (args.batch, args.num_classes)
    assert torch.isfinite(logits).all()

    print("CNNLSTMModel synthetic forward OK")
    print(f"  token_ids: {tuple(token_ids.shape)}")
    print(f"  logits:    {tuple(logits.shape)}  dtype={logits.dtype}")
    print(f"  logit min/max: {logits.min().item():.4f} / {logits.max().item():.4f}")


if __name__ == "__main__":
    main()
