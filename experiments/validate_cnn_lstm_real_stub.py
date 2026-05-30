"""Short training loop on real text with stub token ids (sanity check, not BPE).

    python -m experiments.validate_cnn_lstm_real_stub --epochs 5 --fetch-dataset
"""

from __future__ import annotations

import argparse
import os
import random
import sys

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.dataset import DEFAULT_CHANCHAL_CSV, DatasetLoader
from src.model import CNNLSTMModel
from src.models import ModelConfig
from src.preprocessing import Preprocessor
from src.stub_tokenisation import stub_char_token_ids


def _accuracy(
    model: nn.Module,
    token_ids: torch.Tensor,
    labels: torch.Tensor,
    device: torch.device,
    batch_size: int,
) -> float:
    model.eval()
    ds = TensorDataset(token_ids, labels)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            pred = logits.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.numel()
    return correct / total if total else 0.0


def _predict_all(
    model: nn.Module,
    token_ids: torch.Tensor,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    model.eval()
    out: list[np.ndarray] = []
    with torch.no_grad():
        for i in range(0, len(token_ids), batch_size):
            batch = token_ids[i : i + batch_size].to(device)
            logits = model(batch)
            out.append(logits.argmax(dim=1).cpu().numpy())
    return np.concatenate(out, axis=0)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", default=DEFAULT_CHANCHAL_CSV)
    p.add_argument("--fetch-dataset", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=32, dest="batch_size")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--vocab-size", type=int, default=10_000, dest="vocab_size")
    p.add_argument("--max-seq-len", type=int, default=256, dest="max_seq_len")
    p.add_argument("--embed-dim", type=int, default=128, dest="embed_dim")
    p.add_argument("--num-filters", type=int, default=64, dest="num_filters")
    p.add_argument("--lstm-hidden", type=int, default=128, dest="lstm_hidden")
    p.add_argument("--lstm-layers", type=int, default=2, dest="lstm_layers")
    p.add_argument("--dropout", type=float, default=0.3)
    p.add_argument(
        "--max-train-samples",
        type=int,
        default=None,
        dest="max_train_samples",
        help="Cap training rows for speed (default: all train rows).",
    )
    p.add_argument(
        "--max-val-samples",
        type=int,
        default=None,
        dest="max_val_samples",
        help="Cap validation rows (default: all val rows).",
    )
    args = p.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    loader = DatasetLoader()
    texts, labels = loader.load(args.dataset, fetch_if_missing=args.fetch_dataset)
    pre = Preprocessor()
    texts = pre.batch_clean(texts)
    paired = [(t, l) for t, l in zip(texts, labels) if t]
    if not paired:
        raise SystemExit("All texts empty after preprocessing.")
    texts, labels = zip(*paired)
    texts, labels = list(texts), list(labels)
    tr, va, _ = loader.split(list(texts), list(labels), seed=args.seed)
    num_classes = loader.num_authors

    train_texts = tr.texts
    train_label = tr.labels
    if args.max_train_samples is not None:
        train_texts = train_texts[: args.max_train_samples]
        train_label = train_label[: args.max_train_samples]
    val_texts = va.texts
    val_label = va.labels
    if args.max_val_samples is not None:
        val_texts = val_texts[: args.max_val_samples]
        val_label = val_label[: args.max_val_samples]

    y_train = torch.tensor(train_label, dtype=torch.long)
    y_val = torch.tensor(val_label, dtype=torch.long)

    x_train = stub_char_token_ids(
        train_texts, vocab_size=args.vocab_size, max_len=args.max_seq_len
    )
    x_val = stub_char_token_ids(
        val_texts, vocab_size=args.vocab_size, max_len=args.max_seq_len
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
    model = CNNLSTMModel(config).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    crit = nn.CrossEntropyLoss()
    random_baseline = 1.0 / num_classes

    train_loader = DataLoader(
        TensorDataset(x_train, y_train), batch_size=args.batch_size, shuffle=True
    )

    with torch.no_grad():
        pre_acc = _accuracy(model, x_val, y_val, device, args.batch_size)
    pre_logits = _predict_all(model, x_val, device, args.batch_size)
    pre_f1 = f1_score(
        val_label, pre_logits, average="macro", zero_division=0
    )

    print("CNN-LSTM validation (real text + stub token ids, not BPE)")
    print(f"  device:        {device}")
    print(f"  num_authors:  {num_classes}  (random baseline acc ~ {100 * random_baseline:.2f}%)")
    print(f"  train / val:  {len(x_train)} / {len(x_val)} samples")
    print()
    print(f"  before training:  val acc {100 * pre_acc:.2f}%  macro-F1 {pre_f1:.4f}")
    print("  " + "-" * 60)

    best_val = pre_acc
    first_loss: float | None = None
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        n = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            logits = model(xb)
            loss = crit(logits, yb)
            loss.backward()
            opt.step()
            total_loss += loss.item() * xb.size(0)
            n += xb.size(0)
        epoch_loss = total_loss / n
        if first_loss is None:
            first_loss = epoch_loss
        v_acc = _accuracy(model, x_val, y_val, device, args.batch_size)
        v_pred = _predict_all(model, x_val, device, args.batch_size)
        v_f1 = f1_score(val_label, v_pred, average="macro", zero_division=0)
        if v_acc > best_val:
            best_val = v_acc
        print(
            f"  epoch {epoch:2d}  train_loss {epoch_loss:.4f}  "
            f"val_acc {100 * v_acc:.2f}%  val_macro_f1 {v_f1:.4f}"
        )

    print("  " + "-" * 60)
    print(
        f"  best val acc:    {100 * best_val:.2f}%  "
        f"(> {100 * random_baseline:.2f}% random; pre-train was {100 * pre_acc:.2f}%)"
    )
    if first_loss is not None and args.epochs > 0:
        print(
            f"  train loss:      first epoch {first_loss:.4f}  "
            f"-> see whether later epochs in the log above show lower loss."
        )

    if best_val <= random_baseline * 1.1:
        print(
            "\n  Warning: val accuracy stayed near chance. Stub encoding is weak;"
            " pipeline check only — use run_cnn_lstm with BPE for real runs.",
            file=sys.stderr,
        )
    if best_val > pre_acc:
        print("\n  OK: validation accuracy improved after training (model + optimization path work).")
    else:
        print(
            "\n  Note: val acc did not beat pre-train; try more epochs, lower lr, or more --max-train-samples."
        )


if __name__ == "__main__":
    main()
