"""Property-based tests for end-to-end pipeline reproducibility.

**Validates: Requirements 9.1**
"""

from __future__ import annotations

import os
import random
import tempfile

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from hypothesis import given, settings
from hypothesis import strategies as st

from src.model import CNNLSTMModel
from src.models import ModelConfig
from src.tokeniser import SubwordTokeniser
from src.trainer import Trainer

# ---------------------------------------------------------------------------
# Tiny model config for fast training
# ---------------------------------------------------------------------------

_MODEL_CONFIG = ModelConfig(
    vocab_size=300,
    embed_dim=8,
    num_filters=4,
    kernel_sizes=[2, 3],
    lstm_hidden=8,
    lstm_layers=1,
    dropout=0.0,
    num_classes=2,
)

_SEQ_LEN = 8   # must be >= max(kernel_sizes) = 3
_BATCH_SIZE = 4

# ---------------------------------------------------------------------------
# Synthetic dataset helpers
# ---------------------------------------------------------------------------

# Fixed synthetic texts and labels (2 authors, 20 samples each)
_AUTHOR_0_TEXTS = [
    f"the quick brown fox jumps over the lazy dog sample {i}" for i in range(20)
]
_AUTHOR_1_TEXTS = [
    f"pack my box with five dozen liquor jugs sample {i}" for i in range(20)
]
_ALL_TEXTS = _AUTHOR_0_TEXTS + _AUTHOR_1_TEXTS
_ALL_LABELS = [0] * 20 + [1] * 20


def _set_seeds(seed: int) -> None:
    """Set all relevant random seeds for reproducibility."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _run_pipeline(seed: int) -> tuple[float, float]:
    """Run a minimal training + evaluation pipeline with the given seed.

    Returns (accuracy, f1_macro) from evaluating the trained model on the
    test set.
    """
    _set_seeds(seed)

    # Train a tiny tokeniser on the synthetic texts
    tokeniser = SubwordTokeniser()
    tokeniser.train(_ALL_TEXTS, vocab_size=300, algorithm="bpe")

    # Encode all texts
    encoded = tokeniser.batch_encode(_ALL_TEXTS, max_length=_SEQ_LEN)

    # Build tensors
    token_ids = torch.tensor(encoded, dtype=torch.long)
    labels_tensor = torch.tensor(_ALL_LABELS, dtype=torch.long)

    # Simple 70/30 split (deterministic, no sklearn randomness needed)
    n_total = len(_ALL_TEXTS)
    n_train = int(n_total * 0.7)

    train_ids = token_ids[:n_train]
    train_labels = labels_tensor[:n_train]
    test_ids = token_ids[n_train:]
    test_labels = labels_tensor[n_train:]

    train_loader = DataLoader(
        TensorDataset(train_ids, train_labels),
        batch_size=_BATCH_SIZE,
        shuffle=False,
    )
    val_loader = DataLoader(
        TensorDataset(test_ids, test_labels),
        batch_size=_BATCH_SIZE,
        shuffle=False,
    )
    test_loader = DataLoader(
        TensorDataset(test_ids, test_labels),
        batch_size=_BATCH_SIZE,
        shuffle=False,
    )

    # Build and train model
    _set_seeds(seed)  # re-seed before model init to cover weight initialisation
    model = CNNLSTMModel(_MODEL_CONFIG)

    trainer = Trainer()
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as tmp:
        checkpoint_path = tmp.name

    trainer.train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=2,
        lr=1e-3,
        patience=5,
        checkpoint_path=checkpoint_path,
    )

    # Evaluate on test set
    metrics = trainer.evaluate(model, test_loader)
    return metrics.accuracy, metrics.f1_macro


@given(seed=st.integers(min_value=0, max_value=2**31 - 1))
@settings(max_examples=5, deadline=None)
def test_pipeline_reproducibility(seed: int) -> None:
    """Property 15: Pipeline Reproducibility.

    For any fixed random seed, executing the full training and evaluation
    pipeline twice on the same dataset should produce identical metric values
    in both runs.

    **Validates: Requirements 9.1**
    """
    accuracy_1, f1_1 = _run_pipeline(seed)
    accuracy_2, f1_2 = _run_pipeline(seed)

    assert accuracy_1 == accuracy_2, (
        f"Accuracy not reproducible with seed={seed}: "
        f"run1={accuracy_1}, run2={accuracy_2}"
    )
    assert f1_1 == f1_2, (
        f"F1-macro not reproducible with seed={seed}: "
        f"run1={f1_1}, run2={f1_2}"
    )
