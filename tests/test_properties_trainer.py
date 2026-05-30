"""Property-based tests for the Trainer.

**Validates: Requirements 6.4, 6.5**
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset
from hypothesis import given, settings
from hypothesis import strategies as st

from src.model import CNNLSTMModel
from src.models import ModelConfig
from src.trainer import Trainer

# Tiny model config kept constant across all examples for speed
_MODEL_CONFIG = ModelConfig(
    vocab_size=20,
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
_NUM_BATCHES = 3


def _make_loader() -> DataLoader:
    """Create a tiny synthetic DataLoader with random (token_ids, labels)."""
    token_ids = torch.randint(0, _MODEL_CONFIG.vocab_size, (_NUM_BATCHES * _BATCH_SIZE, _SEQ_LEN))
    labels = torch.randint(0, _MODEL_CONFIG.num_classes, (_NUM_BATCHES * _BATCH_SIZE,))
    dataset = TensorDataset(token_ids, labels)
    return DataLoader(dataset, batch_size=_BATCH_SIZE, shuffle=False)


@given(
    epochs=st.integers(min_value=1, max_value=5),
    patience=st.integers(min_value=1, max_value=3),
)
@settings(max_examples=20, deadline=None)
def test_training_termination_bound(epochs: int, patience: int) -> None:
    """Property 8: Training Termination Bound.

    For any training configuration with epochs > 0 and patience > 0, the
    Trainer should always terminate after at most `epochs` training iterations,
    regardless of the validation performance trajectory.

    **Validates: Requirements 6.4, 6.5**
    """
    model = CNNLSTMModel(_MODEL_CONFIG)
    train_loader = _make_loader()
    val_loader = _make_loader()
    trainer = Trainer()

    with tempfile.TemporaryDirectory() as tdir:
        checkpoint_path = str(Path(tdir) / "checkpoint.pt")
        history = trainer.train(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=epochs,
            lr=1e-3,
            patience=patience,
            checkpoint_path=checkpoint_path,
        )

    # Req 6.5: training must terminate within the configured max epochs
    assert len(history.train_losses) <= epochs, (
        f"Expected at most {epochs} training epochs, got {len(history.train_losses)}"
    )

    # Req 6.4 / 6.8: val_metrics recorded per epoch, bounded by epochs
    assert len(history.val_metrics) <= epochs, (
        f"Expected at most {epochs} val metric entries, got {len(history.val_metrics)}"
    )

    # Consistency: both lists must have the same length
    assert len(history.train_losses) == len(history.val_metrics), (
        "train_losses and val_metrics must have equal length"
    )


@given(
    epochs=st.integers(min_value=1, max_value=5),
    patience=st.integers(min_value=1, max_value=3),
)
@settings(max_examples=20, deadline=None)
def test_training_history_completeness(epochs: int, patience: int) -> None:
    """Property 17: Training History Completeness.

    For any completed training run, the returned TrainingHistory should contain
    per-epoch validation metrics with a length no greater than the configured
    maximum number of epochs.

    Specifically:
    - train_losses has one entry per completed epoch
    - val_metrics has one entry per completed epoch
    - Both lists have equal length
    - Length is <= epochs

    **Validates: Requirements 6.8**
    """
    model = CNNLSTMModel(_MODEL_CONFIG)
    train_loader = _make_loader()
    val_loader = _make_loader()
    trainer = Trainer()

    with tempfile.TemporaryDirectory() as tdir:
        checkpoint_path = str(Path(tdir) / "checkpoint.pt")
        history = trainer.train(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=epochs,
            lr=1e-3,
            patience=patience,
            checkpoint_path=checkpoint_path,
        )

    completed_epochs = len(history.train_losses)

    # Req 6.8: train_losses has one entry per completed epoch, bounded by epochs
    assert completed_epochs <= epochs, (
        f"train_losses length {completed_epochs} exceeds configured epochs {epochs}"
    )

    # Req 6.8: val_metrics has one entry per completed epoch
    assert len(history.val_metrics) == completed_epochs, (
        f"val_metrics length {len(history.val_metrics)} != "
        f"train_losses length {completed_epochs}; both must equal completed epochs"
    )

    # Req 6.8: both lists must have equal length (structural consistency)
    assert len(history.train_losses) == len(history.val_metrics), (
        "train_losses and val_metrics must have equal length"
    )

    # Each val_metrics entry must be a MetricsDict with scalar metrics in [0, 1]
    for i, metrics in enumerate(history.val_metrics):
        assert 0.0 <= metrics.accuracy <= 1.0, (
            f"Epoch {i + 1}: accuracy {metrics.accuracy} out of [0, 1]"
        )
        assert 0.0 <= metrics.f1_macro <= 1.0, (
            f"Epoch {i + 1}: f1_macro {metrics.f1_macro} out of [0, 1]"
        )
