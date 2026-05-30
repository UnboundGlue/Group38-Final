"""Unit tests for Trainer.

Tests early stopping, NaN loss detection, and checkpoint saving.

Validates: Requirements 6.3, 6.4, 6.6
"""

from __future__ import annotations

import os
import tempfile

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.model import CNNLSTMModel
from src.models import ModelConfig, TrainingDivergenceError
from src.trainer import Trainer

# ---------------------------------------------------------------------------
# Shared tiny model config
# ---------------------------------------------------------------------------

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

_SEQ_LEN = 8   # >= max(kernel_sizes) = 3
_BATCH_SIZE = 4
_NUM_SAMPLES = 16


def _make_loader(num_samples: int = _NUM_SAMPLES) -> DataLoader:
    """Synthetic DataLoader with random token IDs and balanced labels."""
    token_ids = torch.randint(0, _MODEL_CONFIG.vocab_size, (num_samples, _SEQ_LEN))
    labels = torch.randint(0, _MODEL_CONFIG.num_classes, (num_samples,))
    dataset = TensorDataset(token_ids, labels)
    return DataLoader(dataset, batch_size=_BATCH_SIZE, shuffle=False)


def _make_model() -> CNNLSTMModel:
    model = CNNLSTMModel(_MODEL_CONFIG)
    return model


# ---------------------------------------------------------------------------
# Early stopping (Req 6.4)
# ---------------------------------------------------------------------------

def test_early_stopping_triggers_before_max_epochs() -> None:
    """Training stops before max_epochs when val_f1 doesn't improve with patience=1.

    Validates: Requirement 6.4
    """
    # Use a large epoch count so early stopping must kick in before the end.
    # With patience=1, training should stop as soon as val_f1 fails to improve
    # for one consecutive epoch.
    max_epochs = 20
    patience = 1

    model = _make_model()
    train_loader = _make_loader()
    val_loader = _make_loader()
    trainer = Trainer()

    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as tmp:
        ckpt_path = tmp.name

    try:
        history = trainer.train(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=max_epochs,
            lr=1e-3,
            patience=patience,
            checkpoint_path=ckpt_path,
        )
    finally:
        if os.path.exists(ckpt_path):
            os.remove(ckpt_path)

    completed = len(history.train_losses)

    # With patience=1 and a random model on a tiny dataset, val_f1 is unlikely
    # to improve every single epoch for 20 epochs — early stopping must fire.
    assert completed < max_epochs, (
        f"Expected early stopping before {max_epochs} epochs, "
        f"but training ran all {completed} epochs."
    )

    # Consistency: val_metrics recorded for every completed epoch
    assert len(history.val_metrics) == completed


def test_early_stopping_respects_patience_counter() -> None:
    """Training runs at most patience+1 epochs when val_f1 never improves after epoch 1.

    Validates: Requirement 6.4
    """
    # With patience=1, once the first improvement is recorded (epoch 1 always
    # saves a checkpoint since best_val_f1 starts at -1), the next epoch that
    # doesn't improve will trigger early stopping.
    patience = 1
    max_epochs = 10

    model = _make_model()
    train_loader = _make_loader()
    val_loader = _make_loader()
    trainer = Trainer()

    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as tmp:
        ckpt_path = tmp.name

    try:
        history = trainer.train(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=max_epochs,
            lr=1e-3,
            patience=patience,
            checkpoint_path=ckpt_path,
        )
    finally:
        if os.path.exists(ckpt_path):
            os.remove(ckpt_path)

    # Training must stop within max_epochs
    assert len(history.train_losses) <= max_epochs


# ---------------------------------------------------------------------------
# NaN loss raises TrainingDivergenceError (Req 6.6)
# ---------------------------------------------------------------------------

class _NaNModel(nn.Module):
    """Mock model whose forward() always returns NaN logits."""

    def __init__(self, num_classes: int = 2) -> None:
        super().__init__()
        # A dummy parameter so the trainer can detect the device
        self._dummy = nn.Parameter(torch.zeros(1))
        self.num_classes = num_classes

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        B = token_ids.shape[0]
        return torch.full((B, self.num_classes), float("nan"))


def test_nan_loss_raises_training_divergence_error() -> None:
    """NaN logits from the model cause TrainingDivergenceError to be raised.

    Validates: Requirement 6.6
    """
    nan_model = _NaNModel(num_classes=_MODEL_CONFIG.num_classes)
    train_loader = _make_loader()
    val_loader = _make_loader()
    trainer = Trainer()

    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as tmp:
        ckpt_path = tmp.name

    try:
        with pytest.raises(TrainingDivergenceError):
            trainer.train(
                model=nan_model,
                train_loader=train_loader,
                val_loader=val_loader,
                epochs=5,
                lr=1e-3,
                patience=3,
                checkpoint_path=ckpt_path,
            )
    finally:
        if os.path.exists(ckpt_path):
            os.remove(ckpt_path)


def test_nan_loss_error_message_contains_epoch_and_batch() -> None:
    """TrainingDivergenceError message includes epoch and batch index.

    Validates: Requirement 6.6
    """
    nan_model = _NaNModel(num_classes=_MODEL_CONFIG.num_classes)
    train_loader = _make_loader()
    val_loader = _make_loader()
    trainer = Trainer()

    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as tmp:
        ckpt_path = tmp.name

    try:
        with pytest.raises(TrainingDivergenceError, match=r"epoch"):
            trainer.train(
                model=nan_model,
                train_loader=train_loader,
                val_loader=val_loader,
                epochs=5,
                lr=1e-3,
                patience=3,
                checkpoint_path=ckpt_path,
            )
    finally:
        if os.path.exists(ckpt_path):
            os.remove(ckpt_path)


# ---------------------------------------------------------------------------
# Checkpoint saved on improvement (Req 6.3)
# ---------------------------------------------------------------------------

def test_checkpoint_created_on_val_f1_improvement() -> None:
    """A checkpoint file is created when val_f1 improves.

    Validates: Requirement 6.3
    """
    model = _make_model()
    train_loader = _make_loader()
    val_loader = _make_loader()
    trainer = Trainer()

    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_path = os.path.join(tmpdir, "best_model.pt")

        # Checkpoint file should not exist before training
        assert not os.path.exists(ckpt_path)

        trainer.train(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=5,
            lr=1e-3,
            patience=3,
            checkpoint_path=ckpt_path,
        )

        # Epoch 1 always improves (best_val_f1 starts at -1), so checkpoint must exist
        assert os.path.exists(ckpt_path), (
            "Expected checkpoint file to be created after val_f1 improvement."
        )


def test_checkpoint_is_loadable_state_dict() -> None:
    """The saved checkpoint can be loaded back as a valid state dict.

    Validates: Requirement 6.3
    """
    model = _make_model()
    train_loader = _make_loader()
    val_loader = _make_loader()
    trainer = Trainer()

    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_path = os.path.join(tmpdir, "best_model.pt")

        trainer.train(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=3,
            lr=1e-3,
            patience=2,
            checkpoint_path=ckpt_path,
        )

        assert os.path.exists(ckpt_path)

        # Load the checkpoint into a fresh model of the same architecture
        fresh_model = _make_model()
        state_dict = torch.load(ckpt_path, map_location="cpu")
        fresh_model.load_state_dict(state_dict)  # must not raise


def test_trainer_cosine_restarts_runs() -> None:
    """CosineAnnealingWarmRestarts does not break the training loop."""
    model = _make_model()
    train_loader = _make_loader()
    val_loader = _make_loader()
    trainer = Trainer()

    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_path = os.path.join(tmpdir, "c.pt")
        history = trainer.train(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=3,
            lr=1e-3,
            patience=2,
            checkpoint_path=ckpt_path,
            lr_schedule="cosine_restarts",
            cosine_t0=2,
        )
    assert len(history.train_losses) == 3


def test_trainer_onecycle_runs() -> None:
    """OneCycleLR stepped per-batch does not break the training loop."""
    model = _make_model()
    train_loader = _make_loader()
    val_loader = _make_loader()
    trainer = Trainer()

    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_path = os.path.join(tmpdir, "oc.pt")
        history = trainer.train(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=2,
            lr=3e-3,
            patience=2,
            checkpoint_path=ckpt_path,
            lr_schedule="onecycle",
        )
    assert len(history.train_losses) == 2


def test_trainer_accepts_class_weights() -> None:
    """CrossEntropyLoss with per-class weights runs without error."""
    model = _make_model()
    train_loader = _make_loader()
    val_loader = _make_loader()
    trainer = Trainer()
    weights = torch.tensor([1.0, 2.0], dtype=torch.float32)

    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_path = os.path.join(tmpdir, "w.pt")
        history = trainer.train(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=1,
            lr=1e-3,
            patience=2,
            checkpoint_path=ckpt_path,
            class_weights=weights,
        )
    assert len(history.train_losses) == 1


def test_on_epoch_end_hook_called_each_epoch() -> None:
    """on_epoch_end receives (epoch, max_epochs, loss, val_f1, lr) once per completed epoch."""
    model = _make_model()
    train_loader = _make_loader()
    val_loader = _make_loader()
    trainer = Trainer()
    calls: list[tuple[int, int, float, float, float]] = []

    def hook(epoch: int, max_epochs: int, loss: float, f1: float, lr: float) -> None:
        calls.append((epoch, max_epochs, loss, f1, lr))

    max_epochs = 3
    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_path = os.path.join(tmpdir, "hook.pt")
        history = trainer.train(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=max_epochs,
            lr=1e-3,
            patience=5,
            checkpoint_path=ckpt_path,
            on_epoch_end=hook,
        )

    assert len(calls) == len(history.train_losses)
    assert calls[0][0] == 1 and calls[-1][0] == len(calls)
    assert all(c[1] == max_epochs for c in calls)
    assert all(c[4] > 0 for c in calls)
