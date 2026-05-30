"""Core data models for the neural authorship attribution pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


# ---------------------------------------------------------------------------
# Dataset models
# ---------------------------------------------------------------------------

@dataclass
class AuthorSample:
    """A single labelled text sample from the authorship dataset."""
    text: str           # raw post/message text
    author_id: int      # integer label (0-indexed)
    author_name: str    # original author string
    source: str         # dataset split: "train" | "val" | "test"


@dataclass
class Split:
    """A dataset partition (train, val, or test)."""
    texts: list[str]
    labels: list[int]
    author_map: dict[int, str]   # id → name mapping


# ---------------------------------------------------------------------------
# Model / training configuration
# ---------------------------------------------------------------------------

@dataclass
class ModelConfig:
    """Hyperparameters for the CNN-LSTM model architecture."""
    vocab_size: int = 10_000
    embed_dim: int = 128
    num_filters: int = 128
    kernel_sizes: list[int] = field(default_factory=lambda: [2, 3, 4])
    lstm_hidden: int = 256
    lstm_layers: int = 2
    dropout: float = 0.5
    max_seq_len: int = 256
    num_classes: int = 2          # must be set to actual number of authors


@dataclass
class TrainingConfig:
    """Hyperparameters controlling the training loop."""
    epochs: int = 50
    batch_size: int = 64
    learning_rate: float = 1e-3
    patience: int = 5
    seed: int = 42
    device: str = "cpu"


# ---------------------------------------------------------------------------
# Metrics and training history
# ---------------------------------------------------------------------------

@dataclass
class MetricsDict:
    """Classification metrics produced by the Evaluator."""
    accuracy: float
    precision_macro: float
    recall_macro: float
    f1_macro: float
    f1_per_class: dict[int, float]          # per-author F1
    confusion_matrix: np.ndarray            # shape [C, C]


@dataclass
class TrainingHistory:
    """Record of per-epoch losses, validation metrics, and LR from a training run."""

    train_losses: list[float] = field(default_factory=list)
    val_metrics: list[MetricsDict] = field(default_factory=list)
    lr_per_epoch: list[float] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Explainability models
# ---------------------------------------------------------------------------

@dataclass
class ShapExplanation:
    """SHAP token-level attributions for a single text."""
    text: str
    token_ids: list[int]
    shap_values: list[float]


@dataclass
class LimeExplanation:
    """LIME local surrogate explanation for a single text."""
    text: str
    feature_weights: list[tuple[str, float]]


@dataclass
class ErrorAnalysisReport:
    """Aggregated misclassification analysis from the ExplainabilityModule."""
    misclassified_indices: list[int]
    top_tokens_per_pair: dict[tuple[int, int], list[str]]   # (true, pred) → tokens
    confusion_pairs: list[tuple[int, int, float]]           # (true, pred, rate)


# ---------------------------------------------------------------------------
# Custom exceptions
# ---------------------------------------------------------------------------

class InsufficientSamplesError(Exception):
    """Raised when an author class has fewer samples than the required minimum."""


class TrainingDivergenceError(Exception):
    """Raised when NaN loss is detected during training, indicating divergence."""
