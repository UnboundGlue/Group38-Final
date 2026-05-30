"""Unit tests for CNNLSTMModel (Task 7.3; synthetic token IDs only).

Tests output shape, NaN-free outputs, ModelConfig acceptance, and
different num_classes values.

Validates: Requirements 5.1, 5.5
"""

from __future__ import annotations

import pytest
import torch

from src.model import CNNLSTMModel
from src.models import ModelConfig

# Small config for fast tests
_SMALL_CONFIG = dict(
    vocab_size=100,
    embed_dim=16,
    num_filters=8,
    kernel_sizes=[2, 3, 4],
    lstm_hidden=16,
    lstm_layers=1,
    dropout=0.0,
    max_seq_len=256,
    num_classes=5,
)

_MIN_SEQ_LEN = max(_SMALL_CONFIG["kernel_sizes"])  # 4


def _make_model(num_classes: int = 5) -> CNNLSTMModel:
    config = ModelConfig(**{**_SMALL_CONFIG, "num_classes": num_classes})
    model = CNNLSTMModel(config)
    model.eval()
    return model


def _random_input(batch_size: int, seq_len: int, vocab_size: int = 100) -> torch.Tensor:
    return torch.randint(low=0, high=vocab_size, size=(batch_size, seq_len))


# ---------------------------------------------------------------------------
# Output shape tests — various batch sizes
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("batch_size", [1, 4, 16])
def test_output_shape_batch_sizes(batch_size: int) -> None:
    """Output shape is [B, num_classes] for batch sizes 1, 4, 16."""
    model = _make_model()
    token_ids = _random_input(batch_size, seq_len=32)
    with torch.no_grad():
        out = model(token_ids)
    assert out.shape == (batch_size, _SMALL_CONFIG["num_classes"])


# ---------------------------------------------------------------------------
# Output shape tests — various sequence lengths
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("seq_len", [4, 32, 128])
def test_output_shape_seq_lengths(seq_len: int) -> None:
    """Output shape is [B, num_classes] for sequence lengths 4, 32, 128."""
    model = _make_model()
    token_ids = _random_input(batch_size=2, seq_len=seq_len)
    with torch.no_grad():
        out = model(token_ids)
    assert out.shape == (2, _SMALL_CONFIG["num_classes"])


# ---------------------------------------------------------------------------
# No NaN in output
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("batch_size,seq_len", [(1, 4), (4, 32), (16, 128)])
def test_no_nan_in_output(batch_size: int, seq_len: int) -> None:
    """Output contains no NaN values for random inputs."""
    model = _make_model()
    token_ids = _random_input(batch_size, seq_len)
    with torch.no_grad():
        out = model(token_ids)
    assert not torch.isnan(out).any(), "Output contains NaN values"
    assert not torch.isinf(out).any(), "Output contains Inf values"


# ---------------------------------------------------------------------------
# ModelConfig acceptance
# ---------------------------------------------------------------------------

def test_model_accepts_model_config() -> None:
    """CNNLSTMModel accepts a ModelConfig and produces correct output shape."""
    config = ModelConfig(**_SMALL_CONFIG)
    model = CNNLSTMModel(config)
    model.eval()
    token_ids = _random_input(batch_size=3, seq_len=16)
    with torch.no_grad():
        out = model(token_ids)
    assert out.shape == (3, config.num_classes)


def test_model_stores_config() -> None:
    """CNNLSTMModel stores the provided ModelConfig."""
    config = ModelConfig(**_SMALL_CONFIG)
    model = CNNLSTMModel(config)
    assert model.config is config


def test_encode_document_vector_shape() -> None:
    """encode() returns [B, lstm_hidden] for reuse as a general representation."""
    model = _make_model()
    token_ids = _random_input(batch_size=3, seq_len=16)
    with torch.no_grad():
        z = model.encode(token_ids)
    assert z.shape == (3, _SMALL_CONFIG["lstm_hidden"])


# ---------------------------------------------------------------------------
# Different num_classes values
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("num_classes", [2, 5, 10, 20])
def test_different_num_classes(num_classes: int) -> None:
    """Output last dimension matches num_classes for various values."""
    model = _make_model(num_classes=num_classes)
    token_ids = _random_input(batch_size=4, seq_len=16)
    with torch.no_grad():
        out = model(token_ids)
    assert out.shape == (4, num_classes)
