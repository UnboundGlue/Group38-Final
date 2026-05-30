"""Property-based tests for the CNN-LSTM model (Task 7.2).

**Validates: Requirements 5.1, 5.5**
"""

from __future__ import annotations

import torch
from hypothesis import given, settings
from hypothesis import strategies as st

from src.model import CNNLSTMModel
from src.models import ModelConfig

# Fixed small config used across all tests
_KERNEL_SIZES = [2, 3, 4]
_MIN_SEQ_LEN = max(_KERNEL_SIZES)  # 4

_BASE_CONFIG_KWARGS = dict(
    embed_dim=16,
    num_filters=8,
    kernel_sizes=_KERNEL_SIZES,
    lstm_hidden=16,
    lstm_layers=1,
    dropout=0.0,
    vocab_size=100,
)


@given(
    batch_size=st.integers(min_value=1, max_value=8),
    seq_len=st.integers(min_value=_MIN_SEQ_LEN, max_value=64),
    num_classes=st.integers(min_value=2, max_value=10),
)
@settings(max_examples=50)
def test_cnn_lstm_output_shape_invariant(
    batch_size: int,
    seq_len: int,
    num_classes: int,
) -> None:
    """Property 5: CNN-LSTM Output Shape Invariant.

    For any batch of token ID tensors of shape [B, T] where B >= 1 and
    T >= max(kernel_sizes), model.forward(token_ids) should return a tensor
    of shape [B, num_classes] with no NaN or Inf values.

    **Validates: Requirements 5.1, 5.5**
    """
    config = ModelConfig(num_classes=num_classes, **_BASE_CONFIG_KWARGS)
    model = CNNLSTMModel(config)
    model.eval()

    token_ids = torch.randint(
        low=0,
        high=config.vocab_size,
        size=(batch_size, seq_len),
    )

    with torch.no_grad():
        output = model.forward(token_ids)

    # Property 5.1: output shape must be [B, num_classes]
    assert output.shape == (batch_size, num_classes), (
        f"Expected shape ({batch_size}, {num_classes}), got {tuple(output.shape)}"
    )

    # Property 5.5: no NaN or Inf values in output
    assert not torch.isnan(output).any(), "Output contains NaN values"
    assert not torch.isinf(output).any(), "Output contains Inf values"
