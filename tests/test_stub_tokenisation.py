"""Tests for stub character tokenisation (used with CNN-LSTM without Task 4)."""

from __future__ import annotations

import torch

from src.stub_tokenisation import stub_char_token_ids


def test_stub_shape_and_padding() -> None:
    out = stub_char_token_ids(["ab", "longer"], vocab_size=100, max_len=8)
    assert out.shape == (2, 8)
    assert out.dtype == torch.long
    assert (out[0, 2:] == 0).all()


def test_stub_respects_vocab() -> None:
    out = stub_char_token_ids(["x"], vocab_size=10, max_len=4)
    assert out.min() >= 0
    assert out.max() < 10
