"""Tests for FastText .vec initialisation helpers."""

from __future__ import annotations

from pathlib import Path
import tempfile

import numpy as np
import torch.nn as nn

from src.pretrained_embeddings import apply_pretrained_to_embedding, load_fasttext_vec


def test_load_fasttext_vec_tiny_file() -> None:
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "tiny.vec"
        p.write_text(
            "3 4\n"
            "hello 0.1 0.2 0.3 0.4\n"
            "world 1.0 0.0 -1.0 2.0\n"
            "##ing 0 0 0 1\n",
            encoding="utf-8",
        )
        table, dim = load_fasttext_vec(p, limit_vectors=10_000)
        assert dim == 4
        assert "hello" in table and "world" in table
        assert np.allclose(table["hello"], np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32))


def test_apply_pretrained_copies_subset_dim() -> None:
    emb = nn.Embedding(4, 3, padding_idx=0)
    emb.weight.data.uniform_(-1, 1)
    table = {"hi": np.array([1.0, 2.0, 3.0], dtype=np.float32)}
    m, n = apply_pretrained_to_embedding(
        emb,
        {0: "pad", 1: "unk", 2: "Hi", 3: "x"},
        table,
        vec_dim_table=3,
        embed_dim=3,
        skip_special_zero_ids=(0, 1),
    )
    assert n == 2
    assert m == 1
    assert emb.weight[2].tolist() == [1.0, 2.0, 3.0]
