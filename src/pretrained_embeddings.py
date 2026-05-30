"""Optional FastText-style .vec initialisation for embedding layers (no extra deps).

Maps a trained subword vocabulary to pretrained word vectors (typically
``wiki-news-300d.vec`` or Common Crawl ``cc.en.300.vec``). Unmatched pieces keep
default row values; use :func:`apply_fasttext_vec_file` after model construction.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch
import torch.nn as nn

if TYPE_CHECKING:
    from .tokeniser import SubwordTokeniser

logger = logging.getLogger(__name__)


def load_fasttext_vec(
    path: str | Path,
    *,
    limit_vectors: int | None = 800_000,
) -> tuple[dict[str, np.ndarray], int]:
    """Load a text ``.vec`` FastText / word2vec-format file (first line: vocab dim).

    Large files are streamed; at most *limit_vectors* rows are read after the
    header to cap memory (omit or set None to read the whole file — may be huge).

    Returns:
        (word_lowercase -> vector, embedding_dimension)
    """
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f".vec file not found: {p.resolve()}")

    table: dict[str, np.ndarray] = {}
    dim = 0
    with p.open(encoding="utf-8", errors="ignore") as f:
        header = f.readline().split()
        if len(header) != 2:
            raise ValueError(f"Unexpected .vec header in {p}: {header!r}")
        _n_claim, dim = int(header[0]), int(header[1])
        for i, line in enumerate(f):
            if limit_vectors is not None and i >= limit_vectors:
                break
            parts = line.rstrip().split()
            if len(parts) != dim + 1:
                continue
            token = parts[0]
            try:
                vec = np.asarray([float(x) for x in parts[1 : dim + 1]], dtype=np.float32)
            except ValueError:
                continue
            table[token.casefold()] = vec

    if not table or dim == 0:
        raise ValueError(f"No vectors parsed from {p}")
    return table, dim


def apply_pretrained_to_embedding(
    embedding: nn.Embedding,
    id_to_piece: dict[int, str],
    vec_table: dict[str, np.ndarray],
    vec_dim_table: int,
    *,
    embed_dim: int,
    freeze_pretrained: bool = False,
    skip_special_zero_ids: tuple[int, ...] = (0, 1),
) -> tuple[int, int]:
    """Copy pretrained rows into ``embedding.weight`` where strings match.

    For each vocab id ``i``, lookups are tried as ``piece.casefold()``,
    ``piece.strip().casefold()``, ``piece.strip().lstrip("▁").casefold()`` (sentencepiece).

    Rows with conflicting vector dimension use the first ``min(vec_dim_table, embed_dim)``
    coefficients; remaining columns keep module default init.

    Returns:
        (num_matched, vocab_size rows considered)
    """
    n_rows, d_emb = embedding.weight.shape
    if d_emb != embed_dim:
        raise ValueError(f"embedding dim {d_emb} != embed_dim argument {embed_dim}")
    copy_w = min(vec_dim_table, embed_dim)

    matched = 0
    considered = 0
    with torch.no_grad():
        for tid in range(n_rows):
            if tid in skip_special_zero_ids:
                continue
            piece = id_to_piece.get(tid, "")
            considered += 1
            variants = (
                piece.casefold(),
                piece.strip().casefold(),
                piece.strip().lstrip("▁").casefold(),
            )
            arr = None
            for key in variants:
                if key in vec_table:
                    arr = vec_table[key]
                    break
            if arr is None:
                continue
            row = embedding.weight[tid]
            row.zero_()
            row[:copy_w].copy_(torch.from_numpy(arr[:copy_w].copy()))
            matched += 1

    embedding.weight.requires_grad_(not freeze_pretrained)
    logger.info(
        "Pretrained init: matched %d / %d token rows (copy_width=%d, vec_dim=%d, embed_dim=%d, freeze=%s)",
        matched,
        considered,
        copy_w,
        vec_dim_table,
        embed_dim,
        freeze_pretrained,
    )
    return matched, considered


def apply_fasttext_vec_file(
    embedding: nn.Embedding,
    tokeniser: SubwordTokeniser,
    vec_path: str | Path,
    *,
    embed_dim: int,
    limit_vectors: int | None = 800_000,
    freeze_pretrained: bool = False,
) -> tuple[int, int]:
    """Convenience: load ``.vec`` and apply to *embedding* using *tokeniser* ids."""
    table, dim = load_fasttext_vec(vec_path, limit_vectors=limit_vectors)
    id_map = tokeniser.id_to_piece_map()
    return apply_pretrained_to_embedding(
        embedding,
        id_map,
        table,
        dim,
        embed_dim=embed_dim,
        freeze_pretrained=freeze_pretrained,
    )
