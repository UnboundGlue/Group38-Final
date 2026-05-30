"""Hash characters to token ids for quick CNN smoke tests (not BPE)."""

from __future__ import annotations

import torch


def stub_char_token_ids(
    texts: list[str],
    *,
    vocab_size: int,
    max_len: int,
) -> torch.Tensor:
    """Map each char to an id in ``1..vocab_size-1``; ``0`` = padding."""
    if vocab_size < 2:
        raise ValueError("vocab_size must be at least 2 (reserve 0 for padding).")
    mod = vocab_size - 1
    rows: list[list[int]] = []
    for text in texts:
        ids: list[int] = []
        for c in text:
            if len(ids) >= max_len:
                break
            ids.append(1 + (ord(c) % mod))
        while len(ids) < max_len:
            ids.append(0)
        rows.append(ids[:max_len])
    return torch.tensor(rows, dtype=torch.long)
