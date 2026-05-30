"""Integration: Task 1 + 2 + 3 — load dataset, split, preprocess splits.

Run after `Preprocessor` (Task 3) is in place; validates the data path to cleaned text.
"""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

from src.dataset import DatasetLoader
from src.preprocessing import Preprocessor


def _write_csv(path: Path, rows: list[tuple[str, str]]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["text", "author"])
        w.writeheader()
        for t, a in rows:
            w.writerow({"text": t, "author": a})


def test_load_split_batch_clean_end_to_end(tmp_path: Path) -> None:
    # Three authors, 15 rows each (>= min_samples for split)
    rows: list[tuple[str, str]] = []
    for a in range(3):
        for i in range(15):
            rows.append(
                (
                    f"Post {i} with noise http://ex.com/a{a}  @user{ i % 2 }  and   spaces",
                    f"author_{a}",
                )
            )
    p = tmp_path / "toy.csv"
    _write_csv(p, rows)

    loader = DatasetLoader()
    texts, labels = loader.load(str(p))
    assert len(texts) == 45
    tr, va, te = loader.split(texts, labels, seed=42)

    pre = Preprocessor()
    for split_name, sp in (("train", tr), ("val", va), ("test", te)):
        cleaned = pre.batch_clean(sp.texts)
        assert len(cleaned) == len(sp.texts), split_name
        for s in cleaned:
            assert "http://" not in s, split_name
            assert "@user" not in s, split_name
        # batch_clean is element-wise clean
        for raw, c in zip(sp.texts, cleaned):
            assert c == pre.clean(raw), split_name

    # Training split still has all author ids (stratified)
    assert set(tr.labels) == set(range(3))
