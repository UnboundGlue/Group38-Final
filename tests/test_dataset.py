"""Unit tests for DatasetLoader (Tasks 2.1, 2.4).

Maps to **Requirement 1** (Dataset Loading and Splitting) acceptance criteria:

| Req   | Topic                                  | Tests (representative) |
|-------|----------------------------------------|-------------------------|
| 1.1   | Load CSV/JSON; 0-indexed author labels | `TestLoadCSV`, `TestLoadJSON`, `TestLoadRowOrderAndMapping` |
| 1.2   | Non-overlapping train/val/test         | `TestSplit::test_no_overlap_between_splits`, `test_all_samples_accounted_for` |
| 1.3   | Stratified: every author in each split | `TestSplit::test_all_authors_in_each_split` |
| 1.4   | `InsufficientSamplesError` + author id | `TestSplit::test_insufficient_samples_*` |
| 1.5   | `num_authors`, `samples_per_author`    | `TestLoadCSV::test_num_authors_set`, `test_samples_per_author_set` |
| —     | Chanchal CSV (`Text`/`Author`, `b'…'`) | `TestLoadChanchalGitHubCSV` |

Split **ratio** checks: `test_approximate_train_ratio`, `test_val_test_ratio_bounds`.
"""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path

import pytest

from src.dataset import (
    DatasetLoader,
    DEFAULT_CHANCHAL_200_CSV,
    DEFAULT_CHANCHAL_CSV,
    resolve_evaluation_dataset_path,
)
from src.models import InsufficientSamplesError, Split


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_csv(path: str, rows: list[tuple[str, str]]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["text", "author"])
        writer.writeheader()
        for text, author in rows:
            writer.writerow({"text": text, "author": author})


def _make_json(path: str, rows: list[tuple[str, str]]) -> None:
    data = [{"text": t, "author": a} for t, a in rows]
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh)


def _make_json_chanchal_keys(path: str, rows: list[tuple[str, str]]) -> None:
    data = [{"Text": t, "Author": a} for t, a in rows]
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh)


def _synthetic_rows(n_authors: int = 3, samples_per: int = 15) -> list[tuple[str, str]]:
    rows = []
    for a in range(n_authors):
        for s in range(samples_per):
            rows.append((f"text from author {a} sample {s}", f"author_{a}"))
    return rows


# ---------------------------------------------------------------------------
# load() — CSV
# ---------------------------------------------------------------------------

def _make_chanchal_style_csv(path: str, rows: list[tuple[str, str]]) -> None:
    """CSV with Text/Author headers like chanchalIITP/AuthorIdentification."""
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["Text", "Author"])
        writer.writeheader()
        for text, author in rows:
            writer.writerow({"Text": text, "Author": author})


class TestLoadChanchalGitHubCSV:
    """Regression: public dataset from chanchalIITP/AuthorIdentification uses Text/Author."""

    def test_text_author_column_headers(self, tmp_path):
        p = str(tmp_path / "sample.csv")
        big_rows = []
        for i in range(10):
            big_rows.append((f"post {i}", "100"))
        for i in range(10):
            big_rows.append((f"other {i}", "200"))
        _make_chanchal_style_csv(p, big_rows)
        loader = DatasetLoader()
        texts, labels = loader.load(p)
        assert len(texts) == 20
        assert set(labels) == {0, 1}
        # Deterministic: sorted author ids "100" < "200" -> 100 -> 0, 200 -> 1
        assert loader.author_map[0] == "100"
        assert loader.author_map[1] == "200"

    def test_byte_string_literal_text_cells(self, tmp_path):
        """Tweets in the published CSVs are often stored as b'...' string literals."""
        cell = "b'Short test tweet. http://x.test/path'"
        p = str(tmp_path / "bytes.csv")
        _make_chanchal_style_csv(
            p,
            [(cell, str(aid)) for aid in (1, 2, 3) for _ in range(12)],
        )
        loader = DatasetLoader()
        texts, _ = loader.load(p)
        assert all(not t.startswith("b'") for t in texts)
        assert "Short test tweet" in texts[0]
        assert "http://x.test/path" in texts[0]


class TestLoadPathResolution:
    def test_missing_file_raises_file_not_found(self) -> None:
        loader = DatasetLoader()
        with pytest.raises(FileNotFoundError, match="not found"):
            loader.load("data/nonexistent_data_file_12345.csv", fetch_if_missing=False)


class TestLoadCSV:
    def test_returns_texts_and_labels(self, tmp_path):
        rows = _synthetic_rows(3, 5)
        p = str(tmp_path / "data.csv")
        _make_csv(p, rows)
        loader = DatasetLoader()
        texts, labels = loader.load(p)
        assert len(texts) == len(rows)
        assert len(labels) == len(rows)

    def test_labels_are_zero_indexed(self, tmp_path):
        rows = _synthetic_rows(4, 5)
        p = str(tmp_path / "data.csv")
        _make_csv(p, rows)
        loader = DatasetLoader()
        _, labels = loader.load(p)
        assert set(labels) == {0, 1, 2, 3}

    def test_num_authors_set(self, tmp_path):
        rows = _synthetic_rows(5, 5)
        p = str(tmp_path / "data.csv")
        _make_csv(p, rows)
        loader = DatasetLoader()
        loader.load(p)
        assert loader.num_authors == 5

    def test_samples_per_author_set(self, tmp_path):
        rows = _synthetic_rows(3, 7)
        p = str(tmp_path / "data.csv")
        _make_csv(p, rows)
        loader = DatasetLoader()
        loader.load(p)
        assert all(v == 7 for v in loader.samples_per_author.values())

    def test_two_column_numeric_headers_text_then_author(self, tmp_path) -> None:
        """Some exports use ``0`` / ``1`` as column names (text, author order)."""
        p = str(tmp_path / "idx.csv")
        with open(p, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=["0", "1"])
            w.writeheader()
            for i in range(12):
                w.writerow({"0": f"tweet sample {i} text", "1": str(100 + (i % 2))})
        loader = DatasetLoader()
        texts, labels = loader.load(p)
        assert len(texts) == 12
        assert "tweet sample" in texts[0]
        assert set(labels) == {0, 1}

    def test_author_map_populated(self, tmp_path):
        rows = _synthetic_rows(3, 5)
        p = str(tmp_path / "data.csv")
        _make_csv(p, rows)
        loader = DatasetLoader()
        loader.load(p)
        assert len(loader.author_map) == 3
        assert all(isinstance(k, int) for k in loader.author_map)
        assert all(isinstance(v, str) for v in loader.author_map.values())


# ---------------------------------------------------------------------------
# load() — JSON
# ---------------------------------------------------------------------------

class TestLoadRowOrderAndMapping:
    """Requirement 1.1: row order preserved; deterministic label mapping (sorted author names)."""

    def test_row_order_preserved_and_labels_deterministic(self, tmp_path):
        rows = [("a", "zebra"), ("b", "apple"), ("c", "zebra")]
        p = str(tmp_path / "order.csv")
        _make_csv(p, rows)
        loader = DatasetLoader()
        texts, labels = loader.load(p)
        assert len(texts) == len(labels) == 3
        assert texts == ["a", "b", "c"]
        # Sorted author names: apple -> 0, zebra -> 1
        assert (texts[0], labels[0]) == ("a", 1)
        assert (texts[1], labels[1]) == ("b", 0)
        assert (texts[2], labels[2]) == ("c", 1)

    def test_json_chanchal_style_keys(self, tmp_path):
        rows = _synthetic_rows(2, 12)
        p = str(tmp_path / "chanchal.json")
        _make_json_chanchal_keys(p, rows)
        loader = DatasetLoader()
        texts, labels = loader.load(p)
        assert len(texts) == len(rows)
        assert set(labels) == {0, 1}


class TestLoadJSON:
    def test_json_returns_same_count(self, tmp_path):
        rows = _synthetic_rows(2, 10)
        p = str(tmp_path / "data.json")
        _make_json(p, rows)
        loader = DatasetLoader()
        texts, labels = loader.load(p)
        assert len(texts) == len(rows)
        assert len(labels) == len(rows)

    def test_json_labels_zero_indexed(self, tmp_path):
        rows = _synthetic_rows(3, 5)
        p = str(tmp_path / "data.json")
        _make_json(p, rows)
        loader = DatasetLoader()
        _, labels = loader.load(p)
        assert set(labels) == {0, 1, 2}


# ---------------------------------------------------------------------------
# split()
# ---------------------------------------------------------------------------

class TestSplit:
    def _loader_with_data(self, tmp_path, n_authors=3, samples_per=15):
        rows = _synthetic_rows(n_authors, samples_per)
        p = str(tmp_path / "data.csv")
        _make_csv(p, rows)
        loader = DatasetLoader()
        texts, labels = loader.load(p)
        return loader, texts, labels

    def test_returns_three_splits(self, tmp_path):
        loader, texts, labels = self._loader_with_data(tmp_path)
        result = loader.split(texts, labels)
        assert len(result) == 3

    def test_splits_are_split_objects(self, tmp_path):
        loader, texts, labels = self._loader_with_data(tmp_path)
        train, val, test = loader.split(texts, labels)
        for s in (train, val, test):
            assert isinstance(s, Split)

    def test_no_overlap_between_splits(self, tmp_path):
        loader, texts, labels = self._loader_with_data(tmp_path, n_authors=3, samples_per=20)
        train, val, test = loader.split(texts, labels)
        train_set = set(train.texts)
        val_set = set(val.texts)
        test_set = set(test.texts)
        assert train_set.isdisjoint(val_set)
        assert train_set.isdisjoint(test_set)
        assert val_set.isdisjoint(test_set)

    def test_all_samples_accounted_for(self, tmp_path):
        loader, texts, labels = self._loader_with_data(tmp_path, n_authors=3, samples_per=20)
        train, val, test = loader.split(texts, labels)
        total = len(train.texts) + len(val.texts) + len(test.texts)
        assert total == len(texts)

    def test_all_authors_in_each_split(self, tmp_path):
        loader, texts, labels = self._loader_with_data(tmp_path, n_authors=3, samples_per=20)
        train, val, test = loader.split(texts, labels)
        all_ids = set(labels)
        assert set(train.labels) == all_ids
        assert set(val.labels) == all_ids
        assert set(test.labels) == all_ids

    def test_author_map_propagated(self, tmp_path):
        loader, texts, labels = self._loader_with_data(tmp_path)
        train, val, test = loader.split(texts, labels)
        for s in (train, val, test):
            assert s.author_map == loader.author_map

    def test_approximate_train_ratio(self, tmp_path):
        loader, texts, labels = self._loader_with_data(tmp_path, n_authors=3, samples_per=30)
        train, val, test = loader.split(texts, labels, train_ratio=0.7, val_ratio=0.15)
        n = len(texts)
        # Allow ±5% tolerance due to stratification rounding
        assert abs(len(train.texts) / n - 0.7) < 0.05

    def test_val_test_ratio_bounds(self, tmp_path):
        """Task 2.4: split ratios — val and test each near 15% of the full set (defaults)."""
        loader, texts, labels = self._loader_with_data(tmp_path, n_authors=3, samples_per=40)
        train, val, test = loader.split(texts, labels, train_ratio=0.7, val_ratio=0.15)
        n = len(texts)
        for name, sp, target in (
            ("val", val, 0.15),
            ("test", test, 0.15),
        ):
            assert abs(len(sp.texts) / n - target) < 0.05, (
                f"{name} fraction {len(sp.texts)/n:.3f} not near {target}"
            )

    def test_split_reproducible_with_same_seed(self, tmp_path):
        """Same seed → identical partition (train set as frozenset of texts)."""
        loader, texts, labels = self._loader_with_data(tmp_path, n_authors=3, samples_per=25)
        t1, v1, s1 = loader.split(texts, labels, seed=12345)
        t2, v2, s2 = loader.split(texts, labels, seed=12345)
        assert set(t1.texts) == set(t2.texts)
        assert set(v1.texts) == set(v2.texts)
        assert set(s1.texts) == set(s2.texts)
        t3, _, _ = loader.split(texts, labels, seed=99999)
        assert set(t1.texts) != set(t3.texts)

    def test_insufficient_samples_raises(self, tmp_path):
        # author_0 has 5 samples, below default threshold of 10
        rows = (
            [(f"text {i}", "author_0") for i in range(5)]
            + [(f"text {i}", "author_1") for i in range(15)]
            + [(f"text {i}", "author_2") for i in range(15)]
        )
        p = str(tmp_path / "data.csv")
        _make_csv(p, rows)
        loader = DatasetLoader()
        texts, labels = loader.load(p)
        with pytest.raises(InsufficientSamplesError):
            loader.split(texts, labels)

    def test_insufficient_samples_error_names_author(self, tmp_path):
        rows = (
            [(f"text {i}", "rare_author") for i in range(3)]
            + [(f"text {i}", "common_author") for i in range(20)]
        )
        p = str(tmp_path / "data.csv")
        _make_csv(p, rows)
        loader = DatasetLoader()
        texts, labels = loader.load(p)
        with pytest.raises(InsufficientSamplesError, match="rare_author"):
            loader.split(texts, labels)

    def test_custom_min_samples_threshold(self, tmp_path):
        # 8 samples per author — passes with min_samples=5, fails with default 10
        rows = _synthetic_rows(3, 8)
        p = str(tmp_path / "data.csv")
        _make_csv(p, rows)
        loader = DatasetLoader()
        texts, labels = loader.load(p)
        # Should not raise with lower threshold
        train, val, test = loader.split(texts, labels, min_samples=5)
        assert len(train.texts) > 0

    def test_stratified_kfold_partitions_and_covers_folds(self, tmp_path):
        """3-fold: each full sample in exactly one test fold; train+val+test = full set."""
        loader, texts, labels = self._loader_with_data(tmp_path, n_authors=3, samples_per=30)
        folds = list(loader.iter_stratified_kfold(texts, labels, n_splits=3, seed=0))
        assert len(folds) == 3
        seen_test: set[str] = set()
        n = len(texts)
        for tr, va, te, fi in folds:
            assert 0 <= fi < 3
            assert len(tr.texts) + len(va.texts) + len(te.texts) == n
            seen_test.update(te.texts)
        assert len(seen_test) == n


def test_resolve_evaluation_dataset_path_explicit_wins_then_preset_then_bundle_then_default(tmp_path):
    custom = tmp_path / "mine.csv"
    custom.write_text("x")
    explicit_res = resolve_evaluation_dataset_path(
        explicit=str(custom), preset="chanchal_50", training_cli_dataset="ignored.csv"
    )
    assert Path(explicit_res) == Path(str(custom)).expanduser()
    got50 = resolve_evaluation_dataset_path(explicit=None, preset="chanchal_50", training_cli_dataset="ignored.csv")
    assert got50 == DEFAULT_CHANCHAL_CSV
    got200p = resolve_evaluation_dataset_path(explicit=None, preset="chanchal_200", training_cli_dataset=None)
    assert got200p == DEFAULT_CHANCHAL_200_CSV
    from_bundle = tmp_path / "from_bundle.csv"
    from_bundle.write_text("")
    bundled = resolve_evaluation_dataset_path(explicit=None, preset=None, training_cli_dataset=str(from_bundle))
    assert Path(bundled) == Path(str(from_bundle)).expanduser()
    default_only = resolve_evaluation_dataset_path(explicit=None, preset=None, training_cli_dataset=None)
    assert default_only == DEFAULT_CHANCHAL_200_CSV

