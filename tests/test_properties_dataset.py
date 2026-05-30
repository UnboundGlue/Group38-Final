"""Property-based tests for DatasetLoader (Task 2.2, 2.3).

**Validates: Requirements 1.3**
"""

from __future__ import annotations

from hypothesis import given, settings
from hypothesis import strategies as st

from src.dataset import DatasetLoader
from src.models import Split


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

def _dataset_strategy(min_authors: int = 2, max_authors: int = 5,
                       min_samples: int = 10, max_samples: int = 30):
    """Generate (texts, labels, loader) where every author has >= min_samples."""
    return st.integers(min_value=min_authors, max_value=max_authors).flatmap(
        lambda n_authors: st.lists(
            st.integers(min_value=min_samples, max_value=max_samples),
            min_size=n_authors,
            max_size=n_authors,
        ).map(lambda counts: _build_dataset(n_authors, counts))
    )


def _build_dataset(n_authors: int, counts: list[int]):
    """Build (texts, labels, loader) from author count list."""
    texts: list[str] = []
    labels: list[int] = []
    for author_id, count in enumerate(counts):
        for sample_idx in range(count):
            texts.append(f"sample text for author {author_id} number {sample_idx}")
            labels.append(author_id)

    loader = DatasetLoader()
    # Populate author_map so split() can reference author names in errors
    loader.author_map = {i: f"author_{i}" for i in range(n_authors)}
    loader.num_authors = n_authors

    return texts, labels, loader


# ---------------------------------------------------------------------------
# Property 3: Stratified Split Coverage
# ---------------------------------------------------------------------------

@given(_dataset_strategy())
@settings(max_examples=20)
def test_stratified_split_coverage(dataset):
    """**Validates: Requirements 1.3**

    For any dataset where every author class has at least the minimum sample
    threshold (10), calling split() should produce three partitions such that
    every author class present in the full dataset appears in the train,
    validation, and test splits.
    """
    texts, labels, loader = dataset

    train, val, test = loader.split(texts, labels)

    all_author_ids = set(labels)

    assert set(train.labels) == all_author_ids, (
        f"Train split is missing authors: {all_author_ids - set(train.labels)}"
    )
    assert set(val.labels) == all_author_ids, (
        f"Val split is missing authors: {all_author_ids - set(val.labels)}"
    )
    assert set(test.labels) == all_author_ids, (
        f"Test split is missing authors: {all_author_ids - set(test.labels)}"
    )


# ---------------------------------------------------------------------------
# Property 4: Non-Overlapping Splits
# ---------------------------------------------------------------------------

@given(_dataset_strategy())
@settings(max_examples=20)
def test_non_overlapping_splits(dataset):
    """**Validates: Requirements 1.2**

    For any dataset, the train, validation, and test partitions produced by
    split() should be pairwise disjoint — no sample should appear in more than
    one partition. Also verifies that the total count of samples across all
    three splits equals the original dataset size.
    """
    texts, labels, loader = dataset

    train, val, test = loader.split(texts, labels)

    train_texts = set(train.texts)
    val_texts = set(val.texts)
    test_texts = set(test.texts)

    assert train_texts.isdisjoint(val_texts), (
        f"Train and val splits share samples: {train_texts & val_texts}"
    )
    assert train_texts.isdisjoint(test_texts), (
        f"Train and test splits share samples: {train_texts & test_texts}"
    )
    assert val_texts.isdisjoint(test_texts), (
        f"Val and test splits share samples: {val_texts & test_texts}"
    )

    total = len(train.texts) + len(val.texts) + len(test.texts)
    assert total == len(texts), (
        f"Total samples across splits ({total}) != original dataset size ({len(texts)})"
    )
