"""
Property-based tests for BaselineFeatureExtractor (Tasks 6.2, 6.3).

Property 12: Baseline Feature Matrix Row Count
    For any non-empty corpus, fit_transform produces exactly N rows.
    Validates Requirements 4.1, 4.2.

Property 13: Baseline Reproducibility
    Two extractors built with identical config and fed identical input
    produce element-wise-identical sparse matrices.
    Validates Requirement 4.4.
"""
from __future__ import annotations

import numpy as np
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from src.features import BaselineFeatureExtractor

# A text strategy that's varied enough to exercise tokenisation but
# constrained enough to keep tests fast and avoid pathological inputs
# (e.g. empty strings, which sklearn rejects with a ValueError when
# the entire corpus is empty).
_word = st.text(
    alphabet=st.characters(whitelist_categories=("Ll", "Lu")),
    min_size=1,
    max_size=8,
)
_sentence = st.lists(_word, min_size=1, max_size=12).map(" ".join)
_corpus = st.lists(_sentence, min_size=2, max_size=20)

_method = st.sampled_from(["bow", "tfidf", "char", "word"])


# ---------------------------------------------------------------------------
# Property 12 — row count preservation
# ---------------------------------------------------------------------------


@given(corpus=_corpus, method=_method)
@settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
def test_fit_transform_row_count_equals_input_count(corpus, method):
    """fit_transform must return exactly len(corpus) rows."""
    fx = BaselineFeatureExtractor(method=method)
    try:
        X = fx.fit_transform(corpus)
    except ValueError:
        # sklearn raises if every document tokenises to nothing — that's
        # fine, the property only applies when fit_transform succeeds.
        return
    assert X.shape[0] == len(corpus)


@given(train=_corpus, unseen=_corpus, method=_method)
@settings(max_examples=30, suppress_health_check=[HealthCheck.too_slow])
def test_transform_row_count_equals_input_count(train, unseen, method):
    """transform on M unseen texts must return exactly M rows."""
    fx = BaselineFeatureExtractor(method=method)
    try:
        fx.fit_transform(train)
    except ValueError:
        return
    X = fx.transform(unseen)
    assert X.shape[0] == len(unseen)
    # And the column count is locked to the training vocabulary.
    assert X.shape[1] == fx.vocabulary_size


# ---------------------------------------------------------------------------
# Property 13 — reproducibility under identical config + input
# ---------------------------------------------------------------------------


@given(corpus=_corpus, method=_method)
@settings(max_examples=30, suppress_health_check=[HealthCheck.too_slow])
def test_fit_transform_is_deterministic(corpus, method):
    """Same config + same input => bitwise-identical sparse matrices."""
    fx_a = BaselineFeatureExtractor(method=method, random_seed=7)
    fx_b = BaselineFeatureExtractor(method=method, random_seed=7)
    try:
        X_a = fx_a.fit_transform(corpus)
        X_b = fx_b.fit_transform(corpus)
    except ValueError:
        return

    # Same shape, same vocabulary, same values.
    assert X_a.shape == X_b.shape
    assert fx_a.vectorizer_.vocabulary_ == fx_b.vectorizer_.vocabulary_

    # Compare densely — sparse matrices don't have a direct == that
    # returns a single boolean; converting to arrays is fine for the
    # small corpora hypothesis generates here.
    np.testing.assert_array_equal(X_a.toarray(), X_b.toarray())
