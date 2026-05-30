"""
Unit tests for BaselineFeatureExtractor (Task 6.4).
Covers Requirements 4.1–4.4.
"""
from __future__ import annotations

import pytest
from scipy.sparse import csr_matrix, issparse

from src.features import BaselineFeatureExtractor


# Small fixed corpus — three "authors", two samples each, distinguishable
# enough that all four feature methods produce a non-empty vocabulary.
TRAIN_TEXTS = [
    "the quick brown fox jumps over the lazy dog",
    "a quick brown fox is faster than a lazy dog",
    "she sells sea shells by the sea shore",
    "sea shells are sold by the seashore",
    "to be or not to be that is the question",
    "whether tis nobler in the mind to suffer",
]

UNSEEN_TEXTS = [
    "an entirely fresh sentence with novel vocabulary",
    "the dog and the fox",  # shares tokens with TRAIN_TEXTS
]


# ---------------------------------------------------------------------------
# Construction & validation
# ---------------------------------------------------------------------------


def test_invalid_method_raises():
    with pytest.raises(ValueError, match="method must be one of"):
        BaselineFeatureExtractor(method="banana")  # type: ignore[arg-type]


def test_default_ngram_per_method():
    assert BaselineFeatureExtractor("bow").ngram_range == (1, 1)
    assert BaselineFeatureExtractor("tfidf").ngram_range == (1, 1)
    assert BaselineFeatureExtractor("word").ngram_range == (1, 2)
    assert BaselineFeatureExtractor("char").ngram_range == (3, 5)


def test_explicit_ngram_overrides_default():
    fx = BaselineFeatureExtractor("char", ngram_range=(2, 4))
    assert fx.ngram_range == (2, 4)


def test_transform_before_fit_raises():
    fx = BaselineFeatureExtractor("tfidf")
    with pytest.raises(ValueError, match="before fit_transform"):
        fx.transform(["hello world"])


def test_vocabulary_size_before_fit_raises():
    fx = BaselineFeatureExtractor("bow")
    with pytest.raises(ValueError, match="not fitted"):
        _ = fx.vocabulary_size


# ---------------------------------------------------------------------------
# fit_transform — all four methods (Requirement 4.1, 4.2, 4.3)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("method", ["bow", "tfidf", "char", "word"])
def test_fit_transform_returns_sparse_matrix_with_correct_shape(method):
    fx = BaselineFeatureExtractor(method=method)
    X = fx.fit_transform(TRAIN_TEXTS)
    assert issparse(X), f"{method} did not return a sparse matrix"
    assert X.shape[0] == len(TRAIN_TEXTS)
    assert X.shape[1] > 0, f"{method} produced an empty vocabulary"
    assert fx.vocabulary_size == X.shape[1]


@pytest.mark.parametrize("method", ["bow", "tfidf", "char", "word"])
def test_fit_transform_then_transform_produces_same_columns(method):
    fx = BaselineFeatureExtractor(method=method)
    X_train = fx.fit_transform(TRAIN_TEXTS)
    X_unseen = fx.transform(UNSEEN_TEXTS)
    # transform must not refit — column count is fixed by the training vocab
    assert X_unseen.shape == (len(UNSEEN_TEXTS), X_train.shape[1])


def test_bow_produces_integer_counts():
    fx = BaselineFeatureExtractor("bow")
    X = fx.fit_transform(TRAIN_TEXTS)
    # CountVectorizer output dtype is integer
    assert X.dtype.kind in ("i", "u")


def test_tfidf_produces_float_values():
    fx = BaselineFeatureExtractor("tfidf")
    X = fx.fit_transform(TRAIN_TEXTS)
    assert X.dtype.kind == "f"
    # tf-idf values are non-negative
    assert X.min() >= 0.0


# ---------------------------------------------------------------------------
# n-gram range configurability (Requirement 4.3)
# ---------------------------------------------------------------------------


def test_word_ngram_range_affects_vocabulary_size():
    fx_unigram = BaselineFeatureExtractor("word", ngram_range=(1, 1))
    fx_bigram = BaselineFeatureExtractor("word", ngram_range=(1, 2))
    fx_unigram.fit_transform(TRAIN_TEXTS)
    fx_bigram.fit_transform(TRAIN_TEXTS)
    # Adding bigrams can only grow the vocabulary, never shrink it.
    assert fx_bigram.vocabulary_size > fx_unigram.vocabulary_size


def test_char_ngram_range_affects_vocabulary_size():
    fx_narrow = BaselineFeatureExtractor("char", ngram_range=(3, 3))
    fx_wide = BaselineFeatureExtractor("char", ngram_range=(3, 5))
    fx_narrow.fit_transform(TRAIN_TEXTS)
    fx_wide.fit_transform(TRAIN_TEXTS)
    assert fx_wide.vocabulary_size > fx_narrow.vocabulary_size


# ---------------------------------------------------------------------------
# transform on unseen texts (Requirement 4.2)
# ---------------------------------------------------------------------------


def test_transform_drops_unseen_tokens_silently():
    fx = BaselineFeatureExtractor("bow")
    fx.fit_transform(TRAIN_TEXTS)
    # Text containing only unseen tokens -> all-zero row, but still valid shape
    fully_unseen = ["zzz qqq xqx"]
    X = fx.transform(fully_unseen)
    assert X.shape == (1, fx.vocabulary_size)
    assert X.nnz == 0  # no non-zero entries — every token was OOV


def test_transform_preserves_seen_tokens():
    fx = BaselineFeatureExtractor("bow")
    fx.fit_transform(TRAIN_TEXTS)
    # "the dog" — both tokens appeared in training
    X = fx.transform(["the dog"])
    assert X.nnz >= 2


# ---------------------------------------------------------------------------
# max_features / min_df pass-through
# ---------------------------------------------------------------------------


def test_max_features_caps_vocabulary():
    fx = BaselineFeatureExtractor("tfidf", max_features=5)
    fx.fit_transform(TRAIN_TEXTS)
    assert fx.vocabulary_size <= 5


def test_min_df_filters_rare_terms():
    # min_df=2 drops every term appearing in only one document
    fx = BaselineFeatureExtractor("bow", min_df=2)
    fx.fit_transform(TRAIN_TEXTS)
    # Sanity: with min_df=1 we'd get a much larger vocab
    fx_loose = BaselineFeatureExtractor("bow", min_df=1)
    fx_loose.fit_transform(TRAIN_TEXTS)
    assert fx.vocabulary_size < fx_loose.vocabulary_size


# ---------------------------------------------------------------------------
# random_seed pass-through (it's stored for downstream use, not consumed
# by sklearn vectorizers, but the API contract is that it round-trips)
# ---------------------------------------------------------------------------


def test_random_seed_is_stored():
    fx = BaselineFeatureExtractor("tfidf", random_seed=42)
    assert fx.random_seed == 42
