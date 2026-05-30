"""Sklearn wrappers for BoW, TF-IDF, and character/word n-gram features."""

from __future__ import annotations

from typing import Iterable, Literal, Sequence

from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

Method = Literal["bow", "tfidf", "char", "word"]
_VALID_METHODS: tuple[Method, ...] = ("bow", "tfidf", "char", "word")

# Keys in metrics.json vs internal method names.
BASELINE_EXPERIMENT_METHODS: tuple[tuple[str, Method], ...] = (
    ("bow", "bow"),
    ("tfidf", "tfidf"),
    ("char_ngram", "char"),
    ("word_ngram", "word"),
)


class BaselineFeatureExtractor:
    """Fit a vectoriser on train text; transform train/val/test without refitting."""

    def __init__(
        self,
        method: Method = "tfidf",
        *,
        ngram_range: tuple[int, int] | None = None,
        max_features: int | None = None,
        min_df: int | float = 1,
        lowercase: bool = True,
        random_seed: int | None = None,
    ) -> None:
        if method not in _VALID_METHODS:
            raise ValueError(
                f"method must be one of {_VALID_METHODS}, got {method!r}"
            )
        self.method: Method = method
        self.ngram_range: tuple[int, int] = ngram_range or self._default_ngram(method)
        self.max_features = max_features
        self.min_df = min_df
        self.lowercase = lowercase
        self.random_seed = random_seed
        self.vectorizer_: CountVectorizer | TfidfVectorizer | None = None

    @staticmethod
    def _default_ngram(method: Method) -> tuple[int, int]:
        if method == "char":
            return (3, 5)
        if method == "word":
            return (1, 2)
        return (1, 1)  # bow, tfidf

    def _build_vectorizer(self) -> CountVectorizer | TfidfVectorizer:
        common = dict(
            ngram_range=self.ngram_range,
            max_features=self.max_features,
            min_df=self.min_df,
            lowercase=self.lowercase,
        )
        if self.method == "bow":
            return CountVectorizer(**common)
        if self.method == "tfidf":
            return TfidfVectorizer(**common)
        if self.method == "word":
            return TfidfVectorizer(analyzer="word", **common)
        # char uses char_wb (word-boundary n-grams), not straddling whitespace.
        return TfidfVectorizer(analyzer="char_wb", **common)

    def fit_transform(self, texts: Sequence[str] | Iterable[str]) -> csr_matrix:
        texts = list(texts)
        self.vectorizer_ = self._build_vectorizer()
        return self.vectorizer_.fit_transform(texts)

    def transform(self, texts: Sequence[str] | Iterable[str]) -> csr_matrix:
        if self.vectorizer_ is None:
            raise ValueError(
                "transform() called before fit_transform(); "
                "the extractor has no vocabulary yet."
            )
        return self.vectorizer_.transform(list(texts))

    @property
    def vocabulary_size(self) -> int:
        """Number of features in the fitted vocabulary."""
        if self.vectorizer_ is None:
            raise ValueError("Extractor is not fitted.")
        return len(self.vectorizer_.vocabulary_)
