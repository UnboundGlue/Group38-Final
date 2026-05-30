"""Text preprocessing for the neural authorship attribution pipeline."""

from __future__ import annotations

import re


class Preprocessor:
    """Strip URLs and @mentions; collapse whitespace. Punctuation kept by default."""

    # Compiled patterns (class-level for efficiency)
    _URL_RE = re.compile(
        r"https?://\S+|www\.\S+",
        re.IGNORECASE,
    )
    _MENTION_RE = re.compile(r"@\w+")
    _WHITESPACE_RE = re.compile(r"\s+")
    _PUNCTUATION_RE = re.compile(r"[^\w\s]")

    def __init__(self, preserve_punctuation: bool = True) -> None:
        self.preserve_punctuation = preserve_punctuation

    def clean(self, text: str) -> str:
        """Return cleaned text, or empty string if nothing remains."""
        if not text:
            return ""

        # Remove URLs
        text = self._URL_RE.sub(" ", text)

        # Remove @mentions
        text = self._MENTION_RE.sub(" ", text)

        # Optionally strip punctuation
        if not self.preserve_punctuation:
            text = self._PUNCTUATION_RE.sub(" ", text)

        # Normalise whitespace (collapse multiple spaces/tabs/newlines)
        text = self._WHITESPACE_RE.sub(" ", text).strip()

        return text

    def batch_clean(self, texts: list[str]) -> list[str]:
        """Apply :meth:`clean` to each string; output list has the same length."""
        return [self.clean(t) for t in texts]
