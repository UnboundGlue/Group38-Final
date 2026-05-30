"""Subword tokeniser wrapping HuggingFace tokenizers library (BPE / WordPiece)."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer
from tokenizers.models import BPE, WordPiece
from tokenizers.trainers import BpeTrainer, WordPieceTrainer
from tokenizers.pre_tokenizers import Punctuation, Sequence, Whitespace

logger = logging.getLogger(__name__)

# Special token constants
_PAD = "[PAD]"
_UNK = "[UNK]"
_CLS = "[CLS]"
_SEP = "[SEP]"
_SPECIAL_TOKENS = [_PAD, _UNK, _CLS, _SEP]
_PAD_ID = 0
_UNK_ID = 1

_DEFAULT_PATH = "artifacts/tokeniser.json"


class SubwordTokeniser:
    """Subword tokeniser using BPE or WordPiece via HuggingFace tokenizers."""

    def __init__(self) -> None:
        self._tokenizer: Tokenizer | None = None

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(
        self,
        texts: list[str],
        vocab_size: int = 10_000,
        algorithm: str = "bpe",
        *,
        isolate_punctuation: bool = True,
    ) -> None:
        """Train tokeniser vocabulary on corpus.

        Args:
            texts: Non-empty list of training strings.
            vocab_size: Target vocabulary size. Must be > 256.
            algorithm: One of ``"bpe"`` or ``"wordpiece"``.
            isolate_punctuation: If True, split punctuation into separate pre-tokens before BPE.

        Raises:
            ValueError: If preconditions are violated.
        """
        if not texts:
            raise ValueError("texts must be non-empty")
        if vocab_size <= 256:
            raise ValueError("vocab_size must be > 256")
        if algorithm not in {"bpe", "wordpiece"}:
            raise ValueError(f"algorithm must be 'bpe' or 'wordpiece', got {algorithm!r}")

        if algorithm == "bpe":
            model = BPE(unk_token=_UNK)
            trainer = BpeTrainer(
                vocab_size=vocab_size,
                special_tokens=_SPECIAL_TOKENS,
                show_progress=False,
            )
        else:
            model = WordPiece(unk_token=_UNK)
            trainer = WordPieceTrainer(
                vocab_size=vocab_size,
                special_tokens=_SPECIAL_TOKENS,
                show_progress=False,
            )

        tokenizer = Tokenizer(model)
        if isolate_punctuation:
            tokenizer.pre_tokenizer = Sequence([Whitespace(), Punctuation()])
        else:
            tokenizer.pre_tokenizer = Whitespace()
        tokenizer.train_from_iterator(texts, trainer=trainer)
        self._tokenizer = tokenizer

    # ------------------------------------------------------------------
    # Encoding / decoding
    # ------------------------------------------------------------------

    def encode(self, text: str, max_length: int = 256) -> list[int]:
        """Encode *text* to a padded/truncated token ID sequence of length *max_length*.

        Unknown tokens are mapped to ``[UNK]`` (id=1).
        Sequences longer than *max_length* are right-truncated and a warning is logged.
        Sequences shorter than *max_length* are right-padded with ``[PAD]`` (id=0).

        Returns:
            List of exactly *max_length* integer token IDs.
        """
        self._require_trained()
        encoding = self._tokenizer.encode(text)  # type: ignore[union-attr]
        ids = encoding.ids

        if len(ids) > max_length:
            logger.warning(
                "Token sequence length %d exceeds max_length=%d; truncating.",
                len(ids),
                max_length,
            )
            ids = ids[:max_length]
        elif len(ids) < max_length:
            ids = ids + [_PAD_ID] * (max_length - len(ids))

        return ids

    def batch_encode(self, texts: list[str], max_length: int = 256) -> np.ndarray:
        """Encode a list of texts into a 2-D integer array.

        Returns:
            ``np.ndarray`` of shape ``[N, max_length]`` with dtype ``int64``.
        """
        self._require_trained()
        rows = [self.encode(text, max_length) for text in texts]
        return np.array(rows, dtype=np.int64)

    def decode(self, ids: list[int]) -> str:
        """Decode token IDs back to text, skipping ``[PAD]`` tokens.

        Returns:
            Reconstructed text string.
        """
        self._require_trained()
        # Filter out PAD tokens before decoding
        filtered = [i for i in ids if i != _PAD_ID]
        return self._tokenizer.decode(filtered)  # type: ignore[union-attr]

    # ------------------------------------------------------------------
    # Vocabulary info
    # ------------------------------------------------------------------

    def vocab_size(self) -> int:
        """Return the actual vocabulary size of the trained tokeniser."""
        self._require_trained()
        return self._tokenizer.get_vocab_size()  # type: ignore[union-attr]

    def id_to_piece_map(self) -> dict[int, str]:
        """Map each id ``0 .. vocab_size-1`` to its vocabulary piece string."""
        self._require_trained()
        n = self.vocab_size()
        out: dict[int, str] = {}
        for i in range(n):
            t = self._tokenizer.id_to_token(i)  # type: ignore[union-attr]
            out[i] = t if t is not None else ""
        return out

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str = _DEFAULT_PATH) -> None:
        """Persist the trained tokeniser to *path* (JSON format).

        The parent directory is created automatically if it does not exist.

        Args:
            path: Destination file path. Defaults to ``"artifacts/tokeniser.json"``.
        """
        self._require_trained()
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._tokenizer.save(path)  # type: ignore[union-attr]

    def load(self, path: str = _DEFAULT_PATH) -> None:
        """Load a previously saved tokeniser from *path* into this instance.

        Args:
            path: Source file path. Defaults to ``"artifacts/tokeniser.json"``.
        """
        self._tokenizer = Tokenizer.from_file(path)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _require_trained(self) -> None:
        if self._tokenizer is None:
            raise RuntimeError(
                "Tokeniser has not been trained yet. Call train() or load() first."
            )
