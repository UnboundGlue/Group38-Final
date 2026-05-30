"""DatasetLoader: load and split social media authorship datasets.

Also provides `EVAL_DATASET_PRESETS`, :func:`resolve_evaluation_dataset_path`
(shortcuts used when evaluating saved CNN bundles with ``--preset-dataset``), and
:func:`ensure_authoridentification_dataset` to shallow-clone the
Chanchal et al. ``AuthorIdentification`` GitHub release when CSVs are not yet
local, plus :func:`authoridentification_clone_path` / ``DEFAULT_CHANCHAL_CSV``
for default locations.
"""

from __future__ import annotations

import ast
import csv
import json
import logging
import shutil
import subprocess
from collections import Counter
from collections.abc import Iterator
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit

from src.models import InsufficientSamplesError, Split

logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# Public Chanchal et al. dataset (GitHub) — clone location & convenience paths
# -----------------------------------------------------------------------------

AUTHORIDENTIFICATION_GIT_URL = (
    "https://github.com/chanchalIITP/AuthorIdentification.git"
)
AUTHORIDENTIFICATION_DIRNAME = "AuthorIdentification"

# Relative to project root, after :func:`ensure_authoridentification_dataset` has run
DEFAULT_CHANCHAL_CSV = (
    "data/AuthorIdentification/Dataset/"
    "Dataset_with_varying_number_of_tweets/50_tweets_per_user.csv"
)
# More text per author (same release); often needed for better neural scores.
DEFAULT_CHANCHAL_200_CSV = (
    "data/AuthorIdentification/Dataset/"
    "Dataset_with_varying_number_of_tweets/200_tweets_per_user.csv"
)

# Convenience keys for evaluating a saved model without typing long paths (--preset-dataset).
EVAL_DATASET_PRESETS: dict[str, str] = {
    "chanchal_50": DEFAULT_CHANCHAL_CSV,
    "chanchal_200": DEFAULT_CHANCHAL_200_CSV,
}


def resolve_evaluation_dataset_path(
    *,
    explicit: str | None,
    preset: str | None,
    training_cli_dataset: object | None,
) -> str:
    """Pick a CSV/JSON path when loading a saved CNN-LSTM bundle for evaluation.

    Precedence:

    #. Non-empty ``explicit`` (CLI ``--dataset``).
    #. ``preset`` in :data:`EVAL_DATASET_PRESETS` (CLI ``--preset-dataset``).
    #. Non-empty ``training_cli_dataset`` from bundled ``training.json`` ``cli_args``.
    #. :data:`DEFAULT_CHANCHAL_200_CSV`.

    Returned paths may be repo-relative strings (same style as defaults).
    """
    if explicit is not None and str(explicit).strip():
        return str(Path(explicit).expanduser())
    if preset is not None:
        if preset not in EVAL_DATASET_PRESETS:
            raise ValueError(f"unknown preset: {preset!r}")
        return EVAL_DATASET_PRESETS[preset]
    if training_cli_dataset is not None:
        s = str(training_cli_dataset).strip()
        if s:
            return str(Path(s).expanduser())
    return DEFAULT_CHANCHAL_200_CSV


def _project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def authoridentification_clone_path() -> Path:
    """Directory where the AuthorIdentification repository is shallow-cloned."""
    return _project_root() / "data" / AUTHORIDENTIFICATION_DIRNAME


def _has_authoridentification_csvs(clone_root: Path) -> bool:
    dataset_root = clone_root / "Dataset"
    if not dataset_root.is_dir():
        return False
    return any(dataset_root.rglob("*.csv"))


def _run_git(*args: str, cwd: Path | None = None) -> None:
    try:
        subprocess.run(
            ["git", *args], check=True, cwd=cwd, capture_output=True, text=True
        )
    except FileNotFoundError as e:
        raise RuntimeError(
            "git is not on PATH. Install Git (https://git-scm.com/) or clone manually, e.g.:\n"
            f"  git clone {AUTHORIDENTIFICATION_GIT_URL} data/{AUTHORIDENTIFICATION_DIRNAME}"
        ) from e
    except subprocess.CalledProcessError as e:
        err = (e.stderr or e.stdout or "").strip()
        raise RuntimeError(f"git failed ({' '.join(e.cmd)}): {err}") from e


def ensure_authoridentification_dataset(
    *,
    clone_root: Path | str | None = None,
    force: bool = False,
) -> Path:
    """Ensure CSVs from chanchalIITP/AuthorIdentification are available locally.

    If ``{clone_root}/Dataset/**/*.csv`` is missing, runs a shallow ``git clone``
    (or ``git pull`` in an existing clone). Requires **git** on ``PATH``.

    Parameters
    ----------
    clone_root:
        Where to place the repository (default: ``data/AuthorIdentification``
        under the project root).
    force:
        If *clone_root* exists but is not a valid clone, remove it and clone
        again.

    Returns
    -------
    Path
        The directory containing the ``Dataset/`` folder (the clone root).

    Raises
    ------
    RuntimeError
        If *git* is missing, clone/pull fails, or no CSVs appear under
        ``Dataset/`` after the operation.
    """
    target = Path(clone_root) if clone_root is not None else authoridentification_clone_path()
    target = target.resolve()

    if _has_authoridentification_csvs(target):
        logger.debug("AuthorIdentification dataset already present at %s", target)
        return target

    if target.exists() and (target / ".git").is_dir():
        logger.info("Updating existing AuthorIdentification clone: %s", target)
        _run_git("pull", "--ff-only", cwd=target)
        if _has_authoridentification_csvs(target):
            return target
        raise RuntimeError(
            f"git pull in {target} did not provide CSVs under {target / 'Dataset'}. "
            "Remove the folder or re-run with force=True."
        )

    if target.exists():
        if not force:
            raise RuntimeError(
                f"Path {target} exists but is not a valid AuthorIdentification clone. "
                "Delete it or call ensure_authoridentification_dataset(..., force=True)."
            )
        logger.warning("Removing %s (--force)", target)
        shutil.rmtree(target)

    target.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Cloning %s into %s ...", AUTHORIDENTIFICATION_GIT_URL, target)
    _run_git("clone", "--depth", "1", AUTHORIDENTIFICATION_GIT_URL, str(target))

    if not _has_authoridentification_csvs(target):
        raise RuntimeError(
            f"Clone completed but no CSVs found under {target / 'Dataset'}"
        )
    return target


def _resolve_dataset_file(path: str) -> str | None:
    """Return an absolute path to *path* if it exists, checking CWD and project root."""
    p = Path(path)
    if p.is_file():
        return str(p.resolve())
    rooted = _project_root() / path
    if rooted.is_file():
        return str(rooted.resolve())
    return None


def _normalize_csv_field_names(fieldnames: list[str] | None) -> dict[str, str]:
    """Map lowercase name -> actual column name in the header row."""
    if not fieldnames:
        return {}
    return {name.strip().lower(): name for name in fieldnames if name is not None}


def _get_text_author_columns(row: dict[str, str], field_map: dict[str, str]) -> tuple[str, str]:
    """Resolve (text, author) values using flexible column names.

    Supports ``text`` / ``Text`` and ``author`` / ``Author`` as used in
    [chanchalIITP/AuthorIdentification](https://github.com/chanchalIITP/AuthorIdentification)
    CSV exports.
    """
    def col(*candidates: str) -> str | None:
        for c in candidates:
            key = field_map.get(c.lower())
            if key is not None and key in row:
                return row[key]
        return None

    text_v = col("text", "tweet", "message", "content")
    auth_v = col("author", "user", "user_id", "label")
    if text_v is None or auth_v is None:
        # Some exports use two columns with no name, or numeric headers "0" / "1"
        # (e.g. headerless data read as first row, or pandas-style index export).
        keys = [k for k in row if k is not None]
        if len(keys) == 2 and text_v is None and auth_v is None:
            k0, k1 = keys[0], keys[1]
            text_v, auth_v = row[k0], row[k1]
        else:
            raise ValueError(
                "CSV must contain text and author columns (e.g. text/author or Text/Author), "
                "or exactly two columns in order: text, then author. "
                f"Found columns: {list(row.keys())}"
            )
    return text_v, auth_v


def _coerce_text_cell(raw: str) -> str:
    """Turn a CSV text cell into a normal string.

    The public Twitter CSVs store tweets as Python byte-string literals
    (``b'...'``). Newlines and quotes may appear inside fields; we try
    ``ast.literal_eval`` for well-formed literals, then fall back to stripping
    the ``b'…'`` wrapper.
    """
    s = raw.strip()
    if not s:
        return s

    # Strip one layer of CSV-style double-quoting if the whole cell is quoted
    if len(s) >= 2 and s[0] == '"' and s[-1] == '"':
        s = s[1:-1].replace('""', '"')

    if s.startswith(("b'", 'b"')):
        try:
            lit = ast.literal_eval(s)
            if isinstance(lit, bytes):
                return lit.decode("utf-8", errors="replace")
            if isinstance(lit, str):
                return lit
        except (ValueError, SyntaxError, MemoryError):
            pass
        # Fallback: unwrap b'...' / b"..." without full literal_eval
        if s.startswith("b'") and s.endswith("'") and len(s) > 2:
            return s[2:-1].encode("latin-1", errors="replace").decode("unicode_escape", errors="replace")
        if s.startswith('b"') and s.endswith('"') and len(s) > 2:
            return s[2:-1].encode("latin-1", errors="replace").decode("unicode_escape", errors="replace")

    return s


def _coerce_author_cell(raw: str) -> str:
    """Normalise author id to a stable string label (CSV may use numeric ids)."""
    return str(raw).strip()


class DatasetLoader:
    """Load CSV/JSON authorship datasets and produce stratified splits."""

    def __init__(self) -> None:
        self.author_map: dict[int, str] = {}       # int id → author name
        self.num_authors: int = 0
        self.samples_per_author: dict[int, int] = {}  # int id → count

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load(
        self,
        path: str,
        *,
        fetch_if_missing: bool = False,
    ) -> tuple[list[str], list[int]]:
        """Load dataset from *path* (CSV or JSON) and return (texts, labels).

        CSV files must have a text column and an author column. Headers are
        matched case-insensitively, e.g. ``Text`` / ``Author`` (as in the
        Chanchal et al. release) or ``text`` / ``author``. If no named
        columns match but the file has **exactly two** columns, they are
        read as *text* then *author* (e.g. numeric ``0`` / ``1`` headers).
        Files are read as UTF-8 with BOM stripped (``utf-8-sig``).

        JSON files must be a list of objects; each object should provide text
        and author fields (any casing, e.g. ``Text`` / ``Author``).

        Author names or ids are mapped to 0-indexed integer labels (sorted
        by string value for a deterministic id order).

        Parameters
        ----------
        path:
            Path to a ``.csv`` or ``.json`` file. Relative paths are also tried
            under the project root (so ``data/...`` works from any CWD if the
            project root is the usual checkout layout).
        fetch_if_missing:
            If the file is not found, call :func:`ensure_authoridentification_dataset`
            to shallow-clone the public GitHub release, then resolve *path* again.
        """
        found = _resolve_dataset_file(path)
        if found is None and fetch_if_missing:
            ensure_authoridentification_dataset()
            found = _resolve_dataset_file(path)
        if found is None:
            msg = f"Dataset file not found: {path!r} (tried CWD and project root)"
            if not fetch_if_missing and path.startswith("data/AuthorIdentification/"):
                msg += (
                    "\nCall load(..., fetch_if_missing=True) or "
                    "ensure_authoridentification_dataset() first."
                )
            raise FileNotFoundError(msg)
        path = found

        if path.endswith(".json"):
            raw = self._load_json(path)
        else:
            raw = self._load_csv(path)

        # Build author → id mapping (sorted for determinism)
        unique_authors = sorted({author for _, author in raw})
        name_to_id: dict[str, int] = {name: idx for idx, name in enumerate(unique_authors)}
        self.author_map = {idx: name for name, idx in name_to_id.items()}

        texts: list[str] = []
        labels: list[int] = []
        for text, author in raw:
            texts.append(text)
            labels.append(name_to_id[author])

        self.num_authors = len(unique_authors)
        counts = Counter(labels)
        self.samples_per_author = dict(counts)

        return texts, labels

    def split(
        self,
        texts: list[str],
        labels: list[int],
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        seed: int = 42,
        min_samples: int = 10,
    ) -> tuple[Split, Split, Split]:
        """Stratified split into train / val / test partitions.

        Parameters
        ----------
        texts:       list of text strings
        labels:      corresponding integer author labels
        train_ratio: fraction of data for training (default 0.70)
        val_ratio:   fraction of data for validation (default 0.15)
        seed:        random seed for reproducibility (default 42)
        min_samples: minimum samples per author; raises
                     ``InsufficientSamplesError`` if any author falls below
                     this threshold (default 10)

        Returns
        -------
        (train, val, test) as ``Split`` namedtuples
        """
        # Validate minimum samples per author
        counts = Counter(labels)
        for author_id, count in counts.items():
            if count < min_samples:
                author_name = self.author_map.get(author_id, str(author_id))
                raise InsufficientSamplesError(
                    f"Author '{author_name}' (id={author_id}) has only {count} "
                    f"sample(s), which is below the minimum threshold of {min_samples}."
                )

        # --- Step 1: carve out the training split ---
        test_val_ratio = 1.0 - train_ratio  # fraction going to val+test
        sss_train = StratifiedShuffleSplit(
            n_splits=1, test_size=test_val_ratio, random_state=seed
        )
        train_idx, remainder_idx = next(sss_train.split(texts, labels))

        # --- Step 2: split remainder into val / test ---
        remainder_texts = [texts[i] for i in remainder_idx]
        remainder_labels = [labels[i] for i in remainder_idx]

        # val_ratio relative to the full dataset; compute relative to remainder
        val_relative = val_ratio / test_val_ratio
        sss_val = StratifiedShuffleSplit(
            n_splits=1, test_size=1.0 - val_relative, random_state=seed
        )
        val_idx_local, test_idx_local = next(
            sss_val.split(remainder_texts, remainder_labels)
        )

        # Map local indices back to original indices
        val_idx = [remainder_idx[i] for i in val_idx_local]
        test_idx = [remainder_idx[i] for i in test_idx_local]

        def _make_split(indices: list[int]) -> Split:
            return Split(
                texts=[texts[i] for i in indices],
                labels=[labels[i] for i in indices],
                author_map=dict(self.author_map),
            )

        return _make_split(list(train_idx)), _make_split(val_idx), _make_split(test_idx)

    def iter_stratified_kfold(
        self,
        texts: list[str],
        labels: list[int],
        *,
        n_splits: int,
        seed: int,
        min_samples: int = 10,
    ) -> Iterator[tuple[Split, Split, Split, int]]:
        """Yield stratified (train, val, test) for each *k*-fold. Test = ``1/n_splits``; train+val = rest.

        Within the train+val pool, validation is a ``15/85`` slice (same *relative* train:val
        ratio as :meth:`split` on full data: 70% train, 15% val, 15% test).

        Yields
        ------
        (train, val, test, fold_index) per fold, with ``fold_index`` in ``0 .. n_splits-1``.
        """
        if n_splits < 2:
            raise ValueError(f"n_splits must be >= 2, got {n_splits}")
        # Validate minimum samples per author (same as split())
        counts = Counter(labels)
        for author_id, count in counts.items():
            if count < min_samples:
                author_name = self.author_map.get(author_id, str(author_id))
                raise InsufficientSamplesError(
                    f"Author '{author_name}' (id={author_id}) has only {count} "
                    f"sample(s), which is below the minimum threshold of {min_samples}."
                )

        def _make_split(indices: list[int]) -> Split:
            return Split(
                texts=[texts[i] for i in indices],
                labels=[labels[i] for i in indices],
                author_map=dict(self.author_map),
            )

        n = len(labels)
        y = np.asarray(labels)
        X = np.zeros((n, 1), dtype=np.float32)
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        for fold, (train_val_idx, test_idx) in enumerate(skf.split(X, y)):
            train_val_idx = np.asarray(train_val_idx, dtype=np.int64)
            test_idx = np.asarray(test_idx, dtype=np.int64)
            tv_y = y[train_val_idx]
            X_tv = np.zeros((len(train_val_idx), 1), dtype=np.float32)
            sss = StratifiedShuffleSplit(
                n_splits=1,
                test_size=15.0 / 85.0,
                random_state=seed + 10_000 * int(fold),
            )
            tr_local, va_local = next(sss.split(X_tv, tv_y))
            train_idx = train_val_idx[tr_local]
            val_idx = train_val_idx[va_local]
            yield (
                _make_split(train_idx.tolist()),
                _make_split(val_idx.tolist()),
                _make_split(test_idx.tolist()),
                int(fold),
            )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _load_csv(path: str) -> list[tuple[str, str]]:
        rows: list[tuple[str, str]] = []
        with open(path, newline="", encoding="utf-8-sig") as fh:
            reader = csv.DictReader(fh)
            field_map = _normalize_csv_field_names(reader.fieldnames)
            for row in reader:
                raw_text, raw_author = _get_text_author_columns(row, field_map)
                text = _coerce_text_cell(raw_text)
                author = _coerce_author_cell(raw_author)
                rows.append((text, author))
        return rows

    @staticmethod
    def _load_json(path: str) -> list[tuple[str, str]]:
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
        out: list[tuple[str, str]] = []
        for item in data:
            low = {str(k).lower().strip(): v for k, v in item.items()}
            text_val: str | None = None
            for key in ("text", "tweet", "message", "content"):
                if key in low and low[key] is not None:
                    text_val = str(low[key])
                    break
            auth_val: str | None = None
            for key in ("author", "user", "user_id", "label"):
                if key in low and low[key] is not None:
                    auth_val = str(low[key])
                    break
            if text_val is None or auth_val is None:
                raise ValueError(
                    "Each JSON object must include text and author fields "
                    f"(any casing). Found keys: {list(item.keys())}"
                )
            out.append((_coerce_text_cell(text_val), _coerce_author_cell(auth_val)))
        return out
