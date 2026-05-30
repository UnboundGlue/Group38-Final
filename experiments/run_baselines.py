"""Baseline-only pipeline: four feature types × logistic regression and linear SVM.

    python -m experiments.run_baselines --fetch-dataset --seed 42

Writes metrics to ``results/metrics.json`` (default).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import sys
import time

import numpy as np

# Ensure the Repository/src package is importable when the script is run
# directly from the Repository directory.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.dataset import DEFAULT_CHANCHAL_200_CSV, DatasetLoader
from src.preprocessing import Preprocessor
from src.sparse_baselines import nested_sparse_baseline_results

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _set_seed(seed: int) -> None:
    """Set random seeds for sklearn and NumPy."""
    random.seed(seed)
    np.random.seed(seed)


# ---------------------------------------------------------------------------
# CLI argument parsing
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Train and evaluate baseline classifiers for authorship attribution.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--dataset",
        default=DEFAULT_CHANCHAL_200_CSV,
        help=(
            "Path to dataset file (CSV or JSON). Default: Chanchal 200-tweets slice "
            "(same default as run_cnn_lstm; see src/dataset.py)."
        ),
    )
    p.add_argument(
        "--fetch-dataset",
        action="store_true",
        help="If the file is missing, clone chanchalIITP/AuthorIdentification into data/ (needs git).",
    )
    p.add_argument("--seed", type=int, default=42, help="Random seed.")
    p.add_argument(
        "--output",
        default="results/metrics.json",
        help="Path to save the metrics JSON file.",
    )
    return p


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run(args: argparse.Namespace) -> dict:
    """Execute the baseline pipeline and return a results dict.

    Returns:
        Nested dict keyed by method → classifier → metrics.
    """
    _set_seed(args.seed)

    # ------------------------------------------------------------------
    # 1. Load dataset
    # ------------------------------------------------------------------
    logger.info("Loading dataset from %s …", args.dataset)
    loader = DatasetLoader()
    texts, labels = loader.load(args.dataset, fetch_if_missing=args.fetch_dataset)
    logger.info("Loaded %d samples across %d authors.", len(texts), loader.num_authors)

    # ------------------------------------------------------------------
    # 2. Preprocess
    # ------------------------------------------------------------------
    preprocessor = Preprocessor()
    texts = preprocessor.batch_clean(texts)

    # Remove empty texts produced by cleaning
    paired = [(t, l) for t, l in zip(texts, labels) if t]
    if not paired:
        raise ValueError("All texts became empty after preprocessing.")
    texts, labels = zip(*paired)
    texts = list(texts)
    labels = list(labels)

    # ------------------------------------------------------------------
    # 3. Split dataset
    # ------------------------------------------------------------------
    train_split, val_split, test_split = loader.split(
        texts, labels, seed=args.seed
    )
    logger.info(
        "Split sizes — train: %d  val: %d  test: %d",
        len(train_split.texts), len(val_split.texts), len(test_split.texts),
    )

    # ------------------------------------------------------------------
    # 4. Baseline sparse features + sklearn classifiers (:mod:`sparse_baselines`)
    # ------------------------------------------------------------------
    t0 = time.perf_counter()
    baseline_block, baseline_timing = nested_sparse_baseline_results(
        train_split, val_split, test_split, seed=args.seed
    )
    cpu_wall = time.perf_counter() - t0
    results = {
        "schema_version": 2,
        "baselines": baseline_block,
        "timings_seconds": {
            "baselines_cpu_wall": cpu_wall,
            "baselines_detail": baseline_timing,
        },
    }

    # ------------------------------------------------------------------
    # Save metrics JSON
    # ------------------------------------------------------------------
    output_path = args.output
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2)
    logger.info("Baseline metrics saved to %s", output_path)

    return results


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
