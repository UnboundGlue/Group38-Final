"""Smoke tests for :mod:`src.sparse_baselines` (task 12 integration)."""

from __future__ import annotations

from src.models import Split
from src.sparse_baselines import nested_sparse_baseline_results


def test_nested_sparse_baselines_produces_splits_and_metrics_for_bow_logreg() -> None:
    train = Split(
        texts=(
            ["author zero loves cheese and pickles"] * 12
            + ["author one hates cheese unique words"] * 12
        ),
        labels=[0] * 12 + [1] * 12,
        author_map={0: "a0", 1: "a1"},
    )
    val = Split(
        texts=(
            ["zero val cheese pickles"] * 6 + ["one val unique hates"] * 6
        ),
        labels=[0] * 6 + [1] * 6,
        author_map={0: "a0", 1: "a1"},
    )
    test = Split(
        texts=(
            ["zero style cheese pickles here"] * 8 + ["one style unique hates"] * 8
        ),
        labels=[0] * 8 + [1] * 8,
        author_map={0: "a0", 1: "a1"},
    )

    results, timing = nested_sparse_baseline_results(train, val, test, seed=0)

    bow_log = results["bow"]["logreg"]
    assert isinstance(bow_log, dict)
    assert "splits" in bow_log
    assert "timing_seconds" in bow_log
    for key in ("train", "validation", "test"):
        sp = bow_log["splits"][key]
        assert "accuracy" in sp
        assert "f1_macro" in sp
        assert isinstance(sp["confusion_matrix"], list)
    assert "error" not in bow_log
    assert isinstance(timing["total_wall_seconds"], float)
    assert "rows" in timing
