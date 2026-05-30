"""Unit tests for ExplainabilityModule.

Tests cover:
- error_analysis() raises ValueError when no misclassifications (Req 8.6)
- top_tokens_per_pair contains at most top_k tokens per pair (Req 8.4)
- confusion_pairs are sorted by rate descending (Req 8.5)
- misclassified_indices are correctly identified (Req 8.3)
"""

from __future__ import annotations

import numpy as np
import pytest

from src.explainability import ExplainabilityModule, ShapExplanation


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_shap(index: int, predicted_class: int, num_tokens: int = 6) -> ShapExplanation:
    """Create a ShapExplanation with distinct shap values for testing."""
    return ShapExplanation(
        text=f"sample text {index}",
        token_ids=list(range(1, num_tokens + 1)),
        shap_values=np.arange(1, num_tokens + 1, dtype=float),
        predicted_class=predicted_class,
    )


# ---------------------------------------------------------------------------
# Req 8.6 — ValueError on zero misclassifications
# ---------------------------------------------------------------------------

def test_error_analysis_raises_on_no_misclassifications():
    """error_analysis() must raise ValueError when all predictions match labels."""
    module = ExplainabilityModule()
    explanations = [_make_shap(i, i) for i in range(3)]
    predictions = [0, 1, 2]
    labels = [0, 1, 2]  # all correct

    with pytest.raises(ValueError, match="misclassified"):
        module.error_analysis(explanations, predictions, labels)


def test_error_analysis_raises_on_single_correct_prediction():
    """Edge case: single sample, correctly classified — must raise ValueError."""
    module = ExplainabilityModule()
    explanations = [_make_shap(0, 0)]
    with pytest.raises(ValueError):
        module.error_analysis(explanations, [0], [0])


# ---------------------------------------------------------------------------
# Req 8.4 — top_tokens_per_pair contains at most top_k tokens per pair
# ---------------------------------------------------------------------------

def test_top_tokens_per_pair_respects_top_k():
    """top_tokens_per_pair must have at most top_k tokens for each pair."""
    module = ExplainabilityModule()
    top_k = 3
    # 4 misclassified samples all in the same (true=0, pred=1) pair
    explanations = [_make_shap(i, 1, num_tokens=8) for i in range(4)]
    predictions = [1, 1, 1, 1]
    labels = [0, 0, 0, 0]

    report = module.error_analysis(explanations, predictions, labels, top_k=top_k)

    for pair, tokens in report.top_tokens_per_pair.items():
        assert len(tokens) <= top_k, (
            f"Pair {pair} has {len(tokens)} tokens, expected at most {top_k}"
        )


def test_top_tokens_per_pair_top_k_one():
    """With top_k=1, each pair should have exactly 1 token (if any tokens exist)."""
    module = ExplainabilityModule()
    explanations = [_make_shap(0, 1, num_tokens=5)]
    predictions = [1]
    labels = [0]

    report = module.error_analysis(explanations, predictions, labels, top_k=1)

    for pair, tokens in report.top_tokens_per_pair.items():
        assert len(tokens) <= 1


def test_top_tokens_per_pair_multiple_pairs():
    """top_k constraint applies independently to each confusion pair."""
    module = ExplainabilityModule()
    top_k = 2
    # pair (0,1): sample 0; pair (1,0): sample 1
    explanations = [_make_shap(0, 1, num_tokens=6), _make_shap(1, 0, num_tokens=6)]
    predictions = [1, 0]
    labels = [0, 1]

    report = module.error_analysis(explanations, predictions, labels, top_k=top_k)

    assert len(report.top_tokens_per_pair) == 2
    for pair, tokens in report.top_tokens_per_pair.items():
        assert len(tokens) <= top_k


# ---------------------------------------------------------------------------
# Req 8.5 — confusion_pairs sorted by rate descending
# ---------------------------------------------------------------------------

def test_confusion_pairs_sorted_descending():
    """confusion_pairs must be sorted by confusion rate in descending order."""
    module = ExplainabilityModule()
    # 3 labels: 0, 1, 2
    # true=0 has 4 samples; 2 misclassified as 1 → rate 0.5
    # true=1 has 1 sample;  1 misclassified as 2 → rate 1.0
    # true=2 has 2 samples; 1 misclassified as 0 → rate 0.5
    predictions = [1, 1, 0, 0, 2, 0, 2]
    labels =      [0, 0, 0, 0, 1, 2, 2]
    explanations = [_make_shap(i, predictions[i]) for i in range(len(predictions))]

    report = module.error_analysis(explanations, predictions, labels)

    rates = [rate for _, _, rate in report.confusion_pairs]
    assert rates == sorted(rates, reverse=True), (
        f"confusion_pairs not sorted descending: {report.confusion_pairs}"
    )


def test_confusion_pairs_sorted_single_pair():
    """Single confusion pair — trivially sorted, rate should be correct."""
    module = ExplainabilityModule()
    # true=0 has 2 samples, 1 misclassified → rate 0.5
    predictions = [1, 0]
    labels = [0, 0]
    explanations = [_make_shap(i, predictions[i]) for i in range(2)]

    report = module.error_analysis(explanations, predictions, labels)

    assert len(report.confusion_pairs) == 1
    true_l, pred_l, rate = report.confusion_pairs[0]
    assert true_l == 0
    assert pred_l == 1
    assert abs(rate - 0.5) < 1e-9


def test_confusion_pairs_rate_equals_one():
    """When all samples of a class are misclassified, rate should be 1.0."""
    module = ExplainabilityModule()
    predictions = [1, 1]
    labels = [0, 0]
    explanations = [_make_shap(i, 1) for i in range(2)]

    report = module.error_analysis(explanations, predictions, labels)

    assert len(report.confusion_pairs) == 1
    _, _, rate = report.confusion_pairs[0]
    assert abs(rate - 1.0) < 1e-9


# ---------------------------------------------------------------------------
# Req 8.3 — misclassified_indices correctness
# ---------------------------------------------------------------------------

def test_misclassified_indices_correct():
    """misclassified_indices must exactly match positions where pred != label."""
    module = ExplainabilityModule()
    predictions = [0, 1, 0, 2, 1]
    labels =      [0, 0, 0, 2, 0]
    explanations = [_make_shap(i, predictions[i]) for i in range(5)]

    report = module.error_analysis(explanations, predictions, labels)

    expected = {1, 4}  # indices where pred != label
    assert set(report.misclassified_indices) == expected


def test_misclassified_indices_all_wrong():
    """When every prediction is wrong, all indices should be misclassified."""
    module = ExplainabilityModule()
    predictions = [1, 2, 0]
    labels =      [0, 0, 1]
    explanations = [_make_shap(i, predictions[i]) for i in range(3)]

    report = module.error_analysis(explanations, predictions, labels)

    assert set(report.misclassified_indices) == {0, 1, 2}
