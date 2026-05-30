"""Property-based tests for the explainability module.

**Validates: Requirements 8.3, 8.1**
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import torch
from hypothesis import given, settings
from hypothesis import strategies as st

from src.explainability import ExplainabilityModule, ShapExplanation


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_shap_explanation(index: int, predicted_class: int, num_tokens: int = 8) -> ShapExplanation:
    """Create a minimal mock ShapExplanation without running actual SHAP."""
    return ShapExplanation(
        text=f"sample text {index}",
        token_ids=list(range(1, num_tokens + 1)),
        shap_values=np.ones(num_tokens, dtype=float),
        predicted_class=predicted_class,
    )


# ---------------------------------------------------------------------------
# Strategy: generate predictions/labels with at least one mismatch
# ---------------------------------------------------------------------------

@st.composite
def predictions_and_labels_with_mismatch(draw):
    """Generate (predictions, labels) lists with at least one misclassified sample."""
    num_samples = draw(st.integers(min_value=2, max_value=10))
    num_classes = 3

    predictions = draw(
        st.lists(
            st.integers(min_value=0, max_value=num_classes - 1),
            min_size=num_samples,
            max_size=num_samples,
        )
    )
    labels = draw(
        st.lists(
            st.integers(min_value=0, max_value=num_classes - 1),
            min_size=num_samples,
            max_size=num_samples,
        )
    )

    # Ensure at least one mismatch: force index 0 to differ
    if predictions == labels:
        labels = list(labels)
        labels[0] = (predictions[0] + 1) % num_classes

    return predictions, labels


# ---------------------------------------------------------------------------
# Property 14: Explainability Coverage
# ---------------------------------------------------------------------------

@given(predictions_and_labels=predictions_and_labels_with_mismatch())
@settings(max_examples=100, deadline=None)
def test_explainability_coverage(predictions_and_labels) -> None:
    """Property 14: Explainability Coverage.

    For any set of misclassified samples passed to ``error_analysis()``, the
    resulting ``ErrorAnalysisReport`` must contain attributions for every
    misclassified sample in the input list — i.e., every index ``i`` where
    ``predictions[i] != labels[i]`` must appear in
    ``report.misclassified_indices``.

    **Validates: Requirements 8.3**
    """
    predictions, labels = predictions_and_labels
    num_samples = len(predictions)

    # Build mock ShapExplanation objects (no real SHAP computation)
    explanations = [
        _make_shap_explanation(i, predictions[i])
        for i in range(num_samples)
    ]

    # Compute expected misclassified indices
    expected_misclassified = {
        i for i, (p, l) in enumerate(zip(predictions, labels)) if p != l
    }

    module = ExplainabilityModule()
    report = module.error_analysis(explanations, predictions, labels)

    actual_misclassified = set(report.misclassified_indices)

    assert expected_misclassified == actual_misclassified, (
        f"Expected misclassified_indices={sorted(expected_misclassified)}, "
        f"got {sorted(actual_misclassified)}"
    )


# ---------------------------------------------------------------------------
# Property 16: SHAP Explanation Count
# ---------------------------------------------------------------------------

@given(n=st.integers(min_value=1, max_value=10))
@settings(max_examples=20, deadline=None)
def test_shap_explanation_count(n: int) -> None:
    """Property 16: SHAP Explanation Count.

    For any list of N texts passed to ``explain_shap()``, the
    ExplainabilityModule SHALL return exactly N ShapExplanation objects.

    **Validates: Requirements 8.1**
    """
    max_length = 16
    num_classes = 3

    # Build N dummy texts
    texts = [f"text sample {i}" for i in range(n)]
    background_texts = ["background text one", "background text two"]

    # --- Mock model ---
    mock_model = MagicMock(spec=torch.nn.Module)
    mock_model.eval.return_value = None

    def _fake_forward(token_tensor: torch.Tensor) -> torch.Tensor:
        batch_size = token_tensor.shape[0]
        logits = torch.zeros(batch_size, num_classes)
        logits[:, 0] = 1.0
        return logits

    mock_model.side_effect = _fake_forward

    # --- Mock tokeniser ---
    mock_tokeniser = MagicMock()
    mock_tokeniser.batch_encode.side_effect = lambda txts, max_length=max_length: np.zeros(
        (len(txts), max_length), dtype=np.int64
    )

    # --- Mock shap.KernelExplainer ---
    # shap_values returns a list of arrays (one per class), each shape [N, max_length]
    def _make_fake_shap_values(input_ids, nsamples=100):
        n_inputs = input_ids.shape[0]
        return [np.zeros((n_inputs, max_length)) for _ in range(num_classes)]

    mock_explainer_instance = MagicMock()
    mock_explainer_instance.shap_values.side_effect = _make_fake_shap_values

    # shap may not be installed; inject a fake module so the lazy import succeeds
    import sys
    import types

    fake_shap = types.ModuleType("shap")
    fake_shap.KernelExplainer = MagicMock(return_value=mock_explainer_instance)  # type: ignore[attr-defined]

    with patch.dict(sys.modules, {"shap": fake_shap}):
        module = ExplainabilityModule()
        results = module.explain_shap(
            model=mock_model,
            tokeniser=mock_tokeniser,
            texts=texts,
            background_texts=background_texts,
            max_length=max_length,
        )

    assert len(results) == n, (
        f"explain_shap() returned {len(results)} explanations for {n} input texts; "
        f"expected exactly {n}."
    )
    for explanation in results:
        assert isinstance(explanation, ShapExplanation), (
            f"Expected ShapExplanation, got {type(explanation)}"
        )
