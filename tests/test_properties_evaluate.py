"""Property-based tests for the evaluate module.

**Validates: Requirements 7.2**
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from hypothesis import given, settings
from hypothesis import strategies as st
from torch.utils.data import DataLoader, TensorDataset

from src.evaluate import evaluate


class _FixedPredictionModel(nn.Module):
    """Mock model that returns logits producing a fixed set of predictions.

    Given token_ids of shape [B, T], returns logits of shape [B, num_classes]
    where the predicted class for sample i is ``predictions[offset + i]``.
    The logits use a large value (10.0) for the predicted class and 0.0 elsewhere,
    so argmax always returns the desired prediction.
    """

    def __init__(self, predictions: list[int], num_classes: int) -> None:
        super().__init__()
        self._predictions = predictions
        self._num_classes = num_classes
        self._call_count = 0
        # Dummy parameter so next(model.parameters()) works in evaluate()
        self._dummy = nn.Parameter(torch.zeros(1), requires_grad=False)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        batch_size = token_ids.shape[0]
        start = self._call_count
        end = start + batch_size
        batch_preds = self._predictions[start:end]
        self._call_count = end

        logits = torch.zeros(batch_size, self._num_classes)
        for i, cls in enumerate(batch_preds):
            logits[i, cls] = 10.0
        return logits


@given(
    num_classes=st.integers(min_value=2, max_value=5),
    num_samples=st.integers(min_value=4, max_value=20),
    seed=st.integers(min_value=0, max_value=2**31 - 1),
)
@settings(max_examples=100, deadline=None)
def test_metric_bounds(num_classes: int, num_samples: int, seed: int) -> None:
    """Property 6: Metric Bounds.

    For any set of predictions and ground-truth labels, all scalar values in
    the resulting MetricsDict (accuracy, precision_macro, recall_macro, f1_macro,
    and per-class F1 values) must be in the range [0.0, 1.0].

    **Validates: Requirements 7.2**
    """
    rng = np.random.default_rng(seed)

    labels_np = rng.integers(0, num_classes, size=num_samples).tolist()
    predictions_np = rng.integers(0, num_classes, size=num_samples).tolist()

    # Build a DataLoader with dummy token_ids (shape [N, 8]) and the generated labels
    token_ids = torch.zeros(num_samples, 8, dtype=torch.long)
    labels_tensor = torch.tensor(labels_np, dtype=torch.long)
    dataset = TensorDataset(token_ids, labels_tensor)
    loader = DataLoader(dataset, batch_size=4, shuffle=False)

    model = _FixedPredictionModel(predictions_np, num_classes)

    metrics = evaluate(model, loader)

    # All scalar metrics must be in [0.0, 1.0]
    assert 0.0 <= metrics.accuracy <= 1.0, (
        f"accuracy={metrics.accuracy} out of [0, 1]"
    )
    assert 0.0 <= metrics.precision_macro <= 1.0, (
        f"precision_macro={metrics.precision_macro} out of [0, 1]"
    )
    assert 0.0 <= metrics.recall_macro <= 1.0, (
        f"recall_macro={metrics.recall_macro} out of [0, 1]"
    )
    assert 0.0 <= metrics.f1_macro <= 1.0, (
        f"f1_macro={metrics.f1_macro} out of [0, 1]"
    )
    for cls, f1_val in metrics.f1_per_class.items():
        assert 0.0 <= f1_val <= 1.0, (
            f"f1_per_class[{cls}]={f1_val} out of [0, 1]"
        )


@given(
    num_classes=st.integers(min_value=2, max_value=5),
    num_samples=st.integers(min_value=4, max_value=20),
    seed=st.integers(min_value=0, max_value=2**31 - 1),
)
@settings(max_examples=100, deadline=None)
def test_confusion_matrix_sum_invariant(num_classes: int, num_samples: int, seed: int) -> None:
    """Property 7: Confusion Matrix Sum Invariant.

    For any set of predictions and ground-truth labels:
    1. The sum of all elements in the confusion matrix equals the total number
       of evaluated samples.
    2. The trace of the confusion matrix divided by the total number of samples
       equals the accuracy.

    **Validates: Requirements 7.3, 7.4**
    """
    rng = np.random.default_rng(seed)

    labels_np = rng.integers(0, num_classes, size=num_samples).tolist()
    predictions_np = rng.integers(0, num_classes, size=num_samples).tolist()

    token_ids = torch.zeros(num_samples, 8, dtype=torch.long)
    labels_tensor = torch.tensor(labels_np, dtype=torch.long)
    dataset = TensorDataset(token_ids, labels_tensor)
    loader = DataLoader(dataset, batch_size=4, shuffle=False)

    model = _FixedPredictionModel(predictions_np, num_classes)

    metrics = evaluate(model, loader)

    cm = metrics.confusion_matrix
    total = num_samples

    # Property 7.1: sum of all confusion matrix elements == total samples
    assert int(cm.sum()) == total, (
        f"confusion_matrix.sum()={cm.sum()} != total_samples={total}"
    )

    # Property 7.2: trace(cm) / total == accuracy
    trace_acc = float(np.trace(cm)) / total
    assert abs(trace_acc - metrics.accuracy) < 1e-6, (
        f"trace(cm)/total={trace_acc} != accuracy={metrics.accuracy}"
    )
