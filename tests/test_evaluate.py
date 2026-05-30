"""Unit tests for the evaluate() function.

Tests known prediction/label pairs and verifies hand-calculated metric values.

Validates: Requirements 7.1, 7.2, 7.3, 7.4
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.evaluate import evaluate, evaluate_labels


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class _FixedPredictionModel(nn.Module):
    """Mock model that returns logits producing a fixed sequence of predictions.

    Given token_ids of shape [B, T], returns logits of shape [B, num_classes]
    where the predicted class for sample i is ``predictions[offset + i]``.
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


def _make_loader(labels: list[int], batch_size: int = 4) -> DataLoader:
    """Build a DataLoader with dummy token_ids and the given labels."""
    n = len(labels)
    token_ids = torch.zeros(n, 8, dtype=torch.long)
    labels_tensor = torch.tensor(labels, dtype=torch.long)
    dataset = TensorDataset(token_ids, labels_tensor)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False)


# ---------------------------------------------------------------------------
# Test 1: Perfect predictions → accuracy=1.0, f1_macro=1.0
# ---------------------------------------------------------------------------

def test_perfect_predictions_accuracy_and_f1() -> None:
    """Perfect predictions yield accuracy=1.0 and f1_macro=1.0.

    Validates: Requirements 7.1, 7.2
    """
    labels = [0, 0, 1, 1, 2, 2]
    predictions = labels[:]  # identical

    model = _FixedPredictionModel(predictions, num_classes=3)
    loader = _make_loader(labels)

    metrics = evaluate(model, loader)

    assert metrics.accuracy == pytest.approx(1.0)
    assert metrics.f1_macro == pytest.approx(1.0)
    assert metrics.precision_macro == pytest.approx(1.0)
    assert metrics.recall_macro == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Test 2: All wrong predictions (binary case) → accuracy=0.0
# ---------------------------------------------------------------------------

def test_all_wrong_binary_accuracy_zero() -> None:
    """All-wrong binary predictions yield accuracy=0.0.

    Validates: Requirements 7.1, 7.2
    """
    labels =      [0, 0, 0, 1, 1, 1]
    predictions = [1, 1, 1, 0, 0, 0]  # every prediction is wrong

    model = _FixedPredictionModel(predictions, num_classes=2)
    loader = _make_loader(labels)

    metrics = evaluate(model, loader)

    assert metrics.accuracy == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Test 3: Known 2-class case with hand-calculated accuracy=0.5
# ---------------------------------------------------------------------------

def test_known_two_class_accuracy() -> None:
    """y_true=[0,0,1,1], y_pred=[0,1,0,1] → accuracy=0.5.

    Validates: Requirements 7.1, 7.2
    """
    labels =      [0, 0, 1, 1]
    predictions = [0, 1, 0, 1]  # 2 correct out of 4

    model = _FixedPredictionModel(predictions, num_classes=2)
    loader = _make_loader(labels, batch_size=4)

    metrics = evaluate(model, loader)

    assert metrics.accuracy == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# Test 4: Confusion matrix sum equals total samples
# ---------------------------------------------------------------------------

def test_confusion_matrix_sum_equals_total_samples() -> None:
    """Sum of all confusion matrix elements equals the number of samples.

    Validates: Requirement 7.3
    """
    labels =      [0, 0, 1, 1, 2, 2, 0, 1]
    predictions = [0, 1, 1, 0, 2, 0, 0, 2]
    n = len(labels)

    model = _FixedPredictionModel(predictions, num_classes=3)
    loader = _make_loader(labels)

    metrics = evaluate(model, loader)

    assert int(metrics.confusion_matrix.sum()) == n


# ---------------------------------------------------------------------------
# Test 5: Trace of confusion matrix / total == accuracy
# ---------------------------------------------------------------------------

def test_confusion_matrix_trace_equals_accuracy() -> None:
    """trace(confusion_matrix) / total_samples == accuracy.

    Validates: Requirement 7.4
    """
    labels =      [0, 0, 1, 1, 2, 2, 0, 1]
    predictions = [0, 1, 1, 0, 2, 0, 0, 2]
    n = len(labels)

    model = _FixedPredictionModel(predictions, num_classes=3)
    loader = _make_loader(labels)

    metrics = evaluate(model, loader)

    trace_acc = float(np.trace(metrics.confusion_matrix)) / n
    assert trace_acc == pytest.approx(metrics.accuracy, abs=1e-6)


# ---------------------------------------------------------------------------
# Test 6: All scalar metrics are in [0.0, 1.0]
# ---------------------------------------------------------------------------

def test_all_scalar_metrics_in_unit_interval() -> None:
    """All scalar metrics (accuracy, precision, recall, f1, per-class f1) are in [0, 1].

    Validates: Requirement 7.2
    """
    labels =      [0, 1, 0, 1, 2, 2]
    predictions = [1, 0, 0, 1, 2, 0]

    model = _FixedPredictionModel(predictions, num_classes=3)
    loader = _make_loader(labels)

    metrics = evaluate(model, loader)

    assert 0.0 <= metrics.accuracy <= 1.0
    assert 0.0 <= metrics.precision_macro <= 1.0
    assert 0.0 <= metrics.recall_macro <= 1.0
    assert 0.0 <= metrics.f1_macro <= 1.0
    for cls, f1_val in metrics.f1_per_class.items():
        assert 0.0 <= f1_val <= 1.0, f"f1_per_class[{cls}]={f1_val} out of [0, 1]"


# ---------------------------------------------------------------------------
# Test 7: Per-class F1 keys match the classes present in y_true / y_pred
# ---------------------------------------------------------------------------

def test_per_class_f1_keys_match_classes_present() -> None:
    """f1_per_class keys are exactly the union of classes in y_true and y_pred.

    Validates: Requirement 7.1
    """
    labels =      [0, 0, 1, 1]
    predictions = [0, 1, 0, 1]
    expected_classes = {0, 1}

    model = _FixedPredictionModel(predictions, num_classes=2)
    loader = _make_loader(labels, batch_size=4)

    metrics = evaluate(model, loader)

    assert set(metrics.f1_per_class.keys()) == expected_classes


def test_per_class_f1_keys_include_predicted_only_class() -> None:
    """f1_per_class includes a class that appears only in predictions (not labels).

    Validates: Requirement 7.1
    """
    # Class 2 appears only in predictions, not in labels
    labels =      [0, 0, 1, 1]
    predictions = [0, 2, 1, 2]
    expected_classes = {0, 1, 2}

    model = _FixedPredictionModel(predictions, num_classes=3)
    loader = _make_loader(labels, batch_size=4)

    metrics = evaluate(model, loader)

    assert set(metrics.f1_per_class.keys()) == expected_classes


def test_evaluate_matches_evaluate_labels_numpy_path() -> None:
    labels = [0, 0, 1, 1, 2, 2]
    preds = list(labels)
    m_arr = evaluate_labels(np.asarray(labels), np.asarray(preds))
    model = _FixedPredictionModel(preds, num_classes=3)
    loader = _make_loader(labels)
    m_mod = evaluate(model, loader)

    assert m_mod.accuracy == pytest.approx(m_arr.accuracy)
    assert m_mod.f1_macro == pytest.approx(m_arr.f1_macro)
    np.testing.assert_array_equal(m_mod.confusion_matrix, m_arr.confusion_matrix)
