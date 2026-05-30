"""Integration tests for the end-to-end neural authorship attribution pipeline.

Tests:
    1. Full pipeline on a synthetic dataset (50 samples, 5 authors) — all metrics in [0.0, 1.0].
    2. Checkpoint save/load round-trip produces identical predictions.
    3. Metrics dict is JSON-serialisable (metrics file pattern).
    4. Subword tokeniser save/load preserves encodings.

Validates pipeline requirements **9.1–9.5** directly here; **9.6** (CNN + sparse baselines
together) is covered by ``src.sparse_baselines`` tests and ``run_cnn_lstm`` / ``run_baselines``.
"""

from __future__ import annotations

import os
import tempfile

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.evaluate import evaluate, metrics_dict_to_jsonable
from src.model import CNNLSTMModel
from src.models import ModelConfig
from src.preprocessing import Preprocessor
from src.tokeniser import SubwordTokeniser
from src.trainer import Trainer

# ---------------------------------------------------------------------------
# Synthetic dataset: 50 samples, 5 authors, 10 texts each
# ---------------------------------------------------------------------------

_NUM_AUTHORS = 5
_SAMPLES_PER_AUTHOR = 10
_SEQ_LEN = 16  # must be >= max(kernel_sizes) = 3

_AUTHOR_TEXTS: list[list[str]] = [
    [
        f"author{author} writes about topic {i} with unique style and vocabulary words here"
        for i in range(_SAMPLES_PER_AUTHOR)
    ]
    for author in range(_NUM_AUTHORS)
]

_ALL_TEXTS: list[str] = [text for author_texts in _AUTHOR_TEXTS for text in author_texts]
_ALL_LABELS: list[int] = [
    author
    for author in range(_NUM_AUTHORS)
    for _ in range(_SAMPLES_PER_AUTHOR)
]

# Tiny model config as specified in the task
_MODEL_CONFIG = ModelConfig(
    vocab_size=300,
    embed_dim=8,
    num_filters=4,
    kernel_sizes=[2, 3],
    lstm_hidden=8,
    lstm_layers=1,
    dropout=0.0,
    num_classes=_NUM_AUTHORS,
)


def _build_loaders(
    encoded: np.ndarray,
    labels: list[int],
    train_ratio: float = 0.6,
    val_ratio: float = 0.2,
    batch_size: int = 8,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Split encoded data into train/val/test DataLoaders."""
    n = len(labels)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    token_ids = torch.tensor(encoded, dtype=torch.long)
    labels_tensor = torch.tensor(labels, dtype=torch.long)

    train_ids, train_labels = token_ids[:n_train], labels_tensor[:n_train]
    val_ids, val_labels = token_ids[n_train : n_train + n_val], labels_tensor[n_train : n_train + n_val]
    test_ids, test_labels = token_ids[n_train + n_val :], labels_tensor[n_train + n_val :]

    train_loader = DataLoader(TensorDataset(train_ids, train_labels), batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(TensorDataset(val_ids, val_labels), batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(TensorDataset(test_ids, test_labels), batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader


# ---------------------------------------------------------------------------
# Test 1: Full pipeline — no errors, all metrics in [0.0, 1.0]
# ---------------------------------------------------------------------------

def test_full_pipeline_synthetic_dataset() -> None:
    """Run the full pipeline on a synthetic 50-sample / 5-author dataset.

    Steps:
        1. Preprocess texts with Preprocessor.
        2. Train SubwordTokeniser on train split.
        3. Encode all splits.
        4. Build tiny CNNLSTMModel.
        5. Train for 2 epochs with Trainer.
        6. Evaluate with evaluate().
        7. Assert all scalar metrics are in [0.0, 1.0].

    Validates: Requirements 9.5
    """
    # Step 1: Preprocess
    preprocessor = Preprocessor()
    cleaned_texts = preprocessor.batch_clean(_ALL_TEXTS)

    # Determine train split boundary for tokeniser training
    n = len(cleaned_texts)
    n_train = int(n * 0.6)
    train_texts = cleaned_texts[:n_train]

    # Step 2: Train tokeniser on train texts only (Req 3.8)
    tokeniser = SubwordTokeniser()
    tokeniser.train(train_texts, vocab_size=300, algorithm="bpe")

    # Step 3: Encode all texts
    encoded = tokeniser.batch_encode(cleaned_texts, max_length=_SEQ_LEN)

    # Build DataLoaders
    train_loader, val_loader, test_loader = _build_loaders(encoded, _ALL_LABELS)

    # Step 4 & 5: Build model and train for 2 epochs
    model = CNNLSTMModel(_MODEL_CONFIG)

    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_path = os.path.join(tmpdir, "best_model.pt")

        trainer = Trainer()
        history = trainer.train(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=2,
            lr=1e-3,
            patience=5,
            checkpoint_path=ckpt_path,
        )

        # Training history should have entries for both epochs
        assert len(history.train_losses) == 2
        assert len(history.val_metrics) == 2

        # Step 6: Evaluate on test set
        metrics = evaluate(model, test_loader)

    # Step 7: All scalar metrics must be in [0.0, 1.0] (Req 7.2, 9.5)
    assert 0.0 <= metrics.accuracy <= 1.0, f"accuracy={metrics.accuracy} out of range"
    assert 0.0 <= metrics.precision_macro <= 1.0, f"precision={metrics.precision_macro} out of range"
    assert 0.0 <= metrics.recall_macro <= 1.0, f"recall={metrics.recall_macro} out of range"
    assert 0.0 <= metrics.f1_macro <= 1.0, f"f1_macro={metrics.f1_macro} out of range"

    for cls_id, f1_val in metrics.f1_per_class.items():
        assert 0.0 <= f1_val <= 1.0, f"per-class F1 for class {cls_id}={f1_val} out of range"

    # Confusion matrix sanity: sum equals number of test samples
    n_test = len(_ALL_LABELS) - int(len(_ALL_LABELS) * 0.6) - int(len(_ALL_LABELS) * 0.2)
    assert metrics.confusion_matrix.sum() == n_test


# ---------------------------------------------------------------------------
# Test 2: Checkpoint save/load round-trip produces identical predictions
# ---------------------------------------------------------------------------

def test_checkpoint_roundtrip_identical_predictions() -> None:
    """Save a checkpoint and reload it; predictions on the same data must be identical.

    Steps:
        1. Train a model and save checkpoint.
        2. Load checkpoint into a fresh model.
        3. Run inference on the same test data.
        4. Assert predictions are identical.

    Validates: Requirements 9.4
    """
    # Preprocess and encode
    preprocessor = Preprocessor()
    cleaned_texts = preprocessor.batch_clean(_ALL_TEXTS)

    n = len(cleaned_texts)
    n_train = int(n * 0.6)
    train_texts = cleaned_texts[:n_train]

    tokeniser = SubwordTokeniser()
    tokeniser.train(train_texts, vocab_size=300, algorithm="bpe")
    encoded = tokeniser.batch_encode(cleaned_texts, max_length=_SEQ_LEN)

    train_loader, val_loader, test_loader = _build_loaders(encoded, _ALL_LABELS)

    # Step 1: Train model and save checkpoint
    model = CNNLSTMModel(_MODEL_CONFIG)

    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_path = os.path.join(tmpdir, "best_model.pt")

        trainer = Trainer()
        trainer.train(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=2,
            lr=1e-3,
            patience=5,
            checkpoint_path=ckpt_path,
        )

        assert os.path.exists(ckpt_path), "Checkpoint file was not created"

        # Trainer keeps last-epoch weights in memory but saves the *best val* checkpoint;
        # align in-memory weights with the file before comparing to a fresh load.
        model.load_state_dict(torch.load(ckpt_path, map_location="cpu"))

        # Collect predictions from the model with checkpoint weights
        model.eval()
        original_preds: list[int] = []
        with torch.no_grad():
            for inputs, _ in test_loader:
                logits = model(inputs)
                preds = logits.argmax(dim=-1).tolist()
                original_preds.extend(preds)

        # Step 2: Load checkpoint into a fresh model
        fresh_model = CNNLSTMModel(_MODEL_CONFIG)
        state_dict = torch.load(ckpt_path, map_location="cpu")
        fresh_model.load_state_dict(state_dict)

        # Step 3: Run inference with the fresh model on the same test data
        fresh_model.eval()
        loaded_preds: list[int] = []
        with torch.no_grad():
            for inputs, _ in test_loader:
                logits = fresh_model(inputs)
                preds = logits.argmax(dim=-1).tolist()
                loaded_preds.extend(preds)

    # Step 4: Predictions must be identical
    assert original_preds == loaded_preds, (
        f"Predictions differ after checkpoint round-trip.\n"
        f"Original: {original_preds}\n"
        f"Loaded:   {loaded_preds}"
    )


def test_evaluate_outputs_are_json_serializable() -> None:
    """After training, test metrics can be written as JSON (Req 9.2)."""
    import json

    preprocessor = Preprocessor()
    cleaned_texts = preprocessor.batch_clean(_ALL_TEXTS)
    n_train = int(len(cleaned_texts) * 0.6)
    train_texts = cleaned_texts[:n_train]

    tokeniser = SubwordTokeniser()
    tokeniser.train(train_texts, vocab_size=300, algorithm="bpe")
    encoded = tokeniser.batch_encode(cleaned_texts, max_length=_SEQ_LEN)
    train_loader, val_loader, test_loader = _build_loaders(encoded, _ALL_LABELS)

    model = CNNLSTMModel(_MODEL_CONFIG)
    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_path = os.path.join(tmpdir, "best_model.pt")
        Trainer().train(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=1,
            lr=1e-3,
            patience=5,
            checkpoint_path=ckpt_path,
        )
        metrics = evaluate(model, test_loader)

    blob = metrics_dict_to_jsonable(metrics)
    json.dumps(blob)


def test_subword_tokeniser_save_load_preserves_encoding() -> None:
    """Save/load reproduces identical encodings (Req 9.3 artefact contract)."""
    preprocessor = Preprocessor()
    cleaned_texts = preprocessor.batch_clean(_ALL_TEXTS)
    n_train = int(len(cleaned_texts) * 0.6)
    train_texts = cleaned_texts[:n_train]

    tokeniser = SubwordTokeniser()
    tokeniser.train(train_texts, vocab_size=300, algorithm="bpe")

    fd, path = tempfile.mkstemp(suffix=".json")
    os.close(fd)
    try:
        tokeniser.save(path)
        loaded = SubwordTokeniser()
        loaded.load(path)
        a = tokeniser.batch_encode(cleaned_texts[:7], max_length=_SEQ_LEN)
        b = loaded.batch_encode(cleaned_texts[:7], max_length=_SEQ_LEN)
        np.testing.assert_array_equal(a, b)
    finally:
        os.unlink(path)
