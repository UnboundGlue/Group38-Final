# Implementation Plan: Neural Authorship Attribution

## Overview

Implement the full neural authorship attribution pipeline in Python: dataset loading, preprocessing, subword tokenisation (BPE/WordPiece), CNN-LSTM model, training loop, evaluation, baseline comparison, and SHAP/LIME explainability.

## Tasks

- [x] 1. Set up project structure, dependencies, and core data models(Ruan)
  - Create directory layout: `src/`, `tests/`, `experiments/`, `results/`, `artifacts/` (`runs/`, `best_model_bundle/`)
  - Create `requirements.txt` with pinned versions for torch, tokenizers, scikit-learn, numpy, pandas, shap, lime, hypothesis, pytest
  - Implement `AuthorSample`, `Split`, `ModelConfig`, `TrainingConfig`, `MetricsDict`, `TrainingHistory`, `ErrorAnalysisReport` dataclasses in `src/models.py`
  - Define custom exceptions: `InsufficientSamplesError`, `TrainingDivergenceError`
  - _Requirements: 1.4, 5.6, 6.8, 8.3_

- [x] 2. Implement DatasetLoader  (RUan)
  - [x] 2.1 Implement `DatasetLoader` in `src/dataset.py`
    - `load()`: read CSV/JSON, map author names to 0-indexed integer labels, return `(texts, labels)`
    - `split()`: stratified train/val/test split using `sklearn.model_selection.StratifiedShuffleSplit`; raise `InsufficientSamplesError` for authors below threshold
    - Expose `num_authors` and `samples_per_author` statistics
    - _Requirements: 1.1, 1.2, 1.3, 1.4, 1.5_

  - [x]* 2.2 Write property test for stratified split coverage (Property 3)
    - **Property 3: Stratified Split Coverage**
    - **Validates: Requirements 1.3**

  - [x]* 2.3 Write property test for non-overlapping splits (Property 4)
    - **Property 4: Non-Overlapping Splits**
    - **Validates: Requirements 1.2**

  - [x]* 2.4 Write unit tests for DatasetLoader
    - Test CSV and JSON loading, label mapping, `InsufficientSamplesError` on small classes, split ratios
    - _Requirements: 1.1, 1.2, 1.3, 1.4, 1.5_

- [x] 3. Implement Preprocessor(Ruan)
  - [x] 3.1 Implement `Preprocessor` in `src/preprocessing.py`
    - `clean()`: remove URLs, @mentions, normalise whitespace; preserve punctuation by default
    - `batch_clean()`: apply `clean()` to every element, return list of equal length; empty strings pass through
    - _Requirements: 2.1, 2.2, 2.3, 2.4_

  - [x]* 3.2 Write property test for batch clean length preservation (Property 11)
    - **Property 11: Batch Clean Length Preservation**
    - **Validates: Requirements 2.2**

  - [x]* 3.3 Write unit tests for Preprocessor
    - Test URL removal, mention stripping, whitespace normalisation, empty string output
    - _Requirements: 2.1, 2.2, 2.3_

- [ ] 4. Implement SubwordTokeniser(mac)
  - [ ] 4.1 Implement `SubwordTokeniser` in `src/tokeniser.py`
    - Wrap HuggingFace `tokenizers` library for BPE and WordPiece training
    - `train()`: train vocabulary on corpus; enforce `vocab_size` bound
    - `encode()`: pad/truncate to `max_length`; map unknowns to `[UNK]`; log warning on truncation
    - `batch_encode()`: return `np.ndarray` of shape `[N, max_length]`
    - `decode()`: reconstruct text from token IDs
    - `save()` / `load()`: persist vocabulary and merge rules to `artifacts/tokeniser.json`
    - _Requirements: 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7, 3.8, 10.1, 10.2, 10.3_

  - [ ]* 4.2 Write property test for tokenisation round-trip (Property 1)
    - **Property 1: Tokenisation Round-Trip**
    - **Validates: Requirements 3.6, 10.1**

  - [ ]* 4.3 Write property test for tokeniser serialisation round-trip (Property 2)
    - **Property 2: Tokeniser Serialisation Round-Trip**
    - **Validates: Requirements 3.7, 10.2, 10.3**

  - [ ]* 4.4 Write property test for encode length invariant (Property 10)
    - **Property 10: Encode Length Invariant**
    - **Validates: Requirements 3.3**

  - [ ]* 4.5 Write property test for BPE vocabulary size bound (Property 9)
    - **Property 9: BPE Vocabulary Size Bound**
    - **Validates: Requirements 3.2**

  - [ ]* 4.6 Write unit tests for SubwordTokeniser
    - Test padding, truncation with warning, `[UNK]` handling, save/load round-trip
    - _Requirements: 3.3, 3.4, 3.5, 3.7_

- [ ] 5. Checkpoint — Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

- [ ] 6. Implement BaselineFeatureExtractor (mosa)
  - [ ] 6.1 Implement `BaselineFeatureExtractor` in `src/features.py`
    - `fit_transform()`: wrap `CountVectorizer` (bow), `TfidfVectorizer` (tfidf), char n-gram and word n-gram variants; return sparse `[N, F]` matrix
    - `transform()`: apply fitted vocabulary to unseen texts without refitting
    - Expose configurable n-gram ranges and random seed
    - _Requirements: 4.1, 4.2, 4.3, 4.4_

  - [ ]* 6.2 Write property test for baseline feature matrix row count (Property 12)
    - **Property 12: Baseline Feature Matrix Row Count**
    - **Validates: Requirements 4.1, 4.2**

  - [ ]* 6.3 Write property test for baseline reproducibility (Property 13)
    - **Property 13: Baseline Reproducibility**
    - **Validates: Requirements 4.4**

  - [ ]* 6.4 Write unit tests for BaselineFeatureExtractor
    - Test all four methods, n-gram range config, transform on unseen texts
    - _Requirements: 4.1, 4.2, 4.3, 4.4_

- [x] 7. Implement CNNLSTMModel(Ruan)
  - [x] 7.1 Implement `CNNLSTMModel` in `src/model.py`
    - Embedding lookup `[B, T] → [B, T, D]` with dropout
    - Parallel `Conv1d` branches for each kernel size with ReLU + global max-over-time pooling → `[B, num_filters]` each
    - Concatenate multi-scale features, apply dropout, reshape for LSTM input
    - Stacked LSTM; take last-layer hidden state `h_n[-1]`
    - Dropout + `Linear` classification head → logits `[B, num_classes]`
    - Accept all hyperparameters via `ModelConfig`
    - _Requirements: 5.1, 5.2, 5.3, 5.4, 5.6_

  - [x]* 7.2 Write property test for CNN-LSTM output shape invariant (Property 5)
    - **Property 5: CNN-LSTM Output Shape Invariant**
    - **Validates: Requirements 5.1, 5.5**

  - [x]* 7.3 Write unit tests for CNNLSTMModel
    - Test output shape for various batch sizes and sequence lengths, no NaN in output
    - _Requirements: 5.1, 5.5_

- [ ] 8. Implement Trainer(all)
  - [ ] 8.1 Implement `Trainer` in `src/trainer.py`
    - Adam optimiser with gradient clipping `max_norm=1.0`
    - Per-epoch validation with macro-F1; save best checkpoint; reset patience counter on improvement
    - Early stopping when patience counter reaches configured value
    - Detect NaN loss → raise `TrainingDivergenceError` with epoch/batch info
    - Catch `torch.cuda.OutOfMemoryError` → halve batch size and retry epoch
    - Return `TrainingHistory` with per-epoch loss and validation metrics
    - _Requirements: 6.1, 6.2, 6.3, 6.4, 6.5, 6.6, 6.7, 6.8_

  - [ ]* 8.2 Write property test for training termination bound (Property 8)
    - **Property 8: Training Termination Bound**
    - **Validates: Requirements 6.4, 6.5**

  - [ ]* 8.3 Write property test for training history completeness (Property 17)
    - **Property 17: Training History Completeness**
    - **Validates: Requirements 6.8**

  - [ ]* 8.4 Write unit tests for Trainer
    - Test early stopping triggers, NaN loss raises `TrainingDivergenceError`, checkpoint saved on improvement
    - _Requirements: 6.3, 6.4, 6.6_

- [ ] 9. Checkpoint — Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

- [ ] 10. Implement Evaluator (mosa)
  - [ ] 10.1 Implement `evaluate()` in `src/evaluate.py`
    - Compute accuracy, macro-precision, macro-recall, macro-F1, per-class F1, confusion matrix using scikit-learn
    - Run inference in `torch.no_grad()` mode
    - Return `MetricsDict`
    - _Requirements: 7.1, 7.2, 7.3, 7.4, 7.5_

  - [ ]* 10.2 Write property test for metric bounds (Property 6)
    - **Property 6: Metric Bounds**
    - **Validates: Requirements 7.2**

  - [ ]* 10.3 Write property test for confusion matrix sum invariant (Property 7)
    - **Property 7: Confusion Matrix Sum Invariant**
    - **Validates: Requirements 7.3, 7.4**

  - [ ]* 10.4 Write unit tests for Evaluator
    - Test with known prediction/label pairs; verify hand-calculated metric values
    - _Requirements: 7.1, 7.2, 7.3, 7.4_

- [ ] 11. Implement ExplainabilityModule(mac)
  - [ ] 11.1 Implement `ExplainabilityModule` in `src/explainability.py`
    - `explain_shap()`: use `shap.DeepExplainer` or `shap.KernelExplainer` for token-level attributions; return list of `ShapExplanation` objects (one per input text)
    - `explain_lime()`: use `lime.lime_text.LimeTextExplainer` for local surrogate explanation
    - `error_analysis()`: filter misclassified samples, aggregate top-k influential tokens per author pair, identify highest-confusion author pairs; raise error if no misclassified samples
    - _Requirements: 8.1, 8.2, 8.3, 8.4, 8.5, 8.6_

  - [ ]* 11.2 Write property test for explainability coverage (Property 14)
    - **Property 14: Explainability Coverage**
    - **Validates: Requirements 8.3**

  - [ ]* 11.3 Write property test for SHAP explanation count (Property 16)
    - **Property 16: SHAP Explanation Count**
    - **Validates: Requirements 8.1**

  - [ ]* 11.4 Write unit tests for ExplainabilityModule
    - Test `error_analysis()` raises on zero misclassifications, top-k token extraction, confusion pair identification
    - _Requirements: 8.3, 8.4, 8.5, 8.6_

- [x] 12. Implement end-to-end pipeline and experiment scripts
  - [x] 12.1 Implement `experiments/run_cnn_lstm.py`
    - Wire DatasetLoader → Preprocessor → SubwordTokeniser → CNNLSTMModel → Trainer → Evaluator
    - Accept CLI args for dataset path, seed, model/training config overrides
    - Save metrics to `results/metrics.json`, tokeniser to `artifacts/tokeniser.json`, CNN-LSTM bundle per run under `artifacts/runs/<label>_<UTC>/` (`model.pt`, `tokeniser.json`, `training.json`), optional canonical promotion to `artifacts/best_model_bundle/` on strict validation F1 improvement
    - _Requirements: 9.1, 9.2, 9.3, 9.4, 9.6_

  - [x] 12.2 Implement `experiments/run_baselines.py`
    - Wire DatasetLoader → Preprocessor → BaselineFeatureExtractor → SVM/LogReg classifiers → Evaluator
    - Save baseline metrics to `results/metrics.json` alongside CNN-LSTM results
    - _Requirements: 9.2, 9.6_

  - [x]* 12.3 Write property test for pipeline reproducibility (Property 15)
    - **Property 15: Pipeline Reproducibility**
    - **Validates: Requirements 9.1**

  - [x]* 12.4 Write integration tests in `tests/test_pipeline.py`
    - Run full pipeline on synthetic dataset (50 samples, 5 authors); assert no errors and all metrics in `[0.0, 1.0]`
    - Test checkpoint save/load round-trip produces identical predictions
    - _Requirements: 9.1, 9.2, 9.3, 9.4, 9.5, 9.6_

- [ ] 13. Final checkpoint — Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

## Notes

- Tasks marked with `*` are optional and can be skipped for a faster MVP
- All property-based tests use the `hypothesis` library
- Each task references specific requirements for traceability
- Checkpoints ensure incremental validation at key milestones
- The design document uses Python throughout; all implementation is in Python
