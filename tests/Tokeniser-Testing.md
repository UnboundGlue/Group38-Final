# SubwordTokeniser Testing Summary

## ✅ Complete Test Coverage

The test file `tests/test_tokeniser.py` now includes **25 comprehensive tests** covering all methods and edge cases in `src/tokeniser.py`:

### Methods Tested:

1. **`train()`**
   - Empty texts validation
   - Vocab size validation (must be > 256)
   - Invalid algorithm validation
   - BPE algorithm (via fixture)
   - WordPiece algorithm

2. **`encode()`**
   - Padding short sequences
   - Truncating long sequences with warning
   - Unknown character handling
   - RuntimeError before training

3. **`batch_encode()`**
   - Correct shape [N, max_length]
   - Correct dtype (int64)
   - Single-element batch
   - RuntimeError before training

4. **`decode()`**
   - Text reconstruction
   - PAD token filtering
   - RuntimeError before training

5. **`vocab_size()`**
   - Returns positive integer
   - Approximately matches requested size
   - RuntimeError before training

6. **`save()`**
   - Creates parent directories
   - RuntimeError before training

7. **`load()`**
   - Round-trip consistency (save → load → same output)
   - Non-existent file handling

---

## Running the Tests

### Run all tokeniser tests:
```bash
python3 -m pytest tests/test_tokeniser.py -v
```

### Run specific test:
```bash
python3 -m pytest tests/test_tokeniser.py::test_encode_pads_short_text_to_max_length -v
```

### Run with coverage report:
```bash
python3 -m pytest tests/test_tokeniser.py --cov=src.tokeniser --cov-report=term-missing
```

**Result:** All 25 tests pass ✓

---

## Manual Testing with Real Dataset

The script `test_tokeniser_manual.py` demonstrates the tokeniser with the actual Chanchal et al. dataset:

### Run the manual test:
```bash
python3 test_tokeniser_manual.py
```

### What it does:
1. **Loads dataset** (downloads if missing) — 2,500 samples from 50 authors
2. **Trains BPE tokeniser** on 1,000 samples with vocab_size=5000
3. **Tests encoding** — converts text to token IDs with padding
4. **Tests decoding** — reconstructs text from token IDs
5. **Tests batch encoding** — processes multiple texts at once
6. **Tests persistence** — saves and loads tokeniser, verifies consistency
7. **Tests WordPiece** — alternative algorithm

**Result:** All manual tests pass ✓

---

## Test Results

```
============================= test session starts ==============================
platform darwin -- Python 3.10.11, pytest-8.3.5, pluggy-1.6.0
collected 25 items

tests/test_tokeniser.py::test_encode_pads_short_text_to_max_length PASSED [  4%]
tests/test_tokeniser.py::test_encode_padding_ids_are_zero PASSED         [  8%]
tests/test_tokeniser.py::test_encode_truncates_long_text_to_max_length PASSED [ 12%]
tests/test_tokeniser.py::test_encode_logs_warning_on_truncation PASSED   [ 16%]
tests/test_tokeniser.py::test_encode_handles_unknown_characters_without_error PASSED [ 20%]
tests/test_tokeniser.py::test_encode_unknown_characters_produce_unk_id PASSED [ 24%]
tests/test_tokeniser.py::test_save_load_round_trip_produces_same_ids PASSED [ 28%]
tests/test_tokeniser.py::test_encode_before_train_raises_runtime_error PASSED [ 32%]
tests/test_tokeniser.py::test_batch_encode_before_train_raises_runtime_error PASSED [ 36%]
tests/test_tokeniser.py::test_batch_encode_returns_ndarray_of_correct_shape PASSED [ 40%]
tests/test_tokeniser.py::test_batch_encode_dtype_is_int64 PASSED         [ 44%]
tests/test_tokeniser.py::test_batch_encode_single_text PASSED            [ 48%]
tests/test_tokeniser.py::test_train_empty_texts_raises_value_error PASSED [ 52%]
tests/test_tokeniser.py::test_train_vocab_size_too_small_raises_value_error PASSED [ 56%]
tests/test_tokeniser.py::test_train_invalid_algorithm_raises_value_error PASSED [ 60%]
tests/test_tokeniser.py::test_train_wordpiece_algorithm PASSED           [ 64%]
tests/test_tokeniser.py::test_decode_before_train_raises_runtime_error PASSED [ 68%]
tests/test_tokeniser.py::test_decode_reconstructs_text PASSED            [ 72%]
tests/test_tokeniser.py::test_decode_filters_pad_tokens PASSED           [ 76%]
tests/test_tokeniser.py::test_vocab_size_before_train_raises_runtime_error PASSED [ 80%]
tests/test_tokeniser.py::test_vocab_size_returns_positive_integer PASSED [ 84%]
tests/test_tokeniser.py::test_vocab_size_approximately_matches_requested PASSED [ 88%]
tests/test_tokeniser.py::test_save_before_train_raises_runtime_error PASSED [ 92%]
tests/test_tokeniser.py::test_save_creates_parent_directories PASSED     [ 96%]
tests/test_tokeniser.py::test_load_nonexistent_file_raises_exception PASSED [100%]

============================== 25 passed in 0.20s ==============================
```

---

## Files Modified/Created

1. **`tests/test_tokeniser.py`** — Updated with 13 additional tests
2. **`tests/test_tokeniser_manual.py`** — Manual test script using real dataset

Both files are ready for Task 4 validation.
