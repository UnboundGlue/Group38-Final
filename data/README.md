# Data

The paper dataset lives in the public [AuthorIdentification](https://github.com/chanchalIITP/AuthorIdentification) repository.

## Fetch the dataset (recommended)

**From code** (`src/dataset.py`), clone if missing then load:

```python
from src.dataset import DatasetLoader, DEFAULT_CHANCHAL_CSV

loader = DatasetLoader()
texts, labels = loader.load(DEFAULT_CHANCHAL_CSV, fetch_if_missing=True)
```

Or call `ensure_authoridentification_dataset()` yourself before `load()`.

**From the shell**, the same logic is available as:

```bash
python data/fetch_dataset.py
```

This checks for `data/AuthorIdentification/Dataset/` with CSV files. If they are missing, it **shallow-clones** that GitHub repository into `data/AuthorIdentification/`. You need **Git** installed and on your `PATH`.

Experiments also accept `--fetch-dataset` when the `--dataset` path is not present yet.

Options:

- `--output PATH` — clone or verify a different directory
- `--force` — remove a broken or partial `data/AuthorIdentification` and clone again

## Manual copy

You can also copy a single CSV into this folder (e.g. `200_tweets_per_user.csv`) and pass it to the experiments. Files matching `*.csv` here are git-ignored by default.

## Example experiment

From the project root, after a successful fetch the CSVs are under `data/AuthorIdentification/Dataset/...`:

```bash
python -m experiments.run_cnn_lstm --dataset data/AuthorIdentification/Dataset/Dataset_with_varying_number_of_tweets/200_tweets_per_user.csv
```

Use a smaller file (e.g. `50_tweets_per_user.csv`) for quicker runs.
