"""Summarise ``results/metrics.json`` for reporting: timings, split metrics, confusion.

Examples::

    python -m experiments.summarize_errors
    python -m experiments.summarize_errors --metrics results/metrics.json \\
        --dataset data/AuthorIdentification/Dataset/.../200_tweets_per_user.csv

**Split names:** *train* / *validation* / *test* correspond to the stratified
70/15/15 split from :class:`src.dataset.DatasetLoader` — there is no separate
\"post-test\" holdout in this codebase; *test* is the final held-out evaluation.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _load_metrics(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_author_map(dataset_path: str | None) -> dict[int, str]:
    if not dataset_path:
        return {}
    root = _repo_root()
    sys.path.insert(0, str(root))
    from src.dataset import DatasetLoader

    p = Path(dataset_path)
    if not p.is_file():
        tried = root / dataset_path
        if tried.is_file():
            p = tried
        else:
            return {}
    loader = DatasetLoader()
    loader.load(str(p.resolve()), fetch_if_missing=False)
    return dict(loader.author_map)


def _is_legacy_baseline_leaf(leaf: dict) -> bool:
    return "accuracy" in leaf and "splits" not in leaf


def _baseline_matrix_is_legacy(baselines: dict) -> bool:
    """True if any baseline row uses the pre-schema-v2 flat test metrics dict."""
    for clf_map in baselines.values():
        if not isinstance(clf_map, dict):
            continue
        for leaf in clf_map.values():
            if isinstance(leaf, dict) and _is_legacy_baseline_leaf(leaf):
                return True
    return False


def _print_timings(data: dict) -> None:
    ts = data.get("timings_seconds")
    if not ts:
        print("\n## Timings\n(not present — re-run `run_cnn_lstm` to record wall times)\n")
        return
    print("\n## Timings (wall clock, seconds)\n")
    if "cnn_training_wall" in ts:
        print(f"  CNN-LSTM training:              {ts['cnn_training_wall']:.2f}")
        print(f"  CNN-LSTM eval (train split):    {ts['cnn_eval_train_wall']:.4f}")
        print(f"  CNN-LSTM eval (validation):     {ts['cnn_eval_validation_wall']:.4f}")
        print(f"  CNN-LSTM eval (test):           {ts['cnn_eval_test_wall']:.4f}")
    if ts.get("baselines_cpu_wall") is not None:
        print(f"  Baselines block (CPU, total):   {ts['baselines_cpu_wall']:.2f}")
    det = ts.get("baselines_detail") or {}
    rows = det.get("rows") if isinstance(det, dict) else None
    if isinstance(rows, dict):
        print("\n  Per baseline row (sparse features + classifier fit + 3× predict):")
        for name, timing in sorted(rows.items()):
            if isinstance(timing, dict):
                tot = timing.get("row_total", timing)
                print(f"    {name:28s}  total={float(tot):.3f}s")
    print()


def _print_cnn_splits(data: dict) -> None:
    sp = data.get("cnn_lstm_splits")
    if not sp:
        print("\n## CNN-LSTM by split\n(only headline `cnn_lstm` test metrics — re-run pipeline for train/val/test table)\n")
        return
    print("\n## CNN-LSTM — accuracy / macro-F1 by split\n")
    print(f"{'split':14s}  {'accuracy':>10s}  {'precision_macro':>16s}  {'recall_macro':>14s}  {'f1_macro':>10s}")
    for key in ("train", "validation", "test"):
        m = sp.get(key) or {}
        print(
            f"{key:14s}  {m.get('accuracy', 0):10.4f}  "
            f"{m.get('precision_macro', 0):16.4f}  {m.get('recall_macro', 0):14.4f}  "
            f"{m.get('f1_macro', 0):10.4f}"
        )
    print("\n  (*Train* scores are in-sample predictions — optimistic vs test.)\n")


def _print_baselines_table(data: dict) -> None:
    baselines = data.get("baselines")
    if not baselines:
        print("\n## Baselines\n(missing)\n")
        return
    print("\n## Sparse baselines — accuracy / macro-F1\n")
    if _baseline_matrix_is_legacy(baselines):
        print(f"{'method':14s}  {'clf':8s}  {'acc':>8s}  {'f1_macro':>10s}  (test only, legacy file)\n")
        for method, clf_map in baselines.items():
            for clf, m in clf_map.items():
                if isinstance(m, dict) and "accuracy" in m:
                    print(f"{method:14s}  {clf:8s}  {m['accuracy']:8.4f}  {m['f1_macro']:10.4f}")
        print()
        return

    print(
        f"{'method':14s}  {'clf':8s}  "
        f"{'tr_acc':>7s} {'tr_f1':>7s}  {'val_acc':>7s} {'val_f1':>7s}  {'te_acc':>7s} {'te_f1':>7s}  {'row_s':>7s}\n"
    )
    for method, clf_map in baselines.items():
        for clf, payload in clf_map.items():
            if not isinstance(payload, dict):
                continue
            if "error" in payload:
                print(f"{method:14s}  {clf:8s}  ERROR: {payload['error'][:50]}")
                continue
            spl = payload.get("splits") or {}
            tr, va, te = spl.get("train", {}), spl.get("validation", {}), spl.get("test", {})
            tsec = (payload.get("timing_seconds") or {}).get("row_total", 0.0)
            print(
                f"{method:14s}  {clf:8s}  "
                f"{tr.get('accuracy', 0):7.3f} {tr.get('f1_macro', 0):7.3f}  "
                f"{va.get('accuracy', 0):7.3f} {va.get('f1_macro', 0):7.3f}  "
                f"{te.get('accuracy', 0):7.3f} {te.get('f1_macro', 0):7.3f}  "
                f"{float(tsec):7.2f}"
            )
    print()


def _print_confusion_analysis(
    data: dict,
    id2name: dict[int, str],
    *,
    top_pairs: int,
    worst_f1: int,
) -> None:
    cnn = data.get("cnn_lstm")
    if not cnn or "confusion_matrix" not in cnn:
        print("\n## Error analysis\n(no CNN confusion matrix)\n")
        return

    import numpy as np

    cm = np.array(cnn["confusion_matrix"], dtype=int)
    f1s = {int(k): float(v) for k, v in (cnn.get("f1_per_class") or {}).items()}
    n_class = cm.shape[0]

    print("\n## CNN-LSTM test — hardest classes (lowest per-class F1)\n")
    worst = sorted(f1s.keys(), key=lambda i: f1s[i])[:worst_f1]
    for i in worst:
        name = id2name.get(i, "?")
        print(f"  id {i:3d}  F1={f1s[i]:.3f}  author={name}")

    print(f"\n## CNN-LSTM test — top {top_pairs} off-diagonal confusion counts (true → pred)\n")
    pairs: list[tuple[int, int, int]] = []
    for i in range(n_class):
        for j in range(n_class):
            if i != j and cm[i, j] > 0:
                pairs.append((int(cm[i, j]), i, j))
    pairs.sort(reverse=True)
    for cnt, i, j in pairs[:top_pairs]:
        ti = id2name.get(i, str(i))
        pj = id2name.get(j, str(j))
        print(f"  {cnt:3d}×  true={i:2d} ({ti}) → pred={j:2d} ({pj})")
    print()


def main() -> int:
    root = _repo_root()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--metrics",
        type=Path,
        default=root / "results" / "metrics.json",
        help="Path to metrics JSON (from run_cnn_lstm or run_baselines).",
    )
    ap.add_argument(
        "--dataset",
        default=None,
        help="Optional CSV path to map class ids → author ids (same file as training).",
    )
    ap.add_argument("--top-pairs", type=int, default=12, help="Confusion pairs to list.")
    ap.add_argument("--worst-f1", type=int, default=10, help="Lowest per-class F1 count.")
    args = ap.parse_args()

    path = args.metrics
    if not path.is_file():
        print(f"Metrics file not found: {path}", file=sys.stderr)
        return 1

    data = _load_metrics(path)
    ver = data.get("schema_version", 1)
    print(f"# Metrics summary  (path={path}, schema_version={ver})\n")

    id2name = _load_author_map(args.dataset)
    if args.dataset and not id2name:
        print(f"(Warning: could not load author map from {args.dataset})\n")

    _print_timings(data)
    _print_cnn_splits(data)
    _print_baselines_table(data)
    if "cnn_lstm" in data:
        _print_confusion_analysis(
            data, id2name, top_pairs=args.top_pairs, worst_f1=args.worst_f1
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
