"""Persist and describe self-contained CNN-LSTM training bundles.

Each bundle is a directory with::

    model.pt         # torch state_dict written by training
    tokeniser.json   # copy aligned with embeddings
    training.json    # metrics trace; ``seeds`` + ``tuning`` + ``model_config`` + ``cli_args``

Each training invocation should use a fresh ``artifacts/runs/<label>_<UTC>/``
folder: the trainer overwrites only ``model.pt`` inside that run when a better
checkpoint is found. :func:`save_run_bundle` materialises ``model.pt``,
``tokeniser.json``, and ``training.json`` there (skipping redundant copies when
the checkpoint already lives at ``dest/model.pt``).

For a **single canonical** directory shared across runs (e.g.
``artifacts/best_model_bundle``), use :func:`promote_bundle_if_improved` — it
overwrites only when the new run's validation macro-F1 **strictly** beats the
value recorded in the incumbent ``training.json`` (or nothing valid is stored yet).
"""

from __future__ import annotations

import json
import shutil
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import argparse

from .models import ModelConfig, TrainingHistory

CNN_TUNING_CLI_KEYS: tuple[str, ...] = (
    # Data
    "dataset",
    "fetch_dataset",
    # Architecture (mirror CLI names; overlaps model_config snapshot)
    "vocab_size",
    "embed_dim",
    "num_filters",
    "lstm_hidden",
    "lstm_layers",
    "dropout",
    "max_seq_len",
    # Training loop / optimisation
    "epochs",
    "batch_size",
    "num_workers",
    "lr",
    "patience",
    "weight_decay",
    "label_smoothing",
    "lr_schedule",
    "cosine_t0",
    "onecycle_max_lr",
    "onecycle_div_factor",
    "onecycle_pct_start",
    "onecycle_final_div_factor",
    "class_weight",
    # Embedding init / throughput toggles passed through CLI
    "fasttext_vec",
    "fasttext_limit",
    "freeze_pretrained",
    "no_compile",
    "compile_mode",
    "no_fused_adam",
    "no_amp",
    "checkpoint",
    "run_label",
    "promote_best_dir",
    "no_promote_best",
    "no_live_plot",
    "no_tensorboard_server",
    "tensorboard_port",
    "no_tensorboard_browser",
)


def build_seeds_record(
    *,
    split_seed_used: int,
    cnn_train_seed: int,
    cli_namespace: argparse.Namespace | None,
) -> dict[str, Any]:
    """Who asked for RNG / splits (resolved + CLI echo for ensembles)."""
    out: dict[str, Any] = {
        "split_seed": split_seed_used,
        "cnn_train_seed": cnn_train_seed,
    }
    if cli_namespace is None:
        return out
    out["cli_default_seed"] = getattr(cli_namespace, "seed", None)
    ss = getattr(cli_namespace, "split_seed", None)
    if ss is not None:
        out["cli_split_seed"] = ss
    ens = getattr(cli_namespace, "ensemble_train_seeds", None)
    if ens is not None and str(ens).strip():
        out["ensemble_train_seeds"] = str(ens).strip()
    return out


def history_best_val_f1_macro(history: TrainingHistory) -> float | None:
    """Highest validation macro-F1 seen in *history* (same criterion as checkpoints)."""
    if not history.val_metrics:
        return None
    return max(m.f1_macro for m in history.val_metrics)


def read_bundle_best_val_f1(bundle_dir: str | Path) -> float | None:
    """Read ``best_val_f1_macro`` from *bundle_dir*/training.json, if valid."""
    p = Path(bundle_dir) / "training.json"
    if not p.is_file():
        return None
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        raw = data.get("best_val_f1_macro")
        return float(raw) if raw is not None else None
    except (OSError, json.JSONDecodeError, TypeError, ValueError):
        return None


def promote_bundle_if_improved(
    dest_dir: str | Path,
    *,
    candidate_best_val_f1: float,
    checkpoint_src: str | Path,
    tokeniser_src: str | Path,
    record: dict[str, Any],
) -> tuple[str, float | None]:
    """Overwrite *dest_dir* with a new bundle only if *candidate* beats the incumbent.

    Comparison uses **strict** inequality so equal F1 keeps the older bundle.

    Returns ``("promoted", previous_f1_or_none)`` or ``("kept_incumbent", previous_f1)``.
    """
    dest = Path(dest_dir)
    prev = read_bundle_best_val_f1(dest)
    if prev is None or candidate_best_val_f1 > prev:
        save_run_bundle(
            dest,
            checkpoint_src=checkpoint_src,
            tokeniser_src=tokeniser_src,
            record=record,
        )
        return "promoted", prev
    return "kept_incumbent", prev


def cli_namespace_to_plain_dict(ns: argparse.Namespace) -> dict[str, Any]:
    """JSON-friendly snapshot of argparse flags (training reproducibility metadata)."""
    out: dict[str, Any] = {}
    for k, v in vars(ns).items():
        if k.startswith("_"):
            continue
        if isinstance(v, Path):
            out[k] = str(v)
        elif isinstance(v, (str, int, float, bool, type(None))):
            out[k] = v
        elif isinstance(v, (list, tuple)):
            out[k] = [str(x) if isinstance(x, Path) else x for x in v]
        elif isinstance(v, dict):
            out[k] = {
                str(sk): str(sv) if isinstance(sv, Path) else sv for sk, sv in v.items()
            }
        else:
            out[k] = str(v)
    return out


def build_tuning_record(cli_namespace: argparse.Namespace | None) -> dict[str, Any]:
    """Subset of CLI flags that matter for reproducing training decisions."""
    if cli_namespace is None:
        return {}
    plain = cli_namespace_to_plain_dict(cli_namespace)
    return {k: plain[k] for k in CNN_TUNING_CLI_KEYS if k in plain}


def build_training_record(
    *,
    history: TrainingHistory,
    training_wall_seconds: float,
    lr_schedule: str,
    split_seed: int,
    train_seed: int,
    cli_namespace: argparse.Namespace | None,
    model_config: ModelConfig,
    checkpoint_path_written: str,
    primary_lr_arg: float,
) -> dict[str, Any]:
    """Produce the ``training.json`` payload."""
    nv = len(history.val_metrics)
    nt = len(history.train_losses)
    if nv > 0 and nt > 0:
        n_epochs = min(nv, nt)
    else:
        n_epochs = nv or nt

    best_i = (
        max(range(nv), key=lambda i: history.val_metrics[i].f1_macro)
        if nv > 0
        else None
    )
    per_epoch: list[dict[str, Any]] = []
    for i in range(n_epochs):
        lr_ep = (
            history.lr_per_epoch[i] if i < len(history.lr_per_epoch) else None
        )
        if i < nv:
            vm = history.val_metrics[i]
            val_acc = vm.accuracy
            val_f1 = vm.f1_macro
        else:
            val_acc = None
            val_f1 = None
        train_loss_ep = (
            history.train_losses[i] if i < nt else None
        )
        per_epoch.append(
            {
                "epoch": i + 1,
                "train_loss": train_loss_ep,
                "val_accuracy": val_acc,
                "val_f1_macro": val_f1,
                "lr": lr_ep,
            }
        )

    cli_args = cli_namespace_to_plain_dict(cli_namespace) if cli_namespace else {}

    record: dict[str, Any] = {
        "schema_version": 2,
        "created_iso_utc": datetime.now(timezone.utc).isoformat(),
        "split_seed": split_seed,
        "cnn_train_seed": train_seed,
        "seeds": build_seeds_record(
            split_seed_used=split_seed,
            cnn_train_seed=train_seed,
            cli_namespace=cli_namespace,
        ),
        "tuning": build_tuning_record(cli_namespace),
        "cnn_training_wall_seconds": training_wall_seconds,
        "checkpoint_path_during_run": checkpoint_path_written,
        "lr_schedule": lr_schedule,
        "initial_lr_cli": primary_lr_arg,
        "epochs_completed": n_epochs,
        "per_epoch": per_epoch,
        "model_config": asdict(model_config),
        "cli_args": cli_args,
    }

    if nv > 0 and best_i is not None:
        record["best_val_f1_macro"] = history.val_metrics[best_i].f1_macro
        record["best_epoch"] = best_i + 1
        if nt > 0:
            tail = min(n_epochs, nt) - 1
            if tail >= 0:
                record["final_train_loss"] = history.train_losses[tail]
        if nv > 0:
            vtail = min(n_epochs, nv) - 1
            if vtail >= 0:
                record["final_val_f1_macro"] = history.val_metrics[vtail].f1_macro
        record["final_lr_logged"] = (
            history.lr_per_epoch[n_epochs - 1]
            if len(history.lr_per_epoch) >= n_epochs
            else primary_lr_arg
        )
    return record


def save_run_bundle(
    dest_dir: str | Path,
    *,
    checkpoint_src: str | Path,
    tokeniser_src: str | Path,
    record: dict[str, Any],
) -> Path:
    """Write ``model.pt``, ``tokeniser.json``, ``training.json`` under *dest_dir*."""
    d = Path(dest_dir)
    d.mkdir(parents=True, exist_ok=True)
    dest_model = d / "model.pt"
    src_ckpt = Path(checkpoint_src).resolve()
    if src_ckpt != dest_model.resolve():
        shutil.copy2(checkpoint_src, dest_model)

    dest_tok = d / "tokeniser.json"
    src_tok = Path(tokeniser_src).resolve()
    if src_tok != dest_tok.resolve():
        shutil.copy2(tokeniser_src, dest_tok)
    (d / "training.json").write_text(
        json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return d.resolve()
