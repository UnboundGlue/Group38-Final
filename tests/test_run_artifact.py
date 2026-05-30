"""tests for persisted CNN-LSTM run bundles."""

from __future__ import annotations

import json

import numpy as np

from src.models import MetricsDict, ModelConfig, TrainingHistory
from src.run_artifact import (
    build_training_record,
    history_best_val_f1_macro,
    promote_bundle_if_improved,
    read_bundle_best_val_f1,
    save_run_bundle,
)


def _stub_metrics(f1_macro: float) -> MetricsDict:
    cm = np.eye(2, dtype=np.int64)
    return MetricsDict(
        accuracy=0.5,
        precision_macro=0.5,
        recall_macro=0.5,
        f1_macro=f1_macro,
        f1_per_class={0: 0.5, 1: 0.5},
        confusion_matrix=cm,
    )


def test_history_best_val_f1_macro_matches_max_epoch() -> None:
    h = TrainingHistory(
        val_metrics=[_stub_metrics(0.15), _stub_metrics(0.72)],
    )
    assert history_best_val_f1_macro(h) == 0.72


def test_read_bundle_best_val_f1(tmp_path) -> None:
    ckpt = tmp_path / "fake.pt"
    ckpt.write_bytes(b"x")
    tok = tmp_path / "tok.json"
    tok.write_text("{}")
    hist = TrainingHistory(val_metrics=[_stub_metrics(0.88)])
    cfg = ModelConfig(vocab_size=4, num_classes=2)
    rec = build_training_record(
        history=hist,
        training_wall_seconds=1.0,
        lr_schedule="none",
        split_seed=0,
        train_seed=1,
        cli_namespace=None,
        model_config=cfg,
        checkpoint_path_written="z.pt",
        primary_lr_arg=1e-3,
    )
    dest = tmp_path / "b"
    save_run_bundle(dest, checkpoint_src=ckpt, tokeniser_src=tok, record=rec)
    assert read_bundle_best_val_f1(dest) == 0.88


def test_promote_bundle_fresh_dir(tmp_path) -> None:
    dest = tmp_path / "promo"
    ckpt = tmp_path / "w.pt"
    ckpt.write_bytes(b"W")
    tok = tmp_path / "t.json"
    tok.write_text("{}")
    hist = TrainingHistory(val_metrics=[_stub_metrics(0.61)])
    cfg = ModelConfig(vocab_size=4, num_classes=2)
    rec = build_training_record(
        history=hist,
        training_wall_seconds=1.0,
        lr_schedule="none",
        split_seed=0,
        train_seed=1,
        cli_namespace=None,
        model_config=cfg,
        checkpoint_path_written="z.pt",
        primary_lr_arg=1e-3,
    )
    st, prev = promote_bundle_if_improved(
        dest,
        candidate_best_val_f1=float(rec["best_val_f1_macro"]),
        checkpoint_src=ckpt,
        tokeniser_src=tok,
        record=rec,
    )
    assert st == "promoted" and prev is None
    assert (dest / "model.pt").read_bytes() == b"W"


def test_promote_keeps_incumbent_on_tie_or_weaker(tmp_path) -> None:
    dest = tmp_path / "promo"
    incumbent_ckpt = tmp_path / "old.pt"
    incumbent_ckpt.write_bytes(b"OLD")
    tok = tmp_path / "t.json"
    tok.write_text("{}")
    strong = TrainingHistory(val_metrics=[_stub_metrics(0.90)])
    cfg = ModelConfig(vocab_size=4, num_classes=2)
    rec0 = build_training_record(
        history=strong,
        training_wall_seconds=1.0,
        lr_schedule="none",
        split_seed=0,
        train_seed=1,
        cli_namespace=None,
        model_config=cfg,
        checkpoint_path_written="z.pt",
        primary_lr_arg=1e-3,
    )
    save_run_bundle(
        dest,
        checkpoint_src=incumbent_ckpt,
        tokeniser_src=tok,
        record=rec0,
    )

    weak_ckpt = tmp_path / "new.pt"
    weak_ckpt.write_bytes(b"NEW")
    weak_hist = TrainingHistory(val_metrics=[_stub_metrics(0.90)])
    rec1 = build_training_record(
        history=weak_hist,
        training_wall_seconds=2.0,
        lr_schedule="cosine_restarts",
        split_seed=3,
        train_seed=9,
        cli_namespace=None,
        model_config=cfg,
        checkpoint_path_written="w.pt",
        primary_lr_arg=1e-4,
    )

    st_equal, prev = promote_bundle_if_improved(
        dest,
        candidate_best_val_f1=float(rec1["best_val_f1_macro"]),
        checkpoint_src=weak_ckpt,
        tokeniser_src=tok,
        record=rec1,
    )
    assert st_equal == "kept_incumbent" and prev == 0.90
    assert (dest / "model.pt").read_bytes() == b"OLD"

    weaker_hist = TrainingHistory(val_metrics=[_stub_metrics(0.40)])
    rec2 = build_training_record(
        history=weaker_hist,
        training_wall_seconds=2.0,
        lr_schedule="none",
        split_seed=3,
        train_seed=9,
        cli_namespace=None,
        model_config=cfg,
        checkpoint_path_written="w.pt",
        primary_lr_arg=1e-4,
    )
    st_weak, prev2 = promote_bundle_if_improved(
        dest,
        candidate_best_val_f1=float(rec2["best_val_f1_macro"]),
        checkpoint_src=weak_ckpt,
        tokeniser_src=tok,
        record=rec2,
    )
    assert st_weak == "kept_incumbent" and prev2 == 0.90


def test_save_run_bundle_writes_files(tmp_path) -> None:
    ckpt = tmp_path / "fake.pt"
    ckpt.write_bytes(b"checkpoint-bytes")
    tok = tmp_path / "tok.json"
    tok.write_text('{"pieces": ["a"]}')
    hist = TrainingHistory(
        train_losses=[2.1, 1.9],
        val_metrics=[_stub_metrics(0.3), _stub_metrics(0.35)],
        lr_per_epoch=[1e-3, 5e-4],
    )
    cfg = ModelConfig(vocab_size=100, embed_dim=8, num_classes=7)
    rec = build_training_record(
        history=hist,
        training_wall_seconds=12.34,
        lr_schedule="plateau",
        split_seed=7,
        train_seed=42,
        cli_namespace=None,
        model_config=cfg,
        checkpoint_path_written="artifacts/x.pt",
        primary_lr_arg=1e-3,
    )
    assert rec["schema_version"] == 2
    assert rec["seeds"] == {"split_seed": 7, "cnn_train_seed": 42}
    assert rec["tuning"] == {}
    dest = tmp_path / "bundle"
    save_run_bundle(dest, checkpoint_src=ckpt, tokeniser_src=tok, record=rec)
    assert (dest / "model.pt").read_bytes() == b"checkpoint-bytes"
    raw = json.loads((dest / "training.json").read_text(encoding="utf-8"))
    assert raw["per_epoch"][-1]["lr"] == 5e-4
    assert raw["seeds"]["cnn_train_seed"] == 42


def test_save_run_bundle_same_dest_skips_weight_copy(tmp_path) -> None:
    """When checkpoint already lives at dest/model.pt, avoid shutil.copy2 onto itself."""
    bundle = tmp_path / "run"
    bundle.mkdir()
    model_path = bundle / "model.pt"
    model_path.write_bytes(b"weights")
    tok_global = tmp_path / "tok.json"
    tok_global.write_text("{}")
    hist = TrainingHistory(val_metrics=[_stub_metrics(0.4)])
    cfg = ModelConfig(vocab_size=10, num_classes=3)
    rec = build_training_record(
        history=hist,
        training_wall_seconds=0.1,
        lr_schedule="none",
        split_seed=1,
        train_seed=2,
        cli_namespace=None,
        model_config=cfg,
        checkpoint_path_written=str(model_path),
        primary_lr_arg=1e-3,
    )
    save_run_bundle(bundle, checkpoint_src=model_path, tokeniser_src=tok_global, record=rec)
    assert model_path.read_bytes() == b"weights"
    assert (bundle / "training.json").is_file()
    assert (bundle / "tokeniser.json").is_file()


def test_training_record_seeds_and_tuning_echo_cli() -> None:
    import argparse

    ns = argparse.Namespace(
        seed=100,
        split_seed=200,
        ensemble_train_seeds="42,43,44",
        lr=0.002,
        epochs=80,
        batch_size=64,
        patience=24,
        dataset="data/sample.csv",
        fetch_dataset=True,
        vocab_size=10000,
        weight_decay=0.0002,
        lr_schedule="plateau",
        class_weight="balanced",
        no_amp=False,
        no_compile=True,
        checkpoint="artifacts/checkpoints/model.pt",
        run_label="experiment_a",
        promote_best_dir="artifacts/best_model_bundle",
        no_promote_best=False,
        # keys not in tuning whitelist are only in cli_args
        save_run="cnn",
    )
    hist = TrainingHistory(val_metrics=[_stub_metrics(0.5)])
    cfg = ModelConfig(vocab_size=10000, embed_dim=256, num_classes=50)
    resolved_split = 42
    rec = build_training_record(
        history=hist,
        training_wall_seconds=1.0,
        lr_schedule="plateau",
        split_seed=resolved_split,
        train_seed=42,
        cli_namespace=ns,
        model_config=cfg,
        checkpoint_path_written="x.pt",
        primary_lr_arg=float(ns.lr),
    )
    assert rec["seeds"]["split_seed"] == resolved_split
    assert rec["seeds"]["cnn_train_seed"] == 42
    assert rec["seeds"]["cli_default_seed"] == 100
    assert rec["seeds"]["cli_split_seed"] == 200
    assert rec["seeds"]["ensemble_train_seeds"] == "42,43,44"
    assert rec["tuning"]["lr"] == 0.002
    assert rec["tuning"]["epochs"] == 80
    assert rec["tuning"]["batch_size"] == 64
    assert rec["tuning"]["checkpoint"] == "artifacts/checkpoints/model.pt"
    assert rec["tuning"]["run_label"] == "experiment_a"
    assert rec["tuning"]["promote_best_dir"] == "artifacts/best_model_bundle"
    assert rec["tuning"]["no_promote_best"] is False
    assert "save_run" not in rec["tuning"]
    assert rec["cli_args"]["save_run"] == "cnn"
