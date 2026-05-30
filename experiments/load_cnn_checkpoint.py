"""Load a saved CNN-LSTM run bundle and optionally evaluate on a dataset.

Each bundle directory contains::
    model.pt, tokeniser.json, training.json

Inference only: **loading a bundle never trains** the network. ``--eval`` runs forward
passes on held-out style splits; add ``--eval-full-dataset`` for a **single** accuracy /
macro-F1 over **all** rows after preprocessing (no train/val/test split — useful for a
“whole file” sanity check; for comparability to training paper metrics, prefer the default
split mirrors ``split_seed``).

**Data path** resolution (evaluation): ``--dataset`` > ``--preset-dataset`` >
path stored in the bundle's ``training.json`` ``cli_args`` > default Chanchal
``200_tweets_per_user.csv`` (see ``src.dataset.DEFAULT_CHANCHAL_200_CSV``).
Specifying ``--dataset`` or ``--preset-dataset`` enables evaluation even if you
omit ``--eval``.

When ``--artifact`` points to an ensemble **run root** containing
``ensemble_index.json`` (from ``--ensemble-train-seeds``), every
``member_train*/model.pt`` is loaded and metrics use **majority vote** (tie-break:
mean logit), matching training evaluation. Promoted/canonical bundles remain a single
directory with ``model.pt`` — use ``--promoted-best`` / ``--promote-best-dir`` or
``--artifact`` on one ``member_train*`` bundle for only that member.

Usage::

    python -m experiments.load_cnn_checkpoint --promoted-best --eval --eval-full-dataset --fetch-dataset
    python -m experiments.load_cnn_checkpoint --artifact artifacts/runs/my_run_20260101_120000 \\
        --eval --preset-dataset chanchal_50
    python -m experiments.load_cnn_checkpoint --artifact path/to/bundle \\
        --eval --dataset path/to/other.csv --fetch-dataset
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.dataset import DatasetLoader, resolve_evaluation_dataset_path
from src.evaluate import ensemble_evaluate_majority_vote, evaluate, metrics_dict_to_jsonable
from src.model import CNNLSTMModel
from src.models import ModelConfig
from src.preprocessing import Preprocessor
from src.tokeniser import SubwordTokeniser
from src import training_hardware

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def _make_eval_loader(
    token_ids: np.ndarray,
    labels: list[int],
    batch_size: int,
    *,
    num_workers: int,
    pin_memory: bool,
) -> DataLoader:
    x = torch.tensor(token_ids, dtype=torch.long)
    y = torch.tensor(labels, dtype=torch.long)
    kw: dict = {
        "batch_size": batch_size,
        "shuffle": False,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
    }
    if num_workers > 0:
        kw["persistent_workers"] = True
        kw["prefetch_factor"] = 4
    return DataLoader(TensorDataset(x, y), **kw)


def load_bundle(
    artifact_dir: Path,
    *,
    device: torch.device | None = None,
) -> tuple[dict, CNNLSTMModel, torch.device]:
    """Load ``training.json`` metadata, reconstructed model + weights."""
    root = artifact_dir.expanduser().resolve()
    meta_path = root / "training.json"
    ckpt_path = root / "model.pt"
    tok_path = root / "tokeniser.json"
    for p in (meta_path, ckpt_path, tok_path):
        if not p.is_file():
            raise FileNotFoundError(f"Missing bundle file: {p}")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    cfg_dict = meta.get("model_config") or {}
    cfg = ModelConfig(**cfg_dict)
    dev = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CNNLSTMModel(cfg).to(dev)
    state = torch.load(str(ckpt_path), map_location=dev)
    model.load_state_dict(state)
    model.eval()
    return meta, model, dev


def load_tokeniser(artifact_dir: Path) -> SubwordTokeniser:
    root = artifact_dir.expanduser().resolve()
    tok_path = root / "tokeniser.json"
    if not tok_path.is_file():
        raise FileNotFoundError(f"Missing tokeniser: {tok_path}")
    tk = SubwordTokeniser()
    tk.load(str(tok_path))
    return tk


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    artifact_group = p.add_mutually_exclusive_group(required=True)
    artifact_group.add_argument(
        "--artifact",
        type=Path,
        default=None,
        help="Bundle directory containing model.pt, tokeniser.json, training.json.",
    )
    artifact_group.add_argument(
        "--promoted-best",
        action="store_true",
        help=(
            "Load the canonical best bundle (--promote-best-dir; default artifacts/best_model_bundle), "
            "updated during training only when validation macro-F1 strictly improves."
        ),
    )
    p.add_argument(
        "--promote-best-dir",
        type=str,
        default="artifacts/best_model_bundle",
        dest="promote_best_dir",
        help=(
            "Directory paired with --promoted-best only. Default matches experiments.run_cnn_lstm "
            "``--promote-best-dir``. Use the same path you passed when training if not default."
        ),
    )
    p.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="CSV/JSON path (overrides --preset-dataset and bundle metadata). Enables --eval if set.",
    )
    p.add_argument(
        "--preset-dataset",
        choices=("chanchal_50", "chanchal_200"),
        default=None,
        help=(
            "Use a documented repo-relative Chanchal slice (same paths as DEFAULT_CHANCHAL_* in "
            "src/dataset.py). Enables --eval if set."
        ),
    )
    p.add_argument(
        "--split-seed",
        type=int,
        default=None,
        dest="split_seed",
        help="Stratified split seed (default: split_seed from training.json).",
    )
    p.add_argument("--fetch-dataset", action="store_true", help="Pass through to DatasetLoader.")
    p.add_argument(
        "--eval",
        action="store_true",
        help=(
            "Run inference only (no training): preprocess, rebuild the stratified train/val/test "
            "split (split_seed from bundle unless --split-seed), and print metrics per split."
        ),
    )
    p.add_argument(
        "--eval-full-dataset",
        action="store_true",
        dest="eval_full_dataset",
        help=(
            "With --eval: skip splitting — score every row in the resolved dataset as one pool "
            "(single full_dataset accuracy / macro-F1). Implies --eval."
        ),
    )
    p.add_argument(
        "--metrics-out",
        type=str,
        default=None,
        help="Optional path to write eval metrics JSON (only with --eval).",
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=0,
        dest="batch_size",
        help="0 = heuristic from GPU/CPU like the training driver.",
    )
    p.add_argument(
        "--num-workers",
        type=int,
        default=-1,
        dest="num_workers",
        help="Ignored during --eval (always 0): avoids Windows spawn workers re-importing SciPy DLLs.",
    )
    p.add_argument(
        "--no-amp",
        action="store_true",
        dest="no_amp",
        help="Disable AMP on CUDA during evaluation.",
    )
    p.add_argument(
        "--show-meta",
        action="store_true",
        help="Print a short summary from training.json and exit (no torch forward).",
    )
    return p


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    if getattr(args, "promoted_best", False):
        artifact = Path(args.promote_best_dir).expanduser()
    elif args.artifact is not None:
        artifact = args.artifact
    else:
        parser.error("Provide --artifact DIR or --promoted-best.")

    artifact = artifact.expanduser().resolve()

    if (
        (args.dataset is not None or args.preset_dataset is not None)
        and not args.show_meta
        and not args.eval
    ):
        args.eval = True
        logger.info("Implied --eval because --dataset and/or --preset-dataset were set.")
    if getattr(args, "eval_full_dataset", False):
        args.eval = True

    ens_path = artifact / "ensemble_index.json"
    is_ensemble_root = ens_path.is_file()
    ensemble_index: dict | None = None
    member_roots: list[tuple[int, Path]] = []

    if is_ensemble_root:
        ensemble_index = json.loads(ens_path.read_text(encoding="utf-8"))
        members = ensemble_index.get("members") or []
        if not members:
            sys.exit(f"ensemble_index.json lists no members: {artifact}")
        for i, m in enumerate(members):
            rel = m.get("relative_dir")
            if not rel:
                sys.exit(f"ensemble member #{i} missing relative_dir under {artifact}")
            seed_raw = m.get("train_seed", i)
            try:
                seed = int(seed_raw)
            except (TypeError, ValueError) as err:
                raise SystemExit(
                    f"ensemble member #{i} has invalid train_seed: {seed_raw!r}"
                ) from err
            member_roots.append((seed, artifact / str(rel)))
        member_roots.sort(key=lambda x: x[0])

        meta_path = member_roots[0][1] / "training.json"
        ckpt_suffix = "(ensemble majority vote)"

        missing: list[str] = []
        for seed, mr in member_roots:
            for fname in ("training.json", "tokeniser.json", "model.pt"):
                if not (mr / fname).is_file():
                    missing.append(f"{mr}/{fname}")
        if missing:
            sys.exit("Ensemble bundle incomplete — missing:\n  " + "\n  ".join(missing))

        logger.info(
            "Loaded ensemble run root with %d members (seeds %s)",
            len(member_roots),
            [s for s, _ in member_roots],
        )

    else:
        meta_path = artifact / "training.json"
        ckpt_suffix = ""

    if not meta_path.is_file():
        sys.exit(f"Not a bundle directory (no training.json at {meta_path})")

    meta = json.loads(meta_path.read_text(encoding="utf-8"))

    if args.show_meta:
        if is_ensemble_root and ensemble_index is not None:
            print(
                json.dumps(
                    {"ensemble_index": ensemble_index, "training_json": meta},
                    indent=2,
                    ensure_ascii=False,
                )
            )
            logger.info(
                "Printed ensemble_index + representative training.json from %s",
                meta_path.parent,
            )
        else:
            pe = meta.get("per_epoch") or []
            print(json.dumps(meta, indent=2, ensure_ascii=False))
            logger.info(
                "epochs=%d best_val_f1_macro=%s wall_s=%s",
                meta.get("epochs_completed"),
                meta.get("best_val_f1_macro"),
                meta.get("cnn_training_wall_seconds"),
            )
            if pe:
                last = pe[-1]
                logger.info(
                    "last epoch: train_loss=%s val_f1=%s lr=%s",
                    last.get("train_loss"),
                    last.get("val_f1_macro"),
                    last.get("lr"),
                )
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cuda_amp_infer = not args.no_amp

    if is_ensemble_root:
        cfg_dict = meta.get("model_config") or {}
        model = CNNLSTMModel(ModelConfig(**cfg_dict)).to(device)
        member_ckpts = [mr / "model.pt" for _, mr in member_roots]
        tokeniser = load_tokeniser(member_roots[0][1])
    else:
        _, model, device = load_bundle(artifact, device=device)
        tokeniser = load_tokeniser(artifact)
        member_ckpts = []

    ev_amp_kw: bool | None = False if args.no_amp else None

    if not args.eval:
        kind = (
            "ensemble CNN-LSTM bundle"
            if is_ensemble_root
            else "CNN-LSTM bundle"
        )
        logger.info("Loaded %s from %s on %s (--eval not set).", kind, artifact, device)
        return

    split_seed = args.split_seed if args.split_seed is not None else int(meta["split_seed"])
    cli_snap = meta.get("cli_args") or {}
    dataset_path = resolve_evaluation_dataset_path(
        explicit=args.dataset,
        preset=args.preset_dataset,
        training_cli_dataset=cli_snap.get("dataset"),
    )

    if args.batch_size <= 0:
        mem_gib = training_hardware.gpu_total_memory_gib(0) if device.type == "cuda" else None
        batch_size = training_hardware.suggest_batch_size(
            use_cuda=device.type == "cuda",
            gpu_mem_gib=mem_gib,
            device_index=0,
        )
    else:
        batch_size = args.batch_size
    pin_memory = device.type == "cuda"
    eval_num_workers = 0

    max_seq_len = int(meta["model_config"]["max_seq_len"])

    cfg_dict = meta["model_config"]
    num_classes = int(cfg_dict["num_classes"])

    data_loader_svc = DatasetLoader()
    texts, labels = data_loader_svc.load(dataset_path, fetch_if_missing=args.fetch_dataset)
    if data_loader_svc.num_authors != num_classes:
        logger.warning(
            "Author count mismatch: artifact num_classes=%d but dataset reports %d labels.",
            num_classes,
            data_loader_svc.num_authors,
        )

    preprocessor = Preprocessor()
    texts = preprocessor.batch_clean(texts)
    paired = [(t, l) for t, l in zip(texts, labels) if t]
    if not paired:
        raise ValueError("All texts became empty after preprocessing.")
    texts, labels = zip(*paired)
    texts = list(texts)
    labels = list(labels)

    if args.eval_full_dataset:
        logger.info(
            "Dataset %s — full-dataset eval (%d samples, no split) %s",
            dataset_path,
            len(labels),
            ckpt_suffix,
        )
        all_ids = tokeniser.batch_encode(texts, max_length=max_seq_len)
        loaders = [("full_dataset", all_ids, labels)]
    else:
        logger.info(
            "Dataset %s — stratified split (split_seed=%d) %s",
            dataset_path,
            split_seed,
            ckpt_suffix,
        )
        train_split, val_split, test_split = data_loader_svc.split(texts, labels, seed=split_seed)

        train_ids = tokeniser.batch_encode(train_split.texts, max_length=max_seq_len)
        val_ids = tokeniser.batch_encode(val_split.texts, max_length=max_seq_len)
        test_ids = tokeniser.batch_encode(test_split.texts, max_length=max_seq_len)

        loaders = [
            ("train", train_ids, train_split.labels),
            ("validation", val_ids, val_split.labels),
            ("test", test_ids, test_split.labels),
        ]

    splits_out: dict[str, dict] = {}
    for name, idsarr, lbls in loaders:
        loader = _make_eval_loader(
            idsarr,
            lbls,
            batch_size,
            num_workers=eval_num_workers,
            pin_memory=pin_memory,
        )
        if is_ensemble_root:
            m = ensemble_evaluate_majority_vote(
                model,
                member_ckpts,
                loader,
                device,
                cuda_amp=cuda_amp_infer,
            )
        else:
            m = evaluate(model, loader, use_amp=ev_amp_kw)
        splits_out[name] = metrics_dict_to_jsonable(m)
        logger.info(
            "%s — acc=%.4f macro_f1=%.4f",
            name,
            m.accuracy,
            m.f1_macro,
        )

    payload: dict = {
        "artifact": str(artifact),
        "eval_mode": "full_dataset" if args.eval_full_dataset else "stratified_splits",
        "splits": splits_out,
        "ensemble_majority_vote": is_ensemble_root,
    }
    if args.eval_full_dataset:
        payload["num_samples"] = len(labels)
    else:
        payload["split_seed_used"] = split_seed
    if is_ensemble_root:
        payload["ensemble_checkpoints"] = [str(p.resolve()) for p in member_ckpts]
    if args.metrics_out:
        out_path = args.metrics_out
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)
        logger.info("Wrote %s", out_path)


if __name__ == "__main__":
    main()
