"""Trainer for the CNN-LSTM authorship attribution model."""

from __future__ import annotations

import inspect
import logging
import os
from collections.abc import Callable
import tempfile
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, confusion_matrix
import numpy as np

from .models import MetricsDict, TrainingHistory, TrainingDivergenceError

logger = logging.getLogger(__name__)


def _adam_supports_fused_kw() -> bool:
    sig = inspect.signature(torch.optim.Adam.__init__)
    return "fused" in sig.parameters


def build_adam(
    params,
    *,
    lr: float,
    weight_decay: float,
    use_fused: bool,
):
    """Create Adam; on CUDA, try ``fused=True`` when requested (faster kernels on recent PyTorch)."""
    fused_on = (
        bool(use_fused)
        and torch.cuda.is_available()
        and _adam_supports_fused_kw()
    )
    kw: dict = {"params": params, "lr": lr, "weight_decay": weight_decay}
    if fused_on:
        kw["fused"] = True
        try:
            opt = torch.optim.Adam(**kw)
            logger.info(
                "Using fused Adam CUDA optimizer kernels (fallback off if incompatible)."
            )
            return opt
        except (RuntimeError, TypeError) as exc:
            logger.warning("Fused Adam unavailable (%s); using reference Adam.", exc)
            kw.pop("fused", None)
    kw.pop("fused", None)
    return torch.optim.Adam(**kw)


def _cuda_amp_dtype() -> tuple[torch.dtype, bool]:
    """AMP dtype and whether :class:`GradScaler` is required (fp16 only)."""
    if torch.cuda.is_bf16_supported():
        return torch.bfloat16, False
    return torch.float16, True


def _atomic_torch_save(obj: object, checkpoint_path: str, *, retries: int = 6, delay_s: float = 0.12) -> None:
    """Write PyTorch checkpoint with replace-in-place semantics.

    Overwriting ``checkpoint_path`` in place can raise on Windows when the target
    is temporarily locked (e.g. indexer, antivirus, sync). Saving to a new file
    in the same directory and ``os.replace`` is more robust; retries cover
    lingering locks on replace.
    """
    path = os.path.normpath(checkpoint_path)
    parent = os.path.dirname(path) or "."
    os.makedirs(parent, exist_ok=True)
    last_exc: BaseException | None = None
    fd, tmp_path = tempfile.mkstemp(suffix=".ckpt.tmp", dir=parent)
    os.close(fd)
    try:
        for attempt in range(retries):
            try:
                torch.save(obj, tmp_path)
                os.replace(tmp_path, path)
                return
            except (OSError, RuntimeError) as exc:
                last_exc = exc
                logger.warning(
                    "Checkpoint save attempt %d/%d failed (%s); retrying after %.2fs.",
                    attempt + 1,
                    retries,
                    exc.__class__.__name__,
                    delay_s * (attempt + 1),
                )
                time.sleep(delay_s * (attempt + 1))
        assert last_exc is not None
        raise last_exc
    finally:
        if os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except OSError:
                pass


def _rebuild_train_loader_smaller_batch(train_loader: DataLoader, new_bs: int) -> DataLoader:
    """Rebuild train DataLoader with a smaller batch size (CUDA OOM retry)."""
    n_workers = getattr(train_loader, "num_workers", 0)
    kwargs: dict = {
        "dataset": train_loader.dataset,
        "batch_size": new_bs,
        "shuffle": True,
        "num_workers": n_workers,
        "pin_memory": getattr(train_loader, "pin_memory", False),
        "drop_last": getattr(train_loader, "drop_last", False),
        "collate_fn": train_loader.collate_fn,
    }
    if n_workers > 0:
        kwargs["persistent_workers"] = getattr(train_loader, "persistent_workers", False)
        kwargs["prefetch_factor"] = getattr(train_loader, "prefetch_factor", 2)
    return DataLoader(**kwargs)


class Trainer:
    """Training loop with validation macro-F1, early stopping, and checkpoints."""

    def train(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int,
        lr: float,
        patience: int,
        checkpoint_path: str = "artifacts/checkpoints/best_model.pt",
        *,
        weight_decay: float = 0.0,
        label_smoothing: float = 0.0,
        reduce_lr_on_plateau: bool = False,
        class_weights: torch.Tensor | None = None,
        lr_schedule: str | None = None,
        cosine_t0: int = 8,
        onecycle_max_lr: float | None = None,
        onecycle_div_factor: float = 25.0,
        onecycle_pct_start: float = 0.1,
        onecycle_final_div_factor: float = 10000.0,
        use_amp: bool | None = None,
        adam_fused: bool | None = None,
        on_epoch_end: Callable[[int, int, float, float, float], None] | None = None,
    ) -> TrainingHistory:
        """Run the training loop.

        Args:
            model: The neural network to train.
            train_loader: DataLoader for training data.
            val_loader: DataLoader for validation data.
            epochs: Maximum number of training epochs.
            lr: Base / peak learning rate: Adam LR for ``plateau`` / ``cosine_restarts`` /
                ``none``; **peak** LR for ``onecycle`` when ``onecycle_max_lr`` is unset.
            patience: Early-stopping patience in epochs.
            checkpoint_path: Where to save the best model checkpoint.
            weight_decay: L2 regularisation for Adam (default 0 for backward compatibility).
            label_smoothing: CrossEntropyLoss label smoothing (0 disables).
            reduce_lr_on_plateau: If True, reduce learning rate when val macro-F1 plateaus
                (used only when ``lr_schedule`` is None; legacy).
            class_weights: Per-class loss weights (e.g. inverse frequency); must match num classes.
            lr_schedule: ``"none"`` | ``"plateau"`` | ``"cosine_restarts"`` | ``"onecycle"`` | None.
                ``onecycle`` uses :class:`~torch.optim.lr_scheduler.OneCycleLR`, stepped each batch.
            cosine_t0: Period in epochs for the first restart (cosine_restarts only).
            onecycle_max_lr: Optional override for OneCycle peak LR.
            onecycle_div_factor: OneCycle ``div_factor``.
            onecycle_pct_start: Fraction of steps with increasing LR (onecycle).
            onecycle_final_div_factor: Final LR divisor (onecycle).
            use_amp: If True, use CUDA automatic mixed precision (bf16 when supported, else fp16).
                If None, enable on CUDA only. Hyperparameters (lr, weight decay, etc.) are unchanged.
            adam_fused: If ``False``, never request fused Adam. If ``True`` or ``None`` on CUDA,
                try ``torch.optim.Adam(..., fused=True)`` where supported.
            on_epoch_end: Optional hook called after each epoch with
                ``(epoch, epochs, train_loss, val_f1_macro, lr)`` for live dashboards / plots.

        Returns:
            TrainingHistory with per-epoch train losses and validation metrics.

        Raises:
            TrainingDivergenceError: If NaN loss is detected.
        """
        history = TrainingHistory()

        # Detect device from model parameters
        try:
            device = next(model.parameters()).device
        except StopIteration:
            device = torch.device("cpu")

        use_amp_eff = (device.type == "cuda") if use_amp is None else bool(use_amp)
        try_adam_fused = device.type == "cuda" and adam_fused is not False
        amp_dtype, amp_needs_scaler = (
            _cuda_amp_dtype() if use_amp_eff and device.type == "cuda" else (torch.float32, False)
        )
        scaler: torch.amp.GradScaler | None = None
        if use_amp_eff and device.type == "cuda" and amp_needs_scaler:
            scaler = torch.amp.GradScaler("cuda")
        if use_amp_eff and device.type == "cuda":
            logger.info(
                "CUDA AMP enabled (dtype=%s, grad_scaler=%s) — same LR/schedule as fp32.",
                str(amp_dtype).replace("torch.", ""),
                scaler is not None,
            )

        ce_kw: dict = {"label_smoothing": label_smoothing}
        if class_weights is not None:
            ce_kw["weight"] = class_weights.to(device)
        criterion = nn.CrossEntropyLoss(**ce_kw)

        if lr_schedule is not None:
            mode = lr_schedule
        else:
            mode = "plateau" if reduce_lr_on_plateau else "none"
        if mode not in ("none", "plateau", "cosine_restarts", "onecycle"):
            raise ValueError(
                f"lr_schedule must be 'none', 'plateau', 'cosine_restarts', or 'onecycle'; got {mode!r}"
            )

        scheduler: (
            torch.optim.lr_scheduler.ReduceLROnPlateau
            | torch.optim.lr_scheduler.CosineAnnealingWarmRestarts
            | None
        ) = None
        batch_scheduler: torch.optim.lr_scheduler.OneCycleLR | None = None
        schedule_kind: str = "none"
        max_peak = float(onecycle_max_lr if onecycle_max_lr is not None else lr)

        if mode == "onecycle":
            total_steps = max(1, epochs * len(train_loader))
            optimiser = build_adam(
                model.parameters(),
                lr=max_peak / onecycle_div_factor,
                weight_decay=weight_decay,
                use_fused=try_adam_fused,
            )
            batch_scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimiser,
                max_lr=max_peak,
                total_steps=total_steps,
                pct_start=onecycle_pct_start,
                div_factor=onecycle_div_factor,
                final_div_factor=onecycle_final_div_factor,
            )
            schedule_kind = "onecycle"
        else:
            optimiser = build_adam(
                model.parameters(),
                lr=lr,
                weight_decay=weight_decay,
                use_fused=try_adam_fused,
            )
            if mode == "plateau":
                scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimiser,
                    mode="max",
                    factor=0.5,
                    patience=4,
                    min_lr=1e-6,
                )
                schedule_kind = "plateau"
            elif mode == "cosine_restarts":
                scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                    optimiser, T_0=cosine_t0, T_mult=1, eta_min=1e-6
                )
                schedule_kind = "cosine_restarts"

        best_val_f1 = -1.0
        patience_counter = 0

        # Ensure checkpoint directory exists
        checkpoint_dir = os.path.dirname(checkpoint_path)
        if checkpoint_dir:
            os.makedirs(checkpoint_dir, exist_ok=True)

        for epoch in range(1, epochs + 1):
            epoch_loss = self._run_epoch(
                model=model,
                train_loader=train_loader,
                optimiser=optimiser,
                criterion=criterion,
                device=device,
                epoch=epoch,
                checkpoint_path=checkpoint_path,
                batch_scheduler=batch_scheduler,
                use_amp=use_amp_eff,
                amp_dtype=amp_dtype,
                scaler=scaler,
            )

            history.train_losses.append(epoch_loss)

            # Validation macro-F1 each epoch
            val_metrics = self.evaluate(
                model, val_loader, use_amp=use_amp_eff, amp_dtype=amp_dtype
            )
            history.val_metrics.append(val_metrics)

            lr_now = optimiser.param_groups[0]["lr"]
            logger.info(
                "Epoch %d/%d  loss=%.4f  val_f1=%.4f  lr=%.2e  (%s)",
                epoch, epochs, epoch_loss, val_metrics.f1_macro, lr_now, schedule_kind,
            )
            history.lr_per_epoch.append(float(lr_now))

            if on_epoch_end is not None:
                on_epoch_end(epoch, epochs, epoch_loss, val_metrics.f1_macro, float(lr_now))

            if scheduler is not None:
                if schedule_kind == "plateau":
                    scheduler.step(val_metrics.f1_macro)
                elif schedule_kind != "onecycle":
                    scheduler.step()

            # Best checkpoint + reset patience on improvement
            if val_metrics.f1_macro > best_val_f1:
                best_val_f1 = val_metrics.f1_macro
                _atomic_torch_save(model.state_dict(), checkpoint_path)
                logger.info("Checkpoint saved to %s (val_f1=%.4f)", checkpoint_path, best_val_f1)
                patience_counter = 0
            else:
                patience_counter += 1

            # Early stopping
            if patience_counter >= patience:
                logger.info(
                    "Early stopping at epoch %d (patience=%d exhausted).", epoch, patience
                )
                break

        return history

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _run_epoch(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        optimiser: torch.optim.Optimizer,
        criterion: nn.Module,
        device: torch.device,
        epoch: int,
        checkpoint_path: str,
        batch_scheduler: torch.optim.lr.scheduler.OneCycleLR | None = None,
        *,
        use_amp: bool = False,
        amp_dtype: torch.dtype = torch.float32,
        scaler: torch.amp.GradScaler | None = None,
    ) -> float:
        """One training epoch; halves batch size and retries on CUDA OOM.

        Returns the mean training loss for the epoch.
        """
        try:
            return self._train_one_epoch(
                model,
                train_loader,
                optimiser,
                criterion,
                device,
                epoch,
                batch_scheduler=batch_scheduler,
                use_amp=use_amp,
                amp_dtype=amp_dtype,
                scaler=scaler,
            )
        except torch.cuda.OutOfMemoryError:
            if batch_scheduler is not None:
                raise RuntimeError(
                    "CUDA OOM during training with per-batch LR scheduling (e.g. OneCycle). "
                    "Lower --batch-size or use --lr-schedule plateau."
                ) from None
            # Halve batch and retry
            old_bs = train_loader.batch_size
            new_bs = max(1, old_bs // 2)
            logger.warning(
                "CUDA OOM at epoch %d. Halving batch size: %d → %d and retrying.",
                epoch, old_bs, new_bs,
            )
            new_loader = _rebuild_train_loader_smaller_batch(train_loader, new_bs)
            torch.cuda.empty_cache()
            return self._train_one_epoch(
                model,
                new_loader,
                optimiser,
                criterion,
                device,
                epoch,
                batch_scheduler=None,
                use_amp=use_amp,
                amp_dtype=amp_dtype,
                scaler=scaler,
            )

    def _train_one_epoch(
        self,
        model: nn.Module,
        loader: DataLoader,
        optimiser: torch.optim.Optimizer,
        criterion: nn.Module,
        device: torch.device,
        epoch: int,
        *,
        batch_scheduler: torch.optim.lr_scheduler.OneCycleLR | None = None,
        use_amp: bool = False,
        amp_dtype: torch.dtype = torch.float32,
        scaler: torch.amp.GradScaler | None = None,
    ) -> float:
        """Core training loop for a single epoch.

        Raises:
            TrainingDivergenceError: On NaN loss.
        """
        model.train()
        total_loss = 0.0
        num_batches = 0

        for batch_idx, batch in enumerate(loader):
            # Support both (inputs, labels) tuples and dict-style batches
            if isinstance(batch, (list, tuple)):
                inputs, labels = batch[0], batch[1]
            else:
                inputs = batch["input_ids"]
                labels = batch["labels"]

            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimiser.zero_grad(set_to_none=True)

            amp_enabled = use_amp and device.type == "cuda" and amp_dtype != torch.float32
            with torch.amp.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
                logits = model(inputs)
                loss = criterion(logits, labels)

            # NaN loss → abort
            if not torch.isfinite(loss).item():
                msg = f"NaN/Inf loss detected at epoch {epoch}, batch {batch_idx}."
                logger.error(msg)
                raise TrainingDivergenceError(msg)

            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.unscale_(optimiser)
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimiser)
                scaler.update()
            else:
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimiser.step()
            if batch_scheduler is not None:
                batch_scheduler.step()

            total_loss += float(loss.detach().item())
            num_batches += 1

        return total_loss / num_batches if num_batches > 0 else 0.0

    def evaluate(
        self,
        model: nn.Module,
        loader: DataLoader,
        *,
        use_amp: bool | None = None,
        amp_dtype: torch.dtype | None = None,
    ) -> MetricsDict:
        """Validation metrics for one DataLoader (macro-F1 drives early stopping)."""
        try:
            device = next(model.parameters()).device
        except StopIteration:
            device = torch.device("cpu")

        eff_amp = device.type == "cuda" if use_amp is None else bool(use_amp)
        if amp_dtype is None:
            dt, _ = _cuda_amp_dtype() if eff_amp and device.type == "cuda" else (torch.float32, False)
        else:
            dt = amp_dtype
        amp_enabled = eff_amp and device.type == "cuda" and dt != torch.float32

        model.eval()
        all_preds: list[int] = []
        all_labels: list[int] = []

        with torch.no_grad():
            for batch in loader:
                if isinstance(batch, (list, tuple)):
                    inputs, labels = batch[0], batch[1]
                else:
                    inputs = batch["input_ids"]
                    labels = batch["labels"]

                inputs = inputs.to(device, non_blocking=True)
                with torch.amp.autocast(device_type=device.type, dtype=dt, enabled=amp_enabled):
                    logits = model(inputs)
                preds = logits.argmax(dim=-1).cpu().tolist()
                all_preds.extend(preds)
                all_labels.extend(labels.tolist())

        y_true = np.array(all_labels)
        y_pred = np.array(all_preds)

        classes = np.unique(np.concatenate([y_true, y_pred]))

        acc = float(accuracy_score(y_true, y_pred))
        prec = float(precision_score(y_true, y_pred, average="macro", zero_division=0))
        rec = float(recall_score(y_true, y_pred, average="macro", zero_division=0))
        f1_mac = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
        f1_per = f1_score(y_true, y_pred, average=None, labels=classes, zero_division=0)
        f1_per_class = {int(c): float(f) for c, f in zip(classes, f1_per)}
        conf_mat = confusion_matrix(y_true, y_pred, labels=classes)

        return MetricsDict(
            accuracy=acc,
            precision_macro=prec,
            recall_macro=rec,
            f1_macro=f1_mac,
            f1_per_class=f1_per_class,
            confusion_matrix=conf_mat,
        )
