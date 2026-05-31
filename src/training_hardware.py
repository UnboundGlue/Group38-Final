"""Heuristics for batch size and DataLoader settings from available hardware.

Used by ``experiments.run_cnn_lstm`` so a single command line can scale to GPU VRAM
and CPU count without hand-tuning. Install a **CUDA** build of PyTorch to use a GPU
(see project README); CPU-only wheels still train but cannot use the discrete GPU.
"""

from __future__ import annotations

import os
import sys

import torch


def gpu_total_memory_gib(device_index: int = 0) -> float | None:
    """Return total GPU memory in GiB, or None if CUDA is not available."""
    if not torch.cuda.is_available():
        return None
    props = torch.cuda.get_device_properties(device_index)
    return float(props.total_memory) / (1024.0**3)


def gpu_free_memory_fraction(device_index: int = 0) -> float | None:
    """Fraction ``free/total`` of device memory (:func:`torch.cuda.mem_get_info`).

    Returns ``None`` if CUDA is inactive or the call fails — callers should ignore it.
    """
    if not torch.cuda.is_available():
        return None
    try:
        free_b, total_b = torch.cuda.mem_get_info(device_index)
        if total_b <= 0:
            return None
        return float(free_b) / float(total_b)
    except RuntimeError:
        return None


def _baseline_batch_from_total_vram_gib(total_gib: float) -> int:
    """Batch size tier from PyTorch reported **total** VRAM (before free-VRAM nudges)."""
    # 11.5+ GiB: many “12 GB” laptops report ~11.6–11.9 to PyTorch; keep them off the
    # 64 tier so they aren’t perennially VRAM‑under‑utilised versus desktop 12 GB parts.
    if total_gib >= 24.0:
        return 160
    if total_gib >= 16.0:
        return 128
    if total_gib >= 11.5:
        return 96
    if total_gib >= 8.0:
        return 64
    if total_gib >= 6.0:
        return 48
    return 32


def _round_batch_to_multiple(bs: int, *, step: int = 8, minimum: int = 8) -> int:
    return max(minimum, (max(minimum, bs) + step - 1) // step * step)


def suggest_batch_size(
    *,
    use_cuda: bool,
    gpu_mem_gib: float | None,
    device_index: int = 0,
    respect_free_vram: bool = True,
) -> int:
    """Pick batch size from **total** VRAM tier, optionally scaled down by **free** VRAM.

    When ``respect_free_vram`` is True, :func:`gpu_free_memory_fraction` is consulted:
    if a large fraction of framebuffer is already allocated (another app / compositor),
    batch size is reduced to lower first-step OOM risk. This mirrors OS-reported memory;
    :class:`~src.trainer.Trainer` can still halve after a real CUDA OOM.

    CNN-LSTM on long sequences stays memory-heavy; avoid ``onecycle`` if you rely on OOM retries.
    """
    if not use_cuda or gpu_mem_gib is None:
        return 32

    bs = _baseline_batch_from_total_vram_gib(gpu_mem_gib)
    tier_base = bs
    upscale_cap = min(448, tier_base * 3)

    if respect_free_vram:
        frac = gpu_free_memory_fraction(device_index)
        if frac is not None:
            # Low free framebuffer → tighten before allocation (OOM avoidance).
            if frac < 0.32:
                bs = max(8, bs // 2)
            elif frac < 0.48:
                bs = max(16, (bs * 2) // 3)
            elif frac < 0.62:
                bs = max(16, (bs * 3) // 4)
            # Plenty of framebuffer free vs total → notch up modestly up to upscale_cap.
            # ( Idle GPUs often report very high frac; Trainer still halves on CUDA OOM. )
            elif frac >= 0.82:
                bs = _round_batch_to_multiple(min(upscale_cap, (bs * 3) // 2))
            elif frac >= 0.72:
                bs = _round_batch_to_multiple(min(upscale_cap, (bs * 5) // 4))

    return _round_batch_to_multiple(bs)


def suggest_num_workers() -> int:
    """Background workers for :class:`~torch.utils.data.DataLoader`.

    On Windows returns **0**: ``spawn`` workers re-import the training script (and
    SciPy/sklearn), which often raises ``MemoryError`` or ``worker exited unexpectedly``
    after long GPU runs when host RAM is tight. Linux/macOS use a capped worker count.
    """
    if sys.platform == "win32":
        return 0
    n = os.cpu_count() or 4
    return max(0, min(8, n - 1))


def cuda_build_hint() -> str | None:
    """If PyTorch was built without CUDA, return a short install hint, else None."""
    if torch.cuda.is_available():
        return None
    # torch.version.cuda is set when the wheel includes CUDA; None for CPU-only wheels
    if getattr(torch.version, "cuda", None) is None:
        return (
            "This PyTorch build has no CUDA. For GPU training install a CUDA build from "
            "https://pytorch.org/get-started/locally/ (pick your OS and a CUDA version, "
            "then run the given pip install command)."
        )
    return "CUDA is not available (driver/runtime issue? Check nvidia-smi and CUDA toolkit)."
