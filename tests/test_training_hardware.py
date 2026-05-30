"""Unit tests for src.training_hardware heuristics."""

from __future__ import annotations

import pytest

from src import training_hardware


def test_suggest_batch_size_cuda_tiers_respect_free_vram_off() -> None:
    """Tier checks must not depend on live ``mem_get_info`` (varies by machine)."""
    assert training_hardware.suggest_batch_size(
        use_cuda=True, gpu_mem_gib=4.0, respect_free_vram=False
    ) == 32
    assert training_hardware.suggest_batch_size(
        use_cuda=True, gpu_mem_gib=7.0, respect_free_vram=False
    ) == 48
    assert training_hardware.suggest_batch_size(
        use_cuda=True, gpu_mem_gib=10.0, respect_free_vram=False
    ) == 64
    assert training_hardware.suggest_batch_size(
        use_cuda=True, gpu_mem_gib=11.6, respect_free_vram=False
    ) == 96
    assert training_hardware.suggest_batch_size(
        use_cuda=True, gpu_mem_gib=14.0, respect_free_vram=False
    ) == 96
    assert training_hardware.suggest_batch_size(
        use_cuda=True, gpu_mem_gib=20.0, respect_free_vram=False
    ) == 128


def test_suggest_batch_size_cpu() -> None:
    assert training_hardware.suggest_batch_size(use_cuda=False, gpu_mem_gib=None) == 32


def test_free_vram_heuristic_scales_down(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(training_hardware, "gpu_free_memory_fraction", lambda _idx=0: 0.28)
    # 16 GiB tier baseline 128 → frac < 0.32 → halve → 64
    assert training_hardware.suggest_batch_size(
        use_cuda=True, gpu_mem_gib=16.0, device_index=0, respect_free_vram=True
    ) == 64


def test_free_vram_headroom_scales_up(monkeypatch: pytest.MonkeyPatch) -> None:
    """High free/total fraction should raise batch modestly (capped; trainer can still OOM-halve)."""
    monkeypatch.setattr(training_hardware, "gpu_free_memory_fraction", lambda _idx=0: 0.86)
    # 11.6 GiB → tier 96 → 1.5× from frac ≥ 0.82 → 144 (multiples of 8)
    assert training_hardware.suggest_batch_size(
        use_cuda=True, gpu_mem_gib=11.6, device_index=0, respect_free_vram=True
    ) == 144


def test_suggest_num_workers_is_non_negative() -> None:
    n = training_hardware.suggest_num_workers()
    assert 0 <= n <= 8
