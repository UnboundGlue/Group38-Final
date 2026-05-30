"""Tests for TensorBoard / factory live dashboards."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from tensorboard.backend.event_processing import event_accumulator as tb_ea

from src.training_live_dashboard import (
    LiveTensorBoardDashboard,
    create_live_dashboard,
    shutdown_embedded_tensorboard_server,
    spawn_embedded_tensorboard_server,
)


def test_live_dashboard_uses_scalar_plugin_not_tensor_only_summaries(tmp_path: Path) -> None:
    """Regression: ``new_style`` scalars are stored as tensors; TB Scalars UI then shows ~1 point."""
    tb_root = tmp_path / "logs"
    dash = LiveTensorBoardDashboard(
        log_dir=str(tb_root),
        title="step_test",
        log_initial_lr=0.01,
    )
    for ep in range(1, 6):
        dash.update(ep, train_loss=float(ep), val_f1_macro=0.1 * ep, lr=0.01)
    dash.close()

    acc = tb_ea.EventAccumulator(str(tb_root), size_guidance={})
    acc.Reload()
    assert acc.Tags().get("scalars"), "expect classic scalar summaries for step-by-step charts"
    scalars_tags = list(acc.Tags()["scalars"])
    loss_candidates = [t for t in scalars_tags if t.endswith("train/loss")]
    assert loss_candidates
    pts = acc.Scalars(loss_candidates[0])
    steps = sorted(s.step for s in pts)
    assert steps == [1, 2, 3, 4, 5]


def test_create_live_dashboard_tensorboard_writes_scalars(tmp_path: Path) -> None:
    tb_root = tmp_path / "tb"
    dash = create_live_dashboard(
        title="unit_test_run",
        tensorboard_root=tb_root,
        tensorboard_subdir="sub",
    )
    dash.update(1, train_loss=2.0, val_f1_macro=0.11, lr=1e-3)
    dash.update(2, train_loss=1.5, val_f1_macro=0.22, lr=5e-4)
    dash.close()
    evt = list((tb_root / "sub").glob("events.out.tfevents.*"))
    assert evt, "SummaryWriter should create a tfevents file"


def test_spawn_embedded_tensorboard_server_command(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("tensorboard", reason="pip install tensorboard")
    import src.training_live_dashboard as tld

    recorded: list[list[str]] = []

    class FakeProc:
        def poll(self) -> None:
            return None

        def terminate(self) -> None:
            pass

        def wait(self, timeout: float | None = None) -> int:
            return 0

        def kill(self) -> None:
            pass

    def fake_popen(cmd: list[str], **kwargs: object) -> FakeProc:
        recorded.append(cmd)
        return FakeProc()

    monkeypatch.setattr(tld.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(tld.time, "sleep", lambda _s: None)

    shutdown_embedded_tensorboard_server()
    logdir = tmp_path / "tb_root"
    url = spawn_embedded_tensorboard_server(logdir, port=6010, open_browser=False)
    shutdown_embedded_tensorboard_server()

    assert url == "http://127.0.0.1:6010/"
    assert recorded, "Popen should have been called"
    cmd = recorded[0]
    assert cmd[:4] == [sys.executable, "-m", "tensorboard.main", "--logdir"]
    assert Path(cmd[4]).resolve() == logdir.resolve()
    assert cmd[5:11] == ["--port", "6010", "--host", "127.0.0.1", "--reload_interval", "2"]
    assert cmd[11:13] == ["--reload_multifile", "true"]


@pytest.fixture(autouse=True)
def _cleanup_embedded_tensorboard() -> None:
    yield
    shutdown_embedded_tensorboard_server()


@pytest.fixture(autouse=True)
def _requires_tensorboard() -> None:
    pytest.importorskip("tensorboard", reason="pip install tensorboard")