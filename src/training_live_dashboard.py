"""TensorBoard scalars during training (optional browser UI)."""

from __future__ import annotations

import atexit
import logging
import re
import subprocess
import sys
import time
import webbrowser
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

logger = logging.getLogger(__name__)

_EMBEDDED_TB_PROC: subprocess.Popen[Any] | None = None
_EMBEDDED_TB_ATEXIT_REGISTERED = False


def _tensorboard_cli_available() -> bool:
    try:
        import tensorboard.main  # noqa: F401
    except ImportError:
        return False
    return True


def shutdown_embedded_tensorboard_server() -> None:
    """Terminate the TensorBoard child started by :func:`spawn_embedded_tensorboard_server`, if any."""
    global _EMBEDDED_TB_PROC
    proc = _EMBEDDED_TB_PROC
    if proc is None:
        return
    _EMBEDDED_TB_PROC = None
    if proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=8)
        except subprocess.TimeoutExpired:
            proc.kill()


def _register_embedded_tensorboard_atexit() -> None:
    global _EMBEDDED_TB_ATEXIT_REGISTERED
    if not _EMBEDDED_TB_ATEXIT_REGISTERED:
        atexit.register(shutdown_embedded_tensorboard_server)
        _EMBEDDED_TB_ATEXIT_REGISTERED = True


def spawn_embedded_tensorboard_server(
    logdir: Path,
    *,
    port: int | None = None,
    port_attempts: int = 10,
    open_browser: bool = True,
) -> str | None:
    """Start TensorBoard under *logdir*; return ``http://127.0.0.1:<port>/`` or ``None`` if all attempts fail.

    Uses the same interpreter (``sys.executable -m tensorboard.main``). Replaces any
    previously started embedded server in this interpreter.
    """
    global _EMBEDDED_TB_PROC
    if not _tensorboard_cli_available():
        logger.warning(
            "TensorBoard package unavailable — cannot start embedded server; "
            "`pip install tensorboard` or run `tensorboard --logdir %s` manually.",
            logdir.resolve(),
        )
        return None

    shutdown_embedded_tensorboard_server()

    root = logdir.resolve()
    root.mkdir(parents=True, exist_ok=True)
    ports: list[int]
    if port is None or port <= 0:
        ports = list(range(6006, 6006 + max(1, port_attempts)))
    else:
        ports = [port]

    popen_kw: dict[str, Any] = {
        "stdin": subprocess.DEVNULL,
        "stdout": subprocess.DEVNULL,
        "stderr": subprocess.DEVNULL,
    }
    if sys.platform == "win32":
        # Avoid flashing a console window for the child TensorBoard process.
        popen_kw["creationflags"] = getattr(subprocess, "CREATE_NO_WINDOW", 0)

    for pnum in ports:
        cmd = [
            sys.executable,
            "-m",
            "tensorboard.main",
            "--logdir",
            str(root),
            "--port",
            str(pnum),
            "--host",
            "127.0.0.1",
            "--reload_interval",
            "2",
            "--reload_multifile",
            "true",
        ]
        try:
            proc = subprocess.Popen(cmd, **popen_kw)
        except OSError as exc:
            logger.warning("Could not start TensorBoard subprocess (%s)", exc)
            continue
        time.sleep(0.7)
        if proc.poll() is not None:
            continue
        _EMBEDDED_TB_PROC = proc
        _register_embedded_tensorboard_atexit()
        url = f"http://127.0.0.1:{pnum}/"
        logger.info("TensorBoard server (embedded) → %s  logdir=%s", url, root)
        if open_browser:
            try:
                webbrowser.open(url)
            except Exception as exc:  # pragma: no cover — platform-specific
                logger.debug("webbrowser.open failed: %s", exc)
        return url

    logger.warning(
        "Could not bind TensorBoard on %s — start manually: tensorboard --logdir %s",
        ports,
        root,
    )
    return None


@runtime_checkable
class EpochDashboard(Protocol):
    def update(self, epoch: int, train_loss: float, val_f1_macro: float, lr: float) -> None: ...

    def close(self) -> None: ...


def _tensorboard_tag_safe(title: str) -> str:
    s = re.sub(r"[^\w]+", "_", title.strip().lower()).strip("_")
    return (s[:48] + "_") if s else ""


class LiveTensorBoardDashboard:
    """Log train loss, validation macro-F1, and LR each epoch via ``SummaryWriter``.

    Writes under *log_dir*; use ``tensorboard --logdir <parent>`` with one or many runs.
    """

    def __init__(
        self,
        *,
        log_dir: str,
        title: str = "",
        log_initial_lr: float | None = None,
    ) -> None:
        try:
            from torch.utils.tensorboard import SummaryWriter
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "TensorBoard dashboard requires package 'tensorboard'. "
                "Install with: pip install tensorboard"
            ) from exc
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        # Flush often so TB's file watcher picks up epochs during long-running training.
        writer = SummaryWriter(log_dir=log_dir, max_queue=10, flush_secs=2)
        self._writer: Any | None = writer
        self._tag_prefix = _tensorboard_tag_safe(title)
        logger.info(
            "TensorBoard scalars → %s",
            Path(log_dir).resolve(),
        )
        if log_initial_lr is not None:
            p = self._tag_prefix
            # IMPORTANT: ``new_style=True`` emits tensor summaries classified as tensors, not scalar
            # plugin summaries; the TB Scalars UI then often renders only one point / nothing useful.
            # Default ``simple_value`` scalars accumulate correctly step-by-step in Scalars charts.
            self._writer.add_scalar(f"{p}opt/lr", float(log_initial_lr), 0)
            self._writer.flush()
            logger.info(
                "TensorBoard: logged initial learning rate at step 0; "
                "train/val curves appear after the first epoch completes.",
            )

    def update(self, epoch: int, train_loss: float, val_f1_macro: float, lr: float) -> None:
        if self._writer is None:
            return
        p = self._tag_prefix
        self._writer.add_scalar(f"{p}train/loss", float(train_loss), epoch)
        self._writer.add_scalar(f"{p}val/f1_macro", float(val_f1_macro), epoch)
        self._writer.add_scalar(f"{p}opt/lr", float(lr), epoch)
        self._writer.flush()

    def close(self) -> None:
        if self._writer is not None:
            self._writer.close()
            self._writer = None


class NullEpochDashboard:
    def update(self, epoch: int, train_loss: float, val_f1_macro: float, lr: float) -> None:
        pass

    def close(self) -> None:
        pass


def create_live_dashboard(
    *,
    title: str,
    tensorboard_root: Path,
    tensorboard_subdir: str,
    log_initial_lr: float | None = None,
) -> EpochDashboard:
    """Open a TensorBoard-backed live dashboard, or a no-op if TensorBoard is unavailable."""
    tb_dir = tensorboard_root / tensorboard_subdir
    try:
        return LiveTensorBoardDashboard(
            log_dir=str(tb_dir),
            title=title,
            log_initial_lr=log_initial_lr,
        )
    except RuntimeError as exc:
        logger.warning("%s — training continues without live TensorBoard curves.", exc)
        return NullEpochDashboard()
