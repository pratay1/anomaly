"""Keep-alive utilities: prevent OS sleep and handle thread exceptions."""

from __future__ import annotations

import ctypes
import logging
import platform
import sys
import threading
import time
import types
from contextlib import contextmanager
from typing import Iterator

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Windows: prevent sleep / hibernate / display-off via SetThreadExecutionState
# ---------------------------------------------------------------------------

_ES_CONTINUOUS = 0x80000000
_ES_SYSTEM_REQUIRED = 0x00000001
_ES_DISPLAY_REQUIRED = 0x00000002
_ES_AWAYMODE_REQUIRED = 0x00000040

_kernel32: ctypes.WinDLL | None = None

if platform.system() == "Windows":
    try:
        _kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
    except Exception:
        _kernel32 = None


def prevent_sleep() -> None:
    """Tell the OS to stay awake (system + display)."""
    if _kernel32 is None:
        return
    try:
        _kernel32.SetThreadExecutionState(
            _ES_CONTINUOUS | _ES_SYSTEM_REQUIRED | _ES_DISPLAY_REQUIRED
        )
        logger.info("Windows sleep prevention active")
    except Exception:
        logger.warning("Failed to prevent Windows sleep", exc_info=True)


def allow_sleep() -> None:
    """Re-allow normal sleep behaviour."""
    if _kernel32 is None:
        return
    try:
        _kernel32.SetThreadExecutionState(_ES_CONTINUOUS)
        logger.info("Windows sleep prevention released")
    except Exception:
        logger.warning("Failed to release Windows sleep prevention", exc_info=True)


@contextmanager
def keep_awake() -> Iterator[None]:
    """Context manager: prevent sleep while inside the block."""
    prevent_sleep()
    try:
        yield
    finally:
        allow_sleep()


# ---------------------------------------------------------------------------
# Thread exception hook — log and prevent silent thread death
# ---------------------------------------------------------------------------

_original_thread_run = threading.Thread.run


def _guarded_thread_run(self: threading.Thread) -> None:
    """Wrap every thread's run() so unhandled exceptions are logged."""
    try:
        _original_thread_run(self)
    except Exception:
        logger.exception("Unhandled exception in thread %s", self.name)
        raise
    except KeyboardInterrupt:
        logger.info("KeyboardInterrupt in thread %s", self.name)
        raise


def install_thread_exception_hook() -> None:
    """Monkey-patch threading.Thread.run to catch + log unhandled exceptions."""
    threading.Thread.run = _guarded_thread_run  # type: ignore[assignment]


def install_sys_excepthook() -> None:
    """Install a hook that logs unhandled exceptions before the default handler."""
    _default_excepthook = sys.excepthook

    def _logging_excepthook(
        exc_type: type[BaseException],
        exc_value: BaseException,
        exc_tb: types.TracebackType | None,
    ) -> None:
        logger.critical(
            "Unhandled exception", exc_info=(exc_type, exc_value, exc_tb)
        )
        _default_excepthook(exc_type, exc_value, exc_tb)

    sys.excepthook = _logging_excepthook


# ---------------------------------------------------------------------------
# Periodic CUDA cache cleanup (prevents VRAM creep during long sessions)
# ---------------------------------------------------------------------------

def start_cuda_cleanup_loop(
    interval_seconds: int = 300,
) -> threading.Thread | None:
    """Spawn a daemon thread that empties the CUDA cache every *interval_seconds*.

    Returns the thread (already started) or *None* if CUDA is unavailable.
    """
    try:
        import torch
        if not torch.cuda.is_available():
            return None
    except ImportError:
        return None

    def _cleanup_loop() -> None:
        while True:
            time.sleep(interval_seconds)
            try:
                import torch
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
            except Exception:
                logger.warning("CUDA cleanup failed", exc_info=True)

    t = threading.Thread(target=_cleanup_loop, name="CudaCleanup", daemon=True)
    t.start()
    logger.info("CUDA cleanup loop started (every %ds)", interval_seconds)
    return t


# ---------------------------------------------------------------------------
# Public API: activate all keep-alive protections
# ---------------------------------------------------------------------------

def activate_keepalive() -> None:
    """Call once at startup to enable all keep-alive protections."""
    install_thread_exception_hook()
    install_sys_excepthook()
    prevent_sleep()
    start_cuda_cleanup_loop()
