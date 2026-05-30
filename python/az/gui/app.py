from __future__ import annotations

import os

# Before any az import that loads torch.
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")

import sys

from PyQt6.QtCore import QTimer
from PyQt6.QtWidgets import QApplication

from az.config import Config
from az.gui.main_window import MainWindow
from az.keepalive import activate_keepalive
from az.session import (
    RestartSupervisor,
    is_seamless_restart,
    load_session,
    relaunch_application,
)


def main(argv: list[str] | None = None) -> int:
    argv = argv or sys.argv
    activate_keepalive()
    app = QApplication(argv)

    saved_cfg, window_state = load_session()
    cfg = saved_cfg or Config()
    win = MainWindow(cfg, window_state=window_state)
    win.sync_ui_from_config()
    win.show()

    if is_seamless_restart():
        win.status.showMessage("Session restored — resuming training from checkpoint…")

    supervisor = RestartSupervisor(cfg, win)

    def _on_restart_requested(reason: str) -> None:
        supervisor.stop()
        win.prepare_seamless_restart(reason)
        relaunch_application()
        QTimer.singleShot(200, app.quit)

    supervisor.restart_requested.connect(_on_restart_requested)

    win.assets.ensure_loaded()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
