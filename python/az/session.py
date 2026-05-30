from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from dataclasses import fields
from pathlib import Path
from typing import Any

from platformdirs import user_data_dir
from PyQt6.QtCore import QObject, QTimer, pyqtSignal

from az.config import Config
from az.process_memory import process_rss_bytes

SESSION_VERSION = 1
SEAMLESS_RESTART_ENV = "AZ_SEAMLESS_RESTART"


def session_path() -> Path:
    root = Path(user_data_dir("anomaly", appauthor="anomaly"))
    root.mkdir(parents=True, exist_ok=True)
    return root / "session.json"


def config_to_dict(cfg: Config) -> dict[str, Any]:
    data: dict[str, Any] = {}
    for f in fields(cfg):
        val = getattr(cfg, f.name)
        if isinstance(val, Path):
            data[f.name] = str(val)
        elif f.name == "lr_schedule":
            data[f.name] = [list(pair) for pair in val]
        else:
            data[f.name] = val
    return data


def config_from_dict(data: dict[str, Any]) -> Config:
    cfg = Config()
    path_fields = {"stockfish_path", "run_dir", "brain_path"}
    for f in fields(Config):
        if f.name not in data:
            continue
        val = data[f.name]
        if f.name in path_fields:
            val = Path(val)
        elif f.name == "lr_schedule":
            val = [tuple(pair) for pair in val]
        setattr(cfg, f.name, val)
    return cfg


def save_session(*, cfg: Config, window: dict[str, Any] | None = None) -> None:
    payload = {
        "version": SESSION_VERSION,
        "saved_at": time.time(),
        "config": config_to_dict(cfg),
        "window": window or {},
    }
    path = session_path()
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    tmp.replace(path)


def load_session() -> tuple[Config | None, dict[str, Any]]:
    path = session_path()
    if not path.is_file():
        return None, {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None, {}
    cfg_data = payload.get("config")
    if not isinstance(cfg_data, dict):
        return None, {}
    window = payload.get("window")
    if not isinstance(window, dict):
        window = {}
    return config_from_dict(cfg_data), window


def effective_restart_interval_s(rss_bytes: int, cfg: Config) -> float:
    """Uptime limit before a seamless restart; shrinks exponentially above the memory threshold."""
    base = float(cfg.auto_restart_interval_s)
    threshold = cfg.auto_restart_memory_threshold_mb * 1024 * 1024
    if rss_bytes <= threshold:
        return base
    over_mb = (rss_bytes - threshold) / (1024 * 1024)
    steps = over_mb / max(cfg.auto_restart_memory_step_mb, 1)
    interval = base / (2 ** steps)
    return max(float(cfg.auto_restart_min_interval_s), interval)


def is_seamless_restart() -> bool:
    return os.environ.get(SEAMLESS_RESTART_ENV) == "1"


def relaunch_application() -> None:
    env = os.environ.copy()
    env[SEAMLESS_RESTART_ENV] = "1"
    subprocess.Popen(
        [sys.executable, "-m", "az.scripts.train"],
        env=env,
        cwd=os.getcwd(),
        close_fds=os.name != "nt",
    )


class RestartSupervisor(QObject):
    """Monitors uptime and RSS; triggers a seamless restart when limits are exceeded."""

    restart_requested = pyqtSignal(str)

    def __init__(self, cfg: Config, parent: QObject | None = None):
        super().__init__(parent)
        self.cfg = cfg
        self._started = time.monotonic()
        self._timer = QTimer(self)
        self._timer.setInterval(cfg.auto_restart_check_interval_ms)
        self._timer.timeout.connect(self._check)
        self._timer.start()

    def stop(self) -> None:
        self._timer.stop()

    def uptime_s(self) -> float:
        return time.monotonic() - self._started

    def _check(self) -> None:
        if not self.cfg.auto_restart_enabled:
            return
        rss = process_rss_bytes()
        limit = effective_restart_interval_s(rss, self.cfg)
        uptime = self.uptime_s()
        if uptime < limit:
            return
        rss_mb = rss / (1024 * 1024)
        reason = (
            f"session refresh after {uptime / 60:.0f} min "
            f"(memory {rss_mb:.0f} MB, limit {limit / 60:.1f} min)"
        )
        self.restart_requested.emit(reason)
