from __future__ import annotations

import json

from az.config import Config
from az.session import (
    SESSION_VERSION,
    RestartSupervisor,
    config_from_dict,
    config_to_dict,
    effective_restart_interval_s,
    load_session,
    save_session,
)


def test_config_roundtrip():
    cfg = Config(
        training_opponent="stockfish",
        replay_capacity=12_000,
        num_workers=2,
    )
    restored = config_from_dict(config_to_dict(cfg))
    assert restored.training_opponent == "stockfish"
    assert restored.replay_capacity == 12_000
    assert restored.num_workers == 2


def test_effective_restart_interval_below_threshold():
    cfg = Config(auto_restart_interval_s=3600, auto_restart_memory_threshold_mb=2048)
    assert effective_restart_interval_s(2048 * 1024 * 1024, cfg) == 3600.0


def test_effective_restart_interval_scales_with_memory():
    cfg = Config(
        auto_restart_interval_s=3600,
        auto_restart_memory_threshold_mb=2048,
        auto_restart_memory_step_mb=512,
        auto_restart_min_interval_s=60,
    )
    rss = int(2.5 * 1024 * 1024 * 1024)
    assert effective_restart_interval_s(rss, cfg) == 1800.0
    rss = int(3.0 * 1024 * 1024 * 1024)
    assert effective_restart_interval_s(rss, cfg) == 900.0


def test_effective_restart_interval_respects_floor():
    cfg = Config(
        auto_restart_interval_s=3600,
        auto_restart_memory_threshold_mb=2048,
        auto_restart_memory_step_mb=512,
        auto_restart_min_interval_s=120,
    )
    rss = int(8 * 1024 * 1024 * 1024)
    assert effective_restart_interval_s(rss, cfg) == 120.0


def test_save_and_load_session(tmp_path, monkeypatch):
    session_file = tmp_path / "session.json"
    monkeypatch.setattr("az.session.session_path", lambda: session_file)

    cfg = Config(training_opponent="self", num_workers=3)
    window = {"x": 10, "y": 20, "width": 1200, "height": 800, "splitter_sizes": [700, 500]}
    save_session(cfg=cfg, window=window)

    loaded_cfg, loaded_window = load_session()
    assert loaded_cfg is not None
    assert loaded_cfg.training_opponent == "self"
    assert loaded_cfg.num_workers == 3
    assert loaded_window["splitter_sizes"] == [700, 500]

    payload = json.loads(session_file.read_text(encoding="utf-8"))
    assert payload["version"] == SESSION_VERSION


def test_restart_supervisor_emits_when_uptime_exceeded(qtbot, monkeypatch):
    cfg = Config(
        auto_restart_enabled=True,
        auto_restart_interval_s=60,
        auto_restart_check_interval_ms=50,
    )
    monkeypatch.setattr("az.session.process_rss_bytes", lambda: 512 * 1024 * 1024)
    t = {"now": 1000.0}
    monkeypatch.setattr("az.session.time.monotonic", lambda: t["now"])

    supervisor = RestartSupervisor(cfg)
    reasons: list[str] = []
    supervisor.restart_requested.connect(reasons.append)

    t["now"] = 1070.0
    qtbot.wait(80)
    assert reasons
