from __future__ import annotations

from az.gui.metrics_panel import MetricsPanel
from az.process_memory import process_rss_bytes


def test_process_rss_bytes_positive():
    assert process_rss_bytes() > 0


def test_metrics_panel_memory_samples(qtbot, monkeypatch):
    seq = [100.0 * 1024 * 1024, 120.0 * 1024 * 1024, 140.0 * 1024 * 1024]
    monkeypatch.setattr(
        "az.gui.metrics_panel.process_rss_bytes",
        lambda: int(seq.pop(0)) if seq else int(140.0 * 1024 * 1024),
    )

    panel = MetricsPanel(window=100, memory_interval_ms=50, memory_window=10)
    qtbot.addWidget(panel)
    qtbot.wait(120)

    ys = list(panel._series["memory_mb"])
    assert len(ys) >= 2
    assert ys[0] == 100.0

    vb = panel._plots["memory_mb"].getViewBox()
    y_min, y_max = vb.viewRange()[1]
    assert y_max >= 100.0
    assert y_min <= 140.0
