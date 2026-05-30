from __future__ import annotations

import math
from collections import deque

import pyqtgraph as pg
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import QGridLayout, QGroupBox, QSizePolicy

from az.gui.theme import (
    BG_DEEPEST,
    BORDER_SUBTLE,
    METRIC_COLORS,
    TEXT_DISABLED,
    TEXT_SECONDARY,
)
from az.ipc.events import TrainStep

PLOT_HEIGHT = 108
WIN_RATE_WINDOW = 20


def _style_plot(
    pw: pg.PlotWidget, title: str, color: str
) -> tuple[pg.PlotDataItem, pg.PlotDataItem]:
    pw.setTitle(title, color=TEXT_SECONDARY, size="9pt")
    pw.setBackground(BG_DEEPEST)
    pw.showGrid(x=True, y=True, alpha=0.08)
    pw.getPlotItem().hideAxis("top")
    pw.getPlotItem().hideAxis("right")
    pw.getPlotItem().layout.setContentsMargins(4, 4, 4, 4)
    for axis_name in ("left", "bottom"):
        ax = pw.getPlotItem().getAxis(axis_name)
        ax.setPen(pg.mkPen(BORDER_SUBTLE, width=1))
        ax.setTextPen(pg.mkPen(TEXT_DISABLED))
        ax.setStyle(tickFont=QFont("Segoe UI", 8))
    pw.setDownsampling(auto=True, mode="peak")
    pw.setClipToView(True)
    pw.setMinimumHeight(PLOT_HEIGHT)
    pw.setMaximumHeight(PLOT_HEIGHT)
    pw.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
    vb = pw.getViewBox()
    vb.enableAutoRange(axis=pg.ViewBox.XYAxes, enable=True)
    fill_pen = pg.mkPen(color, width=0)
    fill_brush = pg.mkBrush(color + "40")
    fill = pw.plot([], [], pen=fill_pen, brush=fill_brush, fillLevel=0)
    line = pw.plot([], [], pen=pg.mkPen(color, width=1.8), antialias=True)
    return line, fill


class MetricsPanel(QGroupBox):
    def __init__(self, window: int = 2_000, parent=None):
        super().__init__("Metrics", parent)
        self.setObjectName("metrics_panel")
        layout = QGridLayout(self)
        layout.setContentsMargins(10, 14, 10, 10)
        layout.setSpacing(8)
        pg.setConfigOptions(antialias=True, background=BG_DEEPEST, foreground=TEXT_SECONDARY)
        self._window = window
        self._series: dict[str, deque] = {
            "policy_loss": deque(maxlen=window),
            "value_loss": deque(maxlen=window),
            "total_loss": deque(maxlen=window),
            "lr": deque(maxlen=window),
            "games_per_min": deque(maxlen=window),
            "win_rate": deque(maxlen=window),
        }
        self._x: dict[str, deque] = {k: deque(maxlen=window) for k in self._series}
        self._lines: dict[str, pg.PlotDataItem] = {}
        self._fills: dict[str, pg.PlotDataItem] = {}
        self._plots: dict[str, pg.PlotWidget] = {}
        labels = list(self._series.keys())
        for i, name in enumerate(labels):
            pw = pg.PlotWidget()
            color = METRIC_COLORS.get(name, "#8a8a8a")
            line, fill = _style_plot(pw, name.replace("_", " ").title(), color)
            layout.addWidget(pw, i // 2, i % 2)
            self._lines[name] = line
            self._fills[name] = fill
            self._plots[name] = pw
        self._step = 0
        self._games_count = 0
        self._last_game_time: float | None = None
        self._recent_scores: deque[float] = deque(maxlen=WIN_RATE_WINDOW)

    def on_train_step(self, step: TrainStep) -> None:
        self._step = step.step
        self._push("policy_loss", step.policy_loss, x=step.step)
        self._push("value_loss", step.value_loss, x=step.step)
        self._push("total_loss", step.total_loss, x=step.step)
        self._push("lr", step.lr, x=step.step)

    def on_game_finished(self, agent_score: float | None = None) -> None:
        import time

        now = time.time()
        self._games_count += 1
        if self._last_game_time:
            dt = now - self._last_game_time
            if dt > 0:
                self._push("games_per_min", 60.0 / dt, x=self._games_count)
        self._last_game_time = now
        if agent_score is not None:
            self._recent_scores.append(agent_score)
            avg = sum(self._recent_scores) / len(self._recent_scores)
            self._push("win_rate", avg, x=self._games_count)

    def on_arena(self, win_rate: float) -> None:
        self._push("win_rate", win_rate, x=self._step)

    def _push(self, name: str, y: float, *, x: int | float | None = None) -> None:
        if not math.isfinite(y):
            return
        x_val = self._step if x is None else x
        self._series[name].append(y)
        self._x[name].append(x_val)
        xs = list(self._x[name])
        ys = list(self._series[name])
        self._lines[name].setData(xs, ys)
        self._fills[name].setData(xs, ys)
        if not xs:
            return
        vb = self._plots[name].getViewBox()
        vb.enableAutoRange(axis=pg.ViewBox.XYAxes, enable=True)
        vb.autoRange(padding=0.08)
