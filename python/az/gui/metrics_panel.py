from __future__ import annotations

import math
from collections import deque

import pyqtgraph as pg
from PyQt6.QtCore import QTimer
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
from az.process_memory import process_rss_bytes

PLOT_HEIGHT = 108
WIN_RATE_WINDOW = 20
_CORE_METRICS = ("policy_loss", "value_loss", "total_loss", "lr", "games_per_min", "win_rate")
_LOSS_METRICS = frozenset({"policy_loss", "value_loss", "total_loss"})


def _style_plot(
    pw: pg.PlotWidget, title: str, color: str, *, show_fill: bool
) -> tuple[pg.PlotDataItem, pg.PlotDataItem | None]:
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
    pw.setMinimumHeight(PLOT_HEIGHT)
    pw.setMaximumHeight(PLOT_HEIGHT)
    pw.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
    vb = pw.getViewBox()
    vb.enableAutoRange(axis=pg.ViewBox.XYAxes, enable=False)

    fill: pg.PlotDataItem | None = None
    if show_fill:
        fill_pen = pg.mkPen(color, width=0)
        fill_brush = pg.mkBrush(color + "40")
        fill = pw.plot([], [], pen=fill_pen, brush=fill_brush, fillLevel=0)

    line_kw: dict = {
        "pen": pg.mkPen(color, width=2.0),
        "antialias": True,
    }
    if not show_fill:
        line_kw["symbol"] = "o"
        line_kw["symbolSize"] = 6
        line_kw["symbolBrush"] = pg.mkBrush(color)
        line_kw["symbolPen"] = pg.mkPen(BG_DEEPEST, width=1)
    line = pw.plot([], [], **line_kw)
    if fill is not None:
        line.setZValue(10)
    return line, fill


class MetricsPanel(QGroupBox):
    def __init__(
        self,
        window: int = 2_000,
        *,
        memory_interval_ms: int = 5_000,
        memory_window: int = 360,
        parent=None,
    ):
        super().__init__("Metrics", parent)
        self.setObjectName("metrics_panel")
        layout = QGridLayout(self)
        layout.setContentsMargins(10, 14, 10, 10)
        layout.setSpacing(8)
        pg.setConfigOptions(antialias=True, background=BG_DEEPEST, foreground=TEXT_SECONDARY)
        self._window = window
        self._memory_interval_ms = memory_interval_ms
        self._memory_elapsed_s = 0.0
        self._series: dict[str, deque] = {
            name: deque(maxlen=window) for name in _CORE_METRICS
        }
        self._series["memory_mb"] = deque(maxlen=memory_window)
        self._x: dict[str, deque] = {
            k: deque(maxlen=series.maxlen) for k, series in self._series.items()
        }
        self._lines: dict[str, pg.PlotDataItem] = {}
        self._fills: dict[str, pg.PlotDataItem | None] = {}
        self._plots: dict[str, pg.PlotWidget] = {}
        for i, name in enumerate(_CORE_METRICS):
            self._add_plot(layout, name, i // 2, i % 2)
        self._add_plot(layout, "memory_mb", 3, 0, colspan=2)
        self._plots["memory_mb"].setLabel("bottom", "Time", units="s")
        self._plots["memory_mb"].setLabel("left", "RSS", units="MB")
        self._step = 0
        self._games_count = 0
        self._last_game_time: float | None = None
        self._recent_scores: deque[float] = deque(maxlen=WIN_RATE_WINDOW)
        self._memory_timer = QTimer(self)
        self._memory_timer.setInterval(memory_interval_ms)
        self._memory_timer.timeout.connect(self._sample_memory)
        self._memory_timer.start()
        self._sample_memory()

    def _add_plot(
        self,
        layout: QGridLayout,
        name: str,
        row: int,
        col: int,
        *,
        colspan: int = 1,
    ) -> None:
        pw = pg.PlotWidget()
        color = METRIC_COLORS.get(name, "#8a8a8a")
        title = "Memory (MB)" if name == "memory_mb" else name.replace("_", " ").title()
        line, fill = _style_plot(
            pw,
            title,
            color,
            show_fill=name not in _LOSS_METRICS,
        )
        layout.addWidget(pw, row, col, 1, colspan)
        self._lines[name] = line
        self._fills[name] = fill
        self._plots[name] = pw

    def _sample_memory(self) -> None:
        mb = process_rss_bytes() / (1024 * 1024)
        self._push("memory_mb", mb, x=self._memory_elapsed_s)
        self._memory_elapsed_s += self._memory_interval_ms / 1000.0

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
        fill = self._fills[name]
        if fill is not None:
            fill.setData(xs, ys)
        if not xs:
            return
        vb = self._plots[name].getViewBox()
        y_min, y_max = min(ys), max(ys)
        if y_min == y_max:
            pad = max(abs(y_min) * 0.1, 0.01)
            y_min -= pad
            y_max += pad
        else:
            pad = (y_max - y_min) * 0.08
            y_min -= pad
            y_max += pad
        x_min, x_max = min(xs), max(xs)
        if x_min == x_max:
            x_pad = max(abs(x_min) * 0.05, 1.0)
            x_min -= x_pad
            x_max += x_pad
        else:
            x_pad = (x_max - x_min) * 0.04
            x_min -= x_pad
            x_max += x_pad
        vb.setRange(xRange=(x_min, x_max), yRange=(y_min, y_max), padding=0)
