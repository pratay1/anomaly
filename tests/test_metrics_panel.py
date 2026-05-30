from __future__ import annotations

from az.gui.metrics_panel import MetricsPanel
from az.ipc.events import TrainStep


def test_metrics_panel_autoranges_loss_plots(qtbot):
    panel = MetricsPanel(window=100)
    qtbot.addWidget(panel)

    step = TrainStep(
        step=1,
        policy_loss=6.2,
        value_loss=0.35,
        total_loss=6.55,
        lr=0.01,
    )
    panel.on_train_step(step)

    policy_vb = panel._plots["policy_loss"].getViewBox()
    y_min, y_max = policy_vb.viewRange()[1]
    assert y_max > 1.0
    assert y_min < 6.3
    assert len(panel._series["policy_loss"]) == 1
    assert panel._fills["policy_loss"] is None

    lr_vb = panel._plots["lr"].getViewBox()
    lr_min, lr_max = lr_vb.viewRange()[1]
    assert lr_max <= 0.02
    assert lr_min > -0.01


def test_metrics_panel_win_rate_from_games(qtbot):
    panel = MetricsPanel(window=100)
    qtbot.addWidget(panel)

    panel.on_game_finished(agent_score=1.0)
    panel.on_game_finished(agent_score=0.0)

    ys = list(panel._series["win_rate"])
    assert len(ys) == 2
    assert ys[-1] == 0.5
