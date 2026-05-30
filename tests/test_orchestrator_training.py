from __future__ import annotations

import threading

import pytest

try:
    from PyQt6.QtWidgets import QApplication  # noqa: F401

    HAS_PYQT6 = True
except ImportError:
    HAS_PYQT6 = False

from az.config import Config
from az.training.orchestrator import TrainerOrchestrator


def test_run_learner_steps_without_qt_calls_in_place():
    cfg = Config()
    cfg.replay_capacity = 128
    orch = TrainerOrchestrator(cfg)
    assert orch.learner is not None
    called = threading.Event()

    def fake_train(steps: int) -> list:
        called.set()
        return []

    orch.learner.train_iteration = fake_train  # type: ignore[method-assign]
    orch._run_learner_steps(3)
    assert called.is_set()


@pytest.mark.skipif(not HAS_PYQT6, reason="PyQt6 required")
def test_run_learner_steps_dispatches_to_gui_thread(qtbot):
    from PyQt6.QtCore import QCoreApplication, QThread
    from PyQt6.QtWidgets import QApplication

    app = QApplication.instance() or QApplication([])
    cfg = Config()
    orch = TrainerOrchestrator(cfg)
    assert orch.learner is not None
    seen: list[int] = []

    def fake_train(steps: int) -> list:
        seen.append(steps)
        assert QThread.currentThread() == app.thread()
        return []

    orch.learner.train_iteration = fake_train  # type: ignore[method-assign]

    done = threading.Event()

    def worker():
        orch._run_learner_steps(7)
        done.set()

    t = threading.Thread(target=worker, name="TrainingLoopTest")
    t.start()
    qtbot.waitUntil(done.is_set, timeout=5000)
    t.join(timeout=2)
    assert seen == [7]
    QCoreApplication.processEvents()
