from __future__ import annotations

import logging
import threading
import time
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from queue import Empty, Queue

import az._az_core as core
from az.brain import align_cfg_with_brain, load_brain, save_brain
from az.checkpoint import CheckpointManager
from az.config import Config
from az.eval.arena import play_random_vs_random
from az.ipc.events import ArenaResult, CheckpointSaved, IterationComplete
from az.network.resnet import AlphaZeroResNet
from az.training.central_learner import CentralLearner
from az.training.inference_server import InferenceServer
from az.training.replay_buffer import ReplayBuffer
from az.training.selfplay_worker import ParallelSelfPlayPool
from az.training.stockfish_paths import stockfish_path_error

logger = logging.getLogger(__name__)

try:
    from PyQt6.QtCore import (
        Q_ARG,
        QCoreApplication,
        QMetaObject,
        QObject,
        QThread,
        Qt,
        pyqtSignal,
        pyqtSlot,
    )
except ImportError:
    QObject = object  # type: ignore
    QThread = threading.Thread  # type: ignore
    QCoreApplication = None  # type: ignore
    QMetaObject = None  # type: ignore
    Qt = None  # type: ignore
    Q_ARG = None  # type: ignore

    def pyqtSignal(*args, **kwargs):  # type: ignore
        return None

    def pyqtSlot(*args, **kwargs):  # type: ignore
        def deco(fn):
            return fn

        return deco


class TrainerOrchestrator(QObject):
    checkpoint_saved = pyqtSignal(object)
    arena_result = pyqtSignal(object)
    iteration_complete = pyqtSignal(object)
    move_played = pyqtSignal(object)
    game_finished = pyqtSignal(object)
    mcts_visits = pyqtSignal(str, list)
    training_error = pyqtSignal(str)

    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.cfg.games_per_iteration = self.cfg.games_per_selfplay_iteration()
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.cfg.run_dir = Path("runs") / ts
        self.stop_event = threading.Event()
        self._selfplay_events: Queue = Queue()
        self.queue = core.InferenceQueue()
        align_cfg_with_brain(cfg)
        self.model = AlphaZeroResNet(cfg)
        self.buffer = ReplayBuffer(
            cfg.replay_capacity,
            cfg.encoding_channels * 64,
            cfg.policy_size,
        )
        self.ckpt = CheckpointManager(self.cfg.run_dir)
        self.inference: InferenceServer | None = None
        self.learner: CentralLearner | None = None
        self.selfplay: ParallelSelfPlayPool | None = None
        self._threads: list[threading.Thread] = []
        self._iteration = 0
        self._visit_buffer: dict[str, list] = {}
        load_brain(self.model, device="cpu")
        # SGD must run on the Qt GUI thread (see _run_learner_steps).
        self.learner = CentralLearner(self.model, self.buffer, self.cfg, self.stop_event)

    def start(self) -> None:
        self.inference = InferenceServer(self.queue, self.model, self.cfg, self.stop_event)
        self.inference.start()
        self.selfplay = ParallelSelfPlayPool(
            self.queue, self.buffer, self.cfg, self.stop_event, self._selfplay_events
        )

        def training_loop():
            while not self.stop_event.is_set():
                if self.selfplay is None or self.learner is None or self.inference is None:
                    time.sleep(0.2)
                    continue
                try:
                    self._run_training_iteration()
                except Exception as exc:
                    self._handle_training_failure(exc)
                    time.sleep(1.0)

        t_train = threading.Thread(target=training_loop, daemon=True, name="TrainingLoop")
        t_train.start()
        self._threads = [t_train, self.inference]

        def monitor():
            while not self.stop_event.is_set():
                time.sleep(30)
                if self.learner is None:
                    continue
                try:
                    step = self.learner.global_step
                    path = self.ckpt.save(
                        self.model,
                        self.cfg,
                        step,
                        iteration=self._iteration,
                    )
                    self.checkpoint_saved.emit(CheckpointSaved(str(path), step))
                except Exception as exc:
                    self._handle_training_failure(exc)

        threading.Thread(target=monitor, daemon=True, name="Monitor").start()

    def _run_training_iteration(self) -> None:
        assert self.selfplay is not None and self.learner is not None and self.inference is not None
        n_games = self.cfg.games_per_selfplay_iteration()
        self.cfg.games_per_iteration = n_games
        self.selfplay.run_iteration(n_games)
        if self.stop_event.is_set():
            return
        self._run_learner_steps(self.cfg.train_steps_per_iteration)
        step = self.learner.global_step
        self._iteration += 1
        try:
            brain_path = save_brain(
                self.model,
                self.cfg,
                step,
                iteration=self._iteration,
                run_dir=self.cfg.run_dir,
            )
            self.ckpt.save(
                self.model,
                self.cfg,
                step,
                iteration=self._iteration,
            )
            self.inference.reload_weights(self.model.state_dict())
        except Exception as exc:
            self._handle_training_failure(exc)
            return
        self.iteration_complete.emit(
            IterationComplete(
                iteration=self._iteration,
                games_finished=self.cfg.games_per_iteration,
                train_steps=self.cfg.train_steps_per_iteration,
                brain_path=str(brain_path),
            )
        )
        if step > 0 and step % self.cfg.arena_every_steps == 0:
            try:
                wins, draws, losses = play_random_vs_random(self.cfg.arena_num_games)
            except Exception as exc:
                self._handle_training_failure(exc)
                return
            total = max(wins + draws + losses, 1)
            wr = wins / total
            self.arena_result.emit(
                ArenaResult(win_rate=wr, draws=draws, wins=wins, losses=losses)
            )
        time.sleep(0.01)

    @pyqtSlot(int)
    def _train_iteration_on_gui_thread(self, steps: int) -> None:
        if self.learner is not None:
            self.learner.train_iteration(steps)

    def _run_learner_steps(self, steps: int) -> None:
        """Run SGD on the Qt main thread when a GUI event loop is active."""
        if self.learner is None:
            return
        app = QCoreApplication.instance() if QCoreApplication is not None else None
        on_gui = (
            app is not None
            and QThread is not threading.Thread
            and QThread.currentThread() == app.thread()
        )
        if on_gui or app is None or QMetaObject is None:
            self.learner.train_iteration(steps)
            return
        QMetaObject.invokeMethod(
            self,
            "_train_iteration_on_gui_thread",
            Qt.ConnectionType.BlockingQueuedConnection,
            Q_ARG(int, steps),
        )

    def _handle_training_failure(self, exc: Exception) -> None:
        msg = f"{type(exc).__name__}: {exc}"
        logger.warning("training failure (continuing): %s", msg, exc_info=exc)
        self.training_error.emit(msg)
        if self.cfg.training_opponent == "stockfish":
            self.restart_stockfish()

    def restart_stockfish(self) -> None:
        if self.selfplay is not None:
            self.selfplay.restart_stockfish()

    def stop(self) -> None:
        self.stop_event.set()
        if self.selfplay is not None:
            self.selfplay._close_stockfish()

    def set_training_opponent(self, opponent: str) -> str | None:
        """Switch training mode at runtime. Returns an error message on failure."""
        if opponent == "stockfish":
            err = stockfish_path_error(self.cfg.stockfish_path)
            if err:
                return err
        if self.selfplay is not None:
            self.selfplay.set_training_opponent(opponent)
        else:
            self.cfg.training_opponent = opponent
        self.cfg.games_per_iteration = self.cfg.games_per_selfplay_iteration()
        return None

    def flush_selfplay_events(self) -> int:
        """Drain worker event queue and emit Qt signals on the caller thread."""
        count = 0
        while True:
            try:
                kind, *payload = self._selfplay_events.get_nowait()
            except Empty:
                break
            count += 1
            if kind == "mcts_visits":
                self._visit_buffer[payload[0]] = payload[1]
            elif kind == "move_played":
                mv = payload[0]
                visits = self._visit_buffer.pop(mv.fen, None)
                if visits:
                    mv = replace(mv, visits=visits)
                self.move_played.emit(mv)
            elif kind == "game_finished":
                self._visit_buffer.clear()
                self.game_finished.emit(payload[0])
        return count

    @property
    def learner_signals(self) -> CentralLearner | None:
        return self.learner

    @property
    def selfplay_signals(self) -> TrainerOrchestrator:
        return self


class TrainerOrchestratorThread(QThread):
    training_started = pyqtSignal()

    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.orchestrator = TrainerOrchestrator(cfg)

    def run(self) -> None:
        self.orchestrator.start()
        self.training_started.emit()
        while not self.isInterruptionRequested():
            time.sleep(0.2)
