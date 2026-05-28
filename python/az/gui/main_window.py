from __future__ import annotations

import chess
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QSplitter,
    QStatusBar,
    QVBoxLayout,
    QWidget,
)

from az.brain import resolve_brain_path
from az.config import Config
from az.gui.board_view import BoardView
from az.gui.games_panel import GamesPanel
from az.gui.mcts_panel import MCTSPanel
from az.gui.metrics_panel import MetricsPanel
from az.gui.multi_game_grid import GameState, MultiGameGridDialog
from az.gui.piece_assets import PieceAssetManager
from az.gui.solo_game_dialog import SoloGameDialog
from az.gui.theme import DARK_STYLESHEET
from az.ipc.events import GameFinished, MovePlayed, TrainStep
from az.training.orchestrator import TrainerOrchestratorThread


class MainWindow(QMainWindow):
    def __init__(self, cfg: Config | None = None):
        super().__init__()
        self.cfg = cfg or Config()
        self.cfg.emit_selfplay_visits = True
        self.setWindowTitle("Anomaly")
        self.resize(1320, 820)
        self.setMinimumSize(920, 620)
        self.setStyleSheet(DARK_STYLESHEET)

        central = QWidget()
        self.setCentralWidget(central)
        root = QHBoxLayout(central)
        root.setContentsMargins(12, 12, 12, 8)
        root.setSpacing(0)

        left = QVBoxLayout()
        left.setSpacing(10)
        left.setContentsMargins(0, 0, 8, 0)

        title_row = QVBoxLayout()
        title_row.setSpacing(2)
        self.title = QLabel("Anomaly")
        self.title.setObjectName("title")
        mode_label = (
            "Stockfish training"
            if self.cfg.training_opponent == "stockfish"
            else "self-play"
        )
        subtitle = QLabel(f"{mode_label}")
        subtitle.setObjectName("subtitle")
        self.subtitle = subtitle
        title_row.addWidget(self.title)
        title_row.addWidget(subtitle)
        left.addLayout(title_row)

        self.assets = PieceAssetManager(self)
        board_frame = QFrame()
        board_frame.setObjectName("board_frame")
        board_layout = QVBoxLayout(board_frame)
        board_layout.setContentsMargins(6, 6, 6, 6)
        self.board_view = BoardView(self.assets, anim_ms=self.cfg.board_anim_ms)
        self.board_view.setMinimumSize(440, 440)
        board_layout.addWidget(self.board_view)
        left.addWidget(board_frame, stretch=1)

        self.mcts_panel = MCTSPanel(self.board_view)
        left.addWidget(self.mcts_panel)

        btn_row = QHBoxLayout()
        btn_row.setSpacing(8)
        self.btn_play = QPushButton("Play vs Brain")
        self.btn_play.clicked.connect(self._play_vs_ckpt)
        btn_row.addWidget(self.btn_play, stretch=1)
        self.btn_grid = QPushButton("Game Grid")
        self.btn_grid.clicked.connect(self._show_game_grid)
        btn_row.addWidget(self.btn_grid, stretch=1)
        left.addLayout(btn_row)

        self.btn_stockfish = QPushButton("Train Against Stockfish")
        self.btn_stockfish.setCheckable(True)
        self.btn_stockfish.setChecked(self.cfg.training_opponent == "stockfish")
        self.btn_stockfish.setToolTip(
            "Takes effect at the next iteration start (not instant)"
        )
        self.btn_stockfish.toggled.connect(self._on_stockfish_toggled)
        left.addWidget(self.btn_stockfish)

        right = QVBoxLayout()
        right.setSpacing(10)
        right.setContentsMargins(8, 0, 0, 0)
        self.metrics = MetricsPanel(self.cfg.metrics_window)
        self.games_panel = GamesPanel()
        right.addWidget(self.metrics, stretch=3)
        right.addWidget(self.games_panel, stretch=2)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setHandleWidth(3)
        lw, rw = QWidget(), QWidget()
        lw.setLayout(left)
        rw.setLayout(right)
        splitter.addWidget(lw)
        splitter.addWidget(rw)
        splitter.setSizes([720, 520])
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 2)
        root.addWidget(splitter)

        self.status = QStatusBar()
        self.setStatusBar(self.status)
        self.status.showMessage("Loading piece assets…")

        self.assets.progress.connect(
            lambda d, t: self.status.showMessage(f"Downloading pieces {d}/{t}…")
        )
        self.assets.ready.connect(self._on_assets_ready)

        self.trainer_thread: TrainerOrchestratorThread | None = None
        self._focused_game_id = 0
        self._game_states: dict[int, GameState] = {
            i: GameState() for i in range(self.cfg.num_workers)
        }
        self._grid_dialog: MultiGameGridDialog | None = None
        self._solo_dialogs: dict[int, SoloGameDialog] = {}
        self._event_timer = QTimer(self)
        self._event_timer.setInterval(33)
        self._event_timer.timeout.connect(self._flush_trainer_events)
        self._ply_token = 0
        self._latest_post_fen = chess.STARTING_FEN

    def _on_assets_ready(self) -> None:
        brain = resolve_brain_path()
        self.status.showMessage(f"Assets ready. Brain: {brain.name}. Starting training…")
        self.trainer_thread = TrainerOrchestratorThread(self.cfg)
        o = self.trainer_thread.orchestrator
        o.checkpoint_saved.connect(
            lambda ck: self.status.showMessage(f"Checkpoint saved: {ck.path}")
        )
        o.arena_result.connect(lambda ar: self.metrics.on_arena(ar.win_rate))
        o.iteration_complete.connect(self._on_iteration_complete)
        o.training_error.connect(self._on_training_error)
        self.trainer_thread.training_started.connect(self._wire_trainer_signals)
        self.trainer_thread.start()

    def _wire_trainer_signals(self) -> None:
        o = self.trainer_thread.orchestrator if self.trainer_thread else None
        if not o:
            return
        if o.learner:
            o.learner.train_step.connect(self._on_train)
        o.game_finished.connect(self._on_game)
        o.move_played.connect(self._on_move)
        self._event_timer.start()

    def _flush_trainer_events(self) -> None:
        o = self.trainer_thread.orchestrator if self.trainer_thread else None
        if o:
            o.flush_selfplay_events()

    def _active_game_count(self) -> int:
        return 1 if self.cfg.training_opponent == "stockfish" else self.cfg.num_workers

    def _reset_training_game_states(self) -> None:
        n = self._active_game_count()
        self._game_states = {i: GameState() for i in range(n)}
        self._focused_game_id = 0
        self._ply_token += 1
        self._latest_post_fen = chess.STARTING_FEN
        self.board_view.clear_last_move()
        self.board_view.set_fen(chess.STARTING_FEN, animated=False)
        self.mcts_panel.set_searching()

    def _on_training_error(self, message: str) -> None:
        try:
            self._on_training_error_impl(message)
        except Exception as exc:
            self.status.showMessage(f"UI error (training continues): {exc}")

    def _on_training_error_impl(self, message: str) -> None:
        self.status.showMessage(f"Training error (continuing): {message}")
        orch = self.trainer_thread.orchestrator if self.trainer_thread else None
        if orch is not None and self.cfg.training_opponent == "stockfish":
            orch.restart_stockfish()

    def _on_iteration_complete(self, event) -> None:
        try:
            self._on_iteration_complete_impl(event)
        except Exception as exc:
            self.status.showMessage(f"UI error (training continues): {exc}")

    def _on_iteration_complete_impl(self, event) -> None:
        self.status.showMessage(
            f"Iteration {event.iteration} complete · brain updated: {event.brain_path}"
        )
        for gid in range(self._active_game_count()):
            self._game_states[gid] = GameState()
        if self._grid_dialog is not None:
            self._grid_dialog.reset_iteration(event.iteration)
        self._ply_token += 1
        self._latest_post_fen = chess.STARTING_FEN
        self.board_view.clear_last_move()
        self.board_view.set_fen(chess.STARTING_FEN, animated=False)
        self.mcts_panel.set_searching()

    def _on_train(self, step: TrainStep) -> None:
        try:
            self._on_train_impl(step)
        except Exception as exc:
            self.status.showMessage(f"UI error (training continues): {exc}")

    def _on_train_impl(self, step: TrainStep) -> None:
        self.metrics.on_train_step(step)
        self.status.showMessage(
            f"Step {step.step} | loss {step.total_loss:.4f} | lr {step.lr:.5f}"
        )

    def _on_game(self, game: GameFinished) -> None:
        try:
            self._on_game_impl(game)
        except Exception as exc:
            self.status.showMessage(f"UI error (training continues): {exc}")

    @staticmethod
    def _result_to_text(result: str) -> str:
        if result == "1-0":
            return "White wins"
        if result == "0-1":
            return "Black wins"
        return "Draw"

    def _on_game_impl(self, game: GameFinished) -> None:
        gid = game.game_id
        if gid in self._game_states:
            st = self._game_states[gid]
            st.result = game.result
            st.active = False
            if game.moves_uci:
                st.moves = list(game.moves_uci)
        if self._grid_dialog is not None:
            self._grid_dialog.on_game_finished(game)
        for dlg in self._solo_dialogs.values():
            dlg.on_game_finished(game)
        if gid == self._focused_game_id:
            self.games_panel.add_game(game)
            self.metrics.on_game_finished()
            self._ply_token += 1
            token = self._ply_token
            final_fen = self._game_states[gid].fen if gid in self._game_states else chess.STARTING_FEN
            self._latest_post_fen = final_fen
            self.board_view.set_fen(final_fen, animated=False)
            self.board_view.show_result_overlay(self._result_to_text(game.result))
            QTimer.singleShot(2500, lambda t=token: self._reset_after_game(t))

    def _reset_after_game(self, token: int) -> None:
        if token != self._ply_token:
            return
        self._ply_token += 1
        self._latest_post_fen = chess.STARTING_FEN
        self.board_view.set_fen(chess.STARTING_FEN, animated=False)
        self.mcts_panel.set_searching()

    @staticmethod
    def _fen_after_move(mv: MovePlayed, fallback_fen: str) -> str:
        try:
            board = chess.Board(mv.fen)
            board.push(chess.Move.from_uci(mv.uci))
            return board.fen()
        except Exception:
            try:
                board = chess.Board(fallback_fen)
                board.push(chess.Move.from_uci(mv.uci))
                return board.fen()
            except Exception:
                return fallback_fen

    def _on_move(self, mv: MovePlayed) -> None:
        try:
            self._on_move_impl(mv)
        except Exception as exc:
            self.status.showMessage(f"UI error (training continues): {exc}")

    def _on_move_impl(self, mv: MovePlayed) -> None:
        gid = mv.game_id
        if gid in self._game_states:
            st = self._game_states[gid]
            st.fen = self._fen_after_move(mv, st.fen)
            st.moves.append(mv.uci)
        if self._grid_dialog is not None:
            self._grid_dialog.on_move(mv)
        for dlg in self._solo_dialogs.values():
            dlg.on_move(mv)
        if gid == self._focused_game_id:
            post = self._game_states[gid].fen
            self._show_focused_ply(mv, post)

    def _show_focused_ply(self, mv: MovePlayed, post_fen: str) -> None:
        """Show MCTS on the pre-move position, then apply the move to the board.

        Uses a ply token so timers from superseded moves cannot revert the
        board. We always render the *latest* known post-FEN — never the
        captured one — to avoid back-and-forth flicker when moves arrive
        faster than the reveal delay.
        """
        self._ply_token += 1
        token = self._ply_token
        self._latest_post_fen = post_fen
        self.board_view.clear_last_move()
        from_sq, to_sq = self._move_squares(mv.uci)
        if self.cfg.emit_selfplay_visits and mv.visits and mv.fen:
            self.board_view.set_fen(mv.fen, animated=False)
            self.mcts_panel.update_visits(mv.fen, mv.visits)
            delay = max(0, self.cfg.mcts_reveal_ms)
            t, fs, ts = token, from_sq, to_sq
            QTimer.singleShot(delay, lambda: self._commit_ply(t, fs, ts))
        else:
            self.board_view.set_fen(post_fen, animated=True)
            if from_sq is not None and to_sq is not None:
                self.board_view.set_last_move(from_sq, to_sq)
            self.mcts_panel.set_searching()

    def _commit_ply(self, token: int, from_sq: int | None, to_sq: int | None) -> None:
        if token != self._ply_token:
            return
        self.board_view.set_fen(self._latest_post_fen, animated=True)
        if from_sq is not None and to_sq is not None:
            self.board_view.set_last_move(from_sq, to_sq)
        self.mcts_panel.set_searching()

    @staticmethod
    def _move_squares(uci: str) -> tuple[int | None, int | None]:
        try:
            m = chess.Move.from_uci(uci)
            return m.from_square, m.to_square
        except Exception:
            return None, None

    def _show_game_grid(self) -> None:
        if self._grid_dialog is None:
            self._grid_dialog = MultiGameGridDialog(
                self.cfg.num_workers,
                self.assets,
                self,
                anim_ms=self.cfg.board_anim_ms,
            )
            self._grid_dialog.focus_requested.connect(self._set_focused_game)
            self._grid_dialog.solo_requested.connect(self._open_solo_game)
            self._grid_dialog.finished.connect(self._on_grid_closed)
        for gid, st in self._game_states.items():
            if gid in self._grid_dialog._boards:
                self._grid_dialog._boards[gid].set_fen(st.fen)
        self._grid_dialog.show()
        self._grid_dialog.raise_()
        self._grid_dialog.activateWindow()

    def _on_grid_closed(self) -> None:
        pass

    def _set_focused_game(self, game_id: int) -> None:
        self._focused_game_id = game_id
        st = self._game_states.get(game_id)
        if st:
            self._ply_token += 1
            self._latest_post_fen = st.fen
            self.board_view.clear_last_move()
            self.board_view.set_fen(st.fen, animated=False)
        self.status.showMessage(f"Focused on game {game_id + 1}")

    def _open_solo_game(self, game_id: int) -> None:
        if game_id in self._solo_dialogs and self._solo_dialogs[game_id].isVisible():
            self._solo_dialogs[game_id].raise_()
            return
        st = self._game_states.get(game_id)
        fen = st.fen if st else chess.STARTING_FEN
        dlg = SoloGameDialog(game_id, self.assets, fen, self)
        self._solo_dialogs[game_id] = dlg
        dlg.finished.connect(lambda: self._solo_dialogs.pop(game_id, None))
        dlg.show()

    def _on_stockfish_toggled(self, enabled: bool) -> None:
        opponent = "stockfish" if enabled else "self"
        orch = self.trainer_thread.orchestrator if self.trainer_thread else None
        if orch is not None:
            err = orch.set_training_opponent(opponent)
            if err:
                self.btn_stockfish.blockSignals(True)
                self.btn_stockfish.setChecked(not enabled)
                self.btn_stockfish.blockSignals(False)
                self.status.showMessage(err)
                return
        else:
            self.cfg.training_opponent = opponent

        mode = "Stockfish training" if enabled else "self-play"
        self.subtitle.setText(f"AlphaZero · {mode}")
        self._reset_training_game_states()
        detail = (
            "one serial game per iteration (no parallel self-play)"
            if enabled
            else "parallel self-play"
        )
        self.status.showMessage(f"{mode} enabled — {detail}")

    def _play_vs_ckpt(self) -> None:
        from az.gui.play_mode import PlayVsNetDialog

        dlg = PlayVsNetDialog(None, self.assets, self)
        dlg.exec()

    def closeEvent(self, event):
        self._event_timer.stop()
        if self.trainer_thread and self.trainer_thread.orchestrator:
            self.trainer_thread.orchestrator.stop()
        if self.trainer_thread:
            self.trainer_thread.requestInterruption()
            self.trainer_thread.wait(5000)
        super().closeEvent(event)
