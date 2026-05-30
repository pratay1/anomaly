from __future__ import annotations

from dataclasses import dataclass, field

import chess
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QDialog,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from az.gui.mini_board_view import MiniBoardView
from az.gui.piece_assets import PieceAssetManager
from az.ipc.events import GameFinished, MovePlayed


@dataclass
class GameState:
    fen: str = chess.STARTING_FEN
    moves: list[str] = field(default_factory=list)
    result: str | None = None
    active: bool = True


class MultiGameGridDialog(QDialog):
    """Grid of mini boards showing all parallel self-play games."""

    focus_requested = pyqtSignal(int)
    solo_requested = pyqtSignal(int)

    def __init__(
        self,
        num_games: int,
        assets: PieceAssetManager,
        parent=None,
        anim_ms: int = 140,
    ):
        super().__init__(parent)
        self.num_games = num_games
        self.assets = assets
        self.setWindowTitle("Parallel Games")
        self.resize(900, 520)
        self._anim_ms = anim_ms
        self._states: dict[int, GameState] = {i: GameState() for i in range(num_games)}
        self._boards: dict[int, MiniBoardView] = {}
        self._labels: dict[int, QLabel] = {}

        layout = QVBoxLayout(self)
        header = QHBoxLayout()
        self.iteration_label = QLabel("Iteration: —")
        header.addWidget(self.iteration_label)
        header.addStretch()
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.close)
        header.addWidget(close_btn)
        layout.addLayout(header)

        grid_widget = QWidget()
        grid = QGridLayout(grid_widget)
        cols = 3 if num_games > 3 else num_games
        for gid in range(num_games):
            cell = QVBoxLayout()
            lbl = QLabel(f"Game {gid + 1}")
            lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            board = MiniBoardView(gid, assets, anim_ms=self._anim_ms)
            board.clicked.connect(self._on_board_click)
            board.double_clicked.connect(self._on_board_double_click)
            cell.addWidget(lbl)
            cell.addWidget(board, alignment=Qt.AlignmentFlag.AlignCenter)
            self._boards[gid] = board
            self._labels[gid] = lbl
            row, col = divmod(gid, cols)
            grid.addLayout(cell, row, col)
        layout.addWidget(grid_widget)

    def _on_board_click(self, game_id: int) -> None:
        self.focus_requested.emit(game_id)

    def _on_board_double_click(self, game_id: int) -> None:
        self.solo_requested.emit(game_id)

    def on_move(self, mv: MovePlayed) -> None:
        gid = mv.game_id
        if gid not in self._states:
            return
        st = self._states[gid]
        try:
            board = chess.Board(mv.fen)
            board.push(chess.Move.from_uci(mv.uci))
            st.fen = board.fen()
        except Exception:
            try:
                board = chess.Board(st.fen)
                board.push(chess.Move.from_uci(mv.uci))
                st.fen = board.fen()
            except Exception:
                pass
        st.moves.append(mv.uci)
        if gid in self._boards:
            if mv.visits and mv.fen:
                self._boards[gid].set_heatmap_from_visits(mv.fen, mv.visits)
            self._boards[gid].set_fen(st.fen)
        self._labels[gid].setText(f"Game {gid + 1} · ply {len(st.moves)}")

    def on_mcts_visits(self, game_id: int, fen: str, visits: list) -> None:
        if game_id in self._boards and visits:
            self._boards[game_id].set_heatmap_from_visits(fen, visits)

    def on_game_finished(self, game: GameFinished) -> None:
        gid = game.game_id
        if gid not in self._states:
            return
        st = self._states[gid]
        st.result = game.result
        st.active = False
        if game.moves_uci:
            st.moves = list(game.moves_uci)
        self._labels[gid].setText(f"Game {gid + 1} · {game.result}")

    def reset_iteration(self, iteration: int) -> None:
        self.iteration_label.setText(f"Iteration: {iteration}")
        for gid in range(self.num_games):
            self._states[gid] = GameState()
            if gid in self._boards:
                self._boards[gid].set_fen(chess.STARTING_FEN)
            self._labels[gid].setText(f"Game {gid + 1}")

    def get_state(self, game_id: int) -> GameState | None:
        return self._states.get(game_id)

    def update_board_fen(self, game_id: int, fen: str) -> None:
        if game_id in self._boards:
            self._boards[game_id].set_fen(fen)
