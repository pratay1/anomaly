from __future__ import annotations

import chess
from PyQt6.QtWidgets import QDialog, QLabel, QVBoxLayout

from az.gui.board_view import BoardView
from az.gui.piece_assets import PieceAssetManager
from az.ipc.events import GameFinished, MovePlayed


class SoloGameDialog(QDialog):
    """Full-size live view of a single parallel game."""

    def __init__(
        self,
        game_id: int,
        assets: PieceAssetManager,
        fen: str = chess.STARTING_FEN,
        parent=None,
    ):
        super().__init__(parent)
        self.game_id = game_id
        self._fen = fen
        self._move_count = 0
        self.setWindowTitle(f"Game {game_id + 1} — Solo View")
        self.resize(560, 620)
        layout = QVBoxLayout(self)
        self.status = QLabel(f"Game {game_id + 1} · live")
        layout.addWidget(self.status)
        self.board_view = BoardView(assets)
        layout.addWidget(self.board_view)
        self.board_view.set_fen(fen)

    def on_move(self, mv: MovePlayed) -> None:
        if mv.game_id != self.game_id:
            return
        try:
            board = chess.Board(self._fen)
            board.push(chess.Move.from_uci(mv.uci))
            self._fen = board.fen()
            self.board_view.set_fen(self._fen)
        except Exception:
            self.board_view.set_fen(mv.fen)
            self._fen = mv.fen
        self._move_count += 1
        self.status.setText(f"Game {self.game_id + 1} · ply {self._move_count}")

    def on_game_finished(self, game: GameFinished) -> None:
        if game.game_id != self.game_id:
            return
        self.status.setText(f"Game {self.game_id + 1} · {game.result} ({game.plies} plies)")
