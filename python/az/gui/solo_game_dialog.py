from __future__ import annotations

import chess
from PyQt6.QtWidgets import QDialog, QFrame, QLabel, QVBoxLayout

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
        layout.setContentsMargins(14, 14, 14, 14)
        layout.setSpacing(10)

        header = QFrame()
        header.setObjectName("dialog_card")
        header_layout = QVBoxLayout(header)
        header_layout.setContentsMargins(14, 12, 14, 12)
        header_layout.setSpacing(4)
        title = QLabel(f"Game {game_id + 1}")
        title.setObjectName("dialog_title")
        self.status = QLabel("Live view")
        self.status.setObjectName("dialog_subtitle")
        header_layout.addWidget(title)
        header_layout.addWidget(self.status)
        layout.addWidget(header)

        self.board_view = BoardView(assets)
        board_frame = QFrame()
        board_frame.setObjectName("board_frame")
        board_layout = QVBoxLayout(board_frame)
        board_layout.setContentsMargins(8, 8, 8, 8)
        board_layout.addWidget(self.board_view)
        layout.addWidget(board_frame, stretch=1)
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
        self.status.setText(f"Ply {self._move_count}")

    def on_game_finished(self, game: GameFinished) -> None:
        if game.game_id != self.game_id:
            return
        self.status.setText(f"{game.result} · {game.plies} plies")
