from __future__ import annotations

import threading
from pathlib import Path

import chess
from PyQt6.QtCore import QTimer
from PyQt6.QtGui import QColor, QPen
from PyQt6.QtWidgets import QDialog, QHBoxLayout, QLabel, QPushButton, QVBoxLayout

import az._az_core as core
from az.brain import align_cfg_with_brain, load_brain, resolve_brain_path
from az.config import Config
from az.gui.board_view import BoardView
from az.gui.piece_assets import PieceAssetManager

from az.network.resnet import AlphaZeroResNet
from az.training.inference_server import InferenceServer
from az.training.selfplay_worker import move_to_uci


class PlayVsNetDialog(QDialog):
    def __init__(self, run_dir: Path | None, assets: PieceAssetManager, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Play vs AlphaZero")
        self.resize(560, 680)
        self.setMinimumSize(440, 540)
        self.cfg = Config()
        self.cfg.num_simulations = 100
        self.run_dir = Path(run_dir) if run_dir else resolve_brain_path().parent

        layout = QVBoxLayout(self)
        brain_path = resolve_brain_path()
        self.info_label = QLabel(f"Brain: {brain_path.name}")
        layout.addWidget(self.info_label)

        self.board_view = BoardView(assets)
        layout.addWidget(self.board_view, stretch=1)

        status_row = QHBoxLayout()
        self.status_label = QLabel("Your turn (White)")
        status_row.addWidget(self.status_label, stretch=1)
        self.btn_reset = QPushButton("New Game")
        self.btn_reset.clicked.connect(self._reset_game)
        status_row.addWidget(self.btn_reset)
        layout.addLayout(status_row)

        align_cfg_with_brain(self.cfg)
        self.model = AlphaZeroResNet(self.cfg)
        info = load_brain(self.model, "cpu")
        self.info_label.setText(f"Brain: {brain_path.name} (step {info.step})")

        self.queue = core.InferenceQueue()
        self.stop = threading.Event()
        self.inference = InferenceServer(self.queue, self.model, self.cfg, self.stop)
        self.inference.start()

        self.ch = chess.Board()
        self.cpp_board = core.Board()
        self._selected_sq: int | None = None
        self._selection_rects: list = []
        self._engine_thinking = False
        self._human_turn = True

        self.board_view.set_fen(self.ch.fen(), animated=False)
        self.board_view.square_clicked.connect(self._on_square_clicked)

    def _clear_selection(self) -> None:
        self._selected_sq = None
        for r in self._selection_rects:
            self.board_view.scene().removeItem(r)
        self._selection_rects.clear()

    def _highlight_square(self, sq: int) -> None:
        self._clear_selection()
        f = chess.square_file(sq)
        r = chess.square_rank(sq)
        x = f * BoardView.SQUARE
        y = (7 - r) * BoardView.SQUARE
        rect = self.board_view.scene().addRect(
            x, y, BoardView.SQUARE, BoardView.SQUARE,
            pen=QPen(QColor("#7a7a7a"), 3),
        )
        rect.setZValue(20)
        self._selection_rects.append(rect)

    def _on_square_clicked(self, sq: int) -> None:
        if self._engine_thinking or not self._human_turn:
            return
        if self.ch.is_game_over():
            return

        if self._selected_sq is None:
            p = self.ch.piece_at(sq)
            if p and p.color == self.ch.turn:
                self._selected_sq = sq
                self._highlight_square(sq)
                self.status_label.setText(f"Selected {chess.square_name(sq)} — pick destination")
        else:
            from_sq = self._selected_sq
            self._clear_selection()
            move = chess.Move(from_sq, sq)
            if move in self.ch.legal_moves:
                self._apply_human_move(move)
            else:
                for promo in [chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT]:
                    pm = chess.Move(from_sq, sq, promotion=promo)
                    if pm in self.ch.legal_moves:
                        self._apply_human_move(pm)
                        return
                p = self.ch.piece_at(sq)
                if p and p.color == self.ch.turn:
                    self._selected_sq = sq
                    self._highlight_square(sq)
                    self.status_label.setText(f"Selected {chess.square_name(sq)} — pick destination")
                else:
                    self._selected_sq = None
                    self.status_label.setText("Illegal move — select your piece")

    def _apply_human_move(self, move: chess.Move) -> None:
        uci = move.uci()
        cpp_move = self._uci_to_cpp_move(uci)
        if cpp_move is None:
            self.status_label.setText("Invalid move")
            self._selected_sq = None
            return

        self.cpp_board.make_move(cpp_move)
        self.ch.push(move)
        self.board_view.set_fen(self.ch.fen(), animated=True)
        self.board_view.set_last_move(move.from_square, move.to_square)
        self._human_turn = False
        self._selected_sq = None
        self.status_label.setText("AlphaZero thinking…")
        self.board_view.set_thinking(True)
        QTimer.singleShot(50, self._engine_move)

    def _engine_move(self) -> None:
        if self.ch.is_game_over():
            self._human_turn = False
            self._show_result()
            return
        self._engine_thinking = True
        try:
            mcts = core.MCTS(self.queue, self.cfg.to_mcts_config())
            pi = mcts.run(self.cpp_board, 0.1)
            legal = core.legal_move_indices(self.cpp_board)
            if not legal:
                self._engine_thinking = False
                self._human_turn = False
                self._show_result()
                return
            idx = max(legal, key=lambda i: pi[i])
            mv = core.index_to_move(self.cpp_board, idx)
            uci = move_to_uci(mv)
            move = chess.Move.from_uci(uci)
            self.cpp_board.make_move(mv)
            self.ch.push(move)
            self.board_view.set_fen(self.ch.fen(), animated=True)
            self.board_view.set_last_move(move.from_square, move.to_square)
        finally:
            self._engine_thinking = False
            self.board_view.set_thinking(False)

        if self.ch.is_game_over():
            self._human_turn = False
            self._show_result()
        else:
            self._human_turn = True
            self.status_label.setText("Your turn (White)")

    def _uci_to_cpp_move(self, uci: str) -> core.Move | None:
        try:
            for m in self.cpp_board.generate_legal_moves():
                if move_to_uci(m) == uci:
                    return m
        except Exception:
            pass
        return None

    def _show_result(self) -> None:
        if self.ch.is_checkmate():
            winner = "Black" if self.ch.turn == chess.WHITE else "White"
            self.status_label.setText(f"Checkmate! {winner} wins.")
        elif self.ch.is_stalemate():
            self.status_label.setText("Stalemate — draw.")
        elif self.ch.is_insufficient_material():
            self.status_label.setText("Draw — insufficient material.")
        elif self.ch.is_fifty_moves():
            self.status_label.setText("Draw — fifty-move rule.")
        elif self.ch.is_repetition():
            self.status_label.setText("Draw — threefold repetition.")
        else:
            self.status_label.setText("Game over.")

    def _reset_game(self) -> None:
        self._clear_selection()
        self.ch = chess.Board()
        self.cpp_board = core.Board()
        self._selected_sq = None
        self._engine_thinking = False
        self._human_turn = True
        self.board_view.clear_last_move()
        self.board_view.set_fen(self.ch.fen(), animated=False)
        self.board_view.set_thinking(False)
        self.status_label.setText("Your turn (White)")

    def closeEvent(self, event):
        self.stop.set()
        self.inference.join(timeout=2)
        super().closeEvent(event)
