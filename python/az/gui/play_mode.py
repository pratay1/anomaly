from __future__ import annotations

import threading
from pathlib import Path

import chess
from PyQt6.QtCore import QTimer
from PyQt6.QtGui import QColor, QPen
from PyQt6.QtWidgets import (
    QDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
)

import az._az_core as core
from az.brain import align_cfg_with_brain, load_brain, resolve_brain_path
from az.config import Config
from az.gui.board_view import BoardView
from az.gui.piece_assets import PieceAssetManager
from az.network.resnet import AlphaZeroResNet
from az.search import create_search
from az.training.inference_server import InferenceServer
from az.training.selfplay_worker import move_to_uci
from az.training.stockfish import StockfishEngine


class PlayVsNetDialog(QDialog):
    def __init__(self, run_dir: Path | None, assets: PieceAssetManager, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Play vs AlphaZero")
        self.resize(560, 680)
        self.setMinimumSize(440, 540)
        self.cfg = Config()
        self.cfg.mcts_think_time_ms_min = 1000
        self.cfg.mcts_think_time_ms_max = 3000
        self.run_dir = Path(run_dir) if run_dir else resolve_brain_path().parent

        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 14, 14, 14)
        layout.setSpacing(10)

        header = QFrame()
        header.setObjectName("dialog_card")
        header_layout = QVBoxLayout(header)
        header_layout.setContentsMargins(14, 12, 14, 12)
        header_layout.setSpacing(4)
        title = QLabel("Play vs AlphaZero")
        title.setObjectName("dialog_title")
        brain_path = resolve_brain_path()
        self.info_label = QLabel(f"Brain: {brain_path.name}")
        self.info_label.setObjectName("dialog_subtitle")
        header_layout.addWidget(title)
        header_layout.addWidget(self.info_label)
        layout.addWidget(header)

        self.board_view = BoardView(assets)
        board_frame = QFrame()
        board_frame.setObjectName("board_frame")
        board_layout = QVBoxLayout(board_frame)
        board_layout.setContentsMargins(8, 8, 8, 8)
        board_layout.addWidget(self.board_view)
        layout.addWidget(board_frame, stretch=1)

        status_row = QHBoxLayout()
        self.status_label = QLabel("Your turn (White)")
        self.status_label.setObjectName("dialog_subtitle")
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
        self.inference = None
        self.stockfish: StockfishEngine | None = None
        if self.cfg.search_engine == "mcts":
            self.inference = InferenceServer(self.queue, self.model, self.cfg, self.stop)
            self.inference.start()
        else:
            self.stockfish = StockfishEngine(self.cfg)
        self._search = create_search(self.cfg, self.queue, self.stockfish)

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
        scene = self.board_view.scene()
        if scene is None:
            return
        for r in self._selection_rects:
            scene.removeItem(r)
        self._selection_rects.clear()

    def _highlight_square(self, sq: int) -> None:
        self._clear_selection()
        scene = self.board_view.scene()
        if scene is None:
            return
        f = chess.square_file(sq)
        r = chess.square_rank(sq)
        x = f * BoardView.SQUARE
        y = (7 - r) * BoardView.SQUARE
        rect = scene.addRect(
            x, y, BoardView.SQUARE, BoardView.SQUARE,
            pen=QPen(QColor("#7a7a7a"), 3),
        )
        if rect is not None:
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
            elif self._is_promotion_attempt(from_sq, sq):
                promo = self._ask_promotion_piece()
                if promo is not None:
                    pm = chess.Move(from_sq, sq, promotion=promo)
                    if pm in self.ch.legal_moves:
                        self._apply_human_move(pm)
                        return
                self.status_label.setText("Promotion cancelled — select your piece")
            else:
                p = self.ch.piece_at(sq)
                if p and p.color == self.ch.turn:
                    self._selected_sq = sq
                    self._highlight_square(sq)
                    name = chess.square_name(sq)
                    self.status_label.setText(f"Selected {name} — pick destination")
                else:
                    self._selected_sq = None
                    self.status_label.setText("Illegal move — select your piece")

    def _is_promotion_attempt(self, from_sq: int, to_sq: int) -> bool:
        p = self.ch.piece_at(from_sq)
        if not p or p.piece_type != chess.PAWN:
            return False
        promo_rank = 7 if p.color == chess.WHITE else 0
        return chess.square_rank(to_sq) == promo_rank

    def _ask_promotion_piece(self) -> int | None:
        box = QMessageBox(self)
        box.setWindowTitle("Promote pawn")
        box.setText("Choose promotion piece:")
        queen = box.addButton("Queen", QMessageBox.ButtonRole.AcceptRole)
        rook = box.addButton("Rook", QMessageBox.ButtonRole.AcceptRole)
        bishop = box.addButton("Bishop", QMessageBox.ButtonRole.AcceptRole)
        knight = box.addButton("Knight", QMessageBox.ButtonRole.AcceptRole)
        box.addButton(QMessageBox.StandardButton.Cancel)
        box.exec()
        clicked = box.clickedButton()
        if clicked == queen:
            return chess.QUEEN
        if clicked == rook:
            return chess.ROOK
        if clicked == bishop:
            return chess.BISHOP
        if clicked == knight:
            return chess.KNIGHT
        return None

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
        self.status_label.setText("Anomaly thinking…")
        self.board_view.set_thinking(True)
        QTimer.singleShot(50, self._engine_move)

    def _engine_move(self) -> None:
        if self.ch.is_game_over():
            self._human_turn = False
            self._show_result()
            return
        self._engine_thinking = True
        try:
            think_ms = self.cfg.random_think_time_ms()
            pi = self._search.run(self.cpp_board, 0.1, think_ms)
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
        if self.inference is not None:
            self.inference.join(timeout=2)
        if self.stockfish is not None:
            self.stockfish.close()
        super().closeEvent(event)
