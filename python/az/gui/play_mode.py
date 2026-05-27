from __future__ import annotations

from pathlib import Path

import chess
from PyQt6.QtWidgets import QDialog, QLabel, QVBoxLayout

import az._az_core as core
from az.brain import align_cfg_with_brain, load_brain, resolve_brain_path
from az.config import Config
from az.gui.board_view import BoardView
from az.gui.piece_assets import PieceAssetManager
from az.network.resnet import AlphaZeroResNet
from az.training.inference_server import InferenceServer
from az.training.selfplay_worker import move_to_uci


class PlayVsNetDialog(QDialog):
    """Play as human (white) vs latest checkpoint using C++ MCTS + loaded net."""

    def __init__(self, run_dir: Path | None, assets: PieceAssetManager, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Play vs AlphaZero")
        self.resize(560, 620)
        self.cfg = Config()
        self.cfg.num_simulations = 100
        self.run_dir = Path(run_dir) if run_dir else resolve_brain_path().parent
        layout = QVBoxLayout(self)
        brain_path = resolve_brain_path()
        layout.addWidget(QLabel(f"Brain: {brain_path}"))
        self.board_view = BoardView(assets)
        layout.addWidget(self.board_view)

        align_cfg_with_brain(self.cfg)
        self.model = AlphaZeroResNet(self.cfg)
        info = load_brain(self.model, "cpu")
        layout.addWidget(QLabel(f"Loaded step {info.step} from {info.path.name}"))

        import threading

        self.queue = core.InferenceQueue()
        self.stop = threading.Event()
        self.inference = InferenceServer(self.queue, self.model, self.cfg, self.stop)
        self.inference.start()

        self.ch = chess.Board()
        self.cpp_board = core.Board()
        self.board_view.set_fen(self.ch.fen())
        self._engine_move()

    def _engine_move(self):
        if self.ch.is_game_over():
            return
        mcts = core.MCTS(self.queue, self.cfg.to_mcts_config())
        pi = mcts.run(self.cpp_board, 0.1)
        legal = core.legal_move_indices(self.cpp_board)
        if not legal:
            return
        idx = max(legal, key=lambda i: pi[i])
        mv = core.index_to_move(self.cpp_board, idx)
        uci = chess.Move.from_uci(move_to_uci(mv))
        self.cpp_board.make_move(mv)
        self.ch.push(uci)
        self.board_view.set_fen(self.ch.fen())

    def closeEvent(self, event):
        self.stop.set()
        super().closeEvent(event)
