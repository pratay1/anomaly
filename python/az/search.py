"""Unified search interface — Stockfish UCI (default) or legacy MCTS+net."""

from __future__ import annotations

import az._az_core as core
from az.config import Config
from az.training.stockfish import StockfishEngine


class StockfishSearch:
    """Drop-in replacement for C++ MCTS using Stockfish's alpha-beta search."""

    def __init__(self, engine: StockfishEngine, cfg: Config):
        self._engine = engine
        self._cfg = cfg
        self._last_visits: list[dict] = []

    def run(
        self, board: core.Board, temperature: float = 1.0, think_time_ms: int = 0
    ) -> list[float]:
        ms = think_time_ms if think_time_ms > 0 else self._cfg.stockfish_movetime_ms
        pi, visits = self._engine.search(
            board,
            ms,
            multipv=min(10, len(core.legal_move_indices(board)) or 1),
            temperature=temperature,
        )
        self._last_visits = visits
        return pi

    def advance_root(self, move_index: int) -> None:
        pass

    def reset_tree(self) -> None:
        self._last_visits = []

    def root_visits(self, board: core.Board) -> list:
        return self._last_visits


class MCTSSearch:
    """Legacy AlphaZero MCTS backed by the neural network."""

    def __init__(self, queue: core.InferenceQueue, cfg: Config):
        self._mcts = core.MCTS(queue, cfg.to_mcts_config())

    def run(
        self, board: core.Board, temperature: float = 1.0, think_time_ms: int = 0
    ) -> list[float]:
        return self._mcts.run(board, temperature, think_time_ms)

    def advance_root(self, move_index: int) -> None:
        self._mcts.advance_root(move_index)

    def reset_tree(self) -> None:
        self._mcts.reset_tree()

    def root_visits(self, board: core.Board) -> list:
        return self._mcts.root_visits(board)


def create_search(
    cfg: Config,
    queue: core.InferenceQueue | None = None,
    stockfish: StockfishEngine | None = None,
) -> StockfishSearch | MCTSSearch:
    if cfg.search_engine == "stockfish":
        if stockfish is None:
            stockfish = StockfishEngine(cfg)
        return StockfishSearch(stockfish, cfg)
    if queue is None:
        raise ValueError("MCTS search requires an InferenceQueue")
    return MCTSSearch(queue, cfg)
