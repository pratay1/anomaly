"""Tests for Stockfish search integration."""

from __future__ import annotations

from unittest.mock import MagicMock

import az._az_core as core
from az.config import Config
from az.search import StockfishSearch, create_search


def test_create_search_returns_stockfish_by_default():
    cfg = Config()
    assert cfg.search_engine == "stockfish"
    engine = MagicMock()
    search = create_search(cfg, stockfish=engine)
    assert isinstance(search, StockfishSearch)


def test_create_search_mcts_requires_queue():
    cfg = Config()
    cfg.search_engine = "mcts"
    queue = core.InferenceQueue()
    search = create_search(cfg, queue)
    assert search.__class__.__name__ == "MCTSSearch"


def test_stockfish_search_run_delegates_to_engine():
    cfg = Config()
    board = core.Board()
    engine = MagicMock()
    pi = [0.0] * cfg.policy_size
    legal = core.legal_move_indices(board)
    pi[legal[0]] = 1.0
    engine.search.return_value = (pi, [{"move_index": legal[0], "N": 1, "Q": 0.0, "P": 1.0}])
    search = StockfishSearch(engine, cfg)
    out = search.run(board, 1.0, 500)
    assert out == pi
    engine.search.assert_called_once()
    visits = search.root_visits(board)
    assert visits[0]["move_index"] == legal[0]


def test_scores_to_policy_low_temperature_is_argmax():
    from az.training.stockfish import _scores_to_policy

    pi = _scores_to_policy([10.0, 50.0, 30.0], [1, 2, 3], 10, 0.1)
    assert pi[2] == 1.0
    assert sum(pi) == 1.0
