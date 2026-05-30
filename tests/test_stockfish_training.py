from __future__ import annotations

import queue
import threading
from pathlib import Path
from unittest.mock import MagicMock, patch

import az._az_core as core
from az.config import Config
from az.training.replay_buffer import ReplayBuffer
from az.training.selfplay_worker import ParallelSelfPlayPool, play_one_game
from az.training.stockfish_paths import resolve_stockfish_path, stockfish_path_error


def test_play_one_game_stockfish_only_records_mcts_plies():
    cfg = Config()
    cfg.search_engine = "mcts"
    cfg.training_opponent = "stockfish"
    cfg.max_game_length = 4
    cfg.temperature_moves = 0
    stop = threading.Event()
    queue_inf = core.InferenceQueue()
    events: queue.Queue = queue.Queue()

    stockfish = MagicMock()
    stockfish_moves = ["e7e5", "d7d5"]

    def fake_choose(board: core.Board) -> core.Move:
        uci = stockfish_moves.pop(0)
        for idx in core.legal_move_indices(board):
            mv = core.index_to_move(board, idx)
            from az.training.selfplay_worker import move_to_uci

            if move_to_uci(mv) == uci:
                return mv
        raise AssertionError(f"unexpected stockfish move {uci}")

    stockfish.choose_move.side_effect = fake_choose

    with patch("az.training.selfplay_worker.create_search") as mock_search_factory:
        mock_search = MagicMock()
        mock_search.run.return_value = [0.0] * cfg.policy_size
        mock_search.root_visits.return_value = []
        mock_search_factory.return_value = mock_search

        examples = play_one_game(
            queue_inf,
            cfg,
            stop,
            game_id=0,
            game_seq=0,
            stockfish=stockfish,
            event_sink=events,
        )

    assert len(examples) == 2
    assert stockfish.choose_move.call_count == 2
    move_events = [payload[0] for kind, *payload in _drain(events) if kind == "move_played"]
    assert len(move_events) == 4


def test_game_seq_alternates_mcts_color():
    cfg = Config()
    cfg.search_engine = "mcts"
    cfg.training_opponent = "stockfish"
    stop = threading.Event()
    queue_inf = core.InferenceQueue()
    stockfish = MagicMock()

    seen_colors: list[int] = []

    def fake_run(board, temp, think_ms=0):
        seen_colors.append(board.side_to_move())
        pi = [0.0] * cfg.policy_size
        legal = core.legal_move_indices(board)
        if legal:
            pi[legal[0]] = 1.0
        return pi

    def fake_stockfish_move(board: core.Board) -> core.Move:
        legal = core.legal_move_indices(board)
        return core.index_to_move(board, legal[0])

    stockfish.choose_move.side_effect = fake_stockfish_move

    with patch("az.training.selfplay_worker.create_search") as mock_search_factory:
        mock_search = MagicMock()
        mock_search.run.side_effect = fake_run
        mock_search.root_visits.return_value = []
        mock_search_factory.return_value = mock_search

        cfg.max_game_length = 1
        play_one_game(
            queue_inf, cfg, stop, game_seq=0, stockfish=stockfish
        )
        cfg.max_game_length = 2
        play_one_game(
            queue_inf, cfg, stop, game_seq=1, stockfish=stockfish
        )

    assert seen_colors == [core.Color.White, core.Color.Black]


def test_stockfish_mode_runs_parallel_games():
    cfg = Config()
    cfg.num_workers = 3
    cfg.training_opponent = "stockfish"
    stop = threading.Event()
    queue_inf = core.InferenceQueue()
    buf = ReplayBuffer(1000, cfg.encoding_channels * 64, cfg.policy_size)
    pool = ParallelSelfPlayPool(queue_inf, buf, cfg, stop)

    call_count = 0

    def fake_play(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        return []

    with patch("az.training.selfplay_worker.play_one_game", side_effect=fake_play):
        pool.run_iteration(3)

    assert call_count == 3
    assert cfg.games_per_selfplay_iteration() == 3


def test_stockfish_mode_publishes_events_to_outbound_queue():
    cfg = Config()
    cfg.num_workers = 1
    cfg.training_opponent = "stockfish"
    stop = threading.Event()
    queue_inf = core.InferenceQueue()
    buf = ReplayBuffer(1000, cfg.encoding_channels * 64, cfg.policy_size)
    outbound: queue.Queue = queue.Queue()
    pool = ParallelSelfPlayPool(queue_inf, buf, cfg, stop, outbound)

    def fake_play(*args, **kwargs):
        # Workers write to their local event_queue; the pool drains it to outbound.
        event_sink = kwargs["event_sink"]
        event_sink.put(("move_played", "live"))
        return []

    with patch("az.training.selfplay_worker.play_one_game", side_effect=fake_play):
        pool.run_iteration(1)

    assert outbound.get_nowait() == ("move_played", "live")


def test_set_training_opponent_closes_stockfish_engine():
    cfg = Config()
    cfg.search_engine = "mcts"
    cfg.training_opponent = "stockfish"
    stop = threading.Event()
    queue_inf = core.InferenceQueue()
    buf = ReplayBuffer(1000, cfg.encoding_channels * 64, cfg.policy_size)
    pool = ParallelSelfPlayPool(queue_inf, buf, cfg, stop)

    engine = MagicMock()
    with patch.object(pool, "_stockfish", engine):
        pool.set_training_opponent("self")
    engine.close.assert_called_once()
    assert pool._stockfish is None
    assert cfg.training_opponent == "self"


def test_resolve_stockfish_path_env(monkeypatch, tmp_path):
    fake = tmp_path / "my-stockfish"
    fake.write_text("")
    monkeypatch.setenv("ANOMALY_STOCKFISH_PATH", str(fake))
    assert resolve_stockfish_path() == fake.resolve()


def test_stockfish_path_error_when_missing():
    assert stockfish_path_error(Path("/nonexistent/stockfish")) is not None


def _drain(q: queue.Queue) -> list:
    items = []
    while True:
        try:
            items.append(q.get_nowait())
        except queue.Empty:
            break
    return items
