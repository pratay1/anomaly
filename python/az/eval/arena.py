from __future__ import annotations

import random

import az._az_core as core
from az.config import Config
from az.network.resnet import AlphaZeroResNet


def play_random_vs_random(num_games: int = 10) -> tuple[int, int, int]:
    wins = losses = draws = 0
    for _ in range(num_games):
        board = core.Board()
        for _ply in range(400):
            res = board.result()
            if res != core.GameResult.Ongoing:
                break
            moves = board.generate_legal_moves()
            if not moves:
                break
            m = random.choice(moves)
            board.make_move(m)
        res = board.result()
        if res == core.GameResult.WhiteWin:
            wins += 1
        elif res == core.GameResult.BlackWin:
            losses += 1
        else:
            draws += 1
    return wins, draws, losses


def evaluate_vs_random(
    cfg: Config,
    model: AlphaZeroResNet,
    queue: core.InferenceQueue,
    num_games: int = 10,
) -> float:
    """Fraction of games won by MCTS+net (playing white and black alternately)."""
    import threading

    from az.training.inference_server import InferenceServer

    mcts_cfg = cfg.to_mcts_config()
    mcts_cfg.add_root_noise = False
    think_ms = (cfg.mcts_think_time_ms_min + cfg.mcts_think_time_ms_max) // 2
    stop = threading.Event()
    inf = InferenceServer(queue, model, cfg, stop)
    inf.start()
    wins = 0.0
    for g in range(num_games):
        board = core.Board()
        mcts = core.MCTS(queue, mcts_cfg)
        for _ply in range(cfg.max_game_length):
            res = board.result()
            if res != core.GameResult.Ongoing:
                break
            pi = mcts.run(board, 0.1, think_ms)
            legal = core.legal_move_indices(board)
            if not legal:
                break
            best = max(legal, key=lambda i: pi[i])
            board.make_move(core.index_to_move(board, best))
        res = board.result()
        is_white = g % 2 == 0
        if res == core.GameResult.WhiteWin:
            wins += 1.0 if is_white else 0.0
        elif res == core.GameResult.BlackWin:
            wins += 1.0 if not is_white else 0.0
    stop.set()
    return wins / max(num_games, 1)
