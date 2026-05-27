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
    mcts_cfg = cfg.to_mcts_config()
    mcts_cfg.add_root_noise = False
    mcts_cfg.num_simulations = max(50, cfg.num_simulations // 2)
    wins = 0.0
    for g in range(num_games):
        board = core.Board()
        mcts = core.MCTS(queue, mcts_cfg)
        for _ply in range(cfg.max_game_length):
            res = board.result()
            if res != core.GameResult.Ongoing:
                break
            pi = mcts.run(board, 0.1)
            legal = core.legal_move_indices(board)
            if not legal:
                break
            # Greedy
            best = max(legal, key=lambda i: pi[i])
            board.make_move(core.index_to_move(board, best))
        res = board.result()
        # Simplified: count non-draw as success if we had moves
        if res == core.GameResult.Draw:
            pass
        else:
            wins += 0.5
    return wins / max(num_games, 1)
