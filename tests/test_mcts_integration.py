"""MCTS + inference server integration test."""

import threading

import torch

from az.config import Config
from az.network.resnet import AlphaZeroResNet
from az.training.inference_server import InferenceServer


def test_mcts_forward_terminates():
    import az._az_core as core

    cfg = Config()
    cfg.num_simulations = 5
    cfg.max_batch = 8
    cfg.max_wait_us = 3000
    queue = core.InferenceQueue()
    model = AlphaZeroResNet(cfg)
    stop = threading.Event()
    srv = InferenceServer(queue, model, cfg, stop)
    srv.start()

    board = core.Board()
    mcts = core.MCTS(queue, cfg.to_mcts_config())
    pi = mcts.run(board, 1.0)

    stop.set()
    srv.join(timeout=3)

    assert len(pi) == cfg.policy_size
    assert abs(sum(pi) - 1.0) < 0.01


def test_mcts_advance_root():
    import az._az_core as core

    cfg = Config()
    cfg.num_simulations = 5
    cfg.max_batch = 8
    cfg.max_wait_us = 3000
    queue = core.InferenceQueue()
    model = AlphaZeroResNet(cfg)
    stop = threading.Event()
    srv = InferenceServer(queue, model, cfg, stop)
    srv.start()

    board = core.Board()
    mcts = core.MCTS(queue, cfg.to_mcts_config())
    _pi = mcts.run(board, 0.1)
    legal = core.legal_move_indices(board)
    assert legal, "Expected at least one legal move"

    idx = max(legal, key=lambda i: _pi[i])
    mv = core.index_to_move(board, idx)
    board.make_move(mv)
    mcts.advance_root(idx)

    # Run again after advancing root
    pi2 = mcts.run(board, 1.0)
    assert abs(sum(pi2) - 1.0) < 0.01

    stop.set()
    srv.join(timeout=3)
