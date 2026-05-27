from __future__ import annotations

import threading
from types import SimpleNamespace

from az.config import Config
from az.network.resnet import AlphaZeroResNet
from az.training.central_learner import CentralLearner
from az.training.replay_buffer import ReplayBuffer


def test_central_learner_gradient_avg():
    cfg = Config()
    cfg.batch_size = 8
    cfg.gradient_avg_workers = 2
    cfg.train_steps_per_iteration = 2
    model = AlphaZeroResNet(cfg)
    buf = ReplayBuffer(100, cfg.encoding_channels * 64, cfg.policy_size)
    import az._az_core as core

    state = core.encode(core.Board())
    pi = [0.0] * cfg.policy_size
    legal = core.legal_move_indices(core.Board())
    if legal:
        pi[legal[0]] = 1.0
    ex = SimpleNamespace(state=state, policy=pi, value=0.5)
    buf.add_batch([ex] * 64)

    stop = threading.Event()
    learner = CentralLearner(model, buf, cfg, stop)
    steps = learner.train_iteration(3)
    assert len(steps) >= 1
    assert all(s.total_loss == s.total_loss for s in steps)  # finite
    assert learner.global_step >= 1
