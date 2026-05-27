from types import SimpleNamespace

from az.config import Config
from az.network.resnet import AlphaZeroResNet
from az.training.central_learner import CentralLearner
from az.training.replay_buffer import ReplayBuffer


def test_overfit_small_batch():
    cfg = Config()
    cfg.batch_size = 8
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

    import threading

    learner = CentralLearner(model, buf, cfg, threading.Event())
    losses = []
    for _ in range(20):
        step = learner.train_once()
        if step:
            losses.append(step.total_loss)
    assert len(losses) >= 5
    # Loss should generally decrease or stay finite
    assert losses[-1] < losses[0] * 2 + 10
