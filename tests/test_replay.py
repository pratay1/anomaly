import threading
from types import SimpleNamespace

from az.config import Config
from az.training.replay_buffer import ReplayBuffer


def test_replay_buffer():
    cfg = Config()
    buf = ReplayBuffer(1000, cfg.encoding_channels * 64, cfg.policy_size)
    ex = SimpleNamespace(
        state=[0.0] * (cfg.encoding_channels * 64),
        policy=[0.0] * cfg.policy_size,
        value=1.0,
    )
    buf.add_batch([ex] * 10)
    assert len(buf) == 10
    s, p, v = buf.sample(4)
    assert s.shape == (4, cfg.encoding_channels * 64)


def test_replay_buffer_thread_safe():
    cfg = Config()
    buf = ReplayBuffer(1000, cfg.encoding_channels * 64, cfg.policy_size)
    ex = SimpleNamespace(
        state=[0.0] * (cfg.encoding_channels * 64),
        policy=[0.0] * cfg.policy_size,
        value=1.0,
    )

    def writer():
        for _ in range(200):
            buf.add_batch([ex])

    threads = [threading.Thread(target=writer) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(buf) > 0
