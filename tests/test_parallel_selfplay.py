from __future__ import annotations

import threading
import time
from types import SimpleNamespace
from unittest.mock import patch

import az._az_core as core
from az.config import Config
from az.training import selfplay_worker
from az.training.replay_buffer import ReplayBuffer
from az.training.selfplay_worker import ParallelSelfPlayPool


def test_parallel_iteration_waits_for_all():
    cfg = Config()
    cfg.num_workers = 3
    stop = threading.Event()
    queue = core.InferenceQueue()
    buf = ReplayBuffer(1000, cfg.encoding_channels * 64, cfg.policy_size)
    pool = ParallelSelfPlayPool(queue, buf, cfg, stop)

    played: list[int] = []

    def fake_play(*args, **kwargs):
        game_id = kwargs.get("game_id", args[3] if len(args) > 3 else 0)
        played.append(game_id)
        time.sleep(0.02 * (game_id + 1))
        return [
            SimpleNamespace(
                state=[0.0] * (cfg.encoding_channels * 64),
                policy=[0.0] * cfg.policy_size,
                value=0.0,
            )
        ]

    with patch.object(selfplay_worker, "play_one_game", side_effect=fake_play):
        examples = pool.run_iteration(3)
    assert len(played) == 3
    assert set(played) == {0, 1, 2}
    assert len(examples) == 3
    assert len(buf) == 3
