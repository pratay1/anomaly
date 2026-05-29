#!/usr/bin/env python3
"""Integration repro: MCTS + inference (should finish in <30s)."""

import json
import sys
import threading
import time

import az._az_core as core
from az.config import Config
from az.network.resnet import AlphaZeroResNet
from az.training.inference_server import InferenceServer


def agent_log(msg: str, data: dict, hypothesis_id: str) -> None:
    try:
        with open("debug-3d7764.log", "a", encoding="utf-8") as f:
            f.write(
                json.dumps(
                    {
                        "sessionId": "3d7764",
                        "runId": "post-fix",
                        "hypothesisId": hypothesis_id,
                        "location": "debug_integration.py",
                        "message": msg,
                        "data": data,
                        "timestamp": int(time.time() * 1000),
                    }
                )
                + "\n"
            )
    except OSError:
        pass


def main() -> int:
    cfg = Config()
    cfg.max_batch = 8
    cfg.max_wait_us = 3000
    queue = core.InferenceQueue()
    model = AlphaZeroResNet(cfg)
    stop = threading.Event()
    srv = InferenceServer(queue, model, cfg, stop)
    srv.start()
    agent_log("inference_server_started", {}, "H1")

    board = core.Board()
    mcts_cfg = cfg.to_mcts_config()
    mcts_cfg.num_simulations = 5
    mcts = core.MCTS(queue, mcts_cfg)
    agent_log("mcts_run_begin", {}, "H1")
    t0 = time.time()
    pi = mcts.run(board, 1.0, 0)
    elapsed = time.time() - t0
    agent_log(
        "mcts_run_done",
        {"elapsed_s": elapsed, "policy_sum": float(sum(pi)), "policy_len": len(pi)},
        "H1",
    )
    stop.set()
    srv.join(timeout=3)
    print(f"MCTS OK in {elapsed:.2f}s, policy_sum={sum(pi):.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
