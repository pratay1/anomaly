#!/usr/bin/env python3
"""Headless training smoke test (no GUI)."""

import time

from az.config import Config
from az.keepalive import activate_keepalive
from az.training.orchestrator import TrainerOrchestrator


def main():
    activate_keepalive()
    cfg = Config()
    cfg.mcts_think_time_ms_min = 200
    cfg.mcts_think_time_ms_max = 500
    cfg.max_batch = 16
    cfg.num_workers = 2
    orch = TrainerOrchestrator(cfg)
    orch.start()
    print("Training started… (60s smoke test)")
    time.sleep(60)
    orch.stop()
    print("Done.")


if __name__ == "__main__":
    main()
