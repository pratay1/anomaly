#!/usr/bin/env python3
"""Headless training smoke test (no GUI)."""

import time

from az.config import Config
from az.training.orchestrator import TrainerOrchestrator


def main():
    cfg = Config()
    cfg.num_simulations = 25
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
