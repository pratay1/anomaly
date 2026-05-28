from __future__ import annotations

import sys
from pathlib import Path

from az.brain import load_brain
from az.config import Config
from az.gui import app as gui_app
from az.network.resnet import AlphaZeroResNet


def main():
    """Launch GUI in play-oriented mode (loads canonical anomaly.pt brain)."""
    cfg = Config()
    if len(sys.argv) > 1:
        cfg.brain_path = Path(sys.argv[1])
    model = AlphaZeroResNet(cfg)
    load_brain(model, "cpu", path=cfg.brain_path if len(sys.argv) > 1 else None)
    return gui_app.main(sys.argv)


if __name__ == "__main__":
    raise SystemExit(main())
