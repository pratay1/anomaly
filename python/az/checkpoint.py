from __future__ import annotations

import json
from pathlib import Path

import torch

from az.brain import save_brain
from az.config import Config
from az.network.resnet import AlphaZeroResNet


class CheckpointManager:
    def __init__(self, run_dir: Path):
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.best_path = self.run_dir / "best.pt"
        self.meta_path = self.run_dir / "meta.json"

    def save(
        self,
        model: AlphaZeroResNet,
        cfg: Config,
        step: int,
        win_rate: float = 0.0,
        tag: str | None = None,
        iteration: int = 0,
    ) -> Path:
        path = self.run_dir / (f"ckpt_{step}.pt" if tag is None else f"ckpt_{tag}.pt")
        payload = {
            "state_dict": model.state_dict(),
            "step": step,
            "win_rate": win_rate,
            "hparams": cfg.__dict__,
        }
        tmp = path.with_suffix(".tmp")
        torch.save(payload, tmp)
        tmp.replace(path)
        meta = {"step": step, "win_rate": win_rate, "latest": str(path)}
        self.meta_path.write_text(json.dumps(meta, indent=2))
        if win_rate >= 0.5 or not self.best_path.exists():
            torch.save(payload, self.best_path)
        save_brain(
            model,
            cfg,
            step,
            win_rate=win_rate,
            iteration=iteration,
            run_dir=self.run_dir,
        )
        return path

    def load_best(self, model: AlphaZeroResNet, device: str = "cpu") -> int:
        if not self.best_path.exists():
            return 0
        data = torch.load(self.best_path, map_location=device, weights_only=False)
        model.load_state_dict(data["state_dict"])
        return int(data.get("step", 0))
