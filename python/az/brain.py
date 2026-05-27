from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import torch

from az.config import Config
from az.network.resnet import AlphaZeroResNet


@dataclass
class BrainInfo:
    path: Path
    step: int
    iteration: int = 0
    win_rate: float = 0.0
    run_dir: str | None = None


def _project_root() -> Path:
    """Walk up from az package to find project root (contains pyproject.toml or runs/)."""
    here = Path(__file__).resolve().parent
    for parent in [here, *here.parents]:
        if (parent / "pyproject.toml").exists():
            return parent
        if (parent / "runs").is_dir() and (parent / "python" / "az").is_dir():
            return parent
    return Path.cwd()


def resolve_brain_path() -> Path:
    """Return canonical anomaly.pt path for dev or frozen exe."""
    env = os.environ.get("ANOMALY_BRAIN_PATH")
    if env:
        return Path(env).resolve()

    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent / "anomaly.pt"

    return _project_root() / "anomaly.pt"


def resolve_brain_meta_path(brain_path: Path | None = None) -> Path:
    path = brain_path or resolve_brain_path()
    return path.with_suffix(".json")


def _brain_payload(
    model: AlphaZeroResNet,
    cfg: Config,
    step: int,
    win_rate: float = 0.0,
    iteration: int = 0,
) -> dict:
    return {
        "state_dict": model.state_dict(),
        "step": step,
        "win_rate": win_rate,
        "iteration": iteration,
        "hparams": cfg.__dict__,
    }


def save_brain(
    model: AlphaZeroResNet,
    cfg: Config,
    step: int,
    path: Path | None = None,
    win_rate: float = 0.0,
    iteration: int = 0,
    run_dir: Path | str | None = None,
) -> Path:
    """Atomically write anomaly.pt and sidecar anomaly.json."""
    brain_path = Path(path) if path else resolve_brain_path()
    brain_path.parent.mkdir(parents=True, exist_ok=True)
    payload = _brain_payload(model, cfg, step, win_rate, iteration)
    tmp = brain_path.with_suffix(".pt.tmp")
    torch.save(payload, tmp)
    tmp.replace(brain_path)

    meta = {
        "step": step,
        "win_rate": win_rate,
        "iteration": iteration,
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "brain_path": str(brain_path),
        "run_dir": str(run_dir) if run_dir else None,
    }
    meta_path = resolve_brain_meta_path(brain_path)
    meta_tmp = meta_path.with_suffix(".json.tmp")
    meta_tmp.write_text(json.dumps(meta, indent=2))
    meta_tmp.replace(meta_path)
    return brain_path


def infer_arch_from_state_dict(state_dict: dict) -> tuple[int, int]:
    block_ids = [
        int(k.split(".")[1])
        for k in state_dict
        if k.startswith("blocks.") and k.endswith(".conv1.weight")
    ]
    num_blocks = (max(block_ids) + 1) if block_ids else 0
    stem = state_dict.get("stem.0.weight")
    channels = int(stem.shape[0]) if stem is not None else 0
    return num_blocks, channels


def align_cfg_with_brain(cfg: Config, path: Path | None = None) -> None:
    """Match network shape in cfg to checkpoint before constructing the model."""
    brain_path = Path(path) if path else resolve_brain_path()
    if not brain_path.exists():
        return

    data = torch.load(brain_path, map_location="cpu", weights_only=False)
    hparams = data.get("hparams")
    sd = data.get("state_dict", {})
    ck_blocks, ck_channels = infer_arch_from_state_dict(sd)
    if isinstance(hparams, dict):
        if "num_res_blocks" in hparams:
            cfg.num_res_blocks = int(hparams["num_res_blocks"])
        if "channels" in hparams:
            cfg.channels = int(hparams["channels"])
    elif ck_blocks and ck_channels:
        cfg.num_res_blocks = ck_blocks
        cfg.channels = ck_channels


def load_brain(
    model: AlphaZeroResNet,
    device: str = "cpu",
    path: Path | None = None,
) -> BrainInfo:
    """Load canonical brain weights into model. Returns BrainInfo with step 0 if missing."""
    brain_path = Path(path) if path else resolve_brain_path()
    if not brain_path.exists():
        return BrainInfo(path=brain_path, step=0, iteration=0)

    data = torch.load(brain_path, map_location=device, weights_only=False)
    model.load_state_dict(data["state_dict"])
    step = int(data.get("step", 0))
    iteration = int(data.get("iteration", 0))
    win_rate = float(data.get("win_rate", 0.0))
    run_dir = None
    meta_path = resolve_brain_meta_path(brain_path)
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text())
            run_dir = meta.get("run_dir")
            iteration = int(meta.get("iteration", iteration))
        except (json.JSONDecodeError, OSError):
            pass
    return BrainInfo(
        path=brain_path,
        step=step,
        iteration=iteration,
        win_rate=win_rate,
        run_dir=run_dir,
    )
