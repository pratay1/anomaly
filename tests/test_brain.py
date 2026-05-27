from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

from az.brain import load_brain, resolve_brain_meta_path, resolve_brain_path, save_brain
from az.config import Config
from az.network.resnet import AlphaZeroResNet


def test_resolve_brain_path_dev(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "pyproject.toml").write_text("[project]\nname='test'\n")
    with patch("az.brain._project_root", return_value=tmp_path):
        path = resolve_brain_path()
    assert path == tmp_path / "anomaly.pt"


def test_resolve_brain_path_env(tmp_path, monkeypatch):
    custom = tmp_path / "custom.pt"
    monkeypatch.setenv("ANOMALY_BRAIN_PATH", str(custom))
    assert resolve_brain_path() == custom.resolve()


def test_save_and_load_brain(tmp_path, monkeypatch):
    monkeypatch.setenv("ANOMALY_BRAIN_PATH", str(tmp_path / "anomaly.pt"))
    cfg = Config()
    model = AlphaZeroResNet(cfg)
    path = save_brain(model, cfg, step=42, iteration=3)
    assert path.exists()
    meta_path = resolve_brain_meta_path(path)
    assert meta_path.exists()
    meta = json.loads(meta_path.read_text())
    assert meta["step"] == 42
    assert meta["iteration"] == 3

    model2 = AlphaZeroResNet(cfg)
    info = load_brain(model2, device="cpu", path=path)
    assert info.step == 42
    assert info.iteration == 3


def test_load_brain_missing_returns_zero(tmp_path):
    cfg = Config()
    model = AlphaZeroResNet(cfg)
    missing = tmp_path / "missing.pt"
    info = load_brain(model, device="cpu", path=missing)
    assert info.step == 0
