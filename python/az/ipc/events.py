from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class TrainStep:
    step: int
    policy_loss: float
    value_loss: float
    total_loss: float
    lr: float
    critic_loss: float = 0.0


@dataclass
class MovePlayed:
    fen: str
    uci: str
    from_sq: int
    to_sq: int
    game_id: int = 0
    visits: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class GameFinished:
    result: str
    plies: int
    examples_count: int
    game_id: int = 0
    moves_uci: list[str] = field(default_factory=list)


@dataclass
class IterationComplete:
    iteration: int
    games_finished: int
    train_steps: int
    brain_path: str


@dataclass
class CheckpointSaved:
    path: str
    step: int
    win_rate: float = 0.0


@dataclass
class ArenaResult:
    win_rate: float
    draws: int
    wins: int
    losses: int
