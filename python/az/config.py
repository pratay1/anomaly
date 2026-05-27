from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

@dataclass
class Config:
    # Network (lighter defaults — less VRAM/RAM pressure on consumer GPUs)
    num_res_blocks: int = 6
    channels: int = 64
    policy_size: int = 4672
    encoding_channels: int = 119

    # MCTS (search "thinking" per move — lower = faster self-play, less compute per ply)
    num_simulations: int = 32
    c_puct_base: float = 19652.0
    c_puct_init: float = 1.25
    dirichlet_alpha: float = 0.3
    dirichlet_eps: float = 0.25
    temperature_moves: int = 15

    # Self-play
    num_workers: int = 2
    games_per_iteration: int = 4
    max_game_length: int = 120
    training_opponent: str = "self"  # "self" or "stockfish"
    stockfish_path: Path = field(
        default_factory=lambda: Path(r"C:\Users\prata\stockfish\stockfish.exe")
    )
    stockfish_movetime_ms: int = 50

    # Training
    batch_size: int = 64
    gradient_avg_workers: int = 2
    lr_schedule: list[tuple[int, float]] = field(
        default_factory=lambda: [(0, 0.01), (20_000, 0.001), (100_000, 0.0001)]
    )
    weight_decay: float = 1e-4
    replay_capacity: int = 50_000
    train_steps_per_iteration: int = 40
    momentum: float = 0.9

    # Inference server (higher max_wait_us batches more parallel MCTS evals per GPU forward)
    max_batch: int = 16
    max_wait_us: int = 8_000

    # Self-play telemetry (off during training — saves root_visits + Qt event traffic)
    emit_selfplay_visits: bool = False
    device: str = "cuda"

    # Arena
    arena_every_steps: int = 500
    arena_num_games: int = 10
    arena_opponent: str = "random"

    # GUI
    live_game_fps_cap: int = 8
    metrics_window: int = 2_000
    board_anim_ms: int = 160
    mcts_reveal_ms: int = 400  # GUI: show search heatmap before applying the move

    # Paths
    run_dir: Path = field(default_factory=lambda: Path("runs") / "default")
    brain_path: Path = field(default_factory=lambda: Path("anomaly.pt"))

    def games_per_selfplay_iteration(self) -> int:
        """Stockfish mode runs one serial game per iteration for stability."""
        return 1 if self.training_opponent == "stockfish" else self.num_workers

    def learning_rate(self, step: int) -> float:
        lr = self.lr_schedule[0][1]
        for s, v in self.lr_schedule:
            if step >= s:
                lr = v
        return lr

    def to_mcts_config(self):
        import az._az_core as core

        cfg = core.MCTSConfig()
        cfg.num_simulations = self.num_simulations
        cfg.c_puct_base = self.c_puct_base
        cfg.c_puct_init = self.c_puct_init
        cfg.dirichlet_alpha = self.dirichlet_alpha
        cfg.dirichlet_eps = self.dirichlet_eps
        cfg.add_root_noise = True
        return cfg

    def to_selfplay_config(self):
        import az._az_core as core

        cfg = core.SelfPlayConfig()
        cfg.mcts = self.to_mcts_config()
        cfg.temperature_moves = self.temperature_moves
        cfg.max_game_length = self.max_game_length
        return cfg
