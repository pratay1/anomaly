# AlphaZero Chess (Anomaly)

Educational **AlphaZero-style** chess system: a fast **C++17** core (bitboards, magic move generation, MCTS with PUCT, batched inference) talks to **PyTorch** training code and a **PyQt6** desktop app for live self-play, metrics, and optional human play against the network.

| Layer | Role |
|--------|------|
| `az._az_core` | Board, encoding, legal moves, MCTS, self-play workers, `InferenceQueue` (pybind11 extension built by CMake via scikit-build) |
| `az.training` | Replay buffer, central learner, inference server, orchestration |
| `az.network` | AlphaZero-style residual tower (policy + value) |
| `az.gui` | **Anomaly** main window: boards, MCTS heatmap, multi-game grid, Stockfish toggle, “Play vs Brain” |
| `az.eval` | Arena helpers (e.g. evaluation vs random baseline) |

## Features

- **Self-play** with configurable parallel workers (`Config.num_workers`) and MCTS simulations per move (`Config.num_simulations`).
- **Training opponent**: pure self-play, or games against **Stockfish** (`Config.training_opponent`, GUI toggle). Stockfish mode uses **one game per iteration** for stability (`games_per_selfplay_iteration()`).
- **Inference batching**: MCTS threads enqueue evaluations; `InferenceServer` batches GPU forwards (`max_batch`, `max_wait_us`).
- **Checkpoints** under each run: `runs/<timestamp>/` with `ckpt_*.pt`, `best.pt`, `meta.json`, plus sync to the canonical brain (below).
- **GUI**: Wikimedia SVG pieces (cached via `platformdirs`), board animations, metrics charts (`pyqtgraph`), optional root-visit heatmaps (`emit_selfplay_visits`), **Game Grid** / solo board, **Play vs Brain** dialog (`python-chess` for human moves).
- **Packaged binary**: optional PyInstaller build producing `anomaly.exe` with `anomaly.pt` beside the executable.

## Requirements

- **Python** 3.10+
- **CMake** 3.22+, **C++17** compiler (GCC, Clang, or MSVC)
- **PyTorch** 2.x with a GPU recommended (`Config.device` defaults to `"cuda"`; CPU works but is slow)
- Runtime deps (installed with the package): NumPy, PyQt6, pyqtgraph, platformdirs, python-chess
- **Stockfish** (optional): executable on `PATH` or path set in `Config.stockfish_path` (default name `stockfish`)

## Install

Editable install builds the C++ extension into the environment:

```bash
pip install -e "."
```

Development tools (pytest, ruff, mypy, pytest-qt):

```bash
pip install -e ".[dev]"
```

Frozen **anomaly** bundle build helpers:

```bash
pip install -e ".[package]"
```

## Command-line entry points

| Command | Description |
|---------|-------------|
| `az-train` | Opens the **Anomaly** PyQt6 app and training UI (same underlying app as `python -m az.gui.app`). |
| `az-play` | Same GUI entry point (alias for play-oriented launch scripts; brain resolution uses `anomaly.pt` / `ANOMALY_BRAIN_PATH` like the rest of the app). |
| `az-benchmark` | Runs a short ResNet forward throughput benchmark on CUDA if available, else CPU. |

Equivalent module invocation:

```bash
python -m az.gui.app
```

## Headless smoke test

No GUI — runs `TrainerOrchestrator` for a fixed sleep window (useful on CI or over SSH):

```bash
python scripts/train_headless.py
```

## Tests and quality

```bash
pytest tests/
ruff check python tests
mypy python/az
```

C++ perft-style checks live under `cpp/tests/`.

## Canonical brain (`anomaly.pt`)

Training and checkpoints update the **canonical** weights file **`anomaly.pt`** at the **project root** (discovered by walking up from the `az` package to a directory containing `pyproject.toml`). Sidecar metadata: **`anomaly.json`**.

Override location:

```bash
export ANOMALY_BRAIN_PATH=/path/to/your.pt
```

In a frozen build, the default is next to the executable.

Load from Python (external tools / notebooks):

```python
from az.brain import resolve_brain_path, load_brain
from az.config import Config
from az.network.resnet import AlphaZeroResNet

cfg = Config()
model = AlphaZeroResNet(cfg)
info = load_brain(model, device="cpu")
print(info.path, info.step)
```

`load_brain` / `align_cfg_with_brain` keep network widths in sync with saved hyperparameters when resuming.

## Configuration

Important fields live on **`az.config.Config`** (dataclass). Defaults are tuned for smaller consumer GPUs (e.g. fewer residual blocks/channels than a full-scale AlphaZero).

- **MCTS**: `num_simulations`, `c_puct_*`, Dirichlet root noise, `temperature_moves`
- **Self-play**: `num_workers`, `max_game_length`, `training_opponent`, `stockfish_path`, `stockfish_movetime_ms`
- **Training**: `batch_size`, `replay_capacity`, `train_steps_per_iteration`, `lr_schedule`, `gradient_avg_workers`
- **Inference server**: `max_batch`, `max_wait_us`
- **GUI**: `live_game_fps_cap`, `metrics_window`, `board_anim_ms`, `mcts_reveal_ms`
- **Telemetry**: `emit_selfplay_visits` — workers emit visit distributions for heatmaps; leave off for maximum self-play throughput
- **Arena telemetry**: `arena_every_steps`, `arena_num_games` (periodic lightweight arena stats for the metrics panel)

Each training session writes to **`runs/<YYYYMMDD_HHMMSS>/`** (`CheckpointManager`).

## Parallel training flow

1. **Self-play** workers run MCTS in C++, sharing an **`InferenceQueue`**.
2. **`InferenceServer`** (Python thread) runs the ResNet on batched encodings and returns policies/values.
3. Positions append to **`ReplayBuffer`**.
4. **`CentralLearner`** performs optimizer steps (SGD + momentum, LR schedule).
5. Weights save to the run directory and **`save_brain`** updates **`anomaly.pt`** / **`anomaly.json`**; the inference server **`reload_weights`** so the next iteration uses the new net.

**Throughput tips**

- Lower **`num_simulations`** for more games per hour.
- MCTS **reuses the search tree** across plies where possible (`advance_root` in the core).
- Tighter **`max_wait_us`** trades latency vs batch size; higher batches more pending evaluations per forward.
- Disable visit emission in workers when you do not need the GUI heatmap.

## Build `anomaly.exe` (Windows-oriented)

```bash
python scripts/build_anomaly.py
```

Output under `dist/`. At runtime the app reads/writes **`anomaly.pt`** next to the executable unless `ANOMALY_BRAIN_PATH` is set.

## Repository layout

- **`cpp/`** — Core library + `bindings.cpp` for `az._az_core`
- **`python/az/`** — Installable package (GUI, training, network, brain IO, IPC event types)
- **`scripts/`** — `build_anomaly.py`, `train_headless.py`, `debug_integration.py`
- **`tests/`** — Pytest suite

## Architecture (one paragraph)

Self-play threads execute **C++ MCTS**, which enqueues neural evaluations on **`InferenceQueue`**. A Python **`InferenceServer`** thread drains the queue, batches tensors, runs **`AlphaZeroResNet`**, and pushes results back. The **central learner** samples from **`ReplayBuffer`** and updates weights on the main thread (Torch dynamo disabled for compatibility). The Qt **`MainWindow`** connects orchestrator signals (`MovePlayed`, `GameFinished`, `TrainStep`, etc.) to boards, charts, and the MCTS panel.

## License

MIT
