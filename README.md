# AlphaZero Chess

Educational AlphaZero-style chess engine with:

- **C++** core: bitboard board, move generation, AlphaZero encoding, MCTS (PUCT), inference queue
- **Python** training: PyTorch ResNet, replay buffer, learner, self-play orchestration
- **PyQt6** GUI: live self-play board with Wikimedia SVG pieces, animations, metrics charts, MCTS heatmap

## Requirements

- Python 3.10+
- CMake 3.22+
- C++17 compiler (MSVC, clang, or gcc)
- PyTorch, PyQt6, pyqtgraph

## Install

```bash
pip install -e ".[dev]"
```

## Run training GUI

```bash
az-train
# or
python -m az.gui.app
```

## Canonical brain (`anomaly.pt`)

Training writes a single findable model brain to **`anomaly.pt`** at the project root (or beside `anomaly.exe` when packaged). Metadata lives in **`anomaly.json`**.

External GUIs can load the brain with:

```python
from az.brain import resolve_brain_path, load_brain
from az.config import Config
from az.network.resnet import AlphaZeroResNet

cfg = Config()
model = AlphaZeroResNet(cfg)
info = load_brain(model, device="cpu")
print(info.path, info.step)
```

Override path with env `ANOMALY_BRAIN_PATH`.

## Build `anomaly.exe`

```bash
pip install -e ".[package]"
python scripts/build_anomaly.py
```

Output: `dist/anomaly.exe`. At runtime, `anomaly.pt` is read/written next to the executable.

## Parallel training

Each iteration runs **5 parallel self-play games** (configurable via `Config.num_workers`). All games must finish before the central learner averages gradients and updates `anomaly.pt`. Use **Show Game Grid** in the GUI to watch all games; double-click a mini board for solo view.

**Throughput tuning** (more games per hour without extra GPU work per position):

- `Config.num_simulations` — MCTS depth per move (default **64**; lower = faster thinking)
- MCTS **reuses the search tree** across plies (`advance_root`) instead of rebuilding each move
- `Config.emit_selfplay_visits` — set `True` in the GUI for heatmaps; leave `False` for max speed
- `Config.max_wait_us` — microsecond batching window for the inference server (default **12000**)

## Headless / benchmarks

```bash
az-benchmark
pytest tests/
```

## Architecture

Self-play threads call C++ MCTS, which batches neural net inference through `InferenceQueue` fulfilled by a Python `InferenceServer`. The learner samples positions from `ReplayBuffer` and updates the ResNet. Qt signals stream moves and metrics to the GUI.

## License

MIT
