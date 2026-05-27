# AGENTS.md

## Cursor Cloud specific instructions

### Overview

AlphaZero Chess: C++ core (pybind11) + Python (PyTorch) training + PyQt6 GUI. Single-process application, no external services required.

### Prerequisites (system packages)

The C++ extension build requires `python3-dev` and Qt runtime requires `libegl1`. These are installed in the VM snapshot but may be needed if rebuilding from scratch:

```
sudo apt-get install -y python3-dev libegl1
```

### Development commands

| Task | Command |
|------|---------|
| Install deps | `pip install -e ".[dev]"` |
| Lint | `ruff check python/ tests/` |
| Tests | `QT_QPA_PLATFORM=offscreen pytest tests/ -v` |
| Benchmark | `az-benchmark` |
| Headless training | `python3 scripts/train_headless.py` |
| GUI training | `az-train` (requires display or xvfb) |

### Important notes

- **PATH**: Scripts install to `/home/ubuntu/.local/bin`. Ensure it is on PATH (`export PATH="/home/ubuntu/.local/bin:$PATH"`).
- **Qt offscreen mode**: Tests require `QT_QPA_PLATFORM=offscreen` since pytest-qt tries to load Qt GUI during collection.
- **C++ extension**: The editable install (`pip install -e ".[dev]"`) compiles the `_az_core` native module via scikit-build-core + CMake. If C++ sources change, re-run the install command.
- **Thread cleanup abort**: The C++ self-play threads may emit "terminate called without an active exception" on process exit after `orch.stop()`. This is benign — it occurs after clean shutdown during thread join/cleanup at interpreter exit.
- **No GPU required**: Training falls back to CPU automatically when CUDA is unavailable.
