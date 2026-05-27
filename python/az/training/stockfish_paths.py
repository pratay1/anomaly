from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path


def resolve_stockfish_path() -> Path:
    """Locate a Stockfish binary via env, PATH, or common install locations."""
    env = os.environ.get("ANOMALY_STOCKFISH_PATH")
    if env:
        path = Path(env).expanduser()
        if path.is_file():
            return path.resolve()

    for name in ("stockfish", "stockfish.exe"):
        found = shutil.which(name)
        if found:
            return Path(found).resolve()

    candidates: list[Path] = []
    if sys.platform == "win32":
        candidates.extend(
            [
                Path(r"C:\Program Files\Stockfish\stockfish.exe"),
                Path.home() / "stockfish" / "stockfish.exe",
                Path(r"C:\Users\prata\stockfish\stockfish.exe"),
            ]
        )
    else:
        candidates.extend(
            [
                Path("/usr/games/stockfish"),
                Path("/usr/bin/stockfish"),
                Path("/opt/homebrew/bin/stockfish"),
            ]
        )

    for path in candidates:
        if path.is_file():
            return path.resolve()

    return Path("stockfish")


def stockfish_path_error(path: Path | None = None) -> str | None:
    """Return a user-facing error if Stockfish is not available at path."""
    resolved = path if path is not None else resolve_stockfish_path()
    if resolved.is_file():
        return None
    return (
        f"Stockfish not found at {resolved}. "
        "Install Stockfish and ensure it is on PATH, or set ANOMALY_STOCKFISH_PATH."
    )
