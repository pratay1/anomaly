#!/usr/bin/env python3
"""Build anomaly.exe with PyInstaller. Output: dist/anomaly.exe"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    spec = root / "anomaly.spec"
    if not spec.exists():
        print(f"Missing spec: {spec}", file=sys.stderr)
        return 1
    cmd = [sys.executable, "-m", "PyInstaller", "--noconfirm", str(spec)]
    print("Running:", " ".join(cmd))
    result = subprocess.run(cmd, cwd=str(root))
    if result.returncode == 0:
        out = root / "dist" / "anomaly.exe"
        print(f"\nBuilt: {out}")
        print("At runtime, anomaly.pt and anomaly.json are written beside anomaly.exe.")
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
