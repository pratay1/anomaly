"""Patch stale native builds after merges when Python outpaces _az_core."""

from __future__ import annotations

import json
import time
from pathlib import Path


def _debug_log(hypothesis_id: str, message: str, data: dict) -> None:
    # #region agent log
    try:
        payload = {
            "sessionId": "16f4c8",
            "runId": "pre-fix",
            "hypothesisId": hypothesis_id,
            "location": "az/core_compat.py",
            "message": message,
            "data": data,
            "timestamp": int(time.time() * 1000),
        }
        log_path = Path(__file__).resolve().parents[2] / "debug-16f4c8.log"
        with log_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(payload) + "\n")
    except OSError:
        pass
    # #endregion


def ensure_board_api() -> None:
    import az._az_core as core

    methods = [name for name in dir(core.Board) if not name.startswith("_")]
    has_is_legal = hasattr(core.Board, "is_legal")
    _debug_log(
        "A",
        "Board native API snapshot",
        {"methods": methods, "has_is_legal": has_is_legal, "module": core.__file__},
    )

    if has_is_legal:
        return

    def is_legal(self, mv):  # type: ignore[no-untyped-def]
        for m in self.generate_legal_moves():
            if m.from_sq == mv.from_sq and m.to_sq == mv.to_sq and m.promotion == mv.promotion:
                return True
        return False

    core.Board.is_legal = is_legal  # type: ignore[method-assign]
    _debug_log("A", "Applied Board.is_legal Python shim", {"reason": "missing from native module"})


ensure_board_api()
