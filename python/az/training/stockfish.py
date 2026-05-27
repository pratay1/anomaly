from __future__ import annotations

from pathlib import Path

import chess
import chess.engine

import az._az_core as core
from az.config import Config


def _sq_uci(sq: int) -> str:
    return chr(ord("a") + (sq % 8)) + str((sq // 8) + 1)


def _move_to_uci(m: core.Move) -> str:
    uci = _sq_uci(m.from_sq) + _sq_uci(m.to_sq)
    pmap = {
        core.PieceType.Knight: "n",
        core.PieceType.Bishop: "b",
        core.PieceType.Rook: "r",
        core.PieceType.Queen: "q",
    }
    if m.promotion in pmap:
        uci += pmap[m.promotion]
    return uci


class StockfishEngine:
    """Synchronous UCI engine for serial Stockfish training (one caller thread)."""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self._engine: chess.engine.SimpleEngine | None = None

    def _restart(self) -> chess.engine.SimpleEngine:
        if self._engine is not None:
            try:
                self._engine.quit()
            except Exception:
                pass
            self._engine = None
        path = Path(self.cfg.stockfish_path)
        if not path.is_file():
            raise FileNotFoundError(f"Stockfish not found at {path}")
        self._engine = chess.engine.SimpleEngine.popen_uci(str(path))
        return self._engine

    def choose_move(self, board: core.Board) -> core.Move:
        fen = board.fen()
        limit = chess.engine.Limit(time=self.cfg.stockfish_movetime_ms / 1000.0)
        last_err: Exception | None = None
        for _ in range(2):
            try:
                engine = self._engine if self._engine is not None else self._restart()
                result = engine.play(chess.Board(fen), limit)
                if result.move is None:
                    raise RuntimeError("Stockfish returned no move")
                uci = result.move.uci()
                ch_move = result.move
                break
            except (chess.engine.EngineError, OSError) as exc:
                last_err = exc
                self._restart()
        else:
            raise last_err or RuntimeError("Stockfish failed to return a move")

        legal = core.legal_move_indices(board)
        for idx in legal:
            mv = core.index_to_move(board, idx)
            if _move_to_uci(mv) == uci:
                return mv
            try:
                if chess.Move.from_uci(_move_to_uci(mv)) == ch_move:
                    return mv
            except ValueError:
                pass
        raise RuntimeError(f"Stockfish move {uci!r} is not legal in position {fen!r}")

    def close(self) -> None:
        if self._engine is not None:
            try:
                self._engine.quit()
            except Exception:
                pass
            self._engine = None
