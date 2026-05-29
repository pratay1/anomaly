from __future__ import annotations

import concurrent.futures
from pathlib import Path

import chess
import chess.engine

import az._az_core as core
from az.config import Config
from az.training.stockfish_paths import stockfish_path_error


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


def _kill_engine(engine: chess.engine.SimpleEngine | None) -> None:
    """Force-kill the engine process to prevent hangs on quit()."""
    if engine is None:
        return
    try:
        if engine._process is not None:
            engine._process.kill()
            engine._process.wait(timeout=5)
    except Exception:
        pass
    try:
        engine.quit()
    except Exception:
        pass


class StockfishEngine:
    """Synchronous UCI engine for serial Stockfish training (one caller thread)."""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self._engine: chess.engine.SimpleEngine | None = None

    def _restart(self) -> chess.engine.SimpleEngine:
        if self._engine is not None:
            _kill_engine(self._engine)
            self._engine = None
        path = Path(self.cfg.stockfish_path)
        err = stockfish_path_error(path)
        if err:
            raise FileNotFoundError(err)
        self._engine = chess.engine.SimpleEngine.popen_uci(str(path))
        return self._engine

    def _query_uci(self, board: core.Board, movetime_ms: int) -> tuple[str, chess.Move]:
        """Run the engine for movetime_ms and return (uci_str, chess.Move)."""
        fen = board.fen()
        limit = chess.engine.Limit(time=movetime_ms / 1000.0)
        timeout_s = movetime_ms / 1000.0 + 5.0
        last_err: Exception | None = None
        for _ in range(4):
            try:
                engine = self._engine if self._engine is not None else self._restart()
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                    board_obj = chess.Board(fen)
                    fut = pool.submit(engine.play, board_obj, limit)
                    result = fut.result(timeout=timeout_s)
                if result.move is None:
                    raise RuntimeError("Stockfish returned no move")
                return result.move.uci(), result.move
            except (chess.engine.EngineError, OSError, concurrent.futures.TimeoutError) as exc:
                last_err = exc
                _kill_engine(self._engine)
                self._engine = None
                self._restart()
        raise last_err or RuntimeError("Stockfish failed to return a move")

    def _uci_to_core_move(
        self, board: core.Board, uci: str, ch_move: chess.Move
    ) -> tuple[core.Move, int]:
        """Resolve a UCI string to (core.Move, move_index).  Raises if not legal."""
        legal = core.legal_move_indices(board)
        for idx in legal:
            mv = core.index_to_move(board, idx)
            if _move_to_uci(mv) == uci:
                return mv, idx
            try:
                if chess.Move.from_uci(_move_to_uci(mv)) == ch_move:
                    return mv, idx
            except ValueError:
                pass
        raise RuntimeError(
            f"Stockfish move {uci!r} is not legal in position {board.fen()!r}"
        )

    def choose_move(self, board: core.Board) -> core.Move:
        uci, ch_move = self._query_uci(board, self.cfg.stockfish_movetime_ms)
        mv, _ = self._uci_to_core_move(board, uci, ch_move)
        return mv

    def choose_move_with_idx(
        self, board: core.Board, movetime_ms: int
    ) -> tuple[core.Move, int]:
        """Return (move, move_index) using a specific time limit.

        Used by the Stockfish Critic to get SF's opinion at Anomaly's think budget.
        """
        uci, ch_move = self._query_uci(board, movetime_ms)
        return self._uci_to_core_move(board, uci, ch_move)

    def close(self) -> None:
        if self._engine is not None:
            _kill_engine(self._engine)
            self._engine = None
