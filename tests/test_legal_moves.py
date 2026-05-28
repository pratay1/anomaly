"""Chess legality: no king capture, must leave check, promotion round-trip."""

import az._az_core as core
import az.core_compat  # noqa: F401
import chess
from az.training.selfplay_worker import move_to_uci


def test_cannot_capture_enemy_king():
    # Black knight can reach white king square; must not be offered as legal.
    board = core.Board.from_fen("8/8/5n2/8/8/8/8/4K3 b - - 0 1")
    for m in board.generate_legal_moves():
        assert m.captured not in (core.Piece.WK, core.Piece.BK)
    assert not any(m.to_sq == 4 for m in board.generate_legal_moves())


def test_must_leave_check():
    fen = "8/8/8/8/8/8/5q2/4K3 w - - 0 1"
    board = core.Board.from_fen(fen)
    assert board.in_check(core.Color.White)
    ch = chess.Board(fen)
    assert ch.is_check()
    for m in board.generate_legal_moves():
        b2 = core.Board.from_fen(fen)
        b2.make_move(m)
        assert not b2.in_check(core.Color.White), move_to_uci(m)


def test_queen_promotion_roundtrip():
    fen = "8/4P3/8/8/8/8/8/4K2k w - - 0 1"
    board = core.Board.from_fen(fen)
    legal = core.legal_move_indices(board)
    assert legal
    e7 = 4 + 6 * 8  # pawn on e7
    promo_moves = [m for m in board.generate_legal_moves() if m.from_sq == e7 and m.promotion]
    assert len(promo_moves) == 4
    for mv in promo_moves:
        assert board.is_legal(mv)
        b2 = core.Board.from_fen(fen)
        b2.make_move(mv)
        piece = b2.at(mv.to_sq)
        assert piece in (core.Piece.WQ, core.Piece.WR, core.Piece.WB, core.Piece.WN)


def test_promotion_matches_python_chess():
    fen = "8/4P3/8/8/8/8/8/4K2k w - - 0 1"
    board = core.Board.from_fen(fen)
    ch = chess.Board(fen)
    cpp_uci = {move_to_uci(m) for m in board.generate_legal_moves()}
    py_uci = {m.uci() for m in ch.legal_moves}
    assert cpp_uci == py_uci
