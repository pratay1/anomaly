from __future__ import annotations

import random

import az._az_core as core

PIECE_VALUES: dict[core.Piece, int] = {
    core.Piece.WP: 1,
    core.Piece.WN: 3,
    core.Piece.WB: 3,
    core.Piece.WR: 5,
    core.Piece.WQ: 9,
    core.Piece.WK: 0,
    core.Piece.BP: -1,
    core.Piece.BN: -3,
    core.Piece.BB: -3,
    core.Piece.BR: -5,
    core.Piece.BQ: -9,
    core.Piece.BK: 0,
}


def material_eval(board: core.Board) -> float:
    score = 0.0
    for sq in range(64):
        p = board.at(sq)
        if p != getattr(core.Piece, "None"):
            score += PIECE_VALUES.get(p, 0)
    return score


def random_move(board: core.Board) -> core.Move:
    moves = board.generate_legal_moves()
    return random.choice(moves)
