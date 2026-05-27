from __future__ import annotations

import chess
from PyQt6.QtCore import QEasingCurve, QPointF, QVariantAnimation
from PyQt6.QtSvgWidgets import QGraphicsSvgItem


def is_same_piece_slide(
    old: chess.Board, new: chess.Board, from_sq: int, to_sq: int
) -> bool:
    """True when the moving piece keeps type and color (safe to slide SVG)."""
    old_p = old.piece_at(from_sq)
    new_p = new.piece_at(to_sq)
    if old_p is None or new_p is None:
        return False
    return old_p.piece_type == new_p.piece_type and old_p.color == new_p.color


def detect_move(old: chess.Board, new: chess.Board) -> tuple[int, int] | None:
    """Return (from_sq, to_sq) if exactly one legal move explains the delta."""
    if old.fen().split()[0] == new.fen().split()[0]:
        return None
    candidates: list[tuple[int, int]] = []
    for move in old.legal_moves:
        b = old.copy()
        b.push(move)
        if b.board_fen() == new.board_fen() and b.turn == new.turn:
            candidates.append((move.from_square, move.to_square))
    if len(candidates) == 1:
        return candidates[0]
    return None


def animate_piece(
    item: QGraphicsSvgItem,
    from_pos: QPointF,
    to_pos: QPointF,
    duration_ms: int = 160,
) -> QVariantAnimation:
    item.setPos(from_pos)
    anim = QVariantAnimation()
    anim.setDuration(duration_ms)
    anim.setStartValue(from_pos)
    anim.setEndValue(to_pos)
    anim.setEasingCurve(QEasingCurve.Type.OutQuad)
    anim.valueChanged.connect(lambda p: item.setPos(p))
    anim.start()
    return anim
