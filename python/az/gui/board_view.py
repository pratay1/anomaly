from __future__ import annotations

import chess
from PyQt6.QtCore import QPointF, QPropertyAnimation, Qt, QVariantAnimation
from PyQt6.QtGui import QBrush, QColor, QFont, QPen
from PyQt6.QtSvgWidgets import QGraphicsSvgItem
from PyQt6.QtWidgets import (
    QGraphicsDropShadowEffect,
    QGraphicsOpacityEffect,
    QGraphicsRectItem,
    QGraphicsScene,
    QGraphicsSimpleTextItem,
    QGraphicsView,
)

from az.gui.board_anim import animate_piece, detect_move, is_same_piece_slide
from az.gui.piece_assets import PieceAssetManager
from az.gui.theme import (
    COORD_TEXT,
    DARK_SQUARE,
    HEAT_HIGH,
    HEAT_LOW,
    HEAT_MID,
    LAST_MOVE_HIGHLIGHT,
    LIGHT_SQUARE,
    THINKING_GLOW,
)


class _PulseRing(QGraphicsRectItem):
    """Subtle border pulse while MCTS is searching."""

    def __init__(self, size: float):
        super().__init__(0, 0, size, size)
        pen = QPen(QColor(THINKING_GLOW), 2)
        pen.setCosmetic(True)
        self.setPen(pen)
        self.setBrush(QBrush(Qt.BrushStyle.NoBrush))
        self.setZValue(20)
        self._effect = QGraphicsOpacityEffect()
        self._effect.setOpacity(0.0)
        self.setGraphicsEffect(self._effect)
        self._anim: QPropertyAnimation | None = None

    def pulse(self, active: bool) -> None:
        if self._anim:
            self._anim.stop()
            self._anim = None
        if not active:
            self.setVisible(False)
            return
        self.setVisible(True)
        self._anim = QPropertyAnimation(self._effect, b"opacity")
        self._anim.setDuration(1100)
        self._anim.setStartValue(0.15)
        self._anim.setEndValue(0.75)
        self._anim.setLoopCount(-1)
        self._anim.start()


class BoardView(QGraphicsView):
    """Main chess board — premium look, slide animations on normal moves."""

    SQUARE = 64
    COORD_MARGIN = 18

    def __init__(self, assets: PieceAssetManager, anim_ms: int = 160, parent=None):
        super().__init__(parent)
        self.assets = assets
        self._anim_ms = anim_ms
        self._last_board: chess.Board | None = None
        self._active_anims: list[QVariantAnimation] = []
        self.setScene(QGraphicsScene(self))
        self.setRenderHints(self.renderHints())
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setStyleSheet("background: transparent; border: none;")

        shadow = QGraphicsDropShadowEffect(self)
        shadow.setBlurRadius(28)
        shadow.setOffset(0, 4)
        shadow.setColor(QColor(0, 0, 0, 160))
        self.setGraphicsEffect(shadow)

        self._heat_overlays: dict[int, QGraphicsRectItem] = {}
        self._last_move_overlays: list[QGraphicsRectItem] = []
        self._pieces: dict[int, QGraphicsSvgItem] = {}
        self._build_board()
        self._pulse = _PulseRing(8 * self.SQUARE)
        self._scene().addItem(self._pulse)
        self._pulse.setVisible(False)
        self.set_fen(chess.STARTING_FEN, animated=False)

    def _scene(self) -> QGraphicsScene:
        scene = self.scene()
        assert scene is not None
        return scene

    def _sq_pos(self, sq: int) -> QPointF:
        f = chess.square_file(sq)
        r = chess.square_rank(sq)
        return QPointF(f * self.SQUARE, (7 - r) * self.SQUARE)

    def _stop_anims(self) -> None:
        for anim in self._active_anims:
            anim.stop()
        self._active_anims.clear()

    def _build_board(self) -> None:
        for r in range(8):
            for f in range(8):
                sq = chess.square(f, 7 - r)
                color = LIGHT_SQUARE if (f + r) % 2 == 0 else DARK_SQUARE
                x, y = f * self.SQUARE, r * self.SQUARE
                rect = QGraphicsRectItem(x, y, self.SQUARE, self.SQUARE)
                rect.setBrush(QBrush(QColor(color)))
                rect.setPen(QPen(Qt.PenStyle.NoPen))
                rect.setZValue(0)
                self._scene().addItem(rect)

                heat = QGraphicsRectItem(x, y, self.SQUARE, self.SQUARE)
                heat.setBrush(QBrush(QColor(0, 0, 0, 0)))
                heat.setPen(QPen(Qt.PenStyle.NoPen))
                heat.setVisible(False)
                heat.setZValue(3)
                self._heat_overlays[sq] = heat
                self._scene().addItem(heat)

        border = QGraphicsRectItem(0, 0, 8 * self.SQUARE, 8 * self.SQUARE)
        pen = QPen(QColor(COORD_TEXT))
        pen.setWidth(1)
        pen.setCosmetic(True)
        border.setPen(pen)
        border.setBrush(QBrush(Qt.BrushStyle.NoBrush))
        border.setZValue(15)
        self._scene().addItem(border)

        self._add_coordinates()

        m = self.COORD_MARGIN
        self.setSceneRect(-m, -m, 8 * self.SQUARE + 2 * m, 8 * self.SQUARE + 2 * m)

    def _add_coordinates(self) -> None:
        font = QFont("Segoe UI", 9)
        font.setWeight(QFont.Weight.DemiBold)
        files = "abcdefgh"
        m = self.COORD_MARGIN
        for f in range(8):
            t = QGraphicsSimpleTextItem(files[f])
            t.setFont(font)
            t.setBrush(QBrush(QColor(COORD_TEXT)))
            br = t.boundingRect()
            cx = f * self.SQUARE + self.SQUARE / 2 - br.width() / 2
            t.setPos(cx, 8 * self.SQUARE + (m - br.height()) / 2)
            t.setZValue(15)
            self._scene().addItem(t)
        for r in range(8):
            t = QGraphicsSimpleTextItem(str(8 - r))
            t.setFont(font)
            t.setBrush(QBrush(QColor(COORD_TEXT)))
            br = t.boundingRect()
            cy = r * self.SQUARE + self.SQUARE / 2 - br.height() / 2
            t.setPos(-m + (m - br.width()) / 2, cy)
            t.setZValue(15)
            self._scene().addItem(t)

    def _heat_color(self, alpha: float) -> QColor:
        t = min(1.0, max(0.0, alpha))
        low, mid, high = QColor(HEAT_LOW), QColor(HEAT_MID), QColor(HEAT_HIGH)
        if t < 0.5:
            u = t * 2
            c = QColor(
                int(low.red() + (mid.red() - low.red()) * u),
                int(low.green() + (mid.green() - low.green()) * u),
                int(low.blue() + (mid.blue() - low.blue()) * u),
            )
        else:
            u = (t - 0.5) * 2
            c = QColor(
                int(mid.red() + (high.red() - mid.red()) * u),
                int(mid.green() + (high.green() - mid.green()) * u),
                int(mid.blue() + (high.blue() - mid.blue()) * u),
            )
        c.setAlpha(int(60 + 150 * t))
        return c

    def clear_heatmap(self) -> None:
        self.set_heatmap({})

    def set_heatmap(self, visits: dict[int, float]) -> None:
        for sq, overlay in self._heat_overlays.items():
            alpha = visits.get(sq, 0.0)
            if alpha > 0:
                overlay.setBrush(QBrush(self._heat_color(alpha)))
                overlay.setVisible(True)
            else:
                overlay.setVisible(False)

    def set_heatmap_from_visits(self, fen: str, visits: list) -> None:
        if not visits:
            self.clear_heatmap()
            return
        heat: dict[int, float] = {}
        max_n = max((v.get("N", getattr(v, "N", 0)) for v in visits)) or 1
        try:
            import az._az_core as core

            board = core.Board.from_fen(fen)
            for v in visits:
                n = v.get("N", getattr(v, "N", 0))
                idx = v.get("move_index", getattr(v, "move_index", -1))
                if idx < 0:
                    continue
                mv = core.index_to_move(board, idx)
                heat[mv.from_sq] = max(heat.get(mv.from_sq, 0), n / max_n)
                heat[mv.to_sq] = max(heat.get(mv.to_sq, 0), n / max_n)
        except Exception:
            pass
        self.set_heatmap(heat)

    def clear_last_move(self) -> None:
        for item in self._last_move_overlays:
            self._scene().removeItem(item)
        self._last_move_overlays.clear()

    def set_last_move(self, from_sq: int, to_sq: int) -> None:
        self.clear_last_move()
        for sq in (from_sq, to_sq):
            f = chess.square_file(sq)
            r = chess.square_rank(sq)
            x = f * self.SQUARE
            y = (7 - r) * self.SQUARE
            overlay = QGraphicsRectItem(x, y, self.SQUARE, self.SQUARE)
            overlay.setBrush(QBrush(QColor(LAST_MOVE_HIGHLIGHT)))
            overlay.setPen(QPen(Qt.PenStyle.NoPen))
            overlay.setZValue(7)
            self._scene().addItem(overlay)
            self._last_move_overlays.append(overlay)

    def set_thinking(self, active: bool) -> None:
        self._pulse.pulse(active)

    def _find_move(self, board: chess.Board, from_sq: int, to_sq: int) -> chess.Move | None:
        for m in board.legal_moves:
            if m.from_square == from_sq and m.to_square == to_sq:
                return m
        return None

    def _needs_full_sync(self, board: chess.Board, move: chess.Move) -> bool:
        if move.promotion or board.is_castling(move) or board.is_en_passant(move):
            return True
        return False

    def _animate_slide(
        self, new_board: chess.Board, from_sq: int, to_sq: int
    ) -> bool:
        if to_sq in self._pieces:
            self._scene().removeItem(self._pieces[to_sq])
            del self._pieces[to_sq]

        item = self._pieces.get(from_sq)
        if item is None:
            return False

        del self._pieces[from_sq]
        self._pieces[to_sq] = item

        from_pos = self._sq_pos(from_sq)
        to_pos = self._sq_pos(to_sq)
        anim = animate_piece(item, from_pos, to_pos, self._anim_ms)
        self._active_anims.append(anim)

        def _done(a: QVariantAnimation = anim) -> None:
            if a in self._active_anims:
                self._active_anims.remove(a)

        anim.finished.connect(_done)
        self._last_board = new_board.copy()
        return True

    def _rebuild_all(self, board: chess.Board) -> None:
        self._stop_anims()
        for item in list(self._pieces.values()):
            self._scene().removeItem(item)
        self._pieces.clear()
        for sq in chess.SQUARES:
            p = board.piece_at(sq)
            if p is None:
                continue
            sym = p.symbol()
            renderer = self.assets.renderer_for_piece_char(sym)
            if not renderer:
                continue
            item = QGraphicsSvgItem()
            item.setSharedRenderer(renderer)
            item.setScale(self.SQUARE / 45.0)
            item.setPos(self._sq_pos(sq))
            item.setZValue(10)
            self._scene().addItem(item)
            self._pieces[sq] = item
        self._last_board = board.copy()

    def set_fen(self, fen: str, animated: bool = True) -> None:
        try:
            new_board = chess.Board(fen)
        except ValueError:
            return

        old_board = self._last_board
        self.clear_heatmap()

        if old_board is None or not animated:
            self._rebuild_all(new_board)
            return

        move_pair = detect_move(old_board, new_board)
        if move_pair is None:
            self._rebuild_all(new_board)
            return

        from_sq, to_sq = move_pair
        move = self._find_move(old_board, from_sq, to_sq)
        if move is None or self._needs_full_sync(old_board, move):
            self._rebuild_all(new_board)
            return

        if not is_same_piece_slide(old_board, new_board, from_sq, to_sq):
            self._rebuild_all(new_board)
            return

        if not self._animate_slide(new_board, from_sq, to_sq):
            self._rebuild_all(new_board)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.fitInView(self.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)
