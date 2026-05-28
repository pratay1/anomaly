from __future__ import annotations

import chess
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QBrush, QColor, QPen
from PyQt6.QtSvgWidgets import QGraphicsSvgItem
from PyQt6.QtWidgets import QGraphicsRectItem, QGraphicsScene, QGraphicsView

from az.gui.piece_assets import PieceAssetManager
from az.gui.theme import (
    DARK_SQUARE,
    HEAT_HIGH,
    HEAT_LOW,
    HEAT_MID,
    LIGHT_SQUARE,
)


class MiniBoardView(QGraphicsView):
    """Compact board for game grid — FEN-synced, no incremental animation."""

    clicked = pyqtSignal(int)
    double_clicked = pyqtSignal(int)

    SQUARE = 32

    def __init__(self, game_id: int, assets: PieceAssetManager, anim_ms: int = 140, parent=None):
        super().__init__(parent)
        self.game_id = game_id
        self.assets = assets
        _ = anim_ms
        self.setScene(QGraphicsScene(self))
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setMinimumSize(self.SQUARE * 8 + 4, self.SQUARE * 8 + 4)
        self.setMaximumSize(self.SQUARE * 8 + 4, self.SQUARE * 8 + 4)
        self.setStyleSheet(
            "background: transparent; border: 1px solid #2a2a2a; border-radius: 4px;"
        )
        self._heat_overlays: dict[int, QGraphicsRectItem] = {}
        self._pieces: dict[int, QGraphicsSvgItem] = {}
        self._build_board()
        self.set_fen(chess.STARTING_FEN)

    def _scene(self) -> QGraphicsScene:
        scene = self.scene()
        assert scene is not None
        return scene

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
        self.setSceneRect(0, 0, 8 * self.SQUARE, 8 * self.SQUARE)

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
        c.setAlpha(int(40 + 120 * t))
        return c

    def set_heatmap_from_visits(self, fen: str, visits: list) -> None:
        heat: dict[int, float] = {}
        if visits:
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
        for sq, overlay in self._heat_overlays.items():
            alpha = heat.get(sq, 0.0)
            if alpha > 0:
                overlay.setBrush(QBrush(self._heat_color(alpha)))
                overlay.setVisible(True)
            else:
                overlay.setVisible(False)

    def _rebuild_all(self, board: chess.Board) -> None:
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
            f = chess.square_file(sq)
            rnk = chess.square_rank(sq)
            item.setPos(f * self.SQUARE, (7 - rnk) * self.SQUARE)
            item.setZValue(5)
            self._scene().addItem(item)
            self._pieces[sq] = item

    def clear_last_move(self) -> None:
        pass

    def set_last_move(self, from_sq: int, to_sq: int) -> None:
        pass

    def set_fen(self, fen: str, animated: bool = False) -> None:
        _ = animated
        try:
            board = chess.Board(fen)
        except ValueError:
            return
        for overlay in self._heat_overlays.values():
            overlay.setVisible(False)
        self._rebuild_all(board)

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.clicked.emit(self.game_id)
        super().mousePressEvent(event)

    def mouseDoubleClickEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.double_clicked.emit(self.game_id)
        super().mouseDoubleClickEvent(event)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.fitInView(self.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)
