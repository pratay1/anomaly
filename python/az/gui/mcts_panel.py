from __future__ import annotations

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from az.gui.board_view import BoardView
from az.gui.theme import BG_ELEVATED, BORDER_SUBTLE

_CARDS_ROW_HEIGHT = 76
_PANEL_HEIGHT = 118


class _MoveCard(QFrame):
    """Single candidate move — always visible; skeleton or live data."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("mcts_card")
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.setFixedHeight(_CARDS_ROW_HEIGHT)
        self.setMinimumWidth(72)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 6, 8, 6)
        layout.setSpacing(4)

        self.move_lbl = QLabel("—")
        self.move_lbl.setObjectName("mcts_move")
        self.move_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.move_lbl)

        self.bar = QFrame()
        self.bar.setObjectName("mcts_bar")
        self.bar.setFixedHeight(28)
        layout.addWidget(self.bar)

        self.stat_lbl = QLabel("· · ·")
        self.stat_lbl.setObjectName("mcts_stat")
        self.stat_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.stat_lbl)
        self.set_skeleton()

    def set_skeleton(self) -> None:
        self.setObjectName("mcts_card_skeleton")
        self.move_lbl.setText("—")
        self.stat_lbl.setText("· · ·")
        self.stat_lbl.setToolTip("")
        self.bar.setStyleSheet(
            f"""
            QFrame#mcts_bar {{
                background: {BG_ELEVATED};
                border-radius: 4px;
                border: 1px solid {BORDER_SUBTLE};
            }}
            """
        )

    def set_data(self, uci: str, n: int, q: float, p: float, frac: float) -> None:
        self.setObjectName("mcts_card")
        self.move_lbl.setText(uci)
        self.stat_lbl.setText(f"N{n} · Q{q:+.2f}")
        self.stat_lbl.setToolTip(f"Prior P = {p:.3f}")
        t = min(1.0, max(0.0, frac))
        top_stop = 1.0 - t
        self.bar.setStyleSheet(
            f"""
            QFrame#mcts_bar {{
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 {BG_ELEVATED},
                    stop:{top_stop:.3f} {BG_ELEVATED},
                    stop:{top_stop:.3f} #7a7a7a,
                    stop:1 #5a5a5a);
                border-radius: 4px;
                border: 1px solid {BORDER_SUBTLE};
            }}
            """
        )


class MCTSPanel(QGroupBox):
    def __init__(self, board_view: BoardView, parent=None):
        super().__init__("Search", parent)
        self.setObjectName("mcts_panel")
        self.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
        self.setFixedHeight(_PANEL_HEIGHT)
        self.board_view = board_view

        outer = QVBoxLayout(self)
        outer.setContentsMargins(10, 8, 10, 10)
        outer.setSpacing(6)

        header = QHBoxLayout()
        header.setSpacing(8)
        self.status = QLabel("Waiting for search…")
        self.status.setObjectName("mcts_status")
        header.addWidget(self.status)
        header.addStretch()
        outer.addLayout(header)

        self._cards_row = QWidget()
        self._cards_row.setFixedHeight(_CARDS_ROW_HEIGHT)
        cards_layout = QHBoxLayout(self._cards_row)
        cards_layout.setContentsMargins(0, 0, 0, 0)
        cards_layout.setSpacing(6)
        self._cards: list[_MoveCard] = []
        for _ in range(5):
            card = _MoveCard()
            cards_layout.addWidget(card, stretch=1)
            self._cards.append(card)
        outer.addWidget(self._cards_row)

    def _fill_skeletons(self, from_index: int = 0) -> None:
        for card in self._cards[from_index:]:
            card.set_skeleton()

    def update_visits(self, fen: str, visits: list) -> None:
        if not visits:
            self._fill_skeletons()
            return
        self.board_view.set_thinking(False)
        self.board_view.set_heatmap_from_visits(fen, visits)
        max_n = max((v.get("N", getattr(v, "N", 0)) for v in visits)) or 1
        sorted_v = sorted(
            visits,
            key=lambda v: v.get("N", getattr(v, "N", 0)),
            reverse=True,
        )[:5]
        self.status.setText(f"{len(visits)} edges · best N={max_n}")
        for i, card in enumerate(self._cards):
            if i >= len(sorted_v):
                card.set_skeleton()
                continue
            v = sorted_v[i]
            n = v.get("N", getattr(v, "N", 0))
            q = v.get("Q", getattr(v, "Q", 0))
            p = v.get("P", getattr(v, "P", 0))
            idx = v.get("move_index", getattr(v, "move_index", -1))
            try:
                import az._az_core as core

                mv = core.index_to_move(core.Board.from_fen(fen), idx)
                uci = self._uci(mv)
            except Exception:
                uci = str(idx)
            card.set_data(uci, n, q, p, n / max_n)

    def set_searching(self) -> None:
        self.board_view.set_thinking(True)
        self.status.setText("Searching…")
        self._fill_skeletons()

    @staticmethod
    def _uci(m) -> str:
        files = "abcdefgh"
        return (
            f"{files[m.from_sq % 8]}{m.from_sq // 8 + 1}"
            f"{files[m.to_sq % 8]}{m.to_sq // 8 + 1}"
        )
