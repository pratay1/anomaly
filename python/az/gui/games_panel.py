from __future__ import annotations

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import QGroupBox, QListWidget, QListWidgetItem, QSizePolicy, QVBoxLayout

from az.ipc.events import GameFinished


class GamesPanel(QGroupBox):
    replay_requested = pyqtSignal(list)

    def __init__(self, parent=None):
        super().__init__("Recent Games", parent)
        self.setObjectName("games_panel")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 10, 8, 8)
        layout.setSpacing(4)
        self.setMaximumHeight(180)
        self.list = QListWidget()
        self.list.setAlternatingRowColors(True)
        self.list.setSpacing(1)
        self.list.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.list.itemClicked.connect(self._on_click)
        layout.addWidget(self.list)
        self._games: list[GameFinished] = []
        self._max_items = 50

    def add_game(self, game: GameFinished) -> None:
        self._games.insert(0, game)
        text = (
            f"G{game.game_id + 1}  {game.result:>4}  ·  "
            f"{game.plies:3} plies  ·  {game.examples_count} pos"
        )
        item = QListWidgetItem(text)
        item.setTextAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        self.list.insertItem(0, item)
        while len(self._games) > self._max_items:
            self._games.pop()
            self.list.takeItem(self.list.count() - 1)

    def _on_click(self, item: QListWidgetItem) -> None:
        row = self.list.row(item)
        if row < len(self._games):
            g = self._games[row]
            if g.moves_uci:
                self.replay_requested.emit(g.moves_uci)
