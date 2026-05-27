from __future__ import annotations

from pathlib import Path

from platformdirs import user_cache_dir
from PyQt6.QtCore import QByteArray, QObject, QUrl, pyqtSignal
from PyQt6.QtGui import QPixmap
from PyQt6.QtNetwork import QNetworkAccessManager, QNetworkReply, QNetworkRequest
from PyQt6.QtSvg import QSvgRenderer

# Cburnett Wikimedia Commons SVG set (12 pieces)
PIECE_URLS = {
    "wP": "https://upload.wikimedia.org/wikipedia/commons/4/45/Chess_plt45.svg",
    "wN": "https://upload.wikimedia.org/wikipedia/commons/7/70/Chess_nlt45.svg",
    "wB": "https://upload.wikimedia.org/wikipedia/commons/b/b1/Chess_blt45.svg",
    "wR": "https://upload.wikimedia.org/wikipedia/commons/7/72/Chess_rlt45.svg",
    "wQ": "https://upload.wikimedia.org/wikipedia/commons/1/15/Chess_qlt45.svg",
    "wK": "https://upload.wikimedia.org/wikipedia/commons/4/42/Chess_klt45.svg",
    "bP": "https://upload.wikimedia.org/wikipedia/commons/c/c7/Chess_pdt45.svg",
    "bN": "https://upload.wikimedia.org/wikipedia/commons/e/ef/Chess_ndt45.svg",
    "bB": "https://upload.wikimedia.org/wikipedia/commons/9/98/Chess_bdt45.svg",
    "bR": "https://upload.wikimedia.org/wikipedia/commons/f/ff/Chess_rdt45.svg",
    "bQ": "https://upload.wikimedia.org/wikipedia/commons/4/47/Chess_qdt45.svg",
    "bK": "https://upload.wikimedia.org/wikipedia/commons/f/f0/Chess_kdt45.svg",
}

# Minimal fallback SVG (white pawn) used if download fails
FALLBACK_SVG = {
    "wP": '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 45 45"><path fill="#fff" stroke="#000" d="M22.5 9c-2.21 0-4 1.79-4 4 0 .89.29 1.71.78 2.38C17.33 16.5 16 18.59 16 21c0 2.03.94 3.84 2.41 5.03C17.06 27.18 16 29.4 16 32v9h13v-9c0-2.6-1.06-4.82-2.41-5.97 1.47-1.19 2.41-3 2.41-5.03 0-2.41-1.33-4.5-3.28-5.62.49-.67.78-1.49.78-2.38 0-2.21-1.79-4-4-4z"/></svg>',
    "bP": '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 45 45"><path fill="#000" stroke="#fff" d="M22.5 9c-2.21 0-4 1.79-4 4 0 .89.29 1.71.78 2.38C17.33 16.5 16 18.59 16 21c0 2.03.94 3.84 2.41 5.03C17.06 27.18 16 29.4 16 32v9h13v-9c0-2.6-1.06-4.82-2.41-5.97 1.47-1.19 2.41-3 2.41-5.03 0-2.41-1.33-4.5-3.28-5.62.49-.67.78-1.49.78-2.38 0-2.21-1.79-4-4-4z"/></svg>',
}


class PieceAssetManager(QObject):
    ready = pyqtSignal()
    progress = pyqtSignal(int, int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.cache_dir = Path(user_cache_dir("alphazero-chess")) / "pieces"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.renderers: dict[str, QSvgRenderer] = {}
        self._nam = QNetworkAccessManager(self)
        self._pending: list[str] = []
        self._done = 0

    def ensure_loaded(self) -> None:
        keys = list(PIECE_URLS.keys())
        for key in keys:
            path = self.cache_dir / f"{key}.svg"
            if path.exists():
                self._load_file(key, path)
                self._done += 1
            else:
                self._pending.append(key)
        if not self._pending:
            self.ready.emit()
            return
        self.progress.emit(self._done, len(keys))
        for key in self._pending:
            self._download(key)

    def _load_file(self, key: str, path: Path) -> None:
        r = QSvgRenderer(str(path))
        if r.isValid():
            self.renderers[key] = r

    def _load_bytes(self, key: str, data: bytes) -> None:
        r = QSvgRenderer(QByteArray(data))
        if r.isValid():
            self.renderers[key] = r
            (self.cache_dir / f"{key}.svg").write_bytes(data)

    def _download(self, key: str) -> None:
        url = PIECE_URLS[key]
        req = QNetworkRequest(QUrl(url))
        reply = self._nam.get(req)
        reply.finished.connect(lambda r=reply, k=key: self._on_finished(r, k))

    def _on_finished(self, reply: QNetworkReply, key: str) -> None:
        if reply.error() == QNetworkReply.NetworkError.NoError:
            self._load_bytes(key, reply.readAll().data())
        else:
            fb = FALLBACK_SVG.get(key) or FALLBACK_SVG["wP"]
            self._load_bytes(key, fb.encode("utf-8"))
        reply.deleteLater()
        self._done += 1
        total = len(PIECE_URLS)
        self.progress.emit(self._done, total)
        if self._done >= total:
            # Fill missing with fallback pawn
            for k in PIECE_URLS:
                if k not in self.renderers:
                    fb = FALLBACK_SVG.get("wP" if k[0] == "w" else "bP") or FALLBACK_SVG["wP"]
                    self._load_bytes(k, fb.encode("utf-8"))
            self.ready.emit()

    def renderer_for_piece_char(self, c: str) -> QSvgRenderer | None:
        if c == "." or c == " ":
            return None
        color = "w" if c.isupper() else "b"
        kind = c.upper()
        key = f"{color}{kind}"
        return self.renderers.get(key)

    def pixmap(self, key: str, size: int = 64) -> QPixmap:
        r = self.renderers.get(key)
        if not r:
            return QPixmap(size, size)
        pm = QPixmap(size, size)
        pm.fill()
        from PyQt6.QtGui import QPainter

        p = QPainter(pm)
        r.render(p)
        p.end()
        return pm
