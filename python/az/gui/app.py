from __future__ import annotations



import os



# Before any az import that loads torch.

os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")



import sys



from PyQt6.QtWidgets import QApplication



from az.config import Config

from az.gui.main_window import MainWindow





def main(argv: list[str] | None = None) -> int:

    argv = argv or sys.argv

    app = QApplication(argv)

    cfg = Config()

    win = MainWindow(cfg)

    win.show()

    win.assets.ensure_loaded()

    return app.exec()





if __name__ == "__main__":

    raise SystemExit(main())

