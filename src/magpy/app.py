"""Application entry point: build the QApplication and show the MainWindow."""

from __future__ import annotations

import sys
from typing import Optional

from PyQt6.QtWidgets import QApplication

from magpy.main_window import MainWindow


def main(argv: Optional[list[str]] = None) -> int:
    argv = list(sys.argv if argv is None else argv)
    app = QApplication(argv)
    app.setApplicationName("MagPy")

    window = MainWindow()
    if len(argv) > 1:
        window.add_audio_path(argv[1])
    window.showMaximized()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
