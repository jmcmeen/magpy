"""Application entry point: build the QApplication and show the MainWindow.

Importing :mod:`magpy.main_window` pulls in the services seam and, through it,
bioamla + scipy + the ML stack -- about a second of import time on a warm cache,
more on a cold one. To avoid a blank/frozen window during that, we show a static
splash *before* the heavy import (which is why ``MainWindow`` is imported inside
:func:`main`, not at module top). The splash is intentionally static -- threading
the import or animating it fights Qt's single-threaded GUI model for no real win.
"""

from __future__ import annotations

import sys
from typing import Optional

from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QColor, QFont, QPainter, QPixmap
from PyQt6.QtWidgets import QApplication, QSplashScreen


def _build_splash() -> QSplashScreen:
    """A small dark splash drawn in code (no asset files to ship)."""
    pixmap = QPixmap(420, 240)
    pixmap.fill(QColor("#1e1e1e"))

    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    painter.setPen(QColor("#d4d4d4"))
    painter.setFont(QFont("Sans Serif", 40, QFont.Weight.Bold))
    painter.drawText(
        pixmap.rect().adjusted(0, 40, 0, 0),
        Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignTop,
        "MagPy",
    )
    painter.setPen(QColor("#808080"))
    painter.setFont(QFont("Sans Serif", 12))
    painter.drawText(
        pixmap.rect().adjusted(0, 130, 0, 0),
        Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignTop,
        "Bioacoustics Analysis Platform",
    )
    painter.end()

    splash = QSplashScreen(pixmap)
    splash.setWindowFlag(Qt.WindowType.WindowStaysOnTopHint, True)
    return splash


def main(argv: Optional[list[str]] = None) -> int:
    argv = list(sys.argv if argv is None else argv)
    app = QApplication(argv)
    app.setApplicationName("MagPy")

    splash = _build_splash()
    splash.show()
    splash.showMessage(
        "Loading…",
        Qt.AlignmentFlag.AlignBottom | Qt.AlignmentFlag.AlignHCenter,
        QColor("#808080"),
    )

    # Build the window from a one-shot timer *after* the event loop starts, so the
    # splash is actually mapped and painted before the ~1s+ heavy import
    # (bioamla/scipy/ML stack) blocks the thread. Doing the import inline here
    # would freeze before the window manager ever shows the splash -- which is why
    # it appeared not to show at all. Held in a list so it isn't garbage-collected
    # while app.exec() runs.
    holder: list = []

    def _load() -> None:
        from magpy.main_window import MainWindow

        window = MainWindow()
        holder.append(window)
        if len(argv) > 1:
            window.add_audio_path(argv[1])
        window.showMaximized()
        splash.finish(window)

    # A small delay (not 0) gives the window manager time to map + paint the
    # splash before the blocking import starts; 50 ms is imperceptible to startup.
    QTimer.singleShot(50, _load)
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
