"""Smoke test for the application shell.

Nothing else in the suite instantiates :class:`MainWindow`, yet that is where the
screen wiring lives -- a renamed/removed signal on the Home screen, a broken
screen constructor, or a bad nav mapping only surfaces here. This drives the real
shell headless (offscreen Qt) with filesystem + QSettings isolated via Qt's test
mode, so it auto-creates a fresh "Untitled" workspace instead of touching the
user's real data.
"""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest  # noqa: E402
from PyQt6.QtCore import QSettings, QStandardPaths  # noqa: E402
from PyQt6.QtWidgets import QApplication  # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    QStandardPaths.setTestModeEnabled(True)  # redirect AppData/Config to a test dir
    app = QApplication.instance() or QApplication([])
    QSettings("MagPy", "MagPy").clear()  # fresh: no recent workspaces
    return app


def test_shell_constructs_and_defaults_to_untitled(qapp):
    from magpy.main_window import MainWindow

    window = MainWindow()
    assert window._workspace.name == "Untitled"


def test_every_nav_view_is_reachable(qapp):
    from magpy.main_window import MainWindow
    from magpy.screens import HuggingFaceScreen
    from magpy.widgets import ViewType

    window = MainWindow()
    # Navigating to every view must not raise and must change the current view.
    for view_type in ViewType:
        window._navigate_to(view_type)
        assert window._current_view == view_type

    # Hugging Face is now a real screen, not a placeholder.
    assert isinstance(window._screens[ViewType.HUGGINGFACE], HuggingFaceScreen)


def test_home_navigate_signal_drives_the_shell(qapp):
    from magpy.main_window import MainWindow
    from magpy.widgets import ViewType

    window = MainWindow()
    window._home_screen.navigate_requested.emit(ViewType.BATCH)
    assert window._current_view == ViewType.BATCH
