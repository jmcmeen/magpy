"""Shared test setup: keep the suite off the developer's real MagPy state.

The app persists two things outside the repo: ``QSettings`` (recent workspaces,
table columns) and the auto-created "Untitled" workspace under the platform's
app-data directory. ``QStandardPaths`` test mode redirects the second but **not**
``QSettings`` -- on macOS and Windows those stay in the native per-user store --
so settings are pointed at a throwaway INI directory here, before any test (or
any import that builds a ``QSettings``) runs.
"""

from __future__ import annotations

import gc
import os
import tempfile

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtCore import QSettings, QStandardPaths  # noqa: E402

_SETTINGS_DIR = tempfile.mkdtemp(prefix="magpy-test-settings-")
QSettings.setDefaultFormat(QSettings.Format.IniFormat)
QSettings.setPath(QSettings.Format.IniFormat, QSettings.Scope.UserScope, _SETTINGS_DIR)
QStandardPaths.setTestModeEnabled(True)


@pytest.fixture(autouse=True)
def _collect_widgets_between_tests():
    """Free a test's widgets as it ends, not in the middle of a later test.

    Views sit in reference cycles (signal -> lambda -> self), so nothing frees
    them until the cyclic collector next happens to run. When that is partway
    through another test, a pyqtgraph scene is torn down out of order while Qt
    is still using it and the interpreter segfaults -- on Linux CI, not macOS.
    """
    yield
    gc.collect()
