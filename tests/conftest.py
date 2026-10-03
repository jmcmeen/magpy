"""Shared test setup: keep the suite off the developer's real MagPy state.

The app persists two things outside the repo: ``QSettings`` (recent workspaces,
table columns) and the auto-created "Untitled" workspace under the platform's
app-data directory. ``QStandardPaths`` test mode redirects the second but **not**
``QSettings`` -- on macOS and Windows those stay in the native per-user store --
so settings are pointed at a throwaway INI directory here, before any test (or
any import that builds a ``QSettings``) runs.
"""

from __future__ import annotations

import os
import tempfile

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtCore import QSettings, QStandardPaths  # noqa: E402

_SETTINGS_DIR = tempfile.mkdtemp(prefix="magpy-test-settings-")
QSettings.setDefaultFormat(QSettings.Format.IniFormat)
QSettings.setPath(QSettings.Format.IniFormat, QSettings.Scope.UserScope, _SETTINGS_DIR)
QStandardPaths.setTestModeEnabled(True)
