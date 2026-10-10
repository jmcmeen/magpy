"""
The app's persistent settings (recent workspaces, table columns, ...).

Every ``QSettings`` in MagPy comes from :func:`app_settings` so there is exactly
one place that decides where settings live. It asks for Qt's *default format*
explicitly: ``QSettings(org, app)`` always uses the native per-user store and
ignores ``QSettings.setDefaultFormat``, which would leave tests no way to keep
their writes out of the developer's real preferences. Going through the default
format keeps the native store in normal use and lets the test suite redirect
everything to a throwaway directory (see ``tests/conftest.py``).
"""

from __future__ import annotations

from PyQt6.QtCore import QSettings

_ORGANIZATION = "MagPy"
_APPLICATION = "MagPy"


def app_settings() -> QSettings:
    """A handle on MagPy's settings store."""
    return QSettings(
        QSettings.defaultFormat(), QSettings.Scope.UserScope, _ORGANIZATION, _APPLICATION
    )
