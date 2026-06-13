"""
Base Screen - Abstract base class for application screens.

Provides lifecycle methods and common functionality for all screens.
"""

from __future__ import annotations

from abc import abstractmethod
from typing import Optional

from PyQt6.QtWidgets import QWidget


class BaseScreen(QWidget):
    """
    Abstract base class for application screens.

    Screens are the main content areas that can be switched between
    using the navigation bar. Each screen maintains its own state
    and can be activated/deactivated as the user navigates.
    """

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._is_active = False
        self._setup_ui()

    @property
    @abstractmethod
    def screen_name(self) -> str:
        """Return the display name of this screen."""
        pass

    @property
    @abstractmethod
    def screen_icon(self) -> str:
        """Return the icon for this screen."""
        pass

    @abstractmethod
    def _setup_ui(self):
        """Set up the screen's UI. Called once during initialization."""
        pass

    def activate(self):
        """Called when this screen becomes visible."""
        self._is_active = True
        self.on_activate()

    def deactivate(self):
        """Called when this screen is hidden."""
        self._is_active = False
        self.on_deactivate()

    def on_activate(self):
        """Override to perform actions when screen becomes active."""
        pass

    def on_deactivate(self):
        """Override to perform actions when screen becomes inactive."""
        pass

    @property
    def is_active(self) -> bool:
        """Return whether this screen is currently active."""
        return self._is_active
