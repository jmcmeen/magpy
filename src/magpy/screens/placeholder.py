"""
Placeholder screen -- a reusable "Coming Soon" card for not-yet-built views.

Catalog screens, Settings, and other stubs share this so every nav button lands
on a consistent view instead of a blank widget. Swap a screen out for a real
implementation when it lands; the nav wiring doesn't change.
"""

from __future__ import annotations

from typing import Optional

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QFrame, QLabel, QVBoxLayout, QWidget

from .base import BaseScreen


def coming_soon_card(icon: str, title: str, description: str) -> QFrame:
    """Build the dashed 'Coming Soon' card used by placeholder views."""
    card = QFrame()
    card.setStyleSheet(
        "QFrame { background-color: #252526; border: 2px dashed #3c3c3c;"
        " border-radius: 8px; }"
    )
    layout = QVBoxLayout(card)
    layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
    layout.setSpacing(12)

    icon_label = QLabel(icon)
    icon_label.setStyleSheet("font-size: 48px; border: none;")
    icon_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
    layout.addWidget(icon_label)

    title_label = QLabel(title)
    title_label.setStyleSheet(
        "font-size: 24px; font-weight: bold; color: #d4d4d4; border: none;"
    )
    title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
    layout.addWidget(title_label)

    desc_label = QLabel(description)
    desc_label.setStyleSheet("font-size: 14px; color: #858585; border: none;")
    desc_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
    layout.addWidget(desc_label)

    badge = QLabel("Coming Soon")
    badge.setStyleSheet(
        "background-color: #0e639c; color: white; padding: 8px 16px;"
        " border-radius: 4px; font-weight: bold;"
    )
    badge.setAlignment(Qt.AlignmentFlag.AlignCenter)
    layout.addWidget(badge)
    return card


class PlaceholderScreen(BaseScreen):
    """A configurable BaseScreen that just shows a 'Coming Soon' card."""

    def __init__(
        self,
        name: str,
        icon: str,
        description: str,
        parent: Optional[QWidget] = None,
    ) -> None:
        # Set fields before super().__init__, which calls _setup_ui().
        self._name = name
        self._icon = icon
        self._description = description
        super().__init__(parent)

    @property
    def screen_name(self) -> str:
        return self._name

    @property
    def screen_icon(self) -> str:
        return self._icon

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.addWidget(coming_soon_card(self._icon, self._name, self._description))
