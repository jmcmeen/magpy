"""
Explore Screen - embedding-space exploration (clustering & novelty).

Home for `bioamla.cluster` (fit / reduce / analyze / novelty): cluster model
embeddings, reduce dimensionality for visualization, and surface novel sounds.
Placeholder for now.
"""

from __future__ import annotations

from PyQt6.QtWidgets import QVBoxLayout

from .base import BaseScreen
from .placeholder import coming_soon_card


class ExploreScreen(BaseScreen):
    @property
    def screen_name(self) -> str:
        return "Explore"

    @property
    def screen_icon(self) -> str:
        return "🔭"

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.addWidget(
            coming_soon_card(
                "🔭",
                "Explore",
                "Explore audio in embedding space.\n\n"
                "- Cluster embeddings\n"
                "- Reduce dimensionality for visualization\n"
                "- Detect novel / out-of-distribution sounds\n"
                "- Surface structure across a dataset",
            )
        )
