"""
Datasets Screen - Manage audio datasets for training and analysis.

Provides tools for organizing, labeling, and preparing datasets.
"""

from __future__ import annotations


from PyQt6.QtWidgets import QVBoxLayout

from .base import BaseScreen
from .placeholder import coming_soon_card


class DatasetsScreen(BaseScreen):
    """
    Datasets screen: build training data from annotated sources.

    This is also **where audio editing lives** -- transforms produce *new*
    derived audio artifacts (source audio stays immutable), so analysis output
    never goes stale. Will provide: clip extraction from annotations, editing/
    augmentation, train/val/test partitioning, label prep, and manifests/stats.
    """

    @property
    def screen_name(self) -> str:
        return "Datasets"

    @property
    def screen_icon(self) -> str:
        return "📁"

    def _setup_ui(self):
        """Set up the datasets screen UI."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.addWidget(
            coming_soon_card(
                "📁",
                "Datasets",
                "Build training data — and the home for audio editing.\n\n"
                "- Extract labeled clips from annotations\n"
                "- Edit & augment (trim, normalize, denoise, pitch/time, noise)\n"
                "- Partition into train / val / test\n"
                "- Prepare labels; build manifests & stats\n\n"
                "Edits write new artifacts — sources stay immutable.",
            )
        )
