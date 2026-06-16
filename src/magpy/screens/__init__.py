"""MagPy screens -- the multi-view workspace, ported from the legacy layout.

Each screen is a self-contained pure-Qt :class:`BaseScreen`. The honed
navigation/layout was brought over largely verbatim; most views are still
placeholders pending the rebuild.
"""

from .base import BaseScreen
from .home_screen import HomeScreen
from .datasets_screen import DatasetsScreen
from .training_screen import TrainingScreen
from .batch_screen import BatchScreen
from .explore_screen import ExploreScreen
from .huggingface_screen import HuggingFaceScreen
from .placeholder import PlaceholderScreen, coming_soon_card
from .settings_screen import SettingsScreen
from .catalog_screen import CatalogConfig, CatalogField, CatalogScreen
from .catalog_configs import (
    CATALOG_CONFIGS,
    EBIRD_CONFIG,
    INATURALIST_CONFIG,
    MACAULAY_CONFIG,
    XENO_CANTO_CONFIG,
)

__all__ = [
    "BaseScreen",
    "HomeScreen",
    "DatasetsScreen",
    "TrainingScreen",
    "BatchScreen",
    "ExploreScreen",
    "HuggingFaceScreen",
    "PlaceholderScreen",
    "coming_soon_card",
    "SettingsScreen",
    "CatalogScreen",
    "CatalogConfig",
    "CatalogField",
    "CATALOG_CONFIGS",
    "XENO_CANTO_CONFIG",
    "MACAULAY_CONFIG",
    "INATURALIST_CONFIG",
    "EBIRD_CONFIG",
]
