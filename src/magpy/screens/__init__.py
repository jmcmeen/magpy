"""MagPy screens -- the multi-view workspace, ported from the legacy layout.

Each screen is a self-contained pure-Qt view. Most are :class:`BaseScreen`
subclasses; :class:`AudioScreen` is a ``QMainWindow`` so it can host its own
docks. The honed navigation/layout was brought over largely verbatim; several
views are still placeholders pending the rebuild.
"""

from .base import BaseScreen
from .audio import AudioScreen
from .home import HomeScreen
from .datasets import DatasetsScreen
from .training import TrainingScreen
from .batch import BatchScreen
from .explore import ExploreScreen
from .huggingface import HuggingFaceScreen
from .placeholder import PlaceholderScreen, coming_soon_card
from .settings import SettingsScreen
from .catalog import CatalogConfig, CatalogField, CatalogScreen
from .catalog_configs import (
    CATALOG_CONFIGS,
    EBIRD_CONFIG,
    INATURALIST_CONFIG,
    MACAULAY_CONFIG,
    XENO_CANTO_CONFIG,
)

__all__ = [
    "BaseScreen",
    "AudioScreen",
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
