"""MagPy screens -- the multi-view workspace, ported from the legacy layout.

Each screen is a self-contained pure-Qt view. Most are :class:`BaseScreen`
subclasses; the audio screens (:class:`AudioAnnotationScreen`,
:class:`IndicesScreen`) are ``QMainWindow``\\ s sharing a :class:`BaseAudioScreen`
core so they can host their own docks. The honed navigation/layout was brought
over largely verbatim; several views are still placeholders pending the rebuild.
"""

from .annotation import AudioAnnotationScreen
from .base import BaseScreen
from .batch import BatchScreen
from .catalog import CatalogConfig, CatalogField, CatalogScreen
from .catalog_configs import (
    CATALOG_CONFIGS,
    EBIRD_CONFIG,
    INATURALIST_CONFIG,
    MACAULAY_CONFIG,
    XENO_CANTO_CONFIG,
)
from .datasets import DatasetsScreen
from .explore import ExploreScreen
from .home import HomeScreen
from .huggingface import HuggingFaceScreen
from .indices import IndicesScreen
from .placeholder import PlaceholderScreen, coming_soon_card
from .settings import SettingsScreen
from .training import TrainingScreen

__all__ = [
    "BaseScreen",
    "AudioAnnotationScreen",
    "IndicesScreen",
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
