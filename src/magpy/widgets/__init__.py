"""Widgets -- dumb Qt views. They bind to models and never import bioamla."""

from .annotation_table import AnnotationTable
from .detect_panel import DetectPanel
from .indices_panel import IndicesPanel
from .navigation_bar import NavigationBar, ViewType
from .properties_panel import PropertiesPanel
from .spectrogram_view import SpectrogramView
from .transport_bar import TransportBar
from .waveform_view import WaveformView
from .workspace_panel import WorkspacePanel

__all__ = [
    "SpectrogramView",
    "WaveformView",
    "TransportBar",
    "AnnotationTable",
    "DetectPanel",
    "IndicesPanel",
    "PropertiesPanel",
    "NavigationBar",
    "ViewType",
    "WorkspacePanel",
]
