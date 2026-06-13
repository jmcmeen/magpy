"""
SpectrogramView -- a pyqtgraph view that renders a :class:`SpectrogramImage`,
overlays the document's annotations, and hosts a transient selection region.

The dB image arrives via :meth:`set_image` (the shell computes it). Annotation
overlays bind directly to the :class:`AnnotationSet` model (like the table), so
the view stays in sync without shell glue. The "selection" is a single
draggable time region the user positions and then promotes to an annotation;
it is transient UI state, distinct from committed annotations.

No bioamla imports -- only the MagPy model and the plain ``Annotation`` DTO.
"""

from __future__ import annotations

from typing import Optional

import pyqtgraph as pg
from PyQt6.QtCore import QRectF
from PyQt6.QtWidgets import QVBoxLayout, QWidget

from magpy.models import AnnotationSet
from magpy.services import Annotation, SpectrogramImage

_ANN_BRUSH = pg.mkBrush(255, 235, 59, 40)
_ANN_SELECTED_BRUSH = pg.mkBrush(255, 140, 0, 90)
_SEL_BRUSH = pg.mkBrush(0, 200, 255, 50)


def _default_colormap() -> pg.ColorMap:
    try:
        return pg.colormap.get("magma", source="matplotlib")
    except Exception:
        return pg.colormap.get("CET-L9")  # bundled fallback


class SpectrogramView(QWidget):
    """Renders a dB spectrogram with time (s) on x and frequency (Hz) on y."""

    def __init__(self, annotations: AnnotationSet, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._model = annotations
        self._f_max = 0.0
        self._duration = 0.0
        self._regions: dict[str, pg.LinearRegionItem] = {}
        self._labels: dict[str, pg.TextItem] = {}
        self._selection: Optional[pg.LinearRegionItem] = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self._plot = pg.PlotWidget()
        self._plot.setLabel("bottom", "Time", units="s")
        self._plot.setLabel("left", "Frequency", units="Hz")
        self._image = pg.ImageItem()
        self._image.setColorMap(_default_colormap())
        self._plot.addItem(self._image)
        self._playhead = pg.InfiniteLine(angle=90, movable=False, pen=pg.mkPen("w", width=1))
        self._playhead.setVisible(False)
        self._plot.addItem(self._playhead)
        layout.addWidget(self._plot)

        self._model.added.connect(self._sync_annotations)
        self._model.removed.connect(self._sync_annotations)
        self._model.reset.connect(self._sync_annotations)
        self._model.changed.connect(self._sync_annotations)
        self._model.selectionChanged.connect(self._restyle)

    # --- spectrogram image ------------------------------------------------
    def set_image(self, img: Optional[SpectrogramImage]) -> None:
        """Render ``img``, or clear the view when ``None``."""
        if img is None:
            self._image.clear()
            self._playhead.setVisible(False)
            self.clear_selection()
            return
        # ImageItem default axis order is col-major (image[x, y]); db is
        # (freq, time), so transpose to put time on x and frequency on y.
        self._f_max = img.f_max
        self._duration = img.duration
        self._image.setImage(img.db.T, autoLevels=True)
        self._image.setRect(QRectF(0.0, 0.0, img.duration, img.f_max))
        self._plot.setXRange(0.0, img.duration, padding=0)
        self._plot.setYRange(0.0, img.f_max, padding=0)
        self._playhead.setPos(0.0)
        self._playhead.setVisible(True)

    def set_playhead(self, seconds: float) -> None:
        """Move the playback position line to ``seconds``."""
        self._playhead.setPos(seconds)

    # --- transient selection ---------------------------------------------
    def start_selection(self) -> None:
        """Create a draggable selection region centred in the current view."""
        self.clear_selection()
        x0, x1 = self._plot.viewRange()[0]
        span = (x1 - x0) * 0.1
        mid = (x0 + x1) / 2
        self._selection = pg.LinearRegionItem(
            values=(mid - span, mid + span), brush=_SEL_BRUSH
        )
        self._plot.addItem(self._selection)

    def selection_range(self) -> Optional[tuple[float, float]]:
        """The current selection's (start, end) in seconds, or ``None``."""
        if self._selection is None:
            return None
        lo, hi = self._selection.getRegion()
        return (float(lo), float(hi))

    def clear_selection(self) -> None:
        if self._selection is not None:
            self._plot.removeItem(self._selection)
            self._selection = None

    # --- annotation overlays ---------------------------------------------
    def _sync_annotations(self, *_: object) -> None:
        for region in self._regions.values():
            self._plot.removeItem(region)
        for label in self._labels.values():
            self._plot.removeItem(label)
        self._regions.clear()
        self._labels.clear()
        for ann in self._model.items():
            region = pg.LinearRegionItem(
                values=(ann.start_time, ann.end_time), movable=False, brush=_ANN_BRUSH
            )
            self._plot.addItem(region)
            self._regions[ann.id] = region
            if ann.label:
                text = pg.TextItem(ann.label, anchor=(0, 1), color="w")
                text.setPos(ann.start_time, self._f_max)
                self._plot.addItem(text)
                self._labels[ann.id] = text
        self._restyle(self._model.selected)

    def _restyle(self, selected: Optional[Annotation]) -> None:
        sel_id = selected.id if selected is not None else None
        for ann_id, region in self._regions.items():
            region.setBrush(_ANN_SELECTED_BRUSH if ann_id == sel_id else _ANN_BRUSH)
            region.update()
