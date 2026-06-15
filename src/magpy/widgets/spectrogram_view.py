"""
SpectrogramView -- a pyqtgraph view that renders a :class:`SpectrogramImage`,
overlays the document's annotations, and hosts a transient selection.

The dB image arrives via :meth:`set_image` (the shell computes it). Annotation
overlays bind directly to the :class:`AnnotationSet` model (like the table), so
the view stays in sync without shell glue.

Two selection shapes, both transient (distinct from committed annotations):

* **Time region** (:meth:`start_selection`) -- a full-band span on the time axis,
  for "this stretch of the recording" selections.
* **Time-frequency box** (:meth:`start_box_selection`) -- a draggable/resizable 2-D
  ``RectROI`` capturing a time *and* frequency extent, for "this call in this
  band" selections.

Either way :meth:`selection_bounds` returns ``(start, end, low_freq, high_freq)``
(``low_freq``/``high_freq`` are ``None`` for a time region). Committed annotations
that carry frequency bounds render as filled boxes; those without (time-only
imports, promoted detector candidates) render as full-height regions.

No bioamla imports -- only the MagPy model and the plain ``Annotation`` DTO.
"""

from __future__ import annotations

from typing import Optional

import pyqtgraph as pg
from PyQt6.QtCore import QRectF
from PyQt6.QtWidgets import QGraphicsRectItem, QLabel, QVBoxLayout, QWidget

from magpy.models import AnnotationSet
from magpy.services import Annotation, SpectrogramImage

_ANN_BRUSH = pg.mkBrush(255, 235, 59, 40)
_ANN_SELECTED_BRUSH = pg.mkBrush(255, 140, 0, 90)
_BOX_PEN = pg.mkPen(255, 235, 59, width=1)
_BOX_SELECTED_PEN = pg.mkPen(255, 140, 0, width=2)
_SEL_BRUSH = pg.mkBrush(0, 200, 255, 50)
_SEL_PEN = pg.mkPen(0, 200, 255, width=2)


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
        self._regions: dict[str, pg.LinearRegionItem] = {}  # time-only annotations
        self._boxes: dict[str, QGraphicsRectItem] = {}  # freq-bounded annotations
        self._labels: dict[str, pg.TextItem] = {}
        self._selection: Optional[object] = None  # LinearRegionItem | RectROI

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

        self._cursor_label = QLabel("")
        self._cursor_label.setStyleSheet("color: #858585; padding: 0 4px;")
        layout.addWidget(self._cursor_label)
        self._plot.scene().sigMouseMoved.connect(self._on_mouse_moved)

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
        """Create a draggable full-band time selection centred in the view."""
        self.clear_selection()
        x0, x1 = self._plot.viewRange()[0]
        span = (x1 - x0) * 0.1
        mid = (x0 + x1) / 2
        self._selection = pg.LinearRegionItem(values=(mid - span, mid + span), brush=_SEL_BRUSH)
        self._plot.addItem(self._selection)

    def start_box_selection(self) -> None:
        """Create a draggable/resizable 2-D time-frequency selection box."""
        self.clear_selection()
        (x0, x1), (y0, y1) = self._plot.viewRange()
        x_span = (x1 - x0) * 0.2
        y0c, y1c = y0 + (y1 - y0) * 0.3, y0 + (y1 - y0) * 0.7
        mid = (x0 + x1) / 2
        roi = pg.RectROI(
            [mid - x_span / 2, y0c], [x_span, y1c - y0c],
            pen=_SEL_PEN, movable=True, resizable=True,
        )
        self._selection = roi
        self._plot.addItem(roi)

    def selection_bounds(self) -> Optional[tuple[float, float, Optional[float], Optional[float]]]:
        """Current selection as ``(start, end, low_freq, high_freq)`` or ``None``.

        ``low_freq``/``high_freq`` are ``None`` for a time region. Bounds are
        normalised (start ≤ end, low ≤ high) so a box dragged up/left can't make
        an inverted annotation, and frequency is clamped to ≥ 0.
        """
        if self._selection is None:
            return None
        if isinstance(self._selection, pg.LinearRegionItem):
            lo, hi = sorted(self._selection.getRegion())
            return (float(lo), float(hi), None, None)
        pos, size = self._selection.pos(), self._selection.size()
        start, end = sorted((pos.x(), pos.x() + size.x()))
        low, high = sorted((pos.y(), pos.y() + size.y()))
        return (float(start), float(end), max(0.0, float(low)), max(0.0, float(high)))

    def clear_selection(self) -> None:
        if self._selection is not None:
            self._plot.removeItem(self._selection)
            self._selection = None

    # --- cursor readout ---------------------------------------------------
    def _on_mouse_moved(self, scene_pos: object) -> None:
        if self._image.image is None:  # nothing loaded
            self._cursor_label.setText("")
            return
        vb = self._plot.getPlotItem().vb
        if not self._plot.sceneBoundingRect().contains(scene_pos):
            self._cursor_label.setText("")
            return
        pt = vb.mapSceneToView(scene_pos)
        self._cursor_label.setText(f"t = {pt.x():.3f} s     f = {pt.y():.0f} Hz")

    # --- annotation overlays ---------------------------------------------
    def _sync_annotations(self, *_: object) -> None:
        for item in (*self._regions.values(), *self._boxes.values(), *self._labels.values()):
            self._plot.removeItem(item)
        self._regions.clear()
        self._boxes.clear()
        self._labels.clear()
        for ann in self._model.items():
            if ann.low_freq is not None and ann.high_freq is not None:
                low, high = sorted((ann.low_freq, ann.high_freq))
                box = QGraphicsRectItem(QRectF(ann.start_time, low, ann.duration, high - low))
                box.setPen(_BOX_PEN)
                box.setBrush(_ANN_BRUSH)
                self._plot.addItem(box)  # routed to the ViewBox -> data coords
                self._boxes[ann.id] = box
                label_y = high
            else:
                region = pg.LinearRegionItem(
                    values=(ann.start_time, ann.end_time), movable=False, brush=_ANN_BRUSH
                )
                self._plot.addItem(region)
                self._regions[ann.id] = region
                label_y = self._f_max
            if ann.label:
                text = pg.TextItem(ann.label, anchor=(0, 1), color="w")
                text.setPos(ann.start_time, label_y)
                self._plot.addItem(text)
                self._labels[ann.id] = text
        self._restyle(self._model.selected)

    def _restyle(self, selected: Optional[Annotation]) -> None:
        sel_id = selected.id if selected is not None else None
        for ann_id, region in self._regions.items():
            region.setBrush(_ANN_SELECTED_BRUSH if ann_id == sel_id else _ANN_BRUSH)
            region.update()
        for ann_id, box in self._boxes.items():
            chosen = ann_id == sel_id
            box.setBrush(_ANN_SELECTED_BRUSH if chosen else _ANN_BRUSH)
            box.setPen(_BOX_SELECTED_PEN if chosen else _BOX_PEN)
