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

A controls row above the plot offers colormap, contrast (dB display window),
auto-scroll-with-playhead, and zoom in/out/fit -- all view-local (no model or
bioamla involvement).

No bioamla imports -- only the MagPy model and the plain ``Annotation`` DTO.
"""

from __future__ import annotations

from typing import Optional

import pyqtgraph as pg
from PyQt6.QtCore import QRectF, Qt
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QGraphicsRectItem,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from magpy.models import AnnotationSet
from magpy.services import SpectrogramImage

_ANN_BRUSH = pg.mkBrush(255, 235, 59, 40)
_ANN_SELECTED_BRUSH = pg.mkBrush(255, 140, 0, 90)
_BOX_PEN = pg.mkPen(255, 235, 59, width=1)
_BOX_SELECTED_PEN = pg.mkPen(255, 140, 0, width=2)
_SEL_BRUSH = pg.mkBrush(0, 200, 255, 50)
_SEL_PEN = pg.mkPen(0, 200, 255, width=2)

_COLORMAPS = ("magma", "viridis", "plasma", "inferno", "cividis", "gray")


def _get_colormap(name: str) -> pg.ColorMap:
    try:
        return pg.colormap.get(name, source="matplotlib")
    except Exception:
        try:
            return pg.colormap.get(name)
        except Exception:
            return pg.colormap.get("CET-L9")  # bundled fallback


class SpectrogramView(QWidget):
    """Renders a dB spectrogram with time (s) on x and frequency (Hz) on y."""

    def __init__(self, annotations: AnnotationSet, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._model = annotations
        self._f_max = 0.0
        self._duration = 0.0
        self._db_min: Optional[float] = None
        self._db_max: Optional[float] = None
        self._auto_scroll = False
        self._regions: dict[str, pg.LinearRegionItem] = {}  # time-only annotations
        self._boxes: dict[str, QGraphicsRectItem] = {}  # unselected freq-bounded annotations
        self._labels: dict[str, pg.TextItem] = {}
        self._selection: Optional[object] = None  # LinearRegionItem | RectROI
        self._edit_roi: Optional[pg.RectROI] = None  # the selected box, made editable
        self._edit_id: Optional[str] = None
        self._syncing = False  # guards the rebuild path (build -> signals -> handlers)
        self._editing = False  # guards the edit path (drag -> model.update -> rebuild)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addLayout(self._build_controls())

        self._plot = pg.PlotWidget()
        self._plot.setLabel("bottom", "Time", units="s")
        self._plot.setLabel("left", "Frequency", units="Hz")
        self._image = pg.ImageItem()
        self._image.setColorMap(_get_colormap("magma"))
        self._plot.addItem(self._image)
        self._playhead = pg.InfiniteLine(angle=90, movable=False, pen=pg.mkPen("w", width=1))
        self._playhead.setVisible(False)
        self._plot.addItem(self._playhead)
        layout.addWidget(self._plot)

        self._cursor_label = QLabel("")
        self._cursor_label.setStyleSheet("color: #858585; padding: 0 4px;")
        layout.addWidget(self._cursor_label)
        self._plot.scene().sigMouseMoved.connect(self._on_mouse_moved)
        self._plot.scene().sigMouseClicked.connect(self._on_mouse_clicked)

        self._model.added.connect(self._sync_annotations)
        self._model.removed.connect(self._sync_annotations)
        self._model.reset.connect(self._sync_annotations)
        self._model.changed.connect(self._sync_annotations)
        # Selecting swaps the chosen box to an editable ROI -> full rebuild.
        self._model.selectionChanged.connect(self._sync_annotations)

    # --- controls ---------------------------------------------------------
    def _build_controls(self) -> QHBoxLayout:
        row = QHBoxLayout()
        row.setContentsMargins(4, 2, 4, 2)

        self._cmap_combo = QComboBox()
        self._cmap_combo.addItems(list(_COLORMAPS))
        self._cmap_combo.currentTextChanged.connect(
            lambda name: self._image.setColorMap(_get_colormap(name))
        )
        row.addWidget(QLabel("Colormap"))
        row.addWidget(self._cmap_combo)

        row.addWidget(QLabel("Contrast"))
        self._contrast = QSlider(Qt.Orientation.Horizontal)
        self._contrast.setRange(0, 90)
        self._contrast.setValue(20)
        self._contrast.setFixedWidth(110)
        self._contrast.valueChanged.connect(self._apply_levels)
        row.addWidget(self._contrast)

        self._auto_scroll_box = QCheckBox("Auto-scroll")
        self._auto_scroll_box.toggled.connect(self._set_auto_scroll)
        row.addWidget(self._auto_scroll_box)

        row.addStretch(1)
        for text, slot in (("−", self.zoom_out), ("+", self.zoom_in), ("Fit", self.zoom_fit)):
            b = QPushButton(text)
            b.setFixedWidth(36 if text != "Fit" else 44)
            b.clicked.connect(slot)
            row.addWidget(b)
        return row

    def _set_auto_scroll(self, enabled: bool) -> None:
        self._auto_scroll = enabled

    def _apply_levels(self, *_: object) -> None:
        if self._db_min is None or self._db_max is None:
            return
        # Contrast raises the display floor toward the peak: 0 = full range,
        # higher = only the louder bins remain visible.
        frac = self._contrast.value() / 100.0
        lo = self._db_min + frac * (self._db_max - self._db_min)
        self._image.setLevels((lo, self._db_max))

    def zoom_in(self) -> None:
        self._zoom(0.5)

    def zoom_out(self) -> None:
        self._zoom(2.0)

    def _zoom(self, factor: float) -> None:
        (x0, x1), _ = self._plot.viewRange()
        center = (x0 + x1) / 2
        half = (x1 - x0) * factor / 2
        lo = max(0.0, center - half)
        hi = min(self._duration, center + half) if self._duration else center + half
        self._plot.setXRange(lo, hi, padding=0)

    def zoom_fit(self) -> None:
        if self._duration:
            self._plot.setXRange(0.0, self._duration, padding=0)
            self._plot.setYRange(0.0, self._f_max, padding=0)

    # --- spectrogram image ------------------------------------------------
    def set_image(self, img: Optional[SpectrogramImage]) -> None:
        """Render ``img``, or clear the view when ``None``."""
        if img is None:
            self._image.clear()
            self._playhead.setVisible(False)
            self._db_min = self._db_max = None
            self.clear_selection()
            return
        # ImageItem default axis order is col-major (image[x, y]); db is
        # (freq, time), so transpose to put time on x and frequency on y.
        self._f_max = img.f_max
        self._duration = img.duration
        self._db_min = float(img.db.min())
        self._db_max = float(img.db.max())
        self._image.setImage(img.db.T, autoLevels=False)
        self._image.setRect(QRectF(0.0, 0.0, img.duration, img.f_max))
        self._apply_levels()
        self._plot.setXRange(0.0, img.duration, padding=0)
        self._plot.setYRange(0.0, img.f_max, padding=0)
        self._playhead.setPos(0.0)
        self._playhead.setVisible(True)

    def set_playhead(self, seconds: float) -> None:
        """Move the playback line; auto-scroll the view to follow if enabled."""
        self._playhead.setPos(seconds)
        if not self._auto_scroll or not self._duration:
            return
        (x0, x1), _ = self._plot.viewRange()
        width = x1 - x0
        if width > 0 and seconds > x1 - width * 0.1:  # near the right edge
            new0 = max(0.0, seconds - width * 0.2)
            new1 = new0 + width
            if new1 <= self._duration:
                self._plot.setXRange(new0, new1, padding=0)

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
        # Skip rebuilds we provoke ourselves (an in-progress drag edit), so we
        # don't tear down the ROI the user is holding.
        if self._editing:
            return
        self._syncing = True
        try:
            self._rebuild_overlays()
        finally:
            self._syncing = False

    def _rebuild_overlays(self) -> None:
        for item in (*self._regions.values(), *self._boxes.values(), *self._labels.values()):
            self._plot.removeItem(item)
        self._regions.clear()
        self._boxes.clear()
        self._labels.clear()
        if self._edit_roi is not None:
            self._plot.removeItem(self._edit_roi)
            self._edit_roi = None
            self._edit_id = None

        selected = self._model.selected
        sel_id = selected.id if selected is not None else None
        for ann in self._model.items():
            has_freq = ann.low_freq is not None and ann.high_freq is not None
            if has_freq:
                low, high = sorted((ann.low_freq, ann.high_freq))
                if ann.id == sel_id:
                    # The selected freq-box is editable: a draggable/resizable ROI.
                    roi = pg.RectROI(
                        [ann.start_time, low], [ann.duration, high - low],
                        pen=_BOX_SELECTED_PEN, movable=True, resizable=True,
                    )
                    roi.sigRegionChangeFinished.connect(self._on_box_edited)
                    self._plot.addItem(roi)
                    self._edit_roi = roi
                    self._edit_id = ann.id
                else:
                    box = QGraphicsRectItem(QRectF(ann.start_time, low, ann.duration, high - low))
                    box.setPen(_BOX_PEN)
                    box.setBrush(_ANN_BRUSH)
                    self._plot.addItem(box)  # routed to the ViewBox -> data coords
                    self._boxes[ann.id] = box
                label_y = high
            else:
                region = pg.LinearRegionItem(
                    values=(ann.start_time, ann.end_time), movable=False,
                    brush=_ANN_SELECTED_BRUSH if ann.id == sel_id else _ANN_BRUSH,
                )
                self._plot.addItem(region)
                self._regions[ann.id] = region
                label_y = self._f_max
            if ann.label:
                text = pg.TextItem(ann.label, anchor=(0, 1), color="w")
                text.setPos(ann.start_time, label_y)
                self._plot.addItem(text)
                self._labels[ann.id] = text

    def _on_box_edited(self) -> None:
        """A selected box ROI was dragged/resized -> write the new bounds back."""
        if self._syncing or self._edit_roi is None or self._edit_id is None:
            return
        ann = next((a for a in self._model.items() if a.id == self._edit_id), None)
        if ann is None:
            return
        pos, size = self._edit_roi.pos(), self._edit_roi.size()
        start, end = sorted((pos.x(), pos.x() + size.x()))
        low, high = sorted((pos.y(), pos.y() + size.y()))
        ann.start_time, ann.end_time = float(start), float(end)
        ann.low_freq, ann.high_freq = max(0.0, float(low)), max(0.0, float(high))
        self._editing = True  # suppress the rebuild our update would trigger
        try:
            self._model.update(ann)
        finally:
            self._editing = False
        # Keep the label glued to the moved box (overlays weren't rebuilt).
        text = self._labels.get(ann.id)
        if text is not None:
            text.setPos(ann.start_time, ann.high_freq)

    def _on_mouse_clicked(self, event: object) -> None:
        """Left-click a box/region to select it; click empty space to deselect."""
        if self._image.image is None or self._syncing:
            return
        if event.button() != Qt.MouseButton.LeftButton:
            return
        vb = self._plot.getPlotItem().vb
        pt = vb.mapSceneToView(event.scenePos())
        t, f = pt.x(), pt.y()
        hit = None
        best_area = None
        for ann in self._model.items():
            if not (ann.start_time <= t <= ann.end_time):
                continue
            if ann.low_freq is not None and ann.high_freq is not None:
                low, high = sorted((ann.low_freq, ann.high_freq))
                if not (low <= f <= high):
                    continue
                area = ann.duration * (high - low)
            else:
                area = ann.duration * (self._f_max or 1.0)  # region: full band
            if best_area is None or area < best_area:
                best_area, hit = area, ann
        self._model.select(hit)
