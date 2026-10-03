"""
WaveformView -- pyqtgraph amplitude view that rides above the spectrogram.

It draws a min/max envelope of **the visible window** (re-derived as the view
moves), so a long recording stays cheap to draw zoomed out and still resolves
individual cycles zoomed in. Its time axis is meant to be linked to the
spectrogram's (:meth:`link_time_axis`), so the two always show the same span.

It shares the spectrogram's mouse model (:class:`~magpy.widgets._plot.AudioViewBox`
in time-only mode): dragging draws a *time-only* selection
(:attr:`selectionDrawn`), a click seeks (:attr:`seekRequested`), the wheel zooms
time. It binds to the :class:`AnnotationSet` for overlays and shows the shared
playhead via :meth:`set_playhead`. No bioamla imports.
"""

from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from PyQt6.QtCore import Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import QVBoxLayout, QWidget

from magpy.models import AnnotationSet
from magpy.services import Annotation

from ._plot import AudioViewBox, label_color
from .spectrogram_view import LEFT_AXIS_WIDTH

_MAX_POINTS = 4000  # envelope points drawn for the visible window (~2 per pixel)
_RENDER_DEBOUNCE_MS = 30


def _brush(color: QColor, alpha: int) -> object:
    c = QColor(color)
    c.setAlpha(alpha)
    return pg.mkBrush(c)


class WaveformView(QWidget):
    seekRequested = pyqtSignal(float)  # seconds
    selectionDrawn = pyqtSignal(float, float, object, object)  # start, end, None, None

    def __init__(self, annotations: AnnotationSet, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._model = annotations
        self._data: np.ndarray | None = None  # mono samples (a reference, not a copy)
        self._sample_rate = 0
        self._duration = 0.0
        self._regions: dict[str, pg.LinearRegionItem] = {}

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self._vb = AudioViewBox(time_only=True)
        self._plot = pg.PlotWidget(viewBox=self._vb)
        self._plot.setLabel("left", "Amplitude")
        self._plot.getAxis("left").setWidth(LEFT_AXIS_WIDTH)
        self._plot.getAxis("left").enableAutoSIPrefix(False)  # amplitude isn't an SI quantity
        # The spectrogram below carries the time axis; a second one is noise.
        self._plot.hideAxis("bottom")
        self._plot.hideButtons()
        self._curve = self._plot.plot([], [], pen=pg.mkPen("#4fc3f7", width=1))
        self._playhead = pg.InfiniteLine(angle=90, movable=False, pen=pg.mkPen("w", width=1))
        self._playhead.setZValue(50)
        self._playhead.setVisible(False)
        self._plot.addItem(self._playhead)
        layout.addWidget(self._plot)

        self._render_timer = QTimer(self)
        self._render_timer.setSingleShot(True)
        self._render_timer.setInterval(_RENDER_DEBOUNCE_MS)
        self._render_timer.timeout.connect(self._render_visible)
        self._vb.sigXRangeChanged.connect(lambda *_: self._render_timer.start())
        self._vb.selectionDrawn.connect(self.selectionDrawn)

        self._plot.scene().sigMouseClicked.connect(self._on_clicked)
        self._model.added.connect(self._sync_annotations)
        self._model.removed.connect(self._sync_annotations)
        self._model.reset.connect(self._sync_annotations)
        self._model.changed.connect(self._sync_annotations)
        self._model.selectionChanged.connect(self._restyle)

    def link_time_axis(self, plot_item: pg.PlotItem) -> None:
        """Lock this view's time span to another plot's (the spectrogram's)."""
        self._plot.setXLink(plot_item)

    def set_draw_enabled(self, enabled: bool) -> None:
        """Whether dragging draws a selection (off: dragging pans instead)."""
        self._vb.set_draw_enabled(enabled)

    # --- waveform ---------------------------------------------------------
    def set_audio(self, samples: np.ndarray, sample_rate: int) -> None:
        self._data = self._to_mono(samples)
        self._sample_rate = sample_rate
        self._duration = len(self._data) / sample_rate if sample_rate else 0.0
        # Scale to the recording's own peak: field recordings are often quiet,
        # and a fixed +/-1 axis would draw them as a flat line.
        peak = float(np.max(np.abs(self._data))) if len(self._data) else 0.0
        y = max(peak, 1e-4) * 1.1
        self._vb.setLimits(xMin=0, xMax=self._duration, yMin=-y, yMax=y)
        self._plot.setYRange(-y, y, padding=0)
        self._plot.setXRange(0, self._duration, padding=0)
        self._playhead.setPos(0.0)
        self._playhead.setVisible(True)
        self._render_visible()

    def clear(self) -> None:
        self._data = None
        self._duration = 0.0
        self._curve.setData([], [])
        self._playhead.setVisible(False)

    def set_playhead(self, seconds: float) -> None:
        self._playhead.setPos(seconds)

    @staticmethod
    def _to_mono(samples: np.ndarray) -> np.ndarray:
        data = np.asarray(samples)
        if data.ndim > 1:  # collapse the (smaller) channel axis to mono
            data = data.mean(axis=int(np.argmin(data.shape)))
        return data

    def _render_visible(self) -> None:
        """Redraw the envelope for the time span currently in view."""
        if self._data is None or not self._sample_rate:
            return
        (x0, x1), _ = self._plot.viewRange()
        i0 = max(0, int(x0 * self._sample_rate))
        i1 = min(len(self._data), int(np.ceil(x1 * self._sample_rate)) + 1)
        times, values = self._envelope(self._data[i0:i1], self._sample_rate, i0)
        self._curve.setData(times, values)

    @staticmethod
    def _envelope(
        data: np.ndarray, sample_rate: int, offset: int = 0
    ) -> tuple[np.ndarray, np.ndarray]:
        """Min/max envelope of ``data``, which starts ``offset`` samples into the file.

        Short spans are returned sample-for-sample; longer ones are reduced to a
        min and a max per chunk so peaks survive the downsampling.
        """
        if not sample_rate or len(data) == 0:
            return np.array([]), np.array([])
        if len(data) <= _MAX_POINTS:
            return (np.arange(len(data)) + offset) / sample_rate, data
        factor = len(data) // (_MAX_POINTS // 2)
        n_chunks = len(data) // factor
        chunks = data[: n_chunks * factor].reshape(-1, factor)
        envelope = np.empty(n_chunks * 2, dtype=data.dtype)
        envelope[0::2] = chunks.min(axis=1)
        envelope[1::2] = chunks.max(axis=1)
        chunk_dur = factor / sample_rate
        times = np.repeat(np.arange(n_chunks) * chunk_dur, 2) + offset / sample_rate
        times[1::2] += chunk_dur / 2
        return times, envelope

    # --- interaction ------------------------------------------------------
    def _on_clicked(self, event) -> None:
        if self._duration <= 0 or event.button() != Qt.MouseButton.LeftButton:
            return
        if not self._vb.sceneBoundingRect().contains(event.scenePos()):
            return
        x = self._vb.mapSceneToView(event.scenePos()).x()
        self.seekRequested.emit(min(max(0.0, float(x)), self._duration))

    # --- annotation overlays ---------------------------------------------
    def _sync_annotations(self, *_: object) -> None:
        for region in self._regions.values():
            self._plot.removeItem(region)
        self._regions.clear()
        for ann in self._model.items():
            region = pg.LinearRegionItem(values=(ann.start_time, ann.end_time), movable=False)
            region.setZValue(10)
            self._plot.addItem(region)
            self._regions[ann.id] = region
        self._restyle(self._model.selected)

    def _restyle(self, selected: Annotation | None) -> None:
        sel_id = selected.id if selected is not None else None
        labels = {a.id: a.label for a in self._model.items()}
        for ann_id, region in self._regions.items():
            color = label_color(labels.get(ann_id, ""))
            region.setBrush(_brush(color, 90 if ann_id == sel_id else 40))
            region.update()
