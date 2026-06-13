"""
WaveformView -- pyqtgraph amplitude view, ported from the legacy WaveformWidget.

Keeps the legacy min/max envelope downsampling (the part that matters for long
recordings), and is adapted to the new architecture: it binds to the
:class:`AnnotationSet` for overlays (like :class:`SpectrogramView`), shows the
shared playhead via :meth:`set_playhead`, and emits :attr:`seekRequested` on
Ctrl+click. No bioamla imports.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pyqtgraph as pg
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import QVBoxLayout, QWidget

from magpy.models import AnnotationSet
from magpy.services import Annotation

_MAX_POINTS = 50000  # cap rendered points; longer audio is min/max downsampled
_ANN_BRUSH = pg.mkBrush(255, 235, 59, 40)
_ANN_SELECTED_BRUSH = pg.mkBrush(255, 140, 0, 90)


class WaveformView(QWidget):
    seekRequested = pyqtSignal(float)  # seconds

    def __init__(self, annotations: AnnotationSet, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._model = annotations
        self._duration = 0.0
        self._regions: dict[str, pg.LinearRegionItem] = {}

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self._plot = pg.PlotWidget()
        self._plot.setLabel("bottom", "Time", units="s")
        self._plot.setLabel("left", "Amplitude")
        self._plot.setYRange(-1.1, 1.1, padding=0)
        self._curve = self._plot.plot([], [], pen=pg.mkPen("#4fc3f7", width=1))
        self._playhead = pg.InfiniteLine(angle=90, movable=False, pen=pg.mkPen("w", width=1))
        self._playhead.setVisible(False)
        self._plot.addItem(self._playhead)
        layout.addWidget(self._plot)

        self._plot.scene().sigMouseClicked.connect(self._on_clicked)
        self._model.added.connect(self._sync_annotations)
        self._model.removed.connect(self._sync_annotations)
        self._model.reset.connect(self._sync_annotations)
        self._model.changed.connect(self._sync_annotations)
        self._model.selectionChanged.connect(self._restyle)

    # --- waveform ---------------------------------------------------------
    def set_audio(self, samples: np.ndarray, sample_rate: int) -> None:
        data = self._to_mono(samples)
        self._duration = len(data) / sample_rate if sample_rate else 0.0
        times, values = self._downsample(data, sample_rate)
        self._curve.setData(times, values)
        self._plot.setLimits(xMin=0, xMax=self._duration, yMin=-1.5, yMax=1.5)
        self._plot.setXRange(0, self._duration, padding=0)
        self._playhead.setPos(0.0)
        self._playhead.setVisible(True)

    def clear(self) -> None:
        self._curve.setData([], [])
        self._playhead.setVisible(False)

    def set_playhead(self, seconds: float) -> None:
        self._playhead.setPos(seconds)

    @staticmethod
    def _to_mono(samples: np.ndarray) -> np.ndarray:
        data = np.asarray(samples, dtype=float)
        if data.ndim > 1:  # collapse the (smaller) channel axis to mono
            data = data.mean(axis=int(np.argmin(data.shape)))
        return data

    @staticmethod
    def _downsample(data: np.ndarray, sample_rate: int) -> tuple[np.ndarray, np.ndarray]:
        """Min/max envelope downsampling so long files render quickly and faithfully."""
        if not sample_rate or len(data) == 0:
            return np.array([]), np.array([])
        if len(data) <= _MAX_POINTS:
            return np.arange(len(data)) / sample_rate, data
        factor = len(data) // (_MAX_POINTS // 2)
        n_chunks = len(data) // factor
        chunks = data[: n_chunks * factor].reshape(-1, factor)
        envelope = np.empty(n_chunks * 2, dtype=data.dtype)
        envelope[0::2] = chunks.min(axis=1)
        envelope[1::2] = chunks.max(axis=1)
        chunk_dur = factor / sample_rate
        times = np.repeat(np.arange(n_chunks) * chunk_dur, 2)
        times[1::2] += chunk_dur / 2
        return times, envelope

    # --- interaction ------------------------------------------------------
    def _on_clicked(self, event) -> None:
        if self._duration <= 0 or event.button() != Qt.MouseButton.LeftButton:
            return
        if event.modifiers() != Qt.KeyboardModifier.ControlModifier:
            return
        x = self._plot.plotItem.vb.mapSceneToView(event.scenePos()).x()
        if 0 <= x <= self._duration:
            self.seekRequested.emit(float(x))

    # --- annotation overlays ---------------------------------------------
    def _sync_annotations(self, *_: object) -> None:
        for region in self._regions.values():
            self._plot.removeItem(region)
        self._regions.clear()
        for ann in self._model.items():
            region = pg.LinearRegionItem(
                values=(ann.start_time, ann.end_time), movable=False, brush=_ANN_BRUSH
            )
            self._plot.addItem(region)
            self._regions[ann.id] = region
        self._restyle(self._model.selected)

    def _restyle(self, selected: Optional[Annotation]) -> None:
        sel_id = selected.id if selected is not None else None
        for ann_id, region in self._regions.items():
            region.setBrush(_ANN_SELECTED_BRUSH if ann_id == sel_id else _ANN_BRUSH)
            region.update()
