"""
IndicesScreen -- acoustic-indices analysis of a sound file.

Shares the audio-visualization core (:class:`BaseAudioScreen`): the same
spectrogram/waveform/transport view and file properties, so you can load and play
a file here just like on the annotation screen. The screen-specific addition is
the Indices panel -- whole-file acoustic-index summaries (computed off-thread).

This is a separate nav destination from the annotation screen and keeps its own
:class:`Document`/:class:`PlaybackController`; the two screens do not share
playback state. Indices are a whole-file summary, so there is no annotation
editing here.
"""

from __future__ import annotations

from typing import Optional

from PyQt6.QtCore import Qt, QThreadPool
from PyQt6.QtWidgets import QDockWidget, QMessageBox

from magpy.services import compute_indices
from magpy.widgets import IndicesPanel
from magpy.workers import Worker

from ._audio_base import _DOCK_FEATURES, BaseAudioScreen


class IndicesScreen(BaseAudioScreen):
    """Audio view + acoustic-indices (whole-file summary) panel."""

    def _create_docks(self) -> None:
        self._indices_worker: Optional[Worker] = None
        self._indices_panel = IndicesPanel()
        dock = QDockWidget("Indices", self)
        dock.setObjectName("IndicesDock")
        dock.setWidget(self._indices_panel)
        dock.setFeatures(_DOCK_FEATURES)
        self.addDockWidget(Qt.DockWidgetArea.BottomDockWidgetArea, dock)

    def _wire_extra(self) -> None:
        self._indices_panel.computeRequested.connect(self._compute_indices)

    def _on_audio_reset(self) -> None:
        self._indices_panel.set_summary(None)

    # --- acoustic indices (whole-file summary) ---------------------------
    def _compute_indices(self) -> None:
        audio = self._document.audio
        if audio is None:
            self.statusMessage.emit("Open an audio file before computing indices")
            return
        if self._indices_worker is not None:
            return
        self._indices_panel.set_running(True)
        self.statusMessage.emit("Computing acoustic indices…")
        worker = Worker(compute_indices, audio.samples, audio.sample_rate)
        self._indices_worker = worker
        worker.signals.result.connect(self._indices_panel.set_summary)
        worker.signals.result.connect(
            lambda _s: self.statusMessage.emit("Acoustic indices computed")
        )
        worker.signals.error.connect(
            lambda exc: QMessageBox.critical(self, "Indices failed", str(exc))
        )
        worker.signals.finished.connect(self._on_indices_finished)
        QThreadPool.globalInstance().start(worker)

    def _on_indices_finished(self) -> None:
        self._indices_panel.set_running(False)
        self._indices_worker = None
