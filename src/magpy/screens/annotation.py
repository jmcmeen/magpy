"""
AudioAnnotationScreen -- annotate a sound file.

The audio-annotation workspace: the shared spectrogram/waveform/transport view
(from :class:`BaseAudioScreen`) plus the annotation-specific panels -- an
Annotations table and a Detect panel (the reviewable candidate layer) docked
side-by-side across the bottom for the annotate -> detect workflow. Selections
made on the spectrogram become annotations; a detector's output lands in a
:class:`CandidateSet` that the user reviews and promotes. Annotations persist
into the open workspace bundle automatically on any change.

Acoustic-indices analysis lives on its own :class:`IndicesScreen`; this screen is
purely about creating and curating annotations.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from PyQt6.QtCore import Qt, QThreadPool
from PyQt6.QtWidgets import QDockWidget, QFileDialog, QMessageBox, QToolBar

from magpy.models import CandidateSet
from magpy.services import (
    Annotation,
    candidate_to_annotation,
    detector_label,
    load_annotations,
    run_detection,
    save_annotations,
)
from magpy.widgets import AnnotationTable, DetectPanel
from magpy.workers import Worker

from ._audio_base import _DOCK_FEATURES, BaseAudioScreen

_ANNOTATION_FILTER = "Selection tables (*.txt *.csv);;Raven table (*.txt);;CSV (*.csv)"


class AudioAnnotationScreen(BaseAudioScreen):
    """Audio view + annotation table + detect (reviewable candidate layer)."""

    # --- screen-specific UI -----------------------------------------------
    def _create_docks(self) -> None:
        self._detect_worker: Optional[Worker] = None
        self._candidates = CandidateSet(self)

        # Annotations + Detect dock side-by-side under the spectrogram.
        self._table = AnnotationTable(self._annotations)
        ann_dock = QDockWidget("Annotations", self)
        ann_dock.setObjectName("AnnotationsDock")
        ann_dock.setWidget(self._table)
        ann_dock.setFeatures(_DOCK_FEATURES)
        self.addDockWidget(Qt.DockWidgetArea.BottomDockWidgetArea, ann_dock)

        self._detect_panel = DetectPanel(self._candidates)
        detect_dock = QDockWidget("Detect", self)
        detect_dock.setObjectName("DetectDock")
        detect_dock.setWidget(self._detect_panel)
        detect_dock.setFeatures(_DOCK_FEATURES)
        self.addDockWidget(Qt.DockWidgetArea.BottomDockWidgetArea, detect_dock)

        self.splitDockWidget(ann_dock, detect_dock, Qt.Orientation.Horizontal)

    def _populate_toolbar(self, toolbar: QToolBar, action) -> None:
        toolbar.addSeparator()
        action("New Selection", self._spectrogram.start_selection, "S")
        action("New Box", self._spectrogram.start_box_selection, "B")
        action("Add", self._add_annotation, "Return")
        action("Delete", self._delete_selected, "Delete")
        toolbar.addSeparator()
        action("Import Anns", self._import_annotations)
        action("Export Anns", self._export_annotations)

    def _wire_extra(self) -> None:
        # Persist annotations into the workspace bundle on any change.
        for sig in (
            self._annotations.added,
            self._annotations.removed,
            self._annotations.changed,
            self._annotations.reset,
        ):
            sig.connect(self._autosave_annotations)
        # Detection (reviewable candidate layer).
        self._detect_panel.runRequested.connect(self._run_detection)
        self._detect_panel.promoteRequested.connect(self._promote_candidates)
        self._detect_panel.candidateActivated.connect(
            lambda start, _end: self._playback.seek(start)
        )

    def _on_audio_reset(self) -> None:
        # A new (or cleared) document invalidates the candidate layer.
        self._candidates.clear()

    # --- detection --------------------------------------------------------
    def _run_detection(self, kind: str, params: dict) -> None:
        audio = self._document.audio
        if audio is None:
            self.statusMessage.emit("Open an audio file before running detection")
            return
        if self._detect_worker is not None:
            return  # a run is already in flight
        self._detect_panel.set_running(True)
        self.statusMessage.emit("Detecting…")
        worker = Worker(run_detection, audio.samples, audio.sample_rate, kind, params)
        self._detect_worker = worker
        label = detector_label(kind)
        worker.signals.result.connect(lambda cands: self._candidates.set_all(cands, label))
        worker.signals.result.connect(
            lambda cands: self.statusMessage.emit(f"{len(cands)} detections ({label})")
        )
        worker.signals.error.connect(
            lambda exc: QMessageBox.critical(self, "Detection failed", str(exc))
        )
        worker.signals.finished.connect(self._on_detection_finished)
        QThreadPool.globalInstance().start(worker)

    def _on_detection_finished(self) -> None:
        self._detect_panel.set_running(False)
        self._detect_worker = None

    def _promote_candidates(self, candidates: list) -> None:
        for candidate in candidates:
            self._annotations.add(candidate_to_annotation(candidate))
        self.statusMessage.emit(f"Promoted {len(candidates)} to annotations")

    # --- annotations ------------------------------------------------------
    def _autosave_annotations(self, *_args) -> None:
        if self._suppress_autosave or self._current_audio_path is None:
            return
        self._workspace.save_annotations_for(self._current_audio_path, self._annotations.items())

    def _add_annotation(self) -> None:
        bounds = self._spectrogram.selection_bounds()
        if bounds is None:
            self.statusMessage.emit("No selection — press S (time) or B (box) to start one")
            return
        start, end, low, high = bounds
        self._annotations.add(
            Annotation(start_time=start, end_time=end, low_freq=low, high_freq=high)
        )
        self._spectrogram.clear_selection()

    def _delete_selected(self) -> None:
        selected = self._annotations.selected
        if selected is not None:
            self._annotations.remove(selected)

    def _import_annotations(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Import selection table", "", _ANNOTATION_FILTER
        )
        if not path:
            return
        try:
            anns = load_annotations(path)
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "Could not import", str(exc))
            return
        self._annotations.set_all(anns)  # autosave persists these into the bundle
        self.statusMessage.emit(f"Imported {len(anns)} annotations")

    def _export_annotations(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self, "Export selection table", "", _ANNOTATION_FILTER
        )
        if not path:
            return
        try:
            save_annotations(self._annotations.items(), path)
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "Could not export", str(exc))
            return
        self.statusMessage.emit(f"Exported to {Path(path).name}")
