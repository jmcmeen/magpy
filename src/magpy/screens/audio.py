"""
AudioScreen -- the audio-analysis workspace.

The rebuilt spectrogram/transport/annotation view, extracted from the app shell
into its own screen. It is a ``QMainWindow`` (not a plain :class:`BaseScreen`) so
it can host its own dockable panels -- Properties on the right, and Annotations /
Detect / Indices side-by-side across the bottom for the annotate -> detect ->
indices workflow. It drops into the shell's ``QStackedWidget`` like any other
screen, and its docks show/hide with the page automatically (they are children of
this window, so there is no manual visibility toggle).

This screen plays the audio view-model role: it owns the :class:`Document`, the
:class:`PlaybackController`, and the reviewable :class:`CandidateSet`; computes
the spectrogram on audio change; runs detection / acoustic-indices off-thread;
and persists annotations into the open workspace bundle. It reaches the workspace
through the shared :class:`Workspace` model. Two seams cross back to the shell,
which owns the real status bar and the navigation:

* ``statusMessage(str)`` -- progress/notice text for the shell's status bar, and
* ``navigateRequested()`` -- ask the shell to bring this screen to front (emitted
  when a file is opened).
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from PyQt6.QtCore import Qt, QThreadPool, pyqtSignal
from PyQt6.QtGui import QAction
from PyQt6.QtWidgets import (
    QDockWidget,
    QFileDialog,
    QMainWindow,
    QMessageBox,
    QToolBar,
    QVBoxLayout,
    QWidget,
)

from magpy.models import CandidateSet, Document, PlaybackController, Workspace
from magpy.services import (
    KIND_AUDIO_FILE,
    KIND_FOLDER,
    Annotation,
    PlaybackState,
    candidate_to_annotation,
    compute_indices,
    compute_spectrogram,
    detector_label,
    load_annotations,
    load_audio,
    run_detection,
    save_annotations,
)
from magpy.widgets import (
    AnnotationTable,
    DetectPanel,
    IndicesPanel,
    PropertiesPanel,
    SpectrogramView,
    TransportBar,
    WaveformView,
)
from magpy.workers import Worker

_ANNOTATION_FILTER = "Selection tables (*.txt *.csv);;Raven table (*.txt);;CSV (*.csv)"
_AUDIO_FILTER = "Audio files (*.wav *.flac *.ogg *.mp3 *.m4a);;All files (*)"


class AudioScreen(QMainWindow):
    """Audio-analysis screen: spectrogram/transport/annotation + detect/indices."""

    statusMessage = pyqtSignal(str)
    navigateRequested = pyqtSignal()

    def __init__(self, workspace: Workspace, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._workspace = workspace
        self._document = Document(self)
        self._playback = PlaybackController(self)
        self._annotations = self._document.annotations
        self._candidates = CandidateSet(self)
        self._detect_worker: Optional[Worker] = None
        self._indices_worker: Optional[Worker] = None
        self._current_audio_path: Optional[Path] = None
        self._suppress_autosave = False

        self._build_ui()
        self._wire()

    # --- layout -----------------------------------------------------------
    def _build_ui(self) -> None:
        central = QWidget()
        layout = QVBoxLayout(central)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)
        self._waveform = WaveformView(self._annotations)
        self._spectrogram = SpectrogramView(self._annotations)
        self._transport = TransportBar()
        layout.addWidget(self._create_toolbar())
        layout.addWidget(self._waveform, stretch=1)
        layout.addWidget(self._spectrogram, stretch=2)
        layout.addWidget(self._transport)
        self.setCentralWidget(central)
        self._create_docks()

    def _create_toolbar(self) -> QToolBar:
        """Audio-analysis actions live here, not in the main menu bar."""
        toolbar = QToolBar("Audio")
        toolbar.setMovable(False)

        def action(text: str, slot, shortcut: str | None = None) -> QAction:
            act = QAction(text, self)
            if shortcut:
                act.setShortcut(shortcut)
            act.triggered.connect(slot)
            toolbar.addAction(act)
            return act

        # Source actions: link by reference, or import a copy into the workspace.
        action("Add Audio", self._add_audio_dialog, "Ctrl+O")
        action("Add Folder", self._add_folder_dialog)
        action("Import…", self._import_artifact_dialog)
        toolbar.addSeparator()
        action("New Selection", self._spectrogram.start_selection, "S")
        action("New Box", self._spectrogram.start_box_selection, "B")
        action("Add", self._add_annotation, "Return")
        action("Delete", self._delete_selected, "Delete")
        toolbar.addSeparator()
        action("Import Anns", self._import_annotations)
        action("Export Anns", self._export_annotations)
        return toolbar

    def _create_docks(self) -> None:
        feats = (
            QDockWidget.DockWidgetFeature.DockWidgetMovable
            | QDockWidget.DockWidgetFeature.DockWidgetFloatable
        )
        # Properties stays on the right; annotations/detect/indices dock under
        # the spectrogram (bottom area), side-by-side.
        self._properties = PropertiesPanel()
        prop_dock = QDockWidget("Properties", self)
        prop_dock.setObjectName("PropertiesDock")
        prop_dock.setWidget(self._properties)
        prop_dock.setFeatures(feats)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, prop_dock)

        self._table = AnnotationTable(self._annotations)
        ann_dock = QDockWidget("Annotations", self)
        ann_dock.setObjectName("AnnotationsDock")
        ann_dock.setWidget(self._table)
        ann_dock.setFeatures(feats)
        self.addDockWidget(Qt.DockWidgetArea.BottomDockWidgetArea, ann_dock)

        self._detect_panel = DetectPanel(self._candidates)
        detect_dock = QDockWidget("Detect", self)
        detect_dock.setObjectName("DetectDock")
        detect_dock.setWidget(self._detect_panel)
        detect_dock.setFeatures(feats)
        self.addDockWidget(Qt.DockWidgetArea.BottomDockWidgetArea, detect_dock)

        self._indices_panel = IndicesPanel()
        indices_dock = QDockWidget("Indices", self)
        indices_dock.setObjectName("IndicesDock")
        indices_dock.setWidget(self._indices_panel)
        indices_dock.setFeatures(feats)
        self.addDockWidget(Qt.DockWidgetArea.BottomDockWidgetArea, indices_dock)

        # Side-by-side across the bottom so all three are visible for the
        # annotate -> detect -> indices workflow.
        self.splitDockWidget(ann_dock, detect_dock, Qt.Orientation.Horizontal)
        self.splitDockWidget(detect_dock, indices_dock, Qt.Orientation.Horizontal)

    # --- wiring -----------------------------------------------------------
    def _wire(self) -> None:
        self._document.audioChanged.connect(self._on_audio_changed)
        self._document.audioChanged.connect(self._playback.set_audio)
        self._document.audioChanged.connect(self._properties.set_audio)
        self._playback.positionChanged.connect(self._transport.set_position)
        self._playback.positionChanged.connect(self._spectrogram.set_playhead)
        self._playback.positionChanged.connect(self._waveform.set_playhead)
        self._playback.durationChanged.connect(self._transport.set_duration)
        self._playback.stateChanged.connect(
            lambda s: self._transport.set_playing(s == PlaybackState.PLAYING)
        )
        self._transport.playPauseRequested.connect(self._playback.toggle)
        self._transport.stopRequested.connect(self._playback.stop)
        self._transport.seekRequested.connect(self._playback.seek)
        self._waveform.seekRequested.connect(self._playback.seek)
        self._annotations.selectionChanged.connect(self._properties.set_selection)
        # Reflect in-place edits (e.g. a label change) when the selected one changes.
        self._annotations.changed.connect(
            lambda ann: self._properties.set_selection(ann)
            if ann is self._annotations.selected
            else None
        )
        # Persist annotations into the workspace bundle on any change.
        for sig in (
            self._annotations.added,
            self._annotations.removed,
            self._annotations.changed,
            self._annotations.reset,
        ):
            sig.connect(self._autosave_annotations)

        # A workspace switch invalidates the open document; reset our own state.
        self._workspace.opened.connect(self._on_workspace_changed)

        self._wire_detect()
        self._wire_indices()

    def _on_workspace_changed(self, _bundle_dir: object) -> None:
        self._current_audio_path = None
        self._suppress_autosave = True
        self._document.set_audio(None)
        self._suppress_autosave = False

    # --- artifacts (link / import) ---------------------------------------
    def _add_audio_dialog(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Add audio file", "", _AUDIO_FILTER)
        if path:
            self._workspace.link(path, KIND_AUDIO_FILE)
            self.load_file(path)

    def _add_folder_dialog(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Add audio folder", "")
        if path:
            self._workspace.link(path, KIND_FOLDER)
            self.statusMessage.emit(
                f"Linked folder · {len(self._workspace.audio_files)} audio files"
            )

    def _import_artifact_dialog(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Import audio (copies)", "", _AUDIO_FILTER)
        if not path:
            return
        if QMessageBox.question(
            self, "Import a copy",
            "Import copies the file into the workspace's bundle.\n\nLink instead to "
            "reference it in place without copying. Continue with import?",
        ) != QMessageBox.StandardButton.Yes:
            return
        try:
            artifact = self._workspace.import_(path, KIND_AUDIO_FILE)
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "Could not import", str(exc))
            return
        self.load_file(self._workspace.resolved_path(artifact))

    # --- audio ------------------------------------------------------------
    def add_audio_path(self, path: str | Path) -> None:
        """Link ``path`` into the current workspace and open it (used by the CLI arg)."""
        self._workspace.link(path, KIND_AUDIO_FILE)
        self.load_file(path)

    def load_file(self, path: str | Path) -> None:
        """Load an audio file into the document + its workspace annotations."""
        path = Path(path)
        try:
            audio = load_audio(path)
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "Could not open file", str(exc))
            self.statusMessage.emit("Failed to open file")
            return
        # Suspend autosave: set_audio clears annotations, and we then repopulate
        # from the workspace -- neither should write back over the sidecar.
        self._suppress_autosave = True
        self._document.set_audio(audio)
        self._current_audio_path = path
        self._annotations.set_all(self._workspace.load_annotations_for(path))
        self._suppress_autosave = False
        self.navigateRequested.emit()

    def _on_audio_changed(self, audio: Optional[object]) -> None:
        # A new (or cleared) document invalidates the candidate + indices layers.
        self._candidates.clear()
        self._indices_panel.set_summary(None)
        if audio is None:
            self._spectrogram.set_image(None)
            self._waveform.clear()
            return
        self._waveform.set_audio(audio.samples, audio.sample_rate)
        # Spectrogram of a view window is ~50 ms; safe to compute synchronously.
        image = compute_spectrogram(audio.samples, audio.sample_rate)
        self._spectrogram.set_image(image)
        self.statusMessage.emit(
            f"{audio.path.name}  ·  {audio.duration:.1f}s  ·  {audio.sample_rate} Hz"
        )

    # --- detection (reviewable candidate layer) --------------------------
    def _wire_detect(self) -> None:
        self._detect_panel.runRequested.connect(self._run_detection)
        self._detect_panel.promoteRequested.connect(self._promote_candidates)
        self._detect_panel.candidateActivated.connect(
            lambda start, _end: self._playback.seek(start)
        )

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

    # --- acoustic indices (whole-file summary) ---------------------------
    def _wire_indices(self) -> None:
        self._indices_panel.computeRequested.connect(self._compute_indices)

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
