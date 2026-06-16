"""
BaseAudioScreen -- the shared audio-visualization core for screens that show and
play a sound file.

Both the annotation screen and the acoustic-indices screen present the same audio
view (waveform + spectrogram + transport + file properties) and let you load and
play a file; they differ only in the side panels they add. This base captures the
shared half so it stays DRY: it is a ``QMainWindow`` (so subclasses can host their
own docks), owns its *own* :class:`Document` and :class:`PlaybackController` (the
screens are independent -- they do **not** share playback state), builds the
central waveform/spectrogram/transport stack, the Properties dock, and the source
toolbar (link / import / open), and handles loading audio and computing the
display spectrogram.

Subclasses extend via four hooks -- ``_create_docks`` (add screen-specific docks),
``_populate_toolbar`` (append toolbar actions), ``_wire_extra`` (connect
screen-specific signals), and ``_on_audio_reset`` (clear screen-specific state
when the audio changes). Two seams cross back to the shell, which owns the real
status bar and navigation: ``statusMessage(str)`` and ``navigateRequested()``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from PyQt6.QtCore import Qt, pyqtSignal
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

from magpy.models import Document, PlaybackController, Workspace
from magpy.services import (
    KIND_AUDIO_FILE,
    KIND_FOLDER,
    PlaybackState,
    compute_spectrogram,
    load_audio,
)
from magpy.widgets import PropertiesPanel, SpectrogramView, TransportBar, WaveformView

_AUDIO_FILTER = "Audio files (*.wav *.flac *.ogg *.mp3 *.m4a);;All files (*)"

_DOCK_FEATURES = (
    QDockWidget.DockWidgetFeature.DockWidgetMovable
    | QDockWidget.DockWidgetFeature.DockWidgetFloatable
)


class BaseAudioScreen(QMainWindow):
    """Audio visualization + playback shared by the annotation and indices screens."""

    statusMessage = pyqtSignal(str)
    navigateRequested = pyqtSignal()

    def __init__(self, workspace: Workspace, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._workspace = workspace
        self._document = Document(self)
        self._playback = PlaybackController(self)
        self._annotations = self._document.annotations
        self._current_audio_path: Optional[Path] = None
        self._suppress_autosave = False

        self._build_ui()
        self._wire_base()
        self._wire_extra()
        # A workspace switch invalidates the open document; reset our own state.
        self._workspace.opened.connect(self._on_workspace_changed)

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

        # Properties on the right is common to both screens.
        self._properties = PropertiesPanel()
        prop_dock = QDockWidget("Properties", self)
        prop_dock.setObjectName("PropertiesDock")
        prop_dock.setWidget(self._properties)
        prop_dock.setFeatures(_DOCK_FEATURES)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, prop_dock)

        self._create_docks()  # subclass adds its own docks

    def _create_toolbar(self) -> QToolBar:
        """Source actions live here; subclasses append their own via the hook."""
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
        self._populate_toolbar(toolbar, action)
        return toolbar

    # --- subclass hooks (default no-ops) ----------------------------------
    def _create_docks(self) -> None:
        """Override to add screen-specific docks (annotations, indices, ...)."""

    def _populate_toolbar(self, toolbar: QToolBar, action) -> None:
        """Override to append screen-specific toolbar actions.

        ``action(text, slot, shortcut=None)`` adds and returns a ``QAction``.
        """

    def _wire_extra(self) -> None:
        """Override to connect screen-specific signals (called after the UI exists)."""

    def _on_audio_reset(self) -> None:
        """Override to clear screen-specific state when the audio changes."""

    # --- base wiring ------------------------------------------------------
    def _wire_base(self) -> None:
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
        self._on_audio_reset()  # let the subclass invalidate its own layers
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
