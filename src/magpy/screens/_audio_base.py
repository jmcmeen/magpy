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
toolbar (link / import / open), and handles loading audio, playback (whole file,
a range, looped, at a chosen speed), and rendering the spectrogram window the
view asks for.

Subclasses extend via four hooks -- ``_create_docks`` (add screen-specific docks),
``_populate_toolbar`` (append toolbar actions), ``_wire_extra`` (connect
screen-specific signals), and ``_on_audio_reset`` (clear screen-specific state
when the audio changes). Two seams cross back to the shell, which owns the real
status bar and navigation: ``statusMessage(str)`` and ``navigateRequested()``.
"""

from __future__ import annotations

from pathlib import Path

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QAction
from PyQt6.QtWidgets import (
    QDockWidget,
    QFileDialog,
    QMainWindow,
    QMenu,
    QMessageBox,
    QToolBar,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from magpy.models import Document, PlaybackController, Workspace
from magpy.services import (
    KIND_AUDIO_FILE,
    KIND_FOLDER,
    PlaybackState,
    load_audio,
    render_spectrogram,
)
from magpy.settings import app_settings
from magpy.widgets import PropertiesPanel, SpectrogramView, TransportBar, WaveformView

_SETTINGS_VIEW = "spectrogram"  # settings group shared by both audio screens
_WAVEFORM_HEIGHT = 110  # px; the spectrogram takes everything else

_AUDIO_FILTER = "Audio files (*.wav *.flac *.ogg *.mp3 *.m4a);;All files (*)"

_DOCK_FEATURES = (
    QDockWidget.DockWidgetFeature.DockWidgetMovable
    | QDockWidget.DockWidgetFeature.DockWidgetFloatable
)


class BaseAudioScreen(QMainWindow):
    """Audio visualization + playback shared by the annotation and indices screens."""

    statusMessage = pyqtSignal(str)
    navigateRequested = pyqtSignal()

    def __init__(self, workspace: Workspace, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._workspace = workspace
        self._document = Document(self)
        self._playback = PlaybackController(self)
        self._annotations = self._document.annotations
        self._current_audio_path: Path | None = None
        self._suppress_autosave = False

        self._build_ui()
        self._restore_view_settings()
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
        # The waveform is a slim strip sharing the spectrogram's time axis; the
        # spectrogram is the working surface and gets the rest of the height.
        self._waveform.setFixedHeight(_WAVEFORM_HEIGHT)
        self._waveform.link_time_axis(self._spectrogram.plot_item)
        layout.addWidget(self._create_toolbar())
        layout.addWidget(self._waveform)
        layout.addWidget(self._spectrogram, stretch=1)
        layout.addWidget(self._transport)
        self.setCentralWidget(central)

        # Properties on the right is common to both screens.
        self._properties = PropertiesPanel()
        self._properties_dock = QDockWidget("Properties", self)
        self._properties_dock.setObjectName("PropertiesDock")
        self._properties_dock.setWidget(self._properties)
        self._properties_dock.setFeatures(_DOCK_FEATURES)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self._properties_dock)

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
        self._add_menu_button(
            toolbar,
            "Add",
            (
                ("Audio file…", self._add_audio_dialog, "Ctrl+O"),
                ("Folder of audio…", self._add_folder_dialog, None),
                ("Import a copy into the workspace…", self._import_artifact_dialog, None),
            ),
        )
        self._populate_toolbar(toolbar, action)
        return toolbar

    def _add_menu_button(self, toolbar: QToolBar, text: str, entries: tuple) -> list[QAction]:
        """Add a drop-down button holding ``(text, slot, shortcut)`` entries.

        Grouping related commands keeps the toolbar short enough to fit beside
        the docks on a laptop screen. The actions are also registered on the
        screen so their shortcuts work while the menu is closed.
        """
        button = QToolButton()
        button.setText(f"{text} ▾")
        button.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        button.setStyleSheet("QToolButton::menu-indicator { image: none; }")
        menu = QMenu(button)
        actions = []
        for entry_text, slot, shortcut in entries:
            act = QAction(entry_text, self)
            if shortcut:
                act.setShortcut(shortcut)
            act.triggered.connect(slot)
            menu.addAction(act)
            self.addAction(act)
            actions.append(act)
        button.setMenu(menu)
        toolbar.addWidget(button)
        return actions

    # --- remembered view settings -----------------------------------------
    def _restore_view_settings(self) -> None:
        """Reapply the spectrogram look (colormap, FFT, levels) from last time."""
        settings = app_settings()
        settings.beginGroup(_SETTINGS_VIEW)
        self._spectrogram.apply_view_settings({k: settings.value(k) for k in settings.childKeys()})
        settings.endGroup()

    def _save_view_settings(self) -> None:
        settings = app_settings()
        settings.beginGroup(_SETTINGS_VIEW)
        for key, value in self._spectrogram.view_settings().items():
            settings.setValue(key, value)
        settings.endGroup()

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
        self._transport.playSelectionRequested.connect(self._play_selection)
        self._transport.speedChanged.connect(self._playback.set_speed)
        self._transport.loopToggled.connect(self._playback.set_loop)
        self._waveform.seekRequested.connect(self._playback.seek)
        self._spectrogram.seekRequested.connect(self._playback.seek)
        self._spectrogram.playRangeRequested.connect(self._playback.play_range)
        self._spectrogram.renderRequested.connect(self._render_spectrogram)
        self._spectrogram.viewSettingsChanged.connect(self._save_view_settings)
        self._annotations.selectionChanged.connect(self._properties.set_selection)
        self._annotations.selectionChanged.connect(
            lambda ann: self._transport.set_selection_available(ann is not None)
        )
        # Transport keys work anywhere on the screen (text fields keep their own).
        for shortcut, slot in (
            ("Space", self._playback.toggle),
            ("Shift+Space", self._play_selection),
        ):
            act = QAction(self)
            act.setShortcut(shortcut)
            act.triggered.connect(slot)
            self.addAction(act)
        # Reflect in-place edits (e.g. a label change) when the selected one changes.
        self._annotations.changed.connect(
            lambda ann: (
                self._properties.set_selection(ann) if ann is self._annotations.selected else None
            )
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
        if (
            QMessageBox.question(
                self,
                "Import a copy",
                "Import copies the file into the workspace's bundle.\n\nLink instead to "
                "reference it in place without copying. Continue with import?",
            )
            != QMessageBox.StandardButton.Yes
        ):
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

    def _on_audio_changed(self, audio: object | None) -> None:
        self._on_audio_reset()  # let the subclass invalidate its own layers
        if audio is None:
            self._spectrogram.set_image(None)
            self._waveform.clear()
            return
        self._waveform.set_audio(audio.samples, audio.sample_rate)
        # Announcing the extent makes the view ask for its first window, which
        # _render_spectrogram answers.
        self._spectrogram.set_extent(audio.duration, audio.sample_rate / 2)
        self.statusMessage.emit(
            f"{audio.path.name}  ·  {audio.duration:.1f}s  ·  {audio.sample_rate} Hz"
        )

    def _render_spectrogram(self, t0: float, t1: float, max_cols: int) -> None:
        """Render the window the spectrogram view asked for.

        A screen-width STFT is a few tens of ms whatever the recording length
        (the column budget bounds it), so this runs synchronously.
        """
        audio = self._document.audio
        if audio is None:
            return
        image = render_spectrogram(
            audio.samples,
            audio.sample_rate,
            t0,
            t1,
            self._spectrogram.params(),
            max_cols=max_cols,
        )
        self._spectrogram.set_image(image)

    def _play_selection(self) -> None:
        """Play the selected annotation, or the visible span if none is selected."""
        if self._document.audio is None:
            return
        selected = self._annotations.selected
        if selected is not None:
            self._playback.play_range(selected.start_time, selected.end_time)
        else:
            self._playback.play_range(*self._spectrogram.visible_range())
