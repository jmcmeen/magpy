"""
MainWindow -- the application shell.

Reproduces the legacy navigation layout: a VS Code-style activity bar on the
left switching a ``QStackedWidget`` of full-window screens, a dark theme, and an
AUDIO view with its own docks that appear only on that view. The screens are the
ported (mostly placeholder) views; the AUDIO view hosts the rebuilt
spectrogram/transport/annotation work and is wired through the services seam.

MagPy is **workspace-always**: the shell always has a :class:`Workspace` open --
the last-used bundle, or an auto-created "Untitled" workspace -- so there is no
"nothing open" state. Audio is *linked* into a workspace by reference (never
copied) and the annotations the user makes are persisted into the bundle. The
File menu drives managing workspaces; "Import" is the explicit copy-in escape
hatch.

For now this shell also plays the AUDIO view-model role (owns the Document and
PlaybackController, computes the spectrogram on audio change, mediates
annotations). That moves into a dedicated view-model as the rebuild proceeds.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from PyQt6.QtCore import QSettings, QStandardPaths, Qt, QThreadPool
from PyQt6.QtGui import QAction
from PyQt6.QtWidgets import (
    QDockWidget,
    QFileDialog,
    QHBoxLayout,
    QMainWindow,
    QMessageBox,
    QStackedWidget,
    QToolBar,
    QVBoxLayout,
    QWidget,
)

from magpy.models import CandidateSet, Document, PlaybackController, Workspace
from magpy.screens import (
    EBIRD_CONFIG,
    INATURALIST_CONFIG,
    MACAULAY_CONFIG,
    XENO_CANTO_CONFIG,
    BaseScreen,
    BatchScreen,
    CatalogScreen,
    DatasetsScreen,
    ExploreScreen,
    HomeScreen,
    HuggingFaceScreen,
    SettingsScreen,
    TrainingScreen,
)
from magpy.services import (
    BUNDLE_SUFFIX,
    KIND_AUDIO_FILE,
    KIND_FOLDER,
    Annotation,
    PlaybackState,
    candidate_to_annotation,
    compute_indices,
    compute_spectrogram,
    default_env_path,
    detector_label,
    is_bundle,
    load_annotations,
    load_audio,
    load_into_environ,
    run_detection,
    save_annotations,
)
from magpy.theme import DARK_STYLESHEET
from magpy.widgets import (
    AnnotationTable,
    DetectPanel,
    IndicesPanel,
    NavigationBar,
    PropertiesPanel,
    SpectrogramView,
    TransportBar,
    ViewType,
    WaveformView,
    WorkspacePanel,
)
from magpy.workers import Worker

_ANNOTATION_FILTER = "Selection tables (*.txt *.csv);;Raven table (*.txt);;CSV (*.csv)"
_AUDIO_FILTER = "Audio files (*.wav *.flac *.ogg *.mp3 *.m4a);;All files (*)"
_MAX_RECENT = 10

_VIEW_NAMES = {
    ViewType.HOME: "Home",
    ViewType.AUDIO: "Audio",
    ViewType.DATASETS: "Datasets",
    ViewType.TRAINING: "Training",
    ViewType.EXPLORE: "Explore",
    ViewType.BATCH: "Batch",
    ViewType.INATURALIST: "iNaturalist",
    ViewType.EBIRD: "eBird",
    ViewType.MACAULAY: "Macaulay Library",
    ViewType.XENOCANTO: "Xeno-Canto",
    ViewType.HUGGINGFACE: "Hugging Face",
    ViewType.SETTINGS: "Settings",
}


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setMinimumSize(1100, 720)
        self.resize(1680, 1000)  # generous default; app opens maximized (see app.main)
        self.setStyleSheet(DARK_STYLESHEET)

        self._settings = QSettings("MagPy", "MagPy")
        # Load MagPy's .env into os.environ before any catalog call so saved API
        # keys take effect this session (bioamla reads them lazily).
        config_base = QStandardPaths.writableLocation(
            QStandardPaths.StandardLocation.AppConfigLocation
        )
        self._env_path = default_env_path(config_base or Path.home() / ".magpy")
        load_into_environ(self._env_path)
        self._workspace = Workspace(self)
        self._document = Document(self)
        self._playback = PlaybackController(self)
        self._annotations = self._document.annotations
        self._candidates = CandidateSet(self)
        self._detect_worker: Optional[Worker] = None
        self._indices_worker: Optional[Worker] = None
        self._audio_docks: list[QDockWidget] = []
        self._current_view = ViewType.HOME
        self._current_audio_path: Optional[Path] = None
        self._suppress_autosave = False

        self._setup_ui()
        self._install_shortcuts()
        self._wire()

        self._bootstrap_workspace()
        self._navigate_to(ViewType.HOME)

    # --- layout -----------------------------------------------------------
    def _setup_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        self._nav_bar = NavigationBar()
        self._nav_bar.view_changed.connect(self._on_view_changed)
        main_layout.addWidget(self._nav_bar)

        self._view_stack = QStackedWidget()
        main_layout.addWidget(self._view_stack, stretch=1)

        self._create_audio_view()
        self._create_other_views()
        self._create_workspace_dock()
        self._create_audio_docks()

    def _create_workspace_dock(self) -> None:
        # Always visible: MagPy is workspace-always, so there is always one open.
        # On the right for now (placement still being figured out).
        self._workspace_panel = WorkspacePanel(self._workspace)
        self._workspace_dock = QDockWidget("Workspace", self)
        self._workspace_dock.setObjectName("WorkspaceDock")
        self._workspace_dock.setWidget(self._workspace_panel)
        self._workspace_dock.setFeatures(
            QDockWidget.DockWidgetFeature.DockWidgetMovable
            | QDockWidget.DockWidgetFeature.DockWidgetFloatable
        )
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self._workspace_dock)

    def _create_audio_view(self) -> None:
        audio_view = QWidget()
        layout = QVBoxLayout(audio_view)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)
        self._waveform = WaveformView(self._annotations)
        self._spectrogram = SpectrogramView(self._annotations)
        self._transport = TransportBar()
        layout.addWidget(self._create_audio_toolbar())
        layout.addWidget(self._waveform, stretch=1)
        layout.addWidget(self._spectrogram, stretch=2)
        layout.addWidget(self._transport)
        self._audio_view = audio_view
        self._view_stack.addWidget(audio_view)

    def _create_audio_toolbar(self) -> QToolBar:
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

    def _create_other_views(self) -> None:
        self._home_screen = HomeScreen()
        self._screens = {
            ViewType.HOME: self._home_screen,
            ViewType.DATASETS: DatasetsScreen(self._workspace),
            ViewType.TRAINING: TrainingScreen(self._workspace),
            ViewType.EXPLORE: ExploreScreen(self._workspace),
            ViewType.BATCH: BatchScreen(self._workspace),
            ViewType.INATURALIST: CatalogScreen(INATURALIST_CONFIG, self._workspace),
            ViewType.EBIRD: CatalogScreen(EBIRD_CONFIG, self._workspace),
            ViewType.MACAULAY: CatalogScreen(MACAULAY_CONFIG, self._workspace),
            ViewType.XENOCANTO: CatalogScreen(XENO_CANTO_CONFIG, self._workspace),
            ViewType.HUGGINGFACE: HuggingFaceScreen(self._workspace),
            ViewType.SETTINGS: SettingsScreen(self._env_path),
        }
        self._view_widgets = {ViewType.AUDIO: self._audio_view}
        for view_type, screen in self._screens.items():
            self._view_stack.addWidget(screen)
            self._view_widgets[view_type] = screen

    def _create_audio_docks(self) -> None:
        feats = (
            QDockWidget.DockWidgetFeature.DockWidgetMovable
            | QDockWidget.DockWidgetFeature.DockWidgetFloatable
        )
        self._table = AnnotationTable(self._annotations)
        ann_dock = QDockWidget("Annotations", self)
        ann_dock.setObjectName("AnnotationsDock")
        ann_dock.setWidget(self._table)
        ann_dock.setFeatures(feats)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, ann_dock)

        self._properties = PropertiesPanel()
        prop_dock = QDockWidget("Properties", self)
        prop_dock.setObjectName("PropertiesDock")
        prop_dock.setWidget(self._properties)
        prop_dock.setFeatures(feats)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, prop_dock)

        self._detect_panel = DetectPanel(self._candidates)
        detect_dock = QDockWidget("Detect", self)
        detect_dock.setObjectName("DetectDock")
        detect_dock.setWidget(self._detect_panel)
        detect_dock.setFeatures(feats)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, detect_dock)

        self._indices_panel = IndicesPanel()
        indices_dock = QDockWidget("Indices", self)
        indices_dock.setObjectName("IndicesDock")
        indices_dock.setWidget(self._indices_panel)
        indices_dock.setFeatures(feats)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, indices_dock)

        self.tabifyDockWidget(prop_dock, ann_dock)
        self.tabifyDockWidget(ann_dock, detect_dock)
        self.tabifyDockWidget(detect_dock, indices_dock)
        prop_dock.raise_()  # show Properties by default
        self._audio_docks += [ann_dock, prop_dock, detect_dock, indices_dock]

    def _install_shortcuts(self) -> None:
        # No menu bar: navigation is driven by the left nav, and workspace
        # management lives on the Home screen. Keep keyboard shortcuts as
        # window-level actions for power users.
        for text, shortcut, slot in (
            ("New Workspace", "Ctrl+Shift+N", self._new_workspace_dialog),
            ("Open Workspace", "Ctrl+Shift+O", self._open_workspace_dialog),
        ):
            act = QAction(text, self)
            act.setShortcut(shortcut)
            act.triggered.connect(slot)
            self.addAction(act)

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

        # Workspace: panel activates a file -> load it (it is already an artifact).
        self._workspace_panel.fileActivated.connect(self.load_file)
        self._workspace.opened.connect(self._on_workspace_opened)

        self._wire_detect()
        self._wire_indices()

        # Home dashboard cards navigate to the matching views / manage workspaces.
        home = self._home_screen
        home.navigate_requested.connect(self._navigate_to)
        home.new_workspace_clicked.connect(self._new_workspace_dialog)
        home.open_workspace_clicked.connect(self._open_workspace_dialog)
        if hasattr(home, "recent_workspace_clicked"):
            home.recent_workspace_clicked.connect(self._open_workspace)

    # --- view switching ---------------------------------------------------
    def _navigate_to(self, view_type: ViewType) -> None:
        """Switch views and keep the nav bar's selection in sync."""
        self._nav_bar.set_current_view(view_type)
        self._on_view_changed(view_type)

    def _on_view_changed(self, view_type: ViewType) -> None:
        self._current_view = view_type
        widget = self._view_widgets.get(view_type)
        if widget is not None:
            self._view_stack.setCurrentWidget(widget)
            # Drive the screen lifecycle so screens can populate lazily on first
            # show. The AUDIO view is a bare QWidget (not a BaseScreen), so guard.
            if isinstance(widget, BaseScreen):
                widget.activate()
        is_audio = view_type == ViewType.AUDIO
        for dock in self._audio_docks:
            dock.setVisible(is_audio)
        self._update_title()

    def _update_title(self) -> None:
        view = _VIEW_NAMES.get(self._current_view, "Home")
        ws = self._workspace.name or "—"
        self.setWindowTitle(f"MagPy — {ws} — {view}")

    # --- workspace lifecycle ---------------------------------------------
    def _bootstrap_workspace(self) -> None:
        """Open the last workspace, else create/open the Untitled workspace."""
        for path in self._recent_workspaces():
            if is_bundle(path):
                self._workspace.open(path)
                return
        scratch = self._scratch_dir()
        if is_bundle(scratch):
            self._workspace.open(scratch)
        else:
            self._workspace.create(scratch, "Untitled")

    def _scratch_dir(self) -> Path:
        base = QStandardPaths.writableLocation(QStandardPaths.StandardLocation.AppDataLocation)
        return Path(base or Path.home() / ".magpy") / f"Untitled{BUNDLE_SUFFIX}"

    def _new_workspace_dialog(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self, "New workspace", "", f"MagPy workspace (*{BUNDLE_SUFFIX})"
        )
        if not path:
            return
        if not path.endswith(BUNDLE_SUFFIX):
            path += BUNDLE_SUFFIX
        try:
            self._workspace.create(path, Path(path).stem)
        except Exception as exc:  # noqa: BLE001 - surface to the user
            QMessageBox.critical(self, "Could not create workspace", str(exc))

    def _open_workspace_dialog(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Open workspace bundle", "")
        if path:
            self._open_workspace(path)

    def _open_workspace(self, path: str) -> None:
        if not is_bundle(path):
            QMessageBox.warning(
                self, "Not a workspace",
                f"{path}\n\nis not a MagPy workspace ({BUNDLE_SUFFIX} bundle).",
            )
            return
        self._workspace.open(path)

    def _on_workspace_opened(self, bundle_dir: object) -> None:
        self._remember_workspace(str(bundle_dir))
        # A workspace switch invalidates the open document.
        self._current_audio_path = None
        self._suppress_autosave = True
        self._document.set_audio(None)
        self._suppress_autosave = False
        self._update_title()
        self.statusBar().showMessage(
            f"Workspace: {self._workspace.name}  ·  {len(self._workspace.audio_files)} audio files"
        )

    # --- recent workspaces (QSettings) -----------------------------------
    def _recent_workspaces(self) -> list[str]:
        return list(self._settings.value("recentWorkspaces", []) or [])

    def _remember_workspace(self, path: str) -> None:
        recent = self._recent_workspaces()
        if path in recent:
            recent.remove(path)
        recent.insert(0, path)
        self._settings.setValue("recentWorkspaces", recent[:_MAX_RECENT])
        if hasattr(self._home_screen, "set_recent_workspaces"):
            self._home_screen.set_recent_workspaces(recent[:_MAX_RECENT])

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
            self.statusBar().showMessage(
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
            self.statusBar().showMessage("Failed to open file")
            return
        # Suspend autosave: set_audio clears annotations, and we then repopulate
        # from the workspace -- neither should write back over the sidecar.
        self._suppress_autosave = True
        self._document.set_audio(audio)
        self._current_audio_path = path
        self._annotations.set_all(self._workspace.load_annotations_for(path))
        self._suppress_autosave = False
        self._navigate_to(ViewType.AUDIO)

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
        self.statusBar().showMessage(
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
            self.statusBar().showMessage("Open an audio file before running detection")
            return
        if self._detect_worker is not None:
            return  # a run is already in flight
        self._detect_panel.set_running(True)
        self.statusBar().showMessage("Detecting…")
        worker = Worker(run_detection, audio.samples, audio.sample_rate, kind, params)
        self._detect_worker = worker
        label = detector_label(kind)
        worker.signals.result.connect(lambda cands: self._candidates.set_all(cands, label))
        worker.signals.result.connect(
            lambda cands: self.statusBar().showMessage(f"{len(cands)} detections ({label})")
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
        self.statusBar().showMessage(f"Promoted {len(candidates)} to annotations")

    # --- acoustic indices (whole-file summary) ---------------------------
    def _wire_indices(self) -> None:
        self._indices_panel.computeRequested.connect(self._compute_indices)

    def _compute_indices(self) -> None:
        audio = self._document.audio
        if audio is None:
            self.statusBar().showMessage("Open an audio file before computing indices")
            return
        if self._indices_worker is not None:
            return
        self._indices_panel.set_running(True)
        self.statusBar().showMessage("Computing acoustic indices…")
        worker = Worker(compute_indices, audio.samples, audio.sample_rate)
        self._indices_worker = worker
        worker.signals.result.connect(self._indices_panel.set_summary)
        worker.signals.result.connect(
            lambda _s: self.statusBar().showMessage("Acoustic indices computed")
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
            self.statusBar().showMessage("No selection — press S (time) or B (box) to start one")
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
        self.statusBar().showMessage(f"Imported {len(anns)} annotations")

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
        self.statusBar().showMessage(f"Exported to {Path(path).name}")
