"""
MainWindow -- the application shell.

Reproduces the legacy navigation layout: a VS Code-style activity bar on the
left switching a ``QStackedWidget`` of full-window screens, a dark theme, and an
always-visible Workspace dock. The screens are self-contained views; the audio
spectrogram/annotation work lives in :class:`AudioAnnotationScreen` and the
acoustic-indices analysis in :class:`IndicesScreen` (each its own ``QMainWindow``
page with its own docks, sharing the :class:`BaseAudioScreen` core), reached
through the services seam.

MagPy is **workspace-always**: the shell always has a :class:`Workspace` open --
the last-used bundle, or an auto-created "Untitled" workspace -- so there is no
"nothing open" state. Audio is *linked* into a workspace by reference (never
copied) and the annotations the user makes are persisted into the bundle. The
Home screen / shortcuts drive managing workspaces; "Import" (on the audio screen)
is the explicit copy-in escape hatch.

The shell owns the status bar and navigation; the audio screen reports progress
via ``statusMessage`` and asks to be brought to front via ``navigateRequested``.
"""

from __future__ import annotations

from pathlib import Path

from PyQt6.QtCore import QSettings, QStandardPaths, Qt
from PyQt6.QtGui import QAction
from PyQt6.QtWidgets import (
    QDockWidget,
    QFileDialog,
    QHBoxLayout,
    QMainWindow,
    QMessageBox,
    QStackedWidget,
    QWidget,
)

from magpy.models import Workspace
from magpy.screens import (
    EBIRD_CONFIG,
    INATURALIST_CONFIG,
    MACAULAY_CONFIG,
    XENO_CANTO_CONFIG,
    AudioAnnotationScreen,
    BaseScreen,
    BatchScreen,
    CatalogScreen,
    DatasetsScreen,
    ExploreScreen,
    HomeScreen,
    HuggingFaceScreen,
    IndicesScreen,
    SettingsScreen,
    TrainingScreen,
)
from magpy.services import (
    BUNDLE_SUFFIX,
    default_env_path,
    is_bundle,
    load_into_environ,
)
from magpy.theme import DARK_STYLESHEET
from magpy.widgets import NavigationBar, ViewType, WorkspacePanel

_MAX_RECENT = 10

_VIEW_NAMES = {
    ViewType.HOME: "Home",
    ViewType.AUDIO: "Audio Annotation",
    ViewType.INDICES: "Acoustic Indices",
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
        self._current_view = ViewType.HOME

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

        self._create_views()
        self._create_workspace_dock()

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

    def _create_views(self) -> None:
        # Two independent audio screens (own playback each); annotation is primary.
        self._annotation_screen = AudioAnnotationScreen(self._workspace)
        self._indices_screen = IndicesScreen(self._workspace)
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
        self._view_widgets = {
            ViewType.AUDIO: self._annotation_screen,
            ViewType.INDICES: self._indices_screen,
        }
        self._view_stack.addWidget(self._annotation_screen)
        self._view_stack.addWidget(self._indices_screen)
        for view_type, screen in self._screens.items():
            self._view_stack.addWidget(screen)
            self._view_widgets[view_type] = screen

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
        # Workspace panel activates a file -> load it in the active audio screen.
        self._workspace_panel.fileActivated.connect(self._load_in_active_audio_screen)
        self._workspace.opened.connect(self._on_workspace_opened)

        # Audio screen seams: status text to the shell's bar, navigate to front.
        # Each screen brings *itself* to front when it opens a file.
        for screen, view in (
            (self._annotation_screen, ViewType.AUDIO),
            (self._indices_screen, ViewType.INDICES),
        ):
            screen.statusMessage.connect(self.statusBar().showMessage)
            screen.navigateRequested.connect(
                lambda v=view: self._navigate_to(v)
            )

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
            # show. The audio screens are QMainWindows (not BaseScreens), so guard.
            if isinstance(widget, BaseScreen):
                widget.activate()
        self._update_title()

    def _update_title(self) -> None:
        view = _VIEW_NAMES.get(self._current_view, "Home")
        ws = self._workspace.name or "—"
        self.setWindowTitle(f"MagPy — {ws} — {view}")

    # --- audio entry point (CLI) -----------------------------------------
    def _load_in_active_audio_screen(self, path: str | Path) -> None:
        """Open a workspace file in whichever audio screen is showing.

        On the Indices screen, load there; otherwise load on the annotation
        screen (the default for any other view). The screen brings itself to
        front via ``navigateRequested``.
        """
        screen = (
            self._indices_screen
            if self._current_view == ViewType.INDICES
            else self._annotation_screen
        )
        screen.load_file(path)

    def add_audio_path(self, path: str | Path) -> None:
        """Link ``path`` into the workspace and open it on the annotation screen."""
        self._annotation_screen.add_audio_path(path)

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
        # The audio screen resets its own document off this same signal.
        self._remember_workspace(str(bundle_dir))
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
