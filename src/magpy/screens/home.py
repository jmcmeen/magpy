"""
Home Screen - Landing page for MagPy.

Displays welcome information and quick access to common tasks.
"""

from __future__ import annotations

from typing import Optional

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QGridLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QFrame,
)

from magpy.widgets import ViewType

from .base import BaseScreen

# The tool cards shown on the dashboard, in display order. Each routes to a
# nav view via ``navigate_requested``. Kept in sync with the NavigationBar.
_TOOL_CARDS: tuple[tuple[str, str, str, ViewType], ...] = (
    ("🎵", "Audio Analysis", "Spectrogram, playback, annotations", ViewType.AUDIO),
    ("📁", "Datasets", "Build & augment training data", ViewType.DATASETS),
    ("🧠", "Model Training", "Train & evaluate AST models", ViewType.TRAINING),
    ("🔭", "Explore", "Cluster embeddings & find novelty", ViewType.EXPLORE),
    ("📚", "Batch", "Run ops across many files", ViewType.BATCH),
    ("🦋", "iNaturalist", "Search observations with sound", ViewType.INATURALIST),
    ("🐦", "eBird", "Recent regional observations", ViewType.EBIRD),
    ("🏛️", "Macaulay Library", "Cornell Lab reference audio", ViewType.MACAULAY),
    ("🎼", "Xeno-Canto", "Search the recording archive", ViewType.XENOCANTO),
    ("🤗", "Hugging Face", "Pull datasets & manage cache", ViewType.HUGGINGFACE),
    ("⚙️", "Settings", "Devices, dependencies, API keys", ViewType.SETTINGS),
)


class ActionCard(QPushButton):
    """A clickable card for quick actions."""

    def __init__(
        self,
        icon: str,
        title: str,
        description: str,
        parent: Optional[QWidget] = None,
    ):
        super().__init__(parent)
        self.setFixedSize(200, 140)
        self.setCursor(Qt.CursorShape.PointingHandCursor)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(8)

        # Icon
        icon_label = QLabel(icon)
        icon_label.setStyleSheet("font-size: 32px; background: transparent;")
        icon_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(icon_label)

        # Title
        title_label = QLabel(title)
        title_label.setStyleSheet("""
            font-size: 14px;
            font-weight: bold;
            color: #d4d4d4;
            background: transparent;
        """)
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title_label.setWordWrap(True)
        layout.addWidget(title_label)

        # Description
        desc_label = QLabel(description)
        desc_label.setStyleSheet("""
            font-size: 11px;
            color: #808080;
            background: transparent;
        """)
        desc_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        desc_label.setWordWrap(True)
        layout.addWidget(desc_label)

        layout.addStretch()

        self.setStyleSheet("""
            ActionCard {
                background-color: #2d2d2d;
                border: 1px solid #3c3c3c;
                border-radius: 8px;
            }
            ActionCard:hover {
                background-color: #3c3c3c;
                border-color: #0e639c;
            }
            ActionCard:pressed {
                background-color: #094771;
            }
        """)


class HomeScreen(BaseScreen):
    """
    Home screen - the default landing view for MagPy.

    Provides quick access to common actions and displays project information.
    """

    # Signals for navigation
    new_workspace_clicked = pyqtSignal()
    open_workspace_clicked = pyqtSignal()
    recent_workspace_clicked = pyqtSignal(str)  # bundle dir
    navigate_requested = pyqtSignal(object)  # ViewType of the tool card clicked

    @property
    def screen_name(self) -> str:
        return "Home"

    @property
    def screen_icon(self) -> str:
        return "🏠"

    def _setup_ui(self):
        """Set up the home screen UI.

        Keeps everything on one page: the cards stay full-size, but the tools
        sit in a grid wide enough for two rows (instead of stacking into three),
        the lengthy info columns are gone, and top/bottom stretches centre the
        content so there is no lopsided dead space. A scroll area remains only as
        a fallback for very short windows -- no scrollbar shows when it all fits.
        """
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setStyleSheet("QScrollArea { border: none; background: transparent; }")

        content = QWidget()
        content.setStyleSheet("background: transparent;")
        main_layout = QVBoxLayout(content)
        main_layout.setContentsMargins(40, 24, 40, 24)
        main_layout.setSpacing(20)

        main_layout.addStretch(1)

        # Hero
        title = QLabel("MagPy")
        title.setStyleSheet("font-size: 48px; font-weight: bold; color: #d4d4d4;")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        main_layout.addWidget(title)

        subtitle = QLabel("Bioacoustics Analysis Platform")
        subtitle.setStyleSheet("font-size: 18px; color: #808080;")
        subtitle.setAlignment(Qt.AlignmentFlag.AlignCenter)
        main_layout.addWidget(subtitle)

        # Workspace actions
        main_layout.addWidget(self._section_label("Get Started"))
        workspace_widget = QWidget()
        workspace_layout = QHBoxLayout(workspace_widget)
        workspace_layout.setContentsMargins(0, 0, 0, 0)
        workspace_layout.setSpacing(16)
        workspace_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        for icon, title_text, desc, signal in (
            ("📁", "New Workspace", "Start a new workspace", self.new_workspace_clicked),
            ("📂", "Open Workspace", "Open an existing workspace", self.open_workspace_clicked),
        ):
            card = ActionCard(icon, title_text, desc)
            card.clicked.connect(signal.emit)
            workspace_layout.addWidget(card)
        main_layout.addWidget(workspace_widget)

        # Tool cards -- one per nav view, in a grid wide enough for two rows.
        main_layout.addWidget(self._section_label("Tools"))
        tools_widget = QWidget()
        tools_grid = QGridLayout(tools_widget)
        tools_grid.setContentsMargins(0, 0, 0, 0)
        tools_grid.setSpacing(16)
        tools_grid.setAlignment(Qt.AlignmentFlag.AlignCenter)
        columns = 6
        for index, (icon, title_text, desc, view_type) in enumerate(_TOOL_CARDS):
            card = ActionCard(icon, title_text, desc)
            card.clicked.connect(
                lambda _checked=False, vt=view_type: self.navigate_requested.emit(vt)
            )
            tools_grid.addWidget(card, index // columns, index % columns)
        main_layout.addWidget(tools_widget)

        # Recent workspaces (populated by the shell; hidden when empty)
        self._recent_header = self._section_label("Recent Workspaces")
        self._recent_header.setVisible(False)
        main_layout.addWidget(self._recent_header)

        self._recent_container = QWidget()
        self._recent_layout = QHBoxLayout(self._recent_container)
        self._recent_layout.setContentsMargins(0, 0, 0, 0)
        self._recent_layout.setSpacing(12)
        self._recent_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        main_layout.addWidget(self._recent_container)

        main_layout.addStretch(1)

        # Info cards: About / Related Projects / Acknowledgments, in one row.
        info_layout = QHBoxLayout()
        info_layout.setSpacing(20)
        info_layout.addWidget(
            self._create_info_section(
                "About MagPy",
                "MagPy is an open-source bioacoustics analysis platform for "
                "researchers, conservationists, and citizen scientists: audio "
                "visualization, species detection, and dataset management, built "
                "with Python &amp; PyQt6 on the bioamla library.",
            )
        )
        info_layout.addWidget(
            self._create_info_section(
                "Related Projects",
                "<b>bioamla</b> - bioacoustics ML library (MagPy's engine)<br>"
                "<b>BirdNET</b> - bird sound identification<br>"
                "<b>OpenSoundscape</b> - bioacoustics toolkit<br>"
                "<b>Audacity</b> - open-source audio editor",
            )
        )
        info_layout.addWidget(
            self._create_info_section("Acknowledgments", "Coming soon.")
        )
        main_layout.addLayout(info_layout)

        # Footer: license + version on a single line.
        footer = QLabel(
            "MIT License  ·  Built with Python & PyQt6 on bioamla  ·  Version 0.0.1"
        )
        footer.setStyleSheet("font-size: 11px; color: #606060;")
        footer.setAlignment(Qt.AlignmentFlag.AlignCenter)
        main_layout.addWidget(footer)

        scroll.setWidget(content)

        outer_layout = QVBoxLayout(self)
        outer_layout.setContentsMargins(0, 0, 0, 0)
        outer_layout.addWidget(scroll)

    def _section_label(self, text: str) -> QLabel:
        """A centred section header used across the dashboard."""
        label = QLabel(text)
        label.setStyleSheet("font-size: 15px; font-weight: bold; color: #d4d4d4;")
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        return label

    def _create_info_section(self, title: str, content: str) -> QWidget:
        """An info panel (card) with a title and rich-text body."""
        section = QWidget()
        section.setStyleSheet("QWidget { background-color: #2d2d2d; border-radius: 8px; }")

        layout = QVBoxLayout(section)
        layout.setContentsMargins(16, 14, 16, 14)
        layout.setSpacing(8)

        header = QLabel(title)
        header.setStyleSheet(
            "font-size: 14px; font-weight: bold; color: #d4d4d4; background: transparent;"
        )
        layout.addWidget(header)

        text = QLabel(content)
        text.setStyleSheet("font-size: 11px; color: #a0a0a0; background: transparent;")
        text.setWordWrap(True)
        text.setTextFormat(Qt.TextFormat.RichText)
        layout.addWidget(text)

        layout.addStretch()
        return section

    def set_recent_workspaces(self, paths: list[str]) -> None:
        """Populate the recent-workspaces list (called by the shell)."""
        from pathlib import Path

        while self._recent_layout.count():
            item = self._recent_layout.takeAt(0)
            w = item.widget()
            if w is not None:
                w.deleteLater()
        self._recent_header.setVisible(bool(paths))
        for path in paths:
            btn = QPushButton(Path(path).name)
            btn.setCursor(Qt.CursorShape.PointingHandCursor)
            btn.setToolTip(path)
            btn.setFlat(True)
            btn.setStyleSheet("color: #0e9c9c; background: transparent; border: none;")
            btn.clicked.connect(lambda _checked=False, p=path: self.recent_workspace_clicked.emit(p))
            self._recent_layout.addWidget(btn)
