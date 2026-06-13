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
    QLabel,
    QPushButton,
    QScrollArea,
    QFrame,
)

from .base import BaseScreen


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
    audio_analysis_clicked = pyqtSignal()
    dataset_editor_clicked = pyqtSignal()
    ai_trainer_clicked = pyqtSignal()
    inaturalist_clicked = pyqtSignal()

    @property
    def screen_name(self) -> str:
        return "Home"

    @property
    def screen_icon(self) -> str:
        return "🏠"

    def _setup_ui(self):
        """Set up the home screen UI."""
        # Main scroll area
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setStyleSheet("QScrollArea { border: none; background: transparent; }")

        content = QWidget()
        content.setStyleSheet("background: transparent;")
        main_layout = QVBoxLayout(content)
        main_layout.setContentsMargins(60, 40, 60, 40)
        main_layout.setSpacing(40)

        # Hero section
        hero_layout = QVBoxLayout()
        hero_layout.setSpacing(8)

        title = QLabel("MagPy")
        title.setStyleSheet("""
            QLabel {
                font-size: 56px;
                font-weight: bold;
                color: #d4d4d4;
            }
        """)
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        hero_layout.addWidget(title)

        subtitle = QLabel("Bioacoustics Analysis Platform")
        subtitle.setStyleSheet("""
            QLabel {
                font-size: 20px;
                color: #808080;
            }
        """)
        subtitle.setAlignment(Qt.AlignmentFlag.AlignCenter)
        hero_layout.addWidget(subtitle)

        main_layout.addLayout(hero_layout)

        # Quick actions section
        actions_section = QVBoxLayout()
        actions_section.setSpacing(16)

        actions_header = QLabel("Get Started")
        actions_header.setStyleSheet("""
            QLabel {
                font-size: 18px;
                font-weight: bold;
                color: #d4d4d4;
            }
        """)
        actions_header.setAlignment(Qt.AlignmentFlag.AlignCenter)
        actions_section.addWidget(actions_header)

        # Action cards grid
        cards_widget = QWidget()
        cards_layout = QHBoxLayout(cards_widget)
        cards_layout.setSpacing(16)
        cards_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

        actions = [
            ("📁", "New Workspace", "Start working in a folder", self.new_workspace_clicked),
            ("📂", "Open Workspace", "Open an existing folder", self.open_workspace_clicked),
            ("🎵", "Audio Analysis", "Analyze audio recordings", self.audio_analysis_clicked),
            ("📊", "Dataset Editor", "Manage training datasets", self.dataset_editor_clicked),
            ("🧠", "AI Trainer", "Train detection models", self.ai_trainer_clicked),
            ("🦋", "iNaturalist", "Connect to iNaturalist", self.inaturalist_clicked),
        ]

        for icon, title, desc, signal in actions:
            card = ActionCard(icon, title, desc)
            card.clicked.connect(signal.emit)
            cards_layout.addWidget(card)

        actions_section.addWidget(cards_widget)
        main_layout.addLayout(actions_section)

        # Recent workspaces (populated by the shell; hidden when empty)
        self._recent_header = QLabel("Recent Workspaces")
        self._recent_header.setStyleSheet(
            "font-size: 18px; font-weight: bold; color: #d4d4d4;"
        )
        self._recent_header.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._recent_header.setVisible(False)
        main_layout.addWidget(self._recent_header)

        self._recent_container = QWidget()
        self._recent_layout = QVBoxLayout(self._recent_container)
        self._recent_layout.setSpacing(4)
        self._recent_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        main_layout.addWidget(self._recent_container)

        # Separator
        separator = QFrame()
        separator.setFrameShape(QFrame.Shape.HLine)
        separator.setStyleSheet("background-color: #3c3c3c;")
        separator.setFixedHeight(1)
        main_layout.addWidget(separator)

        # Info sections in columns
        info_layout = QHBoxLayout()
        info_layout.setSpacing(40)

        # About section
        about_section = self._create_info_section(
            "About MagPy",
            """MagPy is an open-source bioacoustics analysis platform designed for
researchers, conservationists, and citizen scientists. It provides tools for
audio visualization, species detection, and data management.

Built with Python and PyQt6, MagPy integrates machine learning models for
automated species identification and supports collaboration through
standardized data formats."""
        )
        info_layout.addWidget(about_section)

        # Related Projects section
        related_section = self._create_info_section(
            "Related Projects",
            """<b>BioAMLA</b> - Bioacoustics Machine Learning Applications<br>
Core library for audio processing and ML inference.<br><br>
<b>BirdNET</b> - Bird sound identification<br>
Deep learning model for bird species detection.<br><br>
<b>OpenSoundscape</b> - Bioacoustics toolkit<br>
Python library for bioacoustic analysis.<br><br>
<b>Audacity</b> - Audio editor<br>
Free, open-source audio software."""
        )
        info_layout.addWidget(related_section)

        # Acknowledgments section
        ack_section = self._create_info_section(
            "Acknowledgments",
            """Coming soon."""
        )
        info_layout.addWidget(ack_section)

        main_layout.addLayout(info_layout)

        # License section
        license_section = QVBoxLayout()
        license_section.setSpacing(8)

        license_header = QLabel("License")
        license_header.setStyleSheet("""
            QLabel {
                font-size: 14px;
                font-weight: bold;
                color: #d4d4d4;
            }
        """)
        license_section.addWidget(license_header)

        license_text = QLabel(
            "MagPy is released under the MIT License. See the LICENSE file for details."
        )
        license_text.setStyleSheet("""
            QLabel {
                font-size: 12px;
                color: #808080;
            }
        """)
        license_text.setWordWrap(True)
        license_section.addWidget(license_text)

        main_layout.addLayout(license_section)

        # Footer
        # TODO: can we make this dynamic to show actual version number and build info?
        footer = QLabel("Version 0.0.1 | Built with Python & PyQt6")
        footer.setStyleSheet("""
            QLabel {
                font-size: 11px;
                color: #606060;
            }
        """)
        footer.setAlignment(Qt.AlignmentFlag.AlignCenter)
        main_layout.addWidget(footer)

        main_layout.addStretch()

        scroll.setWidget(content)

        # Set up the main layout
        outer_layout = QVBoxLayout(self)
        outer_layout.setContentsMargins(0, 0, 0, 0)
        outer_layout.addWidget(scroll)

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

    def _create_info_section(self, title: str, content: str) -> QWidget:
        """Create an info section with title and content."""
        section = QWidget()
        section.setStyleSheet("""
            QWidget {
                background-color: #2d2d2d;
                border-radius: 8px;
            }
        """)

        layout = QVBoxLayout(section)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(12)

        header = QLabel(title)
        header.setStyleSheet("""
            QLabel {
                font-size: 16px;
                font-weight: bold;
                color: #d4d4d4;
                background: transparent;
            }
        """)
        layout.addWidget(header)

        text = QLabel(content)
        text.setStyleSheet("""
            QLabel {
                font-size: 12px;
                color: #a0a0a0;
                line-height: 1.5;
                background: transparent;
            }
        """)
        text.setWordWrap(True)
        text.setTextFormat(Qt.TextFormat.RichText)
        layout.addWidget(text)

        layout.addStretch()

        return section
