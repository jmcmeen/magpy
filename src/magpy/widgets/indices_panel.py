"""
IndicesPanel -- the Audio view's acoustic-indices summary.

A pure view: a "Compute" button (the shell runs the whole-file computation
off-thread through a ``Worker``), a busy indicator, and a read-only table of the
resulting scalar indices with their descriptions as tooltips. Holds no compute
and no bioamla imports; it just renders an :class:`~magpy.services.IndexSummary`
handed to it via :meth:`set_summary`.
"""

from __future__ import annotations

from typing import Optional

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QHeaderView,
    QLabel,
    QProgressBar,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from magpy.services import IndexSummary

_COLUMNS = ["Index", "Value"]


class IndicesPanel(QWidget):
    computeRequested = pyqtSignal()

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        self.compute_button = QPushButton("Compute indices")
        self.compute_button.clicked.connect(self.computeRequested)
        layout.addWidget(self.compute_button)

        self.progress = QProgressBar()
        self.progress.setRange(0, 0)  # indeterminate
        self.progress.setVisible(False)
        layout.addWidget(self.progress)

        self.table = QTableWidget(0, len(_COLUMNS))
        self.table.setHorizontalHeaderLabels(_COLUMNS)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
        self.table.verticalHeader().setVisible(False)
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        layout.addWidget(self.table, stretch=1)

        self._caption = QLabel("Whole-file acoustic indices. Press Compute.")
        self._caption.setWordWrap(True)
        self._caption.setStyleSheet("color: #858585;")
        layout.addWidget(self._caption)

    def set_running(self, running: bool) -> None:
        self.compute_button.setEnabled(not running)
        self.progress.setVisible(running)

    def set_summary(self, summary: Optional[IndexSummary]) -> None:
        """Render an :class:`IndexSummary` (or clear the table when ``None``)."""
        if summary is None:
            self.table.setRowCount(0)
            self._caption.setText("Whole-file acoustic indices. Press Compute.")
            return
        rows = summary.rows()
        self.table.setRowCount(len(rows))
        for r, row in enumerate(rows):
            name = QTableWidgetItem(row.label)
            name.setToolTip(row.description)
            value = QTableWidgetItem("—" if row.value is None else f"{row.value:.4f}")
            value.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            value.setToolTip(row.description)
            self.table.setItem(r, 0, name)
            self.table.setItem(r, 1, value)
        self._caption.setText(
            f"Over {summary.duration:.1f}s · {summary.sample_rate} Hz"
        )
