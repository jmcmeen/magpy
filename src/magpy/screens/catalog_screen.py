"""
CatalogScreen -- a config-driven search/download view over a sound catalog.

One generic :class:`BaseScreen` drives all of the catalog views (Xeno-Canto,
Macaulay, iNaturalist, eBird). Each view is just a :class:`CatalogConfig`: a
source id, a title/icon/blurb, the search form fields, and the
:mod:`magpy.services.catalogs` search function to call. The screen:

* builds the search form from the config's fields,
* runs the search off-thread through a :class:`~magpy.workers.Worker` (network
  I/O) and renders the resulting :class:`~magpy.services.CatalogRecord`s,
* for downloadable catalogs, downloads the selected records into the open
  workspace (again threaded) and **links** them as artifacts, so they appear in
  the Files panel immediately.

Pure Qt + the services seam + the workers bridge -- no bioamla imports. The
screen binds to the :class:`~magpy.models.Workspace` model for the download
destination, mirroring how the rest of the shell treats the workspace.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

from PyQt6.QtCore import QThreadPool, QUrl
from PyQt6.QtGui import QDesktopServices
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
)

from magpy.models import Workspace
from magpy.services import CatalogRecord, KIND_AUDIO_FILE, download_records
from magpy.workers import Worker

from .base import BaseScreen

_COLUMNS = ["Common name", "Scientific", "Quality", "Duration", "Location", "Recordist"]


@dataclass(frozen=True)
class CatalogField:
    """One search-form field. ``kind`` is ``"text"`` or ``"int"``."""

    name: str  # keyword passed to the search function
    label: str
    kind: str = "text"
    default: int = 0
    placeholder: str = ""
    omit_if_zero: bool = False  # for optional int fields (e.g. place_id)


@dataclass(frozen=True)
class CatalogConfig:
    source: str
    title: str
    icon: str
    blurb: str
    search_fn: Callable[..., list[CatalogRecord]]
    fields: tuple[CatalogField, ...] = field(default_factory=tuple)
    downloadable: bool = True


class CatalogScreen(BaseScreen):
    def __init__(self, config: CatalogConfig, workspace: Workspace, parent=None) -> None:
        self._config = config
        self._workspace = workspace
        self._records: list[CatalogRecord] = []
        self._inputs: dict[str, QLineEdit | QSpinBox] = {}
        self._search_worker: Optional[Worker] = None
        self._download_worker: Optional[Worker] = None
        super().__init__(parent)

    @property
    def screen_name(self) -> str:
        return self._config.title

    @property
    def screen_icon(self) -> str:
        return self._config.icon

    # --- layout -----------------------------------------------------------
    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(10)

        header = QLabel(f"{self._config.icon}  {self._config.title}")
        header.setStyleSheet("font-size: 22px; font-weight: bold; color: #d4d4d4;")
        layout.addWidget(header)
        blurb = QLabel(self._config.blurb)
        blurb.setWordWrap(True)
        blurb.setStyleSheet("color: #858585;")
        layout.addWidget(blurb)

        form = QFormLayout()
        for f in self._config.fields:
            if f.kind == "int":
                w: QLineEdit | QSpinBox = QSpinBox()
                w.setRange(0, 10_000_000)
                w.setValue(int(f.default))
            else:
                w = QLineEdit()
                if f.placeholder:
                    w.setPlaceholderText(f.placeholder)
            self._inputs[f.name] = w
            form.addRow(f.label, w)
        layout.addLayout(form)

        button_row = QHBoxLayout()
        self._search_button = QPushButton("Search")
        self._search_button.clicked.connect(self._search)
        button_row.addWidget(self._search_button)
        if self._config.downloadable:
            self._download_button = QPushButton("Download selected → workspace")
            self._download_button.clicked.connect(self._download)
            button_row.addWidget(self._download_button)
        self._open_button = QPushButton("Open page")
        self._open_button.clicked.connect(self._open_page)
        button_row.addWidget(self._open_button)
        button_row.addStretch(1)
        layout.addLayout(button_row)

        self._progress = QProgressBar()
        self._progress.setRange(0, 0)
        self._progress.setVisible(False)
        layout.addWidget(self._progress)

        self._table = QTableWidget(0, len(_COLUMNS))
        self._table.setHorizontalHeaderLabels(_COLUMNS)
        self._table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self._table.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self._table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self._table.verticalHeader().setVisible(False)
        layout.addWidget(self._table, stretch=1)

        self._status = QLabel("Enter search terms and press Search.")
        self._status.setStyleSheet("color: #858585;")
        layout.addWidget(self._status)

    # --- form -> kwargs ---------------------------------------------------
    def _search_kwargs(self) -> dict:
        kwargs: dict = {}
        for f in self._config.fields:
            w = self._inputs[f.name]
            if isinstance(w, QSpinBox):
                value = w.value()
                if f.omit_if_zero and value == 0:
                    continue
                kwargs[f.name] = value
            else:
                kwargs[f.name] = w.text().strip()
        return kwargs

    # --- search -----------------------------------------------------------
    def _search(self) -> None:
        if self._search_worker is not None:
            return
        self._set_busy(True, "Searching…")
        worker = Worker(self._config.search_fn, **self._search_kwargs())
        self._search_worker = worker
        worker.signals.result.connect(self._on_results)
        worker.signals.error.connect(
            lambda exc: self._status.setText(f"Search failed: {exc}")
        )
        worker.signals.finished.connect(self._on_search_finished)
        QThreadPool.globalInstance().start(worker)

    def _on_search_finished(self) -> None:
        self._search_worker = None
        self._set_busy(False)

    def _on_results(self, records: list) -> None:
        self._records = records
        self._table.setRowCount(len(records))
        for row, rec in enumerate(records):
            cells = [rec.common_name, rec.scientific_name, rec.quality,
                     rec.duration, rec.location, rec.recordist]
            for col, text in enumerate(cells):
                self._table.setItem(row, col, QTableWidgetItem(text))
        self._table.resizeColumnsToContents()
        self._status.setText(f"{len(records)} result(s).")

    # --- download / open --------------------------------------------------
    def _selected_records(self) -> list[CatalogRecord]:
        return [self._records[i.row()] for i in self._table.selectionModel().selectedRows()]

    def _download(self) -> None:
        if self._download_worker is not None:
            return
        records = [r for r in self._selected_records() if r.downloadable]
        if not records:
            self._status.setText("Select one or more downloadable rows first.")
            return
        if not self._workspace.is_open or self._workspace.dir is None:
            self._status.setText("Open a workspace to download into.")
            return
        dest = self._workspace.dir / "imported" / "catalogs" / self._config.source
        ids = [r.record_id for r in records]
        self._set_busy(True, f"Downloading {len(ids)}…")
        worker = Worker(download_records, self._config.source, ids, dest)
        self._download_worker = worker
        worker.signals.result.connect(self._on_downloaded)
        worker.signals.error.connect(
            lambda exc: self._status.setText(f"Download failed: {exc}")
        )
        worker.signals.finished.connect(self._on_download_finished)
        QThreadPool.globalInstance().start(worker)

    def _on_download_finished(self) -> None:
        self._download_worker = None
        self._set_busy(False)

    def _on_downloaded(self, paths: list) -> None:
        for path in paths:
            self._workspace.link(path, KIND_AUDIO_FILE)
        self._status.setText(f"Downloaded and linked {len(paths)} file(s).")

    def _open_page(self) -> None:
        records = self._selected_records()
        if records and records[0].url:
            QDesktopServices.openUrl(QUrl(records[0].url))

    # --- busy state -------------------------------------------------------
    def _set_busy(self, busy: bool, message: str = "") -> None:
        self._search_button.setEnabled(not busy)
        if self._config.downloadable:
            self._download_button.setEnabled(not busy)
        self._progress.setVisible(busy)
        if message:
            self._status.setText(message)
