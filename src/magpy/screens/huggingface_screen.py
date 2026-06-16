"""
HuggingFaceScreen -- pull datasets from the Hub and manage the local cache.

Unlike the search-based catalog views, Hugging Face is *pull-by-id*: you name a
dataset repo and download it. So this is a bespoke screen rather than a
:class:`~magpy.screens.catalog_screen.CatalogConfig`. It offers the "basic"
features:

* **Pull a dataset** by ``repo_id`` into the open workspace (threaded; audio that
  lands is *linked* as artifacts, mirroring the catalog downloads), and
* **Inspect / purge the hub cache** -- a table of cached repos with sizes and a
  purge button to reclaim disk.

Pure Qt + the services seam + the workers bridge -- no bioamla imports. Pull of a
public dataset needs no token; private repos read ``HF_TOKEN`` from the
environment (set it on the Settings screen).
"""

from __future__ import annotations

from typing import Optional

from PyQt6.QtCore import QThreadPool, Qt
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
)

from magpy.models import Workspace
from magpy.services import (
    KIND_AUDIO_FILE,
    HFPullResult,
    human_bytes,
    pull_dataset,
    purge_hf_cache,
    scan_hf_cache,
)
from magpy.workers import Worker

from .base import BaseScreen

_CACHE_COLUMNS = ["Repo", "Type", "Size"]


class HuggingFaceScreen(BaseScreen):
    def __init__(self, workspace: Workspace, parent=None) -> None:
        self._workspace = workspace
        self._cache: list = []
        self._pull_worker: Optional[Worker] = None
        self._cache_worker: Optional[Worker] = None
        self._purge_worker: Optional[Worker] = None
        super().__init__(parent)

    @property
    def screen_name(self) -> str:
        return "Hugging Face"

    @property
    def screen_icon(self) -> str:
        return "🤗"

    # --- layout -----------------------------------------------------------
    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(10)

        header = QLabel("🤗  Hugging Face")
        header.setStyleSheet("font-size: 22px; font-weight: bold; color: #d4d4d4;")
        layout.addWidget(header)
        blurb = QLabel(
            "Pull a dataset from the Hub into the workspace, and manage the local "
            "cache. Private repos use HF_TOKEN (set it on Settings)."
        )
        blurb.setWordWrap(True)
        blurb.setStyleSheet("color: #858585;")
        layout.addWidget(blurb)

        # --- pull a dataset ---
        pull_header = QLabel("Pull a dataset")
        pull_header.setStyleSheet("font-size: 15px; font-weight: bold; color: #d4d4d4;")
        layout.addWidget(pull_header)

        form = QFormLayout()
        self._repo_input = QLineEdit()
        self._repo_input.setPlaceholderText("e.g. user/dataset-name")
        form.addRow("Repo ID", self._repo_input)
        self._split_input = QLineEdit()
        self._split_input.setPlaceholderText("optional, e.g. train")
        form.addRow("Split", self._split_input)
        self._sr_input = QSpinBox()
        self._sr_input.setRange(0, 384_000)
        self._sr_input.setValue(16000)
        self._sr_input.setSuffix(" Hz")
        self._sr_input.setSpecialValueText("native")  # 0 -> keep native rate
        form.addRow("Sample rate", self._sr_input)
        layout.addLayout(form)

        pull_row = QHBoxLayout()
        self._pull_button = QPushButton("Pull → workspace")
        self._pull_button.clicked.connect(self._pull)
        pull_row.addWidget(self._pull_button)
        pull_row.addStretch(1)
        layout.addLayout(pull_row)

        # --- cache ---
        cache_header = QLabel("Hub cache")
        cache_header.setStyleSheet("font-size: 15px; font-weight: bold; color: #d4d4d4;")
        layout.addWidget(cache_header)

        cache_row = QHBoxLayout()
        self._scan_button = QPushButton("Scan cache")
        self._scan_button.clicked.connect(self._scan)
        cache_row.addWidget(self._scan_button)
        self._purge_button = QPushButton("Purge cache")
        self._purge_button.clicked.connect(self._purge)
        cache_row.addWidget(self._purge_button)
        cache_row.addStretch(1)
        layout.addLayout(cache_row)

        self._table = QTableWidget(0, len(_CACHE_COLUMNS))
        self._table.setHorizontalHeaderLabels(_CACHE_COLUMNS)
        self._table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self._table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self._table.verticalHeader().setVisible(False)
        layout.addWidget(self._table, stretch=1)

        self._progress = QProgressBar()
        self._progress.setRange(0, 0)
        self._progress.setVisible(False)
        layout.addWidget(self._progress)

        self._status = QLabel("Enter a repo ID and press Pull, or scan the cache.")
        self._status.setStyleSheet("color: #858585;")
        layout.addWidget(self._status)

    # --- pull -------------------------------------------------------------
    def _pull(self) -> None:
        if self._pull_worker is not None:
            return
        repo_id = self._repo_input.text().strip()
        if not repo_id:
            self._status.setText("Enter a repo ID to pull.")
            return
        if not self._workspace.is_open or self._workspace.dir is None:
            self._status.setText("Open a workspace to pull into.")
            return
        dest = self._workspace.dir / "imported" / "huggingface" / repo_id.replace("/", "__")
        sample_rate = self._sr_input.value() or None
        split = self._split_input.text().strip() or None
        self._set_busy(True, f"Pulling {repo_id}…")
        worker = Worker(
            pull_dataset, repo_id, dest, split=split, sample_rate=sample_rate
        )
        self._pull_worker = worker
        worker.signals.result.connect(self._on_pulled)
        worker.signals.error.connect(lambda exc: self._status.setText(f"Pull failed: {exc}"))
        worker.signals.finished.connect(self._on_pull_finished)
        QThreadPool.globalInstance().start(worker)

    def _on_pulled(self, result: HFPullResult) -> None:
        for path in result.audio_files:
            self._workspace.link(path, KIND_AUDIO_FILE)
        labels = f", {len(result.labels)} label(s)" if result.labels else ""
        self._status.setText(
            f"Pulled {result.repo_id}: {result.num_files} file(s)"
            f", linked {len(result.audio_files)} audio{labels}."
        )

    def _on_pull_finished(self) -> None:
        self._pull_worker = None
        self._set_busy(False)

    # --- cache ------------------------------------------------------------
    def _scan(self) -> None:
        if self._cache_worker is not None:
            return
        self._set_busy(True, "Scanning cache…")
        worker = Worker(scan_hf_cache)
        self._cache_worker = worker
        worker.signals.result.connect(self._on_cache)
        worker.signals.error.connect(lambda exc: self._status.setText(f"Scan failed: {exc}"))
        worker.signals.finished.connect(self._on_cache_finished)
        QThreadPool.globalInstance().start(worker)

    def _on_cache(self, repos: list) -> None:
        self._cache = repos
        self._table.setRowCount(len(repos))
        total = 0
        for row, repo in enumerate(repos):
            total += repo.size_bytes
            for col, text in enumerate((repo.repo_id, repo.repo_type, repo.size_human)):
                item = QTableWidgetItem(text)
                if col == 2:
                    item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
                self._table.setItem(row, col, item)
        self._table.resizeColumnsToContents()
        self._status.setText(f"{len(repos)} cached repo(s), {human_bytes(total)} total.")

    def _on_cache_finished(self) -> None:
        self._cache_worker = None
        self._set_busy(False)

    def _purge(self) -> None:
        if self._purge_worker is not None:
            return
        # Destructive and shared: this clears the *entire* HuggingFace hub cache
        # (every tool on the machine, not just MagPy's pulls). Re-downloading can
        # be gigabytes, so confirm before wiping.
        if QMessageBox.question(
            self,
            "Purge Hugging Face cache",
            "This clears the entire local Hugging Face hub cache (all datasets and "
            "models, system-wide — not just MagPy's pulls). Re-downloading may take "
            "a while.\n\nContinue?",
        ) != QMessageBox.StandardButton.Yes:
            return
        self._set_busy(True, "Purging cache…")
        worker = Worker(purge_hf_cache)
        self._purge_worker = worker
        worker.signals.result.connect(self._on_purged)
        worker.signals.error.connect(lambda exc: self._status.setText(f"Purge failed: {exc}"))
        worker.signals.finished.connect(self._on_purge_finished)
        QThreadPool.globalInstance().start(worker)

    def _on_purged(self, result) -> None:
        self._table.setRowCount(0)
        self._cache = []
        note = f" ({len(result.failures)} failed)" if result.failures else ""
        self._status.setText(
            f"Purged {len(result.deleted)} repo(s), freed {result.freed_human}{note}."
        )

    def _on_purge_finished(self) -> None:
        self._purge_worker = None
        self._set_busy(False)

    # --- busy state -------------------------------------------------------
    def _set_busy(self, busy: bool, message: str = "") -> None:
        for btn in (self._pull_button, self._scan_button, self._purge_button):
            btn.setEnabled(not busy)
        self._progress.setVisible(busy)
        if message:
            self._status.setText(message)
