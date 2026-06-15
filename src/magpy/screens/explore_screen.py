"""
Explore Screen -- cluster and visualize the embedding space.

The end of the bioacoustics pipeline: after the Batch screen extracts AST
embeddings (``.npy`` per file), Explore loads them, clusters them, projects them
to 2-D, and draws a scatter coloured by cluster — the interactive "what's in this
soundscape / what's novel" view (bioamla's ``cluster`` group: fit / reduce /
analyze / novelty).

Compute (clustering + UMAP/t-SNE reduction) runs off-thread through a
:class:`~magpy.workers.Worker`; it has no progress hook, so the bar is
indeterminate. The plot (a pyqtgraph scatter) and metrics render on the UI thread
from the returned :class:`~magpy.services.EmbeddingScatter`. Binds to the
Workspace for a default embeddings folder and CSV export location. No bioamla
imports.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import pyqtgraph as pg
from PyQt6.QtCore import QThreadPool
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from magpy.models import Workspace
from magpy.services import (
    CLUSTER_METHODS,
    REDUCE_METHODS,
    cluster_embeddings_dir,
    export_scatter_csv,
)
from magpy.workers import Worker

from .base import BaseScreen

_NOISE_BRUSH = pg.mkBrush(120, 120, 120, 160)
_NOVEL_PEN = pg.mkPen(255, 60, 60, width=2)


class ExploreScreen(BaseScreen):
    def __init__(self, workspace: Workspace, parent: Optional[QWidget] = None) -> None:
        self._workspace = workspace
        self._worker: Optional[Worker] = None
        self._scatter_result = None
        super().__init__(parent)

    @property
    def screen_name(self) -> str:
        return "Explore"

    @property
    def screen_icon(self) -> str:
        return "🔭"

    def _default_dir(self, *parts: str) -> str:
        if self._workspace.is_open and self._workspace.dir is not None:
            return str(self._workspace.dir.joinpath(*parts))
        return ""

    # --- layout -----------------------------------------------------------
    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(8)

        header = QLabel("🔭  Explore")
        header.setStyleSheet("font-size: 22px; font-weight: bold; color: #d4d4d4;")
        layout.addWidget(header)
        sub = QLabel("Cluster and visualize AST embeddings (.npy from Batch → Model embeddings).")
        sub.setWordWrap(True)
        sub.setStyleSheet("color: #858585;")
        layout.addWidget(sub)

        # Embeddings folder.
        in_row = QHBoxLayout()
        in_row.addWidget(QLabel("Embeddings"))
        self._dir_edit = QLineEdit(self._default_dir())
        self._dir_edit.setPlaceholderText("Folder of .npy embedding files")
        in_row.addWidget(self._dir_edit, stretch=1)
        browse = QPushButton("Browse…")
        browse.clicked.connect(self._pick_dir)
        in_row.addWidget(browse)
        layout.addLayout(in_row)

        # Controls.
        ctrl = QHBoxLayout()
        self._cluster_combo = QComboBox()
        self._cluster_combo.addItems(list(CLUSTER_METHODS))
        self._reduce_combo = QComboBox()
        self._reduce_combo.addItems(list(REDUCE_METHODS))
        self._min_size = QSpinBox()
        self._min_size.setRange(2, 1000)
        self._min_size.setValue(5)
        self._n_clusters = QSpinBox()
        self._n_clusters.setRange(0, 1000)
        self._n_clusters.setValue(0)
        self._n_clusters.setToolTip("kmeans/agglomerative target; 0 = auto")
        self._novelty = QCheckBox("Flag novel")
        for lbl, w in (("Cluster", self._cluster_combo), ("Reduce", self._reduce_combo),
                       ("Min size", self._min_size), ("k", self._n_clusters)):
            ctrl.addWidget(QLabel(lbl))
            ctrl.addWidget(w)
        ctrl.addWidget(self._novelty)
        ctrl.addStretch(1)
        layout.addLayout(ctrl)

        run_row = QHBoxLayout()
        self._run_button = QPushButton("Cluster & plot")
        self._run_button.clicked.connect(self._run)
        run_row.addWidget(self._run_button)
        self._export_button = QPushButton("Export CSV")
        self._export_button.clicked.connect(self._export)
        self._export_button.setEnabled(False)
        run_row.addWidget(self._export_button)
        run_row.addStretch(1)
        layout.addLayout(run_row)

        self._progress = QProgressBar()
        self._progress.setRange(0, 0)
        self._progress.setVisible(False)
        layout.addWidget(self._progress)

        # Scatter plot.
        self._plot = pg.PlotWidget()
        self._plot.setBackground("#1e1e1e")
        self._plot.showGrid(x=True, y=True, alpha=0.2)
        self._plot.setLabel("bottom", "dim 1")
        self._plot.setLabel("left", "dim 2")
        self._scatter = pg.ScatterPlotItem(size=9, pen=pg.mkPen(None))
        self._plot.addItem(self._scatter)
        layout.addWidget(self._plot, stretch=1)

        self._status = QLabel("Load a folder of embeddings, then Cluster & plot.")
        self._status.setStyleSheet("color: #858585;")
        layout.addWidget(self._status)

    def _pick_dir(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Embeddings folder", self._dir_edit.text())
        if path:
            self._dir_edit.setText(path)

    # --- run --------------------------------------------------------------
    def _run(self) -> None:
        if self._worker is not None:
            return
        input_dir = self._dir_edit.text().strip()
        if not input_dir or not Path(input_dir).is_dir():
            self._status.setText("Pick a valid embeddings folder first.")
            return
        self._set_running(True)
        worker = Worker(
            cluster_embeddings_dir, input_dir,
            cluster_method=self._cluster_combo.currentText(),
            reduce_method=self._reduce_combo.currentText(),
            min_cluster_size=self._min_size.value(),
            n_clusters=self._n_clusters.value() or None,
            find_novelty=self._novelty.isChecked(),
        )
        self._worker = worker
        worker.signals.result.connect(self._on_result)
        worker.signals.error.connect(lambda exc: self._status.setText(f"Failed: {exc}"))
        worker.signals.finished.connect(self._on_finished)
        QThreadPool.globalInstance().start(worker)

    def _on_finished(self) -> None:
        self._worker = None
        self._set_running(False)

    def _set_running(self, running: bool) -> None:
        self._run_button.setEnabled(not running)
        self._progress.setVisible(running)
        if running:
            self._status.setText("Clustering…")

    def _on_result(self, scatter) -> None:
        self._scatter_result = scatter
        self._export_button.setEnabled(True)
        novel = set(scatter.novel_indices)
        spots = []
        for i in range(len(scatter.labels)):
            label = int(scatter.labels[i])
            brush = _NOISE_BRUSH if label < 0 else pg.mkBrush(pg.intColor(label, hues=12, alpha=200))
            pen = _NOVEL_PEN if i in novel else pg.mkPen(None)
            spots.append({"pos": (float(scatter.x[i]), float(scatter.y[i])), "brush": brush, "pen": pen})
        self._scatter.setData(spots)
        self._plot.setTitle(
            f"{scatter.cluster_method} · {scatter.reduce_method}", color="#d4d4d4", size="10pt")
        self._status.setText(scatter.message)

    def _export(self) -> None:
        if self._scatter_result is None:
            return
        default = self._default_dir("clusters.csv") or "clusters.csv"
        path, _ = QFileDialog.getSaveFileName(self, "Export clusters CSV", default, "CSV (*.csv)")
        if not path:
            return
        try:
            export_scatter_csv(self._scatter_result, path)
        except Exception as exc:  # noqa: BLE001
            self._status.setText(f"Export failed: {exc}")
            return
        self._status.setText(f"Exported to {Path(path).name}")
