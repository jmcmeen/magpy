"""
DetectPanel -- the Audio view's detection controls + reviewable candidate list.

One cohesive dock widget that:

* builds a detector picker and a parameter form **generically** from the
  MagPy-owned :data:`~magpy.services.DETECTOR_SPECS` (so the widget never imports
  bioamla to learn a detector's knobs),
* emits :attr:`runRequested` (the shell runs the detector off-thread through a
  ``Worker`` -- detection is slow on long files),
* shows the resulting candidates in a table bound to a
  :class:`~magpy.models.CandidateSet`, with a confidence filter, and
* lets the user **promote** reviewed candidates into the curated annotation set
  via :attr:`promoteRequested`.

It is a pure view: it holds no detection state (that lives in ``CandidateSet``)
and performs no compute. No bioamla imports.
"""

from __future__ import annotations

from typing import Optional

from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from magpy.models import CandidateSet
from magpy.services import DETECTOR_SPECS, Candidate, DetectorSpec

_COLUMNS = ["Detector", "Start (s)", "End (s)", "Conf.", "Low (Hz)", "High (Hz)"]


def _freq(v: Optional[float]) -> str:
    return "" if v is None else f"{v:.0f}"


class DetectPanel(QWidget):
    runRequested = pyqtSignal(str, dict)  # (detector kind, params)
    promoteRequested = pyqtSignal(list)  # list[Candidate]
    candidateActivated = pyqtSignal(float, float)  # (start, end) for go-to/seek

    def __init__(self, candidates: CandidateSet, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._model = candidates
        self._syncing = False
        self._rows: list[Candidate] = []  # row -> candidate (Candidate is frozen, no id)
        self._param_widgets: dict[str, QDoubleSpinBox | QSpinBox] = {}

        self._build_ui()
        self._on_detector_changed(0)

        self._model.reset.connect(self._rebuild)
        self._model.thresholdChanged.connect(self._rebuild)
        self._model.selectionChanged.connect(self._on_model_selection)
        self.table.itemSelectionChanged.connect(self._on_table_selection)
        self.table.itemDoubleClicked.connect(self._on_double_click)
        self._rebuild()

    # --- layout -----------------------------------------------------------
    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        self.detector_combo = QComboBox()
        for spec in DETECTOR_SPECS:
            self.detector_combo.addItem(spec.label, spec.kind)
        self.detector_combo.currentIndexChanged.connect(self._on_detector_changed)
        layout.addWidget(self.detector_combo)

        self._description = QLabel()
        self._description.setWordWrap(True)
        self._description.setStyleSheet("color: #858585;")
        layout.addWidget(self._description)

        self._form_host = QWidget()
        self._form = QFormLayout(self._form_host)
        self._form.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._form_host)

        run_row = QHBoxLayout()
        self.run_button = QPushButton("Run detection")
        self.run_button.clicked.connect(self._emit_run)
        self.clear_button = QPushButton("Clear")
        self.clear_button.clicked.connect(self._model.clear)
        run_row.addWidget(self.run_button)
        run_row.addWidget(self.clear_button)
        layout.addLayout(run_row)

        self.progress = QProgressBar()
        self.progress.setRange(0, 0)  # indeterminate (detect_all has no progress)
        self.progress.setVisible(False)
        layout.addWidget(self.progress)

        thr_row = QHBoxLayout()
        thr_row.addWidget(QLabel("Min confidence"))
        self.threshold_spin = QDoubleSpinBox()
        self.threshold_spin.setRange(0.0, 1.0)
        self.threshold_spin.setSingleStep(0.05)
        self.threshold_spin.setDecimals(2)
        self.threshold_spin.valueChanged.connect(self._model.set_threshold)
        thr_row.addWidget(self.threshold_spin)
        thr_row.addStretch(1)
        layout.addLayout(thr_row)

        self.table = QTableWidget(0, len(_COLUMNS))
        self.table.setHorizontalHeaderLabels(_COLUMNS)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.verticalHeader().setVisible(False)
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        layout.addWidget(self.table, stretch=1)

        self._count_label = QLabel("No detections yet.")
        self._count_label.setStyleSheet("color: #858585;")
        layout.addWidget(self._count_label)

        promote_row = QHBoxLayout()
        self.promote_button = QPushButton("Promote selected → annotations")
        self.promote_button.clicked.connect(self._promote_selected)
        self.promote_all_button = QPushButton("Promote all")
        self.promote_all_button.clicked.connect(self._promote_all)
        promote_row.addWidget(self.promote_button)
        promote_row.addWidget(self.promote_all_button)
        layout.addLayout(promote_row)

    def _current_spec(self) -> DetectorSpec:
        return DETECTOR_SPECS[self.detector_combo.currentIndex()]

    def _on_detector_changed(self, _index: int) -> None:
        spec = self._current_spec()
        self._description.setText(spec.description)
        # Rebuild the param form for the newly-selected detector.
        while self._form.rowCount():
            self._form.removeRow(0)
        self._param_widgets.clear()
        for p in spec.params:
            if p.decimals == 0:
                w: QDoubleSpinBox | QSpinBox = QSpinBox()
                w.setRange(int(p.minimum), int(p.maximum))
                w.setSingleStep(int(p.step) or 1)
                w.setValue(int(p.default))
            else:
                w = QDoubleSpinBox()
                w.setDecimals(p.decimals)
                w.setRange(p.minimum, p.maximum)
                w.setSingleStep(p.step)
                w.setValue(p.default)
            self._param_widgets[p.name] = w
            self._form.addRow(p.label, w)

    def _current_params(self) -> dict[str, float]:
        return {name: w.value() for name, w in self._param_widgets.items()}

    # --- run / promote ----------------------------------------------------
    def _emit_run(self) -> None:
        self.runRequested.emit(self._current_spec().kind, self._current_params())

    def set_running(self, running: bool) -> None:
        """Toggle the busy state (the shell calls this around the Worker run)."""
        self.run_button.setEnabled(not running)
        self.progress.setVisible(running)

    def _promote_selected(self) -> None:
        picked = [self._rows[i.row()] for i in self.table.selectionModel().selectedRows()]
        if picked:
            self.promoteRequested.emit(picked)

    def _promote_all(self) -> None:
        if self._rows:
            self.promoteRequested.emit(list(self._rows))

    # --- model <-> table --------------------------------------------------
    def _rebuild(self, *_: object) -> None:
        self._syncing = True
        try:
            self._rows = self._model.visible_items()
            self.table.setRowCount(len(self._rows))
            for row, c in enumerate(self._rows):
                values = [c.detector, f"{c.start_time:.3f}", f"{c.end_time:.3f}",
                          f"{c.confidence:.2f}", _freq(c.low_freq), _freq(c.high_freq)]
                for col, text in enumerate(values):
                    self.table.setItem(row, col, QTableWidgetItem(text))
            self._on_model_selection(self._model.selected)
            total = len(self._model)
            shown = len(self._rows)
            if total == 0:
                self._count_label.setText("No detections yet.")
            else:
                det = f" · {self._model.detector}" if self._model.detector else ""
                self._count_label.setText(f"{shown} of {total} shown{det}")
        finally:
            self._syncing = False

    def _on_model_selection(self, candidate: Optional[Candidate]) -> None:
        self._syncing = True
        try:
            if candidate is None or candidate not in self._rows:
                self.table.clearSelection()
            else:
                self.table.selectRow(self._rows.index(candidate))
        finally:
            self._syncing = False

    def _on_table_selection(self) -> None:
        if self._syncing:
            return
        rows = self.table.selectionModel().selectedRows()
        self._model.select(self._rows[rows[0].row()] if rows else None)

    def _on_double_click(self, item: QTableWidgetItem) -> None:
        c = self._rows[item.row()]
        self.candidateActivated.emit(c.start_time, c.end_time)
