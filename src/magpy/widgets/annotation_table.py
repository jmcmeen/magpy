"""
AnnotationTable -- the selection table: one row per annotation, bound to an
:class:`AnnotationSet`.

The label and bounds columns (begin/end time, low/high frequency) are editable in
place -- typing a number is the precise way to place a boundary. A read-only
confidence column shows how sure a detector or classifier was. After them come
the **measurement columns** the user has chosen (:meth:`set_measurement_keys`);
the screen computes the values and pushes them in (:meth:`set_measurements`), so
this stays a view: it observes the model's signals and calls back into it,
holding no annotation state of its own. No bioamla imports.
"""

from __future__ import annotations

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QBrush
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QHeaderView,
    QTableWidget,
    QTableWidgetItem,
)

from magpy.models import AnnotationSet
from magpy.services import MEASUREMENTS_BY_KEY, Annotation

from ._plot import label_color

_FIXED = ["Label", "Begin (s)", "End (s)", "Low (Hz)", "High (Hz)", "Conf."]
_COL_LABEL, _COL_BEGIN, _COL_END, _COL_LOW, _COL_HIGH, _COL_CONFIDENCE = range(6)
_RIGHT = Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter


def _freq(v: float | None) -> str:
    return "" if v is None else f"{v:.0f}"


class AnnotationTable(QTableWidget):
    def __init__(self, annotations: AnnotationSet, parent=None) -> None:
        super().__init__(0, len(_FIXED), parent)
        self._model = annotations
        self._keys: list[str] = []
        self._measurements: dict[str, dict[str, float]] = {}
        self._syncing = False  # guards against signal feedback loops

        self.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.verticalHeader().setDefaultSectionSize(26)
        self._apply_headers()

        self._model.added.connect(self._rebuild)
        self._model.removed.connect(self._rebuild)
        self._model.reset.connect(self._rebuild)
        self._model.changed.connect(self._rebuild)
        self._model.selectionChanged.connect(self._on_model_selection)
        self.itemSelectionChanged.connect(self._on_table_selection)
        self.itemChanged.connect(self._on_item_changed)

        self._rebuild()

    def delete_selected(self) -> None:
        if self._model.selected is not None:
            self._model.remove(self._model.selected)

    def edit_selected_label(self) -> None:
        """Start typing a label for the selected annotation."""
        rows = self.selectionModel().selectedRows()
        if rows:
            self.setFocus()
            self.editItem(self.item(rows[0].row(), _COL_LABEL))

    # --- measurement columns ----------------------------------------------
    def set_measurement_keys(self, keys: list[str] | tuple[str, ...]) -> None:
        """Choose which measurement columns follow the fixed ones."""
        self._keys = [k for k in keys if k in MEASUREMENTS_BY_KEY]
        self._apply_headers()
        self._rebuild()

    def set_measurements(self, measurements: dict[str, dict[str, float]]) -> None:
        """Show computed values, keyed by annotation id then measurement key."""
        self._measurements = measurements
        self._rebuild()

    def _apply_headers(self) -> None:
        specs = [MEASUREMENTS_BY_KEY[k] for k in self._keys]
        self.setColumnCount(len(_FIXED) + len(specs))
        self.setHorizontalHeaderLabels(_FIXED + [spec.header for spec in specs])
        header = self.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
        header.setStretchLastSection(True)
        self.setColumnWidth(_COL_LABEL, 170)
        metrics = header.fontMetrics()
        for col in range(1, self.columnCount()):
            text = self.horizontalHeaderItem(col).text()
            self.setColumnWidth(col, metrics.horizontalAdvance(text) + 32)
        for col, spec in enumerate(specs, start=len(_FIXED)):
            self.horizontalHeaderItem(col).setToolTip(spec.description)

    # --- model -> table ---------------------------------------------------
    def _rebuild(self, *_: object) -> None:
        self._syncing = True
        try:
            items = self._model.items()
            self.setRowCount(len(items))
            for row, ann in enumerate(items):
                self._set_row(row, ann)
            self._on_model_selection(self._model.selected)
        finally:
            self._syncing = False

    def _set_row(self, row: int, ann: Annotation) -> None:
        fixed = [
            ann.label,
            f"{ann.start_time:.3f}",
            f"{ann.end_time:.3f}",
            _freq(ann.low_freq),
            _freq(ann.high_freq),
            "" if ann.confidence is None else f"{ann.confidence:.2f}",
        ]
        for col, text in enumerate(fixed):
            item = QTableWidgetItem(text)
            if col == _COL_LABEL:
                item.setData(Qt.ItemDataRole.UserRole, ann.id)  # label cell carries the id
                item.setForeground(QBrush(label_color(ann.label)))
            else:
                item.setTextAlignment(_RIGHT)
            if col == _COL_CONFIDENCE:
                item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self.setItem(row, col, item)
        values = self._measurements.get(ann.id, {})
        for col, key in enumerate(self._keys, start=len(_FIXED)):
            item = QTableWidgetItem(MEASUREMENTS_BY_KEY[key].format(values.get(key)))
            item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            item.setTextAlignment(_RIGHT)
            self.setItem(row, col, item)

    def _on_model_selection(self, ann: Annotation | None) -> None:
        self._syncing = True
        try:
            if ann is None:
                self.clearSelection()
            else:
                for row in range(self.rowCount()):
                    if self.item(row, 0).data(Qt.ItemDataRole.UserRole) == ann.id:
                        self.selectRow(row)
                        self.scrollToItem(self.item(row, 0))
                        break
        finally:
            self._syncing = False

    # --- table -> model ---------------------------------------------------
    def _on_table_selection(self) -> None:
        if self._syncing:
            return
        rows = self.selectionModel().selectedRows()
        self._model.select(self._find(rows[0].row()) if rows else None)

    def _on_item_changed(self, item: QTableWidgetItem) -> None:
        if self._syncing or item.column() >= _COL_CONFIDENCE:
            return
        ann = self._find(item.row())
        if ann is None:
            return
        if item.column() == _COL_LABEL:
            ann.label = item.text().strip()
        elif not self._apply_number(ann, item.column(), item.text()):
            self._rebuild()  # not a usable number: put the old value back
            return
        self._model.update(ann)

    @staticmethod
    def _apply_number(ann: Annotation, column: int, text: str) -> bool:
        """Write a typed bound into ``ann``; ``False`` if it isn't acceptable."""
        text = text.strip()
        if column in (_COL_LOW, _COL_HIGH) and not text:
            ann.low_freq = ann.high_freq = None  # blank = a time-only annotation
            return True
        try:
            value = float(text)
        except ValueError:
            return False
        if value < 0:
            return False
        if column == _COL_BEGIN and value < ann.end_time:
            ann.start_time = value
        elif column == _COL_END and value > ann.start_time:
            ann.end_time = value
        elif column == _COL_LOW and (ann.high_freq is None or value < ann.high_freq):
            ann.low_freq = value
            ann.high_freq = ann.high_freq if ann.high_freq is not None else value
        elif column == _COL_HIGH and (ann.low_freq is None or value > ann.low_freq):
            ann.high_freq = value
            ann.low_freq = ann.low_freq if ann.low_freq is not None else 0.0
        else:
            return False
        return True

    def _find(self, row: int) -> Annotation | None:
        id_ = self.item(row, 0).data(Qt.ItemDataRole.UserRole)
        return next((a for a in self._model.items() if a.id == id_), None)
