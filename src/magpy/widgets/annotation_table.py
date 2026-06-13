"""
AnnotationTable -- a table view bound to an :class:`AnnotationSet`.

Lists annotations (label editable in place), keeps its selection in sync with
the model both ways, and offers a Delete action for the selected row. It's a
view: it observes the model's signals and calls back into it, holding no
annotation state of its own. No bioamla imports.
"""

from __future__ import annotations

from typing import Optional

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QHeaderView,
    QTableWidget,
    QTableWidgetItem,
)

from magpy.models import AnnotationSet
from magpy.services import Annotation

_COLUMNS = ["Label", "Start (s)", "End (s)", "Low (Hz)", "High (Hz)"]


def _freq(v: Optional[float]) -> str:
    return "" if v is None else f"{v:.0f}"


class AnnotationTable(QTableWidget):
    def __init__(self, annotations: AnnotationSet, parent=None) -> None:
        super().__init__(0, len(_COLUMNS), parent)
        self._model = annotations
        self._syncing = False  # guards against signal feedback loops

        self.setHorizontalHeaderLabels(_COLUMNS)
        self.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.verticalHeader().setVisible(False)
        self.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)

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
        values = [ann.label, f"{ann.start_time:.3f}", f"{ann.end_time:.3f}",
                  _freq(ann.low_freq), _freq(ann.high_freq)]
        for col, text in enumerate(values):
            item = QTableWidgetItem(text)
            if col == 0:
                item.setData(Qt.ItemDataRole.UserRole, ann.id)  # label cell carries the id
            else:
                item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self.setItem(row, col, item)

    def _on_model_selection(self, ann: Optional[Annotation]) -> None:
        self._syncing = True
        try:
            if ann is None:
                self.clearSelection()
            else:
                for row in range(self.rowCount()):
                    if self.item(row, 0).data(Qt.ItemDataRole.UserRole) == ann.id:
                        self.selectRow(row)
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
        if self._syncing or item.column() != 0:
            return
        ann = self._find(item.row())
        if ann is not None:
            ann.label = item.text()
            self._model.update(ann)

    def _find(self, row: int) -> Optional[Annotation]:
        id_ = self.item(row, 0).data(Qt.ItemDataRole.UserRole)
        return next((a for a in self._model.items() if a.id == id_), None)
