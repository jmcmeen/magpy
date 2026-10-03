"""Tests for the selection table: in-place editing and measurement columns."""

from __future__ import annotations

import pytest
from PyQt6.QtWidgets import QApplication

from magpy.models import AnnotationSet
from magpy.services import Annotation
from magpy.widgets import AnnotationTable


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


def _table(qapp):
    model = AnnotationSet()
    ann = Annotation(2.0, 4.0, 1000.0, 3000.0, label="wren")
    model.add(ann)
    return model, ann, AnnotationTable(model)


def _headers(table) -> list[str]:
    return [table.horizontalHeaderItem(c).text() for c in range(table.columnCount())]


def test_typing_bounds_edits_the_annotation_and_is_undoable(qapp):
    model, ann, table = _table(qapp)
    table.item(0, 1).setText("2.5")  # Begin
    table.item(0, 4).setText("3500")  # High
    assert (ann.start_time, ann.high_freq) == (2.5, 3500.0)
    model.undo()
    model.undo()
    assert (model.items()[0].start_time, model.items()[0].high_freq) == (2.0, 3000.0)


@pytest.mark.parametrize(
    ("column", "text"),
    [(1, "9.0"), (2, "1.0"), (3, "5000"), (4, "500"), (1, "abc"), (2, "-1")],
)
def test_unusable_bounds_are_rejected_and_the_cell_reverts(qapp, column, text):
    _model, ann, table = _table(qapp)
    before = table.item(0, column).text()
    table.item(0, column).setText(text)
    assert (ann.start_time, ann.end_time, ann.low_freq, ann.high_freq) == (2.0, 4.0, 1000.0, 3000.0)
    assert table.item(0, column).text() == before


def test_blanking_a_frequency_makes_it_time_only(qapp):
    _model, ann, table = _table(qapp)
    table.item(0, 3).setText("")
    assert ann.low_freq is None and ann.high_freq is None


def test_label_edit_updates_the_model(qapp):
    _model, ann, table = _table(qapp)
    table.item(0, 0).setText("  robin ")
    assert ann.label == "robin"


def test_measurement_columns_follow_the_chosen_keys(qapp):
    _model, ann, table = _table(qapp)
    table.set_measurement_keys(["peak_frequency", "rms_db", "not_a_metric"])
    assert _headers(table)[6:] == ["Peak freq (Hz)", "RMS level (dBFS)"]

    assert table.item(0, 6).text() == ""  # not measured yet
    table.set_measurements({ann.id: {"peak_frequency": 2040.4, "rms_db": -17.26}})
    assert table.item(0, 6).text() == "2040"
    assert table.item(0, 7).text() == "-17.3"

    table.set_measurement_keys([])
    assert table.columnCount() == 6


def test_confidence_column_is_shown_and_read_only(qapp):
    from PyQt6.QtCore import Qt

    model, ann, table = _table(qapp)
    assert table.item(0, 5).text() == ""
    ann.confidence = 0.876
    model.update(ann)
    assert table.item(0, 5).text() == "0.88"
    assert not table.item(0, 5).flags() & Qt.ItemFlag.ItemIsEditable
