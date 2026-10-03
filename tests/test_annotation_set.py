"""Tests for AnnotationSet: undo/redo history, labels, stepping."""

from __future__ import annotations

from magpy.models import AnnotationSet
from magpy.services import Annotation


def _bounds(model: AnnotationSet) -> list[tuple[float, float, str]]:
    return [(a.start_time, a.end_time, a.label) for a in model.items()]


def test_undo_redo_add_and_remove():
    model = AnnotationSet()
    a = Annotation(1.0, 2.0, label="a")
    model.add(a)
    model.add(Annotation(3.0, 4.0, label="b"))
    model.remove(a)
    assert _bounds(model) == [(3.0, 4.0, "b")]

    model.undo()  # un-remove
    assert _bounds(model) == [(1.0, 2.0, "a"), (3.0, 4.0, "b")]
    model.undo()  # un-add b
    assert _bounds(model) == [(1.0, 2.0, "a")]
    model.redo()
    assert _bounds(model) == [(1.0, 2.0, "a"), (3.0, 4.0, "b")]
    model.undo()
    model.undo()
    assert len(model) == 0 and not model.can_undo and model.can_redo


def test_undo_restores_an_in_place_edit():
    model = AnnotationSet()
    ann = Annotation(1.0, 2.0, 1000.0, 2000.0, label="old")
    model.add(ann)
    ann.label = "new"
    ann.end_time = 5.0
    model.update(ann)

    model.undo()
    assert _bounds(model) == [(1.0, 2.0, "old")]
    model.redo()
    assert _bounds(model) == [(1.0, 5.0, "new")]


def test_update_without_a_change_adds_no_history():
    model = AnnotationSet()
    ann = Annotation(1.0, 2.0)
    model.add(ann)
    model.update(ann)  # nothing actually changed
    model.undo()
    assert len(model) == 0


def test_a_new_edit_clears_redo():
    model = AnnotationSet()
    model.add(Annotation(1.0, 2.0))
    model.undo()
    model.add(Annotation(5.0, 6.0))
    assert not model.can_redo


def test_selection_survives_undo_by_id():
    model = AnnotationSet()
    ann = Annotation(1.0, 2.0, label="x")
    model.add(ann)  # add() selects
    ann.label = "y"
    model.update(ann)
    model.undo()
    assert model.selected is not None and model.selected.id == ann.id
    assert model.selected.label == "x"


def test_loading_a_file_starts_a_fresh_history_but_import_is_undoable():
    model = AnnotationSet()
    model.add(Annotation(1.0, 2.0))
    model.set_all([Annotation(7.0, 8.0)])  # opening another recording
    assert not model.can_undo

    model.set_all([Annotation(0.0, 1.0), Annotation(2.0, 3.0)], undoable=True)  # import
    assert len(model) == 2
    model.undo()
    assert _bounds(model) == [(7.0, 8.0, "")]


def test_labels_and_stepping():
    model = AnnotationSet()
    for start, label in ((5.0, "wren"), (1.0, "robin"), (3.0, ""), (7.0, "wren")):
        model.add(Annotation(start, start + 1, label=label))
    assert model.labels() == ["robin", "wren"]

    model.select(None)
    assert model.select_adjacent(+1).start_time == 1.0
    assert model.select_adjacent(+1).start_time == 3.0
    assert model.select_adjacent(-1).start_time == 1.0
    assert model.select_adjacent(-1).start_time == 1.0  # clamps at the ends
