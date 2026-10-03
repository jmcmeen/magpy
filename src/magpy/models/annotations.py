"""
AnnotationSet -- the observable collection of annotations for the open document.

Holds MagPy :class:`~magpy.services.Annotation` records and the current
selection (which annotation is highlighted/being edited -- the transient UI
state bioamla's data type doesn't carry). Views bind to its signals; nothing
here imports bioamla.

It also keeps the **undo history**. Annotations are edited in place (a view
mutates the record, then calls :meth:`update`), so there is no "before" to
capture at the moment of the edit. Instead the set keeps a copy of its last
published state; each change pushes that copy onto the undo stack and takes a
fresh one. Undo/redo then restore whole snapshots, which is simple, cannot drift
out of sync with the edits, and is cheap at the sizes a selection table reaches.
"""

from __future__ import annotations

from dataclasses import replace

from PyQt6.QtCore import QObject, pyqtSignal

from magpy.services import Annotation

_MAX_UNDO = 200


class AnnotationSet(QObject):
    added = pyqtSignal(object)  # Annotation
    removed = pyqtSignal(object)  # Annotation
    changed = pyqtSignal(object)  # Annotation (fields mutated in place)
    reset = pyqtSignal()  # bulk replacement (a table was loaded/cleared, undo/redo)
    selectionChanged = pyqtSignal(object)  # Annotation | None
    historyChanged = pyqtSignal()  # can_undo / can_redo may have changed

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._items: list[Annotation] = []
        self._selected: Annotation | None = None
        self._committed: list[Annotation] = []  # copy of the last published state
        self._undo: list[list[Annotation]] = []
        self._redo: list[list[Annotation]] = []

    def items(self) -> list[Annotation]:
        """A copy of the annotations, ordered by start time."""
        return sorted(self._items, key=lambda a: a.start_time)

    def labels(self) -> list[str]:
        """The distinct non-empty labels in use, sorted."""
        return sorted({a.label for a in self._items if a.label})

    def __len__(self) -> int:
        return len(self._items)

    @property
    def selected(self) -> Annotation | None:
        return self._selected

    def add(self, annotation: Annotation) -> None:
        self._items.append(annotation)
        self._record()
        self.added.emit(annotation)
        self.select(annotation)

    def remove(self, annotation: Annotation) -> None:
        if annotation not in self._items:
            return
        self._items.remove(annotation)
        self._record()
        if self._selected is annotation:
            self.select(None)
        self.removed.emit(annotation)

    def update(self, annotation: Annotation) -> None:
        """Notify that ``annotation``'s fields were mutated in place."""
        if annotation in self._items:
            self._record()
            self.changed.emit(annotation)

    def update_many(self, annotations: list[Annotation]) -> None:
        """Notify that several annotations were mutated in place, as one undo step."""
        if any(a in self._items for a in annotations):
            self._record()
            self.reset.emit()
            self.selectionChanged.emit(self._selected)

    def set_all(self, annotations: list[Annotation], *, undoable: bool = False) -> None:
        """Replace the whole set.

        By default this starts a fresh history (a different recording was
        opened). Pass ``undoable=True`` when it is an edit of the *current*
        recording's annotations, such as importing a selection table over them.
        """
        self._items = list(annotations)
        self._selected = None
        if undoable:
            self._record()
        else:
            self._committed = self._snapshot()
            self._undo.clear()
            self._redo.clear()
            self.historyChanged.emit()
        self.reset.emit()
        self.selectionChanged.emit(None)

    def clear(self) -> None:
        self.set_all([])

    def select(self, annotation: Annotation | None) -> None:
        if annotation is not self._selected:
            self._selected = annotation
            self.selectionChanged.emit(annotation)

    def select_adjacent(self, step: int) -> Annotation | None:
        """Select the next (``+1``) / previous (``-1``) annotation in time order."""
        items = self.items()
        if not items:
            return None
        if self._selected is None:
            target = items[0] if step > 0 else items[-1]
        else:
            index = next((i for i, a in enumerate(items) if a is self._selected), 0)
            target = items[min(max(0, index + step), len(items) - 1)]
        self.select(target)
        return target

    # --- undo / redo ------------------------------------------------------
    @property
    def can_undo(self) -> bool:
        return bool(self._undo)

    @property
    def can_redo(self) -> bool:
        return bool(self._redo)

    def undo(self) -> None:
        if self._undo:
            self._redo.append(self._committed)
            self._restore(self._undo.pop())

    def redo(self) -> None:
        if self._redo:
            self._undo.append(self._committed)
            self._restore(self._redo.pop())

    def _snapshot(self) -> list[Annotation]:
        return [replace(a) for a in self._items]

    def _record(self) -> None:
        """Push the previous state onto the undo stack (if anything changed)."""
        snapshot = self._snapshot()
        if snapshot == self._committed:
            return
        self._undo.append(self._committed)
        del self._undo[:-_MAX_UNDO]
        self._redo.clear()
        self._committed = snapshot
        self.historyChanged.emit()

    def _restore(self, state: list[Annotation]) -> None:
        selected_id = self._selected.id if self._selected is not None else None
        self._committed = state
        self._items = [replace(a) for a in state]
        self._selected = next((a for a in self._items if a.id == selected_id), None)
        self.historyChanged.emit()
        self.reset.emit()
        self.selectionChanged.emit(self._selected)
