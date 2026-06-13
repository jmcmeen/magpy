"""
AnnotationSet -- the observable collection of annotations for the open document.

Holds MagPy :class:`~magpy.services.Annotation` records and the current
selection (which annotation is highlighted/being edited -- the transient UI
state bioamla's data type doesn't carry). Views bind to its signals; nothing
here imports bioamla.
"""

from __future__ import annotations

from typing import Optional

from PyQt6.QtCore import QObject, pyqtSignal

from magpy.services import Annotation


class AnnotationSet(QObject):
    added = pyqtSignal(object)  # Annotation
    removed = pyqtSignal(object)  # Annotation
    changed = pyqtSignal(object)  # Annotation (fields mutated in place)
    reset = pyqtSignal()  # bulk replacement (e.g. a table was loaded/cleared)
    selectionChanged = pyqtSignal(object)  # Annotation | None

    def __init__(self, parent: Optional[QObject] = None) -> None:
        super().__init__(parent)
        self._items: list[Annotation] = []
        self._selected: Optional[Annotation] = None

    def items(self) -> list[Annotation]:
        """A copy of the annotations, ordered by start time."""
        return sorted(self._items, key=lambda a: a.start_time)

    def __len__(self) -> int:
        return len(self._items)

    @property
    def selected(self) -> Optional[Annotation]:
        return self._selected

    def add(self, annotation: Annotation) -> None:
        self._items.append(annotation)
        self.added.emit(annotation)
        self.select(annotation)

    def remove(self, annotation: Annotation) -> None:
        if annotation not in self._items:
            return
        self._items.remove(annotation)
        if self._selected is annotation:
            self.select(None)
        self.removed.emit(annotation)

    def update(self, annotation: Annotation) -> None:
        """Notify that ``annotation``'s fields were mutated in place."""
        if annotation in self._items:
            self.changed.emit(annotation)

    def set_all(self, annotations: list[Annotation]) -> None:
        """Replace the whole set (used when loading a selection table)."""
        self._items = list(annotations)
        self._selected = None
        self.reset.emit()
        self.selectionChanged.emit(None)

    def clear(self) -> None:
        self.set_all([])

    def select(self, annotation: Optional[Annotation]) -> None:
        if annotation is not self._selected:
            self._selected = annotation
            self.selectionChanged.emit(annotation)
