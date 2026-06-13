"""
CandidateSet -- the observable *candidate* layer produced by detectors.

Detections are deliberately kept separate from curated annotations
(:class:`~magpy.models.AnnotationSet`): a detector run fills this set, the user
reviews/filters it, and promotes the keepers into the annotation set. Keeping
them apart preserves the "reviewed vs. machine-suggested" distinction and means
re-running a detector replaces candidates without touching curated work.

Holds MagPy :class:`~magpy.services.Candidate` records (frozen DTOs) plus the
current selection and a confidence filter threshold. Views bind to its signals;
nothing here imports bioamla.
"""

from __future__ import annotations

from typing import Optional

from PyQt6.QtCore import QObject, pyqtSignal

from magpy.services import Candidate


class CandidateSet(QObject):
    reset = pyqtSignal()  # the whole set was replaced (new detector run / cleared)
    selectionChanged = pyqtSignal(object)  # Candidate | None
    thresholdChanged = pyqtSignal(float)  # confidence filter moved

    def __init__(self, parent: Optional[QObject] = None) -> None:
        super().__init__(parent)
        self._items: list[Candidate] = []
        self._selected: Optional[Candidate] = None
        self._threshold: float = 0.0
        self._detector: str = ""

    def items(self) -> list[Candidate]:
        """All candidates, ordered by start time (unfiltered)."""
        return sorted(self._items, key=lambda c: c.start_time)

    def visible_items(self) -> list[Candidate]:
        """Candidates at or above the current confidence threshold."""
        return [c for c in self.items() if c.confidence >= self._threshold]

    def __len__(self) -> int:
        return len(self._items)

    @property
    def selected(self) -> Optional[Candidate]:
        return self._selected

    @property
    def threshold(self) -> float:
        return self._threshold

    @property
    def detector(self) -> str:
        """Label of the detector that produced the current set (for display)."""
        return self._detector

    def set_all(self, candidates: list[Candidate], detector: str = "") -> None:
        """Replace the whole set (a new detector run, or a clear)."""
        self._items = list(candidates)
        self._detector = detector
        self._selected = None
        self.reset.emit()
        self.selectionChanged.emit(None)

    def clear(self) -> None:
        self.set_all([])

    def select(self, candidate: Optional[Candidate]) -> None:
        if candidate is not self._selected:
            self._selected = candidate
            self.selectionChanged.emit(candidate)

    def set_threshold(self, threshold: float) -> None:
        if threshold != self._threshold:
            self._threshold = threshold
            self.thresholdChanged.emit(threshold)
