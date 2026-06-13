"""
Document -- the mutable application state that bioamla (being functional) cannot
hold for us.

A Document represents "the audio the user is currently working on": the loaded
samples plus its :class:`AnnotationSet`. It is a ``QObject`` so views can bind to
its signals and refresh themselves -- the "model" half of MagPy's MVVM layering.
Views read state through it and never reach into the services layer directly.
"""

from __future__ import annotations

from typing import Optional

from PyQt6.QtCore import QObject, pyqtSignal

from magpy.models.annotations import AnnotationSet
from magpy.services import LoadedAudio


class Document(QObject):
    """Holds the currently open audio and its annotations."""

    audioChanged = pyqtSignal(object)  # LoadedAudio | None

    def __init__(self, parent: Optional[QObject] = None) -> None:
        super().__init__(parent)
        self._audio: Optional[LoadedAudio] = None
        self._annotations = AnnotationSet(self)

    @property
    def audio(self) -> Optional[LoadedAudio]:
        """The currently loaded audio, or ``None`` if nothing is open."""
        return self._audio

    @property
    def annotations(self) -> AnnotationSet:
        """The annotations for the open audio."""
        return self._annotations

    def set_audio(self, audio: Optional[LoadedAudio]) -> None:
        """Replace the current audio (and clear annotations) and notify observers."""
        self._audio = audio
        self._annotations.clear()
        self.audioChanged.emit(audio)
