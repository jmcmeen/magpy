"""
PlaybackController -- the Qt-aware view-model over the playback service.

Owns a :class:`~magpy.services.Player` and a ``QTimer`` that polls the player's
position/state on the UI thread while playing, translating them into Qt signals
the transport bar and spectrogram playhead bind to. This is where the audio
engine meets the event loop; it imports the services seam, never bioamla.
"""

from __future__ import annotations

from typing import Optional

from PyQt6.QtCore import QObject, QTimer, pyqtSignal

from magpy.services import LoadedAudio, PlaybackState, Player

_POLL_MS = 33  # ~30 Hz playhead refresh; independent of the audio buffer rate


class PlaybackController(QObject):
    positionChanged = pyqtSignal(float)  # seconds
    durationChanged = pyqtSignal(float)  # seconds
    stateChanged = pyqtSignal(object)  # PlaybackState

    def __init__(self, parent: Optional[QObject] = None) -> None:
        super().__init__(parent)
        self._player = Player()
        self._state = PlaybackState.STOPPED
        self._timer = QTimer(self)
        self._timer.setInterval(_POLL_MS)
        self._timer.timeout.connect(self._tick)

    def set_audio(self, audio: Optional[LoadedAudio]) -> None:
        """Load (or clear) the audio to play and reset transport state."""
        self.stop()
        if audio is None:
            self.durationChanged.emit(0.0)
            self.positionChanged.emit(0.0)
            return
        self._player.load(audio.samples, audio.sample_rate)
        self.durationChanged.emit(self._player.duration)
        self.positionChanged.emit(0.0)

    def play(self) -> None:
        self._player.play()
        self._timer.start()
        self._set_state(self._player.state)

    def pause(self) -> None:
        self._player.pause()
        self._timer.stop()
        self.positionChanged.emit(self._player.position)
        self._set_state(self._player.state)

    def stop(self) -> None:
        self._player.stop()
        self._timer.stop()
        self.positionChanged.emit(0.0)
        self._set_state(PlaybackState.STOPPED)

    def toggle(self) -> None:
        """Play if stopped/paused, pause if playing."""
        if self._state == PlaybackState.PLAYING:
            self.pause()
        else:
            self.play()

    def seek(self, seconds: float) -> None:
        self._player.seek(seconds)
        self.positionChanged.emit(self._player.position)

    @property
    def state(self) -> PlaybackState:
        return self._state

    def _tick(self) -> None:
        self.positionChanged.emit(self._player.position)
        # The player auto-stops when it reaches the end; reflect that in the UI.
        state = self._player.state
        if state != PlaybackState.PLAYING:
            self._timer.stop()
            self.positionChanged.emit(0.0 if state == PlaybackState.STOPPED else self._player.position)
            self._set_state(state)

    def _set_state(self, state: PlaybackState) -> None:
        if state != self._state:
            self._state = state
            self.stateChanged.emit(state)
