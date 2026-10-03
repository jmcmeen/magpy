"""
PlaybackController -- the Qt-aware view-model over the playback service.

Owns a :class:`~magpy.services.Player` and a ``QTimer`` that polls the player's
position/state on the UI thread while playing, translating them into Qt signals
the transport bar and spectrogram playhead bind to. This is where the audio
engine meets the event loop; it imports the services seam, never bioamla.
"""

from __future__ import annotations

from PyQt6.QtCore import QObject, QTimer, pyqtSignal

from magpy.services import LoadedAudio, PlaybackState, Player

_POLL_MS = 33  # ~30 Hz playhead refresh; independent of the audio buffer rate


class PlaybackController(QObject):
    positionChanged = pyqtSignal(float)  # seconds
    durationChanged = pyqtSignal(float)  # seconds
    stateChanged = pyqtSignal(object)  # PlaybackState

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._player = Player()
        self._state = PlaybackState.STOPPED
        self._range: tuple[float, float] | None = None  # bounded playback, if any
        self._loop = False
        self._timer = QTimer(self)
        self._timer.setInterval(_POLL_MS)
        self._timer.timeout.connect(self._tick)

    def set_audio(self, audio: LoadedAudio | None) -> None:
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
        """Play from the current position to the end of the recording."""
        self._range = None
        self._start()

    def play_range(self, start: float, end: float) -> None:
        """Play just ``start``..``end`` seconds (repeating it while looping is on)."""
        if end <= start:
            return
        self._player.pause()
        self._range = (start, end)
        self._player.seek(start)
        self.positionChanged.emit(start)
        self._start()

    def _start(self) -> None:
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
        self._range = None
        self.positionChanged.emit(0.0)
        self._set_state(PlaybackState.STOPPED)

    def toggle(self) -> None:
        """Play if stopped/paused, pause if playing."""
        if self._state == PlaybackState.PLAYING:
            self.pause()
        else:
            self._start()  # resume, keeping any range still in force

    def seek(self, seconds: float) -> None:
        self._range = None  # a manual seek leaves bounded playback
        self._player.seek(seconds)
        self.positionChanged.emit(self._player.position)

    def set_loop(self, loop: bool) -> None:
        """Whether :meth:`play_range` repeats its range until paused."""
        self._loop = loop

    def set_speed(self, speed: float) -> None:
        """Playback speed multiplier (pitch follows speed, like tape)."""
        self._player.set_speed(speed)
        self._set_state(self._player.state)

    @property
    def state(self) -> PlaybackState:
        return self._state

    @property
    def position(self) -> float:
        """Current position in seconds."""
        return self._player.position

    def _tick(self) -> None:
        position = self._player.position
        if self._range is not None and position >= self._range[1]:
            start = self._range[0]
            self._player.pause()
            self._player.seek(start)
            self.positionChanged.emit(start)
            if self._loop:
                self._player.play()
            else:
                self._range = None  # done: the next Play continues from here
                self._timer.stop()
                self._set_state(self._player.state)
            return
        self.positionChanged.emit(position)
        # The player auto-stops when it reaches the end; reflect that in the UI.
        state = self._player.state
        if state != PlaybackState.PLAYING:
            self._timer.stop()
            self.positionChanged.emit(0.0 if state == PlaybackState.STOPPED else position)
            self._set_state(state)

    def _set_state(self, state: PlaybackState) -> None:
        if state != self._state:
            self._state = state
            self.stateChanged.emit(state)
