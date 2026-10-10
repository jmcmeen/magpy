"""
Playback service -- thin wrapper over ``bioamla.audio.AudioPlayer``.

Playback is the one inherently *stateful* bioamla engine MagPy uses, so unlike
the other services this module exposes a small stateful class rather than pure
functions. It still serves the seam's purpose: the bioamla import and types stay
here, and callers get a MagPy-facing API (``PlaybackState``, positions in
seconds) with no bioamla types leaking out.

It is deliberately **not** Qt-aware. The Qt layer (``models.PlaybackController``)
owns a ``QTimer`` that polls :meth:`Player.position` on the UI thread; we do not
use ``AudioPlayer``'s ``on_position_change`` callback because that fires on
sounddevice's audio thread, from which touching Qt objects is unsafe.
"""

from __future__ import annotations

from enum import Enum

import numpy as np

# bioamla import is confined to the services layer.
from bioamla.audio import AudioPlayer
from bioamla.audio import PlaybackState as _BioPlaybackState


class PlaybackState(Enum):
    """MagPy-owned playback state (mirrors bioamla's, kept off the seam)."""

    STOPPED = "stopped"
    PLAYING = "playing"
    PAUSED = "paused"


_STATE_MAP = {
    _BioPlaybackState.STOPPED: PlaybackState.STOPPED,
    _BioPlaybackState.PLAYING: PlaybackState.PLAYING,
    _BioPlaybackState.PAUSED: PlaybackState.PAUSED,
}


SPEEDS = (0.1, 0.25, 0.5, 1.0, 2.0)


class Player:
    """Wraps a single ``AudioPlayer``. Positions and durations are in seconds.

    Playback **speed** works the way a tape deck does: the samples are sent to
    the output at ``speed`` times their true rate, so slowing down also lowers
    the pitch. That is the behaviour wanted for listening to fast or
    high-pitched calls, and it needs no resynthesis. Positions stay in
    recording time (seconds into the file) whatever the speed.
    """

    def __init__(self) -> None:
        self._player = AudioPlayer()
        self._samples: np.ndarray | None = None
        self._sample_rate = 0
        self._speed = 1.0

    @property
    def _loaded(self) -> bool:
        return self._samples is not None

    def load(self, samples: np.ndarray, sample_rate: int) -> None:
        """Load mono/stereo samples for playback (stops any current playback)."""
        self._samples = samples
        self._sample_rate = sample_rate
        self._player.load(samples, self._output_rate())

    def _output_rate(self) -> int:
        return max(1, round(self._sample_rate * self._speed))

    @property
    def speed(self) -> float:
        return self._speed

    def set_speed(self, speed: float) -> None:
        """Change the playback speed, keeping the position and play state."""
        if speed <= 0 or speed == self._speed:
            return
        self._speed = speed
        if self._samples is None:
            return
        position, playing = self.position, self.state == PlaybackState.PLAYING
        self._player.load(self._samples, self._output_rate())  # stops playback
        self.seek(position)
        if playing:
            self.play()

    def play(self) -> None:
        """Start or resume playback. Opens an audio output stream (non-blocking)."""
        if self._loaded:
            self._player.play()

    def pause(self) -> None:
        self._player.pause()

    def stop(self) -> None:
        self._player.stop()

    def seek(self, seconds: float) -> None:
        """Seek to ``seconds`` from the start of the recording."""
        if self._loaded:
            self._player.seek(int(float(seconds) * self._sample_rate), by_sample=True)

    @property
    def position(self) -> float:
        """Current position in seconds of recording time (0.0 when nothing is loaded)."""
        if not self._loaded or not self._sample_rate:
            return 0.0
        return self._player.position.current_sample / self._sample_rate

    @property
    def duration(self) -> float:
        """Total duration in seconds of recording time (0.0 when nothing is loaded)."""
        if not self._loaded or not self._sample_rate:
            return 0.0
        return self._player.position.total_samples / self._sample_rate

    @property
    def state(self) -> PlaybackState:
        """Current playback state."""
        if not self._loaded:
            return PlaybackState.STOPPED
        return _STATE_MAP[self._player.state]
