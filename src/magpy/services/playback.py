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


class Player:
    """Wraps a single ``AudioPlayer``. Positions and durations are in seconds."""

    def __init__(self) -> None:
        self._player = AudioPlayer()
        self._loaded = False

    def load(self, samples: np.ndarray, sample_rate: int) -> None:
        """Load mono/stereo samples for playback (stops any current playback)."""
        self._player.load(samples, sample_rate)
        self._loaded = True

    def play(self) -> None:
        """Start or resume playback. Opens an audio output stream (non-blocking)."""
        if self._loaded:
            self._player.play()

    def pause(self) -> None:
        self._player.pause()

    def stop(self) -> None:
        self._player.stop()

    def seek(self, seconds: float) -> None:
        """Seek to ``seconds`` from the start."""
        if self._loaded:
            self._player.seek(float(seconds))

    @property
    def position(self) -> float:
        """Current playback position in seconds (0.0 when nothing is loaded)."""
        if not self._loaded:
            return 0.0
        return float(self._player.position.current_time)

    @property
    def duration(self) -> float:
        """Total duration in seconds (0.0 when nothing is loaded)."""
        if not self._loaded:
            return 0.0
        return float(self._player.duration)

    @property
    def state(self) -> PlaybackState:
        """Current playback state."""
        if not self._loaded:
            return PlaybackState.STOPPED
        return _STATE_MAP[self._player.state]
