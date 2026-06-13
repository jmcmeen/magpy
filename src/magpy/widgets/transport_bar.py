"""
TransportBar -- play/pause/stop controls, a seek slider, and a time readout.

A dumb view: it emits user intent (``playPauseRequested``, ``stopRequested``,
``seekRequested``) and exposes setters the shell calls in response to controller
signals. It holds no playback state and imports no bioamla.
"""

from __future__ import annotations

from typing import Optional

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import QHBoxLayout, QLabel, QPushButton, QSlider, QWidget

_SLIDER_MAX = 1000  # slider works in integer ticks; we map to seconds


def _fmt(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    return f"{m:d}:{s:02d}"


class TransportBar(QWidget):
    playPauseRequested = pyqtSignal()
    stopRequested = pyqtSignal()
    seekRequested = pyqtSignal(float)  # seconds

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._duration = 0.0
        self._scrubbing = False

        self._play_btn = QPushButton("Play")
        self._stop_btn = QPushButton("Stop")
        self._slider = QSlider(Qt.Orientation.Horizontal)
        self._slider.setRange(0, _SLIDER_MAX)
        self._time_label = QLabel("0:00 / 0:00")

        layout = QHBoxLayout(self)
        layout.addWidget(self._play_btn)
        layout.addWidget(self._stop_btn)
        layout.addWidget(self._slider, stretch=1)
        layout.addWidget(self._time_label)

        self._play_btn.clicked.connect(self.playPauseRequested)
        self._stop_btn.clicked.connect(self.stopRequested)
        # Track scrubbing so position updates don't fight the user's drag.
        self._slider.sliderPressed.connect(lambda: setattr(self, "_scrubbing", True))
        self._slider.sliderReleased.connect(self._on_slider_released)

        self.set_enabled(False)

    def set_enabled(self, enabled: bool) -> None:
        for w in (self._play_btn, self._stop_btn, self._slider):
            w.setEnabled(enabled)

    def set_duration(self, seconds: float) -> None:
        self._duration = seconds
        self.set_enabled(seconds > 0)
        self._update_label(0.0)

    def set_position(self, seconds: float) -> None:
        if not self._scrubbing and self._duration > 0:
            self._slider.setValue(int(seconds / self._duration * _SLIDER_MAX))
        self._update_label(seconds)

    def set_playing(self, playing: bool) -> None:
        self._play_btn.setText("Pause" if playing else "Play")

    def _on_slider_released(self) -> None:
        self._scrubbing = False
        if self._duration > 0:
            self.seekRequested.emit(self._slider.value() / _SLIDER_MAX * self._duration)

    def _update_label(self, position: float) -> None:
        self._time_label.setText(f"{_fmt(position)} / {_fmt(self._duration)}")
