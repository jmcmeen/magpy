"""
TransportBar -- play/pause/stop controls, a seek slider, a time readout, and the
playback options (speed, loop).

A dumb view: it emits user intent (``playPauseRequested``, ``stopRequested``,
``seekRequested``, ``playSelectionRequested``, ``speedChanged``, ``loopToggled``)
and exposes setters the shell calls in response to controller signals. It holds
no playback state and imports no bioamla.
"""

from __future__ import annotations

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSlider,
    QWidget,
)

from magpy.services import SPEEDS

from ._plot import format_time

_SLIDER_MAX = 1000  # slider works in integer ticks; we map to seconds


def _fmt(seconds: float) -> str:
    # Always m:ss.t here (even under a minute) so the readout doesn't jump width.
    m, s = divmod(max(0.0, seconds), 60)
    return f"{int(m):d}:{s:04.1f}" if m < 60 else format_time(seconds, decimals=1)


class TransportBar(QWidget):
    playPauseRequested = pyqtSignal()
    stopRequested = pyqtSignal()
    seekRequested = pyqtSignal(float)  # seconds
    playSelectionRequested = pyqtSignal()
    speedChanged = pyqtSignal(float)
    loopToggled = pyqtSignal(bool)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._duration = 0.0
        self._scrubbing = False

        self._play_btn = QPushButton("Play")
        self._play_btn.setToolTip("Play / pause (Space)")
        self._stop_btn = QPushButton("Stop")
        self._selection_btn = QPushButton("Play selection")
        self._selection_btn.setToolTip("Play the selected annotation (Shift+Space)")
        self._loop_box = QCheckBox("Loop")
        self._loop_box.setToolTip("Repeat the selection until paused")
        self._slider = QSlider(Qt.Orientation.Horizontal)
        self._slider.setRange(0, _SLIDER_MAX)
        self._speed_combo = QComboBox()
        for speed in SPEEDS:
            self._speed_combo.addItem(f"{speed:g}×", speed)
        self._speed_combo.setCurrentIndex(SPEEDS.index(1.0))
        self._speed_combo.setToolTip(
            "Playback speed. Slower also lowers the pitch, which brings fast or\n"
            "high-pitched calls into hearing range."
        )
        self._time_label = QLabel()
        self._time_label.setMinimumWidth(120)
        self._time_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(4, 2, 4, 2)
        layout.addWidget(self._play_btn)
        layout.addWidget(self._stop_btn)
        layout.addWidget(self._selection_btn)
        layout.addWidget(self._loop_box)
        layout.addWidget(self._slider, stretch=1)
        layout.addWidget(QLabel("Speed"))
        layout.addWidget(self._speed_combo)
        layout.addWidget(self._time_label)

        self._play_btn.clicked.connect(self.playPauseRequested)
        self._stop_btn.clicked.connect(self.stopRequested)
        self._selection_btn.clicked.connect(self.playSelectionRequested)
        self._loop_box.toggled.connect(self.loopToggled)
        self._speed_combo.currentIndexChanged.connect(
            lambda _i: self.speedChanged.emit(float(self._speed_combo.currentData()))
        )
        # Track scrubbing so position updates don't fight the user's drag.
        self._slider.sliderPressed.connect(lambda: setattr(self, "_scrubbing", True))
        self._slider.sliderReleased.connect(self._on_slider_released)

        self._has_selection = False
        self.set_enabled(False)
        self._update_label(0.0)

    def set_enabled(self, enabled: bool) -> None:
        for w in (self._play_btn, self._stop_btn, self._slider, self._speed_combo, self._loop_box):
            w.setEnabled(enabled)
        self._selection_btn.setEnabled(enabled and self._has_selection)

    def set_selection_available(self, available: bool) -> None:
        """Enable "Play selection" only while an annotation is selected."""
        self._has_selection = available
        self._selection_btn.setEnabled(available and self._duration > 0)

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
