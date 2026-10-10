"""Tests for PlaybackController's range / loop / speed logic.

The audio engine is replaced with a fake so nothing is sent to a sound device;
the real ``Player`` speed arithmetic is checked against a stub ``AudioPlayer``.
"""

from __future__ import annotations

import numpy as np
import pytest
from PyQt6.QtWidgets import QApplication

from magpy.models import PlaybackController
from magpy.services import PlaybackState, Player


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


class FakePlayer:
    """Stands in for services.Player: a position the test advances by hand."""

    def __init__(self) -> None:
        self.position = 0.0
        self.duration = 60.0
        self.state = PlaybackState.STOPPED
        self.speed = 1.0

    def play(self) -> None:
        self.state = PlaybackState.PLAYING

    def pause(self) -> None:
        self.state = PlaybackState.PAUSED

    def stop(self) -> None:
        self.state = PlaybackState.STOPPED
        self.position = 0.0

    def seek(self, seconds: float) -> None:
        self.position = seconds

    def set_speed(self, speed: float) -> None:
        self.speed = speed


def _controller(qapp) -> tuple[PlaybackController, FakePlayer]:
    controller = PlaybackController()
    fake = FakePlayer()
    controller._player = fake
    return controller, fake


def test_play_range_stops_at_the_end_and_rewinds_to_its_start(qapp):
    controller, fake = _controller(qapp)
    positions = []
    controller.positionChanged.connect(positions.append)

    controller.play_range(10.0, 12.0)
    assert fake.position == 10.0 and controller.state == PlaybackState.PLAYING

    fake.position = 11.0
    controller._tick()
    assert controller.state == PlaybackState.PLAYING

    fake.position = 12.01
    controller._tick()
    assert controller.state == PlaybackState.PAUSED
    assert fake.position == 10.0 and positions[-1] == 10.0


def test_looping_replays_the_range(qapp):
    controller, fake = _controller(qapp)
    controller.set_loop(True)
    controller.play_range(10.0, 12.0)
    fake.position = 12.5
    controller._tick()
    assert controller.state == PlaybackState.PLAYING and fake.position == 10.0


def test_plain_play_ignores_a_finished_range(qapp):
    controller, fake = _controller(qapp)
    controller.play_range(10.0, 12.0)
    fake.position = 12.5
    controller._tick()  # range done
    controller.toggle()  # Play again: continues, unbounded
    fake.position = 30.0
    controller._tick()
    assert controller.state == PlaybackState.PLAYING


def test_seek_leaves_bounded_playback(qapp):
    controller, fake = _controller(qapp)
    controller.play_range(10.0, 12.0)
    controller.seek(40.0)
    fake.position = 41.0
    controller._tick()
    assert controller.state == PlaybackState.PLAYING and fake.position == 41.0


def test_empty_range_is_ignored(qapp):
    controller, fake = _controller(qapp)
    controller.play_range(5.0, 5.0)
    assert controller.state == PlaybackState.STOPPED


class _StubEngine:
    """The slice of bioamla's AudioPlayer that services.Player relies on."""

    class _Position:
        def __init__(self, sample: int, total: int) -> None:
            self.current_sample = sample
            self.total_samples = total

    def __init__(self) -> None:
        self.loaded_rate = 0
        self.sample = 0
        self.total = 0

    def load(self, samples, sample_rate) -> None:
        self.loaded_rate = sample_rate
        self.total = len(samples)
        self.sample = 0

    def seek(self, value, by_sample=False) -> None:
        assert by_sample
        self.sample = int(value)

    @property
    def position(self):
        return self._Position(self.sample, self.total)

    @property
    def state(self):
        from bioamla.audio import PlaybackState as BioState

        return BioState.STOPPED


def test_speed_changes_the_output_rate_but_not_recording_time():
    player = Player()
    engine = _StubEngine()
    player._player = engine
    player.load(np.zeros(48000 * 10, dtype=np.float32), 48000)
    player.seek(4.0)

    player.set_speed(0.25)
    assert engine.loaded_rate == 12000  # quarter speed = quarter output rate
    assert player.position == pytest.approx(4.0)  # still 4 s into the recording
    assert player.duration == pytest.approx(10.0)
