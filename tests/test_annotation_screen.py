"""End-to-end tests for the annotation screen: draw -> annotate -> measure -> persist.

Drives the real screen (offscreen Qt) over a synthetic recording in a temporary
workspace. Nothing is played: playback is only ever asserted through the
controller's range, never started on a device.
"""

from __future__ import annotations

import numpy as np
import pytest
from PyQt6.QtCore import QThreadPool
from PyQt6.QtTest import QTest
from PyQt6.QtWidgets import QApplication

from magpy.models import Workspace
from magpy.screens import AudioAnnotationScreen
from magpy.services import Annotation, Candidate

SR = 22050


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def screen(qapp, tmp_path):
    from bioamla.audio import save_audio

    audio = np.zeros(SR * 8, dtype=np.float32)
    t = np.arange(SR) / SR
    audio[2 * SR : 3 * SR] = 0.5 * np.sin(2 * np.pi * 3000 * t)
    wav = tmp_path / "rec.wav"
    save_audio(str(wav), audio, SR)

    workspace = Workspace()
    workspace.create(tmp_path / "W.magpy", "W")
    screen = AudioAnnotationScreen(workspace)
    screen.resize(1200, 800)
    screen.show()
    screen.add_audio_path(wav)
    qapp.processEvents()
    screen._wav = wav
    yield screen
    QThreadPool.globalInstance().waitForDone(10000)
    screen.close()


def _settle(qapp) -> None:
    """Let the debounced measurement run finish and deliver its result."""
    QTest.qWait(300)
    QThreadPool.globalInstance().waitForDone(10000)
    QTest.qWait(50)
    qapp.processEvents()


def test_loading_shows_the_recording(screen):
    assert screen._document.audio is not None
    assert screen._spectrogram._img is not None
    assert screen._spectrogram._duration == pytest.approx(8.0, abs=0.01)


def test_drawing_makes_a_labelled_annotation_that_persists(qapp, screen):
    screen._label_combo.setEditText("wren")
    screen._spectrogram.selectionDrawn.emit(2.0, 3.0, 2000.0, 4000.0)

    (ann,) = screen._annotations.items()
    assert (ann.label, ann.start_time, ann.low_freq) == ("wren", 2.0, 2000.0)
    assert screen._annotations.selected is ann

    stored = screen._workspace.load_annotations_for(screen._wav)
    assert [(a.label, a.start_time, a.end_time) for a in stored] == [("wren", 2.0, 3.0)]


def test_waveform_drag_makes_a_time_only_annotation(screen):
    screen._waveform.selectionDrawn.emit(4.0, 5.0, None, None)
    (ann,) = screen._annotations.items()
    assert ann.low_freq is None and ann.high_freq is None


def test_measurements_fill_in_and_follow_edits(qapp, screen):
    screen._measure_keys = ["peak_frequency"]
    screen._table.set_measurement_keys(screen._measure_keys)
    screen._spectrogram.selectionDrawn.emit(2.0, 3.0, 2000.0, 4000.0)
    _settle(qapp)
    assert abs(float(screen._table.item(0, 6).text()) - 3000.0) < 100.0

    # Move the box onto silence: the old value must not survive.
    ann = screen._annotations.items()[0]
    ann.start_time, ann.end_time = 5.0, 6.0
    screen._annotations.update(ann)
    _settle(qapp)
    geometry, _values = screen._measured[ann.id]
    assert geometry[:2] == (5.0, 6.0)


def test_undo_and_redo_drive_the_stored_annotations(screen):
    screen._spectrogram.selectionDrawn.emit(2.0, 3.0, 2000.0, 4000.0)
    assert screen._undo_action.isEnabled() and not screen._redo_action.isEnabled()

    screen._undo_action.trigger()
    assert len(screen._annotations) == 0
    assert screen._workspace.load_annotations_for(screen._wav) == []

    screen._redo_action.trigger()
    assert len(screen._workspace.load_annotations_for(screen._wav)) == 1


def test_number_keys_and_enter_apply_labels(screen):
    for start, label in ((1.0, "robin"), (4.0, "wren")):
        screen._annotations.add(Annotation(start, start + 0.5, label=label))
    screen._spectrogram.selectionDrawn.emit(6.0, 6.5, 1000.0, 2000.0)
    new = screen._annotations.selected
    assert [screen._label_combo.itemText(i) for i in range(2)] == ["robin", "wren"]

    screen._apply_numbered_label(2)
    assert new.label == "wren"

    screen._label_combo.setEditText("towhee")
    screen._apply_active_label()
    assert new.label == "towhee"
    assert "towhee" in [screen._label_combo.itemText(i) for i in range(screen._label_combo.count())]


def test_stepping_and_delete(screen):
    for start in (1.0, 3.0, 5.0):
        screen._annotations.add(Annotation(start, start + 0.5))
    screen._annotations.select(None)
    screen._step_selection(+1)
    screen._step_selection(+1)
    assert screen._annotations.selected.start_time == 3.0
    screen._delete_selected()
    assert [a.start_time for a in screen._annotations.items()] == [1.0, 5.0]


def test_promoted_candidates_take_the_active_label(screen):
    screen._label_combo.setEditText("frog")
    screen._candidates.set_all([Candidate(1.0, 2.0, 0.8, 500.0, 900.0)], "Energy")
    assert len(screen._spectrogram._cand_items) == 1
    screen._promote_candidates(screen._candidates.items())
    (ann,) = screen._annotations.items()
    assert ann.label == "frog" and ann.confidence == 0.8


def test_play_selection_targets_the_selected_annotation(screen, monkeypatch):
    ranges = []
    monkeypatch.setattr(screen._playback, "play_range", lambda a, b: ranges.append((a, b)))
    screen._annotations.add(Annotation(2.0, 3.0))
    screen._play_selection()
    screen._annotations.select(None)
    screen._play_selection()  # nothing selected: the visible span
    assert ranges[0] == (2.0, 3.0)
    assert ranges[1][0] == pytest.approx(0.0, abs=0.01)
    assert ranges[1][1] == pytest.approx(8.0, abs=0.01)


def test_reopening_a_file_restores_its_annotations_with_a_clean_history(screen):
    screen._spectrogram.selectionDrawn.emit(2.0, 3.0, 2000.0, 4000.0)
    screen.load_file(screen._wav)
    assert len(screen._annotations) == 1
    assert not screen._annotations.can_undo


def test_spectrogram_settings_are_remembered_by_the_next_screen(qapp, screen, tmp_path):
    view = screen._spectrogram
    view._fft_combo.setCurrentIndex(view._fft_combo.findData(2048))
    view._cmap_combo.setCurrentText("gray_r")
    view._brightness.setValue(70)
    view._follow_box.setChecked(False)

    workspace = Workspace()
    workspace.create(tmp_path / "Other.magpy", "Other")
    fresh = AudioAnnotationScreen(workspace)
    try:
        settings = fresh._spectrogram.view_settings()
        assert settings["n_fft"] == 2048 and settings["colormap"] == "gray_r"
        assert settings["brightness"] == 70 and settings["follow"] is False
    finally:
        # Put the defaults back so later tests see the stock view.
        from magpy.settings import app_settings

        fresh.close()
        app_settings().remove("spectrogram")


def test_corrupt_stored_settings_are_ignored(qapp, screen):
    screen._spectrogram.apply_view_settings({"n_fft": "not a number", "colormap": "nope"})
    assert screen._spectrogram.params().n_fft in (256, 512, 1024, 2048, 4096, 8192)


def test_identify_writes_labels_above_the_threshold_as_one_undo_step(qapp, screen):
    from magpy.services import Identification

    for start in (1.0, 3.0, 5.0):
        screen._annotations.add(Annotation(start, start + 0.5))
    a, b, c = screen._annotations.items()
    results = {
        a.id: Identification("spring peeper", 0.91),
        b.id: Identification("green frog", 0.20),  # below the threshold
    }
    screen._apply_identifications(results, screen._wav, 0.5)

    assert [(x.label, x.confidence) for x in screen._annotations.items()] == [
        ("spring peeper", 0.91),
        ("", None),
        ("", None),
    ]
    stored = screen._workspace.load_annotations_for(screen._wav)
    assert [x.label for x in stored] == ["spring peeper", "", ""]

    screen._annotations.undo()  # the whole run is one step
    assert all(x.label == "" for x in screen._annotations.items())


def test_identify_results_for_another_recording_are_dropped(screen, tmp_path):
    from magpy.services import Identification

    screen._annotations.add(Annotation(1.0, 1.5))
    (ann,) = screen._annotations.items()
    screen._apply_identifications({ann.id: Identification("x", 0.9)}, tmp_path / "other.wav", 0.0)
    assert screen._annotations.items()[0].label == ""
