"""Tests for SpectrogramView's 2-D box selection and box-vs-region rendering.

These drive the real widget headless (offscreen Qt). The freq-bounded box must
land in data coordinates and time-only annotations must stay full-height regions.
"""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np  # noqa: E402
import pytest  # noqa: E402
from PyQt6.QtWidgets import QApplication  # noqa: E402

from magpy.models import AnnotationSet  # noqa: E402
from magpy.services import Annotation, SpectrogramImage  # noqa: E402
from magpy.widgets import SpectrogramView  # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


def _view(qapp):
    model = AnnotationSet()
    view = SpectrogramView(model)
    db = np.zeros((64, 200), dtype="float32")
    view.set_image(
        SpectrogramImage(db=db, freqs=np.linspace(0, 8000, 64), times=np.linspace(0, 10, 200))
    )
    return model, view


def test_selected_freq_box_is_editable_roi_in_data_coords(qapp):
    model, view = _view(qapp)
    ann = Annotation(start_time=2.0, end_time=4.0, low_freq=1000.0, high_freq=3000.0)
    model.add(ann)  # add() auto-selects -> editable ROI
    assert view._edit_id == ann.id and ann.id not in view._boxes
    pos, size = view._edit_roi.pos(), view._edit_roi.size()
    assert (pos.x(), pos.y(), size.x(), size.y()) == (2.0, 1000.0, 2.0, 2000.0)


def test_unselected_freq_box_is_static_rect(qapp):
    model, view = _view(qapp)
    first = Annotation(start_time=2.0, end_time=4.0, low_freq=1000.0, high_freq=3000.0)
    model.add(first)
    model.add(Annotation(start_time=6.0, end_time=7.0, low_freq=500.0, high_freq=1500.0))
    # the second is now selected (ROI); the first demotes to a static data-coord box
    assert first.id in view._boxes
    r = view._boxes[first.id].rect()
    assert (r.x(), r.y(), r.width(), r.height()) == (2.0, 1000.0, 2.0, 2000.0)


def test_drag_edit_writes_bounds_back_without_teardown(qapp):
    model, view = _view(qapp)
    ann = Annotation(start_time=2.0, end_time=4.0, low_freq=1000.0, high_freq=3000.0)
    model.add(ann)
    view._edit_roi.setPos([2.5, 1500.0])
    view._edit_roi.setSize([1.0, 1000.0])
    view._on_box_edited()
    assert (ann.start_time, ann.end_time, ann.low_freq, ann.high_freq) == (2.5, 3.5, 1500.0, 2500.0)
    assert view._edit_id == ann.id  # ROI still held, not torn down


def test_time_only_annotation_renders_as_region(qapp):
    model, view = _view(qapp)
    ann = Annotation(start_time=1.0, end_time=2.0)  # no freq bounds
    model.add(ann)
    assert ann.id in view._regions and ann.id not in view._boxes


def test_box_selection_bounds_normalized_and_clamped(qapp):
    model, view = _view(qapp)
    view.start_box_selection()
    view._selection.setPos([5.0, 2000.0])
    view._selection.setSize([-1.0, -500.0])  # dragged up/left -> negative size
    start, end, low, high = view.selection_bounds()
    assert start < end and low < high and low >= 0.0


def test_time_selection_bounds_have_no_freq(qapp):
    model, view = _view(qapp)
    view.start_selection()
    start, end, low, high = view.selection_bounds()
    assert low is None and high is None and start < end


def test_no_selection_returns_none(qapp):
    _model, view = _view(qapp)
    assert view.selection_bounds() is None


# --- viewport rendering + gestures ------------------------------------------
#
# These drive real mouse events through the widget. pyqtgraph rate-limits move
# events (100/s), so drags pause between moves the way a hand would.

from PyQt6.QtCore import QEvent, QPoint, QPointF, Qt  # noqa: E402
from PyQt6.QtGui import QMouseEvent, QWheelEvent  # noqa: E402
from PyQt6.QtTest import QTest  # noqa: E402

from magpy.services import render_spectrogram  # noqa: E402

_SR = 22050
_NO_MOD = Qt.KeyboardModifier.NoModifier
_LEFT = Qt.MouseButton.LeftButton


def _live_view(qapp, seconds: float = 20.0):
    """A shown view over real audio, wired to render what it asks for."""
    audio = (0.1 * np.random.default_rng(0).standard_normal(int(_SR * seconds))).astype("float32")
    model = AnnotationSet()
    view = SpectrogramView(model)
    view.resize(1000, 500)
    requests: list[tuple[float, float, int]] = []

    def render(t0: float, t1: float, cols: int) -> None:
        requests.append((t0, t1, cols))
        view.set_image(render_spectrogram(audio, _SR, t0, t1, view.params(), max_cols=cols))

    view.renderRequested.connect(render)
    view.show()
    qapp.processEvents()
    view.set_extent(seconds, _SR / 2)
    QTest.qWait(60)  # let the debounced render timer settle
    return model, view, requests


def _px(view, t: float, f: float) -> QPoint:
    return view._plot.mapFromScene(view._vb.mapViewToScene(QPointF(t, f)))


def _drag(qapp, view, start, end, button=_LEFT) -> None:
    viewport = view._plot.viewport()
    a, b = _px(view, *start), _px(view, *end)
    QTest.mousePress(viewport, button, _NO_MOD, a)
    for i in range(1, 9):
        QTest.qWait(15)
        step = QPoint(a.x() + (b.x() - a.x()) * i // 8, a.y() + (b.y() - a.y()) * i // 8)
        QTest.mouseMove(viewport, step)
    QTest.mouseRelease(viewport, button, _NO_MOD, b)
    qapp.processEvents()


def test_announcing_a_recording_requests_one_render_of_the_whole_span(qapp):
    _model, view, requests = _live_view(qapp)
    assert len(requests) == 1
    t0, t1, cols = requests[0]
    assert t0 == 0.0 and t1 == 20.0 and 200 <= cols <= 4000
    assert view._img is not None and view._noise_db is not None


def test_zooming_in_requests_a_sharper_window_with_margin(qapp):
    _model, view, requests = _live_view(qapp)
    view._plot.setXRange(8.0, 10.0, padding=0)
    QTest.qWait(80)
    t0, t1, _cols = requests[-1]
    assert t0 < 8.0 and t1 > 10.0  # margin either side
    assert len(requests) == 2

    view._plot.setXRange(8.2, 10.2, padding=0)  # a small pan stays inside the margin
    QTest.qWait(80)
    assert len(requests) == 2


def test_changing_stft_settings_rerenders(qapp):
    _model, view, requests = _live_view(qapp)
    emitted = []
    view.paramsChanged.connect(emitted.append)
    view._fft_combo.setCurrentIndex(view._fft_combo.findData(2048))
    assert emitted[-1].n_fft == 2048
    assert len(requests) == 2 and view._img.db.shape[0] == 1025


def test_drag_draws_a_normalised_box(qapp):
    _model, view, _ = _live_view(qapp)
    drawn = []
    view.selectionDrawn.connect(lambda *bounds: drawn.append(bounds))

    _drag(qapp, view, (5.0, 2000.0), (8.0, 6000.0))
    _drag(qapp, view, (14.0, 9000.0), (12.0, 7000.0))  # dragged up and to the left

    assert len(drawn) == 2
    for (start, end, low, high), expected in zip(
        drawn, [(5.0, 8.0, 2000.0, 6000.0), (12.0, 14.0, 7000.0, 9000.0)], strict=True
    ):
        assert start < end and low < high
        assert abs(start - expected[0]) < 0.1 and abs(end - expected[1]) < 0.1
        assert abs(low - expected[2]) < 150 and abs(high - expected[3]) < 150


def test_drag_does_not_draw_when_drawing_is_disabled(qapp):
    _model, view, _ = _live_view(qapp)
    view._plot.setXRange(5.0, 15.0, padding=0)
    view.set_draw_enabled(False)
    drawn = []
    view.selectionDrawn.connect(lambda *bounds: drawn.append(bounds))
    _drag(qapp, view, (10.0, 5000.0), (8.0, 5000.0))
    assert not drawn
    assert view._plot.viewRange()[0][0] > 5.5  # it panned instead


def test_click_selects_a_box_and_empty_space_deselects_and_seeks(qapp):
    model, view, _ = _live_view(qapp)
    ann = Annotation(start_time=5.0, end_time=8.0, low_freq=2000.0, high_freq=6000.0)
    model.add(ann)
    model.select(None)
    seeks = []
    view.seekRequested.connect(seeks.append)
    viewport = view._plot.viewport()

    QTest.mouseClick(viewport, _LEFT, _NO_MOD, _px(view, 6.0, 4000.0))
    assert model.selected is ann and not seeks

    QTest.mouseClick(viewport, _LEFT, _NO_MOD, _px(view, 15.0, 4000.0))
    assert model.selected is None
    assert len(seeks) == 1 and abs(seeks[0] - 15.0) < 0.1


def test_double_click_plays_the_box(qapp):
    model, view, _ = _live_view(qapp)
    model.add(Annotation(start_time=5.0, end_time=8.0, low_freq=2000.0, high_freq=6000.0))
    plays = []
    view.playRangeRequested.connect(lambda start, end: plays.append((start, end)))
    viewport = view._plot.viewport()
    point = _px(view, 6.0, 4000.0)

    # What a real double-click delivers: press, release, double-click, release.
    QTest.mouseClick(viewport, _LEFT, _NO_MOD, point)
    qapp.sendEvent(
        viewport,
        QMouseEvent(
            QEvent.Type.MouseButtonDblClick,
            QPointF(point),
            QPointF(viewport.mapToGlobal(point)),
            _LEFT,
            _LEFT,
            _NO_MOD,
        ),
    )
    QTest.mouseRelease(viewport, _LEFT, _NO_MOD, point)
    assert plays == [(5.0, 8.0)]


def test_dragging_the_selected_box_moves_it_instead_of_drawing(qapp):
    model, view, _ = _live_view(qapp)
    ann = Annotation(start_time=5.0, end_time=8.0, low_freq=2000.0, high_freq=6000.0)
    model.add(ann)  # selected -> editable
    drawn = []
    view.selectionDrawn.connect(lambda *bounds: drawn.append(bounds))

    _drag(qapp, view, (6.0, 4000.0), (7.0, 4500.0))

    assert not drawn
    assert abs(ann.start_time - 6.0) < 0.1 and abs(ann.end_time - 9.0) < 0.1
    assert abs(ann.low_freq - 2500.0) < 150
    model.undo()
    assert model.items()[0].start_time == 5.0


def test_wheel_zooms_time_at_the_cursor_and_leaves_frequency(qapp):
    _model, view, _ = _live_view(qapp)
    viewport = view._plot.viewport()
    point = _px(view, 10.0, 5000.0)
    (_, _), y_before = view._plot.viewRange()
    wheel = QWheelEvent(
        QPointF(point),
        QPointF(viewport.mapToGlobal(point)),
        QPoint(0, 0),
        QPoint(0, 480),
        Qt.MouseButton.NoButton,
        _NO_MOD,
        Qt.ScrollPhase.NoScrollPhase,
        False,
    )
    qapp.sendEvent(viewport, wheel)
    (x0, x1), y_after = view._plot.viewRange()
    assert x1 - x0 < 10.0  # zoomed in
    assert abs((x0 + x1) / 2 - 10.0) < 0.5  # about the cursor
    assert [round(v) for v in y_after] == [round(v) for v in y_before]


def test_label_colors_are_stable_and_distinct_within_the_palette():
    from magpy.widgets import _plot

    _plot._assigned.clear()
    labels = [f"species {i}" for i in range(len(_plot._LABEL_PALETTE))]
    colors = [_plot.label_color(label).name() for label in labels]
    assert len(set(colors)) == len(labels)  # no two labels share a colour
    assert [_plot.label_color(label).name() for label in labels] == colors  # and they stick
    assert _plot.label_color("").name() not in colors
    _plot.label_color("one more than the palette holds")  # still returns a colour


def test_candidates_draw_as_overlays(qapp):
    from magpy.models import CandidateSet
    from magpy.services import Candidate

    _model, view = _view(qapp)
    candidates = CandidateSet()
    view.bind_candidates(candidates)
    candidates.set_all(
        [Candidate(1.0, 2.0, 0.9, 1000.0, 3000.0), Candidate(4.0, 5.0, 0.2)], "Energy"
    )
    assert len(view._cand_items) == 2
    candidates.set_threshold(0.5)  # the confidence filter hides the weak one
    assert len(view._cand_items) == 1
    candidates.clear()
    assert not view._cand_items
