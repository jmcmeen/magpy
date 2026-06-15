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
    view.set_image(SpectrogramImage(db=db, freqs=np.linspace(0, 8000, 64), times=np.linspace(0, 10, 200)))
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
