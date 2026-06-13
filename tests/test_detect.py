"""Tests for the detect service seam and the candidate model."""

from __future__ import annotations

import numpy as np
import pytest

from magpy.models import CandidateSet
from magpy.services import (
    DETECTOR_SPECS,
    Candidate,
    candidate_to_annotation,
    run_detection,
)


def test_detector_specs_have_kinds_and_params():
    kinds = {s.kind for s in DETECTOR_SPECS}
    assert {"energy", "ribbit", "peaks", "accelerating"} <= kinds
    for spec in DETECTOR_SPECS:
        assert spec.params, f"{spec.kind} should expose tunable params"


def test_run_detection_returns_candidates():
    sr = 22050
    audio = (np.random.RandomState(0).randn(sr * 3) * 0.1).astype("float32")
    cands = run_detection(audio, sr, "energy", {"threshold_db": -40.0})
    assert all(isinstance(c, Candidate) for c in cands)
    assert all(c.detector == "Band-limited energy" for c in cands)


def test_run_detection_downmixes_stereo():
    sr = 22050
    stereo = (np.random.RandomState(1).randn(2, sr * 2) * 0.1).astype("float32")
    # Should not raise on a (channels, samples) array.
    run_detection(stereo, sr, "energy", {})


def test_run_detection_rejects_unknown_kind():
    with pytest.raises(ValueError):
        run_detection(np.zeros(1000, dtype="float32"), 22050, "bogus", {})


def test_candidate_to_annotation_remaps_freq_and_records_provenance():
    c = Candidate(
        start_time=1.0, end_time=2.0, confidence=0.9,
        low_freq=500.0, high_freq=5000.0, label="x", detector="RIBBIT",
    )
    ann = candidate_to_annotation(c)
    assert (ann.start_time, ann.end_time) == (1.0, 2.0)
    assert ann.low_freq == 500.0 and ann.high_freq == 5000.0
    assert "RIBBIT" in ann.notes


def test_candidate_set_threshold_filters_visible():
    cs = CandidateSet()
    cs.set_all(
        [
            Candidate(0.0, 1.0, confidence=0.2),
            Candidate(1.0, 2.0, confidence=0.8),
        ],
        detector="test",
    )
    assert len(cs.visible_items()) == 2
    cs.set_threshold(0.5)
    assert len(cs.visible_items()) == 1
    assert cs.detector == "test"
