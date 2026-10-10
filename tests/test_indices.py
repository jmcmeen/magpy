"""Tests for the acoustic-indices service seam."""

from __future__ import annotations

import numpy as np

from magpy.services import IndexSummary, compute_indices


def test_compute_indices_returns_summary_with_all_rows():
    sr = 22050
    audio = (np.random.RandomState(3).randn(sr * 3) * 0.1).astype("float32")
    summary = compute_indices(audio, sr)
    assert isinstance(summary, IndexSummary)
    rows = summary.rows()
    keys = {r.key for r in rows}
    assert {"aci", "adi", "aei", "bio", "ndsi", "h_spectral", "h_temporal"} <= keys
    # Descriptions populate tooltips.
    assert all(r.description for r in rows)
    assert summary.duration > 0


def test_compute_indices_downmixes_stereo():
    sr = 22050
    stereo = (np.random.RandomState(4).randn(2, sr * 2) * 0.1).astype("float32")
    summary = compute_indices(stereo, sr)
    assert summary.sample_rate == sr
