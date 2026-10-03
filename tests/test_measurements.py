"""Tests for the per-annotation measurement service (runs bioamla for real)."""

from __future__ import annotations

import csv

import numpy as np

from magpy.services import (
    DEFAULT_MEASUREMENTS,
    MEASUREMENTS,
    MEASUREMENTS_BY_KEY,
    Annotation,
    export_measurements_csv,
    measure_annotations,
)

SR = 22050


def _recording() -> np.ndarray:
    """Silence, with a 3 kHz tone from 2-3 s and a 6 kHz tone from 5-6 s."""
    audio = np.zeros(SR * 8, dtype=np.float32)
    t = np.arange(SR) / SR
    audio[2 * SR : 3 * SR] = 0.5 * np.sin(2 * np.pi * 3000 * t)
    audio[5 * SR : 6 * SR] = 0.25 * np.sin(2 * np.pi * 6000 * t)
    return audio


def test_every_default_is_a_known_measurement():
    assert set(DEFAULT_MEASUREMENTS) <= set(MEASUREMENTS_BY_KEY)
    assert len({spec.key for spec in MEASUREMENTS}) == len(MEASUREMENTS)


def test_measures_each_annotation_from_its_own_region():
    low = Annotation(2.0, 3.0, 2000.0, 4000.0, label="low")
    high = Annotation(5.0, 6.0, 5000.0, 7000.0, label="high")
    values = measure_annotations(
        _recording(), SR, [low, high], ["duration", "peak_frequency", "rms_db"]
    )

    assert abs(values[low.id]["peak_frequency"] - 3000.0) < 100.0
    assert abs(values[high.id]["peak_frequency"] - 6000.0) < 100.0
    assert abs(values[low.id]["duration"] - 1.0) < 1e-3
    # Half the amplitude is ~6 dB down.
    assert 5.0 < values[low.id]["rms_db"] - values[high.id]["rms_db"] < 7.0


def test_degenerate_regions_do_not_abort_the_batch():
    good = Annotation(2.0, 3.0, 2000.0, 4000.0)
    empty = Annotation(100.0, 101.0)  # past the end of the recording
    values = measure_annotations(_recording(), SR, [empty, good], ["peak_frequency"])
    assert values[empty.id] == {}
    assert "peak_frequency" in values[good.id]


def test_unknown_keys_are_ignored():
    ann = Annotation(2.0, 3.0)
    assert measure_annotations(_recording(), SR, [ann], ["not_a_metric"]) == {ann.id: {}}


def test_export_writes_annotation_fields_and_measurement_columns(tmp_path):
    ann = Annotation(2.0, 3.0, 2000.0, 4000.0, label="wren", notes="clear")
    keys = ["duration", "peak_frequency"]
    values = measure_annotations(_recording(), SR, [ann], keys)
    out = tmp_path / "table.csv"
    export_measurements_csv([ann], values, keys, out)

    with open(out, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == 1
    row = rows[0]
    assert row["label"] == "wren" and row["notes"] == "clear"
    assert float(row["start_time"]) == 2.0 and float(row["high_freq"]) == 4000.0
    assert abs(float(row["peak_frequency"]) - 3000.0) < 100.0
