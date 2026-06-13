"""Tests for the batch service seam.

The audio batch ops are fully exercisable on synthetic wavs, so these run them
for real and assert outputs. Model/cluster ops (need downloads / .npy) are only
checked for spec wiring, matching the verifiability boundary.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from magpy.services import BATCH_OPS, BatchOutcome, run_batch_op


@pytest.fixture
def audio_dir(tmp_path: Path) -> Path:
    d = tmp_path / "in"
    d.mkdir()
    sr = 22050
    for i in range(3):
        t = np.linspace(0, 1, sr, endpoint=False)
        sf.write(d / f"tone{i}.wav", (0.3 * np.sin(2 * np.pi * (440 + 110 * i) * t)).astype("float32"), sr)
    return d


def test_batch_ops_catalogue_is_complete():
    keys = {op.key for op in BATCH_OPS}
    # the full bioamla `batch` group
    assert {
        "audio_info", "audio_convert", "audio_resample", "audio_normalize",
        "audio_trim", "audio_filter", "audio_denoise", "audio_segment",
        "audio_visualize", "detect_energy", "detect_ribbit", "detect_peaks",
        "detect_accelerating", "index", "models_predict", "models_embed", "cluster",
    } == keys


def test_max_workers_only_on_supporting_ops():
    # The crash-trap guard: no op exposes a max_workers field unless it declares support.
    for op in BATCH_OPS:
        has_workers_field = any(p.name == "max_workers" for p in op.params)
        assert has_workers_field == op.supports_max_workers, op.key


def test_progress_synthesized_for_loop_ops(audio_dir, tmp_path):
    ticks = []
    out = tmp_path / "out"
    outcome = run_batch_op("index", audio_dir, out, {"recursive": True}, on_progress=lambda d, t: ticks.append((d, t)))
    assert isinstance(outcome, BatchOutcome)
    assert outcome.total == 3 and outcome.successful == 3
    assert ticks and ticks[-1] == (3, 3)
    assert (out / "indices.csv").exists()


def test_resample_writes_outputs(audio_dir, tmp_path):
    out = tmp_path / "rs"
    outcome = run_batch_op("audio_resample", audio_dir, out, {"target_sample_rate": 16000, "max_workers": 1})
    assert outcome.successful == 3
    assert len(list(out.glob("*.wav"))) == 3


def test_filter_requires_a_cutoff(audio_dir, tmp_path):
    # No cutoff set -> the processor builder raises (surfaced as a failed run).
    with pytest.raises(ValueError):
        run_batch_op("audio_filter", audio_dir, tmp_path / "f", {"order": 5})


def test_detect_runs_without_progress_kwarg(audio_dir, tmp_path):
    # detect has max_workers but NOT on_progress; forwarding on_progress would TypeError.
    out = tmp_path / "d"
    outcome = run_batch_op(
        "detect_energy", audio_dir, out,
        {"low_freq": 500.0, "high_freq": 5000.0, "threshold_db": -30.0, "max_workers": 1},
    )
    assert outcome.total == 3
    # Reports the real output (detections_<method>.json from metadata), not the
    # input paths bioamla leaves in BatchResult.output_files.
    assert outcome.output_files and outcome.output_files[0].endswith("detections_energy.json")
    assert "detections" in outcome.message


def test_unknown_op_raises():
    with pytest.raises(ValueError):
        run_batch_op("nope", ".", ".", {})
