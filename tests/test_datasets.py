"""Tests for the datasets service seam (run for real on synthetic annotated audio)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from magpy.services import (
    DatasetOutcome,
    dataset_stats,
    extract_clips,
    merge,
    partition,
)


@pytest.fixture
def annotated_source(tmp_path: Path) -> Path:
    src = tmp_path / "src"
    src.mkdir()
    sr = 22050
    for name in ("rec1", "rec2"):
        t = np.linspace(0, 3, sr * 3, endpoint=False)
        sf.write(src / f"{name}.wav", (0.3 * np.sin(2 * np.pi * 440 * t)).astype("float32"), sr)
        (src / f"{name}.csv").write_text(
            "start_time,end_time,label\n0.4,0.9,frog\n1.4,1.9,bird\n2.1,2.6,frog\n"
        )
    return src


def test_extract_clips_builds_dataset(annotated_source, tmp_path):
    out = tmp_path / "ds"
    outcome = extract_clips(str(annotated_source), str(out))
    assert isinstance(outcome, DatasetOutcome)
    assert (out / "metadata.csv").exists()
    assert outcome.details.get("clips_written", 0) > 0


def test_stats_and_partition(annotated_source, tmp_path):
    out = tmp_path / "ds"
    extract_clips(str(annotated_source), str(out))
    stats = dataset_stats(str(out))
    assert stats.details.get("total_files", 0) > 0
    part = partition(str(out), train=0.6, val=0.2, test=0.2, stratify=False)
    assert part.op == "partition"
    assert isinstance(part.details.get("splits"), dict)


def test_merge_requires_two_datasets(annotated_source, tmp_path):
    out = tmp_path / "ds"
    extract_clips(str(annotated_source), str(out))
    with pytest.raises(ValueError):
        merge([str(out)], str(tmp_path / "merged"))  # only one -> error
