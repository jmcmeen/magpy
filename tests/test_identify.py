"""Tests for the identification service, against a stand-in for the AST engine.

The real engine needs torch and a model download, so the mapping and the
per-clip plumbing are exercised with a fake that classifies by clip loudness.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from magpy.services import Annotation, identify
from magpy.services.identify import identify_annotations

SR = 16000


class FakeEngine:
    loads = 0

    def __init__(self, model_path: str) -> None:
        type(self).loads += 1
        self.model_path = model_path

    def predict_topk(self, audio_path: str, *, top_k: int = 5):
        from bioamla.audio import load_audio

        clip, _sr = load_audio(audio_path)
        loud = float(np.abs(clip).max()) > 0.25
        if float(np.abs(clip).max()) > 0.9:
            raise RuntimeError("clipped")
        label = "loud" if loud else "quiet"
        return SimpleNamespace(
            predicted_label=label,
            confidence=0.8,
            top_k_labels=[label, "other"][:top_k],
            top_k_scores=[0.8, 0.2][:top_k],
        )


def _recording() -> np.ndarray:
    audio = np.full(SR * 6, 0.01, dtype=np.float32)
    audio[1 * SR : 2 * SR] = 0.5
    audio[4 * SR : 5 * SR] = 1.0
    return audio


def test_classifies_each_annotation_with_one_model_load(monkeypatch):
    FakeEngine.loads = 0
    monkeypatch.setattr(identify, "ASTInference", FakeEngine)
    loud, quiet = Annotation(1.0, 2.0), Annotation(2.5, 3.5)
    progress = []

    results = identify_annotations(
        _recording(),
        SR,
        [loud, quiet],
        "some/model",
        on_progress=lambda d, t: progress.append((d, t)),
    )

    assert FakeEngine.loads == 1
    assert results[loud.id].label == "loud" and results[quiet.id].label == "quiet"
    assert results[loud.id].confidence == 0.8
    assert results[loud.id].top_k == (("loud", 0.8), ("other", 0.2))
    assert progress == [(1, 2), (2, 2)]


def test_a_failing_or_too_short_clip_is_skipped_not_fatal(monkeypatch):
    monkeypatch.setattr(identify, "ASTInference", FakeEngine)
    good, failing, tiny = Annotation(1.0, 2.0), Annotation(4.0, 5.0), Annotation(3.0, 3.001)
    results = identify_annotations(_recording(), SR, [failing, tiny, good], "m")
    assert set(results) == {good.id}


def test_no_annotations_does_not_load_a_model(monkeypatch):
    FakeEngine.loads = 0
    monkeypatch.setattr(identify, "ASTInference", FakeEngine)
    assert identify_annotations(_recording(), SR, [], "m") == {}
    assert FakeEngine.loads == 0


def test_when_every_clip_fails_the_cause_is_raised(monkeypatch):
    import pytest

    monkeypatch.setattr(identify, "ASTInference", FakeEngine)
    failing = Annotation(4.0, 5.0)  # the fake engine raises on this (clipped) region
    with pytest.raises(RuntimeError, match="clipped"):
        identify_annotations(_recording(), SR, [failing], "m")
