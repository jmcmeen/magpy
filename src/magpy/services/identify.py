"""
Identification service -- label annotations with an AST classifier.

The seam over ``bioamla.ml.ASTInference`` for the annotation screen's
"Identify" action: given the open recording and some annotations, classify the
audio inside each one and return a MagPy-owned :class:`Identification` (best
label, its confidence, and the runners-up) per annotation.

Two things shape it:

* **The model is loaded once per run**, not once per annotation. Loading an AST
  checkpoint takes seconds (and the first use of a Hugging Face id downloads
  it), so ``bioamla.ml.predict_file`` -- which builds a fresh engine on every
  call -- is the wrong tool for labelling a table of boxes.
* bioamla's engine classifies *files*, so each annotation's clip is written to a
  small temporary WAV first (the same approach as the measurement service).

This needs torch and a model, so it always runs through a
:class:`~magpy.workers.Worker`. It reports ``on_progress`` per annotation (after
the model has loaded), which also makes it cancellable between annotations.
"""

from __future__ import annotations

import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# bioamla import is confined to the services layer. bioamla.ml is lazy: torch and
# transformers load when the engine is constructed, not on this import.
from bioamla.audio import save_audio
from bioamla.ml import ASTInference

from .annotations import Annotation
from .spectrogram import to_mono

DEFAULT_IDENTIFY_MODEL = "bioamla/scp-frogs"
_MIN_CLIP_SECONDS = 0.05


@dataclass(frozen=True)
class Identification:
    """A classifier's verdict on one annotation."""

    label: str
    confidence: float
    top_k: tuple[tuple[str, float], ...] = ()


def identify_annotations(
    samples: np.ndarray,
    sample_rate: int,
    annotations: list[Annotation],
    model_path: str = DEFAULT_IDENTIFY_MODEL,
    *,
    top_k: int = 3,
    on_progress: Callable[[int, int], None] | None = None,
) -> dict[str, Identification]:
    """Classify the audio inside each annotation; returns ``{annotation.id: result}``.

    ``model_path`` is a local model directory or a Hugging Face model id.
    Annotations too short to classify are left out of the result. A model that
    cannot be loaded raises (bioamla's ``ModelError``). A clip that fails is
    skipped so one bad region doesn't lose the rest -- but if *every* clip fails
    the first error is raised, because then the cause is the setup (audio
    decoding, device), not the regions, and the user needs to see it.
    """
    if not annotations:
        return {}
    engine = ASTInference(model_path=model_path)
    mono = to_mono(samples)
    results: dict[str, Identification] = {}
    first_error: Exception | None = None
    with tempfile.TemporaryDirectory(prefix="magpy-identify-") as tmp:
        clip_path = str(Path(tmp) / "clip.wav")
        for done, ann in enumerate(annotations, start=1):
            i0 = max(0, int(ann.start_time * sample_rate))
            i1 = min(len(mono), int(ann.end_time * sample_rate))
            if (i1 - i0) >= _MIN_CLIP_SECONDS * sample_rate:
                try:
                    save_audio(
                        clip_path, np.ascontiguousarray(mono[i0:i1], dtype=np.float32), sample_rate
                    )
                    r = engine.predict_topk(clip_path, top_k=top_k)
                except Exception as exc:  # noqa: BLE001 - skip the clip, keep the run
                    first_error = first_error or exc
                    r = None
                if r is not None:
                    labels = list(r.top_k_labels or [])
                    scores = [float(s) for s in (r.top_k_scores or [])]
                    results[ann.id] = Identification(
                        label=str(r.predicted_label),
                        confidence=float(r.confidence),
                        top_k=tuple(zip(labels, scores, strict=False)),
                    )
            if on_progress is not None:
                on_progress(done, len(annotations))
    if not results and first_error is not None:
        raise first_error
    return results
