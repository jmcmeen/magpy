"""
Detection service -- thin wrapper over ``bioamla.detect`` for the GUI.

Part of the services seam: the only place (besides its sibling ``services``
modules) that imports ``bioamla.detect``. It owns two MagPy-owned things so that
nothing past the seam touches a bioamla type:

* :class:`Candidate` -- a single detected event. Detections are a *reviewable
  candidate layer*, distinct from curated annotations (the user promotes a
  candidate into the :class:`~magpy.models.AnnotationSet`). bioamla's
  ``Detection`` names its frequency bounds ``frequency_low``/``frequency_high``;
  MagPy's ``Annotation`` uses ``low_freq``/``high_freq``. The remap happens here
  (:func:`candidate_to_annotation`) so the widget layer never sees the mismatch.
* :data:`DETECTOR_SPECS` -- a declarative description of each detector and its
  tunable parameters (label, default, range). The detect *panel* builds its form
  generically from this, instead of importing bioamla to introspect the detector
  classes. This is the same pattern as the spectrogram service owning ``n_fft``:
  GUI-facing defaults live in services, not widgets.

Detection runs over the whole file and is slow (energy detection of 60 s of
48 kHz audio ≈ 4 s), so callers run it through a :class:`~magpy.workers.Worker`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .annotations import Annotation

# bioamla import is confined to the services layer.
from bioamla.detect import (
    AcceleratingPatternDetector,
    BandLimitedEnergyDetector,
    CWTPeakDetector,
    Detection,
    RibbitDetector,
    detect_all,
)


@dataclass(frozen=True)
class Candidate:
    """A single detected event -- a *candidate* annotation pending review.

    MagPy-owned so bioamla's ``Detection`` does not leak past the seam. ``detector``
    records which detector produced it (the candidate layer can hold a mix).
    """

    start_time: float
    end_time: float
    confidence: float = 1.0
    low_freq: float | None = None
    high_freq: float | None = None
    label: str = ""
    detector: str = ""

    @property
    def duration(self) -> float:
        return self.end_time - self.start_time


@dataclass(frozen=True)
class ParamSpec:
    """One tunable detector parameter, described for generic form-building.

    ``decimals == 0`` marks an integer parameter (the form uses an int spinbox
    and :func:`run_detection` casts the value back to ``int``).
    """

    name: str
    label: str
    default: float
    minimum: float
    maximum: float
    step: float = 1.0
    decimals: int = 2


@dataclass(frozen=True)
class DetectorSpec:
    """A detector and the subset of its parameters the GUI exposes."""

    kind: str
    label: str
    description: str
    params: tuple[ParamSpec, ...]


# The GUI-facing catalogue of detectors. Defaults mirror bioamla's constructor
# defaults; only the commonly-tuned parameters are surfaced (optional/``None``
# defaults like CWT's freq bounds keep bioamla's behaviour when omitted).
DETECTOR_SPECS: tuple[DetectorSpec, ...] = (
    DetectorSpec(
        kind="energy",
        label="Band-limited energy",
        description="Flags spans where in-band energy exceeds a dB threshold. "
        "General-purpose; fast to reason about.",
        params=(
            ParamSpec("low_freq", "Low freq (Hz)", 500.0, 0.0, 96000.0, 50.0, 0),
            ParamSpec("high_freq", "High freq (Hz)", 5000.0, 0.0, 96000.0, 50.0, 0),
            ParamSpec("threshold_db", "Threshold (dB)", -20.0, -120.0, 0.0, 1.0, 1),
            ParamSpec("min_duration", "Min duration (s)", 0.05, 0.0, 10.0, 0.01, 3),
            ParamSpec("merge_threshold", "Merge gap (s)", 0.1, 0.0, 10.0, 0.01, 3),
            ParamSpec("smoothing_window", "Smoothing (s)", 0.02, 0.0, 1.0, 0.01, 3),
        ),
    ),
    DetectorSpec(
        kind="ribbit",
        label="RIBBIT (pulse rate)",
        description="Finds sounds with a target pulse repetition rate "
        "(e.g. many frog and insect calls).",
        params=(
            ParamSpec("pulse_rate_hz", "Pulse rate (Hz)", 10.0, 0.1, 200.0, 0.5, 2),
            ParamSpec("pulse_rate_tolerance", "Rate tolerance", 0.2, 0.0, 1.0, 0.05, 2),
            ParamSpec("low_freq", "Low freq (Hz)", 500.0, 0.0, 96000.0, 50.0, 0),
            ParamSpec("high_freq", "High freq (Hz)", 5000.0, 0.0, 96000.0, 50.0, 0),
            ParamSpec("window_duration", "Window (s)", 2.0, 0.1, 30.0, 0.1, 2),
            ParamSpec("hop_duration", "Hop (s)", 0.5, 0.01, 10.0, 0.1, 2),
            ParamSpec("min_score", "Min score", 0.3, 0.0, 1.0, 0.05, 2),
            ParamSpec("n_fft", "FFT size", 1024, 128, 8192, 128, 0),
        ),
    ),
    DetectorSpec(
        kind="peaks",
        label="CWT peaks",
        description="Wavelet-based transient/peak detector for short impulsive "
        "events (clicks, taps).",
        params=(
            ParamSpec("min_scale", "Min scale", 1, 1, 200, 1, 0),
            ParamSpec("max_scale", "Max scale", 50, 1, 500, 1, 0),
            ParamSpec("n_scales", "Scales", 20, 1, 200, 1, 0),
            ParamSpec("snr_threshold", "SNR threshold", 2.0, 0.0, 50.0, 0.5, 2),
            ParamSpec("min_peak_distance", "Min peak gap (s)", 0.01, 0.0, 10.0, 0.01, 3),
        ),
    ),
    DetectorSpec(
        kind="accelerating",
        label="Accelerating pattern",
        description="Detects pulse trains that speed up or slow down "
        "(accelerating/decelerating trills).",
        params=(
            ParamSpec("min_pulses", "Min pulses", 5, 2, 100, 1, 0),
            ParamSpec("acceleration_threshold", "Accel. threshold", 1.5, 1.0, 10.0, 0.1, 2),
            ParamSpec("low_freq", "Low freq (Hz)", 500.0, 0.0, 96000.0, 50.0, 0),
            ParamSpec("high_freq", "High freq (Hz)", 5000.0, 0.0, 96000.0, 50.0, 0),
            ParamSpec("min_pulse_rate", "Min pulse rate (Hz)", 2.0, 0.1, 200.0, 0.5, 2),
            ParamSpec("max_pulse_rate", "Max pulse rate (Hz)", 50.0, 0.1, 500.0, 0.5, 2),
            ParamSpec("window_duration", "Window (s)", 3.0, 0.1, 30.0, 0.1, 2),
            ParamSpec("hop_duration", "Hop (s)", 0.5, 0.01, 10.0, 0.1, 2),
        ),
    ),
)

_DETECTOR_CLASSES = {
    "energy": BandLimitedEnergyDetector,
    "ribbit": RibbitDetector,
    "peaks": CWTPeakDetector,
    "accelerating": AcceleratingPatternDetector,
}

_SPECS_BY_KIND = {spec.kind: spec for spec in DETECTOR_SPECS}


def candidate_to_annotation(candidate: Candidate) -> Annotation:
    """Promote a reviewed :class:`Candidate` into a curated :class:`Annotation`.

    This is the freq-field remap boundary: ``Detection``/``Candidate`` carry
    ``low_freq``/``high_freq`` while the candidate's ``detector`` (e.g. "RIBBIT")
    is recorded in the annotation's ``notes`` for provenance.
    """
    return Annotation(
        start_time=candidate.start_time,
        end_time=candidate.end_time,
        low_freq=candidate.low_freq,
        high_freq=candidate.high_freq,
        label=candidate.label,
        confidence=candidate.confidence,
        notes=f"detector={candidate.detector}" if candidate.detector else "",
    )


def detector_label(kind: str) -> str:
    """The display label for a detector kind (falls back to the raw kind)."""
    spec = _SPECS_BY_KIND.get(kind)
    return spec.label if spec else kind


def _coerce_params(kind: str, params: dict[str, float]) -> dict[str, object]:
    """Cast integer-typed params (``decimals == 0``) back to ``int`` for bioamla."""
    spec = _SPECS_BY_KIND.get(kind)
    if spec is None:
        return dict(params)
    int_names = {p.name for p in spec.params if p.decimals == 0}
    return {k: (int(v) if k in int_names else v) for k, v in params.items()}


def _to_mono(audio: np.ndarray) -> np.ndarray:
    """Downmix ``(channels, samples)`` to mono; pass 1-D audio through."""
    if audio.ndim == 2:
        return audio.mean(axis=0)
    return audio


def run_detection(
    audio: np.ndarray,
    sample_rate: int,
    kind: str,
    params: dict[str, float] | None = None,
) -> list[Candidate]:
    """Run one detector over ``audio`` and return MagPy :class:`Candidate` records.

    ``kind`` is one of the :data:`DETECTOR_SPECS` kinds; ``params`` overrides the
    detector's defaults (only the keys present are passed; the rest use bioamla
    defaults). Slow on long files -- run inside a :class:`~magpy.workers.Worker`.
    """
    if kind not in _DETECTOR_CLASSES:
        raise ValueError(f"Unknown detector kind: {kind!r}")
    cls = _DETECTOR_CLASSES[kind]
    detector = cls(**_coerce_params(kind, params or {}))
    detections: list[Detection] = detect_all(_to_mono(audio), sample_rate, [detector])
    label = detector_label(kind)
    return [
        Candidate(
            start_time=d.start_time,
            end_time=d.end_time,
            confidence=d.confidence,
            low_freq=d.frequency_low,
            high_freq=d.frequency_high,
            label=d.label,
            detector=label,
        )
        for d in detections
    ]
