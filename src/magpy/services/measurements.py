"""
Measurement service -- per-annotation acoustic measurements for the selection
table, over ``bioamla.datasets.compute_measurements``.

Owns two MagPy-side things so the table never imports bioamla:

* :data:`MEASUREMENTS` -- a display description (title, unit, decimals, what it
  means) for each metric bioamla can compute, in bioamla's own domain-grouped
  order. The column chooser and the table headers are built from it. A metric
  bioamla adds later still shows up, with a title derived from its key.
* :func:`measure_annotations` -- measure many annotations of one recording.

bioamla's function takes an audio *path* and decodes the whole file on every
call, which is the wrong shape for a GUI that already holds the samples and
re-measures as boxes are dragged (and ruinous on a long recording). So each
annotation's clip is written to a small temporary WAV and measured there, with
the annotation shifted to start at zero. Every metric is relative to the region,
so the shift changes nothing -- and the cost stays proportional to the clip, not
to the recording.
"""

from __future__ import annotations

import csv
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# bioamla import is confined to the services layer.
from bioamla.audio import save_audio
from bioamla.datasets import ALL_METRICS, compute_measurements
from bioamla.datasets import create_annotation as _create_bio_annotation

from .annotations import Annotation
from .spectrogram import to_mono


@dataclass(frozen=True)
class MeasurementSpec:
    """How one measurement is presented (the value itself comes from bioamla)."""

    key: str
    title: str
    unit: str = ""
    decimals: int = 3
    description: str = ""

    @property
    def header(self) -> str:
        return f"{self.title} ({self.unit})" if self.unit else self.title

    def format(self, value: float | None) -> str:
        return "" if value is None else f"{value:.{self.decimals}f}"


_KNOWN = (
    MeasurementSpec("duration", "Duration", "s", 3, "Length of the selection"),
    MeasurementSpec("zero_crossing_rate", "Zero-crossing rate", "", 4, "Sign changes per sample"),
    MeasurementSpec("peak_time", "Peak time", "s", 3, "Time of the loudest sample, from the start"),
    MeasurementSpec("rms", "RMS", "", 4, "Root-mean-square amplitude (linear)"),
    MeasurementSpec("peak", "Peak", "", 4, "Largest absolute amplitude (linear)"),
    MeasurementSpec("crest_factor", "Crest factor", "", 2, "Peak divided by RMS"),
    MeasurementSpec("rms_db", "RMS level", "dBFS", 1, "RMS amplitude in dB re full scale"),
    MeasurementSpec("peak_db", "Peak level", "dBFS", 1, "Peak amplitude in dB re full scale"),
    MeasurementSpec("crest_factor_db", "Crest factor", "dB", 1, "Peak level minus RMS level"),
    MeasurementSpec("dynamic_range", "Dynamic range", "dB", 1, "Spread between loud and quiet"),
    MeasurementSpec("avg_power", "Avg power", "", 5, "Mean squared amplitude"),
    MeasurementSpec("max_power", "Max power", "", 5, "Largest squared amplitude"),
    MeasurementSpec("energy", "Energy", "", 2, "Sum of squared samples"),
    MeasurementSpec("bandwidth", "Box bandwidth", "Hz", 0, "High minus low frequency of the box"),
    MeasurementSpec("centroid", "Center freq", "Hz", 0, "Power-weighted mean frequency"),
    MeasurementSpec("bandwidth_spectral", "Spectral spread", "Hz", 0, "Spread about the centroid"),
    MeasurementSpec("rolloff", "Rolloff 85%", "Hz", 0, "Frequency below which 85% of power lies"),
    MeasurementSpec("peak_frequency", "Peak freq", "Hz", 0, "Frequency with the most power"),
    MeasurementSpec("freq_q1", "Freq 25%", "Hz", 0, "First-quartile frequency"),
    MeasurementSpec("freq_q3", "Freq 75%", "Hz", 0, "Third-quartile frequency"),
    MeasurementSpec("freq_5", "Freq 5%", "Hz", 0, "Frequency below which 5% of power lies"),
    MeasurementSpec("freq_95", "Freq 95%", "Hz", 0, "Frequency below which 95% of power lies"),
    MeasurementSpec("bandwidth_90", "Bandwidth 90%", "Hz", 0, "Freq 95% minus Freq 5%"),
    MeasurementSpec("bandwidth_iqr", "IQR bandwidth", "Hz", 0, "Freq 75% minus Freq 25%"),
    MeasurementSpec("spectral_entropy", "Spectral entropy", "", 3, "Disorder of the spectrum"),
    MeasurementSpec("temporal_entropy", "Temporal entropy", "", 3, "Disorder of the envelope"),
    MeasurementSpec("pfc_min", "PFC min", "Hz", 0, "Lowest point of the peak-frequency contour"),
    MeasurementSpec("pfc_max", "PFC max", "Hz", 0, "Highest point of the peak-frequency contour"),
    MeasurementSpec("pfc_mean", "PFC mean", "Hz", 0, "Mean of the peak-frequency contour"),
    MeasurementSpec("pfc_start", "PFC start", "Hz", 0, "Peak frequency at the start"),
    MeasurementSpec("pfc_end", "PFC end", "Hz", 0, "Peak frequency at the end"),
    MeasurementSpec("pfc_slope", "PFC slope", "Hz/s", 0, "Overall slope of the contour"),
)
_KNOWN_BY_KEY = {spec.key: spec for spec in _KNOWN}

# Everything bioamla can compute, in its order; unknown keys get a derived title.
MEASUREMENTS: tuple[MeasurementSpec, ...] = tuple(
    _KNOWN_BY_KEY.get(key) or MeasurementSpec(key, key.replace("_", " ").capitalize())
    for key in ALL_METRICS
)
MEASUREMENTS_BY_KEY = {spec.key: spec for spec in MEASUREMENTS}
DEFAULT_MEASUREMENTS = ("duration", "peak_frequency", "centroid", "bandwidth_90", "rms_db")


def measure_annotations(
    samples: np.ndarray,
    sample_rate: int,
    annotations: list[Annotation],
    keys: list[str] | tuple[str, ...],
) -> dict[str, dict[str, float]]:
    """Measure each annotation; returns ``{annotation.id: {metric: value}}``.

    Metrics bioamla cannot compute for a region (too short, empty band) are
    simply absent from that annotation's dict. An annotation that fails outright
    gets an empty dict rather than aborting the rest.
    """
    keys = [k for k in keys if k in MEASUREMENTS_BY_KEY]
    if not keys or not annotations:
        return {a.id: {} for a in annotations}
    mono = to_mono(samples)
    results: dict[str, dict[str, float]] = {}
    with tempfile.TemporaryDirectory(prefix="magpy-measure-") as tmp:
        clip_path = str(Path(tmp) / "clip.wav")
        for ann in annotations:
            i0 = max(0, int(ann.start_time * sample_rate))
            i1 = min(len(mono), int(ann.end_time * sample_rate))
            if i1 - i0 < 2:
                results[ann.id] = {}
                continue
            try:
                save_audio(
                    clip_path, np.ascontiguousarray(mono[i0:i1], dtype=np.float32), sample_rate
                )
                region = _create_bio_annotation(
                    start_time=0.0,
                    end_time=(i1 - i0) / sample_rate,
                    label=ann.label,
                    low_freq=ann.low_freq,
                    high_freq=ann.high_freq,
                )
                values = compute_measurements(region, clip_path, keys)
            except Exception:  # noqa: BLE001 - one bad region must not sink the table
                results[ann.id] = {}
                continue
            results[ann.id] = {k: float(v) for k, v in values.items() if np.isfinite(v)}
    return results


def export_measurements_csv(
    annotations: list[Annotation],
    measurements: dict[str, dict[str, float]],
    keys: list[str] | tuple[str, ...],
    path: str | Path,
) -> None:
    """Write the selection table -- annotation fields plus measurement columns."""
    specs = [MEASUREMENTS_BY_KEY[k] for k in keys if k in MEASUREMENTS_BY_KEY]
    header = ["selection", "label", "start_time", "end_time", "low_freq", "high_freq"]
    header += ["confidence", "notes"] + [spec.key for spec in specs]
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(header)
        for number, ann in enumerate(annotations, start=1):
            values = measurements.get(ann.id, {})
            row = [number, ann.label, f"{ann.start_time:.6f}", f"{ann.end_time:.6f}"]
            row += ["" if v is None else f"{v:.3f}" for v in (ann.low_freq, ann.high_freq)]
            row += ["" if ann.confidence is None else f"{ann.confidence:.4f}", ann.notes]
            row += ["" if spec.key not in values else repr(values[spec.key]) for spec in specs]
            writer.writerow(row)
