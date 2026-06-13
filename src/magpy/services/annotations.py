"""
Annotation service -- the MagPy ``Annotation`` data type plus Raven/CSV I/O.

Per the architecture, MagPy owns its annotation type and bioamla's
``datasets.Annotation`` is only touched here, at the I/O boundary. The MagPy
``Annotation`` carries a stable ``id`` for UI tracking (bioamla's does not) and
is mutable (the user edits labels); it lives in the services layer alongside the
other DTOs (``LoadedAudio``, ``SpectrogramImage``) so models/widgets depend on
it without importing bioamla.

Format note: Raven selection tables preserve time/frequency/label but **drop
``confidence`` and ``notes``** (the format has no columns for them). CSV is
full-fidelity. Dispatch is by file extension.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from uuid import uuid4

# bioamla import is confined to the services layer.
from bioamla.datasets import (
    Annotation as _BioAnnotation,
    create_annotation as _create_bio_annotation,
    load_csv_annotations,
    load_raven_selection_table,
    save_csv_annotations,
    save_raven_selection_table,
)


def _new_id() -> str:
    return uuid4().hex[:8]


@dataclass
class Annotation:
    """A committed time-frequency annotation. MagPy-owned and mutable."""

    start_time: float
    end_time: float
    low_freq: float | None = None
    high_freq: float | None = None
    label: str = ""
    confidence: float | None = None
    notes: str = ""
    channel: int = 1
    id: str = field(default_factory=_new_id)

    @property
    def duration(self) -> float:
        return self.end_time - self.start_time


def _to_bioamla(a: Annotation) -> _BioAnnotation:
    # create_annotation validates (e.g. start < end) at the boundary.
    return _create_bio_annotation(
        start_time=a.start_time,
        end_time=a.end_time,
        label=a.label,
        low_freq=a.low_freq,
        high_freq=a.high_freq,
        channel=a.channel,
        confidence=a.confidence,
        notes=a.notes,
    )


def _from_bioamla(b: _BioAnnotation) -> Annotation:
    return Annotation(
        start_time=b.start_time,
        end_time=b.end_time,
        low_freq=b.low_freq,
        high_freq=b.high_freq,
        label=b.label,
        confidence=b.confidence,
        notes=b.notes,
        channel=b.channel,
    )


def _is_csv(path: Path) -> bool:
    return path.suffix.lower() == ".csv"


def load_annotations(path: str | Path) -> list[Annotation]:
    """Load annotations from a Raven selection table (.txt) or CSV (.csv)."""
    path = Path(path)
    bio = load_csv_annotations(str(path)) if _is_csv(path) else load_raven_selection_table(str(path))
    return [_from_bioamla(b) for b in bio]


def save_annotations(annotations: list[Annotation], path: str | Path) -> None:
    """Save annotations as CSV (.csv, full fidelity) or Raven table (otherwise)."""
    path = Path(path)
    bio = [_to_bioamla(a) for a in annotations]
    if _is_csv(path):
        save_csv_annotations(bio, str(path))
    else:
        save_raven_selection_table(bio, str(path))
