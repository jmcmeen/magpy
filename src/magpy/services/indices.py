"""
Acoustic-indices service -- thin wrapper over ``bioamla.indices`` for the GUI.

Part of the services seam. Acoustic indices are summary scalars (ACI, ADI, …)
computed over the **whole file** (the architecture keys derived data to the
immutable artifact, not the view window), so this is *not* an interactive,
view-window computation: a multi-minute file takes seconds. Callers run
:func:`compute_indices` through a :class:`~magpy.workers.Worker`.

Returns a MagPy-owned :class:`IndexSummary` (and a render-ready
:meth:`IndexSummary.rows`) so bioamla's ``AcousticIndices`` does not leak past
the seam.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# bioamla import is confined to the services layer.
from bioamla.indices import AVAILABLE_INDICES, compute_all_indices, describe_index

# Human-readable labels for the indices the GUI shows, in display order.
_INDEX_LABELS: dict[str, str] = {
    "aci": "ACI",
    "adi": "ADI",
    "aei": "AEI",
    "bio": "Bioacoustic",
    "ndsi": "NDSI",
    "h_spectral": "Spectral entropy",
    "h_temporal": "Temporal entropy",
}


@dataclass(frozen=True)
class IndexRow:
    """One index value, ready to show (label + value + tooltip description)."""

    key: str
    label: str
    value: float | None
    description: str


@dataclass(frozen=True)
class IndexSummary:
    """Whole-file acoustic indices. MagPy-owned (no bioamla type leaks past here)."""

    values: dict[str, float | None]
    sample_rate: int
    duration: float

    def rows(self) -> list[IndexRow]:
        """Display rows in canonical order, with descriptions for tooltips."""
        return [
            IndexRow(
                key=key,
                label=_INDEX_LABELS.get(key, key.upper()),
                value=self.values.get(key),
                description=describe_index(key) or "",
            )
            for key in AVAILABLE_INDICES
        ]


def _to_mono(audio: np.ndarray) -> np.ndarray:
    if audio.ndim == 2:
        return audio.mean(axis=0)
    return audio


def compute_indices(
    audio: np.ndarray,
    sample_rate: int,
    *,
    include_entropy: bool = True,
) -> IndexSummary:
    """Compute the whole-file acoustic indices for ``audio``.

    Slow on long files (seconds) -- run inside a :class:`~magpy.workers.Worker`.
    ``include_entropy`` adds the spectral/temporal entropy indices (off by
    default in bioamla; on here so the GUI shows the full set).
    """
    result = compute_all_indices(_to_mono(audio), sample_rate, include_entropy=include_entropy)
    values: dict[str, float | None] = {
        "aci": result.aci,
        "adi": result.adi,
        "aei": result.aei,
        "bio": result.bio,
        "ndsi": result.ndsi,
        "h_spectral": result.h_spectral,
        "h_temporal": result.h_temporal,
    }
    return IndexSummary(
        values=values,
        sample_rate=result.sample_rate or sample_rate,
        duration=result.duration,
    )
