"""
Audio I/O service -- thin wrapper over ``bioamla.audio`` for loading audio.

Returns a MagPy-owned :class:`LoadedAudio` rather than bioamla's ``AudioData``
so that nothing past the services seam depends on a bioamla type. (The two
carry the same information today; the indirection is what absorbs bioamla churn.)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

# bioamla import is confined to the services layer.
from bioamla.audio import list_audio_files as _list_audio_files
from bioamla.audio import load_audio_data


@dataclass(frozen=True)
class LoadedAudio:
    """Decoded audio plus the metadata the GUI needs. MagPy-owned."""

    samples: np.ndarray  # mono or (channels, samples)
    sample_rate: int
    path: Path

    @property
    def duration(self) -> float:
        """Duration in seconds."""
        n = self.samples.shape[-1]
        return n / self.sample_rate if self.sample_rate else 0.0


def find_audio_files(directory: str | Path, *, recursive: bool = True) -> list[Path]:
    """List the audio files under ``directory`` (sorted), for workspace discovery."""
    return sorted(Path(p) for p in _list_audio_files(Path(directory), recursive=recursive))


def load_audio(path: str | Path) -> LoadedAudio:
    """Load an audio file into a :class:`LoadedAudio`."""
    path = Path(path)
    data = load_audio_data(path)
    return LoadedAudio(samples=data.samples, sample_rate=data.sample_rate, path=path)
