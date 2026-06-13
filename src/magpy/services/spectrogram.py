"""
Spectrogram service -- thin wrapper over ``bioamla.viz`` for the GUI.

This is part of the services seam: it is the only place (besides the other
``services`` modules) that imports bioamla, and it owns MagPy's "GUI defaults".

Two such defaults are deliberate and load-bearing:

* ``backend="librosa"`` -- ``bioamla.viz.compute_stft`` defaults to ``"auto"``,
  which selects torch when available. The *first* torch op in a process pays a
  multi-second initialization cost (~18s observed) and contends for the GPU.
  For interactive rendering we pin a CPU backend: steady-state STFT of 30s of
  48 kHz audio is ~50 ms, fast enough to compute synchronously on the UI thread.
* dB-scaled magnitude -- raw STFT magnitude is not display-ready; we convert to
  dB once here so widgets receive a render-ready array.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# bioamla import is confined to the services layer.
from bioamla.viz import compute_stft, spectrogram_to_db

# CPU backend: predictable latency, no torch warmup / GPU contention. See module docstring.
_BACKEND = "librosa"


@dataclass(frozen=True)
class SpectrogramImage:
    """
    A render-ready spectrogram. MagPy-owned (no bioamla types leak past here),
    so widgets can consume it without importing bioamla.

    Attributes:
        db: dB-scaled magnitude, shape ``(n_freqs, n_times)``.
        freqs: Frequency axis in Hz, length ``n_freqs``.
        times: Time axis in seconds, length ``n_times``.
    """

    db: np.ndarray
    freqs: np.ndarray
    times: np.ndarray

    @property
    def f_max(self) -> float:
        """Highest frequency bin in Hz."""
        return float(self.freqs[-1])

    @property
    def duration(self) -> float:
        """Time extent in seconds."""
        return float(self.times[-1])


def compute_spectrogram(
    audio: np.ndarray,
    sample_rate: int,
    *,
    n_fft: int = 2048,
    hop_length: int = 512,
    window: str = "hann",
    top_db: float = 80.0,
) -> SpectrogramImage:
    """
    Compute a render-ready dB spectrogram from a mono audio array.

    Fast enough (~50 ms for 30 s of audio) to call on the UI thread for a single
    view window. Use a worker for whole-file or batch spectrogram generation.
    """
    freqs, times, mag = compute_stft(
        audio,
        sample_rate,
        n_fft=n_fft,
        hop_length=hop_length,
        window=window,
        backend=_BACKEND,
    )
    db = spectrogram_to_db(mag, top_db=top_db)
    return SpectrogramImage(db=db, freqs=freqs, times=times)
