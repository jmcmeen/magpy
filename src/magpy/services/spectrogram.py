"""
Spectrogram service -- thin wrapper over ``bioamla.viz`` for the GUI.

This is part of the services seam: it is the only place (besides the other
``services`` modules) that imports bioamla, and it owns MagPy's "GUI defaults".

The GUI renders **the visible time window**, not the whole file:
:func:`render_spectrogram` takes a time range and a column budget, and picks the
hop so the result never exceeds that budget. Cost is therefore bounded by the
screen width rather than by the recording length -- an hour-long file renders as
fast as a ten-second one, and zooming in re-renders at finer time resolution
until the hop reaches the one implied by :class:`SpectrogramParams`.

Deliberate, load-bearing defaults:

* ``backend="librosa"`` -- ``bioamla.viz.compute_stft`` defaults to ``"auto"``,
  which selects torch when available. The *first* torch op in a process pays a
  multi-second initialization cost (~18s observed) and contends for the GPU.
  For interactive rendering we pin a CPU backend: a screen-width STFT is a few
  tens of milliseconds, fast enough to compute synchronously on the UI thread.
* A **fixed dB reference** (0 dB = a full-scale sine) rather than "relative to
  this image's maximum". With a per-image reference the colours would shift every
  time the view moved; a fixed one keeps levels stable across re-renders and
  makes the brightness/contrast controls mean the same thing everywhere.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

# bioamla import is confined to the services layer.
from bioamla.viz import compute_stft, spectrogram_to_db

# CPU backend: predictable latency, no torch warmup / GPU contention. See module docstring.
_BACKEND = "librosa"

FFT_SIZES = (256, 512, 1024, 2048, 4096, 8192)
OVERLAPS = (0.5, 0.75, 0.875, 0.9375)
WINDOWS = ("hann", "hamming", "blackman", "bartlett", "kaiser", "rectangular")


@dataclass(frozen=True)
class SpectrogramParams:
    """The user-tunable STFT settings (the time/frequency resolution trade-off).

    Attributes:
        n_fft: Window size in samples. Larger = finer frequency, coarser time.
        overlap: Fraction of the window shared by consecutive frames (0-1).
        window: Window function name (one of :data:`WINDOWS`).
    """

    n_fft: int = 1024
    overlap: float = 0.75
    window: str = "hann"

    @property
    def hop_length(self) -> int:
        """Samples between frames at full resolution."""
        return max(1, round(self.n_fft * (1.0 - self.overlap)))


@dataclass(frozen=True)
class SpectrogramImage:
    """
    A render-ready spectrogram. MagPy-owned (no bioamla types leak past here),
    so widgets can consume it without importing bioamla.

    Attributes:
        db: dB-scaled magnitude, shape ``(n_freqs, n_times)``.
        freqs: Frequency axis in Hz, length ``n_freqs``.
        times: Time of each column in seconds from the start of the file.
        t_start: Left edge of the rendered window (defaults to 0).
        t_end: Right edge of the rendered window (defaults to the last column).
        full_resolution: ``True`` when the hop is the one the params ask for, i.e.
            zooming further in cannot reveal more time detail.
    """

    db: np.ndarray
    freqs: np.ndarray
    times: np.ndarray
    t_start: float | None = None
    t_end: float | None = None
    full_resolution: bool = True

    @property
    def f_max(self) -> float:
        """Highest frequency bin in Hz."""
        return float(self.freqs[-1])

    @property
    def t0(self) -> float:
        """Left edge of the image in seconds."""
        return 0.0 if self.t_start is None else self.t_start

    @property
    def t1(self) -> float:
        """Right edge of the image in seconds."""
        return float(self.times[-1]) if self.t_end is None else self.t_end

    @property
    def duration(self) -> float:
        """Time extent in seconds."""
        return self.t1 - self.t0


def to_mono(samples: np.ndarray) -> np.ndarray:
    """Collapse ``(channels, samples)`` / ``(samples, channels)`` audio to mono."""
    if samples.ndim == 1:
        return samples
    return samples.mean(axis=int(np.argmin(samples.shape)))


def _to_db(mag: np.ndarray, n_fft: int) -> np.ndarray:
    # A full-scale sine through a Hann window peaks at n_fft / 4; using that as
    # the reference puts the scale in (approximate) dBFS for every FFT size.
    ref = n_fft / 4.0
    return spectrogram_to_db(mag**2, ref=ref**2, amin=1e-20, top_db=None).astype(np.float32)


def render_spectrogram(
    samples: np.ndarray,
    sample_rate: int,
    t0: float = 0.0,
    t1: float | None = None,
    params: SpectrogramParams | None = None,
    *,
    max_cols: int = 2000,
) -> SpectrogramImage:
    """
    Render the spectrogram of ``samples`` between ``t0`` and ``t1`` seconds.

    The hop is the larger of the one ``params`` implies and the one that keeps
    the image within ``max_cols`` columns, so the cost is bounded by the column
    budget regardless of how long the window is. Zoomed far out, frames are
    spaced wider than the window (an honest overview, not every sample is
    visited); zoomed in, the full requested resolution is used.
    """
    params = params or SpectrogramParams()
    mono = to_mono(samples)
    total = mono.shape[-1]
    duration = total / sample_rate if sample_rate else 0.0
    t1 = duration if t1 is None else t1
    t0 = min(max(0.0, t0), duration)
    t1 = min(max(t0, t1), duration)

    i0 = int(math.floor(t0 * sample_rate))
    i1 = max(i0 + 1, int(math.ceil(t1 * sample_rate)))
    hop = max(params.hop_length, math.ceil((i1 - i0) / max(1, max_cols)))

    # Give the first/last frames real audio to look at where the file has it,
    # instead of the padding a centred STFT would otherwise invent at the edges.
    lead = min(i0, (params.n_fft // 2 // hop + 1) * hop)
    start = i0 - lead
    stop = min(total, i1 + params.n_fft // 2)
    freqs, times, mag = compute_stft(
        np.ascontiguousarray(mono[start:stop], dtype=np.float32),
        sample_rate,
        n_fft=params.n_fft,
        hop_length=hop,
        window=params.window,
        backend=_BACKEND,
    )
    times = times + start / sample_rate
    first = lead // hop
    last = max(first + 1, min(mag.shape[1], first + math.ceil((i1 - i0) / hop)))
    mag, times = mag[:, first:last], times[first:last]

    half_col = hop / sample_rate / 2.0
    return SpectrogramImage(
        db=_to_db(mag, params.n_fft),
        freqs=freqs,
        times=times,
        t_start=float(times[0]) - half_col,
        t_end=float(times[-1]) + half_col,
        full_resolution=hop == params.hop_length,
    )


def compute_spectrogram(
    audio: np.ndarray,
    sample_rate: int,
    *,
    n_fft: int = 1024,
    hop_length: int = 256,
    window: str = "hann",
) -> SpectrogramImage:
    """Whole-signal spectrogram at a fixed hop (for short clips and exports).

    Interactive views should use :func:`render_spectrogram`, which bounds the
    cost by the visible window.
    """
    params = SpectrogramParams(n_fft=n_fft, overlap=1.0 - hop_length / n_fft, window=window)
    return render_spectrogram(audio, sample_rate, params=params, max_cols=10**9)
