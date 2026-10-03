"""Tests for the viewport-bounded spectrogram renderer."""

from __future__ import annotations

import numpy as np

from magpy.services import SpectrogramParams, compute_spectrogram, render_spectrogram

SR = 48000


def _tone(seconds: float, freq: float = 4000.0) -> np.ndarray:
    t = np.arange(int(SR * seconds)) / SR
    return np.sin(2 * np.pi * freq * t).astype(np.float32)


def test_column_budget_bounds_the_image_whatever_the_length():
    audio = np.zeros(SR * 600, dtype=np.float32)  # ten minutes
    img = render_spectrogram(audio, SR, 0.0, 600.0, max_cols=1500)
    assert img.db.shape[1] <= 1500
    assert not img.full_resolution  # hop was widened to fit the budget
    assert img.t0 <= 0.0 and abs(img.t1 - 600.0) < 1.0


def test_zooming_in_reaches_the_requested_resolution():
    params = SpectrogramParams(n_fft=1024, overlap=0.75)
    img = render_spectrogram(_tone(10), SR, 3.0, 3.5, params, max_cols=2000)
    assert img.full_resolution
    assert abs(img.db.shape[1] - 0.5 * SR / params.hop_length) <= 2
    assert abs(img.t0 - 3.0) < 0.01 and abs(img.t1 - 3.5) < 0.01


def test_levels_are_dbfs_and_frequency_axis_is_right():
    img = render_spectrogram(_tone(2, 4000.0), SR)
    profile = img.db.mean(axis=1)
    assert abs(img.freqs[profile.argmax()] - 4000.0) < SR / 1024
    # A full-scale sine sits at ~0 dB re full scale (a little under: bin scalloping).
    assert -3.0 < float(img.db.max()) <= 0.5


def test_levels_do_not_depend_on_the_window_rendered():
    audio = _tone(10)
    whole = render_spectrogram(audio, SR, 0.0, 10.0, max_cols=500)
    part = render_spectrogram(audio, SR, 4.0, 5.0, max_cols=500)
    assert abs(float(whole.db.max()) - float(part.db.max())) < 1.0


def test_event_lands_at_its_time_in_a_window_that_starts_mid_file():
    audio = np.zeros(SR * 10, dtype=np.float32)
    audio[5 * SR] = 1.0
    img = render_spectrogram(audio, SR, 4.0, 6.0, max_cols=4000)
    column = img.db.max(axis=0).argmax()
    assert abs(float(img.times[column]) - 5.0) < 0.01


def test_stereo_is_downmixed():
    stereo = np.stack([_tone(1), _tone(1)])
    assert render_spectrogram(stereo, SR).db.shape == render_spectrogram(_tone(1), SR).db.shape


def test_out_of_range_window_is_clamped():
    img = render_spectrogram(_tone(1), SR, -5.0, 99.0)
    assert img.t0 <= 0.0 and img.t1 <= 1.01


def test_compute_spectrogram_keeps_full_resolution():
    img = compute_spectrogram(_tone(1), SR, n_fft=512, hop_length=128)
    assert img.full_resolution and img.db.shape[0] == 257
