# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

MagPy is a **PyQt6 GUI wrapper around the `bioamla` library** (built against the
bioamla 0.2.x API; requires >= 0.2.3). bioamla owns the bioacoustics domain;
MagPy owns the GUI and the mutable state a functional library can't hold.
**Read `ARCHITECTURE.md`** — it is the source of truth for layering, the verified
bioamla findings, and the open decisions. This file is the short version.

## Workflow

- **The user handles all commits.** Make changes in the working tree and stop
  there. Do not run `git commit`, and do not offer to commit, unless the user
  explicitly asks. (Branching/pushing likewise only on request.)

## Commands

The project uses **`uv`**; a `Makefile` wraps the common tasks:

```bash
make sync      # uv sync -- create/refresh .venv, install deps + dev group
make run       # launch the GUI (magpy-gui -> magpy.app:main); FILE=x.wav opens a file
make test      # uv run pytest
make lint      # uv run ruff check src tests
make fmt       # ruff format + ruff check --fix (src, tests)
make check     # lint + format check + test
make help      # list targets
```

`uv run python -m magpy [file]` also launches the GUI. Run a single test with
`uv run pytest tests/test_x.py::test_y`. Python is `>=3.10`; env interpreter may
be newer — uv provisions per `requires-python`.

## Dependency philosophy (enforced in pyproject.toml)

Runtime deps are deliberately minimal: **`bioamla` + the Qt stack (`PyQt6`,
`pyqtgraph`) + `numpy`** (declared because services/widgets pass arrays directly).
Everything else — scipy, librosa, soundfile, pandas, matplotlib, the torch/
transformers ML stack, sounddevice — arrives **transitively via bioamla** and is
not declared here. Before adding a dependency, check it isn't already transitive.
bioamla pulls the full ML stack unconditionally (~200 packages; no `[ml]` extra),
so referencing ML is cheap but heavy work must be threaded.

**Push-to-bioamla:** if you need true bioacoustics-domain functionality, add it
to bioamla, not MagPy. MagPy contains only GUI/interaction concerns.

## Architecture (see ARCHITECTURE.md for the full version)

MVVM with a thin **services seam**. Layers under `src/magpy/`:

- **`services/`** — thin wrappers over bioamla. **The only layer that imports
  bioamla.** Owns GUI defaults; returns MagPy-owned DTOs (`LoadedAudio`,
  `SpectrogramImage`) so bioamla types don't leak. Stateless, not Qt-aware.
- **`workers/`** — `QThreadPool` bridge (`Worker`/`WorkerSignals`). Runs blocking
  bioamla calls off-thread; maps bioamla's `on_progress(done,total)` →
  `progress` signal. Cancellation is **cooperative** (raise inside the callback;
  bioamla has no `should_cancel`).
- **`models/`** — mutable app state as `QObject`s with signals (`Document`).
- **`widgets/`** — dumb Qt views. **Never import bioamla**; bind to models.
  Includes `NavigationBar`, ported from legacy.
- **`screens/`** — the multi-view workspace, one per nav button, all real: Home,
  Audio (annotation), Indices, Datasets, Training, Explore, Batch, the four
  catalogs, Hugging Face, Settings. Pure Qt, no bioamla. The two audio screens
  share `BaseAudioScreen` (waveform + spectrogram + transport + playback).
- **`main_window.py` / `app.py`** — the shell: a left nav bar switching a
  `QStackedWidget` of screens, the Workspace dock, and the dark theme (`theme.py`).
- **`settings.py`** — `app_settings()`, the only place a `QSettings` is built.

### Rules

- **Widgets/screens never import bioamla** — go through `services/` (compute) and
  `workers/` (threading). This is what contains bioamla's churn to one layer.
- **Don't thread interactive compute** — STFT of a view window is ~50 ms
  (services pin `backend="librosa"` to avoid a ~18 s torch warmup). Thread the
  `batch_*` ops, ML, measurements, and catalog downloads.
- **Render the visible window, never the whole file.** `SpectrogramView` asks for
  `(t0, t1, max_cols)` and the screen answers with `render_spectrogram`; the
  waveform does the same for its envelope. Don't add a whole-file array to the
  interactive path — recordings can be hours long.
- **Annotation edits go through `AnnotationSet`** (`add`/`remove`/`update`/
  `update_many`/`set_all`) so undo, autosave, and the views stay in step.
- **Never construct `QSettings` directly** — use `magpy.settings.app_settings()`.
  Tests redirect it (`tests/conftest.py`); a direct `QSettings("MagPy", "MagPy")`
  writes to the developer's real preferences.
- **Tests must not play audio or block on a dialog.** Assert playback through
  `PlaybackController` with a fake player; a modal `QMessageBox` hangs a headless
  run forever.
- **`bioamla.core.*` / `bioamla.controllers` no longer exist.** The current API
  is flat: `bioamla.{audio,viz,indices,detect,datasets,ml,cluster,catalogs,batch,system}`.

## legacy/

`legacy/magpy/` is the previous generation, built on the removed bioamla API
(`bioamla.controllers`, `core.*`). **It does not import and is out of the build
path** — reference only, for mining UI ideas. Don't wire it into the new package.
