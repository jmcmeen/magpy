# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

MagPy is a **PyQt6 GUI wrapper around the `bioamla` library** (currently being
rebuilt ground-up against bioamla 0.2.0). bioamla owns the bioacoustics domain;
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
make run       # launch the GUI (magpy-gui -> magpy.app:main)
make test      # uv run pytest   (no tests/ yet)
make lint      # uv run ruff check src
make format    # uv run black src && ruff check --fix src
make check     # lint + test
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
bioamla 0.2.0 pulls the full ML stack unconditionally (~201 packages; no `[ml]`
extra), so referencing ML is cheap but heavy work must be threaded.

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
- **`screens/`** — the multi-view workspace (`BaseScreen` subclasses), ported
  from the legacy layout; most are still placeholders. Pure Qt, no bioamla.
  Nav views: Home, Audio, Datasets, Training, iNaturalist, Batch. (Pipeline/
  node-graph, AI Wizard, and Queue were dropped for the first release; their
  ported sources remain under `legacy/` if revisited.)
- **`main_window.py` / `app.py`** — the shell: a left nav bar switching a
  `QStackedWidget` of screens, dark theme (`theme.py`), and an AUDIO view (the
  rebuilt spectrogram/transport/annotation work) whose docks show only on it.

### Rules

- **Widgets/screens never import bioamla** — go through `services/` (compute) and
  `workers/` (threading). This is what contains bioamla's churn to one layer.
- **Don't thread interactive compute** — STFT of a view window is ~50 ms
  (services pin `backend="librosa"` to avoid a ~18 s torch warmup). Thread the
  `batch_*` ops, ML, and catalog downloads.
- **`bioamla.core.*` / `bioamla.controllers` no longer exist.** The current API
  is flat: `bioamla.{audio,viz,indices,detect,datasets,ml,cluster,catalogs,batch,system}`.

## legacy/

`legacy/magpy/` is the previous generation, built on the removed bioamla API
(`bioamla.controllers`, `core.*`). **It does not import and is out of the build
path** — reference only, for mining UI ideas. Don't wire it into the new package.
