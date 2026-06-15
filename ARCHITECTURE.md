# MagPy Architecture

MagPy is a **PyQt6 GUI wrapper around [bioamla](https://github.com/jmcmeen/bioamla)**.
bioamla owns the bioacoustics domain; MagPy owns the interactive UI and the
mutable state a functional library cannot hold. This document records the
architecture and the empirical findings that justify it. It is the source of
truth for the rebuild; `CLAUDE.md` is the short version.

## The shape of bioamla 0.2.0 (why the architecture is what it is)

bioamla is an almost entirely **functional, stateless** library. Its public API
is pure functions over `(numpy array, sample_rate)` or file paths, returning
dataclasses (`AudioData`, `AudioAnalysis`, `AcousticIndices`, `Detection`,
`Annotation`, `AnnotationSet`, …). There is **no session/controller/project
object** to hold. Three consequences drive the design:

1. **All mutable state lives in MagPy.** "The file the user is editing", the
   current selection, view settings — bioamla won't track these, so MagPy needs
   its own model layer (`models/`).
2. **bioamla already implements everything the old MagPy `core/` reimplemented**:
   audio I/O + transforms (`bioamla.audio`), STFT/mel spectrograms
   (`bioamla.viz`), acoustic indices (`bioamla.indices`), detectors
   (`bioamla.detect`), the annotation model + Raven/CSV/JSON I/O and
   measurements (`bioamla.datasets`), ML inference/training (`bioamla.ml`),
   clustering (`bioamla.cluster`), catalog downloads (`bioamla.catalogs`), and a
   playback engine (`bioamla.audio.AudioPlayer`). MagPy must **not** re-implement
   any of it — it wraps it.
3. **bioamla will churn.** It is being rebuilt in parallel. The services seam
   (below) exists specifically to contain that churn in one layer.

## Layering

```
bioamla (external)         functional domain engine
   ▲
services/   thin wrappers over bioamla. THE ONLY LAYER THAT IMPORTS BIOAMLA.
            Owns GUI defaults; converts bioamla dataclasses ↔ MagPy DTOs.
            Mostly stateless functions; the exception is the playback `Player`,
            which wraps an inherently stateful engine. Not Qt-aware.
workers/    QThreadPool bridge: runs a blocking bioamla call off-thread, maps
            on_progress → Qt signal, cooperative cancel. The ONE place blocking
            calls touch the event loop.
models/     mutable app state as QObjects with signals (the MVVM "model"):
            Document (current audio, later: annotations/selection/view), and a
            MagPy-owned Selection/Annotation model.
widgets/    dumb Qt views. Bind to models; emit user-intent. NEVER import bioamla.
screens/    compose widgets + a view-model into a workspace view (not built yet).
main_window app shell: navigation, screen stack, wiring.
```

**Pattern: MVVM with a thin services seam.** Qt's signal/slot system already *is*
data binding, and a functional backend means the "model" is mutable app state
that views observe — so MVVM's QObject view-model is the idiomatic fit. (MVC
controllers would add a layer without the binding payoff.)

### The one rule that matters

**Widgets and screens never import bioamla.** They go through `services/` for
compute and `workers/` for threading, and hold state in `models/`. bioamla types
do not leak past the services seam — services return MagPy-owned DTOs
(`LoadedAudio`, `SpectrogramImage`, …). This is what lets bioamla move without
the UI noticing.

## Verified findings (measured against bioamla 0.2.0, not assumed)

These were measured before the seam was designed; re-verify if bioamla changes.

| Finding | Measurement | Design consequence |
|---|---|---|
| STFT is fast at steady state | `viz.compute_stft` 30s@48k ≈ **50 ms** on `librosa`/`scipy` backends | Spectrogram of a view window computes **synchronously** on the UI thread. |
| `backend="auto"` triggers torch init | first call ≈ **18 s** (torch warmup + GPU) | Services **pin `backend="librosa"`** for interactive rendering; keep `auto`/`torch` off the interactive path. |
| `bioamla.ml` imports lazily | `import bioamla.ml` ≈ **0.01 s**; torch loads only on first ML call | ML stays cheap to reference; do heavy work in a worker; lazy-import ML submodules at screen activation. |
| `AudioPlayer` callbacks fire on the audio thread | `play/pause/stop/seek/position/state`; non-blocking `play()`; `on_position_change`/`on_complete` callbacks run on sounddevice's stream thread | Don't use those callbacks for UI. Wrap in a `QObject` + `QTimer` that **polls** `position`/`state` on the UI thread and emits `positionChanged`/`stateChanged`. |
| Long ops expose progress, not cancel | `batch_*`/`train_ast`/downloads take `on_progress(done,total)` + `max_workers`; **no** `should_cancel` | Worker maps `on_progress`→signal; **cancellation is cooperative** (raise inside the callback at item boundaries). Ops without progress can't be cancelled mid-flight. |

**Threading threshold:** thread the `batch_*` functions, ML inference/training,
and catalog downloads (all already have `on_progress`/`max_workers`), plus
possibly long-file loads. Do **not** thread per-view-window spectrogram/indices.

## Decisions

- **Annotations:** MagPy owns a lightweight `Selection`/`Annotation` model in
  `models/`; `services/` converts to/from `bioamla.datasets.Annotation` only at
  the Raven/CSV I/O boundary. This keeps bioamla types out of widgets and
  preserves the transient-selection (editing/colour/selected UI state) vs.
  committed-annotation distinction that bioamla's frozen dataclass won't carry.
  *(Model not built yet — lands with the annotation slice.)*
- **Project = Workspace, MagPy-owned, workspace-always** (decided; built).
  bioamla 0.2.0 has no `project` module, so MagPy owns the concept. A **Workspace**
  is a `.magpy/` **directory bundle** (`services.workspace_io`: `workspace.json`
  manifest + `annotations/`, `imported/`, `datasets/`, `models/`). The app is
  *always* in a workspace — it reopens the last-used bundle, or auto-creates a
  **Scratch** workspace in `AppDataLocation` — so there is no "nothing open" state
  and no startup gate. (This **supersedes** the earlier "folder pointer + discovery,
  no manifest yet" decision.)
  - **One primitive — the `Artifact`** (`{id, kind, mode, path}`): everything a
    workspace points at (audio file, folder, and later dataset/model) is either
    **`linked`** (referenced by path, *never copied*, shareable across workspaces)
    or **`imported`** (copied into the bundle's `imported/`). The File menu's *Add
    Audio/Folder* link; *Import* copies (with a warning). This closes the
    "build a dataset / train a model in one workspace, use it in another" gap:
    workspace B just *links* A's artifact.
  - **Annotations live in the bundle**, as Raven/CSV sidecars under `annotations/`
    keyed per *file* (`<artifactId>.csv`; folder-discovered files get a composite
    `<folderId>_<relpath>` key; storage is purpose-suffix-ready,
    `<key>__<purpose>.csv`). The shell autosaves them on every `AnnotationSet`
    change and reloads on file activate (suspending autosave while it populates).
    *Import/Export* annotations remain for outside-world interchange.
  - `Workspace` and `Document` stay separate; the shell mediates the
    "activate a file → load audio + its annotations" edge. The bundle format is
    MagPy-owned (`workspace_io` imports no bioamla) and could be promoted to bioamla
    later if it proves clean. Deferred: named annotation-set UI, relink-by-content.
- **Push-to-bioamla principle:** if MagPy needs functionality that is true
  bioacoustics-domain logic, add it to bioamla, not here. MagPy only contains
  GUI/interaction concerns. Gaps/awkwardness in the bioamla API/CLI we hit while
  building are logged in `bioamla_notes.md` (git-ignored) as the upstream queue.
- **Workflow & CLI→screen mapping** (decided). The bioamla CLI groups map onto
  screens around one principle: **source audio is immutable; analysis derives
  metadata from it; editing produces *new* audio.** Editing commands take
  `INPUT OUTPUT` (new file); analysis commands take just `FILE` (read-only). So
  "indices go stale after an edit" can't happen — an edit is a new artifact with
  its own (empty) derived data, not a mutation. This is why **editing lives only in
  Datasets** (preprocessing/augmentation), never as a free tool over analysis
  subjects. Mapping:
  - **Audio = Analysis** (read-only, one source file): spectrogram/waveform/
    transport/annotations (built) + `detect ribbit/energy/peaks/accelerating`
    (→ a **reviewable candidate layer**, distinct from curated annotations, that
    the user promotes into the `AnnotationSet`) + `indices compute` (summary) and
    `indices temporal` (curve along the timeline). All keyed to the immutable
    artifact. (Legacy precedent: detections became selections tagged `source=` +
    confidence — see `legacy/.../widgets/detection_results.py`; we keep them in a
    separate candidate set rather than mixing tagged rows.)
  - **Datasets** — *the home of audio editing*: `audio trim/normalize/denoise/
    filter/gain/resample/pitch-shift/time-stretch/add-noise/segment/convert` +
    `dataset augment/extract-clips/build/partition/merge/manifest/stats/license` +
    `annotation generate-labels/generate-frame-labels/remap/filter/convert`. Each
    transform emits a new artifact (tag provenance: source id + transform chain).
  - **Training** — `models ast` (train/eval/predict) over a dataset.
  - **Explore** — `cluster fit/reduce/analyze/novelty` (embedding-space).
  - **Batch** — the fan-out of the single-file ops above (`batch audio/detect/
    index/cluster/models`), threaded via `workers/`.
  - **Catalogs** (iNaturalist/eBird/Macaulay/Xeno-Canto/Hugging Face) —
    `catalogs inat/ebird/ml/xc/hf`: search/download → linked/imported artifacts.
  - **Settings** — `system devices/deps/version` + `.env`/tokens.
  - `util download/zip/unzip` is plumbing (no screen). Placeholder screens already
    state their mapped role on their Coming-Soon cards.

## Build status

- **Done — slice 1 (spectrogram):** open file → `services.load_audio` →
  `Document` → `services.compute_spectrogram` → `SpectrogramView` (pyqtgraph).
  Plus the `workers.Worker` threading bridge. Verified headless (offscreen Qt).
- **Done — slice 2 (playback):** `services.Player` (wraps `AudioPlayer`) →
  `models.PlaybackController` (QTimer poll → Qt signals) → `TransportBar` +
  a playhead line on `SpectrogramView`. Engine/wiring verified headless; actual
  audio output needs a device (not exercised in the headless harness).
- **Done — slice 3 (annotations):** MagPy-owned `Annotation` DTO + Raven/CSV I/O
  in `services.annotations` (bioamla `datasets.Annotation` touched only here);
  `models.AnnotationSet` (observable, owns selection); `AnnotationTable` (edit
  label, select, delete) and annotation overlays + a promotable selection region
  on `SpectrogramView`; import/export wired in the shell. Verified headless
  (create/edit/select-sync/Raven+CSV roundtrip/delete/clear-on-reopen). Known
  gap: Raven export drops confidence/notes (format limitation — use CSV for
  fidelity). *(Time-frequency box selection landed later — see slice 15.)*
- **Done — slice 4 (navigation shell):** the legacy layout ported back — a VS
  Code-style `NavigationBar` switching a `QStackedWidget` of `BaseScreen` views
  (Home dashboard + Datasets/Training/iNaturalist/Batch placeholders), the dark
  theme (`theme.py`), and an AUDIO view hosting slices 1–3 with docks shown only
  on that view. All screens are pure Qt (no bioamla); ported close to verbatim,
  not rewritten. Verified headless. PyQt6 pinned to the newest (6.11.0).
  Dropped for the first release (preserved in `legacy/`): the Pipeline/node-graph
  editor, the AI Wizard, and the Queue view.
- **Done — slice 5 (workspace, reworked to workspace-always):** `models.Workspace`
  is now manifest-backed (`.magpy/` bundle via `services.workspace_io`), holding
  `Artifact`s that are linked (referenced) or imported (copied). The shell always
  has one open (last-used or Scratch); the File menu does New/Open/Open-Recent +
  Add Audio/Folder (link) + Import (copy); the always-visible "Files" dock
  (`WorkspacePanel`) lists discovered audio with imported/missing tags and
  double-click loads a file plus its bundled annotations (autosaved on edit). Recent
  workspaces persist in `QSettings` and surface on the Home screen. Round-trip tests
  in `tests/` (first tests in the repo). See Decisions.
- **Menu bar is intentionally minimal.** Functionality lives in its view: the
  Audio view has a toolbar (Open Audio, New Selection / Add / Delete, Import /
  Export) — there is no Annotate menu. New views add their own controls, not menus.
- **Done — slice 6 (audio layout):** `WaveformView` (ported from the legacy
  widget, keeping its min/max envelope downsampling; bound to `AnnotationSet` +
  the shared playhead; Ctrl+click to seek) sits above the spectrogram; a
  `PropertiesPanel` (rewritten against `LoadedAudio`/`Annotation`) is tabified
  with the Annotations dock and tracks both the open file and the selected
  annotation (including in-place label/field edits). Verified headless.
- **Done — slice 7 (detect → candidate layer):** `services.detect` wraps
  `bioamla.detect` (the four detectors) and owns two MagPy types so the widget
  never imports bioamla: a `Candidate` DTO and a declarative `DETECTOR_SPECS`
  (label/default/range per param) the `DetectPanel` builds its form from
  generically. Detection runs off-thread through a `Worker` (energy detection of
  60 s ≈ 4 s); results fill an observable `models.CandidateSet` (separate from
  `AnnotationSet`, with a confidence filter) shown in the Detect dock; the user
  promotes reviewed candidates into the curated annotations (`candidate_to_
  annotation` does the `frequency_low/high` → `low_freq/high_freq` remap).
  Verified headless end-to-end (run → table → promote) + unit tests.
- **Done — slice 8 (acoustic indices):** `services.compute_indices` wraps
  `bioamla.indices.compute_all_indices` into a MagPy `IndexSummary` (whole-file,
  so it runs threaded — ~0.4 s/60 s); the `IndicesPanel` dock renders the scalar
  values with descriptions as tooltips. Verified headless + unit tests.
- **Done — slice 9 (Settings = system info + env editor):** the Settings screen
  shows `bioamla.system` version/dependency info (cheap, built in `_setup_ui`) and
  compute-device enumeration (threaded on first activate — `get_device_info`
  triggers the ~1.2 s torch import). Reworked to add an **API-key / environment
  editor**: an editable form over the vars bioamla reads (`XC_API_KEY`,
  `EBIRD_API_KEY`, `HF_TOKEN`, …), saving to MagPy's own `.env`
  (`services.env_io`, no bioamla) *and* applying to `os.environ` so catalogs work
  in-session. bioamla's `load_dotenv` searches site-packages, not MagPy's dir, so
  MagPy loads/applies its keys itself (the shell loads them at startup before any
  catalog call; keys are read lazily). Verified headless.
- **Done — slice 10 (catalogs, minus Hugging Face):** `services.catalogs` is the
  seam over `bioamla.catalogs` — search + download for Xeno-Canto / Macaulay /
  iNaturalist and recent-observation search for eBird — collapsing each source's
  bespoke result (iNaturalist returns **raw dicts**; eBird returns sightings, not
  audio) into one defensively-mapped `CatalogRecord`. **Download is by record id**
  so no bioamla object crosses the seam; keys come from env vars (e.g.
  `EBIRD_API_KEY`) with an optional form-field override. One config-driven
  `CatalogScreen` (`CatalogConfig`/`CatalogField`) drives all four nav views;
  search/download run threaded, and downloads are *linked* into the open
  workspace (so they surface in the Files panel). The network/key path can't be
  smoke-tested offline, so the **mapping** is unit-tested against faked return
  shapes; screen construction + render verified headless. (Removed: the legacy
  bioamla-CLI `TerminalScreen` and the iNaturalist placeholder.)
- **Fixed — screen lifecycle:** `BaseScreen.activate()`/`on_activate()` were
  defined but never called (the shell only did `setCurrentWidget`), so any screen
  populating in `on_activate` stayed blank. `MainWindow._on_view_changed` now calls
  `activate()` on the incoming screen, guarded by `isinstance(BaseScreen)` (the
  AUDIO view is a bare `QWidget`). Lazy, expensive screen population (e.g. Settings'
  device probe) now actually fires.
- **Done — slice 11 (Batch = fan-out of the single-file ops):** `services.batch`
  is the seam over bioamla's `batch` CLI group — all **17** ops (audio info/convert/
  resample/normalize/trim/filter/denoise/segment/visualize, detect energy/ribbit/
  peaks/accelerating, index, models predict/embed, cluster), each a **per-op
  adapter** (not a generic dispatcher) that localises bioamla's signature
  asymmetries. The crash-trap is contained by truthful `supports_progress`/
  `supports_max_workers` flags: `with_progress` is gated on the flag and
  `max_workers` is only ever a form field for ops whose function accepts it, so an
  unsupported kwarg can't be forwarded. For the three ops bioamla doesn't instrument
  (info/index/segment) the adapter loops per file and emits `on_progress` itself —
  giving a determinate bar *and* cooperative cancel where there was neither. Returns
  (`BatchResult`/stats-dict/loop) normalise to one `BatchOutcome`. The config-driven
  `BatchScreen` builds its form from `BatchParam` specs, runs through the `Worker`
  (**first real use of the progress + cancel path**), and can link outputs into the
  workspace. CSV-metadata mode is deferred. The audio ops are **run for real** on
  synthetic wavs (+ unit tests); models/cluster need downloads → spec-wired only.
- **Done — slice 12 (Training = `models ast`):** `services.training` wraps
  `train_ast` / `evaluate_directory` / `predict_file` into MagPy DTOs (curated
  `TRAIN_PARAMS` subset of train_ast's ~28 kwargs; only known keys forwarded). The
  `TrainingScreen` has Train / Evaluate / Predict tabs, all threaded. Training is
  **honest about its limits**: it has no progress/cancel hook, so the UI confirms
  before launch, shows an indeterminate bar, disables Train while running, and has
  **no Cancel button** (TensorBoard under `<training dir>/logs` is the real progress
  view). Heavy/model paths are construct-/wiring-verified, not run.
- **Done — slice 13 (Datasets = build training data):** `services.datasets` is the
  seam over bioamla's dataset-building fan-outs — `extract_labeled_dataset`,
  `partition_dataset`, `merge_datasets`, `batch_augment`, `get_dataset_stats`,
  `build_manifest_from_metadata`+`save_dataset_manifest`, `generate_license_for_
  dataset` — each a per-op adapter normalising bioamla's `dict` returns to one
  `DatasetOutcome`. The `DatasetsScreen` is **tabbed** (Extract clips / Partition /
  Augment / Merge & inspect) rather than an op-picker, because the inputs are
  heterogeneous (file-or-dir source + optional annotations; a dataset dir; multiple
  dirs). No bioamla op here reports progress, so runs use an indeterminate bar with
  no Cancel. Source audio stays immutable — every op writes a new dataset dir or
  sidecar. The ops are **run for real** on synthetic annotated audio (+ unit tests);
  `generate_license` only succeeds when the metadata carries attribution columns
  (else bioamla raises, surfaced as a run error). Reuses `screens/_form.py` for the
  augment param form. (This absorbs the dataset-building CLI ops; per-file *audio
  editing* — trim/normalize/… as single-file → new-artifact — is reachable today via
  the **Batch** screen's directory mode and is not duplicated here.)
- **Shared:** `screens/_form.py` (`make_field`/`read_field`/`collect_params` over the
  generic `BatchParam`) backs the Batch, Training, and Datasets parameter forms.
- **Done — slice 14 (Explore = embedding space):** `services.cluster` wraps
  `bioamla.cluster` (`load_embeddings_batch` → `cluster_embeddings` +
  `reduce_dimensions` → `analyze_clusters_summary` + optional `detect_novelty`)
  into one `cluster_embeddings_dir` call returning a render-ready
  `EmbeddingScatter` (2-D coords, per-point cluster labels, novelty indices,
  silhouette). The `ExploreScreen` loads a folder of `.npy` embeddings (from Batch
  → Model embeddings), runs cluster+reduce off-thread through the `Worker`
  (indeterminate; no progress hook), and draws a **pyqtgraph scatter coloured by
  cluster** (noise grey, novel points red-ringed) plus CSV export. Seam gotcha
  handled: `cluster_embeddings(...).labels` returns a `list`, so it's
  `np.asarray`-coerced before `analyze_clusters_summary` (which does `labels >= 0`).
  Run for real on synthetic embeddings (+ unit tests); PCA/UMAP/t-SNE all work.
- **All nav views are now real** except **Hugging Face** (intentionally deferred —
  the only remaining `PlaceholderScreen`).
- **Done — slice 15 (time-frequency box selection):** `SpectrogramView` gained a
  2-D `RectROI` selection (`start_box_selection`, key **B**) alongside the existing
  full-band time region (key **S**); `selection_bounds()` returns
  `(start, end, low_freq, high_freq)` (freqs `None` for a time region), normalised
  (start ≤ end, low ≤ high) and freq-clamped ≥ 0 so a backwards-dragged box can't
  invert. Committed annotations carrying frequency bounds now render as filled
  **boxes** in ViewBox data coords (verified they track pan/zoom), while time-only
  ones (imports, promoted detector candidates) stay full-height regions — both
  styled on selection. Added a live cursor **t/f readout**. The `Annotation` DTO
  already carried `low_freq/high_freq`, so this was view-layer only; freq bounds
  round-trip through CSV + Raven (verified). Qt tests in `tests/test_spectrogram_view.py`.
- **Done — slice 16 (spectrogram controls + interactive boxes):** a controls row on
  `SpectrogramView` adds a **colormap** picker, a **contrast** slider (raises the dB
  display floor via `ImageItem.setLevels`), **auto-scroll** (the view follows the
  playhead near the right edge during playback), and **zoom** in/out/fit (time axis)
  — all view-local. Annotations became **click-to-select** (hit-test the box/region
  under the cursor; empty click deselects), and the *selected* freq-box promotes to
  an editable/resizable `RectROI` (unselected ones stay filled static rects — no
  handle clutter, the verified look preserved); dragging it writes the new bounds
  back. The edit↔rebuild feedback loop is contained by symmetric guards: an
  `_editing` flag suppresses the rebuild our own `model.update` triggers (so the held
  ROI isn't torn down), and a `_syncing` flag guards the build path; the swap of
  representation falls out of routing `selectionChanged → _sync_annotations`.
  `sigRegionChangeFinished` (not `…Changed`) avoids per-pixel spam. Verified headless
  (ROI lands in data coords, demote-on-reselect, drag-writes-back, click-select,
  deselect) + Qt tests.
- **Next:** the shell still doubles as the AUDIO view-model — extract a dedicated
  one as it grows. Open polish: candidate overlays on the spectrogram, batch
  CSV-metadata mode, per-file audio editing on the Datasets screen (vs. Batch),
  point→file interaction on the Explore scatter (click a point to open its audio),
  and subprocess-isolated training (killable, crash-isolated).
- **Reference:** the previous generation is preserved under `legacy/magpy/`
  (built on the removed `bioamla.controllers`/`core.*` API; does not import).
  Mine it for UI ideas only.
