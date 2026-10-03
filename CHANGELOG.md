# Changelog

All notable changes to this project are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- **Draw to annotate.** Dragging on the spectrogram draws a time-frequency box that becomes an
  annotation; dragging on the waveform makes a time-only one. Clicking selects, clicking empty
  space moves the playhead, double-clicking plays the box. The selected box has handles on
  every corner and edge; time-only annotations can be dragged too.
- **Viewport-rendered spectrogram.** The spectrogram is computed for the visible window at a
  resolution matched to the screen, so recordings of any length open quickly and zooming in
  sharpens the time detail. Levels use a fixed dBFS reference with the display floor set from
  the recording's noise floor, so calls stand out without adjustment.
- **Spectrogram controls**: brightness, contrast, colormap (including grayscale), FFT size,
  overlap, and window function, remembered between sessions. The cursor readout shows time,
  frequency, and level.
- **Navigation**: scroll zooms time at the cursor, Ctrl/Cmd+scroll zooms frequency, sideways
  scroll or right-drag pans; the waveform shares the spectrogram's time axis; time axes switch
  to `m:ss` / `h:mm:ss` on long recordings; zoom-to-selection.
- **Measurements in the selection table.** Choose from every measurement bioamla computes
  (duration, peak/center frequency, bandwidths, frequency percentiles, levels, power, entropy,
  peak-frequency contour); values update as boxes are edited. Export the table with its
  measurements as CSV.
- **Labels**: an active label applied to new boxes, number keys 1-9 to relabel, a colour per
  label on every view, and editable begin/end/low/high cells for exact placement.
- **Undo / redo** for every annotation edit, including imports and model labelling.
- **Playback**: play the selection, loop it, and play at 0.1x-2x speed (pitch follows speed).
  Space plays/pauses, Shift+Space plays the selection; "Follow" keeps the playhead in view.
- **Identify**: label the selected, the unlabelled, or all annotations with an AST classifier
  (Hugging Face id or local folder). The model is loaded once per run; labels and confidence
  are written as a single undo step. A confidence column shows detector and model certainty.
- **Detections on the spectrogram**: detector candidates draw as dashed boxes for review, and
  promoted candidates take the active label.
- A README with a screenshot, feature tour, and gesture/key reference.

### Changed

- Requires **bioamla >= 0.2.3**.
- The annotation screen's layout: the spectrogram is the main surface, the waveform a slim
  strip above it, the selection table runs along the bottom, and Detect is tabbed with
  Properties on the right. The Workspace panel moved to the left, beside the navigation bar.
- Toolbar commands are grouped into **Add** and **Table** menus so the toolbar fits on a
  laptop screen; the Home screen's tool cards reflow to the window width.

### Fixed

- Linking a file that is already in the workspace no longer adds a duplicate entry.
- The test suite no longer reads or overwrites the developer's real MagPy settings: all
  settings go through `magpy.settings.app_settings()`, which tests redirect to a temporary
  directory.
- Removed the `make cli` target, which ran a console script that does not exist.
