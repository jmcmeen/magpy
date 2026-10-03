"""
SpectrogramView -- the main annotation surface: a pyqtgraph view that renders
:class:`SpectrogramImage` tiles, overlays the document's annotations (and the
detector's candidates), and turns mouse gestures into user intent.

**Rendering is viewport-driven.** The view never holds a whole-file spectrogram.
Whenever the visible time range changes it emits :attr:`renderRequested` with the
window it needs (the visible span plus a margin either side, so small pans don't
re-render) and a column budget matched to its pixel width; the screen computes
that window and hands it back through :meth:`set_image`. Recording length
therefore doesn't matter, and zooming in sharpens the time resolution.

**Gestures** (see :class:`~magpy.widgets._plot.AudioViewBox` for the mouse model):

* drag -- draw a time-frequency box (:attr:`selectionDrawn`)
* click a box -- select it (the selected one becomes editable: drag to move,
  handles to resize); click empty space -- deselect and move the playhead
  (:attr:`seekRequested`)
* double-click a box -- play it (:attr:`playRangeRequested`)
* wheel -- zoom time at the cursor; Ctrl/Cmd+wheel -- zoom frequency;
  horizontal scroll or right-drag -- pan

Annotation overlays bind directly to the :class:`AnnotationSet` model (like the
table), so the view stays in sync without shell glue. Annotations that carry
frequency bounds render as boxes; those without (time-only imports, promoted
detector candidates) render as full-height regions. Both are coloured by label.

A keyboard-driven transient selection is also available (:meth:`start_selection`
/ :meth:`start_box_selection` + :meth:`selection_bounds`) for placing a selection
without the mouse.

No bioamla imports -- only the MagPy models and plain service DTOs.
"""

from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from PyQt6.QtCore import QRectF, Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFormLayout,
    QGraphicsRectItem,
    QHBoxLayout,
    QLabel,
    QMenu,
    QSlider,
    QToolButton,
    QVBoxLayout,
    QWidget,
    QWidgetAction,
)

from magpy.models import AnnotationSet, CandidateSet
from magpy.services import (
    FFT_SIZES,
    OVERLAPS,
    WINDOWS,
    Annotation,
    SpectrogramImage,
    SpectrogramParams,
)

from ._plot import AudioViewBox, TimeAxisItem, format_time, label_color

LEFT_AXIS_WIDTH = 64  # shared with WaveformView so the two time axes line up

_SELECTED_PEN = pg.mkPen(255, 255, 255, width=2)
_SEL_BRUSH = pg.mkBrush(0, 200, 255, 50)
_SEL_PEN = pg.mkPen(0, 200, 255, width=2)
_CAND_PEN = pg.mkPen(0, 229, 255, width=1, style=Qt.PenStyle.DashLine)
_CAND_SELECTED_PEN = pg.mkPen(0, 229, 255, width=2)
_CAND_BRUSH = pg.mkBrush(0, 229, 255, 18)

_COLORMAPS = ("magma", "inferno", "viridis", "plasma", "cividis", "gray", "gray_r")

_RENDER_DEBOUNCE_MS = 30
_MAX_COLS = 4000
# Re-render once the on-screen image is this much coarser than the pixels can show.
_STALE_RATIO = 1.5

_HINT = "drag: draw box  ·  click: select / seek  ·  double-click: play  ·  scroll: zoom"


def _get_colormap(name: str) -> pg.ColorMap:
    try:
        return pg.colormap.get(name, source="matplotlib")
    except Exception:
        try:
            return pg.colormap.get(name)
        except Exception:
            return pg.colormap.get("CET-L9")  # bundled fallback


def _with_alpha(color: QColor, alpha: int) -> QColor:
    c = QColor(color)
    c.setAlpha(alpha)
    return c


def _make_box_roi(x: float, y: float, w: float, h: float, pen: object) -> pg.ROI:
    """A movable box with scale handles on every corner and edge."""
    roi = pg.ROI([x, y], [w, h], pen=pen, hoverPen=pen, rotatable=False, removable=False)
    for hx, hy in ((0, 0), (1, 0), (0, 1), (1, 1), (0.5, 0), (0.5, 1), (0, 0.5), (1, 0.5)):
        roi.addScaleHandle([hx, hy], [1 - hx, 1 - hy])
    roi.setZValue(20)
    return roi


class SpectrogramView(QWidget):
    """Renders a dB spectrogram with time (s) on x and frequency (Hz) on y."""

    renderRequested = pyqtSignal(float, float, int)  # (t0, t1, max columns)
    paramsChanged = pyqtSignal(object)  # SpectrogramParams
    viewSettingsChanged = pyqtSignal()  # any control-row choice (for persistence)
    selectionDrawn = pyqtSignal(float, float, object, object)  # start, end, low, high
    seekRequested = pyqtSignal(float)  # seconds
    playRangeRequested = pyqtSignal(float, float)  # (start, end) seconds

    def __init__(self, annotations: AnnotationSet, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._model = annotations
        self._candidates: CandidateSet | None = None
        self._f_max = 0.0
        self._duration = 0.0
        self._img: SpectrogramImage | None = None
        self._noise_db: float | None = None  # estimated once per recording
        self._follow = True
        self._regions: dict[str, pg.LinearRegionItem] = {}  # time-only annotations
        self._boxes: dict[str, QGraphicsRectItem] = {}  # unselected freq-bounded annotations
        self._labels: dict[str, pg.TextItem] = {}
        self._cand_items: list[QGraphicsRectItem] = []
        self._selection: object | None = None  # LinearRegionItem | ROI
        self._edit_roi: pg.ROI | None = None  # the selected box, made editable
        self._edit_id: str | None = None
        self._syncing = False  # guards the rebuild path (build -> signals -> handlers)
        self._editing = False  # guards the edit path (drag -> model.update -> rebuild)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)
        layout.addLayout(self._build_controls())

        self._vb = AudioViewBox()
        self._plot = pg.PlotWidget(viewBox=self._vb, axisItems={"bottom": TimeAxisItem("bottom")})
        self._plot.setLabel("left", "Frequency", units="Hz")
        self._plot.getAxis("left").setWidth(LEFT_AXIS_WIDTH)
        self._plot.hideButtons()
        self._image = pg.ImageItem()
        self._image.setColorMap(_get_colormap(_COLORMAPS[0]))
        self._plot.addItem(self._image)
        self._playhead = pg.InfiniteLine(angle=90, movable=False, pen=pg.mkPen("w", width=1))
        self._playhead.setZValue(50)
        self._playhead.setVisible(False)
        self._plot.addItem(self._playhead)
        layout.addWidget(self._plot, stretch=1)

        readout = QHBoxLayout()
        readout.setContentsMargins(4, 0, 4, 0)
        self._cursor_label = QLabel("")
        self._cursor_label.setStyleSheet("color: #b0b0b0;")
        hint = QLabel(_HINT)
        hint.setStyleSheet("color: #6a6a6a;")
        readout.addWidget(self._cursor_label)
        readout.addStretch(1)
        readout.addWidget(hint)
        layout.addLayout(readout)

        self._render_timer = QTimer(self)
        self._render_timer.setSingleShot(True)
        self._render_timer.setInterval(_RENDER_DEBOUNCE_MS)
        self._render_timer.timeout.connect(self._request_render)
        self._vb.sigXRangeChanged.connect(lambda *_: self._render_timer.start())
        self._vb.sigResized.connect(lambda *_: self._render_timer.start())
        self._vb.selectionDrawn.connect(self.selectionDrawn)

        self._plot.scene().sigMouseMoved.connect(self._on_mouse_moved)
        self._plot.scene().sigMouseClicked.connect(self._on_mouse_clicked)

        self._model.added.connect(self._sync_annotations)
        self._model.removed.connect(self._sync_annotations)
        self._model.reset.connect(self._sync_annotations)
        self._model.changed.connect(self._sync_annotations)
        # Selecting swaps the chosen box to an editable ROI -> full rebuild.
        self._model.selectionChanged.connect(self._sync_annotations)

    @property
    def plot_item(self) -> pg.PlotItem:
        """The underlying plot, so a sibling view can link its time axis to it."""
        return self._plot.getPlotItem()

    def set_draw_enabled(self, enabled: bool) -> None:
        """Whether dragging draws a selection (off: dragging pans instead)."""
        self._vb.set_draw_enabled(enabled)

    # --- controls ---------------------------------------------------------
    def _build_controls(self) -> QHBoxLayout:
        row = QHBoxLayout()
        row.setContentsMargins(4, 2, 4, 2)
        row.setSpacing(6)

        def slider(tooltip: str) -> QSlider:
            s = QSlider(Qt.Orientation.Horizontal)
            s.setRange(0, 100)
            s.setValue(50)
            s.setFixedWidth(72)
            s.setToolTip(tooltip)
            s.valueChanged.connect(self._apply_levels)
            s.valueChanged.connect(lambda _v: self.viewSettingsChanged.emit())
            return s

        self._brightness = slider("Brightness: lower the display floor to reveal fainter sound")
        self._contrast = slider("Contrast: narrow the dynamic range mapped onto the colours")
        row.addWidget(QLabel("Brightness"))
        row.addWidget(self._brightness)
        row.addWidget(QLabel("Contrast"))
        row.addWidget(self._contrast)

        self._cmap_combo = QComboBox()
        self._cmap_combo.addItems(list(_COLORMAPS))
        self._cmap_combo.setToolTip("Colormap")
        self._cmap_combo.setMinimumWidth(96)
        self._cmap_combo.currentTextChanged.connect(
            lambda name: self._image.setColorMap(_get_colormap(name))
        )
        self._cmap_combo.currentTextChanged.connect(lambda _n: self.viewSettingsChanged.emit())
        row.addWidget(self._cmap_combo)

        row.addSpacing(10)
        defaults = SpectrogramParams()
        self._fft_combo = QComboBox()
        for n in FFT_SIZES:
            self._fft_combo.addItem(str(n), n)
        self._fft_combo.setCurrentIndex(FFT_SIZES.index(defaults.n_fft))
        self._fft_combo.setToolTip(
            "FFT window size (samples). Larger: finer frequency detail, blurrier timing."
        )
        self._overlap_combo = QComboBox()
        for o in OVERLAPS:
            self._overlap_combo.addItem(f"{o * 100:g}%", o)
        self._overlap_combo.setCurrentIndex(OVERLAPS.index(defaults.overlap))
        self._overlap_combo.setToolTip("Overlap between consecutive windows")
        self._window_combo = QComboBox()
        self._window_combo.addItems(list(WINDOWS))
        self._window_combo.setToolTip("Window function")
        # The STFT settings are chosen once and left, so they live in a popup
        # behind one button that reads back the current choice.
        panel = QWidget()
        form = QFormLayout(panel)
        form.setContentsMargins(12, 10, 12, 10)
        for label, combo in (
            ("FFT size", self._fft_combo),
            ("Overlap", self._overlap_combo),
            ("Window", self._window_combo),
        ):
            combo.setMinimumWidth(130)
            form.addRow(label, combo)
            combo.currentIndexChanged.connect(self._on_params_changed)
        self._stft_button = QToolButton()
        self._stft_button.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        self._stft_button.setStyleSheet("QToolButton::menu-indicator { image: none; }")
        self._stft_button.setToolTip("Spectrogram settings: FFT size, overlap, window")
        menu = QMenu(self._stft_button)
        holder = QWidgetAction(menu)
        holder.setDefaultWidget(panel)
        menu.addAction(holder)
        self._stft_button.setMenu(menu)
        self._update_stft_button()
        row.addWidget(self._stft_button)

        row.addStretch(1)
        self._follow_box = QCheckBox("Follow")
        self._follow_box.setChecked(True)
        self._follow_box.setToolTip("Keep the playhead in view during playback")
        self._follow_box.toggled.connect(self._set_follow)
        row.addWidget(self._follow_box)

        for text, tip, slot in (
            ("−", "Zoom out", self.zoom_out),
            ("+", "Zoom in", self.zoom_in),
            ("Fit", "Show the whole recording", self.zoom_fit),
            ("Sel", "Zoom to the selected annotation (Z)", self.zoom_to_selection),
        ):
            b = QToolButton()
            b.setText(text)
            b.setToolTip(tip)
            b.setAutoRaise(False)
            b.clicked.connect(slot)
            row.addWidget(b)
        return row

    def params(self) -> SpectrogramParams:
        """The STFT settings currently chosen in the controls row."""
        return SpectrogramParams(
            n_fft=int(self._fft_combo.currentData()),
            overlap=float(self._overlap_combo.currentData()),
            window=self._window_combo.currentText(),
        )

    def _update_stft_button(self) -> None:
        p = self.params()
        self._stft_button.setText(f"FFT {p.n_fft} · {p.overlap * 100:g}% · {p.window} ▾")

    def _on_params_changed(self, *_: object) -> None:
        self._update_stft_button()
        self.paramsChanged.emit(self.params())
        self.viewSettingsChanged.emit()
        self.invalidate()

    def view_settings(self) -> dict[str, object]:
        """The control-row choices, as plain values a settings store can hold."""
        params = self.params()
        return {
            "colormap": self._cmap_combo.currentText(),
            "brightness": self._brightness.value(),
            "contrast": self._contrast.value(),
            "n_fft": params.n_fft,
            "overlap": params.overlap,
            "window": params.window,
            "follow": self._follow_box.isChecked(),
        }

    def apply_view_settings(self, settings: dict[str, object]) -> None:
        """Restore choices saved by :meth:`view_settings` (unknown values are skipped)."""

        def choose(combo: QComboBox, index: int) -> None:
            if index >= 0:
                combo.setCurrentIndex(index)

        try:
            if "colormap" in settings:
                choose(self._cmap_combo, self._cmap_combo.findText(str(settings["colormap"])))
            if "brightness" in settings:
                self._brightness.setValue(int(settings["brightness"]))
            if "contrast" in settings:
                self._contrast.setValue(int(settings["contrast"]))
            if "n_fft" in settings:
                choose(self._fft_combo, self._fft_combo.findData(int(settings["n_fft"])))
            if "overlap" in settings:
                wanted = float(settings["overlap"])
                choose(
                    self._overlap_combo,
                    next((i for i, o in enumerate(OVERLAPS) if abs(o - wanted) < 1e-6), -1),
                )
            if "window" in settings:
                choose(self._window_combo, self._window_combo.findText(str(settings["window"])))
            if "follow" in settings:
                self._follow_box.setChecked(str(settings["follow"]).lower() in ("true", "1"))
        except (TypeError, ValueError):
            pass  # a corrupt stored value must not stop the app opening

    def _set_follow(self, enabled: bool) -> None:
        self._follow = enabled
        self.viewSettingsChanged.emit()

    def _apply_levels(self, *_: object) -> None:
        if self._noise_db is None:
            return
        # Brightness moves the floor relative to the recording's noise floor
        # (brighter = lower floor = fainter sound shows); contrast sets how many
        # dB above the floor span the colormap (more contrast = fewer dB).
        floor = self._noise_db + (50 - self._brightness.value()) * 0.6
        span = 100.0 - self._contrast.value() * 0.9
        self._image.setLevels((floor, floor + span))

    # --- zoom -------------------------------------------------------------
    def zoom_in(self) -> None:
        self._zoom(0.5)

    def zoom_out(self) -> None:
        self._zoom(2.0)

    def _zoom(self, factor: float) -> None:
        (x0, x1), _ = self._plot.viewRange()
        center = (x0 + x1) / 2
        half = (x1 - x0) * factor / 2
        lo = max(0.0, center - half)
        hi = min(self._duration, center + half) if self._duration else center + half
        self._plot.setXRange(lo, hi, padding=0)

    def zoom_fit(self) -> None:
        if self._duration:
            self._plot.setXRange(0.0, self._duration, padding=0)
            self._plot.setYRange(0.0, self._f_max, padding=0)

    def zoom_to_selection(self) -> None:
        """Frame the selected annotation with a little context around it."""
        ann = self._model.selected
        if ann is None or not self._duration:
            return
        pad = max(ann.duration * 0.75, 0.05)
        self._plot.setXRange(
            max(0.0, ann.start_time - pad), min(self._duration, ann.end_time + pad), padding=0
        )
        if ann.low_freq is not None and ann.high_freq is not None:
            fpad = max((ann.high_freq - ann.low_freq) * 0.75, 100.0)
            self._plot.setYRange(
                max(0.0, ann.low_freq - fpad), min(self._f_max, ann.high_freq + fpad), padding=0
            )

    def visible_range(self) -> tuple[float, float]:
        """The visible time span in seconds."""
        (x0, x1), _ = self._plot.viewRange()
        return (max(0.0, x0), min(self._duration, x1) if self._duration else x1)

    # --- spectrogram image ------------------------------------------------
    def set_extent(self, duration: float, f_max: float) -> None:
        """Start showing a new recording of ``duration`` s up to ``f_max`` Hz."""
        self._apply_extent(duration, f_max)
        self._request_render()

    def _apply_extent(self, duration: float, f_max: float) -> None:
        self._duration = duration
        self._f_max = f_max
        self._img = None
        self._noise_db = None
        self._image.clear()
        self.clear_selection()
        self._vb.setLimits(
            xMin=0.0,
            xMax=duration,
            yMin=0.0,
            yMax=f_max,
            minXRange=min(0.01, duration),
            minYRange=min(50.0, f_max),
        )
        self._plot.setXRange(0.0, duration, padding=0)
        self._plot.setYRange(0.0, f_max, padding=0)
        self._playhead.setPos(0.0)
        self._playhead.setVisible(True)

    def set_image(self, img: SpectrogramImage | None) -> None:
        """Show a rendered window, or clear the view when ``None``."""
        if img is None:
            self._img = None
            self._duration = self._f_max = 0.0
            self._noise_db = None
            self._image.clear()
            self._playhead.setVisible(False)
            self.clear_selection()
            return
        if not self._duration:  # no recording announced: the image defines the extent
            self._apply_extent(img.t1, img.f_max)
        self._img = img
        # ImageItem default axis order is col-major (image[x, y]); db is
        # (freq, time), so transpose to put time on x and frequency on y.
        self._image.setImage(img.db.T, autoLevels=False)
        self._image.setRect(QRectF(img.t0, 0.0, img.duration, img.f_max))
        if self._noise_db is None:
            self._noise_db = self._estimate_noise_floor(img.db)
        self._apply_levels()

    @staticmethod
    def _estimate_noise_floor(db: np.ndarray) -> float:
        """The typical background level: the median, ignoring digital silence."""
        if db.size == 0:
            return -100.0
        audible = db[db > float(db.max()) - 120.0]
        return float(np.median(audible if audible.size else db))

    def invalidate(self) -> None:
        """Drop the current image and re-render (the STFT settings changed)."""
        self._img = None
        self._request_render()

    def _request_render(self) -> None:
        if not self._duration:
            return
        (x0, x1), _ = self._plot.viewRange()
        x0, x1 = max(0.0, x0), min(self._duration, x1)
        if x1 <= x0:
            return
        width_px = max(200, int(self._vb.width()))
        img = self._img
        if img is not None:
            wanted = (x1 - x0) / width_px  # seconds per pixel on screen
            have = img.duration / max(1, img.db.shape[1])
            # Within a column of each edge counts: the last frame rarely lands
            # exactly on the end of the recording.
            covered = img.t0 <= x0 + have and img.t1 >= x1 - have
            sharp = img.full_resolution or have <= wanted * _STALE_RATIO
            if covered and sharp:
                return
        # A margin either side means small pans are served from the same image.
        margin = (x1 - x0) / 2
        t0, t1 = max(0.0, x0 - margin), min(self._duration, x1 + margin)
        cols = round(width_px * (t1 - t0) / (x1 - x0))
        self.renderRequested.emit(t0, t1, min(_MAX_COLS, cols))

    def set_playhead(self, seconds: float) -> None:
        """Move the playback line; page the view to keep it visible if following."""
        self._playhead.setPos(seconds)
        if not self._follow or not self._duration:
            return
        (x0, x1), _ = self._plot.viewRange()
        width = x1 - x0
        if width <= 0 or width >= self._duration or x0 <= seconds <= x1:
            return
        new0 = min(max(0.0, seconds - width * 0.1), self._duration - width)
        self._plot.setXRange(new0, new0 + width, padding=0)

    # --- transient selection (keyboard-driven) ----------------------------
    def start_selection(self) -> None:
        """Create a draggable full-band time selection centred in the view."""
        self.clear_selection()
        x0, x1 = self._plot.viewRange()[0]
        span = (x1 - x0) * 0.1
        mid = (x0 + x1) / 2
        self._selection = pg.LinearRegionItem(values=(mid - span, mid + span), brush=_SEL_BRUSH)
        self._plot.addItem(self._selection)

    def start_box_selection(self) -> None:
        """Create a draggable/resizable 2-D time-frequency selection box."""
        self.clear_selection()
        (x0, x1), (y0, y1) = self._plot.viewRange()
        x_span = (x1 - x0) * 0.2
        y0c, y1c = y0 + (y1 - y0) * 0.3, y0 + (y1 - y0) * 0.7
        mid = (x0 + x1) / 2
        roi = _make_box_roi(mid - x_span / 2, y0c, x_span, y1c - y0c, _SEL_PEN)
        self._selection = roi
        self._plot.addItem(roi)

    def selection_bounds(self) -> tuple[float, float, float | None, float | None] | None:
        """Current selection as ``(start, end, low_freq, high_freq)`` or ``None``.

        ``low_freq``/``high_freq`` are ``None`` for a time region. Bounds are
        normalised (start ≤ end, low ≤ high) so a box dragged up/left can't make
        an inverted annotation, and frequency is clamped to ≥ 0.
        """
        if self._selection is None:
            return None
        if isinstance(self._selection, pg.LinearRegionItem):
            lo, hi = sorted(self._selection.getRegion())
            return (float(lo), float(hi), None, None)
        pos, size = self._selection.pos(), self._selection.size()
        start, end = sorted((pos.x(), pos.x() + size.x()))
        low, high = sorted((pos.y(), pos.y() + size.y()))
        return (float(start), float(end), max(0.0, float(low)), max(0.0, float(high)))

    def clear_selection(self) -> None:
        if self._selection is not None:
            self._plot.removeItem(self._selection)
            self._selection = None

    # --- cursor readout ---------------------------------------------------
    def _on_mouse_moved(self, scene_pos: object) -> None:
        if self._img is None or not self._vb.sceneBoundingRect().contains(scene_pos):
            self._cursor_label.setText("")
            return
        pt = self._vb.mapSceneToView(scene_pos)
        text = f"{format_time(pt.x(), decimals=3)} s    {pt.y() / 1000:.2f} kHz"
        level = self._level_at(pt.x(), pt.y())
        if level is not None:
            text += f"    {level:.0f} dB"
        self._cursor_label.setText(text)

    def _level_at(self, t: float, f: float) -> float | None:
        img = self._img
        if img is None or not (img.t0 <= t <= img.t1) or not (0.0 <= f <= img.f_max):
            return None
        col = min(img.db.shape[1] - 1, int((t - img.t0) / img.duration * img.db.shape[1]))
        row = min(img.db.shape[0] - 1, int(f / img.f_max * (img.db.shape[0] - 1) + 0.5))
        return float(img.db[row, col])

    # --- candidate overlays (the detector's reviewable layer) -------------
    def bind_candidates(self, candidates: CandidateSet) -> None:
        """Draw ``candidates`` as dashed boxes under the curated annotations."""
        self._candidates = candidates
        candidates.reset.connect(self._sync_candidates)
        candidates.thresholdChanged.connect(self._sync_candidates)
        candidates.selectionChanged.connect(self._sync_candidates)
        self._sync_candidates()

    def _sync_candidates(self, *_: object) -> None:
        for item in self._cand_items:
            self._plot.removeItem(item)
        self._cand_items.clear()
        if self._candidates is None:
            return
        selected = self._candidates.selected
        for cand in self._candidates.visible_items():
            if cand.low_freq is not None and cand.high_freq is not None:
                low, high = sorted((cand.low_freq, cand.high_freq))
            else:
                low, high = 0.0, self._f_max
            box = QGraphicsRectItem(QRectF(cand.start_time, low, cand.duration, high - low))
            box.setPen(_CAND_SELECTED_PEN if cand is selected else _CAND_PEN)
            box.setBrush(_CAND_BRUSH)
            box.setZValue(5)
            self._plot.addItem(box)
            self._cand_items.append(box)

    # --- annotation overlays ---------------------------------------------
    def _sync_annotations(self, *_: object) -> None:
        # Skip rebuilds we provoke ourselves (an in-progress drag edit), so we
        # don't tear down the ROI the user is holding.
        if self._editing:
            return
        self._syncing = True
        try:
            self._rebuild_overlays()
        finally:
            self._syncing = False

    def _rebuild_overlays(self) -> None:
        for item in (*self._regions.values(), *self._boxes.values(), *self._labels.values()):
            self._plot.removeItem(item)
        self._regions.clear()
        self._boxes.clear()
        self._labels.clear()
        if self._edit_roi is not None:
            self._plot.removeItem(self._edit_roi)
            self._edit_roi = None
            self._edit_id = None

        selected = self._model.selected
        sel_id = selected.id if selected is not None else None
        for ann in self._model.items():
            color = label_color(ann.label)
            is_selected = ann.id == sel_id
            has_freq = ann.low_freq is not None and ann.high_freq is not None
            if has_freq:
                low, high = sorted((ann.low_freq, ann.high_freq))
                if is_selected:
                    # The selected freq-box is editable: a draggable/resizable ROI.
                    roi = _make_box_roi(
                        ann.start_time, low, ann.duration, high - low, _SELECTED_PEN
                    )
                    roi.sigRegionChangeFinished.connect(self._on_box_edited)
                    self._plot.addItem(roi)
                    self._edit_roi = roi
                    self._edit_id = ann.id
                else:
                    box = QGraphicsRectItem(QRectF(ann.start_time, low, ann.duration, high - low))
                    box.setPen(pg.mkPen(color, width=1.5))
                    box.setBrush(pg.mkBrush(_with_alpha(color, 36)))
                    box.setZValue(10)
                    self._plot.addItem(box)  # routed to the ViewBox -> data coords
                    self._boxes[ann.id] = box
                label_y = high
            else:
                region = pg.LinearRegionItem(
                    values=(ann.start_time, ann.end_time),
                    movable=is_selected,
                    brush=pg.mkBrush(_with_alpha(color, 70 if is_selected else 36)),
                    pen=_SELECTED_PEN if is_selected else pg.mkPen(color, width=1),
                )
                region.setZValue(10)
                if is_selected:
                    region.sigRegionChangeFinished.connect(
                        lambda r, a=ann: self._on_region_edited(a, r)
                    )
                self._plot.addItem(region)
                self._regions[ann.id] = region
                label_y = self._f_max
            if ann.label:
                text = pg.TextItem(ann.label, anchor=(0, 1), color=color)
                text.setPos(ann.start_time, label_y)
                text.setZValue(15)
                self._plot.addItem(text)
                self._labels[ann.id] = text

    def _write_back(self, ann: Annotation) -> None:
        """Publish an in-view edit without rebuilding the item being dragged."""
        self._editing = True
        try:
            self._model.update(ann)
        finally:
            self._editing = False

    def _on_box_edited(self) -> None:
        """A selected box ROI was dragged/resized -> write the new bounds back."""
        if self._syncing or self._edit_roi is None or self._edit_id is None:
            return
        ann = next((a for a in self._model.items() if a.id == self._edit_id), None)
        if ann is None:
            return
        pos, size = self._edit_roi.pos(), self._edit_roi.size()
        start, end = sorted((pos.x(), pos.x() + size.x()))
        low, high = sorted((pos.y(), pos.y() + size.y()))
        ann.start_time, ann.end_time = float(start), float(end)
        ann.low_freq, ann.high_freq = max(0.0, float(low)), max(0.0, float(high))
        self._write_back(ann)
        # Keep the label glued to the moved box (overlays weren't rebuilt).
        text = self._labels.get(ann.id)
        if text is not None:
            text.setPos(ann.start_time, ann.high_freq)

    def _on_region_edited(self, ann: Annotation, region: pg.LinearRegionItem) -> None:
        """A selected time-only region was dragged -> write the new span back."""
        if self._syncing:
            return
        start, end = sorted(region.getRegion())
        ann.start_time, ann.end_time = float(start), float(end)
        self._write_back(ann)
        text = self._labels.get(ann.id)
        if text is not None:
            text.setPos(ann.start_time, self._f_max)

    def _hit_test(self, t: float, f: float) -> Annotation | None:
        """The smallest annotation under ``(t, f)``, so nested boxes stay reachable."""
        hit = None
        best_area = None
        for ann in self._model.items():
            if not (ann.start_time <= t <= ann.end_time):
                continue
            if ann.low_freq is not None and ann.high_freq is not None:
                low, high = sorted((ann.low_freq, ann.high_freq))
                if not (low <= f <= high):
                    continue
                area = ann.duration * (high - low)
            else:
                area = ann.duration * (self._f_max or 1.0)  # region: full band
            if best_area is None or area < best_area:
                best_area, hit = area, ann
        return hit

    def _on_mouse_clicked(self, event: object) -> None:
        """Click selects the box under the cursor; empty space deselects + seeks."""
        if not self._duration or self._syncing:
            return
        if event.button() != Qt.MouseButton.LeftButton:
            return
        if not self._vb.sceneBoundingRect().contains(event.scenePos()):
            return  # a click on an axis, not the plot
        pt = self._vb.mapSceneToView(event.scenePos())
        hit = self._hit_test(pt.x(), pt.y())
        if event.double():
            if hit is not None:
                self.playRangeRequested.emit(hit.start_time, hit.end_time)
            return
        self._model.select(hit)
        if hit is None:
            self.seekRequested.emit(min(max(0.0, float(pt.x())), self._duration))
