"""
Shared pyqtgraph plumbing for the audio views (spectrogram + waveform).

* :class:`AudioViewBox` -- the mouse model both views share, replacing
  pyqtgraph's default (left-drag pans, wheel zooms both axes) with one built for
  annotating: **left-drag draws a selection**, the wheel zooms *time* around the
  cursor (Ctrl/Cmd+wheel zooms the vertical axis, a horizontal scroll pans), and
  right- or middle-drag pans.
* :class:`TimeAxisItem` -- a bottom axis that switches from plain seconds to
  ``m:ss`` / ``h:mm:ss`` once the recording is long enough to need it.
* :func:`label_color` -- one colour per annotation label, so the same species
  is the same colour on every view and in the table.

No bioamla imports.
"""

from __future__ import annotations

import zlib

import pyqtgraph as pg
from PyQt6.QtCore import QRectF, Qt, pyqtSignal
from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import QGraphicsRectItem

# Minimum drag, in screen pixels, before a press-and-release counts as drawing a
# selection rather than a click (so a slightly shaky click doesn't make a sliver).
_MIN_DRAG_PX = 4

_DRAW_PEN = pg.mkPen(0, 200, 255, width=1.5)
_DRAW_BRUSH = pg.mkBrush(0, 200, 255, 40)

UNLABELED_COLOR = QColor(255, 235, 59)

# Distinct, dark-background-friendly hues. A label hashes onto one of these.
_LABEL_PALETTE = [
    QColor(c)
    for c in (
        "#4fc3f7",
        "#81c784",
        "#ff8a65",
        "#ba68c8",
        "#f06292",
        "#4db6ac",
        "#dce775",
        "#9575cd",
        "#ffb74d",
        "#7986cb",
        "#a1887f",
        "#e57373",
    )
]


_assigned: dict[str, int] = {}  # label -> palette slot, for this session


def label_color(label: str) -> QColor:
    """A colour for ``label`` (the unlabeled colour for an empty one).

    A label starts from a slot derived from its text, so it usually gets the
    same colour every session; if another label already holds that slot it takes
    the next free one, so labels only share a colour once there are more of them
    than colours.
    """
    if not label:
        return QColor(UNLABELED_COLOR)
    slot = _assigned.get(label)
    if slot is None:
        size = len(_LABEL_PALETTE)
        # crc32, not hash(): str hashes are salted per process, colours must not be.
        slot = zlib.crc32(label.encode("utf-8")) % size
        taken = set(_assigned.values())
        if len(taken) < size:
            while slot in taken:
                slot = (slot + 1) % size
        _assigned[label] = slot
    return QColor(_LABEL_PALETTE[slot])


def format_time(seconds: float, *, decimals: int = 0) -> str:
    """``12.3`` / ``1:05`` / ``1:02:03`` -- plain seconds below a minute."""
    sign = "-" if seconds < 0 else ""
    seconds = abs(seconds)
    if seconds < 60:
        return f"{sign}{seconds:.{decimals}f}"
    whole = int(seconds)
    frac = f"{seconds - whole:.{decimals}f}"[1:] if decimals else ""
    h, rem = divmod(whole, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{sign}{h:d}:{m:02d}:{s:02d}{frac}"
    return f"{sign}{m:d}:{s:02d}{frac}"


class TimeAxisItem(pg.AxisItem):
    """Bottom axis labelling ticks as seconds, ``m:ss`` or ``h:mm:ss``."""

    def __init__(self, *args: object, **kwargs: object) -> None:
        super().__init__(*args, **kwargs)
        self.enableAutoSIPrefix(False)

    def tickStrings(self, values: list[float], scale: float, spacing: float) -> list[str]:
        if spacing >= 1:
            decimals = 0
        elif spacing >= 0.1:
            decimals = 1
        elif spacing >= 0.01:
            decimals = 2
        else:
            decimals = 3
        return [format_time(v, decimals=decimals) for v in values]


class AudioViewBox(pg.ViewBox):
    """ViewBox with the annotate-first mouse model (see the module docstring).

    ``time_only=True`` (the waveform) draws full-height selections and reports
    no frequency bounds.
    """

    # (start, end, low, high); low/high are None for a time-only selection.
    selectionDrawn = pyqtSignal(float, float, object, object)

    def __init__(self, *, time_only: bool = False) -> None:
        super().__init__(enableMenu=False)
        self._time_only = time_only
        self._draw_enabled = True
        self._band: QGraphicsRectItem | None = None
        self.setMouseEnabled(x=True, y=not time_only)

    def set_draw_enabled(self, enabled: bool) -> None:
        """Whether left-drag draws a selection (otherwise it pans)."""
        self._draw_enabled = enabled

    # --- wheel: zoom time at the cursor; modifiers pick another axis -------
    def wheelEvent(self, ev, axis=None):  # noqa: N802 - Qt override
        horizontal = ev.orientation() == Qt.Orientation.Horizontal
        mods = ev.modifiers()
        if horizontal or mods & Qt.KeyboardModifier.ShiftModifier:
            (x0, x1), _ = self.viewRange()
            self.translateBy(x=-(x1 - x0) * ev.delta() / 1200.0)
            self.sigRangeChangedManually.emit(self.state["mouseEnabled"])
            ev.accept()
            return
        zoom_y = mods & (Qt.KeyboardModifier.ControlModifier | Qt.KeyboardModifier.MetaModifier)
        if axis is None:
            axis = 1 if (zoom_y and not self._time_only) else 0
        super().wheelEvent(ev, axis=axis)

    # --- drag: draw (left) or pan (right / middle) -------------------------
    def mouseDragEvent(self, ev, axis=None):  # noqa: N802 - Qt override
        left = ev.button() == Qt.MouseButton.LeftButton
        if axis is None and left and self._draw_enabled:
            ev.accept()
            self._drag_draw(ev)
            return
        if axis is None:
            ev.accept()
            self._drag_pan(ev)
            return
        super().mouseDragEvent(ev, axis=axis)

    def _drag_pan(self, ev) -> None:
        delta = self.mapToView(ev.lastPos()) - self.mapToView(ev.pos())
        self._resetTarget()
        self.translateBy(x=delta.x(), y=None if self._time_only else delta.y())
        self.sigRangeChangedManually.emit(self.state["mouseEnabled"])

    def _drag_draw(self, ev) -> None:
        rect = self._data_rect(ev)
        if ev.isFinish():
            if self._band is not None:
                self.removeItem(self._band)
                self._band = None
            moved = ev.screenPos() - ev.buttonDownScreenPos()
            if max(abs(moved.x()), abs(moved.y())) < _MIN_DRAG_PX or rect.width() <= 0:
                return
            if self._time_only:
                self.selectionDrawn.emit(rect.left(), rect.right(), None, None)
            else:
                self.selectionDrawn.emit(rect.left(), rect.right(), rect.top(), rect.bottom())
            return
        if self._band is None:
            self._band = QGraphicsRectItem()
            self._band.setPen(_DRAW_PEN)
            self._band.setBrush(_DRAW_BRUSH)
            self._band.setZValue(1e6)
            self.addItem(self._band, ignoreBounds=True)
        self._band.setRect(rect)

    def _data_rect(self, ev) -> QRectF:
        """The dragged rectangle in data coords, clamped to the view's limits."""
        a = self.mapToView(ev.buttonDownPos())
        b = self.mapToView(ev.pos())
        (vx0, vx1), (vy0, vy1) = self.viewRange()
        x_lim = self.state["limits"]["xLimits"]
        y_lim = self.state["limits"]["yLimits"]

        def clamp(v: float, lo: float | None, hi: float | None) -> float:
            # pyqtgraph stores "no limit" as None or +/-1e307 depending on version.
            if lo is not None and lo > -1e300:
                v = max(v, lo)
            if hi is not None and hi < 1e300:
                v = min(v, hi)
            return v

        xs = sorted((clamp(a.x(), *x_lim), clamp(b.x(), *x_lim)))
        if self._time_only:
            return QRectF(xs[0], vy0, xs[1] - xs[0], vy1 - vy0)
        ys = sorted((clamp(a.y(), *y_lim), clamp(b.y(), *y_lim)))
        return QRectF(xs[0], ys[0], xs[1] - xs[0], ys[1] - ys[0])
