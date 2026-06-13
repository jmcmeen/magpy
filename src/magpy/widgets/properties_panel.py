"""
PropertiesPanel -- shows metadata for the open audio and the selected annotation.

Rewritten against the new types (`LoadedAudio`, `Annotation`); the legacy panel
depended on the removed `core.*`. A dumb view: the shell pushes data in via
:meth:`set_audio` / :meth:`set_selection`. No bioamla imports.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from PyQt6.QtWidgets import QFormLayout, QGroupBox, QLabel, QVBoxLayout, QWidget

from magpy.services import Annotation, LoadedAudio


def _fmt_duration(seconds: float) -> str:
    m, s = divmod(seconds, 60)
    return f"{int(m)}:{s:05.2f}"


def _channels(samples: np.ndarray) -> int:
    return 1 if samples.ndim == 1 else int(min(samples.shape))


class PropertiesPanel(QWidget):
    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        self._file_rows = self._build_section(layout, "File", ["Name", "Duration", "Sample rate", "Channels"])
        self._sel_rows = self._build_section(
            layout, "Selection", ["Label", "Start", "End", "Duration", "Frequency", "Confidence"]
        )
        layout.addStretch()
        self.clear()

    def _build_section(self, parent: QVBoxLayout, title: str, fields: list[str]) -> dict[str, QLabel]:
        box = QGroupBox(title)
        form = QFormLayout(box)
        rows = {}
        for field in fields:
            value = QLabel("—")
            value.setStyleSheet("color: #d4d4d4;")
            form.addRow(QLabel(f"{field}:"), value)
            rows[field] = value
        parent.addWidget(box)
        return rows

    def set_audio(self, audio: Optional[LoadedAudio]) -> None:
        if audio is None:
            for v in self._file_rows.values():
                v.setText("—")
            return
        self._file_rows["Name"].setText(audio.path.name)
        self._file_rows["Duration"].setText(_fmt_duration(audio.duration))
        self._file_rows["Sample rate"].setText(f"{audio.sample_rate} Hz")
        self._file_rows["Channels"].setText(str(_channels(audio.samples)))

    def set_selection(self, ann: Optional[Annotation]) -> None:
        if ann is None:
            for v in self._sel_rows.values():
                v.setText("—")
            return
        self._sel_rows["Label"].setText(ann.label or "—")
        self._sel_rows["Start"].setText(f"{ann.start_time:.3f} s")
        self._sel_rows["End"].setText(f"{ann.end_time:.3f} s")
        self._sel_rows["Duration"].setText(f"{ann.duration:.3f} s")
        if ann.low_freq is not None and ann.high_freq is not None:
            self._sel_rows["Frequency"].setText(f"{ann.low_freq:.0f}–{ann.high_freq:.0f} Hz")
        else:
            self._sel_rows["Frequency"].setText("—")
        self._sel_rows["Confidence"].setText(
            "—" if ann.confidence is None else f"{ann.confidence:.2f}"
        )

    def clear(self) -> None:
        self.set_audio(None)
        self.set_selection(None)
