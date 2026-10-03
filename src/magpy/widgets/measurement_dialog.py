"""
MeasurementDialog -- pick which measurement columns the selection table shows.

Built generically from the MagPy-owned :data:`~magpy.services.MEASUREMENTS`
descriptions, so it lists whatever the analysis engine can compute without this
widget knowing any of them. No bioamla imports.
"""

from __future__ import annotations

from PyQt6.QtWidgets import (
    QCheckBox,
    QDialog,
    QDialogButtonBox,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from magpy.services import DEFAULT_MEASUREMENTS, MEASUREMENTS

_COLUMNS = 3


class MeasurementDialog(QDialog):
    def __init__(self, selected: list[str] | tuple[str, ...], parent: QWidget | None = None):
        super().__init__(parent)
        self.setWindowTitle("Measurements")
        layout = QVBoxLayout(self)
        intro = QLabel(
            "Each ticked measurement becomes a column in the selection table, computed for "
            "every annotation from the audio inside its box."
        )
        intro.setWordWrap(True)
        layout.addWidget(intro)

        grid = QGridLayout()
        self._boxes: dict[str, QCheckBox] = {}
        per_column = -(-len(MEASUREMENTS) // _COLUMNS)
        for index, spec in enumerate(MEASUREMENTS):
            box = QCheckBox(spec.header)
            box.setToolTip(spec.description)
            box.setChecked(spec.key in selected)
            grid.addWidget(box, index % per_column, index // per_column)
            self._boxes[spec.key] = box
        layout.addLayout(grid)

        presets = QHBoxLayout()
        for text, keys in (
            ("Defaults", DEFAULT_MEASUREMENTS),
            ("All", tuple(spec.key for spec in MEASUREMENTS)),
            ("None", ()),
        ):
            button = QPushButton(text)
            button.clicked.connect(lambda _checked, k=keys: self._set_checked(k))
            presets.addWidget(button)
        presets.addStretch(1)
        layout.addLayout(presets)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _set_checked(self, keys: tuple[str, ...]) -> None:
        for key, box in self._boxes.items():
            box.setChecked(key in keys)

    def selected_keys(self) -> list[str]:
        """The ticked measurement keys, in table order."""
        return [key for key, box in self._boxes.items() if box.isChecked()]
