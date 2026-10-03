"""
Shared parameter-form helpers for the config-driven screens (Batch, Training).

Both screens render a :class:`~magpy.services.BatchParam` (the generic form-field
descriptor: ``kind`` ∈ int/float/str/choice/bool, plus ``optional``) into a Qt
widget and read a typed value back. Keeping the widget<->value mapping in one
place means the two screens cannot drift in how they coerce ``optional`` fields or
choice values.
"""

from __future__ import annotations

from typing import Any

from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QLineEdit,
    QSpinBox,
    QWidget,
)

from magpy.services import BatchParam

# Sentinel: an optional field left blank/zero -> omit from the call (use the
# library default) rather than forwarding a 0/"".
OMIT = object()


def make_field(param: BatchParam) -> QWidget:
    """Build the input widget for ``param`` (seeded with its default)."""
    if param.kind == "bool":
        w: QWidget = QCheckBox()
        w.setChecked(bool(param.default))
    elif param.kind == "choice":
        w = QComboBox()
        w.addItems(list(param.choices))
        if param.default in param.choices:
            w.setCurrentText(str(param.default))
    elif param.kind == "int":
        w = QSpinBox()
        w.setRange(int(param.minimum), int(param.maximum))
        w.setSingleStep(int(param.step) or 1)
        w.setValue(int(param.default))
    elif param.kind == "float":
        w = QDoubleSpinBox()
        w.setDecimals(param.decimals)
        w.setRange(param.minimum, param.maximum)
        w.setSingleStep(param.step)
        w.setValue(float(param.default))
    else:  # str
        w = QLineEdit(str(param.default))
    if param.help:
        w.setToolTip(param.help)
    return w


def read_field(widget: QWidget, param: BatchParam) -> Any:
    """Read a typed value from ``widget``; return :data:`OMIT` to skip it."""
    if isinstance(widget, QCheckBox):
        return widget.isChecked()
    if isinstance(widget, QComboBox):
        return widget.currentText()
    if isinstance(widget, (QSpinBox, QDoubleSpinBox)):
        value = widget.value()
        if param.optional and value == 0:
            return OMIT
        return value
    # QLineEdit
    text = widget.text().strip()
    if param.optional and not text:
        return OMIT
    return text


def collect_params(fields: dict[str, QWidget], params: tuple[BatchParam, ...]) -> dict:
    """Read every field, dropping the ones that resolve to :data:`OMIT`."""
    out: dict = {}
    for param in params:
        value = read_field(fields[param.name], param)
        if value is not OMIT:
            out[param.name] = value
    return out
