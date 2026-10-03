"""
IdentifyDialog -- choose the model and scope for model-assisted labelling.

A plain form: which classifier (a local model folder or a Hugging Face id),
which annotations to run it on, and how confident it must be before its label is
written. It only collects the choices; the screen runs the model. No bioamla
imports.
"""

from __future__ import annotations

from PyQt6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

SCOPE_SELECTED = "selected"
SCOPE_UNLABELED = "unlabeled"
SCOPE_ALL = "all"


class IdentifyDialog(QDialog):
    def __init__(
        self,
        model: str,
        min_confidence: float,
        *,
        has_selection: bool,
        counts: dict[str, int],
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Identify with a model")
        self.setMinimumWidth(520)
        layout = QVBoxLayout(self)
        intro = QLabel(
            "Runs an audio classifier on the sound inside each chosen annotation and "
            "writes its best guess as the label, with the confidence alongside. The "
            "result is one undo step, so nothing is lost by trying it."
        )
        intro.setWordWrap(True)
        layout.addWidget(intro)

        form = QFormLayout()
        model_row = QHBoxLayout()
        self._model = QLineEdit(model)
        self._model.setPlaceholderText("Hugging Face model id, or a local model folder")
        browse = QPushButton("Browse…")
        browse.clicked.connect(self._browse)
        model_row.addWidget(self._model, stretch=1)
        model_row.addWidget(browse)
        form.addRow("Model", model_row)

        self._scope = QComboBox()
        for key, text in (
            (SCOPE_SELECTED, "The selected annotation"),
            (SCOPE_UNLABELED, "Annotations without a label"),
            (SCOPE_ALL, "All annotations (replaces labels)"),
        ):
            self._scope.addItem(f"{text}  ({counts.get(key, 0)})", key)
        self._scope.setCurrentIndex(0 if has_selection else 1)
        form.addRow("Apply to", self._scope)

        self._min_confidence = QDoubleSpinBox()
        self._min_confidence.setRange(0.0, 1.0)
        self._min_confidence.setSingleStep(0.05)
        self._min_confidence.setDecimals(2)
        self._min_confidence.setValue(min_confidence)
        self._min_confidence.setToolTip("Guesses below this confidence are not written")
        form.addRow("Min confidence", self._min_confidence)
        layout.addLayout(form)

        note = QLabel(
            "A Hugging Face model is downloaded the first time it is used, which can "
            "take a while; after that it loads from the local cache."
        )
        note.setWordWrap(True)
        note.setStyleSheet("color: #858585;")
        layout.addWidget(note)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.button(QDialogButtonBox.StandardButton.Ok).setText("Identify")
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _browse(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Choose a model folder", "")
        if path:
            self._model.setText(path)

    def model(self) -> str:
        return self._model.text().strip()

    def scope(self) -> str:
        return str(self._scope.currentData())

    def min_confidence(self) -> float:
        return float(self._min_confidence.value())
