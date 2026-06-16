"""
Batch Screen -- run a bioamla ``batch`` op over a folder, with progress + cancel.

The fan-out of the single-file operations: pick an op (grouped Audio / Detect /
Analyze / Models), point it at an input folder (+ an output folder where needed),
tune its parameters via a form built generically from the op's
:class:`~magpy.services.BatchParam` specs, and Run. Work goes off-thread through a
:class:`~magpy.workers.Worker`; this is the screen that finally exercises the
Worker's **progress + cooperative-cancel** path for the ops bioamla instruments.

Honesty about capabilities is baked into the spec:

* ops with ``supports_progress`` get a determinate bar driven by the progress
  signal and a working **Cancel** (cancellation lands at the next file boundary);
  ops without it get an indeterminate busy bar and no Cancel button (it could not
  be honoured).
* ``max_workers`` only appears as a field for ops whose bioamla function accepts
  it -- so the screen can never forward an unsupported kwarg.

Pure Qt + the services seam + the workers bridge; binds to the Workspace model
for a default input folder and to link audio outputs back in. No bioamla imports.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from PyQt6.QtCore import QThreadPool
from PyQt6.QtWidgets import (
    QComboBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QProgressBar,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from magpy.models import Workspace
from magpy.services import (
    BATCH_OPS,
    KIND_FOLDER,
    BatchOpSpec,
    run_batch_op,
)
from magpy.workers import Worker

from ._form import collect_params, make_field
from .base import BaseScreen


class BatchScreen(BaseScreen):
    def __init__(self, workspace: Workspace, parent: Optional[QWidget] = None) -> None:
        self._workspace = workspace
        self._worker: Optional[Worker] = None
        self._field_widgets: dict[str, object] = {}
        self._last_output: Optional[Path] = None
        super().__init__(parent)

    @property
    def screen_name(self) -> str:
        return "Batch"

    @property
    def screen_icon(self) -> str:
        return "📚"

    # --- layout -----------------------------------------------------------
    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(10)

        header = QLabel("📚  Batch Processing")
        header.setStyleSheet("font-size: 22px; font-weight: bold; color: #d4d4d4;")
        layout.addWidget(header)
        sub = QLabel("Run a single-file operation across an entire folder.")
        sub.setStyleSheet("color: #858585;")
        layout.addWidget(sub)

        # Operation picker (grouped). A separator goes *between* groups, never at
        # index 0 -- otherwise the combo would open on a non-selectable separator
        # whose currentData() is None (blanks the form, crashes Run via _current_spec).
        self._op_combo = QComboBox()
        current_group = None
        for op in BATCH_OPS:
            if op.group != current_group:
                if self._op_combo.count() > 0:
                    self._op_combo.insertSeparator(self._op_combo.count())
                current_group = op.group
            self._op_combo.addItem(f"{op.group} · {op.label}", op.key)
        self._op_combo.currentIndexChanged.connect(self._on_op_changed)
        layout.addWidget(self._op_combo)

        self._description = QLabel()
        self._description.setWordWrap(True)
        self._description.setStyleSheet("color: #858585;")
        layout.addWidget(self._description)

        # Input / output folders.
        self._input_edit = QLineEdit()
        self._input_edit.setPlaceholderText("Input folder of audio files")
        if self._workspace.is_open and self._workspace.dir is not None:
            self._input_edit.setText(str(self._workspace.dir))
        layout.addLayout(self._folder_row("Input folder", self._input_edit, self._pick_input))

        self._output_edit = QLineEdit()
        self._output_edit.setPlaceholderText("Output folder")
        self._output_row = self._folder_row("Output folder", self._output_edit, self._pick_output)
        layout.addLayout(self._output_row)

        # Parameter form (rebuilt per op).
        self._form_host = QWidget()
        self._form = QFormLayout(self._form_host)
        self._form.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._form_host)

        # Run / Cancel.
        run_row = QHBoxLayout()
        self._run_button = QPushButton("Run")
        self._run_button.clicked.connect(self._run)
        self._cancel_button = QPushButton("Cancel")
        self._cancel_button.clicked.connect(self._cancel)
        self._cancel_button.setEnabled(False)
        run_row.addWidget(self._run_button)
        run_row.addWidget(self._cancel_button)
        self._link_button = QPushButton("Link output → workspace")
        self._link_button.clicked.connect(self._link_output)
        self._link_button.setEnabled(False)
        run_row.addWidget(self._link_button)
        run_row.addStretch(1)
        layout.addLayout(run_row)

        self._progress = QProgressBar()
        self._progress.setVisible(False)
        layout.addWidget(self._progress)

        self._results = QTextEdit()
        self._results.setReadOnly(True)
        self._results.setPlaceholderText("Results appear here.")
        layout.addWidget(self._results, stretch=1)

        self._on_op_changed(0)

    def _folder_row(self, label: str, edit: QLineEdit, slot) -> QHBoxLayout:
        row = QHBoxLayout()
        tag = QLabel(label)
        tag.setMinimumWidth(96)
        row.addWidget(tag)
        row.addWidget(edit, stretch=1)
        browse = QPushButton("Browse…")
        browse.clicked.connect(slot)
        row.addWidget(browse)
        return row

    # --- op selection / form ---------------------------------------------
    def _current_spec(self) -> BatchOpSpec:
        key = self._op_combo.currentData()
        return next(op for op in BATCH_OPS if op.key == key)

    def _on_op_changed(self, _index: int) -> None:
        if self._op_combo.currentData() is None:  # a separator row
            return
        spec = self._current_spec()
        self._description.setText(spec.description)
        # Show/hide the output row.
        for i in range(self._output_row.count()):
            w = self._output_row.itemAt(i).widget()
            if w is not None:
                w.setVisible(spec.needs_output)
        # Rebuild the parameter form.
        while self._form.rowCount():
            self._form.removeRow(0)
        self._field_widgets.clear()
        for param in spec.params:
            widget = make_field(param)
            self._field_widgets[param.name] = widget
            self._form.addRow(param.label, widget)
        # Cancel is only meaningful when the op reports progress.
        self._cancel_button.setVisible(spec.supports_progress)

    def _collect_params(self) -> dict:
        spec = self._current_spec()
        return collect_params(self._field_widgets, spec.params)

    # --- folder pickers ---------------------------------------------------
    def _pick_input(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Input folder", self._input_edit.text())
        if path:
            self._input_edit.setText(path)

    def _pick_output(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Output folder", self._output_edit.text())
        if path:
            self._output_edit.setText(path)

    # --- run / cancel -----------------------------------------------------
    def _run(self) -> None:
        if self._worker is not None:
            return
        if self._op_combo.currentData() is None:  # a separator row is selected
            self._results.setPlainText("Pick an operation first.")
            return
        spec = self._current_spec()
        input_dir = self._input_edit.text().strip()
        if not input_dir or not Path(input_dir).is_dir():
            self._results.setPlainText("Pick a valid input folder first.")
            return
        output_dir = self._output_edit.text().strip()
        if spec.needs_output and not output_dir:
            self._results.setPlainText("This operation needs an output folder.")
            return
        self._last_output = Path(output_dir) if output_dir else None
        self._link_button.setEnabled(False)

        params = self._collect_params()
        self._set_running(True, spec)
        worker = Worker(
            run_batch_op, spec.key, input_dir, output_dir, params,
            with_progress=spec.supports_progress,
        )
        self._worker = worker
        if spec.supports_progress:
            worker.signals.progress.connect(self._on_progress)
        worker.signals.result.connect(self._on_result)
        worker.signals.error.connect(
            lambda exc: self._results.setPlainText(f"Failed:\n{exc}")
        )
        worker.signals.finished.connect(self._on_finished)
        QThreadPool.globalInstance().start(worker)

    def _cancel(self) -> None:
        if self._worker is not None:
            self._worker.cancel()
            self._results.append("\nCancelling at next file…")

    def _on_progress(self, done: int, total: int) -> None:
        self._progress.setMaximum(total or 0)
        self._progress.setValue(done)

    def _on_result(self, outcome) -> None:
        lines = [
            f"Operation: {outcome.op}",
            f"Files: {outcome.total}   ok: {outcome.successful}   failed: {outcome.failed}",
        ]
        if outcome.output_dir:
            lines.append(f"Output: {outcome.output_dir}")
        if outcome.output_files:
            lines.append(f"Wrote {len(outcome.output_files)} file(s).")
        if outcome.errors:
            lines.append("\nErrors:")
            lines.extend(f"  • {e}" for e in outcome.errors[:20])
        self._results.setPlainText("\n".join(lines))
        # Offer to link the output folder if it exists.
        if self._last_output is not None and self._last_output.is_dir():
            self._link_button.setEnabled(True)

    def _on_finished(self) -> None:
        self._worker = None
        self._set_running(False, self._current_spec())

    def _set_running(self, running: bool, spec: BatchOpSpec) -> None:
        self._run_button.setEnabled(not running)
        self._op_combo.setEnabled(not running)
        self._cancel_button.setEnabled(running and spec.supports_progress)
        self._progress.setVisible(running)
        if running:
            # Determinate when the op reports progress; busy spinner otherwise.
            self._progress.setRange(0, 100 if spec.supports_progress else 0)
            self._progress.setValue(0)

    # --- workspace --------------------------------------------------------
    def _link_output(self) -> None:
        if self._last_output is None or not self._last_output.is_dir():
            return
        if not self._workspace.is_open:
            self._results.append("\nNo workspace open to link into.")
            return
        self._workspace.link(self._last_output, KIND_FOLDER)
        self._link_button.setEnabled(False)
        self._results.append(f"\nLinked {self._last_output} into the workspace.")
