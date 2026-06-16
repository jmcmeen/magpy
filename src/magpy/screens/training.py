"""
Training Screen -- fine-tune, evaluate, and run an AST classifier.

Three tabs over :mod:`magpy.services.training` (bioamla's ``models ast`` group):

* **Train** -- ``train_ast`` over a dataset (HuggingFace id / metadata CSV /
  class-subdir folder). It is long-running, heavyweight, and **has no progress or
  cancel hook**, so the UI is deliberately honest: a confirm-before-launch warning
  (it runs in-process; closing MagPy or an OOM loses the run), an indeterminate
  busy bar, a disabled Train button while running, and **no Cancel button** (one
  could not be honoured). TensorBoard under ``<training dir>/logs`` is the real
  progress view.
* **Evaluate** -- ``evaluate_directory``: metrics vs. a ground-truth CSV.
* **Predict** -- ``predict_file``: top-k classification of a single file.

All three run off-thread through a :class:`~magpy.workers.Worker`. The screen
binds to the Workspace only for sensible default paths. No bioamla imports.
"""

from __future__ import annotations

from typing import Optional

from PyQt6.QtCore import QThreadPool
from PyQt6.QtWidgets import (
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from magpy.models import Workspace
from magpy.services import (
    TRAIN_PARAMS,
    evaluate_model,
    predict_audio,
    run_training,
)
from magpy.workers import Worker

from ._form import collect_params, make_field
from .base import BaseScreen


class TrainingScreen(BaseScreen):
    def __init__(self, workspace: Workspace, parent: Optional[QWidget] = None) -> None:
        self._workspace = workspace
        self._worker: Optional[Worker] = None
        self._train_fields: dict[str, QWidget] = {}
        super().__init__(parent)

    @property
    def screen_name(self) -> str:
        return "Training"

    @property
    def screen_icon(self) -> str:
        return "🧠"

    def _default_dir(self, *parts: str) -> str:
        if self._workspace.is_open and self._workspace.dir is not None:
            return str(self._workspace.dir.joinpath(*parts))
        return ""

    # --- layout -----------------------------------------------------------
    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(10)

        header = QLabel("🧠  Model Training")
        header.setStyleSheet("font-size: 22px; font-weight: bold; color: #d4d4d4;")
        layout.addWidget(header)
        sub = QLabel("Fine-tune, evaluate, and run an AST audio classifier (bioamla models ast).")
        sub.setWordWrap(True)
        sub.setStyleSheet("color: #858585;")
        layout.addWidget(sub)

        self._tabs = QTabWidget()
        self._tabs.addTab(self._build_train_tab(), "Train")
        self._tabs.addTab(self._build_evaluate_tab(), "Evaluate")
        self._tabs.addTab(self._build_predict_tab(), "Predict")
        layout.addWidget(self._tabs, stretch=1)

        # Shared run state -- always indeterminate (no progress hook anywhere here).
        self._progress = QProgressBar()
        self._progress.setRange(0, 0)
        self._progress.setVisible(False)
        layout.addWidget(self._progress)

        self._results = QTextEdit()
        self._results.setReadOnly(True)
        self._results.setPlaceholderText("Results appear here.")
        self._results.setMaximumHeight(160)
        layout.addWidget(self._results)

    def _path_row(self, edit: QLineEdit, slot, button: str = "Browse…") -> QHBoxLayout:
        row = QHBoxLayout()
        row.addWidget(edit, stretch=1)
        b = QPushButton(button)
        b.clicked.connect(slot)
        row.addWidget(b)
        return row

    # --- Train tab --------------------------------------------------------
    def _build_train_tab(self) -> QWidget:
        tab = QWidget()
        form = QFormLayout(tab)

        self._train_dataset = QLineEdit(self._default_dir("datasets"))
        self._train_dataset.setPlaceholderText("HF dataset id, metadata.csv, or class-subdir folder")
        ds_row = QHBoxLayout()
        ds_row.addWidget(self._train_dataset, stretch=1)
        pick_dir = QPushButton("Folder…")
        pick_dir.clicked.connect(lambda: self._pick_into(self._train_dataset, folder=True))
        pick_csv = QPushButton("CSV…")
        pick_csv.clicked.connect(
            lambda: self._pick_into(self._train_dataset, folder=False, file_filter="CSV (*.csv)"))
        ds_row.addWidget(pick_dir)
        ds_row.addWidget(pick_csv)
        form.addRow("Train dataset", ds_row)

        self._training_dir = QLineEdit(self._default_dir("models"))
        self._training_dir.setPlaceholderText("Output dir for checkpoints / best_model / logs")
        form.addRow("Training dir", self._path_row(
            self._training_dir, lambda: self._pick_into(self._training_dir, folder=True)))

        for param in TRAIN_PARAMS:
            widget = make_field(param)
            self._train_fields[param.name] = widget
            form.addRow(param.label, widget)

        self._train_button = QPushButton("Train")
        self._train_button.clicked.connect(self._train)
        form.addRow(self._train_button)

        note = QLabel(
            "Training runs in-process and cannot be cancelled or show step progress. "
            "Watch TensorBoard in <training dir>/logs. Closing MagPy aborts it."
        )
        note.setWordWrap(True)
        note.setStyleSheet("color: #c4a000;")
        form.addRow(note)
        return tab

    # --- Evaluate tab -----------------------------------------------------
    def _build_evaluate_tab(self) -> QWidget:
        tab = QWidget()
        form = QFormLayout(tab)
        self._eval_model = QLineEdit("bioamla/scp-frogs")
        form.addRow("Model (HF id or path)", self._path_row(
            self._eval_model, lambda: self._pick_into(self._eval_model, folder=True)))
        self._eval_audio = QLineEdit(self._default_dir())
        form.addRow("Audio folder", self._path_row(
            self._eval_audio, lambda: self._pick_into(self._eval_audio, folder=True)))
        self._eval_truth = QLineEdit()
        self._eval_truth.setPlaceholderText("Ground-truth CSV (file + label columns)")
        form.addRow("Ground truth CSV", self._path_row(
            self._eval_truth,
            lambda: self._pick_into(self._eval_truth, folder=False, file_filter="CSV (*.csv)")))
        self._eval_file_col = QLineEdit("file_name")
        form.addRow("File column", self._eval_file_col)
        self._eval_label_col = QLineEdit("label")
        form.addRow("Label column", self._eval_label_col)
        self._eval_button = QPushButton("Evaluate")
        self._eval_button.clicked.connect(self._evaluate)
        form.addRow(self._eval_button)
        return tab

    # --- Predict tab ------------------------------------------------------
    def _build_predict_tab(self) -> QWidget:
        tab = QWidget()
        form = QFormLayout(tab)
        self._pred_model = QLineEdit("bioamla/scp-frogs")
        form.addRow("Model (HF id or path)", self._path_row(
            self._pred_model, lambda: self._pick_into(self._pred_model, folder=True)))
        self._pred_file = QLineEdit()
        self._pred_file.setPlaceholderText("Audio file to classify")
        form.addRow("Audio file", self._path_row(
            self._pred_file, lambda: self._pick_into(
                self._pred_file, folder=False,
                file_filter="Audio (*.wav *.flac *.ogg *.mp3 *.m4a);;All files (*)")))
        self._pred_button = QPushButton("Predict")
        self._pred_button.clicked.connect(self._predict)
        form.addRow(self._pred_button)
        return tab

    # --- pickers ----------------------------------------------------------
    def _pick_into(self, edit: QLineEdit, *, folder: bool, file_filter: str = "") -> None:
        if folder:
            path = QFileDialog.getExistingDirectory(self, "Select folder", edit.text())
        else:
            path, _ = QFileDialog.getOpenFileName(self, "Select file", edit.text(), file_filter)
        if path:
            edit.setText(path)

    # --- run helpers ------------------------------------------------------
    def _buttons(self) -> list[QPushButton]:
        return [self._train_button, self._eval_button, self._pred_button]

    def _start(self, fn, *args, **kwargs) -> None:
        if self._worker is not None:
            return
        for b in self._buttons():
            b.setEnabled(False)
        self._progress.setVisible(True)
        worker = Worker(fn, *args, **kwargs)
        self._worker = worker
        worker.signals.result.connect(self._on_result)
        worker.signals.error.connect(lambda exc: self._results.setPlainText(f"Failed:\n{exc}"))
        worker.signals.finished.connect(self._on_finished)
        QThreadPool.globalInstance().start(worker)

    def _on_finished(self) -> None:
        self._worker = None
        self._progress.setVisible(False)
        for b in self._buttons():
            b.setEnabled(True)

    def _on_result(self, outcome) -> None:
        self._results.setPlainText(self._format(outcome))

    @staticmethod
    def _format(outcome) -> str:
        # TrainOutcome / EvalOutcome / PredictOutcome -> human text (duck-typed).
        if hasattr(outcome, "model_path"):
            return (f"{outcome.message}\nModel: {outcome.model_path}\n"
                    f"Epochs: {outcome.epochs}   accuracy: {outcome.final_accuracy}   "
                    f"loss: {outcome.final_loss}")
        if hasattr(outcome, "f1_score"):
            return (f"{outcome.message}\nSamples: {outcome.total_samples}\n"
                    f"Accuracy: {outcome.accuracy:.3f}   Precision: {outcome.precision:.3f}   "
                    f"Recall: {outcome.recall:.3f}   F1: {outcome.f1_score:.3f}")
        if hasattr(outcome, "predicted_label"):
            lines = [f"Predicted: {outcome.predicted_label}  ({outcome.confidence:.3f})", "", "Top-k:"]
            lines += [f"  {label}: {score:.3f}" for label, score in outcome.top_k]
            return "\n".join(lines)
        return str(outcome)

    # --- actions ----------------------------------------------------------
    def _train(self) -> None:
        dataset = self._train_dataset.text().strip()
        training_dir = self._training_dir.text().strip()
        if not dataset or not training_dir:
            self._results.setPlainText("Set both a train dataset and a training dir.")
            return
        confirm = QMessageBox.question(
            self, "Start training?",
            "Training runs in-process and may take a long time. It cannot be "
            "cancelled once started, and closing MagPy will abort it. Continue?",
        )
        if confirm != QMessageBox.StandardButton.Yes:
            return
        params = collect_params(self._train_fields, TRAIN_PARAMS)
        self._results.setPlainText("Training… (watch TensorBoard for progress)")
        self._start(run_training, dataset, training_dir, params)

    def _evaluate(self) -> None:
        model = self._eval_model.text().strip()
        audio = self._eval_audio.text().strip()
        truth = self._eval_truth.text().strip()
        if not (model and audio and truth):
            self._results.setPlainText("Set a model, an audio folder, and a ground-truth CSV.")
            return
        self._results.setPlainText("Evaluating…")
        self._start(
            evaluate_model, audio, model, truth,
            file_column=self._eval_file_col.text().strip() or "file_name",
            label_column=self._eval_label_col.text().strip() or "label",
        )

    def _predict(self) -> None:
        model = self._pred_model.text().strip()
        path = self._pred_file.text().strip()
        if not (model and path):
            self._results.setPlainText("Set a model and an audio file.")
            return
        self._results.setPlainText("Predicting…")
        self._start(predict_audio, path, model)
