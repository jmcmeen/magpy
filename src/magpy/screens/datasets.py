"""
Datasets Screen -- build training data from annotated source audio.

This is the home of dataset construction (bioamla's ``dataset`` group). Tabs over
:mod:`magpy.services.datasets`:

* **Extract clips** -- ``extract_labeled_dataset``: cut annotated regions out of a
  file/folder into a labeled clip dataset (label subdirs + ``metadata.csv``).
* **Partition** -- ``partition_dataset``: split an existing dataset into
  train/val/test (stratified, grouped, reproducible).
* **Augment** -- ``batch_augment``: write augmented copies (noise / time-stretch /
  pitch / gain) of a folder.
* **Merge & inspect** -- ``merge_datasets``, plus ``get_dataset_stats`` /
  ``build_manifest`` / ``generate_license`` over a dataset folder.

Source audio is immutable: every op writes a *new* directory or sidecar. None of
the bioamla functions report progress, so runs use an indeterminate busy bar with
no Cancel button (it could not be honoured). Threaded via
:class:`~magpy.workers.Worker`; binds to the Workspace only for default paths. No
bioamla imports.
"""

from __future__ import annotations

from typing import Optional

from PyQt6.QtCore import QThreadPool
from PyQt6.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from magpy.models import Workspace
from magpy.services import (
    AUGMENT_PARAMS,
    augment,
    build_manifest,
    dataset_stats,
    extract_clips,
    generate_license,
    merge,
    partition,
)
from magpy.workers import Worker

from ._form import collect_params, make_field
from .base import BaseScreen


class DatasetsScreen(BaseScreen):
    def __init__(self, workspace: Workspace, parent: Optional[QWidget] = None) -> None:
        self._workspace = workspace
        self._worker: Optional[Worker] = None
        self._augment_fields: dict[str, QWidget] = {}
        self._run_buttons: list[QPushButton] = []
        super().__init__(parent)

    @property
    def screen_name(self) -> str:
        return "Datasets"

    @property
    def screen_icon(self) -> str:
        return "📁"

    def _default_dir(self, *parts: str) -> str:
        if self._workspace.is_open and self._workspace.dir is not None:
            return str(self._workspace.dir.joinpath(*parts))
        return ""

    # --- layout -----------------------------------------------------------
    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(10)

        header = QLabel("📁  Datasets")
        header.setStyleSheet("font-size: 22px; font-weight: bold; color: #d4d4d4;")
        layout.addWidget(header)
        sub = QLabel("Build training data from annotated audio — sources stay immutable; "
                     "every op writes a new dataset.")
        sub.setWordWrap(True)
        sub.setStyleSheet("color: #858585;")
        layout.addWidget(sub)

        self._tabs = QTabWidget()
        self._tabs.addTab(self._build_extract_tab(), "Extract clips")
        self._tabs.addTab(self._build_partition_tab(), "Partition")
        self._tabs.addTab(self._build_augment_tab(), "Augment")
        self._tabs.addTab(self._build_merge_inspect_tab(), "Merge & inspect")
        layout.addWidget(self._tabs, stretch=1)

        self._progress = QProgressBar()
        self._progress.setRange(0, 0)  # indeterminate: no dataset op reports progress
        self._progress.setVisible(False)
        layout.addWidget(self._progress)

        self._results = QTextEdit()
        self._results.setReadOnly(True)
        self._results.setPlaceholderText("Results appear here.")
        self._results.setMaximumHeight(160)
        layout.addWidget(self._results)

    def _path_row(self, edit: QLineEdit, *, folder: bool, file_filter: str = "") -> QHBoxLayout:
        row = QHBoxLayout()
        row.addWidget(edit, stretch=1)
        b = QPushButton("Browse…")
        b.clicked.connect(lambda: self._pick_into(edit, folder=folder, file_filter=file_filter))
        row.addWidget(b)
        return row

    def _run_button(self, text: str, slot) -> QPushButton:
        b = QPushButton(text)
        b.clicked.connect(slot)
        self._run_buttons.append(b)
        return b

    def _fraction(self, default: float) -> QDoubleSpinBox:
        w = QDoubleSpinBox()
        w.setRange(0.0, 1.0)
        w.setSingleStep(0.05)
        w.setDecimals(2)
        w.setValue(default)
        return w

    # --- Extract tab ------------------------------------------------------
    def _build_extract_tab(self) -> QWidget:
        tab = QWidget()
        form = QFormLayout(tab)
        self._ex_source = QLineEdit(self._default_dir())
        self._ex_source.setPlaceholderText("Audio file or folder of annotated recordings")
        form.addRow("Source", self._path_row(self._ex_source, folder=True))
        pick_file = QPushButton("Pick file instead…")
        pick_file.clicked.connect(
            lambda: self._pick_into(self._ex_source, folder=False,
                                    file_filter="Audio (*.wav *.flac *.ogg *.mp3);;All files (*)"))
        form.addRow("", pick_file)
        self._ex_anns = QLineEdit()
        self._ex_anns.setPlaceholderText("Optional selection table (.txt/.csv); else sidecars next to audio")
        form.addRow("Annotations", self._path_row(
            self._ex_anns, folder=False, file_filter="Selection tables (*.txt *.csv);;All files (*)"))
        self._ex_out = QLineEdit(self._default_dir("datasets", "clips"))
        form.addRow("Output dataset", self._path_row(self._ex_out, folder=True))
        self._ex_layout = QComboBox()
        self._ex_layout.addItems(["both", "audiofolder", "flat"])
        form.addRow("Layout", self._ex_layout)
        self._ex_sr = QLineEdit("")
        self._ex_sr.setPlaceholderText("e.g. 16000 (blank = keep source)")
        form.addRow("Resample (Hz)", self._ex_sr)
        self._ex_include = QLineEdit()
        self._ex_include.setPlaceholderText("comma-separated labels to keep (blank = all)")
        form.addRow("Include labels", self._ex_include)
        self._ex_exclude = QLineEdit()
        self._ex_exclude.setPlaceholderText("comma-separated labels to drop")
        form.addRow("Exclude labels", self._ex_exclude)
        form.addRow(self._run_button("Extract clips", self._extract))
        return tab

    def _extract(self) -> None:
        source = self._ex_source.text().strip()
        out = self._ex_out.text().strip()
        if not source or not out:
            self._results.setPlainText("Set a source and an output dataset folder.")
            return
        sr_text = self._ex_sr.text().strip()
        self._start(
            extract_clips, source, out,
            annotations=self._ex_anns.text().strip() or None,
            layout=self._ex_layout.currentText(),
            target_sample_rate=int(sr_text) if sr_text.isdigit() else None,
            include_labels=self._ex_include.text().strip() or None,
            exclude_labels=self._ex_exclude.text().strip() or None,
        )

    # --- Partition tab ----------------------------------------------------
    def _build_partition_tab(self) -> QWidget:
        tab = QWidget()
        form = QFormLayout(tab)
        self._pt_dir = QLineEdit(self._default_dir("datasets", "clips"))
        self._pt_dir.setPlaceholderText("Dataset folder with metadata.csv")
        form.addRow("Dataset", self._path_row(self._pt_dir, folder=True))
        splits = QHBoxLayout()
        self._pt_train = self._fraction(0.70)
        self._pt_val = self._fraction(0.15)
        self._pt_test = self._fraction(0.15)
        for lbl, w in (("train", self._pt_train), ("val", self._pt_val), ("test", self._pt_test)):
            splits.addWidget(QLabel(lbl))
            splits.addWidget(w)
        form.addRow("Splits", splits)
        self._pt_mode = QComboBox()
        self._pt_mode.addItems(["subdirs", "column"])
        form.addRow("Mode", self._pt_mode)
        self._pt_group = QLineEdit("source_file")
        self._pt_group.setToolTip(
            "Keep rows sharing this column in one split (prevents clip leakage). Blank to disable.")
        form.addRow("Group by", self._pt_group)
        form.addRow(self._run_button("Partition", self._partition))
        return tab

    def _partition(self) -> None:
        ds = self._pt_dir.text().strip()
        if not ds:
            self._results.setPlainText("Set a dataset folder to partition.")
            return
        total = self._pt_train.value() + self._pt_val.value() + self._pt_test.value()
        if abs(total - 1.0) > 0.001:  # bioamla requires the three fractions to sum to 1.0
            self._results.setPlainText(f"Splits must sum to 1.0 (got {total:.2f}).")
            return
        self._start(
            partition, ds,
            train=self._pt_train.value(), val=self._pt_val.value(), test=self._pt_test.value(),
            mode=self._pt_mode.currentText(),
            group_by=self._pt_group.text().strip() or None,
        )

    # --- Augment tab ------------------------------------------------------
    def _build_augment_tab(self) -> QWidget:
        tab = QWidget()
        form = QFormLayout(tab)
        self._ag_in = QLineEdit(self._default_dir())
        self._ag_in.setPlaceholderText("Input folder of audio")
        form.addRow("Input", self._path_row(self._ag_in, folder=True))
        self._ag_out = QLineEdit(self._default_dir("datasets", "augmented"))
        form.addRow("Output", self._path_row(self._ag_out, folder=True))
        for param in AUGMENT_PARAMS:
            widget = make_field(param)
            self._augment_fields[param.name] = widget
            form.addRow(param.label, widget)
        form.addRow(self._run_button("Augment", self._augment))
        return tab

    def _augment(self) -> None:
        in_dir = self._ag_in.text().strip()
        out = self._ag_out.text().strip()
        if not in_dir or not out:
            self._results.setPlainText("Set an input folder and an output folder.")
            return
        params = collect_params(self._augment_fields, AUGMENT_PARAMS)
        self._start(augment, in_dir, out, params)

    # --- Merge & inspect tab ---------------------------------------------
    def _build_merge_inspect_tab(self) -> QWidget:
        tab = QWidget()
        outer = QVBoxLayout(tab)

        outer.addWidget(QLabel("Merge datasets"))
        self._mg_inputs = QPlainTextEdit()
        self._mg_inputs.setPlaceholderText("One dataset folder per line (≥2).")
        self._mg_inputs.setMaximumHeight(80)
        outer.addWidget(self._mg_inputs)
        add = QPushButton("Add folder…")
        add.clicked.connect(self._add_merge_folder)
        outer.addWidget(add)
        self._mg_out = QLineEdit(self._default_dir("datasets", "merged"))
        outer.addLayout(self._path_row(self._mg_out, folder=True))
        outer.addWidget(self._run_button("Merge", self._merge))

        outer.addSpacing(12)
        outer.addWidget(QLabel("Inspect / finalize a dataset"))
        self._in_dir = QLineEdit(self._default_dir("datasets", "clips"))
        self._in_dir.setPlaceholderText("Dataset folder with metadata.csv")
        outer.addLayout(self._path_row(self._in_dir, folder=True))
        row = QHBoxLayout()
        row.addWidget(self._run_button("Stats", self._stats))
        row.addWidget(self._run_button("Build manifest", self._manifest))
        row.addWidget(self._run_button("Generate license", self._license))
        row.addStretch(1)
        outer.addLayout(row)
        outer.addStretch(1)
        return tab

    def _add_merge_folder(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Add dataset folder", self._mg_out.text())
        if path:
            existing = self._mg_inputs.toPlainText()
            self._mg_inputs.setPlainText((existing + "\n" + path).strip() if existing else path)

    def _merge(self) -> None:
        paths = [p.strip() for p in self._mg_inputs.toPlainText().splitlines() if p.strip()]
        out = self._mg_out.text().strip()
        if len(paths) < 2 or not out:
            self._results.setPlainText("Add at least two dataset folders and an output folder.")
            return
        self._start(merge, paths, out)

    def _stats(self) -> None:
        ds = self._in_dir.text().strip()
        if ds:
            self._start(dataset_stats, ds)

    def _manifest(self) -> None:
        ds = self._in_dir.text().strip()
        if ds:
            self._start(build_manifest, ds)

    def _license(self) -> None:
        ds = self._in_dir.text().strip()
        if ds:
            self._start(generate_license, ds)

    # --- pickers / run plumbing ------------------------------------------
    def _pick_into(self, edit: QLineEdit, *, folder: bool, file_filter: str = "") -> None:
        if folder:
            path = QFileDialog.getExistingDirectory(self, "Select folder", edit.text())
        else:
            path, _ = QFileDialog.getOpenFileName(self, "Select file", edit.text(), file_filter)
        if path:
            edit.setText(path)

    def _start(self, fn, *args, **kwargs) -> None:
        if self._worker is not None:
            return
        for b in self._run_buttons:
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
        for b in self._run_buttons:
            b.setEnabled(True)

    def _on_result(self, outcome) -> None:
        lines = [f"[{outcome.op}] {outcome.message}"]
        if outcome.output_dir:
            lines.append(f"Output: {outcome.output_dir}")
        for k, v in (outcome.details or {}).items():
            if not isinstance(v, (dict, list)):
                lines.append(f"  {k}: {v}")
        self._results.setPlainText("\n".join(lines))
