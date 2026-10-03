"""
AudioAnnotationScreen -- annotate a sound file.

The audio-annotation workspace: the shared spectrogram/waveform/transport view
(from :class:`BaseAudioScreen`) plus the annotation-specific panels -- the
selection table across the bottom (annotations with their measurement columns)
and a Detect panel (the reviewable candidate layer) tabbed beside Properties.

The loop this screen is built around: **drag a box on the spectrogram** and it
becomes an annotation carrying the active label; the table fills in its
measurements. A detector's output lands in a :class:`CandidateSet`, drawn as
dashed boxes on the spectrogram, that the user reviews and promotes. Annotations
persist into the open workspace bundle automatically on any change, and every
edit is undoable.

Acoustic-indices analysis lives on its own :class:`IndicesScreen`; this screen is
purely about creating and curating annotations.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from PyQt6.QtCore import Qt, QThreadPool, QTimer
from PyQt6.QtGui import QAction
from PyQt6.QtWidgets import (
    QComboBox,
    QDialog,
    QDockWidget,
    QFileDialog,
    QLabel,
    QMessageBox,
    QToolBar,
)

from magpy.models import CandidateSet
from magpy.services import (
    DEFAULT_IDENTIFY_MODEL,
    DEFAULT_MEASUREMENTS,
    Annotation,
    candidate_to_annotation,
    detector_label,
    export_measurements_csv,
    identify_annotations,
    load_annotations,
    measure_annotations,
    run_detection,
    save_annotations,
)
from magpy.settings import app_settings
from magpy.widgets import (
    SCOPE_ALL,
    SCOPE_SELECTED,
    SCOPE_UNLABELED,
    AnnotationTable,
    DetectPanel,
    IdentifyDialog,
    MeasurementDialog,
)
from magpy.workers import Worker

from ._audio_base import _DOCK_FEATURES, BaseAudioScreen

_ANNOTATION_FILTER = "Selection tables (*.txt *.csv);;Raven table (*.txt);;CSV (*.csv)"
_MEASURE_DEBOUNCE_MS = 200
_MAX_LABEL_KEYS = 9  # number keys 1-9 apply the first nine labels
_SETTINGS_MEASUREMENTS = "annotation/measurements"
_SETTINGS_IDENTIFY_MODEL = "annotation/identify_model"
_SETTINGS_IDENTIFY_CONFIDENCE = "annotation/identify_min_confidence"


def _geometry(ann: Annotation) -> tuple:
    """What an annotation's measurements depend on (so a relabel doesn't recompute)."""
    return (ann.start_time, ann.end_time, ann.low_freq, ann.high_freq)


class AudioAnnotationScreen(BaseAudioScreen):
    """Audio view + selection table + detect (reviewable candidate layer)."""

    # --- screen-specific UI -----------------------------------------------
    def _create_docks(self) -> None:
        self._detect_worker: Worker | None = None
        self._identify_worker: Worker | None = None
        self._candidates = CandidateSet(self)
        self._settings = app_settings()
        stored = self._settings.value(_SETTINGS_MEASUREMENTS, None)
        self._measure_keys: list[str] = (
            list(DEFAULT_MEASUREMENTS) if stored is None else [str(k) for k in stored]
        )
        # Measurement cache: annotation id -> (geometry it was computed for, values).
        self._measured: dict[str, tuple[tuple, dict[str, float]]] = {}
        self._measure_worker: Worker | None = None
        self._measure_timer = QTimer(self)
        self._measure_timer.setSingleShot(True)
        self._measure_timer.setInterval(_MEASURE_DEBOUNCE_MS)
        self._measure_timer.timeout.connect(self._measure_pending)

        # The selection table runs the full width under the spectrogram.
        self._table = AnnotationTable(self._annotations)
        self._table.set_measurement_keys(self._measure_keys)
        ann_dock = QDockWidget("Selection table", self)
        ann_dock.setObjectName("AnnotationsDock")
        ann_dock.setWidget(self._table)
        ann_dock.setFeatures(_DOCK_FEATURES)
        self.addDockWidget(Qt.DockWidgetArea.BottomDockWidgetArea, ann_dock)

        # Detect is a tall form: it tabs with Properties on the right rather than
        # taking height away from the spectrogram.
        self._detect_panel = DetectPanel(self._candidates)
        detect_dock = QDockWidget("Detect", self)
        detect_dock.setObjectName("DetectDock")
        detect_dock.setWidget(self._detect_panel)
        detect_dock.setFeatures(_DOCK_FEATURES)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, detect_dock)
        self.tabifyDockWidget(self._properties_dock, detect_dock)
        self._properties_dock.raise_()

        self.setCorner(Qt.Corner.BottomRightCorner, Qt.DockWidgetArea.RightDockWidgetArea)
        self._table_dock = ann_dock
        self._docks_sized = False

        self._spectrogram.bind_candidates(self._candidates)

    def showEvent(self, event) -> None:  # noqa: N802 - Qt override
        super().showEvent(event)
        if not self._docks_sized:
            # Dock sizes only stick once the window has a real geometry.
            self._docks_sized = True
            self.resizeDocks([self._table_dock], [230], Qt.Orientation.Vertical)
            self.resizeDocks([self._properties_dock], [300], Qt.Orientation.Horizontal)

    def _populate_toolbar(self, toolbar: QToolBar, action) -> None:
        toolbar.addSeparator()
        toolbar.addWidget(QLabel(" Label "))
        self._label_combo = QComboBox()
        self._label_combo.setEditable(True)
        self._label_combo.setInsertPolicy(QComboBox.InsertPolicy.NoInsert)
        self._label_combo.setFixedWidth(170)
        self._label_combo.lineEdit().setPlaceholderText("label for new boxes")
        self._label_combo.setToolTip(
            "The label new boxes get. Press Enter to apply it to the selected box.\n"
            "Keys 1-9 apply the first nine labels in this list."
        )
        self._label_combo.lineEdit().returnPressed.connect(self._apply_active_label)
        toolbar.addWidget(self._label_combo)

        toolbar.addSeparator()
        self._undo_action = action("Undo", self._annotations.undo, "Ctrl+Z")
        self._redo_action = action("Redo", self._annotations.redo, "Ctrl+Shift+Z")
        action("Delete", self._delete_selected, "Delete")
        toolbar.addSeparator()
        self._add_menu_button(
            toolbar,
            "Table",
            (
                ("Import selection table…", self._import_annotations, None),
                ("Export selection table…", self._export_annotations, "Ctrl+E"),
                ("Choose measurements…", self._choose_measurements, None),
            ),
        )
        self._identify_action = action("Identify…", self._identify_dialog, "I")
        self._identify_action.setToolTip("Label annotations with an audio classifier (I)")

        # Keyboard-only actions (no toolbar button).
        def key(shortcut: str, slot) -> None:
            act = QAction(self)
            act.setShortcut(shortcut)
            act.triggered.connect(slot)
            self.addAction(act)

        key("Backspace", self._delete_selected)
        key("S", self._spectrogram.start_selection)
        key("B", self._spectrogram.start_box_selection)
        key("Return", self._commit_keyboard_selection)
        key("Escape", self._spectrogram.clear_selection)
        key("Z", self._spectrogram.zoom_to_selection)
        key("L", self._focus_label)
        key("N", lambda: self._step_selection(+1))
        key("P", lambda: self._step_selection(-1))
        for number in range(1, _MAX_LABEL_KEYS + 1):
            key(str(number), lambda n=number: self._apply_numbered_label(n))

    def _wire_extra(self) -> None:
        # Persist annotations into the workspace bundle on any change.
        for sig in (
            self._annotations.added,
            self._annotations.removed,
            self._annotations.changed,
            self._annotations.reset,
        ):
            sig.connect(self._autosave_annotations)
            sig.connect(self._on_annotations_edited)
        self._annotations.historyChanged.connect(self._sync_history_actions)
        self._sync_history_actions()
        # Drawing on either view makes an annotation.
        self._spectrogram.selectionDrawn.connect(self._on_selection_drawn)
        self._waveform.selectionDrawn.connect(self._on_selection_drawn)
        # Detection (reviewable candidate layer).
        self._detect_panel.runRequested.connect(self._run_detection)
        self._detect_panel.promoteRequested.connect(self._promote_candidates)
        self._detect_panel.candidateActivated.connect(self._playback.play_range)

    def _on_audio_reset(self) -> None:
        # A new (or cleared) document invalidates the candidate layer.
        self._candidates.clear()
        self._measured.clear()

    # --- detection --------------------------------------------------------
    def _run_detection(self, kind: str, params: dict) -> None:
        audio = self._document.audio
        if audio is None:
            self.statusMessage.emit("Open an audio file before running detection")
            return
        if self._detect_worker is not None:
            return  # a run is already in flight
        self._detect_panel.set_running(True)
        self.statusMessage.emit("Detecting…")
        worker = Worker(run_detection, audio.samples, audio.sample_rate, kind, params)
        self._detect_worker = worker
        label = detector_label(kind)
        worker.signals.result.connect(lambda cands: self._candidates.set_all(cands, label))
        worker.signals.result.connect(
            lambda cands: self.statusMessage.emit(f"{len(cands)} detections ({label})")
        )
        worker.signals.error.connect(
            lambda exc: QMessageBox.critical(self, "Detection failed", str(exc))
        )
        worker.signals.finished.connect(self._on_detection_finished)
        QThreadPool.globalInstance().start(worker)

    def _on_detection_finished(self) -> None:
        self._detect_panel.set_running(False)
        self._detect_worker = None

    def _promote_candidates(self, candidates: list) -> None:
        label = self._active_label()
        for candidate in candidates:
            annotation = candidate_to_annotation(candidate)
            if not annotation.label:
                annotation.label = label
            self._annotations.add(annotation)
        self.statusMessage.emit(f"Promoted {len(candidates)} to annotations")

    # --- model-assisted labelling -----------------------------------------
    def _identify_dialog(self) -> None:
        audio = self._document.audio
        if audio is None or self._identify_worker is not None:
            return
        items = self._annotations.items()
        if not items:
            self.statusMessage.emit("Draw or detect some annotations first, then Identify")
            return
        selected = self._annotations.selected
        dialog = IdentifyDialog(
            str(self._settings.value(_SETTINGS_IDENTIFY_MODEL, DEFAULT_IDENTIFY_MODEL)),
            float(self._settings.value(_SETTINGS_IDENTIFY_CONFIDENCE, 0.0)),
            has_selection=selected is not None,
            counts={
                SCOPE_SELECTED: 1 if selected is not None else 0,
                SCOPE_UNLABELED: sum(1 for a in items if not a.label),
                SCOPE_ALL: len(items),
            },
            parent=self,
        )
        if dialog.exec() != QDialog.DialogCode.Accepted or not dialog.model():
            return
        self._settings.setValue(_SETTINGS_IDENTIFY_MODEL, dialog.model())
        self._settings.setValue(_SETTINGS_IDENTIFY_CONFIDENCE, dialog.min_confidence())
        if dialog.scope() == SCOPE_SELECTED:
            targets = [selected] if selected is not None else []
        elif dialog.scope() == SCOPE_ALL:
            targets = items
        else:
            targets = [a for a in items if not a.label]
        self._run_identify(targets, dialog.model(), dialog.min_confidence())

    def _run_identify(self, targets: list[Annotation], model: str, min_confidence: float) -> None:
        audio = self._document.audio
        if audio is None or not targets:
            self.statusMessage.emit("Nothing to identify")
            return
        self._identify_action.setEnabled(False)
        self.statusMessage.emit(f"Loading {model}…")
        # Copies: the worker must not see edits made while it runs.
        worker = Worker(
            identify_annotations,
            audio.samples,
            audio.sample_rate,
            [replace(a) for a in targets],
            model,
            with_progress=True,
        )
        self._identify_worker = worker
        worker.signals.progress.connect(
            lambda done, total: self.statusMessage.emit(f"Identifying… {done}/{total}")
        )
        worker.signals.result.connect(
            lambda results, path=audio.path: self._apply_identifications(
                results, path, min_confidence
            )
        )
        worker.signals.error.connect(
            lambda exc: QMessageBox.critical(self, "Identify failed", str(exc))
        )
        worker.signals.finished.connect(self._on_identify_finished)
        QThreadPool.globalInstance().start(worker)

    def _apply_identifications(self, results: dict, path: Path, min_confidence: float) -> None:
        audio = self._document.audio
        if audio is None or audio.path != path:
            return  # the recording changed while the model ran
        changed = []
        for ann in self._annotations.items():
            verdict = results.get(ann.id)
            if verdict is None or verdict.confidence < min_confidence:
                continue
            ann.label = verdict.label
            ann.confidence = verdict.confidence
            changed.append(ann)
        self._annotations.update_many(changed)  # one undo step for the whole run
        self.statusMessage.emit(
            f"Identified {len(changed)} of {len(results)} "
            f"(the rest were below {min_confidence:.2f} confidence)"
            if len(changed) < len(results)
            else f"Identified {len(changed)} annotations"
        )

    def _on_identify_finished(self) -> None:
        self._identify_worker = None
        self._identify_action.setEnabled(True)

    # --- labels -----------------------------------------------------------
    def _active_label(self) -> str:
        return self._label_combo.currentText().strip()

    def _refresh_labels(self) -> None:
        """Keep the label list = every label in use (plus the one being typed)."""
        current = self._label_combo.currentText()
        known = [self._label_combo.itemText(i) for i in range(self._label_combo.count())]
        labels = sorted({*known, *self._annotations.labels()} - {""})
        if labels != known:
            self._label_combo.blockSignals(True)
            self._label_combo.clear()
            self._label_combo.addItems(labels)
            self._label_combo.setEditText(current)
            self._label_combo.blockSignals(False)

    def _set_selected_label(self, label: str) -> None:
        selected = self._annotations.selected
        if selected is not None and selected.label != label:
            selected.label = label
            self._annotations.update(selected)

    def _apply_active_label(self) -> None:
        self._set_selected_label(self._active_label())
        self._refresh_labels()
        self._spectrogram.setFocus()

    def _apply_numbered_label(self, number: int) -> None:
        if number > self._label_combo.count():
            return
        label = self._label_combo.itemText(number - 1)
        self._label_combo.setEditText(label)
        self._set_selected_label(label)

    def _focus_label(self) -> None:
        self._label_combo.setFocus()
        self._label_combo.lineEdit().selectAll()

    # --- annotations ------------------------------------------------------
    def _autosave_annotations(self, *_args) -> None:
        if self._suppress_autosave or self._current_audio_path is None:
            return
        self._workspace.save_annotations_for(self._current_audio_path, self._annotations.items())

    def _on_annotations_edited(self, *_args) -> None:
        self._refresh_labels()
        self._measure_timer.start()

    def _sync_history_actions(self) -> None:
        self._undo_action.setEnabled(self._annotations.can_undo)
        self._redo_action.setEnabled(self._annotations.can_redo)

    def _on_selection_drawn(
        self, start: float, end: float, low: float | None, high: float | None
    ) -> None:
        if self._document.audio is None:
            return
        self._annotations.add(
            Annotation(
                start_time=start,
                end_time=end,
                low_freq=low,
                high_freq=high,
                label=self._active_label(),
            )
        )

    def _commit_keyboard_selection(self) -> None:
        """Enter: turn the S/B placed selection into an annotation."""
        bounds = self._spectrogram.selection_bounds()
        if bounds is None:
            self._table.edit_selected_label()  # no pending selection: Enter edits the label
            return
        self._on_selection_drawn(*bounds)
        self._spectrogram.clear_selection()

    def _delete_selected(self) -> None:
        selected = self._annotations.selected
        if selected is not None:
            self._annotations.remove(selected)

    def _step_selection(self, step: int) -> None:
        """Select the next/previous annotation and bring it into view."""
        target = self._annotations.select_adjacent(step)
        if target is None:
            return
        x0, x1 = self._spectrogram.visible_range()
        if not (x0 <= target.start_time and target.end_time <= x1):
            self._spectrogram.zoom_to_selection()

    def _import_annotations(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Import selection table", "", _ANNOTATION_FILTER
        )
        if not path:
            return
        try:
            anns = load_annotations(path)
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "Could not import", str(exc))
            return
        # An edit of this recording's annotations: autosaved, and undoable.
        self._annotations.set_all(anns, undoable=True)
        self.statusMessage.emit(f"Imported {len(anns)} annotations")

    def _export_annotations(self) -> None:
        path, chosen = QFileDialog.getSaveFileName(
            self,
            "Export selection table",
            "",
            "Raven selection table (*.txt);;CSV (*.csv);;CSV with measurements (*.csv)",
        )
        if not path:
            return
        try:
            if "measurements" in chosen:
                self._export_with_measurements(path)
            else:
                save_annotations(self._annotations.items(), path)
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "Could not export", str(exc))
            return
        self.statusMessage.emit(f"Exported to {Path(path).name}")

    # --- measurements -----------------------------------------------------
    def _choose_measurements(self) -> None:
        dialog = MeasurementDialog(self._measure_keys, self)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        self._measure_keys = dialog.selected_keys()
        self._settings.setValue(_SETTINGS_MEASUREMENTS, self._measure_keys)
        self._table.set_measurement_keys(self._measure_keys)
        self._measured.clear()  # different columns: recompute
        self._measure_pending()

    def _stale(self) -> list[Annotation]:
        """Annotations with no measurements for their current geometry."""
        return [
            a
            for a in self._annotations.items()
            if self._measured.get(a.id, (None,))[0] != _geometry(a)
        ]

    def _measure_pending(self) -> None:
        """Measure whatever changed, off-thread, then refresh the table."""
        audio = self._document.audio
        if audio is None or not self._measure_keys or self._measure_worker is not None:
            return
        stale = self._stale()
        if not stale:
            self._publish_measurements()
            return
        # Copies: the worker must not see edits made while it runs.
        batch = [replace(a) for a in stale]
        worker = Worker(
            measure_annotations, audio.samples, audio.sample_rate, batch, list(self._measure_keys)
        )
        self._measure_worker = worker
        worker.signals.result.connect(
            lambda values, batch=batch, path=audio.path: self._on_measured(batch, values, path)
        )
        worker.signals.finished.connect(self._on_measure_finished)
        QThreadPool.globalInstance().start(worker)

    def _on_measured(self, batch: list[Annotation], values: dict, path: Path) -> None:
        audio = self._document.audio
        if audio is None or audio.path != path:
            return  # the recording changed while measuring
        for ann in batch:
            self._measured[ann.id] = (_geometry(ann), values.get(ann.id, {}))
        self._publish_measurements()

    def _on_measure_finished(self) -> None:
        self._measure_worker = None
        if self._stale():  # edits arrived while the worker ran
            self._measure_timer.start()

    def _publish_measurements(self) -> None:
        live = {a.id for a in self._annotations.items()}
        self._measured = {k: v for k, v in self._measured.items() if k in live}
        self._table.set_measurements({k: v[1] for k, v in self._measured.items()})

    def _export_with_measurements(self, path: str) -> None:
        audio = self._document.audio
        items = self._annotations.items()
        values = (
            measure_annotations(audio.samples, audio.sample_rate, items, self._measure_keys)
            if audio is not None
            else {}
        )
        export_measurements_csv(items, values, self._measure_keys, path)
