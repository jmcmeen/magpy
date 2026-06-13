"""
Settings Screen -- API keys / environment, plus system information.

Two concerns:

* **Environment & API keys** -- an editable form over the variables bioamla reads
  (Xeno-Canto / eBird keys, the Hugging Face token, the HF cache dir). Saving
  writes MagPy's own ``.env`` *and* applies the values to ``os.environ`` so the
  catalog screens work in the same session (the keys are read lazily). bioamla's
  own ``load_dotenv`` searches site-packages, not here, so MagPy manages and
  applies its keys itself (see :mod:`magpy.services.env_io`).
* **System info** -- bioamla's ``system`` group: versions, native dependencies
  (both cheap, built synchronously in ``_setup_ui``), and compute-device
  enumeration (triggers the ~1.2 s torch import, so it runs lazily on first
  activation through a :class:`~magpy.workers.Worker`).

Pure Qt + the services seam + the workers bridge -- no bioamla imports here.
"""

from __future__ import annotations

from pathlib import Path

from PyQt6.QtCore import Qt, QThreadPool
from PyQt6.QtWidgets import (
    QCheckBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QPushButton,
    QScrollArea,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from magpy.services import (
    KNOWN_ENV_VARS,
    DependencyReport,
    DeviceReport,
    VersionInfo,
    apply_to_environ,
    dependency_report,
    device_report,
    read_env,
    version_info,
    write_env,
)
from magpy.workers import Worker

from .base import BaseScreen


class SettingsScreen(BaseScreen):
    def __init__(self, env_path: str | Path, parent: QWidget | None = None) -> None:
        # Set before super().__init__: BaseScreen.__init__ calls _setup_ui, which
        # reads the .env to seed the editor.
        self._env_path = Path(env_path)
        super().__init__(parent)

    @property
    def screen_name(self) -> str:
        return "Settings"

    @property
    def screen_icon(self) -> str:
        return "⚙️"

    def _setup_ui(self) -> None:
        self._devices_loaded = False
        self._device_worker = None
        self._key_fields: dict[str, QLineEdit] = {}

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(scroll.Shape.NoFrame)
        outer.addWidget(scroll)

        body = QWidget()
        scroll.setWidget(body)
        layout = QVBoxLayout(body)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(16)

        title = QLabel("Settings")
        title.setStyleSheet("font-size: 22px; font-weight: bold; color: #d4d4d4;")
        layout.addWidget(title)

        # Environment & API keys (cheap -> build + populate now) -----------
        layout.addWidget(self._build_env_box())

        # Versions (cheap) -------------------------------------------------
        version_box = QGroupBox("Versions")
        self._version_form = QFormLayout(version_box)
        layout.addWidget(version_box)
        self._populate_versions(version_info())

        # Native dependencies (cheap) -------------------------------------
        dep_box = QGroupBox("Native dependencies")
        dep_layout = QVBoxLayout(dep_box)
        self._dep_table = QTableWidget(0, 4)
        self._dep_table.setHorizontalHeaderLabels(["Dependency", "Status", "Version", "Used for"])
        self._dep_table.verticalHeader().setVisible(False)
        self._dep_table.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeMode.Stretch)
        dep_layout.addWidget(self._dep_table)
        self._dep_hint = QLabel()
        self._dep_hint.setWordWrap(True)
        self._dep_hint.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self._dep_hint.setStyleSheet("color: #858585;")
        dep_layout.addWidget(self._dep_hint)
        layout.addWidget(dep_box)
        self._populate_dependencies(dependency_report())

        # Compute devices (torch import -> lazy, threaded on activate) ------
        dev_box = QGroupBox("Compute devices")
        dev_layout = QVBoxLayout(dev_box)
        self._device_label = QLabel("Open this screen to detect devices…")
        self._device_label.setStyleSheet("color: #858585;")
        dev_layout.addWidget(self._device_label)
        self._device_table = QTableWidget(0, 4)
        self._device_table.setHorizontalHeaderLabels(["Device", "Type", "ID", "Memory (GB)"])
        self._device_table.verticalHeader().setVisible(False)
        self._device_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        dev_layout.addWidget(self._device_table)
        layout.addWidget(dev_box)

        layout.addStretch(1)

    # --- environment / API keys ------------------------------------------
    def _build_env_box(self) -> QGroupBox:
        box = QGroupBox("Environment & API keys")
        box_layout = QVBoxLayout(box)

        caption = QLabel(
            "Keys are saved to MagPy's .env and applied immediately (catalogs read "
            "them on the next request). Cache-dir changes apply on restart."
        )
        caption.setWordWrap(True)
        caption.setStyleSheet("color: #858585;")
        box_layout.addWidget(caption)

        form = QFormLayout()
        current = read_env(self._env_path)
        for spec in KNOWN_ENV_VARS:
            field = QLineEdit(current.get(spec.name, ""))
            field.setToolTip(spec.purpose)
            if spec.secret:
                field.setEchoMode(QLineEdit.EchoMode.Password)
            label = spec.label + (" (restart)" if spec.applies_on_restart else "")
            self._key_fields[spec.name] = field
            form.addRow(label, field)
        box_layout.addLayout(form)

        controls = QHBoxLayout()
        show = QCheckBox("Show values")
        show.toggled.connect(self._toggle_secrecy)
        controls.addWidget(show)
        controls.addStretch(1)
        save = QPushButton("Save")
        save.clicked.connect(self._save_env)
        controls.addWidget(save)
        box_layout.addLayout(controls)

        self._env_status = QLabel(
            f".env: {self._env_path}" if current else "No keys saved yet."
        )
        self._env_status.setStyleSheet("color: #858585;")
        self._env_status.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        box_layout.addWidget(self._env_status)
        return box

    def _toggle_secrecy(self, show: bool) -> None:
        mode = QLineEdit.EchoMode.Normal if show else QLineEdit.EchoMode.Password
        for spec in KNOWN_ENV_VARS:
            if spec.secret:
                self._key_fields[spec.name].setEchoMode(mode)

    def _save_env(self) -> None:
        values = {name: field.text().strip() for name, field in self._key_fields.items()}
        try:
            write_env(self._env_path, values)
            apply_to_environ(values)
        except Exception as exc:  # noqa: BLE001 - surface to the user
            self._env_status.setText(f"Could not save: {exc}")
            return
        saved = sum(1 for v in values.values() if v)
        self._env_status.setText(f"Saved {saved} value(s) to {self._env_path}")

    def on_activate(self) -> None:
        # Device detection is the only expensive part (torch import); do it once,
        # lazily, the first time the screen is shown.
        if self._devices_loaded:
            return
        self._devices_loaded = True
        self._device_label.setText("Detecting…")
        self._start_device_detection()

    # --- versions ---------------------------------------------------------
    def _populate_versions(self, v: VersionInfo) -> None:
        rows = [
            ("bioamla", v.bioamla_version),
            ("Python", v.python_version.split()[0]),
            ("Platform", v.platform),
            ("PyTorch", v.pytorch_version or "—"),
            ("CUDA", v.cuda_version or "—"),
        ]
        for name, value in rows:
            label = QLabel(value)
            label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
            self._version_form.addRow(f"{name}:", label)

    # --- dependencies -----------------------------------------------------
    def _populate_dependencies(self, report: DependencyReport) -> None:
        self._dep_table.setRowCount(len(report.dependencies))
        for r, dep in enumerate(report.dependencies):
            status = "✓ installed" if dep.installed else "✗ missing"
            cells = [dep.name, status, dep.version or "—", dep.required_for]
            for c, text in enumerate(cells):
                item = QTableWidgetItem(text)
                if c == 1 and not dep.installed:
                    item.setForeground(Qt.GlobalColor.red)
                self._dep_table.setItem(r, c, item)
        self._dep_table.resizeColumnsToContents()
        self._dep_table.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeMode.Stretch)
        if report.all_installed:
            self._dep_hint.setText(f"All dependencies present ({report.os_type}).")
        else:
            self._dep_hint.setText(f"Install missing: {report.install_command}")

    # --- devices (threaded; torch import) ---------------------------------
    def _start_device_detection(self) -> None:
        worker = Worker(device_report)
        self._device_worker = worker
        worker.signals.result.connect(self._populate_devices)
        worker.signals.error.connect(
            lambda exc: self._device_label.setText(f"Device detection failed: {exc}")
        )
        worker.signals.finished.connect(lambda: setattr(self, "_device_worker", None))
        QThreadPool.globalInstance().start(worker)

    def _populate_devices(self, report: DeviceReport) -> None:
        flags = []
        flags.append("CUDA available" if report.cuda_available else "CUDA unavailable")
        if report.mps_available:
            flags.append("MPS available")
        self._device_label.setText(" · ".join(flags))
        self._device_table.setRowCount(len(report.devices))
        for r, dev in enumerate(report.devices):
            cells = [
                dev.name,
                dev.device_type,
                "—" if dev.device_id is None else str(dev.device_id),
                "—" if dev.memory_gb is None else f"{dev.memory_gb:.1f}",
            ]
            for c, text in enumerate(cells):
                self._device_table.setItem(r, c, QTableWidgetItem(text))
