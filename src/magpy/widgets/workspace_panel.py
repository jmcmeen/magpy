"""
WorkspacePanel -- the Workspace Explorer for the open :class:`Workspace`.

Not a flat file list: a tree organised around the workspace's *artifacts*. A
top-level "Audio" group holds each linked/imported source -- a folder expands to
the audio it contains, a standalone file is a leaf -- with **imported** and
**missing** tags. (Datasets/models become sibling groups once those screens emit
artifacts.) Double-clicking a file emits :attr:`fileActivated` (absolute path);
right-click removes the owning artifact. A dumb view bound to the model.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QAction
from PyQt6.QtWidgets import (
    QLabel,
    QMenu,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from magpy.models import Workspace
from magpy.services import KIND_AUDIO_FILE, KIND_FOLDER

_PATH_ROLE = Qt.ItemDataRole.UserRole  # absolute path of an activatable file
_ARTIFACT_ROLE = Qt.ItemDataRole.UserRole + 1  # owning artifact id


class WorkspacePanel(QWidget):
    fileActivated = pyqtSignal(str)  # absolute path

    def __init__(self, workspace: Workspace, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._workspace = workspace

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self._header = QLabel("No workspace open")
        self._header.setStyleSheet("padding: 6px 8px; color: #808080;")
        self._tree = QTreeWidget()
        self._tree.setHeaderHidden(True)
        self._tree.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        layout.addWidget(self._header)
        layout.addWidget(self._tree)

        self._workspace.opened.connect(lambda _dir: self._refresh())
        self._workspace.closed.connect(self._refresh)
        self._workspace.artifactsChanged.connect(self._refresh)
        self._tree.itemDoubleClicked.connect(self._on_item_activated)
        self._tree.customContextMenuRequested.connect(self._on_context_menu)

    # --- population -------------------------------------------------------
    def _refresh(self) -> None:
        self._tree.clear()
        if not self._workspace.is_open:
            self._header.setText("No workspace open")
            return
        files = self._workspace.audio_files
        self._header.setText(f"{self._workspace.name}  ·  {len(files)} files")

        audio_arts = [
            a for a in self._workspace.artifacts if a.kind in (KIND_FOLDER, KIND_AUDIO_FILE)
        ]
        if not audio_arts:
            return
        group = QTreeWidgetItem(["Audio"])
        group.setFirstColumnSpanned(True)
        self._tree.addTopLevelItem(group)
        for art in audio_arts:
            self._add_artifact(group, art)
        group.setExpanded(True)

    def _add_artifact(self, group: QTreeWidgetItem, artifact) -> None:
        from magpy.services import MODE_IMPORTED  # local import keeps the header tidy

        resolved = self._workspace.resolved_path(artifact)
        imported = artifact.mode == MODE_IMPORTED
        if artifact.kind == KIND_FOLDER:
            label = resolved.name + ("  (imported)" if imported else "")
            if not resolved.exists():
                label += "  (missing)"
            folder_item = QTreeWidgetItem([label])
            folder_item.setData(0, _ARTIFACT_ROLE, artifact.id)
            folder_item.setToolTip(0, str(resolved))
            group.addChild(folder_item)
            for f in self._workspace.files_for(artifact):
                self._add_file(folder_item, f, artifact.id, imported=imported)
            folder_item.setExpanded(True)
        else:  # KIND_AUDIO_FILE
            self._add_file(group, resolved, artifact.id, imported=imported)

    def _add_file(self, parent: QTreeWidgetItem, path: Path, artifact_id: str, *, imported: bool) -> None:
        label = path.name
        if imported:
            label += "  (imported)"
        missing = not path.exists()
        if missing:
            label += "  (missing)"
        item = QTreeWidgetItem([label])
        item.setData(0, _PATH_ROLE, str(path))
        item.setData(0, _ARTIFACT_ROLE, artifact_id)
        item.setToolTip(0, str(path))
        if missing:
            item.setForeground(0, Qt.GlobalColor.gray)
        parent.addChild(item)

    # --- interaction ------------------------------------------------------
    def _on_item_activated(self, item: QTreeWidgetItem, _col: int) -> None:
        path = item.data(0, _PATH_ROLE)
        if path:
            self.fileActivated.emit(path)

    def _on_context_menu(self, pos) -> None:
        item = self._tree.itemAt(pos)
        if item is None:
            return
        artifact_id = item.data(0, _ARTIFACT_ROLE)
        if not artifact_id:
            return
        artifact = next((a for a in self._workspace.artifacts if a.id == artifact_id), None)
        if artifact is None:
            return
        menu = QMenu(self)
        remove = QAction(f"Remove “{Path(artifact.path).name}” from workspace", self)
        remove.triggered.connect(lambda: self._workspace.remove_artifact(artifact.id))
        menu.addAction(remove)
        menu.exec(self._tree.mapToGlobal(pos))
