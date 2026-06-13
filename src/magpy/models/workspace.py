"""
Workspace -- MagPy's stateful "project", backed by a ``.magpy`` bundle on disk.

A workspace is the single unit the user opens (exactly one at a time). It owns a
list of :class:`~magpy.services.Artifact`\\ s -- audio files, folders, and later
datasets/models -- each either **linked** (referenced by path, never copied,
shareable across workspaces) or **imported** (copied into the bundle). The audio
the user actually annotates is discovered from those artifacts; the annotations
MagPy produces live inside the bundle as sidecars, keyed per *file*.

Persistence lives in :mod:`magpy.services.workspace_io`; this model wraps it as a
``QObject`` so views can bind to ``opened`` / ``artifactsChanged``. Kept separate
from :class:`Document` (the *open* audio); the shell wires the "activate a file ->
load it into the document, load its annotations" edge.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from PyQt6.QtCore import QObject, pyqtSignal

from magpy.services import (
    KIND_AUDIO_FILE,
    KIND_FOLDER,
    MODE_IMPORTED,
    MODE_LINKED,
    Annotation,
    Artifact,
    WorkspaceManifest,
    annotation_path,
    create_bundle,
    find_audio_files,
    import_copy,
    load_annotations,
    read_manifest,
    resolve_artifact_path,
    save_annotations,
    write_manifest,
)


def _sanitize(rel: str) -> str:
    return rel.replace("/", "_").replace("\\", "_").replace(" ", "_")


class Workspace(QObject):
    opened = pyqtSignal(object)  # bundle dir (Path)
    closed = pyqtSignal()
    artifactsChanged = pyqtSignal()

    def __init__(self, parent: Optional[QObject] = None) -> None:
        super().__init__(parent)
        self._dir: Optional[Path] = None
        self._manifest: Optional[WorkspaceManifest] = None

    # --- identity ---------------------------------------------------------
    @property
    def dir(self) -> Optional[Path]:
        """The ``.magpy`` bundle directory, or ``None`` if nothing is open."""
        return self._dir

    # Back-compat alias used by some views/dialogs as "the workspace location".
    @property
    def root(self) -> Optional[Path]:
        return self._dir

    @property
    def is_open(self) -> bool:
        return self._dir is not None

    @property
    def name(self) -> str:
        return self._manifest.name if self._manifest else ""

    @property
    def artifacts(self) -> list[Artifact]:
        return list(self._manifest.artifacts) if self._manifest else []

    # --- lifecycle --------------------------------------------------------
    def create(self, bundle_dir: str | Path, name: str | None = None) -> None:
        """Create a fresh bundle and open it."""
        self._dir = Path(bundle_dir)
        self._manifest = create_bundle(self._dir, name)
        self.opened.emit(self._dir)

    def open(self, bundle_dir: str | Path) -> None:
        """Open an existing bundle."""
        self._dir = Path(bundle_dir)
        self._manifest = read_manifest(self._dir)
        self.opened.emit(self._dir)

    def close(self) -> None:
        self._dir = None
        self._manifest = None
        self.closed.emit()

    def _persist(self) -> None:
        if self._dir and self._manifest:
            write_manifest(self._dir, self._manifest)

    # --- artifacts --------------------------------------------------------
    def link(self, path: str | Path, kind: str | None = None) -> Artifact:
        """Add ``path`` as a *linked* (referenced, not copied) artifact."""
        path = Path(path)
        artifact = Artifact(
            kind=kind or (KIND_FOLDER if path.is_dir() else KIND_AUDIO_FILE),
            path=str(path),
            mode=MODE_LINKED,
        )
        self._append(artifact)
        return artifact

    def import_(self, path: str | Path, kind: str | None = None) -> Artifact:
        """Copy ``path`` into the bundle and add it as an *imported* artifact."""
        path = Path(path)
        if self._dir is None:
            raise RuntimeError("No workspace open")
        rel = import_copy(self._dir, path)
        artifact = Artifact(
            kind=kind or (KIND_FOLDER if path.is_dir() else KIND_AUDIO_FILE),
            path=rel,
            mode=MODE_IMPORTED,
        )
        self._append(artifact)
        return artifact

    def remove_artifact(self, artifact_id: str) -> None:
        if not self._manifest:
            return
        self._manifest.artifacts = [a for a in self._manifest.artifacts if a.id != artifact_id]
        self._persist()
        self.artifactsChanged.emit()

    def _append(self, artifact: Artifact) -> None:
        if not self._manifest:
            raise RuntimeError("No workspace open")
        self._manifest.artifacts.append(artifact)
        self._persist()
        self.artifactsChanged.emit()

    # --- audio discovery --------------------------------------------------
    @property
    def audio_files(self) -> list[Path]:
        """All audio reachable from the artifacts (folders expanded), de-duplicated."""
        seen: dict[Path, None] = {}
        for art in self.artifacts:
            resolved = resolve_artifact_path(self._dir, art)
            if art.kind == KIND_FOLDER:
                for p in find_audio_files(resolved):
                    seen.setdefault(p, None)
            elif art.kind == KIND_AUDIO_FILE:
                seen.setdefault(resolved, None)
        return sorted(seen)

    def resolved_path(self, artifact: Artifact) -> Path:
        """Absolute on-disk path for ``artifact`` (handles linked vs imported)."""
        return resolve_artifact_path(self._dir, artifact)

    def files_for(self, artifact: Artifact) -> list[Path]:
        """The audio files an artifact contributes (a folder's contents, or itself)."""
        resolved = resolve_artifact_path(self._dir, artifact)
        if artifact.kind == KIND_FOLDER:
            return find_audio_files(resolved)
        if artifact.kind == KIND_AUDIO_FILE:
            return [resolved]
        return []

    def exists(self, artifact: Artifact) -> bool:
        return resolve_artifact_path(self._dir, artifact).exists()

    # --- annotations (keyed per file) ------------------------------------
    def _artifact_for_audio(self, path: Path) -> Optional[Artifact]:
        """The artifact that owns ``path``: a direct file match, else a containing folder."""
        path = Path(path)
        folder: Optional[Artifact] = None
        for art in self.artifacts:
            resolved = resolve_artifact_path(self._dir, art)
            if art.kind == KIND_AUDIO_FILE and resolved == path:
                return art
            if art.kind == KIND_FOLDER and resolved in path.parents:
                folder = art
        return folder

    def artifact_for_path(self, path: str | Path) -> Optional[Artifact]:
        """The artifact owning ``path`` (a file artifact, or a containing folder)."""
        return self._artifact_for_audio(Path(path))

    def annotation_key(self, path: str | Path) -> Optional[str]:
        """Stable per-file key for annotation storage, or ``None`` if unknown."""
        path = Path(path)
        art = self._artifact_for_audio(path)
        if art is None:
            return None
        if art.kind == KIND_AUDIO_FILE:
            return art.id
        rel = path.relative_to(resolve_artifact_path(self._dir, art))
        return f"{art.id}_{_sanitize(rel.as_posix())}"

    def load_annotations_for(self, path: str | Path) -> list[Annotation]:
        """Load the stored annotations for ``path`` (empty list if none/unknown)."""
        if self._dir is None:
            return []
        key = self.annotation_key(path)
        if key is None:
            return []
        sidecar = annotation_path(self._dir, key)
        return load_annotations(sidecar) if sidecar.exists() else []

    def save_annotations_for(self, path: str | Path, annotations: list[Annotation]) -> None:
        """Write ``annotations`` to ``path``'s sidecar inside the bundle."""
        if self._dir is None:
            return
        key = self.annotation_key(path)
        if key is None:
            return
        sidecar = annotation_path(self._dir, key)
        sidecar.parent.mkdir(parents=True, exist_ok=True)
        save_annotations(annotations, sidecar)
