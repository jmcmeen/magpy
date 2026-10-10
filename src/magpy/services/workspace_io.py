"""
Workspace persistence -- the on-disk ``.magpy`` bundle format.

This is MagPy-owned state, **not** a bioamla concept, so unlike its neighbours in
this package it imports no bioamla -- it sits *beside* the seam, reusing the
services layer only as the natural home for non-Qt I/O and DTOs. A workspace is a
``MyStudy.magpy/`` directory bundle:

    MyStudy.magpy/
      workspace.json   # this manifest
      annotations/     # <artifactId>.csv sidecars (purpose-keyed)
      imported/        # copies created by Import (mode="imported")
      datasets/        # datasets produced here (themselves linkable elsewhere)
      models/          # models produced here (themselves linkable elsewhere)

The single primitive is the :class:`Artifact`: anything the workspace points at
(audio, a folder, a dataset, a model) is either ``linked`` (referenced by an
absolute path, never copied -- shareable across workspaces) or ``imported`` (a
copy living under ``imported/``, addressed by a bundle-relative path).
"""

from __future__ import annotations

import json
import shutil
from dataclasses import asdict, dataclass, field
from pathlib import Path
from uuid import uuid4

BUNDLE_SUFFIX = ".magpy"
MANIFEST_NAME = "workspace.json"
MANIFEST_VERSION = 1

# Artifact kinds and modes are kept as plain strings (JSON-friendly).
KIND_AUDIO_FILE = "audio_file"
KIND_FOLDER = "folder"
KIND_DATASET = "dataset"
KIND_MODEL = "model"

MODE_LINKED = "linked"
MODE_IMPORTED = "imported"


def _new_id() -> str:
    return uuid4().hex[:8]


@dataclass
class Artifact:
    """One thing a workspace points at.

    ``path`` is an absolute filesystem path when ``mode == "linked"``, and a path
    relative to the bundle's ``imported/`` dir when ``mode == "imported"``.
    """

    kind: str  # KIND_*
    path: str
    mode: str = MODE_LINKED
    id: str = field(default_factory=_new_id)


@dataclass
class WorkspaceManifest:
    """The decoded contents of ``workspace.json``."""

    name: str
    artifacts: list[Artifact] = field(default_factory=list)
    settings: dict = field(default_factory=dict)
    version: int = MANIFEST_VERSION


# --- bundle path helpers --------------------------------------------------


def manifest_path(bundle_dir: str | Path) -> Path:
    return Path(bundle_dir) / MANIFEST_NAME


def annotations_dir(bundle_dir: str | Path) -> Path:
    return Path(bundle_dir) / "annotations"


def imported_dir(bundle_dir: str | Path) -> Path:
    return Path(bundle_dir) / "imported"


def datasets_dir(bundle_dir: str | Path) -> Path:
    return Path(bundle_dir) / "datasets"


def models_dir(bundle_dir: str | Path) -> Path:
    return Path(bundle_dir) / "models"


def annotation_path(bundle_dir: str | Path, artifact_id: str, purpose: str = "default") -> Path:
    """Sidecar path for an artifact's annotations. ``default`` purpose is unsuffixed."""
    stem = artifact_id if purpose == "default" else f"{artifact_id}__{purpose}"
    return annotations_dir(bundle_dir) / f"{stem}.csv"


def resolve_artifact_path(bundle_dir: str | Path, artifact: Artifact) -> Path:
    """Absolute on-disk path for an artifact, whether linked or imported."""
    if artifact.mode == MODE_IMPORTED:
        return imported_dir(bundle_dir) / artifact.path
    return Path(artifact.path)


# --- manifest I/O ---------------------------------------------------------


def _ensure_subdirs(bundle_dir: Path) -> None:
    for sub in (annotations_dir, imported_dir, datasets_dir, models_dir):
        sub(bundle_dir).mkdir(parents=True, exist_ok=True)


def create_bundle(bundle_dir: str | Path, name: str | None = None) -> WorkspaceManifest:
    """Create a fresh ``.magpy`` bundle (dirs + manifest) and return its manifest."""
    bundle_dir = Path(bundle_dir)
    bundle_dir.mkdir(parents=True, exist_ok=True)
    _ensure_subdirs(bundle_dir)
    manifest = WorkspaceManifest(name=name or bundle_dir.stem)
    write_manifest(bundle_dir, manifest)
    return manifest


def read_manifest(bundle_dir: str | Path) -> WorkspaceManifest:
    """Read ``workspace.json`` from a bundle dir."""
    raw = json.loads(manifest_path(bundle_dir).read_text(encoding="utf-8"))
    artifacts = [
        Artifact(
            kind=a["kind"],
            path=a["path"],
            mode=a.get("mode", MODE_LINKED),
            id=a.get("id") or _new_id(),
        )
        for a in raw.get("artifacts", [])
    ]
    return WorkspaceManifest(
        name=raw.get("name", Path(bundle_dir).stem),
        artifacts=artifacts,
        settings=raw.get("settings", {}),
        version=raw.get("version", MANIFEST_VERSION),
    )


def write_manifest(bundle_dir: str | Path, manifest: WorkspaceManifest) -> None:
    """Write ``manifest`` to the bundle's ``workspace.json`` (pretty, stable order)."""
    bundle_dir = Path(bundle_dir)
    bundle_dir.mkdir(parents=True, exist_ok=True)
    data = {
        "version": manifest.version,
        "name": manifest.name,
        "settings": manifest.settings,
        "artifacts": [asdict(a) for a in manifest.artifacts],
    }
    manifest_path(bundle_dir).write_text(json.dumps(data, indent=2), encoding="utf-8")


def is_bundle(path: str | Path) -> bool:
    """True if ``path`` is a directory holding a ``workspace.json``."""
    path = Path(path)
    return path.is_dir() and manifest_path(path).is_file()


def import_copy(bundle_dir: str | Path, src_path: str | Path) -> str:
    """Copy ``src_path`` into the bundle's ``imported/`` dir; return the relative path.

    The returned value is what an imported :class:`Artifact` stores as ``path``.
    A name clash is disambiguated with a short id prefix rather than overwriting.
    """
    src_path = Path(src_path)
    dest_dir = imported_dir(bundle_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / src_path.name
    if dest.exists():
        dest = dest_dir / f"{_new_id()}_{src_path.name}"
    if src_path.is_dir():
        shutil.copytree(src_path, dest)
    else:
        shutil.copy2(src_path, dest)
    return dest.relative_to(dest_dir).as_posix()
