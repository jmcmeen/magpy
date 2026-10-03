"""
Hugging Face service -- thin wrapper over ``bioamla.catalogs.huggingface``.

Part of the services seam: pull a dataset repo into a local directory and inspect
or purge the Hugging Face hub cache. Like the rest of ``catalogs``, this hits the
network and may need a token (``HF_TOKEN`` -- read from the environment by
``huggingface_hub``; pull of a *public* dataset needs none). bioamla return types
(``PullResult``/``CachedRepo``/``PurgeResult``) are collapsed into MagPy-owned
DTOs here so they never leak past the seam, and the screen runs every call
through a :class:`~magpy.workers.Worker` because they block on I/O.

Push (``push_dataset``/``push_model``) is intentionally out of scope for now --
it needs a prepared local path and a token, beyond the "basic" pull + cache
features. Add it here if/when the screen grows a publish tab.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

# bioamla import is confined to the services layer.
from bioamla.catalogs import huggingface as _hf

from .audio_io import find_audio_files


def human_bytes(size: int) -> str:
    """Render a byte count as a short human string (e.g. ``1.4 GB``)."""
    value = float(size)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if value < 1024 or unit == "TB":
            return f"{value:.0f} {unit}" if unit == "B" else f"{value:.1f} {unit}"
        value /= 1024
    return f"{value:.1f} TB"


@dataclass(frozen=True)
class HFPullResult:
    """The outcome of pulling a dataset, normalised. MagPy-owned."""

    repo_id: str
    dest: Path
    num_files: int
    labels: list[str]
    splits: list[str]
    audio_files: list[Path]  # audio that landed under ``dest`` (for linking)


@dataclass(frozen=True)
class HFCachedRepo:
    """One repo in the local hub cache. MagPy-owned."""

    repo_id: str
    repo_type: str
    size_bytes: int

    @property
    def size_human(self) -> str:
        return human_bytes(self.size_bytes)


@dataclass(frozen=True)
class HFPurgeResult:
    """The outcome of a cache purge, normalised. MagPy-owned."""

    deleted: list[str]
    freed_bytes: int
    failures: list[str] = field(default_factory=list)

    @property
    def freed_human(self) -> str:
        return human_bytes(self.freed_bytes)


def pull_dataset(
    repo_id: str,
    dest: str | Path,
    *,
    split: str | None = None,
    sample_rate: int | None = 16000,
) -> HFPullResult:
    """Pull ``repo_id`` into ``dest`` and return a render-ready result.

    Audio files that appear under ``dest`` are collected so the caller can link
    them into the open workspace.
    """
    dest_path = Path(dest)
    dest_path.mkdir(parents=True, exist_ok=True)
    result = _hf.pull_dataset(
        repo_id,
        str(dest_path),
        split=split or None,
        sample_rate=sample_rate,
        verbose=False,
    )
    files_written = list(getattr(result, "files_written", []) or [])
    return HFPullResult(
        repo_id=str(getattr(result, "repo_id", repo_id)),
        dest=Path(getattr(result, "dest", dest_path)),
        num_files=len(files_written),
        labels=[str(x) for x in (getattr(result, "labels", []) or [])],
        splits=[str(x) for x in (getattr(result, "splits", []) or [])],
        audio_files=sorted(find_audio_files(dest_path)),
    )


def scan_hf_cache(*, models: bool = True, datasets: bool = True) -> list[HFCachedRepo]:
    """List repos in the local hub cache."""
    repos = _hf.scan_cache(models=models, datasets=datasets)
    return [
        HFCachedRepo(
            repo_id=str(getattr(r, "repo_id", "")),
            repo_type=str(getattr(r, "repo_type", "")),
            size_bytes=int(getattr(r, "size_bytes", 0) or 0),
        )
        for r in repos or []
    ]


def purge_hf_cache(*, models: bool = True, datasets: bool = True) -> HFPurgeResult:
    """Delete cached repos, freeing disk. Returns what was removed."""
    result = _hf.purge_cache(models=models, datasets=datasets)
    return HFPurgeResult(
        deleted=[str(x) for x in (getattr(result, "deleted", []) or [])],
        freed_bytes=int(getattr(result, "freed_bytes", 0) or 0),
        failures=[str(x) for x in (getattr(result, "failures", []) or [])],
    )
