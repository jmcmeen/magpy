"""
Datasets service -- the seam over bioamla's dataset-building fan-outs
(``bioamla dataset …`` + manifest/license).

This is the home of turning annotated source audio into a training-ready dataset:
extract labeled clips, partition into train/val/test, offline-augment, merge
datasets, inspect stats, and write a manifest / license. Per the architecture,
**source audio is immutable** -- every op here writes a *new* dataset directory or
sidecar, never mutating the inputs.

None of these bioamla functions expose an ``on_progress`` hook, so the screen runs
them on an indeterminate busy bar (no determinate progress, no cooperative
cancel). All return ``dict[str, Any]`` (or a ``DatasetManifest``); each adapter
normalises to one MagPy-owned :class:`DatasetOutcome`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# bioamla import is confined to the services layer.
from bioamla.datasets import (
    AugmentationConfig,
    batch_augment,
    build_manifest_from_metadata,
    extract_labeled_dataset,
    generate_license_for_dataset,
    get_dataset_stats,
    merge_datasets,
    partition_dataset,
    save_dataset_manifest,
)

from .batch import BatchParam


@dataclass(frozen=True)
class DatasetOutcome:
    """Normalised result of a dataset op. MagPy-owned."""

    op: str
    message: str
    output_dir: str
    details: dict[str, Any] = field(default_factory=dict)


def _labels(text: str | None) -> set[str] | None:
    if not text:
        return None
    items = {s.strip() for s in text.split(",") if s.strip()}
    return items or None


def _summary(d: dict, keys: tuple[str, ...]) -> str:
    parts = [f"{k}={d[k]}" for k in keys if k in d and not isinstance(d[k], (dict, list))]
    return "  ".join(parts)


def extract_clips(
    source: str,
    output_dir: str,
    *,
    annotations: str | None = None,
    layout: str = "both",
    target_sample_rate: int | None = None,
    padding_ms: float = 0.0,
    min_duration: float | None = None,
    include_labels: str | None = None,
    exclude_labels: str | None = None,
) -> DatasetOutcome:
    r = extract_labeled_dataset(
        source, output_dir,
        annotations=annotations or None,
        layout=layout,
        padding_ms=padding_ms,
        target_sample_rate=target_sample_rate or None,
        min_duration=min_duration or None,
        include_labels=_labels(include_labels),
        exclude_labels=_labels(exclude_labels),
        verbose=False,
    )
    return DatasetOutcome(
        op="extract-clips",
        message=_summary(r, ("clips_written", "files_processed", "failed", "skipped"))
        or "Done.",
        output_dir=str(r.get("output_dir", output_dir)),
        details=r,
    )


def partition(
    dataset_dir: str,
    *,
    train: float = 0.7,
    val: float = 0.15,
    test: float = 0.15,
    seed: int = 0,
    stratify: bool = True,
    mode: str = "subdirs",
    group_by: str | None = "source_file",
) -> DatasetOutcome:
    r = partition_dataset(
        dataset_dir, splits=(train, val, test), seed=seed, stratify=stratify,
        mode=mode, group_by=group_by or None, verbose=False,
    )
    splits = r.get("splits")
    if isinstance(splits, dict):
        message = "  ".join(
            f"{k}={v if not isinstance(v, (list, dict)) else len(v)}" for k, v in splits.items()
        )
    else:
        message = "Partitioned."
    return DatasetOutcome(op="partition", message=message or "Partitioned.", output_dir=dataset_dir, details=r)


def merge(dataset_paths: list[str], output_dir: str, *, target_format: str | None = None) -> DatasetOutcome:
    paths = [p for p in dataset_paths if p and p.strip()]
    if len(paths) < 2:
        raise ValueError("Merge needs at least two dataset folders.")
    r = merge_datasets(paths, output_dir, target_format=target_format or None, verbose=False)
    return DatasetOutcome(
        op="merge",
        message=_summary(r, ("total_files", "files_copied", "files_skipped", "datasets_merged"))
        or f"Merged {len(paths)} datasets.",
        output_dir=str(r.get("output_dir", output_dir)),
        details=r,
    )


# Augmentation knobs surfaced in the UI (subset of AugmentationConfig). Each
# transform is a toggle plus its range; `multiply` makes N augmented copies.
AUGMENT_PARAMS: tuple[BatchParam, ...] = (
    BatchParam("add_noise", "Add noise", "bool", False),
    BatchParam("noise_min_snr", "Noise min SNR (dB)", "float", 3.0, 0, 100, 1, 1),
    BatchParam("noise_max_snr", "Noise max SNR (dB)", "float", 30.0, 0, 100, 1, 1),
    BatchParam("time_stretch", "Time stretch", "bool", False),
    BatchParam("time_stretch_min", "Stretch min", "float", 0.8, 0.1, 3, 0.05, 2),
    BatchParam("time_stretch_max", "Stretch max", "float", 1.2, 0.1, 3, 0.05, 2),
    BatchParam("pitch_shift", "Pitch shift", "bool", False),
    BatchParam("pitch_shift_min", "Pitch min (semitones)", "float", -4.0, -24, 24, 1, 1),
    BatchParam("pitch_shift_max", "Pitch max (semitones)", "float", 4.0, -24, 24, 1, 1),
    BatchParam("gain", "Gain", "bool", False),
    BatchParam("gain_min_db", "Gain min (dB)", "float", -12.0, -60, 60, 1, 1),
    BatchParam("gain_max_db", "Gain max (dB)", "float", 12.0, -60, 60, 1, 1),
    BatchParam("sample_rate", "Sample rate (Hz)", "int", 16000, 1000, 384000, 1000, 0),
    BatchParam("multiply", "Copies per file", "int", 1, 1, 50, 1, 0),
)


def augment(input_dir: str, output_dir: str, params: dict, *, recursive: bool = True) -> DatasetOutcome:
    cfg = AugmentationConfig(
        add_noise=bool(params.get("add_noise", False)),
        noise_min_snr=float(params.get("noise_min_snr", 3.0)),
        noise_max_snr=float(params.get("noise_max_snr", 30.0)),
        time_stretch=bool(params.get("time_stretch", False)),
        time_stretch_min=float(params.get("time_stretch_min", 0.8)),
        time_stretch_max=float(params.get("time_stretch_max", 1.2)),
        pitch_shift=bool(params.get("pitch_shift", False)),
        pitch_shift_min=float(params.get("pitch_shift_min", -4.0)),
        pitch_shift_max=float(params.get("pitch_shift_max", 4.0)),
        gain=bool(params.get("gain", False)),
        gain_min_db=float(params.get("gain_min_db", -12.0)),
        gain_max_db=float(params.get("gain_max_db", 12.0)),
        sample_rate=int(params.get("sample_rate", 16000)),
        multiply=int(params.get("multiply", 1)),
    )
    r = batch_augment(input_dir, output_dir, cfg, recursive=recursive, verbose=False)
    return DatasetOutcome(
        op="augment",
        message=_summary(r, ("files_processed", "files_created", "files_failed")) or "Augmented.",
        output_dir=str(r.get("output_dir", output_dir)),
        details=r,
    )


def dataset_stats(dataset_dir: str) -> DatasetOutcome:
    r = get_dataset_stats(dataset_dir)
    return DatasetOutcome(
        op="stats",
        message=_summary(r, ("total_files", "num_categories", "num_licenses")) or "Inspected.",
        output_dir=dataset_dir,
        details=r,
    )


def build_manifest(dataset_dir: str, *, name: str = "", sample_rate: int | None = None) -> DatasetOutcome:
    manifest = build_manifest_from_metadata(dataset_dir, name=name, sample_rate=sample_rate or None)
    out_path = str(Path(dataset_dir) / "dataset.json")
    save_dataset_manifest(manifest, out_path)
    details = {
        "name": getattr(manifest, "name", name),
        "kind": getattr(manifest, "kind", ""),
        "classes": len(getattr(manifest, "label2id", {}) or {}),
        "manifest": out_path,
    }
    return DatasetOutcome(op="manifest", message=f"Wrote {out_path}", output_dir=dataset_dir, details=details)


def generate_license(dataset_dir: str, *, fmt: str = "text") -> DatasetOutcome:
    r = generate_license_for_dataset(Path(dataset_dir), format=fmt)
    return DatasetOutcome(
        op="license",
        message=_summary(r, ("output_file", "entries")) or "License written.",
        output_dir=dataset_dir,
        details=r,
    )
