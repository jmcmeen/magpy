"""
Batch service -- the seam over bioamla's ``batch`` CLI group (fan-out of the
single-file ops over a whole directory).

This wraps the ~17 ``bioamla batch …`` operations: audio info/convert/resample/
normalize/trim/filter/denoise/segment/visualize, detect energy/ribbit/peaks/
accelerating, index, models predict/embed, and cluster. Each is a directory-mode
fan-out; the CSV-metadata mode the CLI also supports is deferred.

Two things make this layer non-trivial, and both are handled per-op rather than
generically (a generic dispatcher would crash on the asymmetries):

* **Signature asymmetry.** Only some bioamla batch functions accept
  ``on_progress(done,total)`` and/or ``max_workers``; forwarding either to a
  function that doesn't take it is a ``TypeError`` that kills the run. So each op
  declares ``supports_progress``/``supports_max_workers`` truthfully, the caller
  gates ``with_progress`` on it, and ``max_workers`` is only ever a form field
  for ops whose function accepts it.
* **Heterogeneous returns.** bioamla returns ``BatchResult`` for some ops, a stats
  ``dict`` for visualize, and nothing useful for the per-file loops. Every adapter
  normalises to one MagPy-owned :class:`BatchOutcome`.

For the three ops bioamla does not instrument (info/index/segment) we discover the
files and loop ourselves, calling ``on_progress`` per file -- which yields a
determinate progress bar *and* cooperative cancellation (the injected callback
raises at the next file boundary) for ops that otherwise had neither.
"""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass, field, is_dataclass
from pathlib import Path
from typing import Any, Callable, Optional

# bioamla import is confined to the services layer.
from bioamla.audio import (
    batch_convert_files,
    batch_resample_files,
    batch_transform_files,
    bandpass_filter,
    get_audio_info,
    highpass_filter,
    lowpass_filter,
    normalize_loudness,
    peak_normalize,
    segment_audio_file,
    spectral_denoise,
    trim_audio,
    trim_silence,
)
from bioamla.cluster import cluster_batch_files
from bioamla.detect import batch_detect_dir
from bioamla.indices import compute_indices_from_file
from bioamla.ml import batch_embed_files, batch_predict_files
from bioamla.viz import batch_generate_spectrograms

from .audio_io import find_audio_files

ProgressFn = Optional[Callable[[int, int], None]]


@dataclass(frozen=True)
class BatchParam:
    """One form field for a batch op. ``kind`` ∈ int/float/str/choice/bool.

    ``optional`` numerics/strings are omitted from the call when blank/zero, so
    bioamla's own default applies.
    """

    name: str
    label: str
    kind: str
    default: Any
    minimum: float = 0.0
    maximum: float = 1_000_000.0
    step: float = 1.0
    decimals: int = 2
    choices: tuple[str, ...] = ()
    help: str = ""
    optional: bool = False


@dataclass(frozen=True)
class BatchOpSpec:
    """A batch operation: how to present it and how to run it."""

    key: str
    label: str
    group: str
    description: str
    adapter: Callable[..., "BatchOutcome"]
    params: tuple[BatchParam, ...] = ()
    supports_progress: bool = False
    supports_max_workers: bool = False
    needs_output: bool = True
    output_hint: str = ""


@dataclass(frozen=True)
class BatchOutcome:
    """Normalised result of a batch run. MagPy-owned."""

    op: str
    total: int
    successful: int
    failed: int
    output_dir: str
    output_files: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    message: str = ""


# --- helpers ----------------------------------------------------------------
def _discover(input_dir: str | Path, recursive: bool) -> list[Path]:
    return find_audio_files(input_dir, recursive=recursive)


def _errs(raw: Any) -> list[str]:
    """Coerce a bioamla errors collection into display strings (capped)."""
    if not raw:
        return []
    out = []
    for e in list(raw)[:50]:
        if isinstance(e, dict):
            out.append("; ".join(f"{k}={v}" for k, v in e.items()))
        else:
            out.append(str(e))
    return out


def _from_batch_result(op: str, output_dir: str | Path, result: Any) -> BatchOutcome:
    """Normalise a bioamla ``BatchResult`` (defensively, via getattr)."""
    total = int(getattr(result, "total_files", 0) or 0)
    successful = int(getattr(result, "successful", 0) or 0)
    failed = int(getattr(result, "failed", 0) or 0)
    output_files = [str(p) for p in (getattr(result, "output_files", None) or [])]
    errors = _errs(getattr(result, "errors", None))
    return BatchOutcome(
        op=op, total=total, successful=successful, failed=failed,
        output_dir=str(output_dir), output_files=output_files, errors=errors,
        message=f"{successful}/{total} ok, {failed} failed",
    )


def _coerce_recursive(params: dict) -> bool:
    return bool(params.get("recursive", True))


def _coerce_workers(params: dict) -> int:
    return int(params.get("max_workers", 1) or 1)


# --- transform family (build a processor, run batch_transform_files) --------
def _build_processor(op: str, p: dict) -> Callable[[Any, int], Any]:
    if op == "normalize":
        if p.get("peak"):
            return lambda a, sr: peak_normalize(a)
        target_db = float(p.get("target_db", -20.0))
        return lambda a, sr: normalize_loudness(a, sr, target_db)
    if op == "trim":
        if p.get("trim_silence"):
            thr = float(p.get("silence_threshold_db", -40.0))
            return lambda a, sr: trim_silence(a, sr, thr)
        start = p.get("start") or None
        end = p.get("end") or None
        return lambda a, sr: trim_audio(a, sr, start, end)
    if op == "filter":
        order = int(p.get("order", 5))
        if p.get("bandpass_low") and p.get("bandpass_high"):
            lo, hi = float(p["bandpass_low"]), float(p["bandpass_high"])
            return lambda a, sr: bandpass_filter(a, sr, lo, hi, order)
        if p.get("lowpass"):
            c = float(p["lowpass"])
            return lambda a, sr: lowpass_filter(a, sr, c, order)
        if p.get("highpass"):
            c = float(p["highpass"])
            return lambda a, sr: highpass_filter(a, sr, c, order)
        raise ValueError("Set a lowpass, highpass, or bandpass (low+high) cutoff.")
    if op == "denoise":
        strength = float(p.get("strength", 1.0))
        return lambda a, sr: spectral_denoise(a, sr, strength)
    raise ValueError(f"No processor for op {op!r}")


def _run_transform(op: str, input_dir, output_dir, params, on_progress) -> BatchOutcome:
    processor = _build_processor(op, params)
    result = batch_transform_files(
        str(input_dir), str(output_dir), processor,
        recursive=_coerce_recursive(params),
        max_workers=_coerce_workers(params),
        on_progress=on_progress,
    )
    return _from_batch_result(op, output_dir, result)


def _run_convert(input_dir, output_dir, params, on_progress) -> BatchOutcome:
    result = batch_convert_files(
        str(input_dir), str(output_dir),
        target_format=str(params.get("target_format", "wav")),
        sample_rate=int(params["sample_rate"]) if params.get("sample_rate") else None,
        channels=int(params["channels"]) if params.get("channels") else None,
        recursive=_coerce_recursive(params),
        max_workers=_coerce_workers(params),
        on_progress=on_progress,
    )
    return _from_batch_result("convert", output_dir, result)


def _run_resample(input_dir, output_dir, params, on_progress) -> BatchOutcome:
    result = batch_resample_files(
        str(input_dir), str(output_dir), int(params["target_sample_rate"]),
        recursive=_coerce_recursive(params),
        max_workers=_coerce_workers(params),
        on_progress=on_progress,
    )
    return _from_batch_result("resample", output_dir, result)


# --- detect (max_workers yes, on_progress no) -------------------------------
_DETECT_RESERVED = {"recursive", "max_workers"}


def _run_detect(method: str, input_dir, output_dir, params, on_progress) -> BatchOutcome:
    detector_params = {k: v for k, v in params.items() if k not in _DETECT_RESERVED}
    result = batch_detect_dir(
        str(input_dir), str(output_dir), method=method,
        recursive=_coerce_recursive(params),
        max_workers=_coerce_workers(params),
        **detector_params,
    )
    outcome = _from_batch_result(f"detect:{method}", output_dir, result)
    # bioamla puts the real output (one detections_<method>.json + the detection
    # count) in metadata; output_files holds the *input* paths. Report the truth.
    meta = getattr(result, "metadata", None) or {}
    out_file = meta.get("output_file")
    total_det = meta.get("total_detections")
    return BatchOutcome(
        op=outcome.op, total=outcome.total, successful=outcome.successful,
        failed=outcome.failed, output_dir=str(output_dir),
        output_files=[str(out_file)] if out_file else [],
        errors=outcome.errors,
        message=f"{outcome.successful}/{outcome.total} files, {total_det} detections"
        if total_det is not None else outcome.message,
    )


# --- per-file loops (info / index / segment): synth progress + cancel -------
def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    fieldnames: list[str] = []
    for row in rows:
        for k in row:
            if k not in fieldnames:
                fieldnames.append(k)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _loop(op: str, input_dir, output_dir, params, on_progress, per_file, csv_name=None):
    """Run ``per_file(path) -> dict | None`` over discovered files with progress."""
    files = _discover(input_dir, _coerce_recursive(params))
    total = len(files)
    rows: list[dict] = []
    errors: list[str] = []
    successful = 0
    for i, f in enumerate(files):
        try:
            row = per_file(f)
            if row is not None:
                rows.append(row)
            successful += 1
        except Exception as exc:  # noqa: BLE001 - collect, keep going
            errors.append(f"{f.name}: {exc}")
        if on_progress is not None:
            on_progress(i + 1, total)  # raises Cancelled on cancel (Worker callback)
    out_files: list[str] = []
    if csv_name and rows:
        out_path = Path(output_dir) / csv_name
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        _write_csv(out_path, rows)
        out_files = [str(out_path)]
    return BatchOutcome(
        op=op, total=total, successful=successful, failed=len(errors),
        output_dir=str(output_dir), output_files=out_files, errors=errors,
        message=f"{successful}/{total} ok, {len(errors)} failed",
    )


def _row_for(obj: Any, extra: dict | None = None) -> dict:
    base = asdict(obj) if is_dataclass(obj) else dict(obj) if isinstance(obj, dict) else {"value": str(obj)}
    if extra:
        base = {**extra, **base}
    return base


def _run_info(input_dir, output_dir, params, on_progress) -> BatchOutcome:
    def per_file(f: Path) -> dict:
        return _row_for(get_audio_info(str(f)), {"file_name": f.name})
    return _loop("audio info", input_dir, output_dir, params, on_progress, per_file, "audio_info.csv")


def _run_index(input_dir, output_dir, params, on_progress) -> BatchOutcome:
    def per_file(f: Path) -> dict:
        return _row_for(compute_indices_from_file(str(f)), {"file_name": f.name})
    return _loop("index", input_dir, output_dir, params, on_progress, per_file, "indices.csv")


def _run_segment(input_dir, output_dir, params, on_progress) -> BatchOutcome:
    duration = float(params["duration"])
    overlap = float(params.get("overlap", 0.0))

    def per_file(f: Path) -> None:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        segment_audio_file(str(f), str(output_dir), duration=duration, overlap=overlap, prefix=f.stem)
        return None
    return _loop("audio segment", input_dir, output_dir, params, on_progress, per_file)


# --- visualize (on_progress yes, max_workers no) ----------------------------
def _run_visualize(input_dir, output_dir, params, on_progress) -> BatchOutcome:
    result = batch_generate_spectrograms(
        str(input_dir), str(output_dir),
        viz_type=str(params.get("plot_type", "mel")),
        recursive=_coerce_recursive(params),
        on_progress=on_progress,
    )
    processed = int(result.get("files_processed", 0)) if isinstance(result, dict) else 0
    failed = int(result.get("files_failed", 0)) if isinstance(result, dict) else 0
    return BatchOutcome(
        op="audio visualize", total=processed + failed, successful=processed, failed=failed,
        output_dir=str(output_dir), message=f"{processed} images, {failed} failed",
    )


# --- ML predict/embed + cluster (need model/.npy: construct-tested only) ----
def _run_predict(input_dir, output_dir, params, on_progress) -> BatchOutcome:
    result = batch_predict_files(
        str(input_dir), model_path=str(params["model_path"]),
        top_k=int(params.get("top_k", 5)),
        min_confidence=float(params.get("min_confidence", 0.0)),
        recursive=_coerce_recursive(params),
        max_workers=_coerce_workers(params),
        on_progress=on_progress,
    )
    base = _from_batch_result("models predict", input_dir, result)
    # The structured per-file predictions live ONLY in result.metadata; bioamla's
    # batch_predict_files writes no file and result.output_files holds summary
    # strings. Persist a predictions.json ourselves so the run produces something.
    predictions = (getattr(result, "metadata", None) or {}).get("predictions", [])
    out_files: list[str] = []
    if predictions:
        out_path = Path(input_dir) / "predictions.json"
        try:
            out_path.write_text(json.dumps(predictions, indent=2, default=str), encoding="utf-8")
            out_files = [str(out_path)]
        except Exception as exc:  # noqa: BLE001 - keep the counts even if the write fails
            base = BatchOutcome(
                op=base.op, total=base.total, successful=base.successful, failed=base.failed,
                output_dir=str(input_dir), errors=[*base.errors, f"write predictions.json: {exc}"],
                message=base.message,
            )
    return BatchOutcome(
        op="models predict", total=base.total, successful=base.successful, failed=base.failed,
        output_dir=str(input_dir), output_files=out_files, errors=base.errors,
        message=f"{base.successful}/{base.total} predicted"
        + (f" → {out_files[0]}" if out_files else ""),
    )


def _run_embed(input_dir, output_dir, params, on_progress) -> BatchOutcome:
    result = batch_embed_files(
        str(input_dir), str(output_dir), model_path=str(params["model_path"]),
        recursive=_coerce_recursive(params),
        max_workers=_coerce_workers(params),
        on_progress=on_progress,
    )
    return _from_batch_result("models embed", output_dir, result)


def _run_cluster(input_dir, output_dir, params, on_progress) -> BatchOutcome:
    # neither max_workers nor on_progress are accepted here.
    result = cluster_batch_files(
        str(input_dir), str(output_dir),
        method=str(params.get("method", "hdbscan")),
        n_clusters=int(params["n_clusters"]) if params.get("n_clusters") else None,
        min_cluster_size=int(params.get("min_cluster_size", 5)),
        min_samples=int(params.get("min_samples", 3)),
        recursive=_coerce_recursive(params),
    )
    return _from_batch_result("cluster", output_dir, result)


# --- shared param fragments -------------------------------------------------
_RECURSIVE = BatchParam("recursive", "Recurse into subfolders", "bool", True)
_WORKERS = BatchParam("max_workers", "Parallel workers", "int", 1, 1, 64, 1, 0)
_LOWF = BatchParam("low_freq", "Low freq (Hz)", "float", 500.0, 0, 96000, 50, 0)
_HIGHF = BatchParam("high_freq", "High freq (Hz)", "float", 5000.0, 0, 96000, 50, 0)


# --- the catalogue ----------------------------------------------------------
BATCH_OPS: tuple[BatchOpSpec, ...] = (
    BatchOpSpec(
        "audio_info", "Audio info → CSV", "Audio",
        "Read duration/sample-rate/channels for every file into a CSV.",
        _run_info, (_RECURSIVE,), supports_progress=True, output_hint="audio_info.csv",
    ),
    BatchOpSpec(
        "audio_convert", "Convert format", "Audio",
        "Convert every file to a target format (optionally resample / re-channel).",
        _run_convert,
        (
            BatchParam("target_format", "Format", "choice", "wav", choices=("wav", "mp3", "flac", "ogg")),
            BatchParam("sample_rate", "Sample rate (Hz)", "int", 0, 0, 384000, 1000, 0, optional=True,
                       help="0 keeps the source rate"),
            BatchParam("channels", "Channels", "int", 0, 0, 8, 1, 0, optional=True, help="0 keeps source"),
            _WORKERS, _RECURSIVE,
        ),
        supports_progress=True, supports_max_workers=True,
    ),
    BatchOpSpec(
        "audio_resample", "Resample", "Audio",
        "Resample every file to a target sample rate (writes .wav).",
        _run_resample,
        (BatchParam("target_sample_rate", "Sample rate (Hz)", "int", 16000, 1000, 384000, 1000, 0), _WORKERS, _RECURSIVE),
        supports_progress=True, supports_max_workers=True,
    ),
    BatchOpSpec(
        "audio_normalize", "Normalize level", "Audio",
        "RMS-normalize to a target dB, or peak-normalize.",
        lambda *a: _run_transform("normalize", *a),
        (
            BatchParam("peak", "Peak normalize (else RMS)", "bool", False),
            BatchParam("target_db", "Target dB (RMS)", "float", -20.0, -120, 0, 1, 1),
            _WORKERS, _RECURSIVE,
        ),
        supports_progress=True, supports_max_workers=True,
    ),
    BatchOpSpec(
        "audio_trim", "Trim", "Audio",
        "Trim to a time range, or strip leading/trailing silence.",
        lambda *a: _run_transform("trim", *a),
        (
            BatchParam("trim_silence", "Trim silence (else time range)", "bool", False),
            BatchParam("start", "Start (s)", "float", 0.0, 0, 100000, 0.1, 2, optional=True),
            BatchParam("end", "End (s)", "float", 0.0, 0, 100000, 0.1, 2, optional=True),
            BatchParam("silence_threshold_db", "Silence threshold (dB)", "float", -40.0, -120, 0, 1, 1),
            _WORKERS, _RECURSIVE,
        ),
        supports_progress=True, supports_max_workers=True,
    ),
    BatchOpSpec(
        "audio_filter", "Filter", "Audio",
        "Apply a lowpass, highpass, or bandpass Butterworth filter.",
        lambda *a: _run_transform("filter", *a),
        (
            BatchParam("lowpass", "Lowpass cutoff (Hz)", "float", 0.0, 0, 192000, 50, 0, optional=True),
            BatchParam("highpass", "Highpass cutoff (Hz)", "float", 0.0, 0, 192000, 50, 0, optional=True),
            BatchParam("bandpass_low", "Bandpass low (Hz)", "float", 0.0, 0, 192000, 50, 0, optional=True),
            BatchParam("bandpass_high", "Bandpass high (Hz)", "float", 0.0, 0, 192000, 50, 0, optional=True),
            BatchParam("order", "Filter order", "int", 5, 1, 12, 1, 0),
            _WORKERS, _RECURSIVE,
        ),
        supports_progress=True, supports_max_workers=True,
    ),
    BatchOpSpec(
        "audio_denoise", "Denoise", "Audio",
        "Spectral noise reduction on every file.",
        lambda *a: _run_transform("denoise", *a),
        (BatchParam("strength", "Strength (0–2)", "float", 1.0, 0, 2, 0.1, 2), _WORKERS, _RECURSIVE),
        supports_progress=True, supports_max_workers=True,
    ),
    BatchOpSpec(
        "audio_segment", "Segment", "Audio",
        "Split every file into fixed-duration (optionally overlapping) clips.",
        _run_segment,
        (
            BatchParam("duration", "Segment duration (s)", "float", 5.0, 0.1, 100000, 0.5, 2),
            BatchParam("overlap", "Overlap (s)", "float", 0.0, 0, 100000, 0.1, 2),
            _RECURSIVE,
        ),
        supports_progress=True,
    ),
    BatchOpSpec(
        "audio_visualize", "Spectrogram images", "Audio",
        "Render a spectrogram/waveform image (PNG) per file.",
        _run_visualize,
        (BatchParam("plot_type", "Type", "choice", "mel", choices=("mel", "stft", "mfcc", "waveform")), _RECURSIVE),
        supports_progress=True,
    ),
    BatchOpSpec(
        "detect_energy", "Detect: energy", "Detect",
        "Band-limited energy detection per file → detections_energy.json.",
        lambda *a: _run_detect("energy", *a),
        (
            _LOWF, _HIGHF,
            BatchParam("threshold_db", "Threshold (dB)", "float", -20.0, -120, 0, 1, 1),
            BatchParam("min_duration", "Min duration (s)", "float", 0.05, 0, 10, 0.01, 3),
            _WORKERS, _RECURSIVE,
        ),
        supports_max_workers=True,
    ),
    BatchOpSpec(
        "detect_ribbit", "Detect: RIBBIT", "Detect",
        "Pulse-rate (RIBBIT) detection per file → detections_ribbit.json.",
        lambda *a: _run_detect("ribbit", *a),
        (
            BatchParam("pulse_rate_hz", "Pulse rate (Hz)", "float", 10.0, 0.1, 200, 0.5, 2),
            BatchParam("pulse_rate_tolerance", "Rate tolerance", "float", 0.2, 0, 1, 0.05, 2),
            _LOWF, _HIGHF,
            BatchParam("window_duration", "Window (s)", "float", 2.0, 0.1, 30, 0.1, 2),
            BatchParam("min_score", "Min score", "float", 0.3, 0, 1, 0.05, 2),
            _WORKERS, _RECURSIVE,
        ),
        supports_max_workers=True,
    ),
    BatchOpSpec(
        "detect_peaks", "Detect: CWT peaks", "Detect",
        "Wavelet peak detection per file → detections_peaks.json.",
        lambda *a: _run_detect("peaks", *a),
        (
            BatchParam("snr_threshold", "SNR threshold", "float", 2.0, 0, 50, 0.5, 2),
            BatchParam("min_peak_distance", "Min peak gap (s)", "float", 0.01, 0, 10, 0.01, 3),
            _WORKERS, _RECURSIVE,
        ),
        supports_max_workers=True,
    ),
    BatchOpSpec(
        "detect_accelerating", "Detect: accelerating", "Detect",
        "Accelerating pulse-train detection per file → detections_accelerating.json.",
        lambda *a: _run_detect("accelerating", *a),
        (
            BatchParam("min_pulses", "Min pulses", "int", 5, 2, 100, 1, 0),
            BatchParam("acceleration_threshold", "Accel. threshold", "float", 1.5, 1, 10, 0.1, 2),
            _LOWF, _HIGHF,
            BatchParam("window_duration", "Window (s)", "float", 3.0, 0.1, 30, 0.1, 2),
            _WORKERS, _RECURSIVE,
        ),
        supports_max_workers=True,
    ),
    BatchOpSpec(
        "index", "Acoustic indices → CSV", "Analyze",
        "Compute acoustic indices (ACI/ADI/AEI/BIO/NDSI/entropy) per file into a CSV.",
        _run_index, (_RECURSIVE,), supports_progress=True, output_hint="indices.csv",
    ),
    BatchOpSpec(
        "models_predict", "Model predict", "Models",
        "Run an AST classifier over every file → predictions.json. Downloads the model.",
        _run_predict,
        (
            BatchParam("model_path", "Model (HF id or path)", "str", "bioamla/scp-frogs"),
            BatchParam("top_k", "Top-K", "int", 5, 1, 100, 1, 0),
            BatchParam("min_confidence", "Min confidence", "float", 0.0, 0, 1, 0.05, 2),
            _WORKERS, _RECURSIVE,
        ),
        supports_progress=True, supports_max_workers=True, needs_output=False,
    ),
    BatchOpSpec(
        "models_embed", "Model embeddings", "Models",
        "Extract AST embeddings (.npy) per file. Downloads the model.",
        _run_embed,
        (
            BatchParam("model_path", "Model (HF id or path)", "str", "MIT/ast-finetuned-audioset-10-10-0.4593"),
            _WORKERS, _RECURSIVE,
        ),
        supports_progress=True, supports_max_workers=True, output_hint=".npy embeddings",
    ),
    BatchOpSpec(
        "cluster", "Cluster embeddings", "Models",
        "Cluster a directory of embedding .npy files → cluster_assignments.json.",
        _run_cluster,
        (
            BatchParam("method", "Method", "choice", "hdbscan",
                       choices=("hdbscan", "kmeans", "dbscan", "agglomerative")),
            BatchParam("n_clusters", "Num clusters", "int", 0, 0, 1000, 1, 0, optional=True,
                       help="kmeans/agglomerative; 0 = auto"),
            BatchParam("min_cluster_size", "Min cluster size", "int", 5, 2, 1000, 1, 0),
            BatchParam("min_samples", "Min samples", "int", 3, 1, 1000, 1, 0),
            _RECURSIVE,
        ),
    ),
)

BATCH_OPS_BY_KEY = {op.key: op for op in BATCH_OPS}


def run_batch_op(op_key: str, input_dir, output_dir, params: dict, on_progress: ProgressFn = None) -> BatchOutcome:
    """Dispatch a batch op by key. ``on_progress`` is injected by the Worker only
    for ops with ``supports_progress`` (gated by the caller), so adapters that
    forward it to bioamla never receive it for a function that can't take it."""
    spec = BATCH_OPS_BY_KEY.get(op_key)
    if spec is None:
        raise ValueError(f"Unknown batch op: {op_key!r}")
    return spec.adapter(input_dir, output_dir, params, on_progress)
