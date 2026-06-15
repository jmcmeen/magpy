"""
Services -- the thin seam over bioamla.

Every bioamla import in MagPy lives in this package. Widgets, models, and
screens call these functions instead of bioamla directly, so that bioamla API
churn is contained here and GUI defaults (compute backend, FFT sizes, ...) have
a single home. Mostly stateless functions; the playback ``Player`` is the
exception (it wraps an inherently stateful engine). Not Qt-aware.
"""

from .annotations import Annotation, load_annotations, save_annotations
from .audio_io import LoadedAudio, find_audio_files, load_audio
from .detect import (
    DETECTOR_SPECS,
    Candidate,
    DetectorSpec,
    ParamSpec,
    candidate_to_annotation,
    detector_label,
    run_detection,
)
from .catalogs import (
    SOURCE_EBIRD,
    SOURCE_INATURALIST,
    SOURCE_MACAULAY,
    SOURCE_XENO_CANTO,
    CatalogRecord,
    download_records,
    search_ebird_region,
    search_inaturalist,
    search_macaulay,
    search_xeno_canto,
)
from .batch import (
    BATCH_OPS,
    BATCH_OPS_BY_KEY,
    BatchOpSpec,
    BatchOutcome,
    BatchParam,
    run_batch_op,
)
from .cluster import (
    CLUSTER_METHODS,
    REDUCE_METHODS,
    EmbeddingScatter,
    cluster_embeddings_dir,
    export_scatter_csv,
)
from .datasets import (
    AUGMENT_PARAMS,
    DatasetOutcome,
    augment,
    build_manifest,
    dataset_stats,
    extract_clips,
    generate_license,
    merge,
    partition,
)
from .env_io import (
    KNOWN_ENV_VARS,
    EnvVarSpec,
    apply_to_environ,
    default_env_path,
    load_into_environ,
    read_env,
    write_env,
)
from .indices import IndexRow, IndexSummary, compute_indices
from .training import (
    TRAIN_PARAMS,
    EvalOutcome,
    PredictOutcome,
    TrainOutcome,
    evaluate_model,
    list_available_models,
    predict_audio,
    run_training,
)
from .system_info import (
    DependencyReport,
    DependencyRow,
    DeviceReport,
    DeviceRow,
    VersionInfo,
    dependency_report,
    device_report,
    version_info,
)
from .playback import PlaybackState, Player
from .spectrogram import SpectrogramImage, compute_spectrogram
from .workspace_io import (
    BUNDLE_SUFFIX,
    KIND_AUDIO_FILE,
    KIND_DATASET,
    KIND_FOLDER,
    KIND_MODEL,
    MODE_IMPORTED,
    MODE_LINKED,
    Artifact,
    WorkspaceManifest,
    annotation_path,
    create_bundle,
    import_copy,
    is_bundle,
    read_manifest,
    resolve_artifact_path,
    write_manifest,
)

__all__ = [
    "LoadedAudio",
    "load_audio",
    "find_audio_files",
    "SpectrogramImage",
    "compute_spectrogram",
    "Player",
    "PlaybackState",
    "Annotation",
    "load_annotations",
    "save_annotations",
    # detection (reviewable candidate layer)
    "Candidate",
    "DetectorSpec",
    "ParamSpec",
    "DETECTOR_SPECS",
    "run_detection",
    "detector_label",
    "candidate_to_annotation",
    # acoustic indices (whole-file summary)
    "IndexSummary",
    "IndexRow",
    "compute_indices",
    # system info (Settings view)
    "VersionInfo",
    "DependencyReport",
    "DependencyRow",
    "DeviceReport",
    "DeviceRow",
    "version_info",
    "dependency_report",
    "device_report",
    # catalogs (search/download external sound libraries)
    "CatalogRecord",
    "SOURCE_XENO_CANTO",
    "SOURCE_MACAULAY",
    "SOURCE_INATURALIST",
    "SOURCE_EBIRD",
    "search_xeno_canto",
    "search_macaulay",
    "search_inaturalist",
    "search_ebird_region",
    "download_records",
    # batch (fan-out of single-file ops over a directory)
    "BatchParam",
    "BatchOpSpec",
    "BatchOutcome",
    "BATCH_OPS",
    "BATCH_OPS_BY_KEY",
    "run_batch_op",
    # datasets (build training data from annotated sources)
    "DatasetOutcome",
    "AUGMENT_PARAMS",
    "extract_clips",
    "partition",
    "merge",
    "augment",
    "dataset_stats",
    "build_manifest",
    "generate_license",
    # cluster / explore (embedding space)
    "EmbeddingScatter",
    "CLUSTER_METHODS",
    "REDUCE_METHODS",
    "cluster_embeddings_dir",
    "export_scatter_csv",
    # AST training / evaluation / prediction
    "TRAIN_PARAMS",
    "TrainOutcome",
    "EvalOutcome",
    "PredictOutcome",
    "run_training",
    "evaluate_model",
    "predict_audio",
    "list_available_models",
    # environment / API-key persistence (MagPy-owned, no bioamla)
    "EnvVarSpec",
    "KNOWN_ENV_VARS",
    "default_env_path",
    "read_env",
    "write_env",
    "apply_to_environ",
    "load_into_environ",
    # workspace persistence (MagPy-owned, no bioamla)
    "Artifact",
    "WorkspaceManifest",
    "BUNDLE_SUFFIX",
    "KIND_AUDIO_FILE",
    "KIND_FOLDER",
    "KIND_DATASET",
    "KIND_MODEL",
    "MODE_LINKED",
    "MODE_IMPORTED",
    "annotation_path",
    "create_bundle",
    "import_copy",
    "is_bundle",
    "read_manifest",
    "resolve_artifact_path",
    "write_manifest",
]
