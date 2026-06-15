"""
Cluster service -- the seam over ``bioamla.cluster`` for the Explore view.

Explore works in **embedding space**: it loads the ``.npy`` embedding files
produced by the Batch screen's "Model embeddings" op, clusters them, reduces them
to 2-D for a scatter plot, scores the clustering, and optionally flags novel
points. One call does the whole pipeline and returns a render-ready
:class:`EmbeddingScatter` so the widget never touches bioamla.

Clustering/reduction (UMAP/t-SNE especially) can be slow, so the caller runs this
through a :class:`~magpy.workers.Worker`. There is no progress hook, so it's an
indeterminate busy bar.

Gotcha handled here: ``cluster_embeddings(...).labels`` comes back as a Python
``list``; bioamla's own ``analyze_clusters_summary`` then fails on ``labels >= 0``.
We coerce to ``np.ndarray`` once, at the seam.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

# bioamla import is confined to the services layer.
from bioamla.cluster import (
    analyze_clusters_summary,
    cluster_embeddings,
    detect_novelty,
    export_clusters_to_csv,
    load_embeddings_batch,
    reduce_dimensions,
)

CLUSTER_METHODS: tuple[str, ...] = ("hdbscan", "kmeans", "dbscan", "agglomerative")
REDUCE_METHODS: tuple[str, ...] = ("pca", "umap", "tsne")


@dataclass(frozen=True)
class EmbeddingScatter:
    """Render-ready 2-D embedding scatter + clustering metrics. MagPy-owned."""

    coords: np.ndarray  # (N, 2) reduced coordinates
    labels: np.ndarray  # (N,) cluster id per point; -1 is noise
    filepaths: list[str]
    novel_indices: list[int] = field(default_factory=list)
    n_clusters: int = 0
    n_noise: int = 0
    silhouette: float | None = None
    cluster_method: str = ""
    reduce_method: str = ""
    message: str = ""

    @property
    def x(self) -> np.ndarray:
        return self.coords[:, 0]

    @property
    def y(self) -> np.ndarray:
        return self.coords[:, 1]


def _stack(arrays: list[np.ndarray]) -> np.ndarray:
    """Stack per-file embeddings into one (N, D) matrix, mean-pooling 2-D ones."""
    rows = [a.mean(axis=0) if a.ndim == 2 else a.reshape(-1) for a in arrays]
    return np.vstack(rows)


def cluster_embeddings_dir(
    input_dir: str | Path,
    *,
    cluster_method: str = "hdbscan",
    reduce_method: str = "pca",
    min_cluster_size: int = 5,
    n_clusters: int | None = None,
    recursive: bool = True,
    find_novelty: bool = False,
) -> EmbeddingScatter:
    """Load ``.npy`` embeddings under ``input_dir`` and cluster + reduce them.

    Slow (UMAP/t-SNE) -- run in a :class:`~magpy.workers.Worker`.
    """
    arrays, filepaths = load_embeddings_batch(Path(input_dir), recursive=recursive)
    if not arrays:
        raise ValueError(f"No .npy embedding files found under {input_dir}")
    matrix = _stack(arrays)

    summary = cluster_embeddings(
        matrix, method=cluster_method,
        n_clusters=n_clusters if n_clusters else None,
        min_cluster_size=min_cluster_size,
    )
    labels = np.asarray(summary.labels)
    coords = reduce_dimensions(matrix, method=reduce_method, n_components=2)

    silhouette: float | None = None
    try:  # silhouette is undefined for <2 clusters / all-noise; don't let it abort
        analysis = analyze_clusters_summary(matrix, labels, filepaths)
        silhouette = analysis.silhouette_score
    except Exception:  # noqa: BLE001
        pass

    novel_indices: list[int] = []
    if find_novelty:
        try:
            nov_summary, _scores, _mask = detect_novelty(matrix)
            novel_indices = [int(i) for i in (nov_summary.novel_indices or [])]
        except Exception:  # noqa: BLE001
            pass

    n_clusters_found = int(getattr(summary, "n_clusters", 0) or 0)
    n_noise = int(getattr(summary, "n_noise", 0) or 0)
    bits = [f"{n_clusters_found} clusters", f"{len(filepaths)} points"]
    if n_noise:
        bits.append(f"{n_noise} noise")
    if silhouette is not None:
        bits.append(f"silhouette {silhouette:.3f}")
    if find_novelty:
        bits.append(f"{len(novel_indices)} novel")

    return EmbeddingScatter(
        coords=np.asarray(coords),
        labels=labels,
        filepaths=[str(p) for p in filepaths],
        novel_indices=novel_indices,
        n_clusters=n_clusters_found,
        n_noise=n_noise,
        silhouette=silhouette,
        cluster_method=cluster_method,
        reduce_method=reduce_method,
        message="  ·  ".join(bits),
    )


def export_scatter_csv(scatter: EmbeddingScatter, output_path: str | Path) -> str:
    """Write the clustering (labels + 2-D coords + filepaths) to a CSV."""
    return export_clusters_to_csv(
        scatter.labels, scatter.filepaths, str(output_path), reduced_embeddings=scatter.coords
    )
