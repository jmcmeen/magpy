"""Tests for the cluster/explore service seam (run for real on synthetic embeddings)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from magpy.services import (
    CLUSTER_METHODS,
    REDUCE_METHODS,
    EmbeddingScatter,
    cluster_embeddings_dir,
    export_scatter_csv,
)


@pytest.fixture
def embeddings_dir(tmp_path: Path) -> Path:
    d = tmp_path / "emb"
    d.mkdir()
    rng = np.random.RandomState(7)
    centers = [rng.randn(32) * 0.1 + k * 5 for k in range(3)]
    for i in range(30):
        np.save(d / f"e{i:02d}.npy", (centers[i % 3] + rng.randn(32) * 0.3).astype("float32"))
    return d


def test_methods_exposed():
    assert "hdbscan" in CLUSTER_METHODS
    assert "pca" in REDUCE_METHODS


def test_cluster_and_reduce(embeddings_dir):
    sc = cluster_embeddings_dir(embeddings_dir, cluster_method="hdbscan", reduce_method="pca",
                                min_cluster_size=3, find_novelty=True)
    assert isinstance(sc, EmbeddingScatter)
    assert sc.coords.shape == (30, 2)
    assert sc.labels.shape == (30,)
    assert sc.n_clusters >= 2
    assert sc.silhouette is not None  # labels coerced to ndarray so analyze doesn't crash


def test_export_csv(embeddings_dir, tmp_path):
    sc = cluster_embeddings_dir(embeddings_dir, reduce_method="pca", min_cluster_size=3)
    out = tmp_path / "clusters.csv"
    export_scatter_csv(sc, out)
    assert out.exists()


def test_empty_dir_raises(tmp_path):
    with pytest.raises(ValueError):
        cluster_embeddings_dir(tmp_path / "nope_empty")
