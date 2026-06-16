"""Tests for the Hugging Face service mapping.

Like the catalog service, the network/token-bound paths can't run offline, so
these target the *mapping* layer by faking bioamla's return shapes
(``PullResult``/``CachedRepo``/``PurgeResult``). The mapping must be defensive
(missing attributes degrade, not crash) and collapse to MagPy-owned DTOs.
"""

from __future__ import annotations

from dataclasses import dataclass

from magpy.services import (
    HFCachedRepo,
    HFPullResult,
    human_bytes,
    huggingface,
    pull_dataset,
    purge_hf_cache,
    scan_hf_cache,
)


@dataclass
class _FakePull:
    repo_id = "user/birds"
    dest = "/tmp/x"
    url = "https://huggingface.co/datasets/user/birds"
    files_written = ["a.wav", "b.wav", "meta.json"]
    labels = ["robin", "crow"]
    splits = ["train", "test"]
    metadata_file = "meta.json"


@dataclass
class _FakeRepo:
    repo_id = "user/birds"
    repo_type = "dataset"
    size_bytes = 1_500_000_000


@dataclass
class _FakePurge:
    deleted = ["user/birds"]
    freed_bytes = 2048
    failures: tuple = ()


def test_human_bytes_scales():
    assert human_bytes(0) == "0 B"
    assert human_bytes(1024) == "1.0 KB"
    assert human_bytes(1_500_000_000).endswith("GB")


def test_pull_dataset_maps_and_collects_audio(monkeypatch, tmp_path):
    (tmp_path / "clip.wav").write_bytes(b"\x00")
    monkeypatch.setattr(huggingface._hf, "pull_dataset", lambda *a, **k: _FakePull())
    out = pull_dataset("user/birds", tmp_path)
    assert isinstance(out, HFPullResult)
    assert out.repo_id == "user/birds"
    assert out.num_files == 3
    assert out.labels == ["robin", "crow"]
    assert out.splits == ["train", "test"]
    # audio under dest is collected from the filesystem for linking
    assert any(p.name == "clip.wav" for p in out.audio_files)


def test_scan_cache_maps_repos(monkeypatch):
    monkeypatch.setattr(huggingface._hf, "scan_cache", lambda **k: [_FakeRepo()])
    out = scan_hf_cache()
    assert len(out) == 1
    repo = out[0]
    assert isinstance(repo, HFCachedRepo)
    assert repo.repo_id == "user/birds"
    assert repo.repo_type == "dataset"
    assert repo.size_human.endswith("GB")


def test_scan_cache_tolerates_missing_attrs(monkeypatch):
    monkeypatch.setattr(huggingface._hf, "scan_cache", lambda **k: [object()])
    out = scan_hf_cache()
    assert len(out) == 1
    assert out[0].repo_id == "" and out[0].size_bytes == 0


def test_purge_cache_maps(monkeypatch):
    monkeypatch.setattr(huggingface._hf, "purge_cache", lambda **k: _FakePurge())
    out = purge_hf_cache()
    assert out.deleted == ["user/birds"]
    assert out.freed_bytes == 2048
    assert out.failures == []
