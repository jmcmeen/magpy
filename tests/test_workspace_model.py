"""Tests for the Workspace model: artifacts, discovery, annotation persistence."""

from __future__ import annotations

from magpy.models import Workspace
from magpy.services import Annotation


def test_link_and_discover_file(tmp_path):
    audio = tmp_path / "a.wav"
    audio.write_bytes(b"\0")
    ws = Workspace()
    ws.create(tmp_path / "W.magpy", "W")

    art = ws.link(audio)
    assert art.mode == "linked"
    assert audio in ws.audio_files
    assert ws.annotation_key(audio) == art.id


def test_annotations_persist_across_reopen(tmp_path):
    audio = tmp_path / "a.wav"
    audio.write_bytes(b"\0")
    bundle = tmp_path / "W.magpy"

    ws = Workspace()
    ws.create(bundle, "W")
    ws.link(audio)
    ws.save_annotations_for(audio, [Annotation(start_time=0.0, end_time=1.0, label="bird")])

    reopened = Workspace()
    reopened.open(bundle)
    anns = reopened.load_annotations_for(audio)
    assert len(anns) == 1
    assert anns[0].label == "bird"


def test_folder_artifact_keys_annotations_per_file(tmp_path):
    folder = tmp_path / "recordings"
    folder.mkdir()
    f1 = folder / "one.wav"
    f2 = folder / "two.wav"
    f1.write_bytes(b"\0")
    f2.write_bytes(b"\0")

    ws = Workspace()
    ws.create(tmp_path / "W.magpy", "W")
    art = ws.link(folder)

    k1 = ws.annotation_key(f1)
    k2 = ws.annotation_key(f2)
    assert k1 != k2
    assert k1.startswith(art.id) and k2.startswith(art.id)


def test_remove_artifact(tmp_path):
    audio = tmp_path / "a.wav"
    audio.write_bytes(b"\0")
    ws = Workspace()
    ws.create(tmp_path / "W.magpy", "W")
    art = ws.link(audio)
    assert ws.audio_files

    ws.remove_artifact(art.id)
    assert ws.artifacts == []
    assert ws.audio_files == []
