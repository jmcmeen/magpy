"""Round-trip tests for the MagPy workspace bundle format (no Qt, no bioamla)."""

from __future__ import annotations

from magpy.services import workspace_io as wio


def test_create_and_read_bundle(tmp_path):
    bundle = tmp_path / "Study.magpy"
    manifest = wio.create_bundle(bundle, "Study")

    assert wio.is_bundle(bundle)
    assert manifest.name == "Study"
    assert manifest.artifacts == []
    for sub in ("annotations", "imported", "datasets", "models"):
        assert (bundle / sub).is_dir()

    again = wio.read_manifest(bundle)
    assert again.name == "Study"
    assert again.version == wio.MANIFEST_VERSION


def test_artifact_round_trip(tmp_path):
    bundle = tmp_path / "W.magpy"
    manifest = wio.create_bundle(bundle, "W")
    art = wio.Artifact(kind=wio.KIND_AUDIO_FILE, path="/audio/a.wav")
    manifest.artifacts.append(art)
    wio.write_manifest(bundle, manifest)

    loaded = wio.read_manifest(bundle)
    assert len(loaded.artifacts) == 1
    got = loaded.artifacts[0]
    assert (got.kind, got.path, got.mode, got.id) == (
        wio.KIND_AUDIO_FILE, "/audio/a.wav", wio.MODE_LINKED, art.id,
    )


def test_import_copy_places_file_in_imported(tmp_path):
    bundle = tmp_path / "W.magpy"
    wio.create_bundle(bundle, "W")
    src = tmp_path / "outside.wav"
    src.write_bytes(b"RIFF....")

    rel = wio.import_copy(bundle, src)
    assert rel == "outside.wav"
    copied = wio.imported_dir(bundle) / rel
    assert copied.is_file()
    assert src.exists()  # original untouched

    # A second import of the same name does not overwrite.
    rel2 = wio.import_copy(bundle, src)
    assert rel2 != rel
    assert (wio.imported_dir(bundle) / rel2).is_file()


def test_annotation_path_purpose_suffix(tmp_path):
    bundle = tmp_path / "W.magpy"
    assert wio.annotation_path(bundle, "abc").name == "abc.csv"
    assert wio.annotation_path(bundle, "abc", "dawn").name == "abc__dawn.csv"


def test_resolve_artifact_path(tmp_path):
    bundle = tmp_path / "W.magpy"
    linked = wio.Artifact(kind=wio.KIND_AUDIO_FILE, path="/x/y.wav")
    imported = wio.Artifact(kind=wio.KIND_AUDIO_FILE, path="y.wav", mode=wio.MODE_IMPORTED)
    assert wio.resolve_artifact_path(bundle, linked).as_posix() == "/x/y.wav"
    assert wio.resolve_artifact_path(bundle, imported) == wio.imported_dir(bundle) / "y.wav"
