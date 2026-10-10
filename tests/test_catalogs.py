"""Tests for the catalog service mapping.

The network/key-bound paths cannot be exercised offline, so these tests target
the *mapping* layer -- the unverifiable risk surface flagged during the build --
by faking the bioamla return shapes. iNaturalist returns raw dicts; the others
return objects; both must map defensively without raising.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from magpy.services import (
    SOURCE_INATURALIST,
    CatalogRecord,
    catalogs,
    download_records,
    search_inaturalist,
    search_xeno_canto,
)


@dataclass
class _FakeXC:
    id = "98765"
    common_name = "Common Blackbird"
    scientific_name = "Turdus merula"
    quality = "A"
    length = "0:23"
    location = "Berlin"
    country = "Germany"
    recordist = "rec"
    license = "CC-BY"
    url = "https://xeno-canto.org/98765"


class _FakeResult:
    def __init__(self, recordings=None, observations=None):
        self.recordings = recordings or []
        self.observations = observations or []


def test_xeno_canto_maps_recording(monkeypatch):
    monkeypatch.setattr(catalogs._xc, "search", lambda **kw: _FakeResult(recordings=[_FakeXC()]))
    out = search_xeno_canto(species="Turdus merula")
    assert len(out) == 1
    rec = out[0]
    assert isinstance(rec, CatalogRecord)
    assert rec.record_id == "98765"
    assert rec.scientific_name == "Turdus merula"
    assert rec.duration == "0:23"


def test_inaturalist_maps_raw_dicts(monkeypatch):
    obs = [
        {
            "id": 42,
            "taxon": {"name": "Turdus migratorius", "preferred_common_name": "American Robin"},
            "user": {"login": "birder"},
            "quality_grade": "research",
            "sounds": [{"file_url": "x"}, {"file_url": "y"}],
            "place_guess": "New York",
            "license_code": "cc-by",
        }
    ]
    monkeypatch.setattr(catalogs._inat, "search", lambda **kw: _FakeResult(observations=obs))
    out = search_inaturalist(taxon_name="Turdus")
    assert len(out) == 1
    rec = out[0]
    assert rec.source == SOURCE_INATURALIST
    assert rec.record_id == "42"
    assert rec.common_name == "American Robin"
    assert "2 sound" in rec.duration
    assert rec.recordist == "birder"


def test_inaturalist_tolerates_missing_keys(monkeypatch):
    # A sparse observation must not raise (defensive mapping).
    monkeypatch.setattr(catalogs._inat, "search", lambda **kw: _FakeResult(observations=[{}]))
    out = search_inaturalist(taxon_name="x")
    assert len(out) == 1
    assert out[0].common_name == ""


def test_download_records_rejects_non_downloadable_source():
    with pytest.raises(ValueError):
        download_records("ebird", ["1"], "/tmp/does-not-matter")
