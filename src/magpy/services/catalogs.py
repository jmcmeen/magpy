"""
Catalog service -- thin wrapper over ``bioamla.catalogs`` for the GUI.

Part of the services seam: search + download against the external sound catalogs
(Xeno-Canto, Macaulay Library, iNaturalist) and the eBird observation API. It
collapses each catalog's bespoke result type into one MagPy-owned
:class:`CatalogRecord`, so the catalog screens render a single shape and bioamla
types never leak past the seam.

Two realities shape this module:

* **It hits the network and may need API keys** (eBird requires one; Xeno-Canto
  v3 and Macaulay may). Keys are read from environment variables with an optional
  per-call override; there is no token editor yet. Callers run every function
  through a :class:`~magpy.workers.Worker` -- these block on I/O.
* **The upstream payloads are not all dataclasses.** iNaturalist observations are
  raw API dicts; eBird records are observations, not audio. The mapping here is
  deliberately *defensive* (``getattr``/``.get`` with ``"—"``/`""` fallbacks) so a
  renamed upstream field degrades to a blank cell instead of crashing a screen
  this layer cannot be smoke-tested against offline.

Download is **by record id** (every catalog's ``download_recording`` accepts the
id as a string / the observation id as an int), so no bioamla recording object is
ever held on the MagPy side.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

# bioamla import is confined to the services layer.
from bioamla.catalogs import (
    EBirdService,
    inat as _inat,
    macaulay as _macaulay,
    xeno_canto as _xc,
)

from .audio_io import find_audio_files

SOURCE_XENO_CANTO = "xeno_canto"
SOURCE_MACAULAY = "macaulay"
SOURCE_INATURALIST = "inaturalist"
SOURCE_EBIRD = "ebird"


@dataclass(frozen=True)
class CatalogRecord:
    """A single catalog search hit, normalised across sources. MagPy-owned.

    ``downloadable`` is False for eBird observations (they are sightings, not
    audio assets); the screen hides its Download action for those.
    """

    source: str
    record_id: str
    common_name: str
    scientific_name: str
    quality: str
    duration: str
    location: str
    country: str
    recordist: str
    license: str
    url: str
    downloadable: bool = True


def _s(value: object) -> str:
    """Coerce a possibly-missing value to a display string (empty for ``None``)."""
    return "" if value is None else str(value)


# --- search -----------------------------------------------------------------
def search_xeno_canto(
    *,
    species: str | None = None,
    country: str | None = None,
    quality: str | None = None,
    max_results: int = 30,
) -> list[CatalogRecord]:
    result = _xc.search(
        species=species or None,
        country=country or None,
        quality=quality or None,
        max_results=max_results,
    )
    return [
        CatalogRecord(
            source=SOURCE_XENO_CANTO,
            record_id=_s(getattr(r, "id", "")),
            common_name=_s(getattr(r, "common_name", "")),
            scientific_name=_s(getattr(r, "scientific_name", "")),
            quality=_s(getattr(r, "quality", "")),
            duration=_s(getattr(r, "length", "")),
            location=_s(getattr(r, "location", "")),
            country=_s(getattr(r, "country", "")),
            recordist=_s(getattr(r, "recordist", "")),
            license=_s(getattr(r, "license", "")),
            url=_s(getattr(r, "url", "")),
        )
        for r in getattr(result, "recordings", [])
    ]


def search_macaulay(
    *,
    scientific_name: str | None = None,
    common_name: str | None = None,
    region: str | None = None,
    min_rating: int = 0,
    max_results: int = 30,
) -> list[CatalogRecord]:
    result = _macaulay.search_audio(
        scientific_name=scientific_name or None,
        region=region or None,
        min_rating=min_rating,
        max_results=max_results,
    ) if not common_name else _macaulay.search(
        common_name=common_name,
        scientific_name=scientific_name or None,
        region=region or None,
        min_rating=min_rating,
        max_results=max_results,
    )
    records = []
    for r in getattr(result, "recordings", []):
        asset = getattr(r, "asset_id", "")
        records.append(
            CatalogRecord(
                source=SOURCE_MACAULAY,
                record_id=_s(asset),
                common_name=_s(getattr(r, "common_name", "")),
                scientific_name=_s(getattr(r, "scientific_name", "")),
                quality=_s(getattr(r, "rating", "")),
                duration=_s(getattr(r, "duration", "")),
                location=_s(getattr(r, "location", "")),
                country=_s(getattr(r, "country", "")),
                recordist=_s(getattr(r, "user_display_name", "")),
                license="",
                url=f"https://macaulaylibrary.org/asset/{asset}" if asset else "",
            )
        )
    return records


def search_inaturalist(
    *,
    taxon_name: str | None = None,
    place_id: int | None = None,
    quality_grade: str | None = "research",
    max_results: int = 30,
) -> list[CatalogRecord]:
    result = _inat.search(
        taxon_name=taxon_name or None,
        place_id=place_id,
        quality_grade=quality_grade or None,
        per_page=max_results,
    )
    records = []
    for obs in getattr(result, "observations", []) or []:
        taxon = obs.get("taxon") or {}
        user = obs.get("user") or {}
        obs_id = obs.get("id")
        records.append(
            CatalogRecord(
                source=SOURCE_INATURALIST,
                record_id=_s(obs_id),
                common_name=_s(taxon.get("preferred_common_name")),
                scientific_name=_s(taxon.get("name")),
                quality=_s(obs.get("quality_grade")),
                duration=_s(len(obs.get("sounds") or [])) + " sound(s)",
                location=_s(obs.get("place_guess")),
                country="",
                recordist=_s(user.get("login") or user.get("name")),
                license=_s(obs.get("license_code")),
                url=_s(obs.get("uri") or f"https://www.inaturalist.org/observations/{obs_id}"),
            )
        )
    return records


def _ebird_service(api_key: str | None) -> EBirdService:
    key = api_key or os.environ.get("EBIRD_API_KEY")
    return EBirdService(api_key=key)


def search_ebird_region(
    *,
    region_code: str,
    back: int = 14,
    max_results: int = 100,
    api_key: str | None = None,
) -> list[CatalogRecord]:
    result = _ebird_service(api_key).get_recent_observations(
        region_code=region_code, back=back, max_results=max_results
    )
    records = []
    for obs in getattr(result, "observations", []) or []:
        subid = getattr(obs, "subid", "")
        how_many = getattr(obs, "how_many", None)
        records.append(
            CatalogRecord(
                source=SOURCE_EBIRD,
                record_id=_s(getattr(obs, "obs_id", "") or subid),
                common_name=_s(getattr(obs, "common_name", "")),
                scientific_name=_s(getattr(obs, "scientific_name", "")),
                quality="",
                duration=_s(getattr(obs, "observation_date", "")),
                location=_s(getattr(obs, "location_name", "")),
                country="",
                recordist=("" if how_many is None else f"{how_many} seen"),
                license="",
                url=f"https://ebird.org/checklist/{subid}" if subid else "",
                downloadable=False,
            )
        )
    return records


# --- download ---------------------------------------------------------------
def download_records(
    source: str,
    record_ids: list[str],
    dest_dir: str | Path,
) -> list[Path]:
    """Download the given records into ``dest_dir`` and return the new audio files.

    Returns the audio files that appeared under ``dest_dir`` (compared to before
    the call), so the caller can link them into the workspace. iNaturalist
    observation ids are coerced to ``int`` as bioamla requires.
    """
    dest = Path(dest_dir)
    dest.mkdir(parents=True, exist_ok=True)
    before = set(find_audio_files(dest))

    if source == SOURCE_XENO_CANTO:
        for rid in record_ids:
            _xc.download_recording(rid, dest, organize_by_species=False)
    elif source == SOURCE_MACAULAY:
        for rid in record_ids:
            _macaulay.download_recording(rid, dest, organize_by_species=False)
    elif source == SOURCE_INATURALIST:
        obs_ids = [int(rid) for rid in record_ids if rid.isdigit()]
        if obs_ids:
            _inat.download_from_observations(obs_ids, str(dest), organize_by_taxon=False)
    else:
        raise ValueError(f"Source {source!r} does not support download")

    after = set(find_audio_files(dest))
    return sorted(after - before)
