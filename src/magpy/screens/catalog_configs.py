"""
Catalog configurations -- the per-source :class:`CatalogConfig` instances.

Each entry wires a search form (field names match the corresponding
:mod:`magpy.services.catalogs` search-function keyword arguments) to its search
function. The generic :class:`~magpy.screens.catalog.CatalogScreen`
renders them. Keep all bioamla/network specifics in the service; these are pure
descriptors.

eBird is search-only (``downloadable=False``) -- it returns sightings, not audio
assets -- and needs an ``EBIRD_API_KEY`` (env var, or the optional form field).
"""

from __future__ import annotations

from magpy.services import (
    SOURCE_EBIRD,
    SOURCE_INATURALIST,
    SOURCE_MACAULAY,
    SOURCE_XENO_CANTO,
    search_ebird_region,
    search_inaturalist,
    search_macaulay,
    search_xeno_canto,
)

from .catalog import CatalogConfig, CatalogField

XENO_CANTO_CONFIG = CatalogConfig(
    source=SOURCE_XENO_CANTO,
    title="Xeno-Canto",
    icon="🎼",
    blurb="Search the Xeno-Canto recording archive and download into the workspace.",
    search_fn=search_xeno_canto,
    fields=(
        CatalogField("species", "Species", placeholder="e.g. Turdus merula"),
        CatalogField("country", "Country", placeholder="e.g. Germany"),
        CatalogField("quality", "Quality", placeholder="A–E (A is best)"),
        CatalogField("max_results", "Max results", kind="int", default=30),
    ),
)

MACAULAY_CONFIG = CatalogConfig(
    source=SOURCE_MACAULAY,
    title="Macaulay Library",
    icon="🏛️",
    blurb="Search the Macaulay Library (Cornell Lab) for reference recordings.",
    search_fn=search_macaulay,
    fields=(
        CatalogField("common_name", "Common name", placeholder="e.g. American Robin"),
        CatalogField("scientific_name", "Scientific name", placeholder="e.g. Turdus migratorius"),
        CatalogField("region", "Region code", placeholder="e.g. US-NY"),
        CatalogField("min_rating", "Min rating", kind="int", default=0),
        CatalogField("max_results", "Max results", kind="int", default=30),
    ),
)

INATURALIST_CONFIG = CatalogConfig(
    source=SOURCE_INATURALIST,
    title="iNaturalist",
    icon="🦋",
    blurb="Search iNaturalist observations that have sound and download them.",
    search_fn=search_inaturalist,
    fields=(
        CatalogField("taxon_name", "Taxon name", placeholder="e.g. Aves, or a species"),
        CatalogField("place_id", "Place ID", kind="int", default=0, omit_if_zero=True),
        CatalogField("max_results", "Max results", kind="int", default=30),
    ),
)

EBIRD_CONFIG = CatalogConfig(
    source=SOURCE_EBIRD,
    title="eBird",
    icon="🐦",
    blurb="Recent eBird observations for a region (sightings, not audio). "
    "Needs an eBird API key (EBIRD_API_KEY or the field below).",
    search_fn=search_ebird_region,
    downloadable=False,
    fields=(
        CatalogField("region_code", "Region code", placeholder="e.g. US-NY"),
        CatalogField("back", "Days back", kind="int", default=14),
        CatalogField("max_results", "Max results", kind="int", default=100),
        CatalogField("api_key", "API key (optional)", placeholder="overrides EBIRD_API_KEY"),
    ),
)

CATALOG_CONFIGS = (
    XENO_CANTO_CONFIG,
    MACAULAY_CONFIG,
    INATURALIST_CONFIG,
    EBIRD_CONFIG,
)
