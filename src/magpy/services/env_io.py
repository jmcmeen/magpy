"""
Environment / API-key persistence -- MagPy-owned, no bioamla import.

bioamla reads its API keys (Xeno-Canto, eBird) and the Hugging Face token lazily
from environment variables, and auto-loads a ``.env`` *on import* via
``python-dotenv`` -- but it searches from its own install location
(site-packages), **not** from anywhere MagPy controls. So MagPy cannot rely on
that to find a MagPy-managed key file. Instead MagPy owns a small ``.env`` of its
own and **sets the values straight into ``os.environ``** (the keys are read
lazily at the first catalog call, so writing the process environment before then
is sufficient -- no restart needed for the keys/tokens).

This module is deliberately Qt-free and bioamla-free: it is plain file I/O over a
``KEY=VALUE`` file plus ``os.environ`` updates. Qt-aware callers (the shell,
Settings screen) decide *where* the file lives (an app-config dir) and pass the
path in via :func:`default_env_path`.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

_ENV_FILENAME = "magpy.env"


@dataclass(frozen=True)
class EnvVarSpec:
    """A known environment variable MagPy can edit, described for the UI.

    ``secret`` masks the value field; ``applies_on_restart`` flags vars consumed
    by a dependency at *its* import time (so editing them mid-session has no
    effect until MagPy is relaunched).
    """

    name: str
    label: str
    purpose: str
    secret: bool = True
    applies_on_restart: bool = False


# The environment variables worth surfacing in Settings, sourced from the bioamla
# env-var audit. Keys/tokens take effect in-session (read lazily); HF_HOME is
# read by huggingface_hub at import, so it is flagged restart-only.
KNOWN_ENV_VARS: tuple[EnvVarSpec, ...] = (
    EnvVarSpec("XC_API_KEY", "Xeno-Canto API key",
               "Required for Xeno-Canto search/download (API v3)."),
    EnvVarSpec("EBIRD_API_KEY", "eBird API key",
               "Required for eBird observation queries. Get one at ebird.org/api/keygen."),
    EnvVarSpec("HF_TOKEN", "Hugging Face token",
               "Hugging Face Hub auth: pushing models/datasets and pulling private/gated ones."),
    EnvVarSpec("HUGGING_FACE_HUB_TOKEN", "Hugging Face token (alt)",
               "Alternate name huggingface_hub honours; set whichever your tooling expects."),
    EnvVarSpec("HF_HOME", "Hugging Face cache dir",
               "Location of the local HF model/dataset cache.",
               secret=False, applies_on_restart=True),
)

_KNOWN_NAMES = tuple(spec.name for spec in KNOWN_ENV_VARS)


def default_env_path(config_base: str | Path) -> Path:
    """The MagPy ``.env`` path under an app-config base directory."""
    return Path(config_base) / _ENV_FILENAME


def read_env(path: str | Path) -> dict[str, str]:
    """Parse a ``KEY=VALUE`` file into a dict. Missing file -> empty dict.

    Tolerant: ignores blank lines, ``#`` comments, and lines without ``=``;
    strips one layer of surrounding single/double quotes from the value.
    """
    path = Path(path)
    if not path.exists():
        return {}
    values: dict[str, str] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
            value = value[1:-1]
        if key:
            values[key] = value
    return values


def write_env(path: str | Path, values: dict[str, str]) -> None:
    """Write ``values`` to a ``KEY=VALUE`` file (only non-empty values).

    Quotes values containing whitespace. Creates parent directories. Empty
    values are omitted so clearing a field removes it from the file.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["# MagPy environment / API keys. Managed via Settings.", ""]
    for key, value in values.items():
        if not value:
            continue
        if any(c.isspace() for c in value):
            value = f'"{value}"'
        lines.append(f"{key}={value}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def apply_to_environ(values: dict[str, str]) -> None:
    """Make ``os.environ`` reflect ``values`` for this process.

    Non-empty values are set; **empty values unset** the corresponding key. This
    is what makes "clear a key and Save" actually revoke the credential in-session
    (bioamla reads the vars lazily), not just remove it from the file.
    """
    for key, value in values.items():
        if value:
            os.environ[key] = value
        else:
            os.environ.pop(key, None)


def load_into_environ(path: str | Path, *, override: bool = False) -> dict[str, str]:
    """Read the MagPy ``.env`` and apply it to ``os.environ``; return what was read.

    ``override=False`` (default) means a value already exported in the real
    environment wins over the file -- matching bioamla's own ``load_dotenv``
    semantics. Called once at startup, before any catalog call.
    """
    values = read_env(path)
    for key, value in values.items():
        if value and (override or key not in os.environ):
            os.environ[key] = value
    return values
