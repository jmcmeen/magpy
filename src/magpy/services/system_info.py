"""
System-info service -- thin wrapper over ``bioamla.system`` for the Settings view.

Part of the services seam. Splits the system queries by cost:

* :func:`version_info` and :func:`dependency_report` are cheap (metadata reads +
  ``shutil`` probes) -- safe to call synchronously on screen activation.
* :func:`device_report` triggers the **torch import** (~1.2 s the first time) to
  enumerate CUDA/MPS devices, so callers run it through a
  :class:`~magpy.workers.Worker` to keep the UI responsive.

All three return MagPy-owned dataclasses so bioamla's ``VersionData`` /
``DependencyReport`` / ``DevicesData`` do not leak past the seam.
"""

from __future__ import annotations

from dataclasses import dataclass

# bioamla import is confined to the services layer.
from bioamla.system import dependency as _dependency
from bioamla.system import util as _util


@dataclass(frozen=True)
class VersionInfo:
    bioamla_version: str
    python_version: str
    platform: str
    pytorch_version: str | None
    cuda_version: str | None


@dataclass(frozen=True)
class DependencyRow:
    name: str
    description: str
    required_for: str
    installed: bool
    version: str | None
    install_hint: str


@dataclass(frozen=True)
class DependencyReport:
    os_type: str
    all_installed: bool
    dependencies: list[DependencyRow]
    install_command: str


@dataclass(frozen=True)
class DeviceRow:
    name: str
    device_type: str
    device_id: int | str | None
    memory_gb: float | None


@dataclass(frozen=True)
class DeviceReport:
    cuda_available: bool
    mps_available: bool
    devices: list[DeviceRow]


def version_info() -> VersionInfo:
    """bioamla / Python / platform / torch versions. Cheap."""
    v = _util.get_version()
    return VersionInfo(
        bioamla_version=v.bioamla_version,
        python_version=v.python_version,
        platform=v.platform,
        pytorch_version=v.pytorch_version,
        cuda_version=v.cuda_version,
    )


def dependency_report() -> DependencyReport:
    """Native dependency probe (FFmpeg/libsndfile/PortAudio). Cheap."""
    r = _dependency.check_all()
    return DependencyReport(
        os_type=r.os_type,
        all_installed=r.all_installed,
        dependencies=[
            DependencyRow(
                name=d.name,
                description=d.description,
                required_for=d.required_for,
                installed=d.installed,
                version=d.version,
                install_hint=d.install_hint,
            )
            for d in r.dependencies
        ],
        install_command=r.install_command,
    )


def device_report() -> DeviceReport:
    """Enumerate compute devices. **Triggers the torch import** -- run threaded."""
    d = _util.get_device_info()
    return DeviceReport(
        cuda_available=d.cuda_available,
        mps_available=d.mps_available,
        devices=[
            DeviceRow(
                name=dev.name,
                device_type=dev.device_type,
                device_id=dev.device_id,
                memory_gb=dev.memory_gb,
            )
            for dev in d.devices
        ],
    )
