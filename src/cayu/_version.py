from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version


def package_version() -> str:
    """Return the installed Cayu distribution version."""

    try:
        return version("cayu")
    except PackageNotFoundError:
        return "0.1.0rc1"


__version__ = package_version()
