"""Durable task contracts and worker configuration.

Public import surface; implementations retain their existing internal owners."""

from typing import Any as _Any

from cayu._api import resolve_export as _resolve_export
from cayu.tasks._exports import EXPORTS as _EXPORTS
from cayu.tasks._exports import PUBLIC_NAMES as _PUBLIC_NAMES

__all__ = _PUBLIC_NAMES


def __getattr__(name: str) -> _Any:
    return _resolve_export(name, globals(), _EXPORTS)


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_EXPORTS))
