"""Virtual egress credentials for explicitly selected runners.

The runner receives only a virtual credential while a trusted broker outside
the runner swaps in the real vault secret and enforces per-request egress
policy. Egress enforcement provides credential non-possession; isolation
strength comes from the selected runner. See ``docs/virtual-egress.md``."""

from typing import Any as _Any

from cayu._api import resolve_export as _resolve_export
from cayu.egress._exports import EXPORTS as _EXPORTS
from cayu.egress._exports import PUBLIC_NAMES as _PUBLIC_NAMES

__all__ = _PUBLIC_NAMES


def __getattr__(name: str) -> _Any:
    return _resolve_export(name, globals(), _EXPORTS)


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_EXPORTS))
