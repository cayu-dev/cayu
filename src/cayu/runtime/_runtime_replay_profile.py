"""Private identity bridge for runtime-contract replay substitution adapters."""

from __future__ import annotations

import threading
import weakref
from typing import Any

_LOCK = threading.RLock()
_SOURCES: dict[int, tuple[weakref.ReferenceType[Any], object]] = {}


def bind_runtime_replay_profile_source(wrapper: object, source: object) -> None:
    """Bind one live replay-only adapter to the component whose behavior it preserves."""

    wrapper_id = id(wrapper)

    def discard(reference: weakref.ReferenceType[Any]) -> None:
        with _LOCK:
            current = _SOURCES.get(wrapper_id)
            if current is not None and current[0] is reference:
                _SOURCES.pop(wrapper_id, None)

    try:
        reference = weakref.ref(wrapper, discard)
    except TypeError as exc:  # pragma: no cover - framework adapters support weak references.
        raise TypeError("Runtime replay profile wrappers must support weak references.") from exc
    with _LOCK:
        _SOURCES[wrapper_id] = (reference, source)


def runtime_replay_profile_source(value: object) -> object:
    """Return the attested live source for a replay adapter, otherwise ``value``."""

    with _LOCK:
        current = _SOURCES.get(id(value))
        if current is None or current[0]() is not value:
            return value
        return current[1]
