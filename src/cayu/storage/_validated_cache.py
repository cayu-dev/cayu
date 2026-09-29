"""Bounded, content-addressed copies of validated SQLite row projections.

Never key by timestamps: external writers and same-tick updates must invalidate
validation too. Cached values are private; every caller receives its own copy.
"""

from __future__ import annotations

import sqlite3
from collections import OrderedDict
from collections.abc import Callable
from copy import deepcopy
from functools import wraps
from threading import Lock
from typing import Any, ParamSpec, TypeVar

_P = ParamSpec("_P")
_T = TypeVar("_T")
_MAX_SOURCE_BYTES = 8 * 1024 * 1024
_MAX_ENTRIES = 64


def _key(value: Any) -> tuple[Any, int]:
    if isinstance(value, sqlite3.Row):
        return _key((tuple(value.keys()), tuple(value)))
    if isinstance(value, dict):
        return _key(tuple(sorted(value.items())))
    if isinstance(value, tuple):
        parts = tuple(_key(item) for item in value)
        return tuple(part[0] for part in parts), 64 + sum(part[1] for part in parts)
    if isinstance(value, str):
        return (str, value), 64 + 4 * len(value)
    if isinstance(value, bytes):
        return (bytes, value), 64 + len(value)
    return (type(value), value), 64


def validated_row_cache(function: Callable[_P, _T]) -> Callable[_P, _T]:
    """Cache successful validation, bounded by entry count and source size.

    Only use for context-independent projections of immutable SQL scalar values.
    Failed validation is never cached. Copies and validation run outside the
    cache lock; duplicate concurrent misses are harmless.
    """

    cache: OrderedDict[Any, tuple[_T, int]] = OrderedDict()
    lock = Lock()
    source_bytes = 0

    @wraps(function)
    def wrapped(*args: _P.args, **kwargs: _P.kwargs) -> _T:
        nonlocal source_bytes
        key, size = _key((args, kwargs))
        if size > _MAX_SOURCE_BYTES:
            return function(*args, **kwargs)
        with lock:
            cached = cache.get(key)
            if cached is not None:
                cache.move_to_end(key)
        if cached is not None:
            return deepcopy(cached[0])
        result = function(*args, **kwargs)
        private = deepcopy(result)
        with lock:
            previous = cache.pop(key, None)
            if previous is not None:
                source_bytes -= previous[1]
            cache[key] = (private, size)
            source_bytes += size
            while source_bytes > _MAX_SOURCE_BYTES or len(cache) > _MAX_ENTRIES:
                _, (_, evicted_size) = cache.popitem(last=False)
                source_bytes -= evicted_size
        return result

    return wrapped
