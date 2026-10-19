"""Reuse JSON admission within one nested validation, never owner decisions.

Snapshots remain ordinary dicts/lists, including when owner validators see them.
An owner may mutate any input: reuse checks and detaches the complete graph in
one traversal, falling back to normal bounded admission on any changed identity.
"""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any

_MAX_CONTAINERS = 2048
_MAX_CHILDREN = 16_384


class SnapshotChanged(Exception):
    pass


class ValidationSnapshots:
    def __init__(self) -> None:
        self.active = True
        self._records: dict[int, tuple[Any, tuple[Any, ...]]] = {}
        self._children = 0

    def remember(self, value: Any) -> bool:
        """Retain exact child references only for an already admitted JSON tree."""
        if type(value) not in (dict, list):
            return True
        if id(value) in self._records:
            return True
        if len(self._records) >= _MAX_CONTAINERS:
            return False
        if self._children + len(value) > _MAX_CHILDREN:
            return False
        entries = tuple(value.items()) if type(value) is dict else tuple(value)
        # Retain parents only after accounting for every descendant container;
        # an uncounted subtree must not remain alive through a parent reference.
        for entry in entries:
            if not self.remember(entry[1] if type(value) is dict else entry):
                return False
        if len(self._records) >= _MAX_CONTAINERS or self._children + len(entries) > _MAX_CHILDREN:
            return False
        self._records[id(value)] = (value, entries)
        self._children += len(entries)
        return True

    def copy(self, value: Any) -> Any:
        """Copy immutable leaves only after matching their parent's saved identity."""
        if type(value) not in (dict, list):
            return value
        record = self._records.get(id(value))
        if record is None or record[0] is not value:
            raise SnapshotChanged
        expected = record[1]
        items = tuple(value.items()) if type(value) is dict else tuple(value)
        if len(items) != len(expected):
            raise SnapshotChanged
        if type(value) is dict:
            copied = {}
            for (key, item), (old_key, old_item) in zip(items, expected, strict=True):
                if key is not old_key or item is not old_item:
                    raise SnapshotChanged
                copied[key] = self.copy(item)
            return copied
        copied_list = []
        for item, old_item in zip(items, expected, strict=True):
            if item is not old_item:
                raise SnapshotChanged
            copied_list.append(self.copy(item))
        return copied_list

    def clear(self) -> None:
        self.active = False
        self._records.clear()
        self._children = 0


_CURRENT: ContextVar[ValidationSnapshots | None] = ContextVar(
    "contract_validation_snapshots", default=None
)


@contextmanager
def validation_snapshot_scope():
    current = _CURRENT.get()
    if current is not None and current.active:
        yield current
        return
    snapshots = ValidationSnapshots()
    token = _CURRENT.set(snapshots)
    try:
        yield snapshots
    finally:
        snapshots.clear()
        _CURRENT.reset(token)


def current_validation_snapshots() -> ValidationSnapshots | None:
    current = _CURRENT.get()
    return current if current is not None and current.active else None
