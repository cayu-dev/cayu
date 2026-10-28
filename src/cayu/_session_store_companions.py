"""Stores that an application store set built beside its session store.

``open_application_stores`` builds some stores, such as the durable budget
ledger and event watcher store, in the same database as the session store and
records them here. ``CayuApp`` looks them up when the caller passes that session
store without the matching store, so an app assembled from those stores is
durable without extra wiring.

Records are keyed by object identity, not by hash or equality, so a custom
session store needs neither. A record lives only as long as its session store.
"""

from __future__ import annotations

import weakref
from typing import Generic, TypeVar

_CompanionT = TypeVar("_CompanionT")


class SessionStoreCompanions(Generic[_CompanionT]):
    """One kind of store recorded beside session stores, keyed by identity."""

    __slots__ = ("_records",)

    def __init__(self) -> None:
        # id(store) -> (weak reference to that store, its companion). The weak
        # reference both drops the record with the store and guards against a
        # reused id.
        self._records: dict[int, tuple[weakref.ref[object], _CompanionT]] = {}

    def attach(self, session_store: object, companion: _CompanionT) -> None:
        """Record ``companion`` beside ``session_store``.

        A store that cannot be weakly referenced is not recorded; an app built on
        it keeps its default.
        """

        key = id(session_store)
        try:
            reference = weakref.ref(session_store, lambda dead, key=key: self._forget(key, dead))
        except TypeError:
            return
        self._records[key] = (reference, companion)

    def get(self, session_store: object) -> _CompanionT | None:
        """Return the companion recorded beside ``session_store``, if any."""

        entry = self._records.get(id(session_store))
        if entry is None or entry[0]() is not session_store:
            return None
        return entry[1]

    def _forget(self, key: int, reference: weakref.ref[object]) -> None:
        entry = self._records.get(key)
        if entry is not None and entry[0] is reference:
            self._records.pop(key, None)


__all__ = ["SessionStoreCompanions"]
