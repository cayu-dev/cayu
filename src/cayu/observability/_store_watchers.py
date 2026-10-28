"""The durable event watcher store an application store set built beside its session store.

``open_application_stores`` builds a durable watcher store in the same database
as the session store and records it here. ``CayuApp`` uses that store when the
caller passes the session store but no ``event_watcher_store``, so watcher
cursors survive restarts and a restarted app does not deliver old events again.
An explicit ``event_watcher_store`` always wins, and any other session store
keeps the in-memory default.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from cayu._session_store_companions import SessionStoreCompanions
from cayu.observability.watchers import EventWatcherStore, InMemoryEventWatcherStore

if TYPE_CHECKING:
    from cayu.sessions.store import SessionStore

_WATCHER_STORES: SessionStoreCompanions[EventWatcherStore] = SessionStoreCompanions()


def attach_session_store_event_watcher_store(
    session_store: object, watcher_store: EventWatcherStore
) -> None:
    """Record ``watcher_store`` as the durable watcher store beside ``session_store``.

    A store that cannot be weakly referenced is not recorded; an app built on it
    without its own ``event_watcher_store`` keeps the in-memory default.
    """

    if not isinstance(watcher_store, EventWatcherStore):
        raise TypeError("watcher_store must be an EventWatcherStore.")
    _WATCHER_STORES.attach(session_store, watcher_store)


def session_store_event_watcher_store(session_store: SessionStore) -> EventWatcherStore | None:
    """Return the watcher store recorded beside ``session_store``, if any."""

    return _WATCHER_STORES.get(session_store)


def resolve_event_watcher_store(
    supplied: EventWatcherStore | None, session_store: SessionStore
) -> EventWatcherStore:
    """Return ``CayuApp``'s watcher store: the supplied one, the recorded one, or in-memory."""

    if supplied is not None:
        return supplied
    watcher_store = session_store_event_watcher_store(session_store)
    return watcher_store if watcher_store is not None else InMemoryEventWatcherStore()


__all__ = [
    "attach_session_store_event_watcher_store",
    "resolve_event_watcher_store",
    "session_store_event_watcher_store",
]
