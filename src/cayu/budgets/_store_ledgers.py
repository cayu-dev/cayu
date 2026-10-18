"""The durable budget ledger an application store set built beside its session store.

``open_application_stores`` builds a durable ledger in the same database as the
session store and records it here. ``CayuApp`` uses that ledger when the caller
passes the session store but no ``budget_ledger``, so an app assembled from those
stores keeps its reservations across restarts without extra wiring. An explicit
``budget_ledger`` always wins, and any other session store keeps the in-memory
default.

Records are keyed by object identity, not by hash or equality, so a custom
session store needs neither. A record lives only as long as its store.
"""

from __future__ import annotations

import weakref
from typing import TYPE_CHECKING

from cayu.budgets.base import BudgetLedger, InMemoryBudgetLedger

if TYPE_CHECKING:
    from cayu.sessions.store import SessionStore

# id(store) -> (weak reference to that store, its ledger). The weak reference
# both drops the record with the store and guards against a reused id.
_LEDGERS: dict[int, tuple[weakref.ref[object], BudgetLedger]] = {}


def _forget(key: int, reference: weakref.ref[object]) -> None:
    entry = _LEDGERS.get(key)
    if entry is not None and entry[0] is reference:
        _LEDGERS.pop(key, None)


def attach_session_store_budget_ledger(session_store: object, ledger: BudgetLedger) -> None:
    """Record ``ledger`` as the durable ledger stored beside ``session_store``.

    A store that cannot be weakly referenced is not recorded; an app built on it
    without its own ``budget_ledger`` keeps the in-memory default.
    """

    if not isinstance(ledger, BudgetLedger):
        raise TypeError("ledger must be a BudgetLedger.")
    key = id(session_store)
    try:
        reference = weakref.ref(session_store, lambda dead, key=key: _forget(key, dead))
    except TypeError:
        return
    _LEDGERS[key] = (reference, ledger)


def session_store_budget_ledger(session_store: SessionStore) -> BudgetLedger | None:
    """Return the ledger recorded beside ``session_store``, if any."""

    entry = _LEDGERS.get(id(session_store))
    if entry is None or entry[0]() is not session_store:
        return None
    return entry[1]


def resolve_budget_ledger(
    supplied: BudgetLedger | None, session_store: SessionStore
) -> BudgetLedger:
    """Return ``CayuApp``'s ledger: the supplied one, the recorded one, or in-memory."""

    if supplied is not None:
        return supplied
    ledger = session_store_budget_ledger(session_store)
    return ledger if ledger is not None else InMemoryBudgetLedger()


__all__ = [
    "attach_session_store_budget_ledger",
    "resolve_budget_ledger",
    "session_store_budget_ledger",
]
