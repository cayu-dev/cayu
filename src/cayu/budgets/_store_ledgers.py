"""The durable budget ledger an application store set built beside its session store.

``open_application_stores`` builds a durable ledger in the same database as the
session store and records it here. ``CayuApp`` uses that ledger when the caller
passes the session store but no ``budget_ledger``, so an app assembled from those
stores keeps its reservations across restarts without extra wiring. An explicit
``budget_ledger`` always wins, and any other session store keeps the in-memory
default.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from cayu._session_store_companions import SessionStoreCompanions
from cayu.budgets.base import BudgetLedger, InMemoryBudgetLedger

if TYPE_CHECKING:
    from cayu.sessions.store import SessionStore

_LEDGERS: SessionStoreCompanions[BudgetLedger] = SessionStoreCompanions()


def attach_session_store_budget_ledger(session_store: object, ledger: BudgetLedger) -> None:
    """Record ``ledger`` as the durable ledger stored beside ``session_store``.

    A store that cannot be weakly referenced is not recorded; an app built on it
    without its own ``budget_ledger`` keeps the in-memory default.
    """

    if not isinstance(ledger, BudgetLedger):
        raise TypeError("ledger must be a BudgetLedger.")
    _LEDGERS.attach(session_store, ledger)


def session_store_budget_ledger(session_store: SessionStore) -> BudgetLedger | None:
    """Return the ledger recorded beside ``session_store``, if any."""

    return _LEDGERS.get(session_store)


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
