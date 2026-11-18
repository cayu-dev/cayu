"""Knowledge granted to work admitted before the membership policy granted any.

Cayu stores the grant each conversation and task was admitted with and never widens it on
its own. Work admitted before ``MembershipAccessPolicy(knowledge=True)`` was turned on keeps
a grant without knowledge, so its knowledge reads find nothing and its writes are refused.
Granting it knowledge is an operator's explicit decision: ``cayu memberships grant-knowledge``
lists that work, and records one grant for each admission it covers. The policy reads the
record and admits that work to the knowledge of the organizations it was admitted in, still
within the person's current memberships at every step.

Each record names one admission by its digest (``cayu.resource_access.binding_sha256``), so
it covers exactly the work listed and whatever that work starts later, which carries the
same admission. The record is the audit: who, which organizations, how many sessions and
tasks, and when.

:class:`PostgresKnowledgeGrants` keeps the records in a table of their own (``TABLE``),
next to Cayu's own tables rather than in the application's schema, so the application's
migrations never see it. ``record`` creates it the first time; until then nothing is
granted. :class:`InMemoryKnowledgeGrants` serves tests.
"""

from __future__ import annotations

import asyncio
import json
import re
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Protocol

from cayu.memberships._transactions import run_transaction

TABLE = "cayu_membership_knowledge_grants"
_DIGEST = re.compile(r"[0-9a-f]{64}")
_CREATE = f"""
CREATE TABLE IF NOT EXISTS {TABLE} (
    binding_sha256 char(64) PRIMARY KEY,
    authority text NOT NULL,
    subject text NOT NULL,
    organizations jsonb NOT NULL,
    sessions integer NOT NULL,
    tasks integer NOT NULL,
    granted_at timestamptz NOT NULL
)
"""


@dataclass(frozen=True, slots=True)
class KnowledgeGrant:
    """One operator decision: the work admitted as ``binding_sha256`` gets knowledge.

    ``organizations`` are the organizations that work was admitted in, and ``sessions`` and
    ``tasks`` how many of each the grant covered when it was recorded.
    """

    binding_sha256: str
    authority: str
    subject: str
    organizations: tuple[str, ...]
    sessions: int
    tasks: int
    granted_at: datetime

    def __post_init__(self) -> None:
        _digest(self.binding_sha256)
        if self.granted_at.utcoffset() is None:
            raise ValueError("granted_at must be an aware timestamp.")


class KnowledgeGrants(Protocol):
    """Where the policy finds the knowledge an operator granted to earlier work."""

    async def granted(self, binding_sha256: str) -> bool: ...

    async def record(self, grants: Iterable[KnowledgeGrant]) -> tuple[KnowledgeGrant, ...]: ...


def _digest(value: str) -> str:
    if type(value) is not str or _DIGEST.fullmatch(value) is None:
        raise ValueError("A binding digest is 64 lowercase hexadecimal characters.")
    return value


class InMemoryKnowledgeGrants:
    """Grants kept in memory, for tests and for trying the policy without PostgreSQL."""

    def __init__(self) -> None:
        self._grants: dict[str, KnowledgeGrant] = {}

    async def granted(self, binding_sha256: str) -> bool:
        return _digest(binding_sha256) in self._grants

    async def record(self, grants: Iterable[KnowledgeGrant]) -> tuple[KnowledgeGrant, ...]:
        """Record ``grants``; returns the ones that weren't recorded before."""

        recorded = []
        for grant in grants:
            if grant.binding_sha256 not in self._grants:
                self._grants[grant.binding_sha256] = grant
                recorded.append(grant)
        return tuple(recorded)

    async def list(self) -> tuple[KnowledgeGrant, ...]:
        return tuple(sorted(self._grants.values(), key=lambda grant: grant.granted_at))


class PostgresKnowledgeGrants:
    """Grants in the ``TABLE`` table, over ``connect`` like
    :class:`~cayu.memberships.PostgresMemberships`: each call opens a connection with
    autocommit off, commits or rolls back, and closes it. Calls run in a worker thread."""

    def __init__(self, connect: Callable[[], Any]) -> None:
        if not callable(connect):
            raise TypeError("connect must be a callable that returns a database connection.")
        self._connect = connect

    async def granted(self, binding_sha256: str) -> bool:
        digest = _digest(binding_sha256)
        return await asyncio.to_thread(self._granted, digest)

    async def record(self, grants: Iterable[KnowledgeGrant]) -> tuple[KnowledgeGrant, ...]:
        """Record ``grants`` in one transaction, creating the table the first time.

        Returns the ones that weren't recorded before; a grant already recorded keeps its
        original row.
        """

        return await asyncio.to_thread(self._record, tuple(grants))

    async def list(self) -> tuple[KnowledgeGrant, ...]:
        """Every recorded grant, oldest first."""

        return await asyncio.to_thread(self._list)

    def _granted(self, digest: str) -> bool:
        def read(cursor: Any) -> bool:
            if not _exists(cursor):
                return False
            cursor.execute(f"SELECT 1 FROM {TABLE} WHERE binding_sha256 = %s", (digest,))
            return cursor.fetchone() is not None

        return self._transaction(read)

    def _record(self, grants: tuple[KnowledgeGrant, ...]) -> tuple[KnowledgeGrant, ...]:
        def write(cursor: Any) -> tuple[KnowledgeGrant, ...]:
            cursor.execute(_CREATE)
            recorded = []
            for grant in grants:
                cursor.execute(
                    f"INSERT INTO {TABLE} (binding_sha256, authority, subject, organizations, "
                    "sessions, tasks, granted_at) VALUES (%s, %s, %s, %s::jsonb, %s, %s, %s) "
                    "ON CONFLICT (binding_sha256) DO NOTHING",
                    (
                        grant.binding_sha256,
                        grant.authority,
                        grant.subject,
                        json.dumps(list(grant.organizations)),
                        grant.sessions,
                        grant.tasks,
                        grant.granted_at,
                    ),
                )
                if cursor.rowcount > 0:
                    recorded.append(grant)
            return tuple(recorded)

        return self._transaction(write, commit=True)

    def _list(self) -> tuple[KnowledgeGrant, ...]:
        def read(cursor: Any) -> tuple[KnowledgeGrant, ...]:
            if not _exists(cursor):
                return ()
            cursor.execute(
                "SELECT binding_sha256, authority, subject, organizations, sessions, tasks, "
                f"granted_at FROM {TABLE} ORDER BY granted_at, binding_sha256"
            )
            return tuple(
                KnowledgeGrant(
                    binding_sha256=digest.strip(),
                    authority=authority,
                    subject=subject,
                    organizations=tuple(
                        json.loads(organizations)
                        if isinstance(organizations, str)
                        else organizations
                    ),
                    sessions=int(sessions),
                    tasks=int(tasks),
                    granted_at=granted_at
                    if granted_at.utcoffset() is not None
                    else granted_at.replace(tzinfo=UTC),
                )
                for (
                    digest,
                    authority,
                    subject,
                    organizations,
                    sessions,
                    tasks,
                    granted_at,
                ) in cursor.fetchall()
            )

        return self._transaction(read)

    def _transaction(self, work: Callable[[Any], Any], *, commit: bool = False) -> Any:
        return run_transaction(self._connect, work, commit=commit, owner="PostgresKnowledgeGrants")


def _exists(cursor: Any) -> bool:
    cursor.execute("SELECT to_regclass(%s) IS NOT NULL", (TABLE,))
    row = cursor.fetchone()
    return bool(row and row[0])
