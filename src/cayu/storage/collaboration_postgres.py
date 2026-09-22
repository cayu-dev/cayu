"""PostgreSQL collaboration identity store with scope-ordered transactions."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager

from cayu.collaboration._ownership import _MutationOwners
from cayu.collaboration.base import CollaborationStore
from cayu.collaboration.participants import CollaborationUnavailable
from cayu.storage._collaboration_repository import _SQLRepository
from cayu.storage.postgres import _PostgresStoreBase


class PostgresCollaborationStore(_PostgresStoreBase, CollaborationStore):
    request_contract_version = 1

    _min_required_revision = 95

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._owners = _MutationOwners()

    @asynccontextmanager
    async def _transaction(self, scope: str, *, write: bool):
        if self._owners.closed and asyncio.current_task() not in self._owners.pending:
            raise CollaborationUnavailable("Collaboration store is closing.")
        await self._ensure_ready()
        async with self._connection() as connection:
            # Reads use the same scope lock: a receipt and its linkage must be
            # reconstructed from one authoritative view, not mixed commits.
            await connection.execute(
                "SELECT pg_advisory_xact_lock(hashtextextended(%s, 1761))", (scope,)
            )
            yield _SQLRepository(connection, scope, postgres=True)

    async def close(self) -> None:
        await self._owners.drain()
        await super().close()
