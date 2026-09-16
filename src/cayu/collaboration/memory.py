"""In-process collaboration identity storage with atomic owned publication."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from copy import deepcopy
from typing import Any, cast

from cayu.collaboration._contracts import ContractValue, snapshot_input
from cayu.collaboration._ownership import _MutationOwners
from cayu.collaboration.base import CollaborationStore, Key, Table
from cayu.collaboration.participants import CollaborationUnavailable


class _MemoryRepository:
    def __init__(self, rows: dict[tuple[Table, Key], object]) -> None:
        self.rows = rows

    async def get(self, table: Table, key: Key) -> object | None:
        return deepcopy(self.rows.get((table, key)))

    async def put(self, table: Table, key: Key, value: ContractValue, *, insert: bool) -> None:
        if insert and (table, key) in self.rows:
            raise CollaborationUnavailable("Collaboration unique record already exists.")
        self.rows[table, key] = snapshot_input(value)

    async def delete(self, table: Table, key: Key) -> None:
        self.rows.pop((table, key), None)

    async def scan(self, table, *, after, limit, allowed):
        rows: list[tuple[Any, dict[str, Any]]] = []
        for (family, key), document in sorted(self.rows.items(), key=lambda item: str(item[0])):
            if family != table or key[0] <= after:
                continue
            assert isinstance(document, dict)
            document = cast("dict[str, Any]", document)
            if allowed is not None:
                ids = (
                    (key[0],)
                    if table == "participants"
                    else tuple(p["participant_id"] for p in document["participants"])
                )
                if not ids or any(identity not in allowed for identity in ids):
                    continue
            rows.append((key[0], document))
        rows.sort(key=lambda item: item[0])
        return [deepcopy(document) for _, document in rows[:limit]]


class InMemoryCollaborationStore(CollaborationStore):
    def __init__(self) -> None:
        self._lock = asyncio.Lock()
        self._scopes: dict[str, dict[tuple[Table, Key], object]] = {}
        self._owners = _MutationOwners()

    @asynccontextmanager
    async def _transaction(self, scope: str, *, write: bool):
        async with self._lock:
            rows = deepcopy(self._scopes.get(scope, {}))
            yield _MemoryRepository(rows)
            if write:
                self._scopes[scope] = rows

    async def close(self) -> None:
        await self._owners.drain()
