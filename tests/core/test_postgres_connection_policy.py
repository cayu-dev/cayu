from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager

import pytest

from cayu.storage.postgres import (
    PostgresEventWatcherStore,
    PostgresKnowledgeStore,
    PostgresTaskStore,
    _configure_store_connection,
    _PostgresStoreBase,
)


class _RecordingConnection:
    def __init__(self) -> None:
        self.prepare_threshold = 5
        self.statements: list[str] = []
        self.commit_count = 0

    async def execute(self, statement: str) -> None:
        self.statements.append(statement)

    async def commit(self) -> None:
        self.commit_count += 1


class _RecordingPool:
    def __init__(self, connection: _RecordingConnection) -> None:
        self.connection_value = connection

    @asynccontextmanager
    async def connection(self):
        yield self.connection_value


def test_read_only_connection_policy_is_transaction_scoped() -> None:
    async def exercise() -> None:
        connection = _RecordingConnection()
        await _configure_store_connection(connection)

        assert connection.prepare_threshold is None
        assert connection.statements == []
        assert connection.commit_count == 0

        store = object.__new__(_PostgresStoreBase)
        store._pool = _RecordingPool(connection)
        store._read_only = True

        async with store._connection() as acquired:
            assert acquired is connection
            assert connection.statements == ["SET TRANSACTION READ ONLY"]

    asyncio.run(exercise())


@pytest.mark.parametrize(
    "store_type",
    (PostgresTaskStore, PostgresKnowledgeStore, PostgresEventWatcherStore),
)
def test_read_only_is_rejected_for_unsupported_postgres_stores(store_type: type) -> None:
    with pytest.raises(ValueError, match="only supported by PostgresSessionStore"):
        store_type("postgresql://example/cayu", read_only=True)
