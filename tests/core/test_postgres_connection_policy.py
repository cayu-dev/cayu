from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager

import pytest

from cayu.storage._diagnostic_inspection import diagnostic_store_inspection
from cayu.storage.evals_postgres import PostgresEvalStore
from cayu.storage.postgres import (
    PostgresAgentWorkContextStore,
    PostgresBudgetLedger,
    PostgresEventWatcherStore,
    PostgresKnowledgeStore,
    PostgresSessionStore,
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


def test_explicit_read_only_postgres_session_declares_read_only_durability() -> None:
    store = PostgresSessionStore("postgresql://example/cayu", read_only=True)

    assert store._read_only is True
    assert store.service_durability.value == "read_only"

    asyncio.run(store.close())


@pytest.mark.parametrize(
    "store_type",
    (
        PostgresAgentWorkContextStore,
        PostgresBudgetLedger,
        PostgresEvalStore,
        PostgresEventWatcherStore,
        PostgresKnowledgeStore,
        PostgresTaskStore,
    ),
)
def test_diagnostic_inspection_forces_builtin_postgres_stores_read_only(
    store_type: type,
) -> None:
    with diagnostic_store_inspection() as inspection:
        store = store_type("postgresql://example/cayu")
        assert store._read_only is True
        if isinstance(store, (PostgresSessionStore, PostgresTaskStore)):
            assert store.service_durability.value == "durable"
        assert store._schema_mode.value == "validate"
        assert store._owns_pool is True
        inspection.verify()

    async def close() -> None:
        await store.close()

    asyncio.run(close())


def test_diagnostic_inspection_rejects_a_caller_owned_postgres_pool() -> None:
    from psycopg_pool import AsyncConnectionPool

    pool = AsyncConnectionPool("", open=False)
    with (
        diagnostic_store_inspection(),
        pytest.raises(ValueError, match="store-owned Postgres connection pool"),
    ):
        PostgresSessionStore(pool=pool)
