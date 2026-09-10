"""Isolated stores for the public verified-worker conformance scenarios."""

from __future__ import annotations

import asyncio
from collections.abc import Iterator
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import pytest

from cayu import (
    BudgetLedger,
    InMemoryBudgetLedger,
    InMemorySessionStore,
    InMemoryTaskStore,
    PostgresBudgetLedger,
    PostgresSessionStore,
    PostgresTaskStore,
    SessionStore,
    SQLiteBudgetLedger,
    SQLiteSessionStore,
    SQLiteTaskStore,
    TaskStore,
)
from cayu.storage.migrations import SchemaMode


async def wait_for_verified_worker_lease_expiry(store: TaskStore, expires_at: datetime) -> None:
    """Observe expiry at the backend's actual lease authority after process loss."""
    if isinstance(store, PostgresTaskStore):
        await store._ensure_ready()
        async with asyncio.timeout(10):
            while True:
                async with store._pool.connection() as connection, connection.cursor() as cur:
                    database_now = await store._database_now(cur)
                if database_now >= expires_at:
                    return
                await asyncio.sleep(0.05)
    else:
        await asyncio.sleep(max(0, (expires_at - datetime.now(UTC)).total_seconds()) + 0.05)


@dataclass(frozen=True)
class VerifiedWorkerStoreFactory:
    backend: str
    directory: Path
    postgres_dsn: str | None = field(default=None, repr=False)

    def budget_ledger(self) -> BudgetLedger:
        if self.backend == "memory":
            return InMemoryBudgetLedger()
        if self.backend == "sqlite":
            return SQLiteBudgetLedger(self.directory / "budget.sqlite")
        if self.backend != "postgres" or self.postgres_dsn is None:
            raise ValueError("Unsupported verified-worker test backend.")
        return PostgresBudgetLedger(self.postgres_dsn, schema_mode=SchemaMode.CREATE)

    def __call__(self) -> tuple[SessionStore, TaskStore]:
        if self.backend == "memory":
            return InMemorySessionStore(), InMemoryTaskStore()
        if self.backend == "sqlite":
            return (
                SQLiteSessionStore(self.directory / "sessions.sqlite"),
                SQLiteTaskStore(self.directory / "tasks.sqlite"),
            )
        if self.backend != "postgres" or self.postgres_dsn is None:
            raise ValueError("Unsupported verified-worker test backend.")
        return (
            PostgresSessionStore(self.postgres_dsn, schema_mode=SchemaMode.CREATE),
            PostgresTaskStore(self.postgres_dsn, schema_mode=SchemaMode.CREATE),
        )


@pytest.fixture
def verified_worker_store_factory(
    backend: str, tmp_path: Path, request: pytest.FixtureRequest
) -> Iterator[VerifiedWorkerStoreFactory]:
    """Construct in the caller's loop; callers close stores before fixture teardown."""
    if backend in {"memory", "sqlite"}:
        yield VerifiedWorkerStoreFactory(backend, tmp_path)
        return
    if backend != "postgres":
        raise ValueError("Unsupported verified-worker test backend.")

    dsn = request.getfixturevalue("verified_work_postgres_dsn")
    yield VerifiedWorkerStoreFactory(backend, tmp_path, dsn)


@pytest.fixture
def verified_work_postgres_dsn(request: pytest.FixtureRequest) -> Iterator[str]:
    """Own one database per case, including cases retaining immutable identities."""
    import psycopg
    from psycopg import sql
    from psycopg.conninfo import make_conninfo

    # Reuse the maintained server owner, but isolate each full lifecycle: some
    # outcomes deliberately retain tasks and immutable contract identities.
    server_dsn = request.getfixturevalue("_postgres_server_dsn")
    database = f"cayu_worker_test_{uuid4().hex}"
    with psycopg.connect(server_dsn, autocommit=True) as connection:
        connection.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(database)))
    try:
        dsn = make_conninfo(server_dsn, dbname=database)
        yield dsn
    finally:
        with psycopg.connect(server_dsn, autocommit=True) as connection:
            connection.execute(
                sql.SQL("DROP DATABASE {} WITH (FORCE)").format(sql.Identifier(database))
            )
