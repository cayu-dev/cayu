"""Native PostgreSQL evidence operations retain transaction and cancellation ownership."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from functools import partial
from types import SimpleNamespace

import pytest
from tests.core.test_native_memory_evidence_ownership import (
    READS,
    TABLES,
    WRITES,
    assert_owner_imports,
    assert_persisted,
    exercise,
    seed,
)


def test_postgres_memory_evidence_imports_without_adapters_or_driver():
    assert_owner_imports("postgres")


@pytest.mark.parametrize("error", [RuntimeError, asyncio.CancelledError])
def test_postgres_memory_evidence_composes_with_native_connection(postgres_dsn, error):
    import psycopg
    from psycopg import sql

    from cayu.storage import _postgres_memory_evidence as owner
    from cayu.storage.migrations import SchemaMode
    from cayu.storage.postgres import PostgresSessionStore

    async def run():
        store = PostgresSessionStore(postgres_dsn, schema_mode=SchemaMode.CREATE)
        try:
            session_id = await seed(store)
        finally:
            await store.close()
        del store
        active = False
        ready = False
        fault = SimpleNamespace(statement=None, error=error)

        class Cursor(psycopg.AsyncCursor):
            async def execute(self, query, params=None, **kwargs):
                result = await super().execute(query, params, **kwargs)
                self.fail_after_write(query)
                return result

            async def executemany(self, query, params_seq, **kwargs):
                result = await super().executemany(query, params_seq, **kwargs)
                self.fail_after_write(query)
                return result

            def fail_after_write(self, query):
                if (
                    fault.statement
                    and isinstance(query, str)
                    and " ".join(query.split()).startswith(fault.statement)
                ):
                    assert active
                    assert (
                        self.connection.info.transaction_status
                        == psycopg.pq.TransactionStatus.INTRANS
                    )
                    raise fault.error("injected evidence write failure")

        async def ensure_ready():
            nonlocal ready
            assert not active
            ready = True

        @asynccontextmanager
        async def connect():
            nonlocal active, ready
            assert ready and not active
            ready, active = False, True
            try:
                async with await psycopg.AsyncConnection.connect(
                    postgres_dsn, cursor_factory=Cursor
                ) as connection:
                    yield connection
            finally:
                active = False

        async def closure_owners(cur, targets):
            assert active
            assert cur.connection.info.transaction_status == psycopg.pq.TransactionStatus.INTRANS
            return await PostgresSessionStore._closure_lineage_owners(None, cur, targets)

        ops = SimpleNamespace(
            **{
                name: partial(getattr(owner, name), connect, ensure_ready=ensure_ready)
                for name in READS
            },
            **{
                name: partial(
                    getattr(owner, name),
                    connect,
                    ensure_ready=ensure_ready,
                    lock_closure_lineage=PostgresSessionStore._lock_closure_lineage,
                    closure_owners=closure_owners,
                )
                for name in WRITES
            },
        )

        async def snapshot():
            assert not active
            async with await psycopg.AsyncConnection.connect(postgres_dsn) as connection:
                result = []
                for table in TABLES:
                    cur = await connection.execute(
                        sql.SQL("SELECT * FROM {} ORDER BY 1, 2").format(sql.Identifier(table))
                    )
                    result.append(tuple(await cur.fetchall()))
                return tuple(result)

        receipts, exposures = await exercise(ops, session_id, snapshot, fault)
        reopened = PostgresSessionStore(postgres_dsn)
        try:
            await assert_persisted(reopened, session_id, receipts, exposures)
        finally:
            await reopened.close()

    asyncio.run(run())
