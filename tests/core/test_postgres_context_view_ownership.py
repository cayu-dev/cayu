"""PostgreSQL context-view operations own admission and their native transactions."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from functools import partial
from types import SimpleNamespace

from tests.core.test_context_views import _manifest_for_store
from tests.core.test_native_context_view_ownership import (
    TABLES,
    assert_owner_imports,
    compose,
    exercise,
)


def test_postgres_context_view_owner_imports_without_adapters_or_driver():
    assert_owner_imports("postgres")


def test_postgres_context_view_owner_composes_with_direct_connection(postgres_dsn):
    import psycopg
    from psycopg import sql
    from psycopg.errors import UniqueViolation
    from psycopg.types.json import Jsonb

    from cayu.storage import _postgres_context_views as owner
    from cayu.storage.migrations import SchemaMode
    from cayu.storage.postgres import PostgresSessionStore

    async def run():
        store = PostgresSessionStore(postgres_dsn, schema_mode=SchemaMode.CREATE)
        try:
            manifest = await _manifest_for_store(store)
        finally:
            await store.close()
        del store
        active = False
        ready = False
        fault = SimpleNamespace(table=None)

        class Cursor(psycopg.AsyncCursor):
            async def execute(self, query, params=None, **kwargs):
                result = await super().execute(query, params, **kwargs)
                if (
                    fault.table
                    and isinstance(query, str)
                    and " ".join(query.split()).startswith(f"INSERT INTO {fault.table} ")
                ):
                    assert active
                    assert (
                        self.connection.info.transaction_status
                        == psycopg.pq.TransactionStatus.INTRANS
                    )
                    raise RuntimeError("injected view write failure")
                return result

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

        # The existing transaction-local session readers require only label
        # decoding. Compose those capabilities without retaining a store.
        readers = SimpleNamespace()
        readers._load_labels = partial(PostgresSessionStore._load_labels, readers)

        def now():
            assert active
            return datetime(2026, 1, 1, tzinfo=UTC)

        ops = compose(
            owner,
            connect,
            {
                "ensure_ready": ensure_ready,
                "store_now": now,
                "load_session": partial(PostgresSessionStore._load, readers),
                "load_for_update": partial(PostgresSessionStore._load_for_update, readers),
                "jsonb": Jsonb,
                "unique_violation": UniqueViolation,
            },
        )

        async def snapshot():
            assert not active
            async with await psycopg.AsyncConnection.connect(postgres_dsn) as connection:
                values = []
                for table in TABLES:
                    cur = await connection.execute(
                        sql.SQL("SELECT * FROM {} ORDER BY 1").format(sql.Identifier(table))
                    )
                    values.append(tuple(await cur.fetchall()))
                return tuple(values)

        await exercise(ops, manifest, snapshot, fault)

    asyncio.run(run())
