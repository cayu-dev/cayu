"""PostgreSQL creation owners preserve native rollback and cancellation."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from functools import partial
from types import SimpleNamespace

import pytest
from tests.core.test_native_session_creation_ownership import (
    TABLES,
    assert_owner_imports,
    assert_persisted,
    compose,
    exercise,
    seed,
)


def test_postgres_creation_owner_imports_without_adapters_or_driver():
    assert_owner_imports("postgres")


@pytest.mark.parametrize("error", [RuntimeError, asyncio.CancelledError])
def test_postgres_creation_owner_composes_with_direct_connection(postgres_dsn, error):
    import psycopg
    from psycopg import sql
    from psycopg.errors import ForeignKeyViolation, UniqueViolation
    from psycopg.types.json import Jsonb

    from cayu.storage import _postgres_session_creation as owner
    from cayu.storage.migrations import SchemaMode
    from cayu.storage.postgres import PostgresSessionStore, _checkpoint_row_values

    async def run():
        store = PostgresSessionStore(postgres_dsn, schema_mode=SchemaMode.CREATE)
        try:
            source = await seed(store)
        finally:
            await store.close()
        del store
        active = False
        ready = False
        fault = SimpleNamespace(table=None, error=error)

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
                    fault.table
                    and isinstance(query, str)
                    and " ".join(query.split()).startswith(f"INSERT INTO {fault.table} ")
                ):
                    assert active
                    assert (
                        self.connection.info.transaction_status
                        == psycopg.pq.TransactionStatus.INTRANS
                    )
                    raise fault.error("injected creation write failure")

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

        # Compose the existing transaction-local readers and authority writers
        # without retaining a store, pool, or readiness implementation.
        readers = SimpleNamespace(public_authority_alias_codec=None)
        for method in ("_load_labels", "_closure_lineage_owners", "_register_public_authorities"):
            setattr(readers, method, partial(getattr(PostgresSessionStore, method), readers))

        async def now(cur):
            assert (
                active
                and cur.connection.info.transaction_status == psycopg.pq.TransactionStatus.INTRANS
            )
            return await PostgresSessionStore._session_store_now(cur)

        capabilities = {
            "ensure_ready": ensure_ready,
            "store_now": now,
            "lock_closure_lineage": PostgresSessionStore._lock_closure_lineage,
            "closure_owners": readers._closure_lineage_owners,
            "register_authorities": readers._register_public_authorities,
            "supports_model_failover": lambda: True,
            "checkpoint_values": _checkpoint_row_values,
            "jsonb": Jsonb,
            "unique_violation": UniqueViolation,
            "foreign_key_violation": ForeignKeyViolation,
        }
        for name, method in {
            "require_available_closure_identity": "_require_available_closure_identity",
            "require_external_creation": "_require_external_creation",
            "load_checkpoint": "_load_checkpoint",
            "upsert_checkpoint": "_upsert_checkpoint",
            "load_session": "_load",
            "load_for_update": "_load_for_update",
            "load_for_key_share": "_load_for_key_share",
            "register_event_authorities": "_register_event_public_authorities",
        }.items():
            capabilities[name] = partial(getattr(PostgresSessionStore, method), readers)
        ops = compose(owner, connect, capabilities)

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

        await exercise(ops, source, snapshot, fault)
        reopened = PostgresSessionStore(postgres_dsn)
        try:
            await assert_persisted(reopened, source)
        finally:
            await reopened.close()

    asyncio.run(run())
