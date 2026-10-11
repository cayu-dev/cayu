"""PostgreSQL alias operations retain transaction and cancellation ownership."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from functools import partial
from types import SimpleNamespace

import pytest
from tests.core.test_native_public_authority_ownership import (
    OPERATIONS,
    TABLES,
    assert_owner_imports,
    codec,
    exercise,
    seed,
)


@asynccontextmanager
async def postgres_composition(postgres_dsn, error):
    import psycopg
    from psycopg import sql

    from cayu.storage import _postgres_base
    from cayu.storage import _postgres_public_authority as owner
    from cayu.storage.migrations import SchemaMode
    from cayu.storage.postgres import PostgresSessionStore

    store = PostgresSessionStore(postgres_dsn, schema_mode=SchemaMode.CREATE)
    try:
        session = await seed(store)
    finally:
        await store.close()
    del store
    active = None
    closed = []
    fault = SimpleNamespace(
        target=None, error=error, block=False, started=asyncio.Event(), release=asyncio.Event()
    )

    class Cursor(psycopg.AsyncCursor):
        async def execute(self, query, params=None, **kwargs):
            result = await super().execute(query, params, **kwargs)
            if (
                isinstance(query, str)
                and fault.target is not None
                and " ".join(query.split()).startswith(fault.target)
            ):
                assert self.connection is active
                assert active.info.transaction_status == psycopg.pq.TransactionStatus.INTRANS
                if fault.block:
                    fault.started.set()
                    await fault.release.wait()
                else:
                    raise fault.error("injected after a native alias write")
            return result

    @asynccontextmanager
    async def connect():
        nonlocal active
        assert active is None
        connection = await psycopg.AsyncConnection.connect(postgres_dsn, cursor_factory=Cursor)
        active = connection
        try:
            async with connection:
                yield connection
        finally:
            active = None
            closed.append(connection)
            assert connection.closed

    async def ensure_ready():
        assert active is None

    async def reconcile(selected, *, read_only=False):
        await owner.reconcile_keys(
            connect,
            codec=selected,
            read_only=read_only,
            acquire_schema_lock=_postgres_base._acquire_schema_transaction_lock,
        )

    async def snapshot():
        results = []
        async with connect() as conn, conn.cursor() as cur:
            for table in TABLES:
                await cur.execute(
                    sql.SQL("SELECT * FROM {} ORDER BY 1, 2").format(sql.Identifier(table))
                )
                results.append(await cur.fetchall())
        return results

    def clean():
        assert active is None and closed and all(connection.closed for connection in closed)

    async def require_current(selected):
        await owner.require_current_configuration(connect, codec=selected)

    ops = SimpleNamespace(
        **{
            name: partial(getattr(owner, name), connect, ensure_ready=ensure_ready)
            for name in OPERATIONS
        }
    )
    try:
        yield ops, session, fault, clean, snapshot, reconcile, require_current
    finally:
        fault.release.set()
        clean()
        # This fixture owns the module's disposable database. Remove its
        # deployment keyring so the next case can seed an unconfigured store.
        async with await psycopg.AsyncConnection.connect(postgres_dsn) as connection:
            await connection.execute(
                "TRUNCATE cayu_public_authority_aliases, cayu_public_authority_alias_config, "
                "cayu_public_authority_alias_keys"
            )


def test_postgres_public_authority_imports_without_adapters_or_driver():
    assert_owner_imports("postgres")


@pytest.mark.parametrize("error", [RuntimeError, asyncio.CancelledError])
def test_postgres_public_authority_composes_and_rolls_back_failed_writes(postgres_dsn, error):
    async def run():
        async with postgres_composition(postgres_dsn, error) as parts:
            await exercise(*parts)

    asyncio.run(run())


def test_postgres_public_authority_cancellation_rolls_back_registration(postgres_dsn):
    async def run():
        async with postgres_composition(postgres_dsn, RuntimeError) as (
            ops,
            _session,
            fault,
            clean,
            snapshot,
            reconcile,
            _require_current,
        ):
            selected = codec()
            await reconcile(selected)
            alias = selected.encode("cancelled-writer", field_name="session_id")
            before = await snapshot()
            fault.target, fault.block = "INSERT INTO cayu_public_authority_aliases", True
            task = asyncio.create_task(
                ops.register_public_authority_alias(
                    alias, field_name="session_id", private_value="cancelled-writer", codec=selected
                )
            )
            try:
                await asyncio.wait_for(fault.started.wait(), 5)
                task.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await asyncio.wait_for(task, 5)
            finally:
                fault.release.set()
                if not task.done():
                    task.cancel()
                await asyncio.gather(task, return_exceptions=True)
            clean()
            fault.target = None
            assert await snapshot() == before
            await ops.register_public_authority_alias(
                alias, field_name="session_id", private_value="cancelled-writer", codec=selected
            )
            assert (
                await ops.resolve_public_authority_alias(
                    alias, field_name="session_id", codec=selected
                )
                == "cancelled-writer"
            )
            clean()

    asyncio.run(run())
