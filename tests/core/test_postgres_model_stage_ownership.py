"""PostgreSQL model-stage composition owns native transactions and cancellation."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from functools import partial
from types import SimpleNamespace

import pytest
from tests.core.test_native_model_stage_ownership import (
    TABLES,
    assert_owner_imports,
    compose,
    exercise,
    preparation,
    seed,
)


@asynccontextmanager
async def postgres_composition(postgres_dsn, error):
    import psycopg
    from psycopg import sql

    from cayu.storage import _postgres_model_stage as owner
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
        enabled=False, error=error, block=False, started=asyncio.Event(), release=asyncio.Event()
    )

    class Cursor(psycopg.AsyncCursor):
        async def execute(self, query, params=None, **kwargs):
            result = await super().execute(query, params, **kwargs)
            if (
                isinstance(query, str)
                and fault.enabled
                and " ".join(query.split()).startswith("INSERT INTO cayu_session_operations")
            ):
                assert self.connection is active
                assert active.info.transaction_status == psycopg.pq.TransactionStatus.INTRANS
                if fault.block:
                    fault.started.set()
                    await fault.release.wait()
                else:
                    raise fault.error("injected after native stage write")
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
        assert active is None and closed and all(conn.closed for conn in closed)

    # Only transaction-local readers and writers are composed. No store, pool
    # or readiness implementation remains alive after seeding.
    readers = SimpleNamespace()
    for name in ("_load_labels", "_load_checkpoint"):
        setattr(readers, name, partial(getattr(PostgresSessionStore, name), readers))

    async def unexpected_events(*args, **kwargs):
        raise AssertionError("plain compaction preparation must not append failover events")

    ops = compose(
        owner,
        dict(
            connect=connect,
            ensure_ready=ensure_ready,
            load_session=partial(PostgresSessionStore._load, readers),
            load_for_update=partial(PostgresSessionStore._load_for_update, readers),
            load_checkpoint=readers._load_checkpoint,
            store_now=PostgresSessionStore._session_store_now,
            reject_steering=partial(PostgresSessionStore._reject_new_work_after_steering, readers),
            upsert_checkpoint=partial(PostgresSessionStore._upsert_checkpoint, readers),
            append_events=unexpected_events,
        ),
    )
    try:
        yield ops, session, fault, clean, snapshot
    finally:
        fault.release.set()
        clean()


def test_postgres_model_stage_imports_without_adapters_or_driver():
    assert_owner_imports("postgres")


@pytest.mark.parametrize("error", [RuntimeError, asyncio.CancelledError])
def test_postgres_model_stage_composes_and_rolls_back_failed_writes(postgres_dsn, error):
    async def run():
        async with postgres_composition(postgres_dsn, error) as parts:
            await exercise(*parts)

    asyncio.run(run())


def test_postgres_model_stage_cancellation_rolls_back_preparation(postgres_dsn):
    async def run():
        async with postgres_composition(postgres_dsn, RuntimeError) as (
            ops,
            session,
            fault,
            clean,
            snapshot,
        ):
            prepared = preparation(session)
            before = await snapshot()
            fault.enabled, fault.block = True, True
            task = asyncio.create_task(ops.prepare(prepared))
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
            fault.enabled = False
            assert await snapshot() == before
            assert not (await ops.prepare(prepared)).replayed
            clean()

    asyncio.run(run())
