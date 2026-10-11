"""Terminal evidence retains PostgreSQL snapshot and connection ownership."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from functools import partial
from types import SimpleNamespace

import pytest
from tests.core.test_native_terminal_evidence_ownership import (
    OPERATIONS,
    assert_export,
    assert_owner_imports,
    exercise,
    seed,
)

from cayu.runtime.evidence_spool import EvidenceSpool, IncrementalEvidenceLimits


@asynccontextmanager
async def postgres_composition(postgres_dsn, error):
    import psycopg

    from cayu.storage import _postgres_terminal_evidence as owner
    from cayu.storage.migrations import SchemaMode
    from cayu.storage.postgres import PostgresSessionStore

    store = PostgresSessionStore(postgres_dsn, schema_mode=SchemaMode.CREATE)
    try:
        evidence = await seed(store)
    finally:
        await store.close()
    del store
    fault = SimpleNamespace(
        enabled=False, error=error, block=False, started=asyncio.Event(), release=asyncio.Event()
    )
    active = None
    ready = False
    closed_connections = []

    class Cursor(psycopg.AsyncCursor):
        async def execute(self, query, params=None, **kwargs):
            result = await super().execute(query, params, **kwargs)
            if isinstance(query, str) and " ".join(query.split()).startswith(
                "SELECT session.status,"
            ):
                assert self.connection is active
                assert active.info.transaction_status == psycopg.pq.TransactionStatus.INTRANS
                if fault.block:
                    fault.started.set()
                    await fault.release.wait()
                if fault.enabled:
                    raise fault.error("injected after a native snapshot read")
            return result

    async def ensure_ready():
        nonlocal ready
        assert active is None
        ready = True

    @asynccontextmanager
    async def connect():
        nonlocal active, ready
        assert ready and active is None
        ready = False
        connection = await psycopg.AsyncConnection.connect(postgres_dsn, cursor_factory=Cursor)
        active = connection
        try:
            async with connection:
                yield connection
        finally:
            active = None
            closed_connections.append(connection)
            assert connection.closed

    readers = SimpleNamespace(_load_labels=partial(PostgresSessionStore._load_labels, None))
    load_session = partial(PostgresSessionStore._load, readers)
    ops = SimpleNamespace(
        **{
            name: partial(
                getattr(owner, name), connect, ensure_ready=ensure_ready, load_session=load_session
            )
            for name in OPERATIONS
        }
    )

    def assert_clean():
        assert active is None and not ready
        assert closed_connections and all(conn.closed for conn in closed_connections)

    yield ops, evidence, fault, assert_clean
    assert_clean()


def test_postgres_terminal_evidence_imports_without_adapters_or_driver():
    assert_owner_imports("postgres")


@pytest.mark.parametrize("error", [RuntimeError, asyncio.CancelledError])
def test_postgres_terminal_evidence_composes_and_releases_failed_snapshots(postgres_dsn, error):
    async def run():
        async with postgres_composition(postgres_dsn, error) as (ops, evidence, fault, clean):
            await exercise(ops, evidence, fault, clean)

    asyncio.run(run())


def test_postgres_terminal_export_cancellation_releases_native_connection(postgres_dsn):
    async def run():
        async with postgres_composition(postgres_dsn, RuntimeError) as (
            ops,
            evidence,
            fault,
            clean,
        ):
            fault.block = True
            with EvidenceSpool(IncrementalEvidenceLimits(batch_records=1)) as spool:
                task = asyncio.create_task(
                    ops.export_terminal_session_evidence(evidence.complete, spool=spool)
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
                assert spool.phase != "snapshot_complete"
                clean()
            fault.block = False
            await assert_export(ops, evidence)
            clean()

    asyncio.run(run())
