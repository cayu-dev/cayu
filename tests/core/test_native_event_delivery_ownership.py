"""Native delivery operations compose without retaining a session store."""

from __future__ import annotations

import asyncio
import os
import subprocess
import sys
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import pytest

import cayu
from cayu.events import Event, EventType
from cayu.sessions.event_delivery import (
    PersistedEventSideEffectClaimLost,
    PersistedEventSideEffectQuery,
    PersistedEventSideEffectStatus,
)
from cayu.sessions.records import SessionIdentity
from cayu.sessions.requests import RunRequest


def _assert_import_without_adapters(backend):
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            """
import importlib
import importlib.abc
import sys

class RejectAdapters(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname in {"cayu.storage.sqlite", "cayu.storage.postgres"}:
            raise AssertionError(f"Delivery owner imported an adapter: {fullname}")

sys.meta_path.insert(0, RejectAdapters())
owner = importlib.import_module(f"cayu.storage._{sys.argv[1]}_event_delivery")
assert callable(owner.claim_persisted_event_side_effect)
assert callable(owner.enqueue_persisted_event_side_effects)
""",
            backend,
        ],
        env={**os.environ, "PYTHONPATH": str(Path(cayu.__file__).resolve().parent.parent)},
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_sqlite_delivery_imports_without_store_adapters():
    _assert_import_without_adapters("sqlite")


def test_sqlite_delivery_composes_with_direct_connection(tmp_path):
    from cayu.storage import _sqlite_connection as sqlite_connection
    from cayu.storage import _sqlite_event_delivery as delivery
    from cayu.storage.sqlite import SQLiteSessionStore

    async def exercise():
        path = tmp_path / "delivery.sqlite"
        store = SQLiteSessionStore(path)
        event = Event(type=EventType.MODEL_COMPLETED, session_id="delivery")
        try:
            await store.create(
                RunRequest(session_id="delivery", agent_name="assistant", messages=[]),
                identity=SessionIdentity(provider_name="provider", model="model"),
            )
            await store.append_event("delivery", event)
        finally:
            await store.close()

        connection = sqlite_connection.connect(path)
        now = datetime(2026, 1, 1, tzinfo=UTC)

        def clock():
            return now

        async def execute(operation):
            result = operation(connection)
            assert not connection.in_transaction
            return result

        try:
            expected = await delivery.get_persisted_event_side_effect_delivery(
                execute, session_id="delivery", event_id=event.id
            )
            claim = await delivery.claim_first_persisted_event_side_effect(
                execute, expected, ownership_clock=clock
            )
            assert claim.event == event and claim.attempt == 1
            renewed = await delivery.renew_persisted_event_side_effect(
                execute, claim, lease_seconds=600, ownership_clock=clock
            )
            assert renewed.lease_expires_at > claim.lease_expires_at
            pending = await delivery.defer_persisted_event_side_effect(
                execute, claim, ownership_clock=clock
            )
            assert (
                pending.attempts == 0 and pending.status is PersistedEventSideEffectStatus.PENDING
            )
            with pytest.raises(PersistedEventSideEffectClaimLost):
                await delivery.mark_persisted_event_side_effect_delivered(
                    execute, claim, ownership_clock=clock
                )
            retried = await delivery.claim_persisted_event_side_effect(
                execute, session_id="delivery", event_id=event.id, ownership_clock=clock
            )
            failed = await delivery.mark_persisted_event_side_effect_failed(
                execute,
                retried,
                error="Sink unavailable",
                max_attempts=3,
                retry_delay_seconds=0,
                ownership_clock=clock,
            )
            retired = await delivery.retire_failed_first_event_delivery(
                execute, failed, ownership_clock=clock
            )
            assert retired.status is PersistedEventSideEffectStatus.DEAD_LETTERED
            assert (
                await delivery.claim_persisted_event_side_effect(execute, ownership_clock=clock)
                is None
            )
            health = await delivery.get_persisted_event_side_effect_health(
                execute, ownership_clock=clock
            )
            assert health.dead_lettered == 1 and health.claimable_total == 0
            page = await delivery.query_persisted_event_side_effect_deliveries(
                execute, PersistedEventSideEffectQuery(), ownership_clock=clock
            )
            assert [row.event_id for row in page.deliveries] == [event.id]
            assert await delivery.list_persisted_event_side_effect_deliveries(
                execute, ownership_clock=clock
            ) == [retired]
        finally:
            connection.close()

    asyncio.run(exercise())


def test_sqlite_enqueue_failure_rolls_back_event_publication(tmp_path, monkeypatch):
    from cayu.storage import _sqlite_event_delivery as delivery
    from cayu.storage.sqlite import SQLiteSessionStore

    async def exercise():
        store = SQLiteSessionStore(tmp_path / "rollback.sqlite")
        try:
            await store.create(
                RunRequest(session_id="rollback", agent_name="assistant", messages=[]),
                identity=SessionIdentity(provider_name="provider", model="model"),
            )
            event = Event(type=EventType.MODEL_COMPLETED, session_id="rollback")
            enqueue = delivery.enqueue_persisted_event_side_effects

            def fail_after_enqueue(connection, session_id, events):
                assert connection.in_transaction
                enqueue(connection, session_id, events)
                assert connection.in_transaction
                raise RuntimeError("Failure after enqueue")

            with monkeypatch.context() as patch:
                patch.setattr(delivery, "enqueue_persisted_event_side_effects", fail_after_enqueue)
                with pytest.raises(RuntimeError, match="Failure after enqueue"):
                    await store.append_event("rollback", event)
            assert await store.load_events("rollback") == []
            assert await store.list_persisted_event_side_effect_deliveries() == []
            await store.append_event("rollback", event)
            assert [
                row.event_id for row in await store.list_persisted_event_side_effect_deliveries()
            ] == [event.id]
        finally:
            await store.close()

    asyncio.run(exercise())


@pytest.fixture
def delivery_postgres_dsn(postgres_dsn):
    from tests.core.postgres_contention_support import drop_cayu_tables

    asyncio.run(drop_cayu_tables(postgres_dsn))
    try:
        yield postgres_dsn
    finally:
        asyncio.run(drop_cayu_tables(postgres_dsn))


def test_postgres_delivery_imports_without_store_adapters():
    _assert_import_without_adapters("postgres")


def test_postgres_delivery_composes_with_direct_connection(delivery_postgres_dsn):
    import psycopg

    from cayu.storage import _postgres_event_delivery as delivery
    from cayu.storage.migrations import SchemaMode
    from cayu.storage.postgres import PostgresSessionStore

    async def exercise():
        session_id = f"delivery-{uuid4()}"
        store = PostgresSessionStore(delivery_postgres_dsn, schema_mode=SchemaMode.CREATE)
        event = Event(type=EventType.MODEL_COMPLETED, session_id=session_id)
        try:
            await store.create(
                RunRequest(session_id=session_id, agent_name="assistant", messages=[]),
                identity=SessionIdentity(provider_name="provider", model="model"),
            )
            await store.append_event(session_id, event)
        finally:
            await store.close()

        callbacks = []

        @asynccontextmanager
        async def connect():
            async with await psycopg.AsyncConnection.connect(delivery_postgres_dsn) as connection:
                yield connection

        async def lock_closure_lineage(cur):
            callbacks.append("closure")
            await cur.execute("SELECT pg_advisory_xact_lock(42)")

        async def store_now(cur):
            assert cur.connection.info.transaction_status == psycopg.pq.TransactionStatus.INTRANS
            callbacks.append("clock")
            await cur.execute("SELECT clock_timestamp()")
            return (await cur.fetchone())[0]

        expected = await delivery.get_persisted_event_side_effect_delivery(
            connect, session_id=session_id, event_id=event.id
        )
        claim = await delivery.claim_first_persisted_event_side_effect(
            connect, expected, lock_closure_lineage=lock_closure_lineage
        )
        assert claim.event == event and claim.attempt == 1
        renewed = await delivery.renew_persisted_event_side_effect(
            connect, claim, lease_seconds=600
        )
        assert renewed.lease_expires_at > claim.lease_expires_at
        pending = await delivery.defer_persisted_event_side_effect(connect, claim)
        assert pending.attempts == 0 and pending.status is PersistedEventSideEffectStatus.PENDING
        with pytest.raises(PersistedEventSideEffectClaimLost):
            await delivery.mark_persisted_event_side_effect_delivered(connect, claim)
        retried = await delivery.claim_persisted_event_side_effect(
            connect,
            session_id=session_id,
            event_id=event.id,
            lock_closure_lineage=lock_closure_lineage,
        )
        failed = await delivery.mark_persisted_event_side_effect_failed(
            connect, retried, error="Sink unavailable", max_attempts=3, retry_delay_seconds=0
        )
        retired = await delivery.retire_failed_first_event_delivery(
            connect, failed, store_now=store_now
        )
        assert retired.status is PersistedEventSideEffectStatus.DEAD_LETTERED
        assert callbacks == ["closure", "closure", "clock"]
        health = await delivery.get_persisted_event_side_effect_health(connect)
        assert health.dead_lettered == 1 and health.claimable_total == 0
        page = await delivery.query_persisted_event_side_effect_deliveries(
            connect, PersistedEventSideEffectQuery()
        )
        assert [row.event_id for row in page.deliveries] == [event.id]
        assert await delivery.list_persisted_event_side_effect_deliveries(connect) == [retired]

    asyncio.run(exercise())


def test_postgres_enqueue_failure_rolls_back_event_publication(delivery_postgres_dsn, monkeypatch):
    import psycopg

    from cayu.storage import _postgres_event_delivery as delivery
    from cayu.storage.migrations import SchemaMode
    from cayu.storage.postgres import PostgresSessionStore

    async def exercise():
        session_id = f"rollback-{uuid4()}"
        store = PostgresSessionStore(delivery_postgres_dsn, schema_mode=SchemaMode.CREATE)
        try:
            await store.create(
                RunRequest(session_id=session_id, agent_name="assistant", messages=[]),
                identity=SessionIdentity(provider_name="provider", model="model"),
            )
            event = Event(type=EventType.MODEL_COMPLETED, session_id=session_id)
            enqueue = delivery.enqueue_persisted_event_side_effects

            async def fail_after_enqueue(cur, sid, events):
                assert (
                    cur.connection.info.transaction_status == psycopg.pq.TransactionStatus.INTRANS
                )
                await enqueue(cur, sid, events)
                assert (
                    cur.connection.info.transaction_status == psycopg.pq.TransactionStatus.INTRANS
                )
                raise RuntimeError("Failure after enqueue")

            with monkeypatch.context() as patch:
                patch.setattr(delivery, "enqueue_persisted_event_side_effects", fail_after_enqueue)
                with pytest.raises(RuntimeError, match="Failure after enqueue"):
                    await store.append_event(session_id, event)
            assert await store.load_events(session_id) == []
            assert await store.list_persisted_event_side_effect_deliveries() == []
            await store.append_event(session_id, event)
            assert [
                row.event_id for row in await store.list_persisted_event_side_effect_deliveries()
            ] == [event.id]
        finally:
            await store.close()

    asyncio.run(exercise())
