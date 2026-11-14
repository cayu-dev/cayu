"""Postgres migrations wait for live writers instead of deadlocking with them.

Revisions 152 and 157 used to lock ``cayu_events`` weakly, copy retained rows,
and then ask for ACCESS EXCLUSIVE to replace triggers. An event writer that had
read ``cayu_events`` before the revision started and inserted after the weak
lock deadlocked with that upgrade (cayu-tech/cayu#4033). The revisions now take
ACCESS EXCLUSIVE first, so such a writer finishes and the revision then runs.
The migration runner also bounds each lock wait with ``lock_timeout`` and runs a
transaction again after a deadlock or a lock timeout, a bounded number of times.
"""

from __future__ import annotations

import asyncio
import functools
import logging
from datetime import UTC, datetime
from uuid import uuid4

import pytest
from psycopg.errors import DeadlockDetected, LockNotAvailable, UniqueViolation

from cayu import Event, EventType, RunRequest
from cayu.sessions.records import SessionIdentity
from cayu.storage import _postgres_base
from cayu.storage._knowledge_lookup_schema import KNOWLEDGE_LOOKUP_TOTALS_REVISION
from cayu.storage._postgres_migration_locks import MigrationLockRetry
from cayu.storage._usage_projection_schema import USAGE_PROJECTION_REVISION
from cayu.storage.migrations import LATEST_REVISION, SchemaMode
from cayu.storage.targets import (
    DATABASE_MIGRATION_LOCK_TIMEOUT_ENV,
    DEFAULT_DATABASE_MIGRATION_LOCK_TIMEOUT_SECONDS,
    SessionStoreTargetError,
    configured_database_migration_lock_timeout,
)

pytestmark = pytest.mark.anyio

_SYNC_DEADLINE_SECONDS = 30


class _RecordedSleep:
    """Stand-in for ``asyncio.sleep`` that records delays and can hold a retry."""

    def __init__(self) -> None:
        self.delays: list[float] = []
        self.entered = asyncio.Event()
        self.release = asyncio.Event()
        self.release.set()

    async def __call__(self, delay: float) -> None:
        self.delays.append(delay)
        self.entered.set()
        await self.release.wait()


async def _run(retry: MigrationLockRetry, failures: list[BaseException]) -> int:
    """Run transactions under ``retry`` until one does not fail; count the runs."""

    runs = 0
    while True:
        async with retry.transaction():
            retry.step = "revision 999"
            runs += 1
            if failures:
                raise failures.pop(0)
        if not retry.retrying:
            return runs


@pytest.mark.parametrize(
    ("error", "reason"),
    [
        (DeadlockDetected("deadlock detected"), "deadlocked"),
        (LockNotAvailable("canceling statement due to lock timeout"), "timed out on a lock"),
    ],
)
async def test_retry_runs_a_transaction_again_after_a_deadlock_or_lock_timeout(
    error, reason, caplog
) -> None:
    sleep = _RecordedSleep()
    retry = MigrationLockRetry(sleep=sleep)

    with caplog.at_level(logging.WARNING, logger="cayu.storage.postgres"):
        assert await _run(retry, [error, type(error)("again")]) == 3

    assert sleep.delays == [1.0, 2.0]
    messages = [record.getMessage() for record in caplog.records]
    assert len(messages) == 2
    assert f"revision 999 {reason} (SQLSTATE {error.sqlstate})" in messages[0]
    assert "attempt 2 of 5" in messages[0]
    assert "attempt 3 of 5" in messages[1]


async def test_retry_does_not_retry_other_errors() -> None:
    sleep = _RecordedSleep()
    retry = MigrationLockRetry(sleep=sleep)

    with pytest.raises(UniqueViolation):
        await _run(retry, [UniqueViolation("duplicate key")])
    with pytest.raises(RuntimeError, match="not a lock failure"):
        await _run(retry, [RuntimeError("not a lock failure")])

    assert sleep.delays == []


async def test_retry_gives_up_after_its_attempts_with_guidance() -> None:
    sleep = _RecordedSleep()
    retry = MigrationLockRetry(attempts=3, sleep=sleep)

    with pytest.raises(LockNotAvailable) as raised:
        await _run(retry, [LockNotAvailable(f"timeout {n}") for n in range(5)])

    assert sleep.delays == [1.0, 2.0]
    assert str(raised.value) == "timeout 2"
    (note,) = raised.value.__notes__
    assert "revision 999 timed out on a lock; gave up after 3 attempts" in note
    assert "Stop the application's writers" in note
    assert DATABASE_MIGRATION_LOCK_TIMEOUT_ENV in note


async def test_retry_gives_each_committed_transaction_a_fresh_budget() -> None:
    sleep = _RecordedSleep()
    retry = MigrationLockRetry(attempts=2, sleep=sleep)

    assert await _run(retry, [DeadlockDetected("first")]) == 2
    assert await _run(retry, [DeadlockDetected("second")]) == 2
    assert sleep.delays == [1.0, 1.0]


def test_migration_lock_timeout_reads_whole_seconds_from_the_environment() -> None:
    env = DATABASE_MIGRATION_LOCK_TIMEOUT_ENV
    assert configured_database_migration_lock_timeout({}) == (
        DEFAULT_DATABASE_MIGRATION_LOCK_TIMEOUT_SECONDS
    )
    assert configured_database_migration_lock_timeout({env: " 5 "}) == 5
    assert configured_database_migration_lock_timeout({env: "0"}) == 0
    for value in ("", "-1", "1.5", "5s", "٣"):
        with pytest.raises(SessionStoreTargetError, match=env):
            configured_database_migration_lock_timeout({env: value})


# Real Postgres: a live writer and a migration racing for cayu_events.


def _event(session_id: str) -> Event:
    return Event(
        id=f"live-writer-{uuid4().hex}",
        type=EventType.MODEL_COMPLETED,
        session_id=session_id,
        timestamp=datetime(2026, 3, 1, tzinfo=UTC),
        payload={
            "usage_metrics": {
                "provider_name": "openai",
                "model": "gpt-x",
                "input_tokens": 7,
                "output_tokens": 3,
                "total_tokens": 10,
            }
        },
    )


async def _open(dsn: str, schema_mode: SchemaMode):
    from cayu.storage.postgres import PostgresSessionStore

    store = PostgresSessionStore(dsn, schema_mode=schema_mode)
    try:
        await store.ensure_schema()
    except BaseException:
        await store.close()
        raise
    return store


async def _rewind(dsn: str, before_revision: int) -> None:
    """Forget revisions from ``before_revision`` on; their objects stay and rerun cleanly."""

    import psycopg

    async with await psycopg.AsyncConnection.connect(dsn, autocommit=True) as conn:
        await conn.execute(
            "DELETE FROM cayu_schema_migrations WHERE revision >= %s", (before_revision,)
        )


async def _recorded_revision(dsn: str) -> int:
    import psycopg

    async with await psycopg.AsyncConnection.connect(dsn, autocommit=True) as conn:
        cursor = await conn.execute("SELECT max(revision) FROM cayu_schema_migrations")
        row = await cursor.fetchone()
        assert row is not None
        return int(row[0])


async def _waiting_event_lock(dsn: str, migration: asyncio.Task) -> str:
    """Wait until a backend waits for a lock on cayu_events and return its mode."""

    import psycopg

    loop = asyncio.get_running_loop()
    deadline = loop.time() + _SYNC_DEADLINE_SECONDS
    async with await psycopg.AsyncConnection.connect(dsn, autocommit=True) as conn:
        while True:
            cursor = await conn.execute(
                "SELECT mode FROM pg_locks WHERE relation = 'cayu_events'::regclass AND NOT granted"
            )
            row = await cursor.fetchone()
            if row is not None:
                return str(row[0])
            if migration.done():
                migration.result()
                raise AssertionError("The migration finished without waiting for the writer.")
            assert loop.time() < deadline, "The migration never waited for the writer."
            await asyncio.sleep(0.05)


@pytest.mark.parametrize(
    "revision",
    [KNOWLEDGE_LOOKUP_TOTALS_REVISION, USAGE_PROJECTION_REVISION],
    ids=["revision-152", "revision-157"],
)
async def test_revision_waits_for_a_writer_that_read_events_before_it_started(
    postgres_dsn, revision, caplog
) -> None:
    writer = await _open(postgres_dsn, SchemaMode.CREATE)
    try:
        session = await writer.create(
            RunRequest(agent_name="assistant", messages=[]),
            identity=SessionIdentity(provider_name="openai", model="gpt-x"),
        )
        await _rewind(postgres_dsn, revision)
        with caplog.at_level(logging.WARNING, logger="cayu.storage.postgres"):
            await _race_live_writer(postgres_dsn, writer, session.id)
        assert not [
            record for record in caplog.records if record.name == "cayu.storage.postgres"
        ], "the revision deadlocked or timed out and was retried"
        assert await _recorded_revision(postgres_dsn) == LATEST_REVISION
    finally:
        await writer.close()


async def _race_live_writer(dsn: str, writer, session_id: str) -> None:
    """Migrate while a writer that read cayu_events first appends an event."""

    import psycopg

    async with (
        await psycopg.AsyncConnection.connect(dsn) as live,
        live.cursor() as cur,
    ):
        # The 0.19.1 event writer read before it inserted, in one transaction,
        # so it held ACCESS SHARE on cayu_events when the migration started.
        await cur.execute(
            "SELECT 1 FROM cayu_events WHERE session_id = %s AND event_id = %s",
            (session_id, "absent"),
        )
        migration = asyncio.create_task(_open(dsn, SchemaMode.MIGRATE))
        try:
            # The revision asks for ACCESS EXCLUSIVE first and holds nothing the
            # writer needs while it waits, so the writer's insert goes through.
            assert await _waiting_event_lock(dsn, migration) == "AccessExclusiveLock"
            await asyncio.wait_for(
                writer._append_events_with_cursor(
                    cur, session_id, [_event(session_id)], expected_run_epoch=None
                ),
                timeout=_SYNC_DEADLINE_SECONDS,
            )
            assert not migration.done()
            await live.commit()
            migrated = await asyncio.wait_for(migration, timeout=120)
        finally:
            if not migration.done():
                migration.cancel()
    await migrated.close()


async def _hold_event_reads(dsn: str):
    import psycopg

    holder = await psycopg.AsyncConnection.connect(dsn)
    await holder.execute("SELECT 1 FROM cayu_events LIMIT 1")
    return holder


async def test_migration_retries_a_revision_that_timed_out_on_a_lock(
    postgres_dsn, monkeypatch, caplog
) -> None:
    monkeypatch.setenv(DATABASE_MIGRATION_LOCK_TIMEOUT_ENV, "1")
    sleep = _RecordedSleep()
    sleep.release.clear()
    monkeypatch.setattr(
        _postgres_base, "MigrationLockRetry", functools.partial(MigrationLockRetry, sleep=sleep)
    )
    store = await _open(postgres_dsn, SchemaMode.CREATE)
    await store.close()
    await _rewind(postgres_dsn, USAGE_PROJECTION_REVISION)

    holder = await _hold_event_reads(postgres_dsn)
    with caplog.at_level(logging.WARNING, logger="cayu.storage.postgres"):
        migration = asyncio.create_task(_open(postgres_dsn, SchemaMode.MIGRATE))
        try:
            # The first attempt gives up after one second behind the reader.
            await asyncio.wait_for(sleep.entered.wait(), timeout=_SYNC_DEADLINE_SECONDS)
            assert await _recorded_revision(postgres_dsn) == USAGE_PROJECTION_REVISION - 1
            await holder.commit()
            sleep.release.set()
            migrated = await asyncio.wait_for(migration, timeout=120)
        finally:
            if not migration.done():
                migration.cancel()
            await holder.close()
    await migrated.close()

    assert sleep.delays == [1.0]
    (record,) = [r for r in caplog.records if r.name == "cayu.storage.postgres"]
    assert f"revision {USAGE_PROJECTION_REVISION} timed out on a lock" in record.getMessage()
    assert await _recorded_revision(postgres_dsn) == LATEST_REVISION


async def test_migration_stops_after_bounded_lock_timeouts(postgres_dsn, monkeypatch) -> None:
    monkeypatch.setenv(DATABASE_MIGRATION_LOCK_TIMEOUT_ENV, "1")
    sleep = _RecordedSleep()
    monkeypatch.setattr(
        _postgres_base,
        "MigrationLockRetry",
        functools.partial(MigrationLockRetry, attempts=2, sleep=sleep),
    )
    store = await _open(postgres_dsn, SchemaMode.CREATE)
    await store.close()
    await _rewind(postgres_dsn, USAGE_PROJECTION_REVISION)

    holder = await _hold_event_reads(postgres_dsn)
    try:
        with pytest.raises(LockNotAvailable) as raised:
            await asyncio.wait_for(_open(postgres_dsn, SchemaMode.MIGRATE), timeout=120)
        assert sleep.delays == [1.0]
        assert "gave up after 2 attempts" in raised.value.__notes__[0]
        assert await _recorded_revision(postgres_dsn) == USAGE_PROJECTION_REVISION - 1
    finally:
        await holder.close()
    migrated = await _open(postgres_dsn, SchemaMode.MIGRATE)
    await migrated.close()
    assert await _recorded_revision(postgres_dsn) == LATEST_REVISION
