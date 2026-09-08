from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from pathlib import Path
from uuid import uuid4

import pytest

from cayu.core import Event, EventType, Message
from cayu.runtime import CayuApp, EventQuery, InMemorySessionStore, RunRequest
from cayu.runtime.event_watchers import (
    EventWatcher,
    EventWatcherClaim,
    EventWatcherDelivery,
    EventWatcherDeliveryStatus,
    EventWatcherLeaseLost,
    EventWatcherStore,
    InMemoryEventWatcherStore,
)
from cayu.runtime.sessions import EventRecord, SessionIdentity
from cayu.storage import SQLiteEventWatcherStore
from cayu.storage.migrations import SchemaMode


class CustomStore(EventWatcherStore):
    """A public-contract adapter: Runtime must not depend on concrete store internals."""

    def __init__(self):
        self.delegate = InMemoryEventWatcherStore()

    async def load_state(self, watcher_name):
        return await self.delegate.load_state(watcher_name)

    async def claim_event(self, **kwargs):
        return await self.delegate.claim_event(**kwargs)

    async def renew_claim(self, claim, *, lease_seconds):
        return await self.delegate.renew_claim(claim, lease_seconds=lease_seconds)

    async def mark_success(self, claim):
        return await self.delegate.mark_success(claim)

    async def mark_failure(self, claim, *, error, max_attempts):
        return await self.delegate.mark_failure(claim, error=error, max_attempts=max_attempts)

    async def list_dead_letters(self, watcher_name, **kwargs):
        return await self.delegate.list_dead_letters(watcher_name, **kwargs)

    async def resolve_dead_letter(self, watcher_name, event_sequence):
        return await self.delegate.resolve_dead_letter(watcher_name, event_sequence)


@pytest.fixture(params=["memory", "sqlite", "postgres", "custom"])
def watcher_store_factory(request, tmp_path):
    if request.param == "memory":
        return lambda: InMemoryEventWatcherStore()
    if request.param == "custom":
        return CustomStore
    if request.param == "sqlite":
        return lambda: SQLiteEventWatcherStore(tmp_path / "watchers.sqlite")
    from cayu.storage import PostgresEventWatcherStore

    dsn = request.getfixturevalue("postgres_dsn")
    return lambda: PostgresEventWatcherStore(dsn, schema_mode=SchemaMode.CREATE)


async def close_store(store):
    if hasattr(store, "close"):
        await store.close()


def record(number=1):
    return EventRecord(
        sequence=number,
        event=Event(
            id=f"event-{number}", type=EventType.BUDGET_LIMIT_REACHED, session_id="watcher-session"
        ),
    )


def test_crash_only_reclaims_dead_letter_without_another_handler_attempt(watcher_store_factory):
    async def run():
        store = watcher_store_factory()
        name = f"crash-{uuid4()}"
        try:
            for attempt in (1, 2):
                claim = await store.claim_event(
                    watcher_name=name, record=record(), lease_seconds=0.06, max_attempts=2
                )
                assert isinstance(claim, EventWatcherClaim) and claim.attempt == attempt
                await asyncio.sleep(0.09)
            outcome = await store.claim_event(
                watcher_name=name, record=record(), lease_seconds=1, max_attempts=2
            )
            assert isinstance(outcome, EventWatcherDelivery)
            assert outcome.status is EventWatcherDeliveryStatus.DEAD_LETTERED
            assert outcome.attempt == 2
            assert (await store.load_state(name)).cursor_sequence == 1
            assert (await store.load_state(name)).dead_lettered_count == 1
            assert (
                await store.claim_event(
                    watcher_name=name, record=record(), lease_seconds=1, max_attempts=2
                )
                is None
            )
            assert len(await store.list_dead_letters(name)) == 1
            next_claim = await store.claim_event(
                watcher_name=name, record=record(2), lease_seconds=1, max_attempts=2
            )
            assert isinstance(next_claim, EventWatcherClaim) and next_claim.attempt == 1
            with pytest.raises(EventWatcherLeaseLost):
                await store.mark_success(claim)
        finally:
            await close_store(store)

    asyncio.run(run())


def test_renewal_fences_expired_or_replaced_claims_and_replays_exact_settlement(
    watcher_store_factory,
):
    async def run():
        store = watcher_store_factory()
        name = f"fence-{uuid4()}"
        try:
            first = await store.claim_event(
                watcher_name=name, record=record(), lease_seconds=0.06, max_attempts=3
            )
            assert isinstance(first, EventWatcherClaim)
            renewed = await store.renew_claim(first, lease_seconds=0.2)
            assert renewed.claim_id == first.claim_id
            await asyncio.sleep(0.09)
            assert (
                await store.claim_event(watcher_name=name, record=record(), lease_seconds=1) is None
            )
            await asyncio.sleep(0.15)
            with pytest.raises(EventWatcherLeaseLost):
                await store.renew_claim(first, lease_seconds=1)
            with pytest.raises(EventWatcherLeaseLost):
                await store.mark_success(first)
            second = await store.claim_event(watcher_name=name, record=record(), lease_seconds=5)
            assert isinstance(second, EventWatcherClaim) and second.claim_id != first.claim_id
            with pytest.raises(EventWatcherLeaseLost):
                await store.mark_failure(first, error="stale", max_attempts=3)
            delivered = await store.mark_success(second)
            assert await store.mark_success(second) == delivered
            third = await store.claim_event(watcher_name=name, record=record(2), lease_seconds=5)
            assert isinstance(third, EventWatcherClaim)
            await store.mark_success(third)
            assert await store.mark_success(second) == delivered
            with pytest.raises(EventWatcherLeaseLost):
                await store.mark_failure(second, error="contradictory", max_attempts=3)
        finally:
            await close_store(store)

    asyncio.run(run())


def test_failure_receipts_and_pending_event_order_are_replay_safe(watcher_store_factory):
    async def run():
        store = watcher_store_factory()
        name = f"retry-{uuid4()}"
        try:
            claim = await store.claim_event(
                watcher_name=name, record=record(), lease_seconds=5, max_attempts=2
            )
            assert isinstance(claim, EventWatcherClaim)
            failed = await store.mark_failure(claim, error="failure", max_attempts=2)
            assert await store.mark_failure(claim, error="failure", max_attempts=2) == failed
            with pytest.raises(EventWatcherLeaseLost):
                await store.claim_event(
                    watcher_name=name, record=record(2), lease_seconds=5, max_attempts=2
                )
            second = await store.claim_event(
                watcher_name=name, record=record(), lease_seconds=5, max_attempts=2
            )
            dead = await store.mark_failure(second, error="failure", max_attempts=2)
            assert dead.status is EventWatcherDeliveryStatus.DEAD_LETTERED
            assert await store.mark_failure(second, error="failure", max_attempts=2) == dead
            assert (await store.load_state(name)).dead_lettered_count == 1
            assert await store.mark_failure(claim, error="failure", max_attempts=2) == failed
            await store.resolve_dead_letter(name, 1)
            assert not await store.list_dead_letters(name)
            assert (
                await store.claim_event(watcher_name=name, record=record(), lease_seconds=5) is None
            )
        finally:
            await close_store(store)

    asyncio.run(run())


async def event_app(store):
    sessions = InMemorySessionStore()
    await sessions.create(
        RunRequest(
            agent_name="assistant",
            session_id="watcher-session",
            messages=[Message.text("user", "hello")],
        ),
        identity=SessionIdentity(provider_name="fake", model="fake"),
    )
    await sessions.append_event("watcher-session", record().event)
    return CayuApp(session_store=sessions, event_watcher_store=store, enable_logging=False)


def test_long_handler_renews_while_another_app_attempts_takeover(watcher_store_factory):
    async def run():
        store = watcher_store_factory()
        app = await event_app(store)
        competitor = CayuApp(
            session_store=app.session_store, event_watcher_store=store, enable_logging=False
        )
        started, release = asyncio.Event(), asyncio.Event()
        calls = 0

        async def handler(_context):
            nonlocal calls
            calls += 1
            started.set()
            await release.wait()

        watcher = EventWatcher(
            name=f"long-{uuid4()}", query=EventQuery(), handler=handler, lease_seconds=0.3
        )
        task = asyncio.create_task(app.run_event_watchers([watcher]))
        try:
            await asyncio.wait_for(started.wait(), timeout=10)
            await asyncio.sleep(0.7)
            result = await competitor.run_event_watchers([watcher])
            assert result[0].blocked_by_active_lease
            assert calls == 1
            release.set()
            completed = await asyncio.wait_for(task, timeout=5)
            assert completed[0].deliveries[0].status is EventWatcherDeliveryStatus.SUCCEEDED
        finally:
            release.set()
            await asyncio.gather(task, return_exceptions=True)
            await close_store(store)

    asyncio.run(run())


def test_renewal_loss_cancels_handler_and_does_not_suppress_other_watchers():
    class LosingStore(InMemoryEventWatcherStore):
        def __init__(self):
            super().__init__()
            self.renewals = 0
            self.publications = []

        async def renew_claim(self, claim, *, lease_seconds):
            if claim.watcher_name == "lost":
                self.renewals += 1
                if self.renewals > 1:
                    raise EventWatcherLeaseLost("stolen")
            return await super().renew_claim(claim, lease_seconds=lease_seconds)

        async def mark_success(self, claim):
            self.publications.append(claim.watcher_name)
            return await super().mark_success(claim)

        async def mark_failure(self, claim, **kwargs):
            self.publications.append(claim.watcher_name)
            return await super().mark_failure(claim, **kwargs)

    async def run():
        store = LosingStore()
        app = await event_app(store)
        cancelled = asyncio.Event()

        async def hanging(_ctx):
            try:
                await asyncio.Event().wait()
            finally:
                cancelled.set()

        results = await asyncio.wait_for(
            app.run_event_watchers(
                [
                    EventWatcher(
                        name="lost", query=EventQuery(), handler=hanging, lease_seconds=0.09
                    ),
                    EventWatcher(name="other", query=EventQuery(), handler=lambda _: None),
                ]
            ),
            timeout=3,
        )
        await asyncio.wait_for(cancelled.wait(), timeout=1)
        assert results[0].deliveries[0].status is EventWatcherDeliveryStatus.LEASE_LOST
        assert results[1].deliveries[0].status is EventWatcherDeliveryStatus.SUCCEEDED
        assert store.publications == ["other"]

    asyncio.run(run())


@pytest.mark.parametrize("commit_first", [False, True])
def test_publication_failure_is_bounded_and_exact_lost_ack_is_reconciled(commit_first):
    class FailingStore(InMemoryEventWatcherStore):
        def __init__(self):
            super().__init__()
            self.calls = 0

        async def mark_success(self, claim):
            if claim.watcher_name != "broken":
                return await super().mark_success(claim)
            self.calls += 1
            if commit_first:
                result = await super().mark_success(claim)
                if self.calls > 1:
                    return result
            raise ConnectionError("publication failed")

    async def run():
        store = FailingStore()
        app = await event_app(store)
        results = await app.run_event_watchers(
            [
                EventWatcher(name="broken", query=EventQuery(), handler=lambda _: None),
                EventWatcher(name="other", query=EventQuery(), handler=lambda _: None),
            ]
        )
        assert results[0].deliveries[0].status is (
            EventWatcherDeliveryStatus.SUCCEEDED
            if commit_first
            else EventWatcherDeliveryStatus.PUBLICATION_FAILED
        )
        assert results[1].deliveries[0].status is EventWatcherDeliveryStatus.SUCCEEDED
        assert store.calls == 2

    asyncio.run(run())


def test_cancelled_synchronous_handler_keeps_renewing_until_thread_stops():
    import threading

    async def run():
        store = InMemoryEventWatcherStore()
        app = await event_app(store)
        competitor = CayuApp(
            session_store=app.session_store, event_watcher_store=store, enable_logging=False
        )
        started, release = threading.Event(), threading.Event()
        calls = 0

        def handler(_context):
            nonlocal calls
            calls += 1
            started.set()
            assert release.wait(timeout=5)

        watcher = EventWatcher(
            name="sync-cancel", query=EventQuery(), handler=handler, lease_seconds=0.3
        )
        task = asyncio.create_task(app.run_event_watchers([watcher]))
        try:
            async with asyncio.timeout(3):
                while not started.is_set():
                    await asyncio.sleep(0.005)
            task.cancel("operator stopped waiting")
            with pytest.raises(asyncio.CancelledError, match="operator stopped waiting"):
                await asyncio.wait_for(task, timeout=0.2)
            await asyncio.sleep(0.7)
            result = await competitor.run_event_watchers([watcher])
            assert result[0].blocked_by_active_lease
            assert calls == 1
        finally:
            release.set()
            async with asyncio.timeout(3):
                while app._event_watcher_supervisor.active(watcher.name):
                    await asyncio.sleep(0.005)
        assert (
            await store.load_state(watcher.name)
        ).delivery_status is EventWatcherDeliveryStatus.FAILED

    asyncio.run(run())


def test_hung_publication_returns_bounded_result_and_preserves_other_watchers():
    class HangingStore(InMemoryEventWatcherStore):
        async def mark_success(self, claim):
            if claim.watcher_name == "hanging":
                await asyncio.Event().wait()
            return await super().mark_success(claim)

    async def run():
        store = HangingStore()
        app = await event_app(store)
        result = await asyncio.wait_for(
            app.run_event_watchers(
                [
                    EventWatcher(
                        name="hanging",
                        query=EventQuery(),
                        handler=lambda _: None,
                        lease_seconds=0.15,
                    ),
                    EventWatcher(name="other", query=EventQuery(), handler=lambda _: None),
                ]
            ),
            timeout=2,
        )
        assert result[0].deliveries[0].status is EventWatcherDeliveryStatus.PUBLICATION_FAILED
        assert result[1].deliveries[0].status is EventWatcherDeliveryStatus.SUCCEEDED
        assert (await store.load_state("hanging")).cursor_sequence == 0

    asyncio.run(run())


def test_unsupported_custom_renewal_never_dispatches_handler():
    class UnsupportedStore(CustomStore):
        async def renew_claim(self, claim, *, lease_seconds):
            return await EventWatcherStore.renew_claim(self, claim, lease_seconds=lease_seconds)

    async def run():
        app = await event_app(UnsupportedStore())
        calls = []
        result = await app.run_event_watchers(
            [EventWatcher(name="unsupported", query=EventQuery(), handler=calls.append)]
        )
        assert not calls
        assert result[0].deliveries[0].status is EventWatcherDeliveryStatus.LEASE_LOST

    asyncio.run(run())


def test_memory_and_sqlite_sample_clock_after_acquiring_authority(tmp_path):
    async def run():
        now = datetime(2026, 1, 1, tzinfo=UTC)

        def clock():
            return now

        stores = (
            InMemoryEventWatcherStore(clock=clock),
            SQLiteEventWatcherStore(tmp_path / "clock.sqlite", clock=clock),
        )
        for store in stores:
            await store._lock.acquire()
            task = asyncio.create_task(
                store.claim_event(watcher_name="clock", record=record(), lease_seconds=1)
            )
            await asyncio.sleep(0)
            now += timedelta(hours=1)
            store._lock.release()
            claim = await task
            assert isinstance(claim, EventWatcherClaim)
            assert claim.lease_expires_at == now + timedelta(seconds=1)
            await close_store(store)

    asyncio.run(run())


def test_postgres_watcher_ignores_worker_clock_skew(postgres_dsn, monkeypatch):
    import cayu.storage.postgres as postgres

    class SkewedClock(datetime):
        @classmethod
        def now(cls, tz=None):
            return datetime(2090, 1, 1, tzinfo=UTC)

    async def run():
        store = postgres.PostgresEventWatcherStore(postgres_dsn, schema_mode=SchemaMode.CREATE)
        try:
            monkeypatch.setattr(postgres, "datetime", SkewedClock)
            before = datetime.now(UTC)
            claim = await store.claim_event(
                watcher_name=f"clock-{uuid4()}", record=record(), lease_seconds=3
            )
            assert isinstance(claim, EventWatcherClaim)
            assert before < claim.lease_expires_at < datetime.now(UTC) + timedelta(seconds=4)
            await store.mark_success(claim)
        finally:
            await store.close()

    asyncio.run(run())


def test_sqlite_crashed_claimants_exhaust_attempts_across_processes(tmp_path):
    import os
    import subprocess
    import sys

    script = tmp_path / "crash_claimant.py"
    script.write_text("""
import asyncio, os, sys
from datetime import UTC, datetime, timedelta
from cayu.core import Event, EventType
from cayu.runtime.sessions import EventRecord
from cayu.runtime.event_watchers import EventWatcherClaim
from cayu.storage import SQLiteEventWatcherStore
async def run():
    now = datetime(2026, 1, 1, tzinfo=UTC) + timedelta(seconds=int(sys.argv[2]) * 2)
    store = SQLiteEventWatcherStore(sys.argv[1], clock=lambda: now)
    claim = await store.claim_event(watcher_name="crashed", record=EventRecord(sequence=1, event=Event(id="crash-event", type=EventType.BUDGET_LIMIT_REACHED, session_id="session")), lease_seconds=1, max_attempts=2)
    assert isinstance(claim, EventWatcherClaim)
    os._exit(73)
asyncio.run(run())
""")
    path = tmp_path / "crash.sqlite"
    env = {**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parents[2] / "src")}
    for attempt in (1, 2):
        result = subprocess.run(
            [sys.executable, str(script), str(path), str(attempt)],
            env=env,
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode == 73, result.stderr

    async def run():
        store = SQLiteEventWatcherStore(path, clock=lambda: datetime(2026, 1, 2, tzinfo=UTC))
        try:
            outcome = await store.claim_event(
                watcher_name="crashed",
                record=EventRecord(
                    sequence=1,
                    event=Event(
                        id="crash-event", type=EventType.BUDGET_LIMIT_REACHED, session_id="session"
                    ),
                ),
                lease_seconds=1,
                max_attempts=2,
            )
            assert isinstance(outcome, EventWatcherDelivery)
            assert outcome.status is EventWatcherDeliveryStatus.DEAD_LETTERED
            assert outcome.attempt == 2
            assert (await store.load_state("crashed")).dead_lettered_count == 1
        finally:
            await store.close()

    asyncio.run(run())


def test_sqlite_migration_preserves_pending_attempt_and_reopened_receipt(tmp_path):
    import sqlite3

    async def run():
        path = tmp_path / "upgrade.sqlite"
        store = SQLiteEventWatcherStore(path)
        claim = await store.claim_event(watcher_name="upgrade", record=record(), lease_seconds=60)
        await store.close()
        with sqlite3.connect(path) as connection:
            connection.execute("DROP TABLE cayu_event_watcher_settlements")
            connection.execute("DELETE FROM cayu_schema_migrations WHERE revision >= 81")
        with pytest.raises(Exception, match="requires >= 81"):
            SQLiteEventWatcherStore(path, schema_mode=SchemaMode.VALIDATE)
        store = SQLiteEventWatcherStore(path, schema_mode=SchemaMode.MIGRATE)
        assert (await store.load_state("upgrade")).pending_attempt == 1
        receipt = await store.mark_success(claim)
        await store.close()
        reopened = SQLiteEventWatcherStore(path, schema_mode=SchemaMode.VALIDATE)
        try:
            assert await reopened.mark_success(claim) == receipt
            assert (await reopened.load_state("upgrade")).cursor_sequence == 1
        finally:
            await reopened.close()

    asyncio.run(run())


def test_custom_store_cannot_publish_a_cursor_that_skips_an_event():
    class InvalidSettlement(InMemoryEventWatcherStore):
        async def mark_success(self, claim):
            result = await super().mark_success(claim)
            return result.model_copy(update={"cursor_sequence": claim.event_sequence + 1})

    async def run():
        app = await event_app(InvalidSettlement())
        results = await app.run_event_watchers(
            [EventWatcher(name="invalid", query=EventQuery(), handler=lambda _: None)]
        )
        assert results[0].deliveries[0].status is EventWatcherDeliveryStatus.PUBLICATION_FAILED

    asyncio.run(run())


def test_sqlite_renewal_contention_does_not_starve_other_watcher_leases(tmp_path):
    import sqlite3

    async def run():
        path = tmp_path / "contended.sqlite"
        store = SQLiteEventWatcherStore(path)
        app = await event_app(store)
        healthy_app = await event_app(InMemoryEventWatcherStore())
        healthy_started, healthy_release = asyncio.Event(), asyncio.Event()
        locked = asyncio.Event()
        writer = sqlite3.connect(path)

        async def healthy_handler(_context):
            healthy_started.set()
            await healthy_release.wait()

        async def contended_handler(_context):
            writer.execute("BEGIN IMMEDIATE")
            locked.set()
            await asyncio.Event().wait()

        healthy = asyncio.create_task(
            healthy_app.run_event_watchers(
                [
                    EventWatcher(
                        name="healthy",
                        query=EventQuery(),
                        handler=healthy_handler,
                        lease_seconds=0.6,
                    )
                ]
            )
        )
        contended = None
        try:
            await asyncio.wait_for(healthy_started.wait(), timeout=10)
            contended = asyncio.create_task(
                app.run_event_watchers(
                    [
                        EventWatcher(
                            name="contended",
                            query=EventQuery(),
                            handler=contended_handler,
                            lease_seconds=0.3,
                        )
                    ]
                )
            )
            await asyncio.wait_for(locked.wait(), timeout=3)
            started = asyncio.get_running_loop().time()
            result = await asyncio.wait_for(contended, timeout=0.5)
            assert asyncio.get_running_loop().time() - started < 0.5
            assert result[0].deliveries[0].status is EventWatcherDeliveryStatus.LEASE_LOST
            # Keep the writer locked beyond the independent watcher's initial
            # lease. Its heartbeat must still run and preserve ownership.
            await asyncio.sleep(0.7)
            assert writer.in_transaction
            healthy_release.set()
            completed = await asyncio.wait_for(healthy, timeout=3)
            assert completed[0].deliveries[0].status is EventWatcherDeliveryStatus.SUCCEEDED
        finally:
            writer.rollback()
            writer.close()
            healthy_release.set()
            pending = [healthy] + ([] if contended is None else [contended])
            for task in pending:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*pending, return_exceptions=True)
            while app._event_watcher_supervisor.active("contended"):
                await asyncio.sleep(0.005)
            await store.close()

    asyncio.run(run())


@pytest.mark.parametrize("cancel_waiter", [False, True])
def test_sqlite_claim_contention_retries_and_releases_connection_on_cancel(tmp_path, cancel_waiter):
    import sqlite3

    async def run():
        path = tmp_path / "retry.sqlite"
        store = SQLiteEventWatcherStore(path)
        writer = sqlite3.connect(path)
        writer.execute("BEGIN IMMEDIATE")
        task = asyncio.create_task(
            store.claim_event(watcher_name="retry", record=record(), lease_seconds=5)
        )
        try:
            await asyncio.sleep(0.05)
            assert not task.done()
            if cancel_waiter:
                task.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await asyncio.wait_for(task, timeout=0.5)
                assert not store._connection.in_transaction
                assert not store._lock.locked()
            writer.rollback()
            if cancel_waiter:
                claim = await store.claim_event(
                    watcher_name="retry", record=record(), lease_seconds=5
                )
            else:
                claim = await asyncio.wait_for(task, timeout=1)
            assert isinstance(claim, EventWatcherClaim)
            assert claim.attempt == 1
            assert (await store.mark_success(claim)).status is EventWatcherDeliveryStatus.SUCCEEDED
        finally:
            writer.rollback()
            writer.close()
            if not task.done():
                task.cancel()
            await asyncio.gather(task, return_exceptions=True)
            await store.close()

    asyncio.run(run())
