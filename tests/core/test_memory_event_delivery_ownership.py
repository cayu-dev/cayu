"""Memory delivery operations compose with explicit state and one owned lock."""

from __future__ import annotations

import asyncio
import os
import subprocess
import sys
from contextlib import asynccontextmanager
from copy import deepcopy
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

import cayu
from cayu.events import Event, EventType
from cayu.sessions import _memory_event_delivery as delivery
from cayu.sessions._closure_records import ClosureProgressIndex
from cayu.sessions.event_delivery import (
    PersistedEventSideEffectClaimLost,
    PersistedEventSideEffectDelivery,
    PersistedEventSideEffectQuery,
    PersistedEventSideEffectStatus,
)
from cayu.sessions.records import EventRecord


def _state(clock):
    now = datetime(2026, 1, 1, tzinfo=UTC)
    records = {}
    deliveries = {}
    for sequence, event_id in enumerate(("a", "b", "c"), 1):
        event = Event(
            id=event_id,
            type=EventType.MODEL_COMPLETED,
            session_id="session",
            timestamp=now,
            payload={"nested": {"value": "original"}},
        )
        records[("session", event_id)] = EventRecord(sequence=sequence, event=event)
        deliveries[("session", event_id)] = PersistedEventSideEffectDelivery(
            session_id="session",
            event_id=event_id,
            event_sequence=sequence,
            status=PersistedEventSideEffectStatus.PENDING,
            updated_at=now,
        )
    return delivery.MemoryEventDeliveryState(deliveries, records, ClosureProgressIndex(), clock)


def test_memory_delivery_imports_without_store_implementations():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            """
import importlib
import importlib.abc
import sys
class RejectStores(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname in {"cayu.sessions.base", "cayu.sessions.store", "cayu.sessions.memory",
                        "cayu.storage.sqlite", "cayu.storage.postgres"}:
            raise AssertionError(fullname)
sys.meta_path.insert(0, RejectStores())
owner = importlib.import_module("cayu.sessions._memory_event_delivery")
assert callable(owner.claim_persisted_event_side_effect)
assert callable(owner.get_persisted_event_side_effect_health)
""",
        ],
        env={**os.environ, "PYTHONPATH": str(Path(cayu.__file__).resolve().parent.parent)},
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_memory_delivery_composes_with_clock_fencing_and_detached_results():
    async def scenario():
        now = datetime(2026, 1, 1, tzinfo=UTC)
        lock = asyncio.Lock()
        entrances = 0

        def clock():
            assert lock.locked()
            return now

        state = _state(clock)

        @asynccontextmanager
        async def locked():
            nonlocal entrances
            async with lock:
                entrances += 1
                yield state

        claim = await delivery.claim_persisted_event_side_effect(locked, lease_seconds=10)
        assert entrances == 1 and not lock.locked()
        assert claim.event_id == "a" and claim.attempt == 1
        assert claim.lease_expires_at == now + timedelta(seconds=10)
        claim.event.payload["nested"]["value"] = "changed output"
        assert state.event_records_by_id[("session", "a")].event.payload["nested"]["value"] == (
            "original"
        )
        now += timedelta(seconds=1)
        renewed = await delivery.renew_persisted_event_side_effect(locked, claim, lease_seconds=20)
        assert renewed.lease_expires_at == now + timedelta(seconds=20)
        failed = await delivery.mark_persisted_event_side_effect_failed(
            locked, claim, error="sink unavailable", max_attempts=3, retry_delay_seconds=5
        )
        assert failed.next_attempt_at == now + timedelta(seconds=5)
        failed.last_error = "changed output"
        assert (
            await delivery.get_persisted_event_side_effect_delivery(
                locked, session_id="session", event_id="a"
            )
        ).last_error == "sink unavailable"
        assert (
            await delivery.claim_persisted_event_side_effect(
                locked, session_id="session", event_id="a"
            )
            is None
        )
        now += timedelta(seconds=5)
        retry = await delivery.claim_persisted_event_side_effect(
            locked, session_id="session", event_id="a"
        )
        assert retry.attempt == 2
        before = deepcopy(state.persisted_event_side_effect_deliveries)
        with pytest.raises(PersistedEventSideEffectClaimLost):
            await delivery.mark_persisted_event_side_effect_delivered(locked, claim)
        assert state.persisted_event_side_effect_deliveries == before and not lock.locked()
        deferred = await delivery.defer_persisted_event_side_effect(locked, retry)
        assert deferred.attempts == 1 and deferred.status == "pending"
        replacement = await delivery.claim_persisted_event_side_effect(
            locked, session_id="session", event_id="a"
        )
        assert replacement.attempt == 2 and replacement.claim_id != retry.claim_id
        done = await delivery.mark_persisted_event_side_effect_delivered(locked, replacement)
        assert done.status == "delivered"
        health = await delivery.get_persisted_event_side_effect_health(locked)
        assert health.observed_at == now and health.delivered == 1 and health.claimable_total == 2
        query = PersistedEventSideEffectQuery(limit=1)
        page = await delivery.query_persisted_event_side_effect_deliveries(locked, query)
        assert [row.event_id for row in page.deliveries] == ["b"] and page.next_cursor
        last = await delivery.query_persisted_event_side_effect_deliveries(
            locked, query.model_copy(update={"cursor": page.next_cursor})
        )
        assert [row.event_id for row in last.deliveries] == ["c"] and last.next_cursor is None
        rows = await delivery.list_persisted_event_side_effect_deliveries(
            locked, claimable_only=True, after_sequence=1, limit=1
        )
        assert [row.event_id for row in rows] == ["b"]
        rows[0].attempts = 99
        assert state.persisted_event_side_effect_deliveries[("session", "b")].attempts == 0

    asyncio.run(scenario())


def test_memory_delivery_first_attempt_and_retirement_are_exact():
    async def scenario():
        state = _state(lambda: datetime(2026, 1, 1, tzinfo=UTC))
        lock = asyncio.Lock()

        @asynccontextmanager
        async def locked():
            async with lock:
                yield state

        expected = state.persisted_event_side_effect_deliveries[("session", "a")].model_copy()
        claim = await delivery.claim_first_persisted_event_side_effect(locked, expected)
        assert claim.attempt == 1
        assert await delivery.claim_first_persisted_event_side_effect(locked, expected) is None
        failed = await delivery.mark_persisted_event_side_effect_failed(
            locked, claim, error="sink unavailable", max_attempts=3, retry_delay_seconds=0
        )
        assert (
            await delivery.retire_failed_first_event_delivery(
                locked, failed.model_copy(update={"last_error": "different"})
            )
            is None
        )
        retired = await delivery.retire_failed_first_event_delivery(locked, failed)
        assert retired.status == "dead_lettered"
        assert (
            await delivery.claim_persisted_event_side_effect(
                locked, session_id="session", event_id="a"
            )
            is None
        )
        state.session_closure_progress[("session", "closure")] = {
            "root_session_id": "session",
            "plan_id": "closure",
            "descendants": [],
        }
        before = deepcopy(state.persisted_event_side_effect_deliveries)
        assert await delivery.claim_persisted_event_side_effect(locked) is None
        assert state.persisted_event_side_effect_deliveries == before

    asyncio.run(scenario())


@pytest.mark.parametrize("operation", ["claim", "read", "health", "closure"])
def test_memory_delivery_resolves_current_state_and_clock_after_lock(operation):
    from cayu.sessions.memory import InMemorySessionStore

    async def scenario():
        def old_clock():
            raise AssertionError("Clock captured before lock acquisition")

        store = InMemorySessionStore(ownership_clock=old_clock)
        now = datetime(2026, 2, 1, tzinfo=UTC)
        state = _state(lambda: now)
        old_deliveries = store._persisted_event_side_effect_deliveries
        if operation == "closure":
            state.session_closure_progress[("session", "closure")] = {
                "root_session_id": "session",
                "plan_id": "closure",
                "descendants": [],
            }
        await store._lock.acquire()
        if operation == "read":
            work = store.get_persisted_event_side_effect_delivery(
                session_id="session", event_id="a"
            )
        elif operation == "health":
            work = store.get_persisted_event_side_effect_health()
        else:
            work = store.claim_persisted_event_side_effect()
        pending = asyncio.create_task(work)
        try:
            await asyncio.sleep(0)
            assert not pending.done()
            for name, value in zip(state._fields, state, strict=True):
                setattr(store, "_" + name, value)
        finally:
            store._lock.release()
        result = await pending
        assert old_deliveries == {} and not store._lock.locked()
        if operation == "read":
            assert result == state.persisted_event_side_effect_deliveries[("session", "a")]
        elif operation == "health":
            assert result.observed_at == now and result.pending == 3
        elif operation == "closure":
            assert result is None
        else:
            assert result.event_id == "a" and result.lease_expires_at == now + timedelta(
                seconds=300
            )

    asyncio.run(scenario())


@pytest.mark.parametrize("write", [False, True])
def test_cancelled_memory_delivery_waiter_preserves_state_and_lock(write):
    async def scenario():
        state = _state(lambda: datetime(2026, 1, 1, tzinfo=UTC))
        before = deepcopy(state)
        lock = asyncio.Lock()
        started = asyncio.Event()

        @asynccontextmanager
        async def locked():
            started.set()
            async with lock:
                yield state

        await lock.acquire()
        work = (
            delivery.claim_persisted_event_side_effect(locked)
            if write
            else delivery.get_persisted_event_side_effect_health(locked)
        )
        pending = asyncio.create_task(work)
        try:
            await started.wait()
            pending.cancel()
            with pytest.raises(asyncio.CancelledError):
                await pending
            assert state == before and lock.locked()
        finally:
            lock.release()
        assert (await delivery.claim_persisted_event_side_effect(locked)).attempt == 1
        assert not lock.locked()

    asyncio.run(scenario())


def test_memory_delivery_missing_source_does_not_publish_claim():
    async def scenario():
        state = _state(lambda: datetime(2026, 1, 1, tzinfo=UTC))
        state.event_records_by_id.clear()
        before = deepcopy(state)
        lock = asyncio.Lock()

        @asynccontextmanager
        async def locked():
            async with lock:
                yield state

        with pytest.raises(RuntimeError, match="lost its source event"):
            await delivery.claim_persisted_event_side_effect(locked)
        assert state == before and not lock.locked()

    asyncio.run(scenario())
