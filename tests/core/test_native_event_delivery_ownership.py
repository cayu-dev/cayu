"""Native delivery operations compose without retaining a session store."""

from __future__ import annotations

import asyncio
import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import pytest

import cayu
from cayu.events import Event, EventType
from cayu.sessions.base import RunRequest
from cayu.sessions.event_delivery import (
    PersistedEventSideEffectClaimLost,
    PersistedEventSideEffectQuery,
    PersistedEventSideEffectStatus,
)
from cayu.sessions.records import SessionIdentity


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
