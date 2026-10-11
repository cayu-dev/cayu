"""Watcher batches compose independently while retaining live store lookup."""

from __future__ import annotations

import asyncio
import os
import subprocess
import sys
from pathlib import Path

from cayu.events import Event, EventType
from cayu.observability.watchers import (
    EventWatcher,
    EventWatcherDeliveryStatus,
    InMemoryEventWatcherStore,
)
from cayu.runtime._event_projection import public_event_id
from cayu.runtime._event_watcher_delivery import EventWatcherSupervisor, dispatch_event_watchers
from cayu.sessions.event_queries import EventQuery
from cayu.sessions.memory import InMemorySessionStore
from cayu.sessions.records import EventRecord, SessionIdentity
from cayu.sessions.requests import RunRequest
from cayu.vaults.redaction import SecretRedactor


async def _sessions(*values: str) -> InMemorySessionStore:
    store = InMemorySessionStore()
    await store.create(
        RunRequest(session_id="events", agent_name="assistant", messages=[]),
        identity=SessionIdentity(provider_name="fake", model="fake"),
    )
    for value in values:
        await store.append_event(
            "events",
            Event(
                type=EventType.BUDGET_LIMIT_REACHED,
                session_id="events",
                payload={"value": value},
            ),
        )
    return store


async def _dispatch_without_application() -> None:
    class RecordingStore(InMemoryEventWatcherStore):
        def __init__(self):
            super().__init__()
            self.claimed_ids: list[str] = []

        async def claim_event(self, **kwargs):
            self.claimed_ids.append(kwargs["record"].event.id)
            return await super().claim_event(**kwargs)

    sessions = await _sessions("first", "second")
    records = await sessions.query_events(EventQuery())
    store = RecordingStore()
    supervisor = EventWatcherSupervisor()
    seen: list[str] = []

    def project(record: EventRecord) -> EventRecord:
        return EventRecord(
            sequence=record.sequence,
            event=record.event.model_copy(update={"id": f"handler-{record.sequence}"}),
        )

    async def handler(context):
        seen.append(context.record.event.id)

    watchers = [
        EventWatcher(name="one", query=EventQuery(), handler=handler, batch_size=1),
        EventWatcher(name="two", query=EventQuery(), handler=handler, batch_size=2),
    ]

    async def dispatch(limit):
        return await dispatch_event_watchers(
            watchers,
            limit=limit,
            get_session_store=lambda: sessions,
            get_watcher_store=lambda: store,
            supervisor=supervisor,
            redactor=SecretRedactor(),
            project_record=project,
        )

    try:
        first = await dispatch(2)
        assert [len(result.deliveries) for result in first] == [1, 1]
        assert seen == ["handler-1", "handler-1"]
        assert store.claimed_ids == [records[0].event.id] * 2
        assert all(result.error is None for result in first)
        assert all(
            result.deliveries[0].status is EventWatcherDeliveryStatus.SUCCEEDED
            and result.deliveries[0].event_id == public_event_id(records[0].sequence)
            for result in first
        )
        second = await dispatch(10)
        assert [len(result.deliveries) for result in second] == [1, 1]
        assert seen == ["handler-1", "handler-1", "handler-2", "handler-2"]
        assert store.claimed_ids == [records[0].event.id] * 2 + [records[1].event.id] * 2
        assert all(not result.deliveries for result in await dispatch(10))
        assert (await store.load_state("one")).cursor_sequence == records[-1].sequence
        assert (await store.load_state("two")).cursor_sequence == records[-1].sequence
    finally:
        assert await supervisor.drain(timeout_s=5)


def test_watcher_batch_executes_without_application_or_engine() -> None:
    script = """
import asyncio
import importlib.abc
import sys

blocked = {"cayu.applications", "cayu.runtime._session_engine"}
class RejectOwners(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname in blocked:
            raise AssertionError(f"Watcher batch imported {fullname}")
sys.meta_path.insert(0, RejectOwners())
from tests.core.test_event_watcher_composition import _dispatch_without_application
asyncio.run(_dispatch_without_application())
assert not blocked.intersection(sys.modules)
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=Path(__file__).resolve().parents[2],
        env=os.environ.copy(),
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_public_watcher_batch_resolves_replaced_stores_after_await(monkeypatch) -> None:
    from cayu.applications import CayuApp

    async def scenario() -> None:
        entered, release = asyncio.Event(), asyncio.Event()

        class GatedStore(InMemoryEventWatcherStore):
            async def load_state(self, watcher_name):
                state = await super().load_state(watcher_name)
                entered.set()
                await release.wait()
                return state

        original_sessions = await _sessions("original")
        replacement_sessions = await _sessions("replacement")
        original_store, replacement_store = GatedStore(), InMemoryEventWatcherStore()
        app = CayuApp(
            session_store=original_sessions,
            event_watcher_store=original_store,
            enable_logging=False,
        )
        seen: list[str] = []

        async def handler(context):
            seen.append(context.record.event.payload["value"])

        watcher = EventWatcher(name="replacement", query=EventQuery(), handler=handler)
        caller = None
        try:
            with monkeypatch.context() as patch:
                patch.delattr(app, "_session_engine")
                caller = asyncio.create_task(app.run_event_watchers([watcher], limit=1))
                await asyncio.wait_for(entered.wait(), timeout=5)
                patch.setattr(app, "session_store", replacement_sessions)
                patch.setattr(app, "event_watcher_store", replacement_store)
                release.set()
                results = await asyncio.wait_for(caller, timeout=5)
                assert seen == ["replacement"]
                assert results[0].error is None
                assert results[0].deliveries[0].status is EventWatcherDeliveryStatus.SUCCEEDED
                assert (await replacement_store.load_state(watcher.name)).cursor_sequence == 1
                assert (await original_store.load_state(watcher.name)).cursor_sequence == 0
        finally:
            release.set()
            if caller is not None and not caller.done():
                caller.cancel()
                await asyncio.gather(caller, return_exceptions=True)
            assert (await app.aclose(timeout_s=5)).settled

    asyncio.run(scenario())
