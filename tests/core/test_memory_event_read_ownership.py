"""Memory event reads compose over explicit state without a session-store host."""

from __future__ import annotations

import asyncio
import os
import subprocess
import sys
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from pathlib import Path

import pytest
from tests._session_provenance import fixture_session_invocation

import cayu
from cayu.events import Event, EventType
from cayu.runtime._cost_accounting_index import CostEventIndex
from cayu.runtime._cost_accounting_refresh import CostAccountingAuthority
from cayu.runtime._usage_accounting import SessionUsageCache
from cayu.sessions import _memory_event_reads as reads
from cayu.sessions._access_rules import (
    SessionAccessDenied,
    SessionAccessRule,
    SessionAccessScope,
    SessionAccessSelector,
    _query_bounds,
    _SessionAccessBounds,
)
from cayu.sessions.event_queries import EventQuery, EventQueryResultTooLarge
from cayu.sessions.records import EventRecord, Session


def _state() -> reads.MemoryEventReadState:
    now = datetime(2026, 1, 1, tzinfo=UTC)
    session = Session(
        id="session",
        agent_name="assistant",
        provider_name="fake",
        model="fake",
        causal_budget_id="session",
        invocation=fixture_session_invocation("session"),
        created_at=now,
        updated_at=now,
        last_activity_at=now,
        labels={"team": "one"},
    )
    records = [
        EventRecord(
            sequence=index,
            event=Event(
                id=f"event-{index}",
                session_id=session.id,
                type=EventType.SESSION_STARTED,
                timestamp=now,
                payload={"nested": {"value": index}},
            ),
        )
        for index in (1, 2)
    ]
    return reads.MemoryEventReadState(
        accounting_generation=3,
        checkpoints={},
        cost_accounting_authority=CostAccountingAuthority(),
        cost_event_index=CostEventIndex(),
        event_records=records,
        event_records_by_id={(session.id, r.event.id): r for r in records},
        events={session.id: [r.event for r in records]},
        interaction_event_records={},
        latest_interaction_event_records_by_sequence={},
        pending_action_event_history={},
        session_event_records={session.id: records},
        session_usage_cache=SessionUsageCache(),
        sessions={session.id: session},
        transcript_interaction_ids={},
        transcripts={},
        type_event_records={str(EventType.SESSION_STARTED): records},
        user_input_supersession_event_records={},
        workflow_attempt_event_records={},
        workflow_fenced_step_event_records={},
        workflow_step_event_records={},
    )


def test_event_reads_import_without_store_implementations() -> None:
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
        if fullname in {"cayu.sessions.base", "cayu.sessions.memory", "cayu.sessions.store",
                        "cayu.storage.sqlite", "cayu.storage.postgres"}:
            raise AssertionError(fullname)
sys.meta_path.insert(0, RejectStores())
owner = importlib.import_module("cayu.sessions._memory_event_reads")
assert callable(owner.query_events)
assert callable(owner.load_terminal_session_evidence)
""",
        ],
        env={**os.environ, "PYTHONPATH": str(Path(cayu.__file__).resolve().parent.parent)},
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_event_reads_compose_with_independent_locked_state() -> None:
    async def scenario():
        state = _state()
        lock = asyncio.Lock()
        contexts = 0

        @asynccontextmanager
        async def read():
            nonlocal contexts
            async with lock:
                contexts += 1
                yield state

        records = await reads.query_events(read, EventQuery(session_id="session", limit=1))
        assert [r.event.id for r in records] == ["event-1"]
        records[0].event.payload["nested"]["value"] = 999
        assert state.event_records[0].event.payload["nested"]["value"] == 1
        assert await reads.event_exists(read, EventQuery(session_id="session", after_sequence=1))
        assert not await reads.event_exists(
            read, EventQuery(session_id="session", after_sequence=2)
        )
        events = await reads.load_events(read, "session")
        events[0].payload.clear()
        assert state.events["session"][0].payload
        assert (await reads.read_usage_accounting(read, EventQuery())).generation == 3
        with pytest.raises(EventQueryResultTooLarge):
            await reads.query_events_bounded(read, EventQuery(), max_bytes=1)
        assert contexts == 6
        assert not lock.locked()

    asyncio.run(scenario())


def test_event_reads_apply_access_bounds_with_independent_state() -> None:
    async def scenario():
        state = _state()

        @asynccontextmanager
        async def read():
            yield state

        scope = SessionAccessScope(
            read=(
                SessionAccessRule(selectors=(SessionAccessSelector(key="team", values=("two",)),)),
            )
        )
        token = _query_bounds.set(_SessionAccessBounds(scope, scope))
        try:
            assert await reads.query_events(read, EventQuery()) == []
            assert not await reads.event_exists(read, EventQuery())
            from cayu.budgets.pricing import ModelPrice, PriceBook

            with pytest.raises(SessionAccessDenied):
                await reads.read_cost_accounting(
                    read,
                    EventQuery(),
                    PriceBook(
                        prices=(
                            ModelPrice.fixed(
                                provider_name="fake",
                                model="fake",
                                input_per_million=1,
                                output_per_million=1,
                            ),
                        )
                    ),
                    additional_events=(state.event_records[0].event,),
                )
        finally:
            _query_bounds.reset(token)

    asyncio.run(scenario())


def test_store_event_view_is_acquired_after_the_writer_lock() -> None:
    from cayu.sessions.memory import InMemorySessionStore

    async def scenario():
        store = InMemorySessionStore()
        state = _state()
        await store._lock.acquire()
        pending = asyncio.create_task(store.query_events(EventQuery(session_id="session")))
        try:
            await asyncio.sleep(0)
            assert not pending.done()
            # Writers may replace a complete dictionary while a read is waiting.
            store._sessions = state.sessions
            store._session_event_records = state.session_event_records
        finally:
            store._lock.release()
        assert [r.event.id for r in await pending] == ["event-1", "event-2"]
        store._session_event_records = {}
        assert await store.query_events(EventQuery(session_id="session")) == []

    asyncio.run(scenario())


def test_accounting_cancellation_releases_the_read_context() -> None:
    from cayu.runtime._usage_accounting import USAGE_ACCOUNTING_PAGE_SIZE

    async def scenario():
        state = _state()
        source = state.event_records[0].event.model_copy(update={"type": EventType.MODEL_COMPLETED})
        records = [
            EventRecord(sequence=i + 1, event=source) for i in range(USAGE_ACCOUNTING_PAGE_SIZE + 1)
        ]
        state = state._replace(event_records=records)
        lock = asyncio.Lock()
        started = asyncio.Event()

        @asynccontextmanager
        async def read():
            async with lock:
                started.set()
                yield state

        task = asyncio.create_task(reads.read_usage_accounting(read, EventQuery()))
        await started.wait()
        assert lock.locked()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert not lock.locked()

    asyncio.run(scenario())
