from __future__ import annotations

import asyncio

import pytest

from cayu.core import Event, EventType, Message
from cayu.runtime import InMemorySessionStore, RunRequest, SessionIdentity
from cayu.runtime._event_writer import RuntimeEventWriter
from cayu.runtime.budgets import BudgetWindow, InMemoryBudgetStore
from cayu.runtime.event_sinks import EventSink


class _RecordingSink(EventSink):
    def __init__(self) -> None:
        self.events: list[Event] = []

    async def emit(self, event: Event) -> None:
        self.events.append(event)


class _FailingSink(EventSink):
    async def emit(self, event: Event) -> None:
        raise RuntimeError("sink unavailable")


async def _session_store(session_id: str) -> InMemorySessionStore:
    store = InMemorySessionStore()
    await store.create(
        RunRequest(
            agent_name="assistant",
            session_id=session_id,
            messages=[Message.text("user", "go")],
        ),
        identity=SessionIdentity(provider_name="fake", model="fake-model"),
    )
    return store


def test_emit_persists_forwards_cost_event_and_fans_out() -> None:
    async def scenario() -> tuple[list[Event], list[Event], list[Event], Event]:
        store = await _session_store("writer_single")
        budget_store = InMemoryBudgetStore()
        sink = _RecordingSink()
        writer = RuntimeEventWriter(
            session_store=store,
            budget_store=budget_store,
            event_sinks=[sink],
        )
        event = Event(type=EventType.MODEL_COMPLETED, session_id="writer_single")

        emitted = await writer.emit(event)
        persisted = await store.load_events("writer_single")
        budget_events = await budget_store.load_events_for_budget(
            scope="app",
            key=None,
            window=BudgetWindow.all_time(),
        )
        return persisted, budget_events, sink.events, emitted

    persisted, budget_events, sink_events, emitted = asyncio.run(scenario())

    assert [event.id for event in persisted] == [emitted.id]
    assert [event.id for event in budget_events] == [emitted.id]
    assert [event.id for event in sink_events] == [emitted.id]


def test_emit_many_copies_events_before_persisting_and_fanout() -> None:
    async def scenario() -> tuple[list[Event], list[Event], list[Event]]:
        store = await _session_store("writer_batch")
        sink = _RecordingSink()
        writer = RuntimeEventWriter(
            session_store=store,
            budget_store=InMemoryBudgetStore(),
            event_sinks=[sink],
        )
        source = [
            Event(type="custom.example.one", session_id="writer_batch", payload={"value": 1}),
            Event(type="custom.example.two", session_id="writer_batch", payload={"value": 2}),
        ]

        emitted = await writer.emit_many("writer_batch", source)
        source[0].payload["value"] = 99
        return emitted, await store.load_events("writer_batch"), sink.events

    emitted, persisted, sink_events = asyncio.run(scenario())

    expected = [{"value": 1}, {"value": 2}]
    assert [event.payload for event in emitted] == expected
    assert [event.payload for event in persisted] == expected
    assert [event.payload for event in sink_events] == expected


def test_emit_many_rejects_event_for_different_session() -> None:
    async def scenario() -> None:
        writer = RuntimeEventWriter(
            session_store=await _session_store("writer_target"),
            budget_store=InMemoryBudgetStore(),
            event_sinks=[],
        )

        with pytest.raises(ValueError, match="session_id does not match"):
            await writer.emit_many(
                "writer_target",
                [Event(type="custom.example", session_id="writer_other")],
            )

    asyncio.run(scenario())


def test_sink_failure_is_durable_and_does_not_block_later_sink() -> None:
    async def scenario() -> tuple[list[Event], list[Event]]:
        store = await _session_store("writer_sink_failure")
        recorder = _RecordingSink()
        writer = RuntimeEventWriter(
            session_store=store,
            budget_store=InMemoryBudgetStore(),
            event_sinks=[_FailingSink(), recorder],
        )
        event = Event(type=EventType.SESSION_STARTED, session_id="writer_sink_failure")

        await writer.emit(event)
        return await store.load_events("writer_sink_failure"), recorder.events

    persisted, recorded = asyncio.run(scenario())

    assert [event.type for event in persisted] == [
        EventType.SESSION_STARTED,
        EventType.RUNTIME_SINK_FAILED,
    ]
    assert persisted[1].payload == {
        "sink": "_FailingSink",
        "error": "sink unavailable",
        "error_type": "RuntimeError",
        "event_id": persisted[0].id,
        "event_type": EventType.SESSION_STARTED,
    }
    assert [event.type for event in recorded] == [EventType.SESSION_STARTED]
