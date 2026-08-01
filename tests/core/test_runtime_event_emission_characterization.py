"""Black-box characterization for CayuApp event persistence and sink fan-out."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator

from cayu.core import AgentSpec, Event, EventType, Message
from cayu.providers import ModelProvider, ModelRequest, ModelStreamEvent
from cayu.runtime import (
    CayuApp,
    InMemorySessionStore,
    RunRequest,
    SessionIdentity,
    SessionStatus,
)
from cayu.runtime._event_projection import public_event_id, public_event_sequence
from cayu.runtime.budgets import BudgetStore
from cayu.runtime.event_sinks import EventSink
from cayu.runtime.sessions import EventQuery, EventRecord, SessionQuery
from cayu.vaults import SecretRedactor


class _OneShotProvider(ModelProvider):
    name = "fake"

    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
        yield ModelStreamEvent.text_delta("done")
        yield ModelStreamEvent.completed({"finish_reason": "stop"})


class _StoreObservingSink(EventSink):
    def __init__(self, store: InMemorySessionStore) -> None:
        self._store = store
        self.persisted_before_delivery: list[bool] = []

    async def emit(self, event: Event) -> None:
        sequence = public_event_sequence(event.id)
        persisted = (
            []
            if sequence is None
            else await self._store.query_events(
                EventQuery(
                    session_id=event.session_id,
                    after_sequence=sequence - 1,
                    limit=1,
                )
            )
        )
        self.persisted_before_delivery.append(bool(persisted and persisted[0].sequence == sequence))


class _FailingSink(EventSink):
    async def emit(self, event: Event) -> None:
        raise RuntimeError("sink unavailable")


class _MutatingSink(EventSink):
    async def emit(self, event: Event) -> None:
        event.payload["mutated"] = True


class _RecordingSink(EventSink):
    def __init__(self) -> None:
        self.events: list[Event] = []

    async def emit(self, event: Event) -> None:
        self.events.append(event)


class _RecordingBudgetStore(BudgetStore):
    def __init__(self) -> None:
        self.events: list[Event] = []

    async def append_event(self, event: Event) -> None:
        self.events.append(event.model_copy(deep=True))

    async def load_events_for_budget(self, *, scope, key, window) -> list[Event]:
        return [event.model_copy(deep=True) for event in self.events]


async def _collect_run(app: CayuApp, session_id: str) -> list[Event]:
    app.register_provider(_OneShotProvider(), default=True)
    app.register_agent(AgentSpec(name="assistant", model="fake-model"))
    return [
        event
        async for event in app.run(
            RunRequest(
                agent_name="assistant",
                session_id=session_id,
                messages=[Message.text("user", "go")],
            )
        )
    ]


def test_runtime_event_is_persisted_before_sink_delivery() -> None:
    store = InMemorySessionStore()
    sink = _StoreObservingSink(store)
    app = CayuApp(session_store=store, event_sinks=[sink], enable_logging=False)

    events = asyncio.run(_collect_run(app, "sess_persist_before_sink"))

    assert len(sink.persisted_before_delivery) == len(events)
    assert all(sink.persisted_before_delivery)


def test_generated_session_and_interaction_ids_survive_short_secret_collisions() -> None:
    async def scenario() -> tuple[list[Event], list[EventRecord]]:
        store = InMemorySessionStore()
        app = CayuApp(
            session_store=store,
            secret_redactor=SecretRedactor("-"),
            enable_logging=False,
        )
        app.register_provider(_OneShotProvider(), default=True)
        app.register_agent(AgentSpec(name="assistant", model="model"))

        events = [
            event
            async for event in app.run(
                RunRequest(
                    agent_name="assistant",
                    messages=[Message.text("user", "go")],
                )
            )
        ]
        sessions = (await store.list_sessions(SessionQuery(limit=1))).sessions
        assert len(sessions) == 1
        records = await store.query_events(EventQuery(session_id=sessions[0].id, limit=100))
        return events, records

    events, records = asyncio.run(scenario())

    assert events[-1].type == EventType.SESSION_COMPLETED
    assert all(
        event.id == public_event_id(record.sequence)
        for event, record in zip(events, records, strict=True)
    )
    assert all("-" in record.event.session_id for record in records)
    assert all(record.event.interaction_id is not None for record in records[:-2])


def test_failing_sink_does_not_block_later_sink() -> None:
    store = InMemorySessionStore()
    recorder = _RecordingSink()
    app = CayuApp(
        session_store=store,
        event_sinks=[_FailingSink(), recorder],
        enable_logging=False,
    )

    events = asyncio.run(_collect_run(app, "sess_later_sink"))
    persisted = asyncio.run(store.load_events("sess_later_sink"))
    persisted_records = asyncio.run(store.query_events(EventQuery(session_id="sess_later_sink")))
    session = asyncio.run(store.load("sess_later_sink"))

    assert [event.type for event in events] == [
        EventType.INTERACTION_STARTED,
        EventType.SESSION_STARTED,
        EventType.MODEL_STARTED,
        EventType.MODEL_TEXT_DELTA,
        EventType.MODEL_COMPLETED,
        EventType.INTERACTION_COMPLETED,
        EventType.TURN_COMPLETED,
        EventType.SESSION_COMPLETED,
    ]
    assert [event.id for event in events] == [
        public_event_id(record.sequence)
        for record in persisted_records
        if record.event.type != EventType.RUNTIME_SINK_FAILED
    ]
    assert [event.id for event in recorder.events] == [event.id for event in events]
    assert session is not None
    assert session.status == SessionStatus.COMPLETED
    sink_failures = [event for event in persisted if event.type == EventType.RUNTIME_SINK_FAILED]
    assert len(sink_failures) == len(events)
    assert all(event.type != EventType.RUNTIME_SINK_FAILED for event in recorder.events)


def test_sink_mutation_cannot_rewrite_returned_or_later_sink_events() -> None:
    recorder = _RecordingSink()
    app = CayuApp(event_sinks=[_MutatingSink(), recorder], enable_logging=False)

    events = asyncio.run(_collect_run(app, "sess_sink_mutation"))

    assert events[1].type == EventType.SESSION_STARTED
    assert events[1].payload == {"agent_name": "assistant"}
    assert recorder.events[1].payload == {"agent_name": "assistant"}


def test_model_completed_is_forwarded_to_budget_store_once() -> None:
    budget_store = _RecordingBudgetStore()
    app = CayuApp(budget_store=budget_store, enable_logging=False)

    events = asyncio.run(_collect_run(app, "sess_budget_forwarding"))

    completed = [event for event in events if event.type == EventType.MODEL_COMPLETED]
    assert len(completed) == 1
    assert [event.type for event in budget_store.events] == [EventType.MODEL_COMPLETED]


def test_emit_events_batch_is_persisted_before_fanout() -> None:
    async def scenario() -> tuple[list[Event], list[Event], list[EventRecord], list[bool]]:
        store = InMemorySessionStore()
        observer = _StoreObservingSink(store)
        recorder = _RecordingSink()
        app = CayuApp(
            session_store=store,
            event_sinks=[observer, recorder],
            enable_logging=False,
        )
        await store.create(
            RunRequest(
                agent_name="assistant",
                session_id="sess_batch_emit",
                messages=[Message.text("user", "go")],
            ),
            identity=SessionIdentity(provider_name="fake", model="fake-model"),
        )
        source = [
            Event(
                type="custom.example.first",
                session_id="sess_batch_emit",
                payload={"position": 1},
            ),
            Event(
                type="custom.example.second",
                session_id="sess_batch_emit",
                payload={"position": 2},
            ),
        ]

        emitted = await app.emit_events("sess_batch_emit", source)
        persisted = await store.load_events("sess_batch_emit")
        records = await store.query_events(EventQuery(session_id="sess_batch_emit"))
        return emitted, persisted, records, observer.persisted_before_delivery

    emitted, persisted, records, observed = asyncio.run(scenario())

    assert [event.id for event in emitted] == [
        public_event_id(record.sequence) for record in records
    ]
    assert [event.type for event in emitted] == [event.type for event in persisted]
    assert observed == [True, True]


def test_emit_event_returns_public_identity_while_store_retains_private_authority() -> None:
    async def scenario() -> tuple[Event, EventRecord]:
        store = InMemorySessionStore()
        app = CayuApp(session_store=store, enable_logging=False)
        await store.create(
            RunRequest(
                agent_name="assistant",
                session_id="sess_public_emit",
                messages=[Message.text("user", "go")],
            ),
            identity=SessionIdentity(provider_name="fake", model="fake-model"),
        )
        emitted = await app.emit_event(
            Event(
                type="custom.example",
                session_id="sess_public_emit",
                payload={"safe": True},
            )
        )
        records = await store.query_events(EventQuery(session_id="sess_public_emit"))
        assert len(records) == 1
        return emitted, records[0]

    emitted, record = asyncio.run(scenario())

    assert emitted.id == public_event_id(record.sequence)
    assert emitted.id != record.event.id
