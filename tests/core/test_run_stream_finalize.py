"""An abandoned run/resume event stream must finalize — not strand — the session.

When a consumer closes the event stream mid-run (client disconnect, abandoned async
generator), GeneratorExit reaches the run generator. The runtime must transition the
still-RUNNING session to INTERRUPTED and persist a terminal event, so the session
stays observable and resumable instead of being stranded in RUNNING forever.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from typing import NamedTuple

from cayu.core import AgentSpec, Message
from cayu.core.events import Event, EventType
from cayu.providers import ModelProvider, ModelRequest, ModelStreamEvent
from cayu.runtime import (
    CayuApp,
    InMemorySessionStore,
    ResumeRequest,
    RunRequest,
    SessionIdentity,
    SessionStatus,
)


class FakeProvider(ModelProvider):
    name = "fake"

    def __init__(self, batches: list[list[ModelStreamEvent]]) -> None:
        self.event_batches = batches
        self.requests: list[ModelRequest] = []

    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
        batch_index = len(self.requests)
        self.requests.append(request)
        if batch_index >= len(self.event_batches):
            raise AssertionError(f"No fake provider event batch for request {batch_index}")
        for event in self.event_batches[batch_index]:
            yield event


class Harness(NamedTuple):
    app: CayuApp
    store: InMemorySessionStore
    provider: FakeProvider


def _batch(text: str) -> list[ModelStreamEvent]:
    return [
        ModelStreamEvent.text_delta(text),
        ModelStreamEvent.completed({"finish_reason": "stop"}),
    ]


def _build(batches: list[list[ModelStreamEvent]]) -> Harness:
    store = InMemorySessionStore()
    provider = FakeProvider(batches)
    app = CayuApp(session_store=store, enable_logging=False)
    app.register_provider(provider, default=True)
    app.register_agent(AgentSpec(name="assistant", model="fake-model"))
    return Harness(app, store, provider)


def _abandoned_terminal_event(events: list[Event]) -> Event:
    interrupted = [event for event in events if event.type == EventType.SESSION_INTERRUPTED]
    assert len(interrupted) == 1
    return interrupted[0]


def _assert_turn_completed_before_abandoned_terminal(events: list[Event]) -> None:
    event_types = [event.type for event in events]
    assert event_types.count(EventType.TURN_COMPLETED) == 1
    assert event_types.index(EventType.TURN_COMPLETED) < event_types.index(
        EventType.SESSION_INTERRUPTED
    )
    turn = next(event for event in events if event.type == EventType.TURN_COMPLETED)
    assert turn.payload["status"] == "interrupted"


def test_abandoned_run_stream_finalizes_running_session() -> None:
    # The first provider batch is consumed by the post-abandon resume, proving the
    # abandoned run never reached the model and the session stayed resumable.
    h = _build([_batch("answer after abandonment")])

    async def scenario() -> tuple[Event, list[Event]]:
        stream = h.app.run(
            RunRequest(
                agent_name="assistant",
                session_id="sess_abandoned_run",
                messages=[Message.text("user", "hello")],
            )
        )
        first_event = await anext(stream)
        assert first_event.type == EventType.SESSION_STARTED
        # The consumer walks away mid-run (e.g. SSE client disconnect).
        await stream.aclose()
        events = await h.store.load_events("sess_abandoned_run")
        return first_event, events

    _, events = asyncio.run(scenario())

    session = asyncio.run(h.store.load("sess_abandoned_run"))
    assert session is not None
    assert session.status == SessionStatus.INTERRUPTED
    _assert_turn_completed_before_abandoned_terminal(events)
    terminal = _abandoned_terminal_event(events)
    assert terminal.payload == {
        "interruption_type": "runtime_interrupted",
        "reason": "event_stream_closed",
        "abandoned": True,
    }

    # The finalized session is resumable — it was not stranded in RUNNING.
    async def resume() -> None:
        async for _ in h.app.resume(
            ResumeRequest(
                session_id="sess_abandoned_run",
                messages=[Message.text("user", "continue")],
            )
        ):
            pass

    asyncio.run(resume())
    resumed = asyncio.run(h.store.load("sess_abandoned_run"))
    assert resumed is not None
    assert resumed.status == SessionStatus.COMPLETED
    assert len(h.provider.requests) == 1


def test_abandoned_resume_stream_finalizes_running_session() -> None:
    h = _build([_batch("first answer")])

    async def scenario() -> None:
        async for _ in h.app.run(
            RunRequest(
                agent_name="assistant",
                session_id="sess_abandoned_resume",
                messages=[Message.text("user", "hello")],
            )
        ):
            pass
        stream = h.app.resume(
            ResumeRequest(
                session_id="sess_abandoned_resume",
                messages=[Message.text("user", "continue")],
            )
        )
        first_event = await anext(stream)
        assert first_event.type == EventType.SESSION_RESUMED
        await stream.aclose()

    asyncio.run(scenario())

    session = asyncio.run(h.store.load("sess_abandoned_resume"))
    assert session is not None
    assert session.status == SessionStatus.INTERRUPTED
    events = asyncio.run(h.store.load_events("sess_abandoned_resume"))
    event_types = [event.type for event in events]
    assert event_types.count(EventType.TURN_COMPLETED) == 2
    assert event_types[-2:] == [EventType.TURN_COMPLETED, EventType.SESSION_INTERRUPTED]
    assert events[-2].payload["status"] == "interrupted"
    terminal = _abandoned_terminal_event(events)
    assert terminal.payload["reason"] == "event_stream_closed"
    assert terminal.payload["abandoned"] is True


def test_finalize_abandoned_session_by_id_finalizes_and_is_idempotent() -> None:
    # The shared finalizer that the run-factory-window and tool-approval GeneratorExit
    # guards call to close a session stranded RUNNING before _run_session's own finalizer.
    h = _build([_batch("x")])

    async def scenario() -> None:
        await h.store.create(
            RunRequest(
                agent_name="assistant",
                session_id="sess_strand",
                messages=[Message.text("user", "hi")],
            ),
            identity=SessionIdentity(provider_name="fake", model="fake-model"),
        )
        await h.store.transition_status(
            "sess_strand",
            from_statuses={SessionStatus.PENDING},
            to_status=SessionStatus.RUNNING,
        )

        await h.app._finalize_abandoned_session_by_id("sess_strand")
        first = await h.store.load("sess_strand")
        assert first is not None and first.status == SessionStatus.INTERRUPTED

        # Idempotent: a second call (e.g. also reached by _run_session's finalizer) no-ops.
        await h.app._finalize_abandoned_session_by_id("sess_strand")
        second = await h.store.load("sess_strand")
        assert second is not None and second.status == SessionStatus.INTERRUPTED

        # Unknown session id is a safe no-op.
        await h.app._finalize_abandoned_session_by_id("does-not-exist")

        events = await h.store.load_events("sess_strand")
        interrupted = [e for e in events if e.type == EventType.SESSION_INTERRUPTED]
        assert len(interrupted) == 1
        assert interrupted[0].payload["abandoned"] is True

    asyncio.run(scenario())


def test_completed_run_stream_close_is_a_no_op() -> None:
    # Closing an already-finished stream must not rewrite the terminal status.
    h = _build([_batch("first answer")])

    async def scenario() -> None:
        stream = h.app.run(
            RunRequest(
                agent_name="assistant",
                session_id="sess_completed_close",
                messages=[Message.text("user", "hello")],
            )
        )
        async for _ in stream:
            pass
        await stream.aclose()

    asyncio.run(scenario())

    session = asyncio.run(h.store.load("sess_completed_close"))
    assert session is not None
    assert session.status == SessionStatus.COMPLETED
    events = asyncio.run(h.store.load_events("sess_completed_close"))
    assert not [event for event in events if event.type == EventType.SESSION_INTERRUPTED]
