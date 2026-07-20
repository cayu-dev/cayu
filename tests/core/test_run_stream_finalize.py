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


class RecordingReleaseStore(InMemorySessionStore):
    def __init__(self) -> None:
        super().__init__()
        self.release_calls: dict[str, int] = {}

    async def release_run_fence(self, session_id: str) -> None:
        self.release_calls[session_id] = self.release_calls.get(session_id, 0) + 1
        await super().release_run_fence(session_id)


class Harness(NamedTuple):
    app: CayuApp
    store: RecordingReleaseStore
    provider: FakeProvider


def _batch(text: str) -> list[ModelStreamEvent]:
    return [
        ModelStreamEvent.text_delta(text),
        ModelStreamEvent.completed({"finish_reason": "stop"}),
    ]


def _build(batches: list[list[ModelStreamEvent]]) -> Harness:
    store = RecordingReleaseStore()
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
    assert h.store.release_calls["sess_abandoned_run"] == 1

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

        try:
            await h.app._recovery_coordinator.finalize_abandoned_session_by_id("sess_strand")
            first = await h.store.load("sess_strand")
            assert first is not None and first.status == SessionStatus.INTERRUPTED

            # Idempotent: a second call (e.g. also reached by _run_session's finalizer) no-ops.
            await h.app._recovery_coordinator.finalize_abandoned_session_by_id("sess_strand")
            second = await h.store.load("sess_strand")
            assert second is not None and second.status == SessionStatus.INTERRUPTED

            # Unknown session id is a safe no-op.
            await h.app._recovery_coordinator.finalize_abandoned_session_by_id("does-not-exist")

            events = await h.store.load_events("sess_strand")
            interrupted = [e for e in events if e.type == EventType.SESSION_INTERRUPTED]
            assert len(interrupted) == 1
            assert interrupted[0].payload["abandoned"] is True
        finally:
            await h.store.release_run_fence("sess_strand")

    asyncio.run(scenario())


def test_abandoned_session_contains_cancellation_group_from_terminal_cleanup() -> None:
    h = _build([_batch("x")])

    async def scenario() -> None:
        await h.store.create(
            RunRequest(
                agent_name="assistant",
                session_id="sess_grouped_abandonment",
                messages=[Message.text("user", "hi")],
            ),
            identity=SessionIdentity(provider_name="fake", model="fake-model"),
        )

        async def fail_terminal(_request):  # type: ignore[no-untyped-def]
            raise BaseExceptionGroup(
                "terminal cleanup cancelled and failed",
                [asyncio.CancelledError(), RuntimeError("cleanup failed")],
            )
            yield  # pragma: no cover

        h.app._recovery_coordinator._emit_terminal_event_with_hooks = fail_terminal
        await h.app._recovery_coordinator.finalize_abandoned_session_by_id(
            "sess_grouped_abandonment"
        )
        session = await h.store.load("sess_grouped_abandonment")
        assert session is not None
        assert session.status is SessionStatus.INTERRUPTED

    asyncio.run(scenario())


def test_finalize_abandoned_session_does_not_require_registered_environment() -> None:
    h = _build([_batch("x")])

    async def scenario() -> None:
        await h.store.create(
            RunRequest(
                agent_name="assistant",
                session_id="sess_missing_environment",
                environment_name="retired-environment",
                messages=[Message.text("user", "hi")],
            ),
            identity=SessionIdentity(provider_name="fake", model="fake-model"),
        )
        await h.store.transition_status(
            "sess_missing_environment",
            from_statuses={SessionStatus.PENDING},
            to_status=SessionStatus.RUNNING,
        )

        try:
            await h.app._recovery_coordinator.finalize_abandoned_session_by_id(
                "sess_missing_environment"
            )
            session = await h.store.load("sess_missing_environment")
            assert session is not None
            assert session.status == SessionStatus.INTERRUPTED
            terminal = _abandoned_terminal_event(
                await h.store.load_events("sess_missing_environment")
            )
            assert terminal.payload["reason"] == "event_stream_closed"
            assert terminal.environment_name == "retired-environment"
        finally:
            await h.store.release_run_fence("sess_missing_environment")

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
