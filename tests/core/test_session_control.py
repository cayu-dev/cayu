from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator

import pytest

import cayu.runtime._session_control as session_control_module
from cayu.core import Event, EventType, Message
from cayu.runtime import (
    EventOrder,
    EventQuery,
    EventRecord,
    InMemorySessionStore,
    RunRequest,
    SessionIdentity,
    SessionStatus,
)
from cayu.runtime._session_control import (
    SessionControl,
    SessionInterruptedByRequest,
)


class _CountingSessionStore(InMemorySessionStore):
    def __init__(self) -> None:
        super().__init__()
        self.load_calls = 0
        self.event_queries: list[EventQuery] = []

    async def load(self, session_id: str):
        self.load_calls += 1
        return await super().load(session_id)

    async def query_events(self, query: EventQuery) -> list[EventRecord]:
        self.event_queries.append(query)
        return await super().query_events(query)


async def _create_session(
    store: InMemorySessionStore,
    session_id: str,
    *,
    status: SessionStatus = SessionStatus.RUNNING,
) -> None:
    await store.create(
        RunRequest(
            agent_name="assistant",
            session_id=session_id,
            messages=[Message.text("user", "hello")],
        ),
        identity=SessionIdentity(provider_name="fake", model="fake-model"),
    )
    if status != SessionStatus.PENDING:
        await store.update_status(session_id, status)


def test_stream_interrupt_poll_throttles_reads_and_signal_bypasses_throttle() -> None:
    store = _CountingSessionStore()
    control = SessionControl[object](session_store=store)

    async def scenario() -> None:
        await _create_session(store, "sess_poll_signal", status=SessionStatus.INTERRUPTING)
        store.load_calls = 0
        poll = control.stream_interrupt_poll("sess_poll_signal")

        await poll.raise_if_interrupted()
        assert store.load_calls == 0

        control.signal_interrupt("sess_poll_signal")
        assert control.interrupt_signalled("sess_poll_signal") is True
        with pytest.raises(SessionInterruptedByRequest):
            await poll.raise_if_interrupted()
        assert store.load_calls == 1

        control.discard_interrupt_signal("sess_poll_signal")
        assert control.interrupt_signalled("sess_poll_signal") is False

    asyncio.run(scenario())


def test_stream_interrupt_poll_reads_once_per_expired_check(monkeypatch) -> None:
    monkeypatch.setattr(session_control_module, "STREAM_INTERRUPT_POLL_INTERVAL_S", 0.0)
    store = _CountingSessionStore()
    control = SessionControl[object](session_store=store)

    async def scenario() -> None:
        await _create_session(store, "sess_poll_expired")
        store.load_calls = 0
        poll = control.stream_interrupt_poll("sess_poll_expired")

        await poll.raise_if_interrupted()
        assert store.load_calls == 1

        await store.update_status("sess_poll_expired", SessionStatus.INTERRUPTING)
        with pytest.raises(SessionInterruptedByRequest):
            await poll.raise_if_interrupted()
        assert store.load_calls == 2

    asyncio.run(scenario())


def test_latest_interrupted_event_uses_one_bounded_tail_query() -> None:
    store = _CountingSessionStore()
    control = SessionControl[object](session_store=store)

    async def scenario() -> None:
        await _create_session(store, "sess_latest_interrupted")
        await store.append_events(
            "sess_latest_interrupted",
            [
                Event(
                    type=EventType.SESSION_INTERRUPTED,
                    session_id="sess_latest_interrupted",
                    payload={"reason": "first"},
                ),
                Event(
                    type=EventType.MODEL_TEXT_DELTA,
                    session_id="sess_latest_interrupted",
                    payload={"delta": "noise"},
                ),
                Event(
                    type=EventType.SESSION_INTERRUPTED,
                    session_id="sess_latest_interrupted",
                    payload={"reason": "second"},
                ),
            ],
        )
        store.load_calls = 0

        latest = await control.latest_interrupted_event("sess_latest_interrupted")

        assert latest is not None
        assert latest.payload["reason"] == "second"
        assert store.load_calls == 0
        assert len(store.event_queries) == 1
        query = store.event_queries[0]
        assert query.session_id == "sess_latest_interrupted"
        assert query.event_type == EventType.SESSION_INTERRUPTED
        assert query.order_by == EventOrder.SEQUENCE_DESC
        assert query.limit == 1

        latest.payload["reason"] = "mutated"
        stored = await store.load_events("sess_latest_interrupted")
        assert stored[-1].payload["reason"] == "second"

    asyncio.run(scenario())


def test_latest_interrupted_event_distinguishes_absent_event_and_session() -> None:
    store = _CountingSessionStore()
    control = SessionControl[object](session_store=store)

    async def scenario() -> None:
        await _create_session(store, "sess_no_interrupted_event")
        store.load_calls = 0

        assert await control.latest_interrupted_event("sess_no_interrupted_event") is None
        assert store.load_calls == 1

        with pytest.raises(KeyError):
            await control.latest_interrupted_event("sess_missing")
        assert store.load_calls == 2

    asyncio.run(scenario())


def test_active_run_registry_cancels_other_owners_and_cleans_empty_sessions() -> None:
    control = SessionControl[object](session_store=InMemorySessionStore())

    async def scenario() -> None:
        current_task = asyncio.current_task()
        assert current_task is not None
        release = asyncio.Event()
        other_task = asyncio.create_task(release.wait())
        current_run = control.register_active_task(
            "sess_active",
            current_task,
            task_id="task-current",
            task_started=True,
            task_finished=False,
        )
        other_run = control.register_active_task(
            "sess_active",
            other_task,
            task_id="task-other",
            task_started=False,
            task_finished=False,
        )
        try:
            assert control.active_runs("sess_active") == (current_run, other_run)
            assert control.has_active_tasks("sess_active") is True
            assert control.cancel_active_runs("sess_active") is True
            with pytest.raises(asyncio.CancelledError):
                await other_task
            assert current_task.cancelling() == 0
        finally:
            control.unregister_active_task("sess_active", current_task)
            control.unregister_active_task("sess_active", other_task)
            if not other_task.done():
                release.set()
                await other_task

        assert control.active_runs("sess_active") == ()
        assert control.has_active_tasks("sess_active") is False

    asyncio.run(scenario())


def test_interruption_markers_keep_active_wait_alive_until_terminal_event(
    monkeypatch,
) -> None:
    monkeypatch.setattr(session_control_module, "ACTIVE_INTERRUPTED_EVENT_WAIT_ATTEMPTS", 2)
    monkeypatch.setattr(session_control_module, "ACTIVE_INTERRUPTED_EVENT_WAIT_INTERVAL_S", 0.0)
    store = _CountingSessionStore()
    control = SessionControl[object](session_store=store)

    async def scenario() -> None:
        await _create_session(store, "sess_wait", status=SessionStatus.INTERRUPTING)
        control.begin_interruption_request("sess_wait")
        control.begin_emitting_interrupted("sess_wait")
        try:
            assert control.is_interruption_request_active("sess_wait") is True
            assert control.is_emitting_interrupted("sess_wait") is True
            assert await control.wait_for_active_interrupted_event("sess_wait") is None
            assert len(store.event_queries) == 2
        finally:
            control.end_emitting_interrupted("sess_wait")
            control.end_interruption_request("sess_wait")

        assert control.is_interruption_request_active("sess_wait") is False
        assert control.is_emitting_interrupted("sess_wait") is False

    asyncio.run(scenario())


def test_wait_for_interrupted_event_matches_durable_request_identity() -> None:
    store = InMemorySessionStore()
    control = SessionControl[object](session_store=store)

    async def scenario() -> None:
        await _create_session(store, "sess_wait_identity", status=SessionStatus.INTERRUPTED)
        event = Event(
            type=EventType.SESSION_INTERRUPTED,
            session_id="sess_wait_identity",
            payload={"interruption_request_id": "request-1"},
        )
        await store.append_events("sess_wait_identity", [event])

        matched = await control.wait_for_interrupted_event(
            "sess_wait_identity",
            interruption_request_id="request-1",
        )
        assert matched is not None
        assert matched.id == event.id

        assert (
            await control.wait_for_interrupted_event(
                "sess_wait_identity",
                interruption_request_id="request-2",
            )
            is None
        )

    asyncio.run(scenario())


def test_out_of_band_events_are_copied_and_drained_after_stream_events() -> None:
    control = SessionControl[object](session_store=InMemorySessionStore())

    async def scenario() -> list[Event]:
        current_task = asyncio.current_task()
        assert current_task is not None
        control.register_active_task(
            "sess_oob",
            current_task,
            task_id=None,
            task_started=False,
            task_finished=False,
        )
        queued = Event(
            type="custom.workflow.progress",
            session_id="sess_oob",
            payload={"progress": 1},
        )
        control.queue_out_of_band_event(queued)
        queued.payload["progress"] = 99

        async def source() -> AsyncIterator[Event]:
            yield Event(type=EventType.MODEL_STARTED, session_id="sess_oob")

        try:
            return [
                event
                async for event in control.stream_with_out_of_band_events(
                    "sess_oob",
                    source(),
                )
            ]
        finally:
            control.unregister_active_task("sess_oob", current_task)

    events = asyncio.run(scenario())

    assert [event.type for event in events] == [
        EventType.MODEL_STARTED,
        "custom.workflow.progress",
    ]
    assert events[1].payload == {"progress": 1}
