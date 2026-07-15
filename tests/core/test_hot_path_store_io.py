"""Focused tests for incremental usage-tracker store I/O behavior."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator

import cayu.runtime.app as runtime_app_module
from cayu.core import AgentSpec, Event, EventType, Message
from cayu.providers import ModelProvider, ModelRequest, ModelStreamEvent
from cayu.runtime import (
    CayuApp,
    InMemorySessionStore,
    RunLimits,
    RunRequest,
    SessionIdentity,
    SessionStatus,
)
from cayu.runtime.usage import session_usage_summary


class _FakeProvider(ModelProvider):
    name = "fake"

    def __init__(self, events: list[ModelStreamEvent]) -> None:
        self.events = events

    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
        for event in self.events:
            yield event


class _CountingSessionStore(InMemorySessionStore):
    def __init__(self) -> None:
        super().__init__()
        self.load_calls = 0
        self.load_events_calls = 0

    async def load(self, session_id: str):
        self.load_calls += 1
        return await super().load(session_id)

    async def load_events(self, session_id: str) -> list[Event]:
        self.load_events_calls += 1
        return await super().load_events(session_id)


def _register_streaming_agent(store: InMemorySessionStore, delta_count: int) -> CayuApp:
    provider = _FakeProvider(
        [
            *[ModelStreamEvent.text_delta(f"chunk {index} ") for index in range(delta_count)],
            ModelStreamEvent.completed(
                {
                    "finish_reason": "stop",
                    "usage": {"input_tokens": 7, "output_tokens": 4, "total_tokens": 11},
                }
            ),
        ]
    )
    app = CayuApp(session_store=store, enable_logging=False)
    app.register_provider(provider, default=True)
    app.register_agent(AgentSpec(name="assistant", model="fake-model"))
    return app


async def _create_running_session(store: InMemorySessionStore, session_id: str) -> None:
    await store.create(
        RunRequest(
            agent_name="assistant",
            session_id=session_id,
            messages=[Message.text("user", "hello")],
        ),
        identity=SessionIdentity(provider_name="fake", model="fake-model"),
    )
    await store.update_status(session_id, SessionStatus.RUNNING)


def test_streaming_does_not_load_session_per_delta():
    delta_count = 200
    store = _CountingSessionStore()
    app = _register_streaming_agent(store, delta_count)

    async def run() -> list[Event]:
        return [
            event
            async for event in app.run(
                RunRequest(
                    agent_name="assistant",
                    session_id="sess_stream_poll",
                    messages=[Message.text("user", "hello")],
                )
            )
        ]

    events = asyncio.run(run())

    assert events[-1].type == EventType.SESSION_COMPLETED
    delta_events = [event for event in events if event.type == EventType.MODEL_TEXT_DELTA]
    assert len(delta_events) == delta_count
    # Per-delta interrupt checks are throttled; only the phase-boundary checks
    # (and at most a couple of interval expiries on a slow machine) load the
    # session, instead of one load per streamed delta.
    assert store.load_calls <= 20


def test_run_with_limits_does_not_load_full_event_log():
    store = _CountingSessionStore()
    app = _register_streaming_agent(store, delta_count=3)

    async def run() -> list[Event]:
        return [
            event
            async for event in app.run(
                RunRequest(
                    agent_name="assistant",
                    session_id="sess_limits_no_full_load",
                    messages=[Message.text("user", "hello")],
                    limits=RunLimits(max_total_tokens=1000, max_tool_calls=5),
                )
            )
        ]

    events = asyncio.run(run())

    assert events[-1].type == EventType.SESSION_COMPLETED
    # Limit checks tail-query only usage-bearing event types; the full event
    # log (dominated by per-delta stream events) is never loaded.
    assert store.load_events_calls == 0


def test_session_usage_tracker_accumulates_tail_events_incrementally():
    store = InMemorySessionStore()
    app = CayuApp(session_store=store, enable_logging=False)

    def usage_event(step: int) -> Event:
        return Event(
            type=EventType.MODEL_COMPLETED,
            session_id="sess_tracker",
            agent_name="assistant",
            payload={
                "provider_name": "fake",
                "model": "fake-model",
                "usage": {
                    "input_tokens": 10 * step,
                    "output_tokens": step,
                    "total_tokens": 11 * step,
                },
            },
        )

    async def run():
        await _create_running_session(store, "sess_tracker")
        tracker = runtime_app_module._SessionUsageTracker(app, session_id="sess_tracker")

        assert await tracker.usage_events() == []

        await store.append_events(
            "sess_tracker",
            [
                Event(
                    type=EventType.MODEL_TEXT_DELTA,
                    session_id="sess_tracker",
                    payload={"delta": "noise"},
                ),
                usage_event(1),
                Event(
                    type=EventType.TOOL_CALL_STARTED,
                    session_id="sess_tracker",
                    tool_name="echo",
                    payload={"tool_call_id": "call_1"},
                ),
            ],
        )
        first = await tracker.usage_events()
        assert [event.type for event in first] == [
            EventType.MODEL_COMPLETED,
            EventType.TOOL_CALL_STARTED,
        ]

        # No new events: the cached tail is returned as-is.
        assert await tracker.usage_events() is first

        await store.append_events(
            "sess_tracker",
            [
                Event(
                    type=EventType.MODEL_TEXT_DELTA,
                    session_id="sess_tracker",
                    payload={"delta": "more noise"},
                ),
                usage_event(2),
            ],
        )
        second = await tracker.usage_events()
        assert [event.type for event in second] == [
            EventType.MODEL_COMPLETED,
            EventType.TOOL_CALL_STARTED,
            EventType.MODEL_COMPLETED,
        ]

        all_events = await store.load_events("sess_tracker")
        tracked_summary = session_usage_summary("sess_tracker", second)
        full_summary = session_usage_summary("sess_tracker", all_events)
        assert tracked_summary == full_summary
        assert tracked_summary.usage.total_tokens == 33
        assert tracked_summary.tool_calls == 1
        assert tracked_summary.model_steps == 2

    asyncio.run(run())


class _MidRefreshAppendingStore(InMemorySessionStore):
    """Deterministically replays the issue-#101 watermark race.

    On the first ``query_events`` call it appends one ``model.completed`` and
    one ``tool.call.started`` event after computing its result. With the old
    per-type refresh (two queries sharing one watermark), the second query of
    the same refresh returned only the tool event, the shared watermark jumped
    past the model event, and its tokens were lost forever. A single
    ``event_types`` query sees nothing mid-refresh, so the next refresh picks
    both events up.
    """

    def __init__(self) -> None:
        super().__init__()
        self.query_events_calls = 0
        self._raced = False

    async def query_events(self, query=None):
        self.query_events_calls += 1
        records = await super().query_events(query)
        if not self._raced:
            self._raced = True
            await super().append_events(
                "sess_race",
                [
                    Event(
                        type=EventType.MODEL_COMPLETED,
                        session_id="sess_race",
                        agent_name="assistant",
                        payload={
                            "provider_name": "fake",
                            "model": "fake-model",
                            "usage": {
                                "input_tokens": 10,
                                "output_tokens": 1,
                                "total_tokens": 11,
                            },
                        },
                    ),
                    Event(
                        type=EventType.TOOL_CALL_STARTED,
                        session_id="sess_race",
                        tool_name="echo",
                        payload={"tool_call_id": "call_race"},
                    ),
                ],
            )
        return records


def test_session_usage_tracker_never_skips_events_appended_mid_refresh():
    store = _MidRefreshAppendingStore()
    app = CayuApp(session_store=store, enable_logging=False)

    async def run():
        await _create_running_session(store, "sess_race")
        tracker = runtime_app_module._SessionUsageTracker(app, session_id="sess_race")

        # First refresh: the store races an append mid-refresh. The refresh
        # must be ONE store query, and must not advance the watermark past
        # the raced events it never saw.
        assert await tracker.usage_events() == []
        assert store.query_events_calls == 1

        # Second refresh recovers BOTH raced events — the old two-query
        # refresh skipped the model.completed event forever.
        events = await tracker.usage_events()
        assert [event.type for event in events] == [
            EventType.MODEL_COMPLETED,
            EventType.TOOL_CALL_STARTED,
        ]
        assert store.query_events_calls == 2

        summary = session_usage_summary("sess_race", events)
        assert summary.usage.total_tokens == 11
        assert summary.tool_calls == 1

    asyncio.run(run())
