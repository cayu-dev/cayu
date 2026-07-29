from __future__ import annotations

import asyncio

import pytest

from cayu.core import AgentSpec, Event, EventType, Message
from cayu.providers import ModelProvider, ModelRequest, ModelStreamEvent
from cayu.runtime import (
    CayuApp,
    EventQuery,
    InMemorySessionStore,
    RunRequest,
    RuntimeHook,
    RuntimeHookContext,
    SessionStatus,
)
from cayu.runtime import sessions as sessions_module


class CompletingProvider(ModelProvider):
    name = "completing"

    def __init__(self) -> None:
        self.requests: list[ModelRequest] = []

    async def stream(self, request: ModelRequest):
        self.requests.append(request)
        yield ModelStreamEvent.text_delta("done")
        yield ModelStreamEvent.completed({"finish_reason": "stop"})


class FailingProvider(ModelProvider):
    name = "failing"

    def __init__(self) -> None:
        self.requests: list[ModelRequest] = []

    async def stream(self, request: ModelRequest):
        self.requests.append(request)
        raise RuntimeError("provider failed")
        yield  # pragma: no cover


class CommitThenLoseInteractionTransitionAcknowledgementStore(InMemorySessionStore):
    def __init__(self) -> None:
        super().__init__()
        self.lost_acknowledgement = False
        self.attempted_event_ids: list[str] = []

    async def publish_interaction_transition(
        self,
        session_id,
        *,
        event,
        from_statuses,
        to_status,
        only_if_no_queued_messages=False,
    ):
        self.attempted_event_ids.append(event.id)
        result = await super().publish_interaction_transition(
            session_id,
            event=event,
            from_statuses=from_statuses,
            to_status=to_status,
            only_if_no_queued_messages=only_if_no_queued_messages,
        )
        if not self.lost_acknowledgement:
            self.lost_acknowledgement = True
            raise ConnectionError("interaction transition acknowledgement lost")
        return result


@pytest.mark.parametrize(
    ("provider", "expected_status", "expected_event_type"),
    [
        (CompletingProvider(), SessionStatus.COMPLETED, EventType.INTERACTION_COMPLETED),
        (FailingProvider(), SessionStatus.FAILED, EventType.INTERACTION_FAILED),
    ],
)
def test_runtime_reconstructs_interaction_transition_after_commit_acknowledgement_loss(
    provider: ModelProvider,
    expected_status: SessionStatus,
    expected_event_type: EventType,
) -> None:
    async def run() -> None:
        store = CommitThenLoseInteractionTransitionAcknowledgementStore()
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_provider(provider)
        app.register_agent(AgentSpec(name="assistant", model="fake-model"))

        events = [
            event
            async for event in app.run(
                RunRequest(
                    agent_name="assistant",
                    session_id=f"sess_{expected_status}",
                    messages=[Message.text("user", "run")],
                )
            )
        ]

        session = await store.load(f"sess_{expected_status}")
        assert session is not None
        assert session.status is expected_status
        assert store.lost_acknowledgement is True
        assert len(store.attempted_event_ids) == 2
        assert len(set(store.attempted_event_ids)) == 1
        durable = await store.load_events(session.id)
        assert sum(event.type == expected_event_type for event in durable) == 1
        assert sum(event.type == expected_event_type for event in events) == 1
        assert len(provider.requests) == 1  # type: ignore[attr-defined]

    asyncio.run(run())


def test_terminal_interaction_closes_attribution_before_session_finalization() -> None:
    class TerminalHook(RuntimeHook):
        async def after_session_completed(self, context: RuntimeHookContext) -> None:
            await context.emit_custom_event(
                "custom.after-session-completed",
                payload={"session_id": context.session.id},
            )

    async def run() -> None:
        session_id = "sess_terminal_interaction_attribution"
        store = InMemorySessionStore()
        app = CayuApp(
            session_store=store,
            runtime_hooks=[TerminalHook()],
            enable_logging=False,
        )
        app.register_provider(CompletingProvider())
        app.register_agent(AgentSpec(name="assistant", model="fake-model"))

        await _collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id=session_id,
                messages=[Message.text("user", "run")],
            ),
        )

        events = await store.load_events(session_id)
        terminal_index = next(
            index
            for index, event in enumerate(events)
            if event.type == EventType.INTERACTION_COMPLETED
        )
        interaction_id = events[terminal_index].interaction_id
        assert interaction_id is not None
        assert all(event.interaction_id is None for event in events[terminal_index + 1 :])

        turn_completed = next(event for event in events if event.type == EventType.TURN_COMPLETED)
        assert turn_completed.payload["interaction_ids"] == [interaction_id]

    asyncio.run(run())


def test_run_stream_carries_interaction_context_across_consumer_tasks() -> None:
    async def run() -> None:
        session_id = "sess_cross_task_interaction_context"
        store = InMemorySessionStore()
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_provider(CompletingProvider())
        app.register_agent(AgentSpec(name="assistant", model="fake-model"))

        stream = app.run(
            RunRequest(
                agent_name="assistant",
                session_id=session_id,
                messages=[Message.text("user", "run")],
            )
        )
        events: list[Event] = []
        while True:
            try:
                events.append(await asyncio.create_task(anext(stream)))
            except StopAsyncIteration:
                break
            assert sessions_module._current_session_interaction_id(session_id) is None

        interaction_id = next(
            event.interaction_id for event in events if event.type == EventType.INTERACTION_STARTED
        )
        assert interaction_id is not None
        for event_type in (
            EventType.MODEL_STARTED,
            EventType.MODEL_TEXT_DELTA,
            EventType.MODEL_COMPLETED,
            EventType.INTERACTION_COMPLETED,
        ):
            matching = [event for event in events if event.type == event_type]
            assert matching
            assert {event.interaction_id for event in matching} == {interaction_id}
        assert sessions_module._current_session_interaction_id(session_id) is None

    asyncio.run(run())


def test_in_memory_interaction_query_uses_interaction_scoped_candidates() -> None:
    async def run() -> None:
        session_id = "sess_interaction_candidate_index"
        store = InMemorySessionStore()
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_provider(CompletingProvider())
        app.register_agent(AgentSpec(name="assistant", model="fake-model"))

        await _collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id=session_id,
                messages=[Message.text("user", "run")],
            ),
        )
        interaction_id = next(
            event.interaction_id
            for event in await store.load_events(session_id)
            if event.type == EventType.INTERACTION_STARTED
        )
        assert interaction_id is not None
        await store.append_events(
            session_id,
            [
                Event(
                    id=f"evt_unrelated_{index}",
                    type="custom.unrelated",
                    session_id=session_id,
                    interaction_id="interaction-unrelated",
                )
                for index in range(32)
            ],
        )

        query = EventQuery(session_id=session_id, interaction_id=interaction_id)
        candidates = store._query_candidate_records(query, frozenset())
        records = await store.query_events(query)

        assert candidates
        assert len(candidates) == len(records)
        assert all(record.event.interaction_id == interaction_id for record in candidates)

    asyncio.run(run())


async def _collect_events(app: CayuApp, request: RunRequest) -> list:
    return [event async for event in app.run(request)]
