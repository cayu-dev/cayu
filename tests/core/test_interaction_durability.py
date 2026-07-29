from __future__ import annotations

import asyncio

import pytest

from cayu.core import AgentSpec, EventType, Message
from cayu.providers import ModelProvider, ModelRequest, ModelStreamEvent
from cayu.runtime import CayuApp, InMemorySessionStore, RunRequest, SessionStatus


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
