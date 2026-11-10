from __future__ import annotations

import asyncio

import pytest

from cayu.applications import CayuApp
from cayu.events import Event
from cayu.observability.events import InMemoryEventSink
from cayu.sessions.event_delivery import PersistedEventSideEffectStatus
from cayu.sessions.memory import InMemorySessionStore
from cayu.sessions.records import SessionIdentity
from cayu.sessions.requests import RunRequest


@pytest.mark.parametrize(
    "payload",
    [
        {"tool_call_id": " "},
        {"parent_session_id": "\t"},
        {"model_attempt_id": "\n"},
        {"auxiliary_inference": {"tool_call_id": " "}},
    ],
)
def test_custom_payload_text_is_not_required_to_be_private_sink_authority(payload) -> None:
    async def scenario() -> None:
        store = InMemorySessionStore()
        session_id = "custom-payload-text"
        await store.create(
            RunRequest(agent_name="assistant", session_id=session_id, messages=[]),
            identity=SessionIdentity(provider_name="fake", model="fake-model"),
        )
        sink = InMemoryEventSink()
        app = CayuApp(session_store=store, event_sinks=[sink], enable_logging=False)
        event = Event(type="custom.observation", session_id=session_id, payload=payload)
        try:
            emitted = await app.emit_events(session_id, [event])
            assert emitted[0].payload == payload
            assert sink.events[0].payload == payload
            delivery = await store.get_persisted_event_side_effect_delivery(
                session_id=session_id, event_id=event.id
            )
            assert delivery is not None
            assert delivery.status is PersistedEventSideEffectStatus.DELIVERED
        finally:
            await app.aclose()

    asyncio.run(scenario())
