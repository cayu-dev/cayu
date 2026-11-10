from __future__ import annotations

import asyncio

import pytest

from cayu.budgets.base import InMemoryBudgetStore
from cayu.events import Event, EventType
from cayu.observability.events import InMemoryEventSink
from cayu.runtime._event_writer import RuntimeEventWriter
from cayu.sessions.event_delivery import PersistedEventSideEffectStatus
from cayu.sessions.memory import InMemorySessionStore
from cayu.sessions.records import SessionIdentity
from cayu.sessions.requests import RunRequest


async def _scenario(batch_size: int) -> None:
    store = InMemorySessionStore()
    deferred = []
    for index in range(batch_size):
        session_id = f"deferred-child-{index}"
        await store.create(
            RunRequest(agent_name="assistant", session_id=session_id, messages=[]),
            identity=SessionIdentity(provider_name="fake", model="fake-model"),
        )
        event = Event(type=EventType.SESSION_COMPLETED, session_id=session_id)
        await store.append_event(session_id, event)
        deferred.append(event)
    healthy = Event(type="custom.ready", session_id="ready-session")
    await store.create(
        RunRequest(agent_name="assistant", session_id=healthy.session_id, messages=[]),
        identity=SessionIdentity(provider_name="fake", model="fake-model"),
    )
    await store.append_event(healthy.session_id, healthy)
    sink = InMemoryEventSink()

    async def parent_still_settling(_claim) -> bool:
        # This is the documented false outcome of a foreground continuation
        # while its parent still owns execution; it must not spend attempts.
        return False

    writer = RuntimeEventWriter(
        session_store=store,
        budget_store=InMemoryBudgetStore(),
        event_sinks=[sink],
        continue_foreground_parent=parent_still_settling,
    )
    for _ in range(3):
        await writer.recover_persisted_side_effects(limit=batch_size)

    first = await store.get_persisted_event_side_effect_delivery(
        session_id=deferred[0].session_id, event_id=deferred[0].id
    )
    assert first is not None
    assert (first.status, first.attempts) == (PersistedEventSideEffectStatus.PENDING, 0)
    ready = await store.get_persisted_event_side_effect_delivery(
        session_id=healthy.session_id, event_id=healthy.id
    )
    assert ready is not None
    assert ready.status is PersistedEventSideEffectStatus.DELIVERED
    assert [event.type for event in sink.events] == ["custom.ready"]


@pytest.mark.parametrize("batch_size", [1, 1000])
def test_deferred_page_head_does_not_starve_later_event_recovery(batch_size: int) -> None:
    # 1000 is the batch size used by the optional server recovery loop.
    asyncio.run(_scenario(batch_size))
