from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta

import pytest

from cayu.budgets.base import InMemoryBudgetStore
from cayu.events import Event, EventType
from cayu.observability.events import InMemoryEventSink
from cayu.runtime._event_writer import RuntimeEventWriter
from cayu.sessions.event_delivery import PersistedEventSideEffectStatus
from cayu.sessions.memory import InMemorySessionStore
from cayu.sessions.records import SessionIdentity
from cayu.sessions.requests import RunRequest


@pytest.mark.parametrize("batch_size", [1, 3])
@pytest.mark.parametrize("initial_state", ["deferred", "failed", "leased"])
def test_recovery_revisits_earlier_candidates_under_continuous_arrivals(
    batch_size: int, initial_state: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def scenario() -> None:
        store = InMemorySessionStore()
        session_id = "continuous-arrivals"
        await store.create(
            RunRequest(agent_name="assistant", session_id=session_id, messages=[]),
            identity=SessionIdentity(provider_name="fake", model="fake-model"),
        )
        now = datetime(2026, 7, 16, tzinfo=UTC)
        monkeypatch.setattr(store, "_ownership_clock", lambda: now)
        earlier = [
            Event(type=EventType.SESSION_COMPLETED, session_id=session_id)
            for _ in range(batch_size)
        ]
        healthy = [Event(type="custom.ready", session_id=session_id) for _ in range(batch_size)]
        await store.append_events(session_id, [*earlier, *healthy])
        if initial_state != "deferred":
            for event in earlier:
                claim = await store.claim_persisted_event_side_effect(
                    session_id=session_id,
                    event_id=event.id,
                    lease_seconds=60,
                )
                assert claim is not None
                if initial_state == "failed":
                    await store.mark_persisted_event_side_effect_failed(
                        claim,
                        error="try later",
                        max_attempts=3,
                        retry_delay_seconds=60,
                    )
        ready = False
        continuations: list[str] = []

        async def continue_parent(claim) -> bool:
            continuations.append(claim.event_id)
            return ready

        sink = InMemoryEventSink()
        writer = RuntimeEventWriter(
            session_store=store,
            budget_store=InMemoryBudgetStore(),
            event_sinks=[sink],
            continue_foreground_parent=continue_parent,
        )
        await writer.recover_persisted_side_effects(limit=batch_size)
        if initial_state == "deferred":
            assert continuations == [event.id for event in earlier]
        else:
            assert continuations == []

        ready = True
        now += timedelta(seconds=61)
        for _ in range(10):
            # Keep every tail page full, even after the earlier candidates
            # become ready or their retry/lease deadline expires.
            await store.append_events(
                session_id,
                [Event(type="custom.arrival", session_id=session_id) for _ in range(batch_size)],
            )
            recovered = await writer.recover_persisted_side_effects(limit=batch_size)
            assert len(recovered) <= batch_size

        for event in [*earlier, *healthy]:
            delivery = await store.get_persisted_event_side_effect_delivery(
                session_id=session_id,
                event_id=event.id,
            )
            assert delivery is not None
            assert delivery.status is PersistedEventSideEffectStatus.DELIVERED
        for event in earlier:
            assert continuations.count(event.id) == (2 if initial_state == "deferred" else 1)
        assert sum(event.type == EventType.SESSION_COMPLETED for event in sink.events) == batch_size

    asyncio.run(scenario())
