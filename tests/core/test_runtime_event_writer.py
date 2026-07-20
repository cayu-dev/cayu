from __future__ import annotations

import asyncio
import logging
from datetime import UTC, datetime, timedelta

import pytest

from cayu.core import Event, EventType, Message
from cayu.runtime import (
    CayuApp,
    InMemorySessionStore,
    PersistedEventSideEffectClaimLost,
    PersistedEventSideEffectStatus,
    RunRequest,
    SessionIdentity,
)
from cayu.runtime._event_writer import RuntimeEventWriter
from cayu.runtime.budgets import BudgetWindow, InMemoryBudgetStore
from cayu.runtime.event_sinks import EventSink
from cayu.runtime.sessions import EventQuery, EventRecord


class _RecordingSink(EventSink):
    def __init__(self) -> None:
        self.events: list[Event] = []

    async def emit(self, event: Event) -> None:
        self.events.append(event)


class _FailingSink(EventSink):
    async def emit(self, event: Event) -> None:
        raise RuntimeError("sink unavailable")


class _FlakySink(EventSink):
    def __init__(self, *, failures: int) -> None:
        self._failures = failures
        self.events: list[Event] = []

    async def emit(self, event: Event) -> None:
        if self._failures:
            self._failures -= 1
            raise RuntimeError("sink unavailable")
        self.events.append(event.model_copy(deep=True))


class _ConcurrentClaimSessionStore(InMemorySessionStore):
    async def append_event(self, session_id: str, event: Event) -> None:
        await super().append_event(session_id, event)
        claim = await self.claim_persisted_event_side_effect(
            session_id=session_id,
            event_id=event.id,
        )
        assert claim is not None


class _PendingSnapshotSessionStore(InMemorySessionStore):
    def __init__(self) -> None:
        super().__init__()
        self._decline_next_exact_claim = True

    async def claim_persisted_event_side_effect(
        self,
        *,
        session_id=None,
        event_id=None,
        lease_seconds=300.0,
    ):
        if session_id is not None and self._decline_next_exact_claim:
            self._decline_next_exact_claim = False
            return None
        return await super().claim_persisted_event_side_effect(
            session_id=session_id,
            event_id=event_id,
            lease_seconds=lease_seconds,
        )


class _LostAcknowledgementSessionStore(InMemorySessionStore):
    async def mark_persisted_event_side_effect_delivered(self, claim):
        raise PersistedEventSideEffectClaimLost("replacement worker owns the claim")


class _BrokenAcknowledgementSessionStore(InMemorySessionStore):
    def __init__(self) -> None:
        super().__init__()
        self._ack_failures_remaining = 1

    async def claim_persisted_event_side_effect(
        self,
        *,
        session_id=None,
        event_id=None,
        lease_seconds=300.0,
    ):
        return await super().claim_persisted_event_side_effect(
            session_id=session_id,
            event_id=event_id,
            lease_seconds=min(lease_seconds, 0.001),
        )

    async def mark_persisted_event_side_effect_delivered(self, claim):
        if self._ack_failures_remaining:
            self._ack_failures_remaining -= 1
            raise ConnectionError("handoff store unavailable")
        return await super().mark_persisted_event_side_effect_delivered(claim)


class _LostFailureAcknowledgementSessionStore(InMemorySessionStore):
    async def mark_persisted_event_side_effect_failed(
        self,
        claim,
        *,
        error,
        max_attempts,
        retry_delay_seconds,
    ):
        raise PersistedEventSideEffectClaimLost("replacement worker owns the claim")


class _BrokenFailureBookkeepingSessionStore(InMemorySessionStore):
    async def mark_persisted_event_side_effect_failed(
        self,
        claim,
        *,
        error,
        max_attempts,
        retry_delay_seconds,
    ):
        raise RuntimeError("handoff store unavailable")


class _BrokenSinkDiagnosticSessionStore(InMemorySessionStore):
    async def append_event(self, session_id: str, event: Event) -> None:
        if event.type == EventType.RUNTIME_SINK_FAILED:
            raise RuntimeError("diagnostic store unavailable")
        await super().append_event(session_id, event)


class _FailingBudgetStore(InMemoryBudgetStore):
    def __init__(self, *, fail_event_id: str | None = None) -> None:
        super().__init__()
        self._fail_event_id = fail_event_id

    async def append_event(self, event: Event) -> None:
        if self._fail_event_id is None or event.id == self._fail_event_id:
            raise RuntimeError("budget unavailable")
        await super().append_event(event)


async def _session_store(session_id: str) -> InMemorySessionStore:
    store = InMemorySessionStore()
    await store.create(
        RunRequest(
            agent_name="assistant",
            session_id=session_id,
            messages=[Message.text("user", "go")],
        ),
        identity=SessionIdentity(provider_name="fake", model="fake-model"),
    )
    return store


def test_emit_persists_forwards_cost_event_and_fans_out() -> None:
    async def scenario() -> tuple[list[Event], list[Event], list[Event], Event]:
        store = await _session_store("writer_single")
        budget_store = InMemoryBudgetStore()
        sink = _RecordingSink()
        writer = RuntimeEventWriter(
            session_store=store,
            budget_store=budget_store,
            event_sinks=[sink],
        )
        event = Event(type=EventType.MODEL_COMPLETED, session_id="writer_single")

        emitted = await writer.emit(event)
        persisted = await store.load_events("writer_single")
        budget_events = await budget_store.load_events_for_budget(
            scope="app",
            key=None,
            window=BudgetWindow.all_time(),
        )
        return persisted, budget_events, sink.events, emitted

    persisted, budget_events, sink_events, emitted = asyncio.run(scenario())

    assert [event.id for event in persisted] == [emitted.id]
    assert [event.id for event in budget_events] == [emitted.id]
    assert [event.id for event in sink_events] == [emitted.id]


def test_persist_leaves_side_effect_delivery_for_recovery() -> None:
    async def scenario() -> tuple[list[Event], list[Event], list[Event]]:
        store = await _session_store("writer_persist_only")
        sink = _RecordingSink()
        writer = RuntimeEventWriter(
            session_store=store,
            budget_store=InMemoryBudgetStore(),
            event_sinks=[sink],
        )
        event = Event(
            type=EventType.ENVIRONMENT_BINDING_FINALIZE_FAILED,
            session_id="writer_persist_only",
        )

        persisted = await writer.persist(event)
        before_recovery = list(sink.events)
        recovered = await writer.recover_persisted_side_effects()
        return [persisted], before_recovery, recovered

    persisted, before_recovery, recovered = asyncio.run(scenario())

    assert before_recovery == []
    assert [event.id for event in recovered] == [persisted[0].id]


def test_is_persisted_uses_bounded_event_id_query() -> None:
    class _QueryTrackingStore(InMemorySessionStore):
        def __init__(self) -> None:
            super().__init__()
            self.queries: list[EventQuery | None] = []

        async def load_events(self, session_id: str) -> list[Event]:
            raise AssertionError("is_persisted must not load the complete session history")

        async def query_events(self, query: EventQuery | None = None) -> list[EventRecord]:
            self.queries.append(query)
            return await super().query_events(query)

    async def scenario() -> tuple[bool, bool, list[EventQuery | None], str, str]:
        store = _QueryTrackingStore()
        await store.create(
            RunRequest(
                agent_name="assistant",
                session_id="writer_reconciliation",
                messages=[Message.text("user", "go")],
            ),
            identity=SessionIdentity(provider_name="fake", model="fake-model"),
        )
        writer = RuntimeEventWriter(
            session_store=store,
            budget_store=InMemoryBudgetStore(),
            event_sinks=[],
        )
        persisted = Event(type=EventType.SESSION_STARTED, session_id="writer_reconciliation")
        await store.append_event(persisted.session_id, persisted)
        missing = Event(type=EventType.SESSION_STARTED, session_id="writer_reconciliation")
        return (
            await writer.is_persisted(persisted),
            await writer.is_persisted(missing),
            store.queries,
            persisted.id,
            missing.id,
        )

    persisted, missing, queries, persisted_id, missing_id = asyncio.run(scenario())

    assert persisted is True
    assert missing is False
    assert queries == [
        EventQuery(session_id="writer_reconciliation", event_id=persisted_id, limit=1),
        EventQuery(session_id="writer_reconciliation", event_id=missing_id, limit=1),
    ]


def test_recover_persisted_side_effects_delivers_once_without_replaying_origin() -> None:
    async def scenario() -> tuple[list[Event], list[Event], list[Event], list[Event]]:
        store = await _session_store("writer_recovery")
        event = Event(type=EventType.MODEL_COMPLETED, session_id="writer_recovery")
        await store.append_event(event.session_id, event)

        budget_store = InMemoryBudgetStore()
        sink = _RecordingSink()
        writer = RuntimeEventWriter(
            session_store=store,
            budget_store=budget_store,
            event_sinks=[sink],
        )

        first = await writer.recover_persisted_side_effects()
        second = await writer.recover_persisted_side_effects()
        budget_events = await budget_store.load_events_for_budget(
            scope="app",
            key=None,
            window=BudgetWindow.all_time(),
        )
        return first, second, budget_events, sink.events

    first, second, budget_events, sink_events = asyncio.run(scenario())

    assert [event.type for event in first] == [EventType.MODEL_COMPLETED]
    assert second == []
    assert [event.id for event in budget_events] == [first[0].id]
    assert [event.id for event in sink_events] == [first[0].id]


def test_emit_closes_its_durable_side_effect_handoff() -> None:
    async def scenario() -> tuple[list[Event], list[Event], list[Event]]:
        store = await _session_store("writer_emit_handoff")
        budget_store = InMemoryBudgetStore()
        sink = _RecordingSink()
        writer = RuntimeEventWriter(
            session_store=store,
            budget_store=budget_store,
            event_sinks=[sink],
        )

        emitted = await writer.emit(
            Event(type=EventType.MODEL_COMPLETED, session_id="writer_emit_handoff")
        )
        recovered = await writer.recover_persisted_side_effects()
        budget_events = await budget_store.load_events_for_budget(
            scope="app",
            key=None,
            window=BudgetWindow.all_time(),
        )
        return [emitted, *recovered], budget_events, sink.events

    observed, budget_events, sink_events = asyncio.run(scenario())

    assert len(observed) == 1
    assert [event.id for event in budget_events] == [observed[0].id]
    assert [event.id for event in sink_events] == [observed[0].id]


def test_emit_does_not_fail_when_recovery_worker_wins_the_claim_race() -> None:
    async def scenario():
        store = _ConcurrentClaimSessionStore()
        await store.create(
            RunRequest(
                agent_name="assistant",
                session_id="writer_claim_race",
                messages=[Message.text("user", "go")],
            ),
            identity=SessionIdentity(provider_name="fake", model="fake-model"),
        )
        budget_store = InMemoryBudgetStore()
        writer = RuntimeEventWriter(
            session_store=store,
            budget_store=budget_store,
            event_sinks=[],
        )

        emitted = await writer.emit(
            Event(type=EventType.MODEL_COMPLETED, session_id="writer_claim_race")
        )
        budget_events = await budget_store.load_events_for_budget(
            scope="app",
            key=None,
            window=BudgetWindow.all_time(),
        )
        return emitted, budget_events, await store.list_persisted_event_side_effect_deliveries()

    emitted, budget_events, deliveries = asyncio.run(scenario())

    assert [event.id for event in budget_events] == [emitted.id]
    assert deliveries[0].event_id == emitted.id
    assert deliveries[0].status is PersistedEventSideEffectStatus.LEASED


def test_emit_tolerates_budget_fallback_failure_when_recovery_worker_owns_claim() -> None:
    async def scenario():
        store = _ConcurrentClaimSessionStore()
        await store.create(
            RunRequest(
                agent_name="assistant",
                session_id="writer_claim_race_budget_failure",
                messages=[Message.text("user", "go")],
            ),
            identity=SessionIdentity(provider_name="fake", model="fake-model"),
        )
        writer = RuntimeEventWriter(
            session_store=store,
            budget_store=_FailingBudgetStore(),
            event_sinks=[],
        )

        emitted = await writer.emit(
            Event(
                type=EventType.MODEL_COMPLETED,
                session_id="writer_claim_race_budget_failure",
            )
        )
        delivery = await store.get_persisted_event_side_effect_delivery(
            session_id=emitted.session_id,
            event_id=emitted.id,
        )
        return emitted, delivery

    emitted, delivery = asyncio.run(scenario())

    assert delivery is not None
    assert delivery.event_id == emitted.id
    assert delivery.status is PersistedEventSideEffectStatus.LEASED


def test_emit_tolerates_pending_snapshot_and_leaves_sink_delivery_recoverable() -> None:
    async def scenario():
        store = _PendingSnapshotSessionStore()
        await store.create(
            RunRequest(
                agent_name="assistant",
                session_id="writer_pending_snapshot",
                messages=[Message.text("user", "go")],
            ),
            identity=SessionIdentity(provider_name="fake", model="fake-model"),
        )
        budget_store = InMemoryBudgetStore()
        sink = _RecordingSink()
        writer = RuntimeEventWriter(
            session_store=store,
            budget_store=budget_store,
            event_sinks=[sink],
        )

        emitted = await writer.emit(
            Event(type=EventType.MODEL_COMPLETED, session_id="writer_pending_snapshot")
        )
        pending = await store.get_persisted_event_side_effect_delivery(
            session_id=emitted.session_id,
            event_id=emitted.id,
        )
        assert sink.events == []
        recovered = await writer.recover_persisted_side_effects()
        delivered = await store.get_persisted_event_side_effect_delivery(
            session_id=emitted.session_id,
            event_id=emitted.id,
        )
        budget_events = await budget_store.load_events_for_budget(
            scope="app",
            key=None,
            window=BudgetWindow.all_time(),
        )
        return emitted, pending, recovered, delivered, budget_events, sink.events

    emitted, pending, recovered, delivered, budget_events, sink_events = asyncio.run(scenario())

    assert pending is not None
    assert pending.status is PersistedEventSideEffectStatus.PENDING
    assert [event.id for event in recovered] == [emitted.id]
    assert delivered is not None
    assert delivered.status is PersistedEventSideEffectStatus.DELIVERED
    assert [event.id for event in budget_events] == [emitted.id]
    assert [event.id for event in sink_events] == [emitted.id]


def test_fan_out_does_not_resurrect_ineligible_budget_deliveries() -> None:
    async def scenario():
        store = await _session_store("writer_ineligible_budget")
        events = [
            Event(
                type=EventType.MODEL_COMPLETED,
                session_id="writer_ineligible_budget",
                payload={"state": state},
            )
            for state in ("failed", "delivered", "dead_lettered")
        ]
        await store.append_events("writer_ineligible_budget", events)
        claims = []
        for event in events:
            claim = await store.claim_persisted_event_side_effect(
                session_id=event.session_id,
                event_id=event.id,
            )
            assert claim is not None
            claims.append(claim)
        await store.mark_persisted_event_side_effect_failed(
            claims[0],
            error="try later",
            max_attempts=3,
            retry_delay_seconds=60,
        )
        await store.mark_persisted_event_side_effect_delivered(claims[1])
        await store.mark_persisted_event_side_effect_failed(
            claims[2],
            error="permanent failure",
            max_attempts=1,
            retry_delay_seconds=0,
        )
        budget_store = InMemoryBudgetStore()
        sink = _RecordingSink()
        writer = RuntimeEventWriter(
            session_store=store,
            budget_store=budget_store,
            event_sinks=[sink],
        )

        await writer.fan_out_persisted(events)
        budget_events = await budget_store.load_events_for_budget(
            scope="app",
            key=None,
            window=BudgetWindow.all_time(),
        )
        return budget_events, sink.events

    budget_events, sink_events = asyncio.run(scenario())

    assert budget_events == []
    assert sink_events == []


def test_emit_tolerates_acknowledgement_after_claim_is_replaced() -> None:
    async def scenario():
        store = _LostAcknowledgementSessionStore()
        await store.create(
            RunRequest(
                agent_name="assistant",
                session_id="writer_lost_ack",
                messages=[Message.text("user", "go")],
            ),
            identity=SessionIdentity(provider_name="fake", model="fake-model"),
        )
        sink = _RecordingSink()
        writer = RuntimeEventWriter(
            session_store=store,
            budget_store=InMemoryBudgetStore(),
            event_sinks=[sink],
        )

        emitted = await writer.emit(
            Event(type=EventType.SESSION_STARTED, session_id="writer_lost_ack")
        )
        return emitted, sink.events

    emitted, sink_events = asyncio.run(scenario())

    assert [event.id for event in sink_events] == [emitted.id]


def test_emit_tolerates_delivery_acknowledgement_failure(
    caplog: pytest.LogCaptureFixture,
) -> None:
    async def scenario():
        store = _BrokenAcknowledgementSessionStore()
        await store.create(
            RunRequest(
                agent_name="assistant",
                session_id="writer_broken_ack",
                messages=[Message.text("user", "go")],
            ),
            identity=SessionIdentity(provider_name="fake", model="fake-model"),
        )
        sink = _RecordingSink()
        writer = RuntimeEventWriter(
            session_store=store,
            budget_store=InMemoryBudgetStore(),
            event_sinks=[sink],
        )

        emitted = await writer.emit(
            Event(type=EventType.SESSION_STARTED, session_id="writer_broken_ack")
        )
        delivery = await store.get_persisted_event_side_effect_delivery(
            session_id=emitted.session_id,
            event_id=emitted.id,
        )
        await asyncio.sleep(0.01)
        recovered = await writer.recover_persisted_side_effects()
        delivered = await store.get_persisted_event_side_effect_delivery(
            session_id=emitted.session_id,
            event_id=emitted.id,
        )
        return emitted, delivery, recovered, delivered, sink.events

    with caplog.at_level(logging.ERROR, logger="cayu.runtime._event_writer"):
        emitted, leased, recovered, delivered, sink_events = asyncio.run(scenario())

    assert [event.id for event in sink_events] == [emitted.id, emitted.id]
    assert leased is not None
    assert leased.status is PersistedEventSideEffectStatus.LEASED
    assert [event.id for event in recovered] == [emitted.id]
    assert delivered is not None
    assert delivered.status is PersistedEventSideEffectStatus.DELIVERED
    assert "delivery acknowledgement failed" in caplog.text
    assert "ConnectionError" in caplog.text


def test_in_memory_side_effect_deadlines_start_after_lock_acquisition(monkeypatch) -> None:
    class StoreClock(datetime):
        current = datetime(2026, 7, 16, 10, 0, tzinfo=UTC)

        @classmethod
        def now(cls, tz=None):
            return cls.current

    async def scenario() -> None:
        store = await _session_store("writer_lock_relative_deadlines")
        event = Event(type="custom.locked", session_id="writer_lock_relative_deadlines")
        await store.append_event(event.session_id, event)
        monkeypatch.setattr("cayu.runtime.sessions.datetime", StoreClock)

        await store._lock.acquire()
        claim_task = asyncio.create_task(
            store.claim_persisted_event_side_effect(
                session_id=event.session_id,
                event_id=event.id,
                lease_seconds=30,
            )
        )
        await asyncio.sleep(0)
        StoreClock.current += timedelta(minutes=1)
        store._lock.release()
        claim = await claim_task
        assert claim is not None
        assert claim.lease_expires_at == StoreClock.current + timedelta(seconds=30)
        assert (
            await store.claim_persisted_event_side_effect(
                session_id=event.session_id,
                event_id=event.id,
            )
            is None
        )

        await store._lock.acquire()
        failure_task = asyncio.create_task(
            store.mark_persisted_event_side_effect_failed(
                claim,
                error="try later",
                max_attempts=3,
                retry_delay_seconds=30,
            )
        )
        await asyncio.sleep(0)
        StoreClock.current += timedelta(minutes=1)
        store._lock.release()
        failed = await failure_task
        assert failed.updated_at == StoreClock.current
        assert failed.next_attempt_at == StoreClock.current + timedelta(seconds=30)
        assert (
            await store.claim_persisted_event_side_effect(
                session_id=event.session_id,
                event_id=event.id,
            )
            is None
        )

    asyncio.run(scenario())


def test_budget_failure_is_not_masked_when_failure_bookkeeping_also_fails() -> None:
    async def scenario() -> None:
        store = _BrokenFailureBookkeepingSessionStore()
        await store.create(
            RunRequest(
                agent_name="assistant",
                session_id="writer_budget_bookkeeping",
                messages=[Message.text("user", "go")],
            ),
            identity=SessionIdentity(provider_name="fake", model="fake-model"),
        )
        writer = RuntimeEventWriter(
            session_store=store,
            budget_store=_FailingBudgetStore(),
            event_sinks=[],
        )

        with pytest.raises(RuntimeError, match="budget unavailable") as raised:
            await writer.emit(
                Event(
                    type=EventType.MODEL_COMPLETED,
                    session_id="writer_budget_bookkeeping",
                )
            )
        assert raised.value.__notes__ == [
            "Persisted event side-effect failure bookkeeping also failed: "
            "RuntimeError: handoff store unavailable"
        ]

    asyncio.run(scenario())


def test_budget_failure_survives_lost_failure_acknowledgement() -> None:
    async def scenario() -> None:
        store = _LostFailureAcknowledgementSessionStore()
        await store.create(
            RunRequest(
                agent_name="assistant",
                session_id="writer_budget_lost_failure_ack",
                messages=[Message.text("user", "go")],
            ),
            identity=SessionIdentity(provider_name="fake", model="fake-model"),
        )
        writer = RuntimeEventWriter(
            session_store=store,
            budget_store=_FailingBudgetStore(),
            event_sinks=[],
        )

        with pytest.raises(RuntimeError, match="budget unavailable") as raised:
            await writer.emit(
                Event(
                    type=EventType.MODEL_COMPLETED,
                    session_id="writer_budget_lost_failure_ack",
                )
            )
        assert not getattr(raised.value, "__notes__", None)

    asyncio.run(scenario())


def test_sink_diagnostic_failure_does_not_skip_later_sinks_or_failure_state() -> None:
    async def scenario():
        store = _BrokenSinkDiagnosticSessionStore()
        await store.create(
            RunRequest(
                agent_name="assistant",
                session_id="writer_sink_diagnostic",
                messages=[Message.text("user", "go")],
            ),
            identity=SessionIdentity(provider_name="fake", model="fake-model"),
        )
        recorder = _RecordingSink()
        writer = RuntimeEventWriter(
            session_store=store,
            budget_store=InMemoryBudgetStore(),
            event_sinks=[_FailingSink(), recorder],
        )

        emitted = await writer.emit(
            Event(type=EventType.SESSION_STARTED, session_id="writer_sink_diagnostic")
        )
        deliveries = await store.list_persisted_event_side_effect_deliveries()
        return emitted, recorder.events, deliveries

    emitted, recorded, deliveries = asyncio.run(scenario())

    assert [event.id for event in recorded] == [emitted.id]
    assert len(deliveries) == 1
    assert deliveries[0].status is PersistedEventSideEffectStatus.FAILED
    assert deliveries[0].last_error == (
        "RuntimeError: sink unavailable; runtime.sink.failed persistence failed: "
        "RuntimeError: diagnostic store unavailable"
    )


def test_sink_failure_is_authoritative_when_failure_bookkeeping_also_fails() -> None:
    async def scenario() -> None:
        store = _BrokenFailureBookkeepingSessionStore()
        await store.create(
            RunRequest(
                agent_name="assistant",
                session_id="writer_sink_bookkeeping",
                messages=[Message.text("user", "go")],
            ),
            identity=SessionIdentity(provider_name="fake", model="fake-model"),
        )
        writer = RuntimeEventWriter(
            session_store=store,
            budget_store=InMemoryBudgetStore(),
            event_sinks=[_FailingSink()],
        )

        with pytest.raises(RuntimeError, match="sink unavailable") as raised:
            await writer.emit(
                Event(
                    type=EventType.SESSION_STARTED,
                    session_id="writer_sink_bookkeeping",
                )
            )
        assert raised.value.__notes__ == [
            "Persisted event side-effect failure bookkeeping also failed: "
            "RuntimeError: handoff store unavailable"
        ]

    asyncio.run(scenario())


def test_emit_many_copies_events_before_persisting_and_fanout() -> None:
    async def scenario() -> tuple[list[Event], list[Event], list[Event]]:
        store = await _session_store("writer_batch")
        sink = _RecordingSink()
        writer = RuntimeEventWriter(
            session_store=store,
            budget_store=InMemoryBudgetStore(),
            event_sinks=[sink],
        )
        source = [
            Event(type="custom.example.one", session_id="writer_batch", payload={"value": 1}),
            Event(type="custom.example.two", session_id="writer_batch", payload={"value": 2}),
        ]

        emitted = await writer.emit_many("writer_batch", source)
        source[0].payload["value"] = 99
        return emitted, await store.load_events("writer_batch"), sink.events

    emitted, persisted, sink_events = asyncio.run(scenario())

    expected = [{"value": 1}, {"value": 2}]
    assert [event.payload for event in emitted] == expected
    assert [event.payload for event in persisted] == expected
    assert [event.payload for event in sink_events] == expected


def test_persist_many_exposes_durable_boundary_before_fanout() -> None:
    async def scenario() -> tuple[list[Event], list[Event], list[Event], list[Event]]:
        store = await _session_store("writer_persist_batch")
        sink = _RecordingSink()
        writer = RuntimeEventWriter(
            session_store=store,
            budget_store=InMemoryBudgetStore(),
            event_sinks=[sink],
        )
        source = [
            Event(
                type="custom.example.persisted",
                session_id="writer_persist_batch",
                payload={"value": 1},
            )
        ]

        persisted_result = await writer.persist_many("writer_persist_batch", source)
        source[0].payload["value"] = 99
        before_fanout = list(sink.events)
        await writer.fan_out_persisted(persisted_result)
        return (
            persisted_result,
            await store.load_events("writer_persist_batch"),
            before_fanout,
            sink.events,
        )

    persisted_result, stored, before_fanout, after_fanout = asyncio.run(scenario())

    assert [event.payload for event in persisted_result] == [{"value": 1}]
    assert [event.payload for event in stored] == [{"value": 1}]
    assert before_fanout == []
    assert [event.payload for event in after_fanout] == [{"value": 1}]


def test_emit_many_rejects_event_for_different_session() -> None:
    async def scenario() -> None:
        writer = RuntimeEventWriter(
            session_store=await _session_store("writer_target"),
            budget_store=InMemoryBudgetStore(),
            event_sinks=[],
        )

        with pytest.raises(ValueError, match="session_id does not match"):
            await writer.emit_many(
                "writer_target",
                [Event(type="custom.example", session_id="writer_other")],
            )

    asyncio.run(scenario())


def test_sink_failure_is_durable_and_does_not_block_later_sink() -> None:
    async def scenario() -> tuple[list[Event], list[Event]]:
        store = await _session_store("writer_sink_failure")
        recorder = _RecordingSink()
        writer = RuntimeEventWriter(
            session_store=store,
            budget_store=InMemoryBudgetStore(),
            event_sinks=[_FailingSink(), recorder],
        )
        event = Event(type=EventType.SESSION_STARTED, session_id="writer_sink_failure")

        await writer.emit(event)
        return await store.load_events("writer_sink_failure"), recorder.events

    persisted, recorded = asyncio.run(scenario())

    assert [event.type for event in persisted] == [
        EventType.SESSION_STARTED,
        EventType.RUNTIME_SINK_FAILED,
    ]
    assert persisted[1].payload == {
        "sink": "_FailingSink",
        "error": "sink unavailable",
        "error_type": "RuntimeError",
        "event_id": persisted[0].id,
        "event_type": EventType.SESSION_STARTED,
    }
    assert [event.type for event in recorded] == [EventType.SESSION_STARTED]


def test_failed_sink_delivery_is_recovered_and_closed(monkeypatch) -> None:
    monkeypatch.setattr(
        "cayu.runtime._event_writer._PERSISTED_SIDE_EFFECT_RETRY_DELAY_SECONDS",
        0,
    )

    async def scenario():
        store = await _session_store("writer_sink_retry")
        sink = _FlakySink(failures=1)
        writer = RuntimeEventWriter(
            session_store=store,
            budget_store=InMemoryBudgetStore(),
            event_sinks=[sink],
        )

        emitted = await writer.emit(
            Event(type=EventType.SESSION_STARTED, session_id="writer_sink_retry")
        )
        failed = await store.list_persisted_event_side_effect_deliveries()
        recovered = await writer.recover_persisted_side_effects()
        closed = await store.list_persisted_event_side_effect_deliveries()
        return emitted, failed, recovered, closed, sink.events

    emitted, failed, recovered, closed, sink_events = asyncio.run(scenario())

    assert [(state.status, state.attempts) for state in failed] == [
        (PersistedEventSideEffectStatus.FAILED, 1)
    ]
    assert [event.id for event in recovered] == [emitted.id]
    assert [(state.status, state.attempts) for state in closed] == [
        (PersistedEventSideEffectStatus.DELIVERED, 2)
    ]
    assert [event.id for event in sink_events] == [emitted.id]


def test_repeated_sink_failure_dead_letters_without_recursive_handoffs(monkeypatch) -> None:
    monkeypatch.setattr(
        "cayu.runtime._event_writer._PERSISTED_SIDE_EFFECT_RETRY_DELAY_SECONDS",
        0,
    )

    async def scenario():
        store = await _session_store("writer_sink_dead_letter")
        writer = RuntimeEventWriter(
            session_store=store,
            budget_store=InMemoryBudgetStore(),
            event_sinks=[_FailingSink()],
        )

        await writer.emit(
            Event(type=EventType.SESSION_STARTED, session_id="writer_sink_dead_letter")
        )
        await writer.recover_persisted_side_effects()
        await writer.recover_persisted_side_effects()
        assert await writer.recover_persisted_side_effects() == []
        return (
            await store.list_persisted_event_side_effect_deliveries(),
            await store.load_events("writer_sink_dead_letter"),
        )

    deliveries, events = asyncio.run(scenario())

    assert [(state.status, state.attempts) for state in deliveries] == [
        (PersistedEventSideEffectStatus.DEAD_LETTERED, 3)
    ]
    assert [event.type for event in events] == [
        EventType.SESSION_STARTED,
        EventType.RUNTIME_SINK_FAILED,
        EventType.RUNTIME_SINK_FAILED,
        EventType.RUNTIME_SINK_FAILED,
    ]


def test_budget_retry_after_crash_is_idempotent_by_event_identity() -> None:
    async def scenario():
        store = await _session_store("writer_budget_retry")
        event = Event(type=EventType.MODEL_COMPLETED, session_id="writer_budget_retry")
        await store.append_event(event.session_id, event)
        claim = await store.claim_persisted_event_side_effect(lease_seconds=0.000001)
        assert claim is not None
        budget_store = InMemoryBudgetStore()
        await budget_store.append_event(claim.event)
        await asyncio.sleep(0.001)

        writer = RuntimeEventWriter(
            session_store=store,
            budget_store=budget_store,
            event_sinks=[],
        )
        recovered = await writer.recover_persisted_side_effects()
        budget_events = await budget_store.load_events_for_budget(
            scope="app",
            key=None,
            window=BudgetWindow.all_time(),
        )
        return recovered, budget_events

    recovered, budget_events = asyncio.run(scenario())

    assert len(recovered) == 1
    assert [event.id for event in budget_events] == [recovered[0].id]


def test_recovery_continues_after_one_poison_budget_delivery() -> None:
    async def scenario():
        store = await _session_store("writer_poison_delivery")
        poison = Event(type=EventType.MODEL_COMPLETED, session_id="writer_poison_delivery")
        healthy = Event(type="custom.healthy", session_id="writer_poison_delivery")
        await store.append_events(poison.session_id, [poison, healthy])
        sink = _RecordingSink()
        writer = RuntimeEventWriter(
            session_store=store,
            budget_store=_FailingBudgetStore(fail_event_id=poison.id),
            event_sinks=[sink],
        )

        recovered = await writer.recover_persisted_side_effects()
        states = await store.list_persisted_event_side_effect_deliveries()
        return poison, healthy, recovered, states, sink.events

    poison, healthy, recovered, states, sink_events = asyncio.run(scenario())

    assert [event.id for event in recovered] == [healthy.id]
    assert [event.id for event in sink_events] == [healthy.id]
    assert [(state.event_id, state.status) for state in states] == [
        (poison.id, PersistedEventSideEffectStatus.FAILED),
        (healthy.id, PersistedEventSideEffectStatus.DELIVERED),
    ]


def test_cayu_app_exposes_persisted_event_side_effect_recovery() -> None:
    async def scenario():
        store = await _session_store("app_event_recovery")
        event = Event(type="custom.recover", session_id="app_event_recovery")
        await store.append_event(event.session_id, event)
        sink = _RecordingSink()
        app = CayuApp(session_store=store, event_sinks=[sink], enable_logging=False)

        recovered = await app.recover_persisted_event_side_effects()
        return recovered, sink.events

    recovered, sink_events = asyncio.run(scenario())

    assert [event.type for event in recovered] == ["custom.recover"]
    assert [event.id for event in sink_events] == [recovered[0].id]


def test_recovery_limit_counts_claimable_events_not_live_leases() -> None:
    async def scenario():
        store = await _session_store("writer_live_lease")
        leased = Event(type="custom.leased", session_id="writer_live_lease")
        pending = Event(type="custom.pending", session_id="writer_live_lease")
        await store.append_events(leased.session_id, [leased, pending])
        assert (
            await store.claim_persisted_event_side_effect(
                session_id=leased.session_id,
                event_id=leased.id,
            )
            is not None
        )
        sink = _RecordingSink()
        writer = RuntimeEventWriter(
            session_store=store,
            budget_store=InMemoryBudgetStore(),
            event_sinks=[sink],
        )

        recovered = await writer.recover_persisted_side_effects(limit=1)
        return recovered, sink.events

    recovered, sink_events = asyncio.run(scenario())

    assert [event.type for event in recovered] == ["custom.pending"]
    assert [event.id for event in sink_events] == [recovered[0].id]
