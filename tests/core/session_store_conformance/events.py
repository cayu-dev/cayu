"""Shared session-store events contract cases."""

from __future__ import annotations

import asyncio
from typing import Any, cast

import pytest
from tests.core.session_store_conformance.support import (
    _close_store,
    _identity,
    _IterationCountingEventIds,
    _open_store,
    _reopen_store,
)

from cayu.events import (
    Event,
    EventType,
)
from cayu.messages import (
    Message,
)
from cayu.sessions.event_delivery import (
    PERSISTED_EVENT_SIDE_EFFECT_ERROR_MAX_BYTES,
    PersistedEventSideEffectClaimLost,
    PersistedEventSideEffectDelivery,
    PersistedEventSideEffectStatus,
)
from cayu.sessions.memory import InMemorySessionStore
from cayu.sessions.requests import (
    RunRequest,
)

pytestmark = pytest.mark.postgres


def test_in_memory_event_append_does_not_scan_existing_event_ids() -> None:
    async def run() -> None:
        store = InMemorySessionStore()
        session_id = "sess_event_id_membership"
        await store.create(
            RunRequest(
                agent_name="assistant",
                session_id=session_id,
                messages=[Message.text("user", "append without scanning history")],
            ),
            identity=_identity(),
        )
        existing_events = [
            Event(
                id=f"existing-event-{index}",
                type=EventType.MODEL_STARTED,
                session_id=session_id,
            )
            for index in range(32)
        ]
        await store.append_events(session_id, existing_events)

        tracked_ids = _IterationCountingEventIds(store._event_ids[session_id])
        store._event_ids[session_id] = cast("set[str]", tracked_ids)
        appended = Event(
            id="new-event",
            type=EventType.MODEL_COMPLETED,
            session_id=session_id,
        )
        await store.append_event(session_id, appended)

        assert tracked_ids.iterated_values == 0
        assert tracked_ids.membership_checks == 1
        assert tracked_ids.add_calls == 1
        assert [event.id for event in await store.load_events(session_id)] == [
            *(event.id for event in existing_events),
            appended.id,
        ]

    asyncio.run(run())


def test_session_store_conformance_side_effect_deferral_retains_delivery_and_fences_old_claim(
    session_store_case,
) -> None:
    async def run() -> None:
        store = await _open_store(session_store_case)
        try:
            session = await store.create(
                RunRequest(
                    agent_name="assistant",
                    session_id="deferred-child-wakeup",
                    messages=[Message.text("user", "go")],
                ),
                identity=_identity(),
            )
            event = Event(type=EventType.SESSION_COMPLETED, session_id=session.id)
            await store.append_event(session.id, event)
            previous_claim = None
            for _ in range(4):
                claim = await store.claim_persisted_event_side_effect(
                    session_id=session.id, event_id=event.id
                )
                assert claim is not None and claim.attempt == 1
                if previous_claim is not None:
                    assert claim.claim_id != previous_claim.claim_id
                    with pytest.raises(PersistedEventSideEffectClaimLost):
                        await store.mark_persisted_event_side_effect_delivered(previous_claim)
                    with pytest.raises(PersistedEventSideEffectClaimLost):
                        await store.defer_persisted_event_side_effect(previous_claim)
                deferred = await store.defer_persisted_event_side_effect(claim)
                assert deferred.status is PersistedEventSideEffectStatus.PENDING
                assert deferred.attempts == 0
                assert deferred.claim_id is None
                assert deferred.lease_expires_at is None
                previous_claim = claim
                store = await _reopen_store(session_store_case, store)
            candidates = await store.list_persisted_event_side_effect_deliveries(
                claimable_only=True, limit=1
            )
            assert len(candidates) == 1 and candidates[0].event_id == event.id
            claim = await store.claim_persisted_event_side_effect(
                session_id=session.id, event_id=event.id
            )
            assert claim is not None
            delivered = await store.mark_persisted_event_side_effect_delivered(claim)
            assert delivered.status is PersistedEventSideEffectStatus.DELIVERED
        finally:
            await _close_store(store)

    asyncio.run(run())


def test_session_store_conformance_side_effect_renewal_requires_live_exact_claim(
    session_store_case,
) -> None:
    async def run() -> None:
        store = await _open_store(session_store_case)
        try:
            session = await store.create(
                RunRequest(
                    agent_name="assistant",
                    session_id="renewed-child-wakeup",
                    messages=[Message.text("user", "go")],
                ),
                identity=_identity(),
            )
            event = Event(type=EventType.SESSION_COMPLETED, session_id=session.id)
            await store.append_event(session.id, event)
            claim = await store.claim_persisted_event_side_effect(
                session_id=session.id,
                event_id=event.id,
                lease_seconds=3,
            )
            assert claim is not None
            renewed = await store.renew_persisted_event_side_effect(claim, lease_seconds=30)
            assert renewed.claim_id == claim.claim_id
            assert renewed.attempts == claim.attempt
            assert renewed.lease_expires_at is not None
            assert renewed.lease_expires_at > claim.lease_expires_at
            for invalid in (True, 0, -1, float("nan"), float("inf"), 86_401):
                with pytest.raises(ValueError, match="lease_seconds"):
                    await store.renew_persisted_event_side_effect(claim, lease_seconds=invalid)
            assert (
                await store.get_persisted_event_side_effect_delivery(
                    session_id=session.id, event_id=event.id
                )
                == renewed
            )
            # A shorter renewal cannot shorten an already-owned interval.
            shorter = await store.renew_persisted_event_side_effect(claim, lease_seconds=1)
            assert shorter.lease_expires_at == renewed.lease_expires_at
            store = await _reopen_store(session_store_case, store)
            await asyncio.sleep(3.1)
            assert (
                await store.claim_persisted_event_side_effect(
                    session_id=session.id, event_id=event.id
                )
                is None
            )
            await store.defer_persisted_event_side_effect(claim)
            expired = await store.claim_persisted_event_side_effect(
                session_id=session.id,
                event_id=event.id,
                lease_seconds=0.05,
            )
            assert expired is not None
            await asyncio.sleep(0.1)
            # Expiry alone revokes renewal; no competing claimant is needed.
            with pytest.raises(PersistedEventSideEffectClaimLost):
                await store.renew_persisted_event_side_effect(expired)
            replacement = await store.claim_persisted_event_side_effect(
                session_id=session.id, event_id=event.id
            )
            assert replacement is not None
            assert replacement.claim_id not in {claim.claim_id, expired.claim_id}
            for stale in (claim, expired):
                with pytest.raises(PersistedEventSideEffectClaimLost):
                    await store.renew_persisted_event_side_effect(stale)
            delivered = await store.mark_persisted_event_side_effect_delivered(replacement)
            assert delivered.status is PersistedEventSideEffectStatus.DELIVERED
            with pytest.raises(PersistedEventSideEffectClaimLost):
                await store.renew_persisted_event_side_effect(replacement)
        finally:
            await _close_store(store)

    asyncio.run(run())


def test_session_store_conformance_persisted_event_side_effect_recovery(
    session_store_case,
) -> None:
    async def run() -> None:
        store = await _open_store(session_store_case)
        try:
            session = await store.create(
                RunRequest(
                    agent_name="assistant",
                    session_id="sess_side_effect_recovery",
                    messages=[Message.text("user", "persist")],
                ),
                identity=_identity(),
            )
            event = Event(type=EventType.MODEL_COMPLETED, session_id=session.id)
            await store.append_event(session.id, event)
            pending = await store.get_persisted_event_side_effect_delivery(
                session_id=session.id,
                event_id=event.id,
            )
            assert pending is not None
            assert pending.status is PersistedEventSideEffectStatus.PENDING
            assert (
                await store.get_persisted_event_side_effect_delivery(
                    session_id=session.id,
                    event_id="missing-event",
                )
                is None
            )
            await store.append_event(
                session.id,
                Event(
                    type=EventType.RUNTIME_SINK_FAILED,
                    session_id=session.id,
                    payload={"event_id": event.id},
                ),
            )

            store = await _reopen_store(session_store_case, store)
            first_claim = await store.claim_persisted_event_side_effect()
            assert first_claim is not None
            assert first_claim.event.id == event.id
            assert first_claim.attempt == 1
            leased = await store.get_persisted_event_side_effect_delivery(
                session_id=session.id,
                event_id=event.id,
            )
            assert leased is not None
            assert leased.status is PersistedEventSideEffectStatus.LEASED
            failed = await store.mark_persisted_event_side_effect_failed(
                first_claim,
                error="sink unavailable",
                max_attempts=2,
                retry_delay_seconds=0,
            )
            assert failed.status is PersistedEventSideEffectStatus.FAILED
            loaded_failed = await store.get_persisted_event_side_effect_delivery(
                session_id=session.id,
                event_id=event.id,
            )
            assert loaded_failed == failed

            store = await _reopen_store(session_store_case, store)
            second_claim = await store.claim_persisted_event_side_effect()
            assert second_claim is not None
            assert second_claim.event.id == event.id
            assert second_claim.attempt == 2
            delivered = await store.mark_persisted_event_side_effect_delivered(second_claim)
            assert delivered.status is PersistedEventSideEffectStatus.DELIVERED
            loaded_delivered = await store.get_persisted_event_side_effect_delivery(
                session_id=session.id,
                event_id=event.id,
            )
            assert loaded_delivered == delivered
            assert await store.claim_persisted_event_side_effect() is None
            states = await store.list_persisted_event_side_effect_deliveries()
            assert [(state.event_id, state.status, state.attempts) for state in states] == [
                (event.id, PersistedEventSideEffectStatus.DELIVERED, 2)
            ]
        finally:
            await _close_store(store)

    asyncio.run(run())


def test_session_store_conformance_rejects_nonportable_side_effect_errors_atomically(
    session_store_case,
) -> None:
    async def run() -> None:
        store = await _open_store(session_store_case)
        try:
            session = await store.create(
                RunRequest(
                    agent_name="assistant",
                    session_id="sess_side_effect_error_portability",
                    messages=[Message.text("user", "persist")],
                ),
                identity=_identity(),
            )
            event = Event(type=EventType.SESSION_STARTED, session_id=session.id)
            await store.append_event(session.id, event)
            claim = await store.claim_persisted_event_side_effect(
                session_id=session.id,
                event_id=event.id,
            )
            assert claim is not None
            leased = await store.get_persisted_event_side_effect_delivery(
                session_id=session.id,
                event_id=event.id,
            )
            assert leased is not None

            invalid_errors = (
                "sink\x00secret",
                "sink \ud800 secret",
                "x" * (PERSISTED_EVENT_SIDE_EFFECT_ERROR_MAX_BYTES + 1),
            )
            for error in invalid_errors:
                with pytest.raises(ValueError):
                    await store.mark_persisted_event_side_effect_failed(
                        claim,
                        error=error,
                        max_attempts=3,
                        retry_delay_seconds=0,
                    )
                assert (
                    await store.get_persisted_event_side_effect_delivery(
                        session_id=session.id,
                        event_id=event.id,
                    )
                    == leased
                )

            failed = await store.mark_persisted_event_side_effect_failed(
                claim,
                error="portable failure",
                max_attempts=3,
                retry_delay_seconds=0,
            )
            assert failed.status is PersistedEventSideEffectStatus.FAILED
            assert failed.last_error == "portable failure"
        finally:
            await _close_store(store)

    asyncio.run(run())


def test_legacy_unbounded_side_effect_error_remains_readable() -> None:
    delivery = PersistedEventSideEffectDelivery(
        session_id="sess_legacy_side_effect_error",
        event_id="event_legacy_side_effect_error",
        event_sequence=1,
        status=PersistedEventSideEffectStatus.FAILED,
        last_error="é" * (PERSISTED_EVENT_SIDE_EFFECT_ERROR_MAX_BYTES + 1),
    )

    assert delivery.last_error is not None
    assert len(delivery.last_error.encode("utf-8")) <= (PERSISTED_EVENT_SIDE_EFFECT_ERROR_MAX_BYTES)
    assert delivery.last_error.endswith("... [truncated]")


def test_session_store_conformance_persisted_event_side_effect_claim_fencing(
    session_store_case,
) -> None:
    async def run() -> None:
        store = await _open_store(session_store_case)
        try:
            session = await store.create(
                RunRequest(
                    agent_name="assistant",
                    session_id="sess_side_effect_fencing",
                    messages=[Message.text("user", "persist")],
                ),
                identity=_identity(),
            )
            event = Event(type=EventType.SESSION_STARTED, session_id=session.id)
            await store.append_event(session.id, event)

            # Keep the initial lease comfortably live across a loaded PostgreSQL
            # round trip; the old 50ms window made this immediate fencing check
            # depend on CI scheduler latency rather than store behavior.
            stale = await store.claim_persisted_event_side_effect(lease_seconds=0.5)
            assert stale is not None
            pending = Event(type="custom.pending", session_id=session.id)
            await store.append_event(session.id, pending)
            claimable = await store.list_persisted_event_side_effect_deliveries(
                claimable_only=True,
                limit=1,
            )
            assert [delivery.event_id for delivery in claimable] == [pending.id]
            await asyncio.sleep(0.6)
            replacement = await store.claim_persisted_event_side_effect()
            assert replacement is not None
            assert replacement.event.id == event.id
            assert replacement.attempt == 2
            with pytest.raises(PersistedEventSideEffectClaimLost, match="no longer active"):
                await store.mark_persisted_event_side_effect_delivered(stale)
            dead_lettered = await store.mark_persisted_event_side_effect_failed(
                replacement,
                error="still unavailable",
                max_attempts=2,
                retry_delay_seconds=0,
            )
            assert dead_lettered.status is PersistedEventSideEffectStatus.DEAD_LETTERED
            loaded_dead_lettered = await store.get_persisted_event_side_effect_delivery(
                session_id=session.id,
                event_id=event.id,
            )
            assert loaded_dead_lettered == dead_lettered
            pending_claim = await store.claim_persisted_event_side_effect()
            assert pending_claim is not None
            assert pending_claim.event.id == pending.id
            await store.mark_persisted_event_side_effect_delivered(pending_claim)
            assert await store.claim_persisted_event_side_effect() is None
        finally:
            await _close_store(store)

    asyncio.run(run())


def test_session_store_conformance_persisted_event_side_effect_retry_spacing_and_paging(
    session_store_case,
    monkeypatch,
) -> None:
    async def run() -> None:
        store = await _open_store(session_store_case)
        try:
            session = await store.create(
                RunRequest(
                    agent_name="assistant",
                    session_id="sess_side_effect_retry_spacing",
                    messages=[Message.text("user", "persist")],
                ),
                identity=_identity(),
            )
            events = [
                Event(type=f"custom.page.{index}", session_id=session.id) for index in range(3)
            ]
            await store.append_events(session.id, events)

            async def exercise_retry_clock():
                claim = await store.claim_persisted_event_side_effect(
                    session_id=session.id,
                    event_id=events[0].id,
                )
                assert claim is not None
                failed = await store.mark_persisted_event_side_effect_failed(
                    claim,
                    error="try later",
                    max_attempts=3,
                    retry_delay_seconds=60,
                )
                assert failed.next_attempt_at is not None
                assert failed.next_attempt_at > failed.updated_at
                assert (
                    await store.claim_persisted_event_side_effect(
                        session_id=session.id,
                        event_id=events[0].id,
                    )
                    is None
                )

            if session_store_case[0] == "postgres":

                class NodeClockMustNotBeRead:
                    @classmethod
                    def now(cls, *args, **kwargs):
                        raise AssertionError("Postgres handoff eligibility must use DB time")

                with monkeypatch.context() as context:
                    context.setattr("cayu.storage.postgres.datetime", NodeClockMustNotBeRead)
                    await exercise_retry_clock()
            else:
                await exercise_retry_clock()

            claimable = await store.list_persisted_event_side_effect_deliveries(
                claimable_only=True,
            )
            assert [state.event_id for state in claimable] == [events[1].id, events[2].id]

            first_page = await store.list_persisted_event_side_effect_deliveries(limit=2)
            second_page = await store.list_persisted_event_side_effect_deliveries(
                after_sequence=first_page[-1].event_sequence,
                limit=2,
            )
            assert [state.event_id for state in [*first_page, *second_page]] == [
                event.id for event in events
            ]
            assert second_page[0].event_sequence > first_page[-1].event_sequence
        finally:
            await _close_store(store)

    asyncio.run(run())


def test_session_store_conformance_validates_event_batch_preamble(
    session_store_case,
) -> None:
    async def run() -> None:
        session_store = await _open_store(session_store_case)
        try:
            await session_store.create(
                RunRequest(
                    agent_name="assistant",
                    session_id="sess_event_preamble",
                    messages=[Message.text("user", "events")],
                ),
                identity=_identity(),
            )
            append_events: Any = session_store.append_events

            with pytest.raises(TypeError, match="Session events must be a list."):
                await append_events("sess_event_preamble", ())
            with pytest.raises(TypeError, match="Session events must be Event instances."):
                await append_events("sess_event_preamble", ["not-an-event"])
            with pytest.raises(ValueError, match="Event session_id does not match target session."):
                await session_store.append_events(
                    "sess_event_preamble",
                    [
                        Event(
                            id="evt_wrong_session",
                            type=EventType.SESSION_STARTED,
                            session_id="sess_other",
                        )
                    ],
                )
            with pytest.raises(ValueError, match="Event already exists for session"):
                await session_store.append_events(
                    "sess_event_preamble",
                    [
                        Event(
                            id="evt_duplicate",
                            type=EventType.SESSION_STARTED,
                            session_id="sess_event_preamble",
                        ),
                        Event(
                            id="evt_duplicate",
                            type=EventType.SESSION_STARTED,
                            session_id="sess_event_preamble",
                        ),
                    ],
                )
        finally:
            await _close_store(session_store)

    asyncio.run(run())
