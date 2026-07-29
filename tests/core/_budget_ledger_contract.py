from __future__ import annotations

import asyncio
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Protocol
from unittest.mock import patch
from uuid import UUID

import pytest
from tests.core._execution_unit_fixtures import model_attempt_identity

from cayu._validation import DurableValueError
from cayu.runtime import BudgetLedger, BudgetLimit
from cayu.runtime.budgets import BudgetReservationIdentityConflict


class MutableClock(Protocol):
    value: datetime

    def __call__(self) -> datetime: ...


async def assert_reservation_identity_collision_is_rejected(
    ledger: BudgetLedger,
    limit: BudgetLimit,
) -> None:
    first_identity = model_attempt_identity()
    second_identity = model_attempt_identity()
    fixed_uuid = UUID("11111111-1111-1111-1111-111111111111")
    reserve_kwargs = {
        "limit": limit,
        "agent_name": "assistant",
        "provider_name": "fake",
        "model": "fake-model",
    }

    with patch("cayu.runtime.budgets.uuid4", return_value=fixed_uuid):
        first = await ledger.reserve(
            session_id="sess_reservation_identity_winner",
            model_attempt_identity=first_identity,
            **reserve_kwargs,
        )
        assert first.accepted is True
        assert first.record is not None

        with pytest.raises(
            BudgetReservationIdentityConflict,
            match="reused a reservation identity",
        ):
            await ledger.reserve(
                session_id="sess_reservation_identity_collision",
                model_attempt_identity=second_identity,
                **reserve_kwargs,
            )

    reconciled = await ledger.reconcile(
        reservation_id=first.record.reservation_id,
        actual_amount=Decimal("0.01"),
        reason="winner retained after collision",
    )
    assert (reconciled.model_step_id, reconciled.model_attempt_id) == (
        first_identity.model_step_id,
        first_identity.model_attempt_id,
    )


async def assert_portable_text_boundaries(
    ledger: BudgetLedger,
    limit: BudgetLimit,
) -> None:
    for invalid_text, code in (
        ("workload-secret-value\x00", "nul_character"),
        ("workload-secret-value\ud800", "unicode_surrogate"),
    ):
        with pytest.raises(DurableValueError) as invalid_reservation:
            await ledger.reserve(
                limit=limit,
                session_id=invalid_text,
                agent_name="assistant",
                provider_name="fake",
                model="fake-model",
                model_attempt_identity=model_attempt_identity(),
            )
        assert invalid_reservation.value.code == code
        assert "workload-secret-value" not in str(invalid_reservation.value)

    reserved = await ledger.reserve(
        limit=limit,
        session_id="sess_portable_text",
        agent_name="assistant",
        provider_name="fake",
        model="fake-model",
        model_attempt_identity=model_attempt_identity(),
    )
    assert reserved.accepted is True
    assert reserved.record is not None

    for invalid_text, code in (
        ("workload-secret-value\x00", "nul_character"),
        ("workload-secret-value\ud800", "unicode_surrogate"),
    ):
        with pytest.raises(DurableValueError) as invalid_release:
            await ledger.release(
                reservation_id=reserved.record.reservation_id,
                reason=invalid_text,
            )
        assert invalid_release.value.code == code
        assert "workload-secret-value" not in str(invalid_release.value)
        assert await ledger.heartbeat(reservation_id=reserved.record.reservation_id) is True

    released = await ledger.release(
        reservation_id=reserved.record.reservation_id,
        reason="portable boundary verified",
    )
    assert released.status == "released"


async def assert_idempotent_terminal_settlements(
    ledger: BudgetLedger,
    limit: BudgetLimit,
    *,
    clock: MutableClock,
) -> None:
    first_identity = model_attempt_identity()
    first = await ledger.reserve(
        limit=limit,
        session_id="sess_idempotent_reconcile",
        agent_name="assistant",
        provider_name="fake",
        model="fake-model",
        model_attempt_identity=first_identity,
    )
    assert first.accepted is True
    assert first.record is not None
    assert (first.model_step_id, first.model_attempt_id) == (
        first_identity.model_step_id,
        first_identity.model_attempt_id,
    )
    assert (first.record.model_step_id, first.record.model_attempt_id) == (
        first_identity.model_step_id,
        first_identity.model_attempt_id,
    )

    occurred_at = clock.value
    reconciled, concurrent_reconciliation = await asyncio.gather(
        ledger.reconcile(
            reservation_id=first.record.reservation_id,
            actual_amount=Decimal("0.01"),
            reason="actual usage",
            occurred_at=occurred_at,
        ),
        ledger.reconcile(
            reservation_id=first.record.reservation_id,
            actual_amount=Decimal("0.01"),
            reason="actual usage",
            occurred_at=occurred_at,
        ),
    )
    assert concurrent_reconciliation == reconciled
    assert (reconciled.model_step_id, reconciled.model_attempt_id) == (
        first_identity.model_step_id,
        first_identity.model_attempt_id,
    )
    reconciliation_retry = await ledger.reconcile(
        reservation_id=first.record.reservation_id,
        actual_amount=Decimal("0.01"),
        reason="actual usage",
        occurred_at=occurred_at + timedelta(seconds=40),
    )
    assert reconciliation_retry == reconciled

    with pytest.raises(ValueError, match="conflicting reconciliation"):
        await ledger.reconcile(
            reservation_id=first.record.reservation_id,
            actual_amount=Decimal("0.02"),
            reason="different outcome",
        )
    with pytest.raises(ValueError, match="not active"):
        await ledger.release(
            reservation_id=first.record.reservation_id,
            reason="cannot release charged spend",
        )

    clock.value = occurred_at + timedelta(seconds=1)
    second_identity = model_attempt_identity()
    second = await ledger.reserve(
        limit=limit,
        session_id="sess_idempotent_release",
        agent_name="assistant",
        provider_name="fake",
        model="fake-model",
        model_attempt_identity=second_identity,
    )
    assert second.accepted is True
    assert second.actual == Decimal("0.23")
    assert second.record is not None
    assert (second.model_step_id, second.model_attempt_id) == (
        second_identity.model_step_id,
        second_identity.model_attempt_id,
    )

    released, concurrent_release = await asyncio.gather(
        ledger.release(
            reservation_id=second.record.reservation_id,
            reason="provider not dispatched",
        ),
        ledger.release(
            reservation_id=second.record.reservation_id,
            reason="provider not dispatched",
        ),
    )
    assert concurrent_release == released
    assert (released.model_step_id, released.model_attempt_id) == (
        second_identity.model_step_id,
        second_identity.model_attempt_id,
    )
    release_retry = await ledger.release(
        reservation_id=second.record.reservation_id,
        reason="provider not dispatched",
    )
    assert release_retry == released

    with pytest.raises(ValueError, match="conflicting release"):
        await ledger.release(
            reservation_id=second.record.reservation_id,
            reason="different outcome",
        )
    with pytest.raises(ValueError, match="not active"):
        await ledger.reconcile(
            reservation_id=second.record.reservation_id,
            actual_amount=Decimal("0.22"),
            reason="cannot charge explicitly released reservation",
        )

    # The original reconciliation is now outside the rolling window. If the
    # idempotent retry had moved its accounting timestamp forward, its charge
    # would still contribute to this admission result.
    clock.value = occurred_at + timedelta(seconds=90)
    third = await ledger.reserve(
        limit=limit,
        session_id="sess_after_idempotent_settlements",
        agent_name="assistant",
        provider_name="fake",
        model="fake-model",
        model_attempt_identity=model_attempt_identity(),
    )
    assert third.accepted is True
    assert third.actual == Decimal("0.22")


async def assert_crash_safe_dispatch_and_settlement_outbox(
    ledger: BudgetLedger,
    limit: BudgetLimit,
    *,
    clock: MutableClock,
    ttl_seconds: int,
) -> None:
    """Exercise the shared dispatch fence and terminal audit-outbox contract."""

    completion_identity = model_attempt_identity()
    reserved = await ledger.reserve(
        limit=limit,
        session_id="sess_crash_safe_completion",
        agent_name="assistant",
        environment_name="sandbox",
        settlement_event_payload={"audit_context": "trusted"},
        provider_name="fake",
        model="fake-model",
        model_attempt_identity=completion_identity,
    )
    assert reserved.accepted is True
    assert reserved.record is not None
    reserved.record.settlement_event_payload["forged"] = "caller mutation"
    dispatch_id = f"{completion_identity.model_step_id}:dispatch:0"
    dispatched_records = await ledger.mark_dispatched(
        reservation_ids=(reserved.record.reservation_id,),
        dispatch_id=dispatch_id,
        dispatched_at=clock.value,
    )
    assert len(dispatched_records) == 1
    dispatched = dispatched_records[0]
    assert dispatched.dispatch_id == dispatch_id
    assert await ledger.mark_dispatched(
        reservation_ids=(reserved.record.reservation_id,),
        dispatch_id=dispatch_id,
        dispatched_at=clock.value + timedelta(seconds=1),
    ) == (dispatched,)
    with pytest.raises(ValueError, match="Dispatched budget reservation cannot be released"):
        await ledger.release(
            reservation_id=reserved.record.reservation_id,
            reason="must retain post-dispatch capacity",
        )

    clock.value += timedelta(seconds=ttl_seconds)
    blocked = await ledger.reserve(
        limit=limit,
        session_id="sess_crash_safe_blocked",
        agent_name="assistant",
        provider_name="fake",
        model="fake-model",
        model_attempt_identity=model_attempt_identity(),
    )
    assert blocked.accepted is False

    reconciled_at = clock.value
    reconciled, duplicate = await asyncio.gather(
        ledger.reconcile(
            reservation_id=reserved.record.reservation_id,
            actual_amount=Decimal("0.01"),
            settlement_kind="completed",
            reason="model completed",
            occurred_at=reconciled_at,
        ),
        ledger.reconcile(
            reservation_id=reserved.record.reservation_id,
            actual_amount=Decimal("0.01"),
            settlement_kind="completed",
            reason="model completed",
            occurred_at=reconciled_at,
        ),
    )
    assert duplicate == reconciled
    assert reconciled.settlement_kind == "completed"
    settlement = await ledger.load_settlement(reconciled.settlement_id)
    assert settlement is not None
    assert settlement.reconciliation == reconciled
    assert settlement.event.payload["settlement_id"] == reconciled.settlement_id
    assert settlement.event.payload["actual_amount"] == "0.01"
    assert settlement.event.payload["audit_context"] == "trusted"
    assert "forged" not in settlement.event.payload
    assert settlement.event.environment_name == "sandbox"
    assert settlement.event_published is False
    assert await ledger.list_pending_settlements(
        session_id=reserved.record.session_id,
    ) == [settlement]

    with pytest.raises(ValueError, match="conflicting settlement"):
        await ledger.reconcile(
            reservation_id=reserved.record.reservation_id,
            actual_amount=Decimal("0.01"),
            settlement_kind="conservative",
            reason="model completed",
            occurred_at=reconciled_at,
        )

    published = await ledger.mark_settlement_event_published(
        settlement_id=settlement.settlement_id,
        event_id=settlement.event.id,
    )
    assert published.event_published is True
    assert (
        await ledger.mark_settlement_event_published(
            settlement_id=settlement.settlement_id,
            event_id=settlement.event.id,
        )
        == published
    )
    assert await ledger.list_pending_settlements(session_id=reserved.record.session_id) == []

    pending = await ledger.reserve(
        limit=limit,
        session_id="sess_crash_safe_predispatch",
        agent_name="assistant",
        provider_name="fake",
        model="fake-model",
        model_attempt_identity=model_attempt_identity(),
    )
    assert pending.accepted is True
    assert pending.record is not None
    clock.value += timedelta(seconds=ttl_seconds)
    replacement = await ledger.reserve(
        limit=limit,
        session_id="sess_crash_safe_replacement",
        agent_name="assistant",
        provider_name="fake",
        model="fake-model",
        model_attempt_identity=model_attempt_identity(),
    )
    assert replacement.accepted is True
    pending_releases = await ledger.list_pending_settlements(
        session_id=pending.record.session_id,
    )
    assert len(pending_releases) == 1, pending_releases
    assert pending_releases[0].settlement_kind == "released"
    assert pending_releases[0].reconciliation.reason == (
        f"Reservation expired: not reconciled within {ttl_seconds}s."
    )

    audit_limit = limit.model_copy(
        update={"max_estimated_cost": Decimal("2")},
        deep=True,
    )
    conservative = await ledger.reserve(
        limit=audit_limit,
        session_id="sess_conservative_settlement",
        agent_name="assistant",
        provider_name="fake",
        model="fake-model",
        model_attempt_identity=model_attempt_identity(),
    )
    assert conservative.record is not None
    await ledger.mark_dispatched(
        reservation_ids=(conservative.record.reservation_id,),
        dispatch_id="dispatch:conservative",
    )
    conservative_reconciliation = await ledger.reconcile(
        reservation_id=conservative.record.reservation_id,
        actual_amount=conservative.record.reserved_amount,
        settlement_kind="conservative",
        reason="provider usage unknown after dispatch; charged reserved amount",
        occurred_at=clock.value,
    )
    assert (
        await ledger.reconcile(
            reservation_id=conservative.record.reservation_id,
            actual_amount=conservative.record.reserved_amount,
            settlement_kind="conservative",
            reason="provider usage unknown after dispatch; charged reserved amount",
            occurred_at=clock.value + timedelta(seconds=1),
        )
        == conservative_reconciliation
    )
    conservative_settlement = await ledger.load_settlement(
        conservative_reconciliation.settlement_id
    )
    assert conservative_settlement is not None
    assert conservative_settlement.settlement_kind == "conservative"
    assert conservative_settlement.event_published is False

    releasable = await ledger.reserve(
        limit=audit_limit,
        session_id="sess_release_settlement",
        agent_name="assistant",
        provider_name="fake",
        model="fake-model",
        model_attempt_identity=model_attempt_identity(),
    )
    assert releasable.record is not None
    released = await ledger.release(
        reservation_id=releasable.record.reservation_id,
        reason="provider not dispatched",
    )
    assert (
        await ledger.release(
            reservation_id=releasable.record.reservation_id,
            reason="provider not dispatched",
        )
        == released
    )
    release_settlement = await ledger.load_settlement(released.settlement_id)
    assert release_settlement is not None
    assert release_settlement.settlement_kind == "released"
    assert release_settlement.event_published is False

    atomic_limit = limit.model_copy(
        update={"max_estimated_cost": Decimal("1")},
        deep=True,
    )
    first = await ledger.reserve(
        limit=atomic_limit,
        session_id="sess_atomic_dispatch_first",
        agent_name="assistant",
        provider_name="fake",
        model="fake-model",
        model_attempt_identity=model_attempt_identity(),
    )
    second = await ledger.reserve(
        limit=atomic_limit,
        session_id="sess_atomic_dispatch_second",
        agent_name="assistant",
        provider_name="fake",
        model="fake-model",
        model_attempt_identity=model_attempt_identity(),
    )
    assert first.record is not None
    assert second.record is not None
    await ledger.mark_dispatched(
        reservation_ids=(first.record.reservation_id,),
        dispatch_id="dispatch:atomic:first",
    )
    with pytest.raises(ValueError, match="conflicting dispatch"):
        await ledger.mark_dispatched(
            reservation_ids=(
                first.record.reservation_id,
                second.record.reservation_id,
            ),
            dispatch_id="dispatch:atomic:combined",
        )
    clock.value += timedelta(seconds=ttl_seconds)
    trigger_reap = await ledger.reserve(
        limit=atomic_limit,
        session_id="sess_atomic_dispatch_trigger",
        agent_name="assistant",
        provider_name="fake",
        model="fake-model",
        model_attempt_identity=model_attempt_identity(),
    )
    assert trigger_reap.accepted is True
    second_releases = await ledger.list_pending_settlements(
        session_id=second.record.session_id,
    )
    assert len(second_releases) == 1
    assert second_releases[0].reservation_id == second.record.reservation_id
