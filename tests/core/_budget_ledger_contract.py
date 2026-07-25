from __future__ import annotations

import asyncio
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Protocol

import pytest
from tests.core._execution_unit_fixtures import model_attempt_identity

from cayu._validation import DurableValueError
from cayu.runtime import BudgetLedger, BudgetLimit


class MutableClock(Protocol):
    value: datetime

    def __call__(self) -> datetime: ...


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
