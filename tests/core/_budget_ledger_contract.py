from __future__ import annotations

import asyncio
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Protocol

import pytest

from cayu.runtime import BudgetLedger, BudgetLimit


class MutableClock(Protocol):
    value: datetime

    def __call__(self) -> datetime: ...


async def assert_idempotent_terminal_settlements(
    ledger: BudgetLedger,
    limit: BudgetLimit,
    *,
    clock: MutableClock,
) -> None:
    first = await ledger.reserve(
        limit=limit,
        session_id="sess_idempotent_reconcile",
        agent_name="assistant",
        provider_name="fake",
        model="fake-model",
    )
    assert first.accepted is True
    assert first.record is not None

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
    second = await ledger.reserve(
        limit=limit,
        session_id="sess_idempotent_release",
        agent_name="assistant",
        provider_name="fake",
        model="fake-model",
    )
    assert second.accepted is True
    assert second.actual == Decimal("0.23")
    assert second.record is not None

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
    )
    assert third.accepted is True
    assert third.actual == Decimal("0.22")
