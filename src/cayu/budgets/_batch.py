"""Private, atomic multi-ceiling reservation material for ledger owners.

This is an accounting primitive, not an authenticated collaboration entrance.
The runtime must qualify binding authority before it can advertise that capability.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal

from cayu.budgets.base import (
    BudgetReservationIdentityConflict,
    BudgetReservationRecord,
    BudgetReservationResult,
    _copy_effective_budget_limit,
    _EffectiveBudgetLimit,
    _reservation_result,
)
from cayu.runtime.execution_units import ModelAttemptIdentity


@dataclass(frozen=True)
class BudgetBatchMember:
    limit: _EffectiveBudgetLimit
    record: BudgetReservationRecord


@dataclass(frozen=True)
class BudgetBatchResult:
    records: tuple[BudgetReservationRecord, ...]
    # Post-reservation projected usage for each record, in record order.
    projected_usage: tuple[Decimal, ...] = ()
    failure: BudgetReservationResult | None = None


def prepare_batch(members: tuple[BudgetBatchMember, ...]) -> tuple[BudgetBatchMember, ...]:
    """Detach all input before locking; bind membership and every ceiling definition."""
    if type(members) is not tuple or not 1 <= len(members) <= 16:
        raise ValueError("A budget batch requires between one and sixteen ceilings.")
    prepared = []
    for member in members:
        if type(member) is not BudgetBatchMember:
            raise TypeError("Budget batch members must be BudgetBatchMember instances.")
        limit = _copy_effective_budget_limit(member.limit)
        if type(member.record) is not BudgetReservationRecord:
            raise TypeError("Budget batch members require reservation records.")
        record = BudgetReservationRecord.model_validate(member.record.model_dump(mode="python"))
        if (
            limit.reservation is None
            or limit.action != "interrupt"
            or record.budget_limit_id != limit.budget_limit_id
            or record.scope != limit.scope
            or record.key != limit.key
            or record.window != limit.window
            or record.currency != limit.currency
            or record.status != "active"
            or record.dispatch_id is not None
            or record.actual_amount is not None
            or record.reason is not None
        ):
            raise ValueError("Budget batch member conflicts with its ceiling or initial state.")
        prepared.append(BudgetBatchMember(limit, record))
    if len({m.record.reservation_id for m in prepared}) != len(prepared) or len(
        {m.limit.budget_limit_id for m in prepared}
    ) != len(prepared):
        raise ValueError("Budget batch reservation and ceiling identities must be distinct.")

    def identity(r: BudgetReservationRecord) -> tuple[object, ...]:
        return (
            r.session_id,
            r.agent_name,
            r.provider_name,
            r.model,
            r.environment_name,
            r.model_step_id,
            r.model_attempt_id,
            r.billing_identity,
        )

    if any(identity(m.record) != identity(prepared[0].record) for m in prepared):
        raise ValueError("Budget batch members must describe the same dispatch.")
    return tuple(prepared)


def replay_batch(
    members: tuple[BudgetBatchMember, ...],
    existing: tuple[BudgetReservationRecord | None, ...],
) -> BudgetBatchResult | None:
    if all(r is None for r in existing):
        return None
    if any(r is None for r in existing):
        raise BudgetReservationIdentityConflict("Budget batch has partial or conflicting identity.")
    for member, record in zip(members, existing, strict=True):
        assert record is not None
        expected = member.record.model_copy(
            update={
                "created_at": record.created_at,
                "updated_at": record.updated_at,
                "dispatch_id": record.dispatch_id,
                "dispatched_at": record.dispatched_at,
            }
        )
        if record != expected or record.status != "active":
            raise BudgetReservationIdentityConflict(
                "Budget batch replay conflicts with durable state."
            )
    return BudgetBatchResult(tuple(r.model_copy(deep=True) for r in existing if r is not None))


def batch_failure(member: BudgetBatchMember, used: Decimal) -> BudgetReservationResult | None:
    record = member.record
    projected = used + record.reserved_amount
    if projected <= member.limit.max_estimated_cost:
        return None
    return _reservation_result(
        limit=member.limit,
        model_attempt_identity=ModelAttemptIdentity(
            model_step_id=record.model_step_id,
            model_attempt_id=record.model_attempt_id,
        ),
        accepted=False,
        requested=record.reserved_amount,
        actual=projected,
        message="Atomic budget admission exceeded an applicable ceiling.",
    )


def stamp_batch(
    members: tuple[BudgetBatchMember, ...],
    now: datetime,
    projected_usage: tuple[Decimal, ...] = (),
) -> BudgetBatchResult:
    return BudgetBatchResult(
        tuple(
            m.record.model_copy(update={"created_at": now, "updated_at": now}, deep=True)
            for m in members
        ),
        projected_usage=projected_usage,
    )
