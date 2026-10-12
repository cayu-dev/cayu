"""In-memory persisted-event delivery over explicit locked state."""

from __future__ import annotations

import math
from collections.abc import Callable
from contextlib import AbstractAsyncContextManager
from datetime import UTC, datetime, timedelta
from typing import Any, NamedTuple

from cayu._validation import require_durable_clean_nonblank as require_clean_nonblank
from cayu.sessions import _closure_records as closure_records
from cayu.sessions import event_delivery as session_event_rules
from cayu.sessions import event_delivery as side_effect_health
from cayu.sessions.event_delivery import (
    PersistedEventSideEffectClaim,
    PersistedEventSideEffectClaimLost,
    PersistedEventSideEffectDelivery,
    PersistedEventSideEffectHealth,
    PersistedEventSideEffectPage,
    PersistedEventSideEffectQuery,
    PersistedEventSideEffectStatus,
    validate_persisted_event_side_effect_error,
)
from cayu.sessions.records import EventRecord


class MemoryEventDeliveryState(NamedTuple):
    """Borrowed state resolved under the caller's lock for each operation.

    Claim decisions, clock observations and updates share that lock. State
    construction, event publication and deletion remain with the caller.
    No view may outlive its lock scope.
    """

    persisted_event_side_effect_deliveries: dict[tuple[str, str], PersistedEventSideEffectDelivery]
    event_records_by_id: dict[tuple[str, str], EventRecord]
    session_closure_progress: dict[tuple[str, str], dict[str, Any]]
    ownership_clock: Callable[[], datetime]


LockedEventDeliveryState = Callable[[], AbstractAsyncContextManager[MemoryEventDeliveryState]]


async def claim_first_persisted_event_side_effect(
    locked: LockedEventDeliveryState, expected: PersistedEventSideEffectDelivery
) -> PersistedEventSideEffectClaim | None:
    expected = session_event_rules._copy_pending_first_event_delivery(expected)
    return await _claim_persisted_event_side_effect(
        locked, session_id=expected.session_id, event_id=expected.event_id, expected=expected
    )


async def claim_persisted_event_side_effect(
    locked: LockedEventDeliveryState,
    *,
    session_id: str | None = None,
    event_id: str | None = None,
    lease_seconds: float = 300.0,
) -> PersistedEventSideEffectClaim | None:
    return await _claim_persisted_event_side_effect(
        locked, session_id=session_id, event_id=event_id, lease_seconds=lease_seconds
    )


async def _claim_persisted_event_side_effect(
    locked: LockedEventDeliveryState,
    *,
    session_id: str | None = None,
    event_id: str | None = None,
    lease_seconds: float = 300.0,
    expected: PersistedEventSideEffectDelivery | None = None,
) -> PersistedEventSideEffectClaim | None:
    if session_id is not None:
        session_id = require_clean_nonblank(session_id, "session_id")
    if event_id is not None:
        event_id = require_clean_nonblank(event_id, "event_id")
    if (session_id is None) != (event_id is None):
        raise ValueError("session_id and event_id must be supplied together.")
    if type(lease_seconds) not in {int, float} or lease_seconds <= 0:
        raise ValueError("lease_seconds must be greater than 0.")
    async with locked() as state:
        now = state.ownership_clock()
        deliveries = sorted(
            state.persisted_event_side_effect_deliveries.values(),
            key=lambda delivery: delivery.event_sequence,
        )
        closure_targets = {
            target
            for owner in state.session_closure_progress.values()
            for target in closure_records._closure_progress_targets(owner)
        }
        for delivery in deliveries:
            if expected is not None and delivery != expected:
                continue
            if delivery.session_id in closure_targets:
                continue
            if session_id is not None and (
                delivery.session_id != session_id or delivery.event_id != event_id
            ):
                continue
            if delivery.status in {
                PersistedEventSideEffectStatus.DELIVERED,
                PersistedEventSideEffectStatus.DEAD_LETTERED,
            }:
                continue
            if (
                delivery.status is PersistedEventSideEffectStatus.LEASED
                and delivery.lease_expires_at is not None
                and delivery.lease_expires_at > now
            ):
                continue
            if (
                delivery.status is PersistedEventSideEffectStatus.FAILED
                and delivery.next_attempt_at is not None
                and delivery.next_attempt_at > now
            ):
                continue
            record = state.event_records_by_id.get((delivery.session_id, delivery.event_id))
            if record is None:
                raise RuntimeError("Persisted side-effect delivery lost its source event.")
            claim = PersistedEventSideEffectClaim(
                session_id=delivery.session_id,
                event_id=delivery.event_id,
                event_sequence=delivery.event_sequence,
                event=record.event,
                attempt=delivery.attempts + 1,
                lease_expires_at=now + timedelta(seconds=float(lease_seconds)),
            )
            state.persisted_event_side_effect_deliveries[
                (delivery.session_id, delivery.event_id)
            ] = delivery.model_copy(
                update={
                    "status": PersistedEventSideEffectStatus.LEASED,
                    "attempts": claim.attempt,
                    "claim_id": claim.claim_id,
                    "lease_expires_at": claim.lease_expires_at,
                    "next_attempt_at": None,
                    "last_error": None,
                    "updated_at": now,
                },
                deep=True,
            )
            return claim
        return None


async def get_persisted_event_side_effect_delivery(
    locked: LockedEventDeliveryState,
    *,
    session_id: str,
    event_id: str,
) -> PersistedEventSideEffectDelivery | None:
    session_id = require_clean_nonblank(session_id, "session_id")
    event_id = require_clean_nonblank(event_id, "event_id")
    async with locked() as state:
        delivery = state.persisted_event_side_effect_deliveries.get((session_id, event_id))
        return None if delivery is None else delivery.model_copy(deep=True)


async def retire_failed_first_event_delivery(
    locked: LockedEventDeliveryState, expected: PersistedEventSideEffectDelivery
) -> PersistedEventSideEffectDelivery | None:
    expected = session_event_rules._copy_failed_first_delivery_retirement(expected)
    key = (expected.session_id, expected.event_id)
    async with locked() as state:
        if state.persisted_event_side_effect_deliveries.get(key) != expected:
            return None
        retired = expected.model_copy(
            update={
                "status": PersistedEventSideEffectStatus.DEAD_LETTERED,
                "next_attempt_at": None,
                "updated_at": state.ownership_clock(),
            }
        )
        state.persisted_event_side_effect_deliveries[key] = retired
        return retired.model_copy(deep=True)


async def mark_persisted_event_side_effect_delivered(
    locked: LockedEventDeliveryState,
    claim: PersistedEventSideEffectClaim,
) -> PersistedEventSideEffectDelivery:
    claim = PersistedEventSideEffectClaim.model_validate(claim)
    async with locked() as state:
        delivery = _matching_persisted_event_side_effect_claim_unlocked(state, claim)
        updated = delivery.model_copy(
            update={
                "status": PersistedEventSideEffectStatus.DELIVERED,
                "claim_id": None,
                "lease_expires_at": None,
                "next_attempt_at": None,
                "last_error": None,
                "updated_at": state.ownership_clock(),
            },
            deep=True,
        )
        state.persisted_event_side_effect_deliveries[(claim.session_id, claim.event_id)] = updated
        return updated.model_copy(deep=True)


async def mark_persisted_event_side_effect_failed(
    locked: LockedEventDeliveryState,
    claim: PersistedEventSideEffectClaim,
    *,
    error: str,
    max_attempts: int,
    retry_delay_seconds: float,
) -> PersistedEventSideEffectDelivery:
    claim = PersistedEventSideEffectClaim.model_validate(claim)
    error = validate_persisted_event_side_effect_error(error)
    if type(max_attempts) is not int or max_attempts < 1:
        raise ValueError("max_attempts must be an integer greater than or equal to 1.")
    if (
        type(retry_delay_seconds) not in {int, float}
        or not math.isfinite(retry_delay_seconds)
        or retry_delay_seconds < 0
    ):
        raise ValueError("retry_delay_seconds must be a finite non-negative number.")
    dead_lettered = claim.attempt >= max_attempts
    async with locked() as state:
        now = state.ownership_clock()
        delivery = _matching_persisted_event_side_effect_claim_unlocked(state, claim)
        updated = delivery.model_copy(
            update={
                "status": (
                    PersistedEventSideEffectStatus.DEAD_LETTERED
                    if dead_lettered
                    else PersistedEventSideEffectStatus.FAILED
                ),
                "claim_id": None,
                "lease_expires_at": None,
                "next_attempt_at": (
                    None if dead_lettered else now + timedelta(seconds=float(retry_delay_seconds))
                ),
                "last_error": error,
                "updated_at": now,
            },
            deep=True,
        )
        state.persisted_event_side_effect_deliveries[(claim.session_id, claim.event_id)] = updated
        return updated.model_copy(deep=True)


async def defer_persisted_event_side_effect(
    locked: LockedEventDeliveryState,
    claim: PersistedEventSideEffectClaim,
) -> PersistedEventSideEffectDelivery:
    claim = PersistedEventSideEffectClaim.model_validate(claim)
    async with locked() as state:
        delivery = _matching_persisted_event_side_effect_claim_unlocked(state, claim)
        updated = delivery.model_copy(
            update={
                "status": PersistedEventSideEffectStatus.PENDING,
                "attempts": delivery.attempts - 1,
                "claim_id": None,
                "lease_expires_at": None,
                "next_attempt_at": None,
                "last_error": None,
                "updated_at": state.ownership_clock(),
            },
            deep=True,
        )
        state.persisted_event_side_effect_deliveries[(claim.session_id, claim.event_id)] = updated
        return updated.model_copy(deep=True)


async def get_persisted_event_side_effect_health(
    locked: LockedEventDeliveryState,
) -> PersistedEventSideEffectHealth:
    async with locked() as state:
        now = state.ownership_clock().astimezone(UTC)
        rows = list(state.persisted_event_side_effect_deliveries.values())
        counts = dict.fromkeys(side_effect_health.CONDITIONS, 0)
        timestamps: dict[str, datetime | None] = dict.fromkeys(
            [
                "oldest_claimable_at",
                "oldest_pending_at",
                "oldest_failed_at",
                "oldest_dead_letter_at",
                "earliest_live_lease_expires_at",
            ]
        )
        max_attempts = 0

        def oldest(key: str, value: datetime) -> None:
            previous = timestamps[key]
            timestamps[key] = value if previous is None else min(previous, value)

        for row in rows:
            if row.status == "leased":
                counts[
                    "leased_expired" if side_effect_health.claimable(row, now) else "leased_live"
                ] += 1
                if row.lease_expires_at is not None and row.lease_expires_at > now:
                    oldest("earliest_live_lease_expires_at", row.lease_expires_at)
            elif row.status == "failed":
                counts["failed_retryable"] += 1
                counts["failed_deferred"] += int(
                    row.next_attempt_at is not None and row.next_attempt_at > now
                )
                counts["repeatedly_failing"] += int(row.attempts > 1)
                oldest("oldest_failed_at", row.updated_at)
            else:
                counts[str(row.status)] += 1
            if row.status == "pending":
                oldest("oldest_pending_at", row.updated_at)
            if row.status == "dead_lettered":
                oldest("oldest_dead_letter_at", row.updated_at)
            if row.status != "delivered":
                counts["outstanding_total"] += 1
                max_attempts = max(max_attempts, row.attempts)
            if row.status in {"pending", "failed", "leased"}:
                live = (
                    row.status == "leased"
                    and row.lease_expires_at is not None
                    and row.lease_expires_at > now
                )
                threshold = side_effect_health.PERSISTED_EVENT_SIDE_EFFECT_MAX_ATTEMPTS - (
                    0 if live else 1
                )
                counts["final_attempt_boundary"] += int(row.attempts >= threshold)
            if side_effect_health.claimable(row, now):
                counts["claimable_total"] += 1
                eligible = row.updated_at
                if row.status == "leased" and row.lease_expires_at is not None:
                    eligible = row.lease_expires_at
                elif row.status == "failed" and row.next_attempt_at is not None:
                    eligible = max(eligible, row.next_attempt_at)
                oldest("oldest_claimable_at", eligible)
        return side_effect_health.finish_health(
            {**counts, **timestamps, "max_outstanding_attempts": max_attempts},
            now,
        )


async def query_persisted_event_side_effect_deliveries(
    locked: LockedEventDeliveryState,
    query: PersistedEventSideEffectQuery,
) -> PersistedEventSideEffectPage:
    query = PersistedEventSideEffectQuery.model_validate(query)
    key = side_effect_health.cursor_key(query)
    async with locked() as state:
        now = state.ownership_clock().astimezone(UTC)
        rows = sorted(
            (
                row
                for row in state.persisted_event_side_effect_deliveries.values()
                if (key is None or (row.session_id, row.event_id) > key)
                and (query.statuses is None or row.status in query.statuses)
                and (not query.outstanding_only or row.status != "delivered")
                and (not query.claimable_only or side_effect_health.claimable(row, now))
            ),
            key=lambda row: (row.session_id, row.event_id),
        )[: query.limit + 1]
        return side_effect_health.page(rows, query, now)


async def list_persisted_event_side_effect_deliveries(
    locked: LockedEventDeliveryState,
    *,
    statuses: set[PersistedEventSideEffectStatus] | None = None,
    claimable_only: bool = False,
    after_sequence: int | None = None,
    limit: int = 100,
) -> list[PersistedEventSideEffectDelivery]:
    if type(limit) is not int or not 1 <= limit <= 1000:
        raise ValueError("limit must be between 1 and 1000.")
    if type(claimable_only) is not bool:
        raise TypeError("claimable_only must be a bool.")
    if after_sequence is not None and (type(after_sequence) is not int or after_sequence < 0):
        raise ValueError("after_sequence must be a non-negative integer.")
    selected_statuses = (
        None
        if statuses is None
        else {PersistedEventSideEffectStatus(status) for status in statuses}
    )
    async with locked() as state:
        now = state.ownership_clock()
        deliveries = sorted(
            state.persisted_event_side_effect_deliveries.values(),
            key=lambda delivery: delivery.event_sequence,
        )
        return [
            delivery.model_copy(deep=True)
            for delivery in deliveries
            if after_sequence is None or delivery.event_sequence > after_sequence
            if selected_statuses is None or delivery.status in selected_statuses
            if not claimable_only
            or (
                delivery.status
                in {
                    PersistedEventSideEffectStatus.PENDING,
                    PersistedEventSideEffectStatus.FAILED,
                }
                and (
                    delivery.status is not PersistedEventSideEffectStatus.FAILED
                    or delivery.next_attempt_at is None
                    or delivery.next_attempt_at <= now
                )
            )
            or (
                delivery.status is PersistedEventSideEffectStatus.LEASED
                and delivery.lease_expires_at is not None
                and delivery.lease_expires_at <= now
            )
        ][:limit]


async def renew_persisted_event_side_effect(
    locked: LockedEventDeliveryState,
    claim: PersistedEventSideEffectClaim,
    *,
    lease_seconds: float = 300.0,
) -> PersistedEventSideEffectDelivery:
    claim = PersistedEventSideEffectClaim.model_validate(claim)
    if type(lease_seconds) not in {int, float} or not 0 < lease_seconds <= 86_400:
        raise ValueError("lease_seconds must be positive and at most 86400.")
    async with locked() as state:
        if (
            claim.session_id,
            claim.event_id,
        ) not in state.persisted_event_side_effect_deliveries:
            raise PersistedEventSideEffectClaimLost(
                "Persisted event side-effect claim is no longer active."
            )
        delivery = _matching_persisted_event_side_effect_claim_unlocked(state, claim)
        now = state.ownership_clock()
        if delivery.lease_expires_at is None or delivery.lease_expires_at <= now:
            raise PersistedEventSideEffectClaimLost(
                "Persisted event side-effect claim is no longer active."
            )
        renewed = delivery.model_copy(
            update={
                "lease_expires_at": max(
                    delivery.lease_expires_at, now + timedelta(seconds=float(lease_seconds))
                ),
                "updated_at": now,
            },
            deep=True,
        )
        state.persisted_event_side_effect_deliveries[(claim.session_id, claim.event_id)] = renewed
        return renewed.model_copy(deep=True)


def _matching_persisted_event_side_effect_claim_unlocked(
    state: MemoryEventDeliveryState,
    claim: PersistedEventSideEffectClaim,
) -> PersistedEventSideEffectDelivery:
    delivery = state.persisted_event_side_effect_deliveries.get((claim.session_id, claim.event_id))
    if delivery is None:
        raise ValueError("Persisted event side-effect delivery was not found.")
    if (
        delivery.status is not PersistedEventSideEffectStatus.LEASED
        or delivery.claim_id != claim.claim_id
        or delivery.attempts != claim.attempt
    ):
        raise PersistedEventSideEffectClaimLost(
            "Persisted event side-effect claim is no longer active."
        )
    return delivery
