from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest
from pydantic import ValidationError

from cayu._validation import MAX_DURABLE_JSON_INTEGER, compact_json_utf8_size
from cayu.runtime._durable_operation_ownership import (
    DURABLE_OPERATION_OWNERSHIP_MAX_ID_CHARS,
    DurableOperationOwnership,
    DurableOperationOwnershipAction,
    DurableOperationOwnershipDisposition,
    DurableOperationOwnershipResult,
    DurableOperationOwnershipState,
    DurableOperationOwnershipTransition,
    transition_durable_operation_ownership,
)

_NOW = datetime(2026, 9, 3, tzinfo=UTC)


def _transition(
    action: DurableOperationOwnershipAction,
    *,
    operation_id: str = "operation-1",
    claim_id: str = "claim-1",
    owner_id: str = "worker-1",
    generation: int | None = None,
) -> DurableOperationOwnershipTransition:
    return DurableOperationOwnershipTransition(
        operation_id=operation_id,
        claim_id=claim_id,
        owner_id=owner_id,
        action=action,
        generation=generation,
        lease_seconds=(
            30
            if action
            in {
                DurableOperationOwnershipAction.CLAIM,
                DurableOperationOwnershipAction.RENEW,
            }
            else None
        ),
    )


def _claim(
    current: DurableOperationOwnership | None = None,
    *,
    request: DurableOperationOwnershipTransition | None = None,
    now: datetime = _NOW,
    operation_active: bool = True,
) -> DurableOperationOwnershipResult:
    return transition_durable_operation_ownership(
        current,
        request or _transition(DurableOperationOwnershipAction.CLAIM),
        store_now=now,
        operation_active=operation_active,
    )


def _ownership(**changes: object) -> DurableOperationOwnership:
    ownership = _claim().ownership
    assert ownership is not None
    return DurableOperationOwnership.model_validate(
        {**ownership.model_dump(mode="python"), **changes}
    )


def test_ownership_value_is_strict_bounded_and_secret_free_in_shape() -> None:
    acquired = _claim()
    ownership = acquired.ownership
    assert ownership is not None
    assert ownership.model_dump(mode="json") == {
        "record_type": "cayu.durable-operation-ownership",
        "schema_version": 1,
        "operation_id": "operation-1",
        "claim_id": "claim-1",
        "owner_id": "worker-1",
        "generation": 1,
        "state": "active",
        "acquired_at": "2026-09-03T00:00:00Z",
        "renewed_at": "2026-09-03T00:00:00Z",
        "lease_expires_at": "2026-09-03T00:00:30Z",
        "released_at": None,
        "settled_at": None,
    }
    assert compact_json_utf8_size(ownership.model_dump(mode="json")) < 1_024

    for invalid in (
        {"operation_id": "x" * (DURABLE_OPERATION_OWNERSHIP_MAX_ID_CHARS + 1)},
        {"generation": True},
        {"schema_version": 1.0},
        {"lease_expires_at": _NOW},
    ):
        with pytest.raises(ValidationError):
            DurableOperationOwnership.model_validate(
                {**ownership.model_dump(mode="python"), **invalid}
            )


def test_claim_acquires_unowned_operation_and_reconciles_exact_acknowledgement() -> None:
    acquired = _claim()
    assert acquired.disposition is DurableOperationOwnershipDisposition.ACQUIRED
    assert acquired.owns_exact_claim is True

    replayed = _claim(acquired.ownership, now=_NOW + timedelta(seconds=1))
    assert replayed.disposition is (DurableOperationOwnershipDisposition.EQUIVALENT_LIVE_OWNER)
    assert replayed.ownership == acquired.ownership
    assert replayed.owns_exact_claim is True


def test_live_foreign_claim_is_fenced_without_mutation() -> None:
    acquired = _claim()
    foreign = _claim(
        acquired.ownership,
        request=_transition(
            DurableOperationOwnershipAction.CLAIM,
            claim_id="claim-2",
            owner_id="worker-2",
        ),
        now=_NOW + timedelta(seconds=29),
    )
    assert foreign.disposition is DurableOperationOwnershipDisposition.FENCED
    assert foreign.ownership == acquired.ownership
    assert foreign.owns_exact_claim is False


def test_takeover_uses_store_time_and_increments_the_fence() -> None:
    acquired = _claim()
    assert acquired.ownership is not None
    at_expiry = _claim(
        acquired.ownership,
        request=_transition(
            DurableOperationOwnershipAction.CLAIM,
            claim_id="claim-2",
            owner_id="worker-2",
        ),
        now=acquired.ownership.lease_expires_at,
    )
    assert at_expiry.disposition is (DurableOperationOwnershipDisposition.EXPIRED_TAKEN_OVER)
    assert at_expiry.ownership is not None
    assert at_expiry.ownership.generation == 2
    assert at_expiry.ownership.claim_id == "claim-2"

    stale_renew = transition_durable_operation_ownership(
        at_expiry.ownership,
        _transition(DurableOperationOwnershipAction.RENEW, generation=1),
        store_now=_NOW + timedelta(seconds=31),
        operation_active=True,
    )
    assert stale_renew.disposition is DurableOperationOwnershipDisposition.FENCED
    assert stale_renew.ownership == at_expiry.ownership


def test_exact_live_renewal_never_moves_lease_evidence_backward() -> None:
    acquired = _claim()
    assert acquired.ownership is not None
    renewed = transition_durable_operation_ownership(
        acquired.ownership,
        _transition(DurableOperationOwnershipAction.RENEW, generation=1),
        store_now=_NOW + timedelta(seconds=10),
        operation_active=True,
    )
    assert renewed.disposition is DurableOperationOwnershipDisposition.RENEWED
    assert renewed.ownership is not None
    assert renewed.ownership.renewed_at == _NOW + timedelta(seconds=10)
    assert renewed.ownership.lease_expires_at == _NOW + timedelta(seconds=40)

    clock_regressed = transition_durable_operation_ownership(
        renewed.ownership,
        _transition(DurableOperationOwnershipAction.RENEW, generation=1),
        store_now=_NOW + timedelta(seconds=5),
        operation_active=True,
    )
    assert clock_regressed.disposition is DurableOperationOwnershipDisposition.RENEWED
    assert clock_regressed.ownership == renewed.ownership


@pytest.mark.parametrize(
    ("action", "disposition", "state"),
    (
        (
            DurableOperationOwnershipAction.RELEASE,
            DurableOperationOwnershipDisposition.RELEASED,
            DurableOperationOwnershipState.RELEASED,
        ),
        (
            DurableOperationOwnershipAction.SETTLE,
            DurableOperationOwnershipDisposition.SETTLED,
            DurableOperationOwnershipState.SETTLED,
        ),
    ),
)
def test_exact_claim_can_release_or_settle(
    action: DurableOperationOwnershipAction,
    disposition: DurableOperationOwnershipDisposition,
    state: DurableOperationOwnershipState,
) -> None:
    acquired = _claim()
    assert acquired.ownership is not None
    result = transition_durable_operation_ownership(
        acquired.ownership,
        _transition(action, generation=1),
        store_now=_NOW + timedelta(seconds=1),
        operation_active=True,
    )
    assert result.disposition is disposition
    assert result.ownership is not None
    assert result.ownership.state is state
    assert result.ownership.lease_expires_at is None

    replayed = transition_durable_operation_ownership(
        result.ownership,
        _transition(action, generation=1),
        store_now=_NOW + timedelta(seconds=2),
        operation_active=False,
    )
    assert replayed.disposition is disposition
    assert replayed.ownership == result.ownership


def test_released_claim_requires_a_new_claim_identity() -> None:
    acquired = _claim()
    assert acquired.ownership is not None
    released = transition_durable_operation_ownership(
        acquired.ownership,
        _transition(DurableOperationOwnershipAction.RELEASE, generation=1),
        store_now=_NOW + timedelta(seconds=1),
        operation_active=True,
    )
    stale_replay = _claim(released.ownership, now=_NOW + timedelta(seconds=2))
    assert stale_replay.disposition is DurableOperationOwnershipDisposition.FENCED

    reacquired = _claim(
        released.ownership,
        request=_transition(
            DurableOperationOwnershipAction.CLAIM,
            claim_id="claim-2",
        ),
        now=_NOW + timedelta(seconds=2),
    )
    assert reacquired.disposition is DurableOperationOwnershipDisposition.ACQUIRED
    assert reacquired.ownership is not None
    assert reacquired.ownership.generation == 2


def test_claim_identity_cannot_be_rebound_to_another_owner() -> None:
    acquired = _claim()
    conflict = _claim(
        acquired.ownership,
        request=_transition(
            DurableOperationOwnershipAction.CLAIM,
            owner_id="worker-2",
        ),
        now=_NOW + timedelta(minutes=1),
    )
    assert conflict.disposition is DurableOperationOwnershipDisposition.IDENTITY_CONFLICT
    assert conflict.ownership == acquired.ownership


def test_operation_identity_conflict_and_advanced_phase_are_typed() -> None:
    acquired = _claim()
    conflict = _claim(
        acquired.ownership,
        request=_transition(
            DurableOperationOwnershipAction.CLAIM,
            operation_id="operation-2",
        ),
    )
    assert conflict.disposition is DurableOperationOwnershipDisposition.IDENTITY_CONFLICT

    advanced = _claim(acquired.ownership, operation_active=False)
    assert advanced.disposition is DurableOperationOwnershipDisposition.OPERATION_ADVANCED
    assert advanced.ownership == acquired.ownership

    inactive_conflict = _claim(
        acquired.ownership,
        request=_transition(
            DurableOperationOwnershipAction.CLAIM,
            operation_id="operation-2",
        ),
        operation_active=False,
    )
    assert inactive_conflict.disposition is (DurableOperationOwnershipDisposition.IDENTITY_CONFLICT)


@pytest.mark.parametrize(
    "action",
    (
        DurableOperationOwnershipAction.RENEW,
        DurableOperationOwnershipAction.RELEASE,
        DurableOperationOwnershipAction.SETTLE,
    ),
)
def test_expired_claim_cannot_act_and_settled_operation_cannot_reopen(
    action: DurableOperationOwnershipAction,
) -> None:
    acquired = _claim()
    assert acquired.ownership is not None
    expired = transition_durable_operation_ownership(
        acquired.ownership,
        _transition(action, generation=1),
        store_now=acquired.ownership.lease_expires_at,
        operation_active=True,
    )
    assert expired.disposition is DurableOperationOwnershipDisposition.FENCED

    settled = transition_durable_operation_ownership(
        acquired.ownership,
        _transition(DurableOperationOwnershipAction.SETTLE, generation=1),
        store_now=_NOW + timedelta(seconds=1),
        operation_active=True,
    )
    reopened = _claim(
        settled.ownership,
        request=_transition(
            DurableOperationOwnershipAction.CLAIM,
            claim_id="claim-2",
        ),
    )
    assert reopened.disposition is DurableOperationOwnershipDisposition.OPERATION_ADVANCED


def test_generation_exhaustion_and_indeterminate_outcomes_fail_closed() -> None:
    acquired = _claim()
    assert acquired.ownership is not None
    exhausted = DurableOperationOwnership.model_validate(
        {
            **acquired.ownership.model_dump(mode="python"),
            "generation": MAX_DURABLE_JSON_INTEGER,
            "lease_expires_at": _NOW + timedelta(seconds=1),
        }
    )
    result = _claim(
        exhausted,
        request=_transition(
            DurableOperationOwnershipAction.CLAIM,
            claim_id="claim-2",
        ),
        now=_NOW + timedelta(seconds=1),
    )
    assert result.disposition is DurableOperationOwnershipDisposition.IDENTITY_CONFLICT

    indeterminate = DurableOperationOwnershipResult(
        disposition=DurableOperationOwnershipDisposition.INDETERMINATE,
        observed_at=_NOW,
    )
    assert indeterminate.ownership is None
    assert indeterminate.owns_exact_claim is False


@pytest.mark.parametrize(
    "disposition",
    (
        DurableOperationOwnershipDisposition.ACQUIRED,
        DurableOperationOwnershipDisposition.RENEWED,
        DurableOperationOwnershipDisposition.EQUIVALENT_LIVE_OWNER,
        DurableOperationOwnershipDisposition.EXPIRED_TAKEN_OVER,
    ),
)
def test_live_result_rejects_lease_expired_at_observation(
    disposition: DurableOperationOwnershipDisposition,
) -> None:
    with pytest.raises(ValueError):
        DurableOperationOwnershipResult(
            disposition=disposition,
            observed_at=_NOW + timedelta(seconds=30),
            ownership=_ownership(),
        )


@pytest.mark.parametrize(
    "disposition",
    (
        DurableOperationOwnershipDisposition.RELEASED,
        DurableOperationOwnershipDisposition.SETTLED,
    ),
)
def test_terminal_result_rejects_active_ownership_evidence(
    disposition: DurableOperationOwnershipDisposition,
) -> None:
    with pytest.raises(ValueError):
        DurableOperationOwnershipResult(
            disposition=disposition,
            observed_at=_NOW,
            ownership=_ownership(),
        )
