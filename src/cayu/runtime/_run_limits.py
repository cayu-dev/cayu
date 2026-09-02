from __future__ import annotations

import asyncio
import contextlib
import time
from collections.abc import AsyncIterator, Awaitable, Callable, Collection, Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from decimal import Decimal
from typing import Generic, Literal, TypeVar, cast

from cayu._exception_groups import add_exception_note_safely, exception_cause
from cayu._exception_state import exception_state, set_exception_state
from cayu._task_wait import (
    await_shielded_task_outcome,
    unexpected_child_cancellation_error,
)
from cayu._validation import (
    copy_durable_json_object,
    copy_json_value,
    require_clean_nonblank,
)
from cayu.core.billing import (
    UNRESOLVED_BILLING_IDENTITY,
    BillingIdentity,
    BillingIdentityState,
    ResolvedBillingIdentity,
    copy_billing_identity,
    resolved_billing_identity,
)
from cayu.core.events import (
    Event,
    EventType,
    event_with_runtime_envelope_authority,
    event_with_runtime_generated_id,
    event_with_runtime_nested_payload_authority,
    event_with_runtime_payload_authority,
)
from cayu.providers import ModelProviderError
from cayu.providers._credential_boundary import provider_cancellation_failures
from cayu.runtime._event_writer import RuntimeEventWriter
from cayu.runtime._run_limit_accounting import RunBudgetAccountingAuthority
from cayu.runtime._session_queries import query_all_event_records
from cayu.runtime.budgets import (
    MODEL_COMPLETION_BUDGET_SETTLEMENTS_KEY,
    BudgetCheck,
    BudgetLedger,
    BudgetLimit,
    BudgetPolicy,
    BudgetReconciliation,
    BudgetReconciliationPricing,
    BudgetReservationIdentityConflict,
    BudgetReservationRecord,
    BudgetReservationRecoveryContext,
    BudgetReservationResult,
    BudgetSettlementCursor,
    BudgetSettlementFallback,
    BudgetSettlementRecord,
    BudgetStore,
    _budget_reservation_amount,
    _budget_settlement_record,
    _copy_effective_budget_limit,
    _effective_budget_limit_id,
    _effective_budget_limits,
    _EffectiveBudgetLimit,
    _expired_reservation_reason,
    _is_expired_reservation_reason,
    _operation_budget_limits_for_session,
    budget_actual_cost_for_event,
    budget_check_from_events,
    budget_check_payload,
    budget_limits_for_session,
    budget_price,
    budget_reconciliation_from_payload,
    budget_reconciliation_payload,
    budget_reconciliation_preview,
    budget_reconciliation_pricing,
    budget_release_preview,
    budget_reservation_authority_sha256,
    budget_reservation_payload,
    budget_settlement_id,
    events_for_budget_window,
    has_deferred_contextual_price,
    model_completion_budget_settlements,
    new_budget_reservation_id,
    request_budget_limits_for_session,
)
from cayu.runtime.costs import SessionCostSummary, estimate_session_cost
from cayu.runtime.execution_profiles import (
    event_with_execution_profile_fingerprint_authority,
)
from cayu.runtime.execution_units import (
    ModelAttemptIdentity,
    ModelStepIdentity,
    copy_model_attempt_identity,
    copy_model_step_identity,
)
from cayu.runtime.sessions import (
    EventQuery,
    EventRecord,
    Session,
    SessionRunFenced,
    SessionStore,
)
from cayu.runtime.stop_policy import (
    RunLimits,
    StopDecision,
    StopLimit,
    first_reached_limit,
    has_run_limits,
)
from cayu.runtime.usage import (
    USAGE_BEARING_EVENT_TYPES,
    SessionUsageSummary,
    build_aggregate_usage_metrics,
    session_usage_summary,
)


def _event_with_budget_authority(
    event: Event,
    *,
    execution_identity: ModelStepIdentity | ModelAttemptIdentity | None = None,
    execution_profile_fingerprint: str | None = None,
    additional_fields: Collection[str] = (),
) -> Event:
    """Attest runtime-owned accounting linkage at the budget control boundary."""

    fields = [field_name for field_name in additional_fields if field_name in event.payload]
    if type(execution_identity) is ModelAttemptIdentity:
        identity_payload = copy_model_attempt_identity(execution_identity).payload()
    elif type(execution_identity) is ModelStepIdentity:
        identity_payload = copy_model_step_identity(execution_identity).payload()
    elif execution_identity is None:
        identity_payload = {}
    else:
        raise TypeError("Budget execution identity has an unsupported type.")
    fields.extend(
        field_name
        for field_name, value in identity_payload.items()
        if event.payload.get(field_name) == value
    )
    attributed = (
        event_with_runtime_payload_authority(event, *dict.fromkeys(fields)) if fields else event
    )
    return event_with_execution_profile_fingerprint_authority(
        attributed,
        execution_profile_fingerprint,
    )


UNKNOWN_POST_DISPATCH_BUDGET_REASON = (
    "provider usage unknown after dispatch; charged reserved amount"
)
PRE_PROVIDER_DISPATCH_BUDGET_RELEASE_REASON = "model completion abandoned before provider dispatch"

_OperationResultT = TypeVar("_OperationResultT")
_StreamResultT = TypeVar("_StreamResultT")


def _merge_events_by_id(*groups: list[Event]) -> list[Event]:
    """Merge durable and in-flight views without double-counting one event."""

    merged: list[Event] = []
    seen: set[str] = set()
    for group in groups:
        for event in group:
            if event.id in seen:
                continue
            seen.add(event.id)
            merged.append(event)
    return merged


def _execution_identity_payload(
    identity: ModelStepIdentity | ModelAttemptIdentity | None,
) -> dict[str, str]:
    if identity is None:
        return {}
    if type(identity) is ModelAttemptIdentity:
        return copy_model_attempt_identity(identity).payload()
    if type(identity) is ModelStepIdentity:
        return copy_model_step_identity(identity).payload()
    raise TypeError("Execution identity must be a ModelStepIdentity or ModelAttemptIdentity.")


@dataclass(frozen=True)
class _PublishableBudgetReservationAuthority:
    reservation_id: str
    settlement_fallback: BudgetSettlementFallback
    billing_identity: BillingIdentity | None


def _new_publishable_budget_reservation_authority(
    event_writer: RuntimeEventWriter,
    *,
    limit: _EffectiveBudgetLimit,
    model_attempt_identity: ModelAttemptIdentity,
    session_id: str,
    agent_name: str,
    environment_name: str | None,
    provider_name: str,
    model: str,
    settlement_event_payload: dict[str, object],
    billing_identity: BillingIdentity | None,
    reserved_amount: Decimal,
    fallback_settled_at: datetime,
    reservation_ttl_seconds: int | None,
) -> _PublishableBudgetReservationAuthority:
    """Allocate authority only after its terminal fallback is publishable."""

    reservation_id = new_budget_reservation_id()
    raw_expiration_reason = (
        None
        if reservation_ttl_seconds is None
        else _expired_reservation_reason(reservation_ttl_seconds)
    )
    raw_fallback = BudgetSettlementFallback(
        settled_at=fallback_settled_at,
        expiration_reason=raw_expiration_reason,
    )
    record = BudgetReservationRecord(
        reservation_id=reservation_id,
        budget_limit_id=limit.budget_limit_id,
        model_step_id=model_attempt_identity.model_step_id,
        model_attempt_id=model_attempt_identity.model_attempt_id,
        scope=limit.scope,
        key=limit.key,
        window=limit.window,
        currency=limit.currency,
        session_id=session_id,
        agent_name=agent_name,
        environment_name=environment_name,
        provider_name=provider_name,
        model=model,
        billing_identity=copy_billing_identity(billing_identity),
        settlement_event_payload=copy_durable_json_object(
            settlement_event_payload,
            "settlement_event_payload",
        ),
        settlement_fallback=raw_fallback,
        reserved_amount=reserved_amount,
        created_at=fallback_settled_at,
        updated_at=fallback_settled_at,
    )
    reservation = BudgetStepReservation(
        limit=limit,
        record=record,
        request_billing_identity=copy_billing_identity(billing_identity),
    )
    conservative = _publication_safe_reconciliation(
        event_writer.prepare,
        reservation=reservation,
        reconciliation=budget_reconciliation_preview(
            record,
            actual_amount=reserved_amount,
            settlement_kind="conservative",
            reason=raw_fallback.reconciliation_reason,
            occurred_at=raw_fallback.settled_at,
        ),
    )
    release = _publication_safe_reconciliation(
        event_writer.prepare,
        reservation=reservation,
        reconciliation=budget_release_preview(
            record,
            reason=raw_fallback.release_reason,
            occurred_at=raw_fallback.settled_at,
        ),
    )
    expiration = (
        None
        if raw_expiration_reason is None
        else _publication_safe_reconciliation(
            event_writer.prepare,
            reservation=reservation,
            reconciliation=budget_release_preview(
                record,
                reason=raw_expiration_reason,
                occurred_at=raw_fallback.settled_at,
            ),
        )
    )
    if conservative.actual_amount != reserved_amount:
        raise RuntimeError("Budget publication changed conservative fallback accounting.")
    if release.billing_identity != conservative.billing_identity or (
        expiration is not None and expiration.billing_identity != conservative.billing_identity
    ):
        raise RuntimeError("Budget publication changed fallback billing authority.")
    return _PublishableBudgetReservationAuthority(
        reservation_id=reservation_id,
        settlement_fallback=BudgetSettlementFallback(
            settled_at=conservative.settled_at,
            reconciliation_reason=conservative.reason or raw_fallback.reconciliation_reason,
            release_reason=release.reason or raw_fallback.release_reason,
            expiration_reason=(None if expiration is None else expiration.reason),
        ),
        billing_identity=copy_billing_identity(conservative.billing_identity),
    )


def _interaction_bound_settlement_event_payload(
    event_writer: RuntimeEventWriter,
    *,
    session_id: str,
    agent_name: str,
    environment_name: str | None,
    payload: dict[str, object] | None = None,
) -> dict[str, object]:
    """Bind future ledger-owned settlement evidence to its current interaction."""

    copied = copy_durable_json_object(payload or {}, "settlement_event_payload")
    attribution_probe = event_writer.prepare_budget_settlement_template(
        Event(
            type=EventType.BUDGET_RECONCILED,
            session_id=session_id,
            agent_name=agent_name,
            environment_name=environment_name,
            payload={},
        )
    )
    copied["interaction_id"] = attribution_probe.interaction_id
    return copied


def _validate_ledger_reservation_result(
    result: BudgetReservationResult,
    *,
    expected_reservation_id: str,
    limit: _EffectiveBudgetLimit,
    model_attempt_identity: ModelAttemptIdentity,
    session_id: str,
    agent_name: str,
    provider_name: str,
    model: str,
    environment_name: str | None,
    settlement_event_payload: dict[str, object],
    settlement_fallback: BudgetSettlementFallback,
    billing_identity: BillingIdentity | None,
    expected_requested_amount: Decimal,
) -> BudgetReservationResult:
    """Detach and validate one untrusted custom-ledger reservation response."""

    if type(result) is not BudgetReservationResult:
        raise TypeError("Budget ledger reserve() must return a BudgetReservationResult.")
    if result.record is not None and type(result.record) is not BudgetReservationRecord:
        raise TypeError(
            "Budget ledger reservation records must be BudgetReservationRecord instances."
        )
    result = BudgetReservationResult.model_validate(result.model_dump(mode="python"))
    expected_identity = copy_model_attempt_identity(model_attempt_identity)
    if (
        result.budget_limit_id != limit.budget_limit_id
        or result.model_step_id != expected_identity.model_step_id
        or result.model_attempt_id != expected_identity.model_attempt_id
        or result.scope != limit.scope
        or result.key != limit.key
        or result.window != limit.window
        or result.currency != limit.currency
        or result.maximum != limit.max_estimated_cost
        or result.action != limit.action
    ):
        raise RuntimeError("Budget ledger reservation result changed its requested identity.")
    if result.requested != expected_requested_amount:
        raise RuntimeError("Budget ledger reservation result changed its requested amount.")
    if not result.accepted:
        if result.record is not None:
            raise RuntimeError("Rejected budget reservation unexpectedly returned a record.")
        if result.actual <= result.maximum:
            raise RuntimeError("Rejected budget reservation did not exceed its configured maximum.")
        return result
    if result.actual < result.requested or result.actual > result.maximum:
        raise RuntimeError("Accepted budget reservation violated its configured maximum.")
    record = result.record
    if record is None:
        raise RuntimeError("Accepted budget reservation did not return a record.")
    if record.reservation_id != expected_reservation_id:
        raise RuntimeError("Budget ledger reused a reservation identity.")
    if (
        record.budget_limit_id != limit.budget_limit_id
        or record.model_step_id != expected_identity.model_step_id
        or record.model_attempt_id != expected_identity.model_attempt_id
        or record.scope != limit.scope
        or record.key != limit.key
        or record.window != limit.window
        or record.currency != limit.currency
        or record.session_id != session_id
        or record.agent_name != agent_name
        or record.environment_name != environment_name
        or record.provider_name != provider_name
        or record.model != model
        or record.billing_identity != billing_identity
        or record.settlement_event_payload != settlement_event_payload
        or record.settlement_fallback != settlement_fallback
        or record.dispatch_id is not None
        or record.dispatched_at is not None
        or record.reserved_amount != result.requested
        or record.status != "active"
        or record.actual_amount is not None
        or record.reason is not None
    ):
        raise RuntimeError("Budget ledger reservation record changed its requested identity.")
    return result


def _validate_ledger_settlement_record(
    settlement: BudgetSettlementRecord,
) -> BudgetSettlementRecord:
    """Detach and revalidate one untrusted custom-ledger outbox value."""

    if type(settlement) is not BudgetSettlementRecord:
        raise TypeError("Budget ledger settlements must be BudgetSettlementRecord instances.")
    if type(settlement.reconciliation) is not BudgetReconciliation:
        raise TypeError(
            "Budget ledger settlement reconciliations must be BudgetReconciliation instances."
        )
    if type(settlement.event) is not Event:
        raise TypeError("Budget ledger settlement events must be Event instances.")
    validated = BudgetSettlementRecord.model_validate(settlement.model_dump(mode="python"))
    # Deserialization deliberately strips in-process provenance. The record's
    # model validator has now positively bound the deterministic event and its
    # accounting identities to the settlement record. Re-attest only that
    # validated copy before handing it to the event writer.
    event = event_with_runtime_generated_id(validated.event)
    envelope_fields = ["session_id"]
    payload_fields = [
        "reservation_id",
        "settlement_id",
        "budget_limit_id",
        "model_step_id",
        "model_attempt_id",
    ]
    if "execution_profile_fingerprint" in event.payload:
        payload_fields.append("execution_profile_fingerprint")
    if event.interaction_id is not None:
        if event.payload.get("interaction_id") != event.interaction_id:
            raise RuntimeError("Budget ledger settlement event changed its interaction identity.")
        envelope_fields.append("interaction_id")
        payload_fields.append("interaction_id")
    event = event_with_runtime_payload_authority(
        event_with_runtime_envelope_authority(event, *envelope_fields),
        *payload_fields,
    )
    return validated.model_copy(
        update={"event": event},
        deep=True,
    )


def _validate_ledger_settlement_page(
    page: list[BudgetSettlementRecord],
    *,
    session_id: str | None,
    after: BudgetSettlementCursor | None,
    limit: int,
) -> list[BudgetSettlementRecord]:
    """Validate bounded ordering and ownership before traversing an outbox page."""

    if type(page) is not list:
        raise TypeError("Budget ledger pending settlements must be returned as a list.")
    if len(page) > limit:
        raise RuntimeError("Budget ledger returned an oversized settlement page.")
    validated: list[BudgetSettlementRecord] = []
    previous_key = None if after is None else (after.settled_at, after.settlement_id)
    for raw_settlement in page:
        settlement = _validate_ledger_settlement_record(raw_settlement)
        key = (settlement.reconciliation.settled_at, settlement.settlement_id)
        if settlement.event_published:
            raise RuntimeError("Budget ledger returned a published settlement as pending.")
        if session_id is not None and settlement.session_id != session_id:
            raise RuntimeError("Budget ledger returned a settlement for another session.")
        if previous_key is not None and key <= previous_key:
            raise RuntimeError("Budget ledger returned a non-monotonic settlement page.")
        validated.append(settlement)
        previous_key = key
    return validated


def _validate_ledger_limit_unchanged(
    limit: _EffectiveBudgetLimit,
    *,
    expected: _EffectiveBudgetLimit,
) -> None:
    """Fail closed if a custom ledger mutates its detached limit argument."""

    try:
        detached = _copy_effective_budget_limit(limit)
    except Exception:
        raise RuntimeError(
            "Budget ledger reservation result changed its requested identity."
        ) from None
    if detached != expected:
        raise RuntimeError("Budget ledger reservation result changed its requested identity.")


def _validate_ledger_reconciliation(
    reconciliation: BudgetReconciliation,
    *,
    reservation: BudgetStepReservation,
    expected_status: Literal["reconciled", "released"],
    expected_settlement_kind: Literal["completed", "conservative", "released"],
    expected_actual_amount: Decimal | None,
    expected_reason: str | None,
    expected_billing_identity: BillingIdentity | None,
    expected_pricing: BudgetReconciliationPricing | None = None,
) -> BudgetReconciliation:
    """Detach and validate one untrusted custom-ledger settlement response."""

    return _validate_ledger_reconciliation_against_reservation_record(
        reconciliation,
        record=reservation.record,
        expected_status=expected_status,
        expected_settlement_kind=expected_settlement_kind,
        expected_actual_amount=expected_actual_amount,
        expected_reason=expected_reason,
        expected_billing_identity=expected_billing_identity,
        expected_pricing=expected_pricing,
    )


def _validate_ledger_reconciliation_against_reservation_record(
    reconciliation: BudgetReconciliation,
    *,
    record: BudgetReservationRecord,
    expected_status: Literal["reconciled", "released"],
    expected_settlement_kind: Literal["completed", "conservative", "released"],
    expected_actual_amount: Decimal | None,
    expected_reason: str | None,
    expected_billing_identity: BillingIdentity | None,
    expected_pricing: BudgetReconciliationPricing | None = None,
) -> BudgetReconciliation:
    """Validate an untrusted settlement against its original durable reservation."""

    if type(reconciliation) is not BudgetReconciliation:
        raise TypeError("Budget ledger settlement must return a BudgetReconciliation.")
    try:
        reconciliation = BudgetReconciliation.model_validate(
            reconciliation.model_dump(mode="python")
        )
    except Exception:
        raise RuntimeError("Budget ledger settlement changed its requested outcome.") from None
    actual_pricing = _reconciliation_pricing_evidence(reconciliation)
    if actual_pricing != expected_pricing:
        raise RuntimeError("Budget ledger settlement changed its requested pricing evidence.")
    expected_released_amount = (
        record.reserved_amount
        if expected_status == "released"
        else max(record.reserved_amount - (expected_actual_amount or Decimal("0")), Decimal("0"))
    )
    reason_matches = reconciliation.reason == expected_reason or (
        expected_status == "released" and _is_expired_reservation_reason(reconciliation.reason)
    )
    if (
        reconciliation.reservation_id != record.reservation_id
        or reconciliation.settlement_id != budget_settlement_id(record.reservation_id)
        or reconciliation.settlement_kind != expected_settlement_kind
        or reconciliation.budget_limit_id != record.budget_limit_id
        or reconciliation.model_step_id != record.model_step_id
        or reconciliation.model_attempt_id != record.model_attempt_id
        or reconciliation.execution_profile_fingerprint
        != record.settlement_event_payload.get("execution_profile_fingerprint")
        or reconciliation.reserved_amount != record.reserved_amount
        or reconciliation.status != expected_status
        or reconciliation.actual_amount != expected_actual_amount
        or reconciliation.released_amount != expected_released_amount
        or not reason_matches
        or reconciliation.billing_identity != expected_billing_identity
    ):
        raise RuntimeError("Budget ledger settlement changed its requested outcome.")
    return reconciliation


def _validate_ledger_reservation_against_recovery_context(
    record: BudgetReservationRecord,
    *,
    context: BudgetReservationRecoveryContext,
    dispatch_id: str,
) -> BudgetReservationRecord:
    """Validate mutable ledger state against its frozen pre-dispatch authority."""

    if type(record) is not BudgetReservationRecord:
        raise TypeError("Budget ledger reservations must be BudgetReservationRecord instances.")
    try:
        record = BudgetReservationRecord.model_validate(record.model_dump(mode="python"))
    except Exception:
        raise RuntimeError("Budget ledger changed its reservation authority.") from None
    if record.dispatch_id not in {None, dispatch_id}:
        raise RuntimeError("Budget ledger changed its reservation authority.")
    if budget_reservation_authority_sha256(record) != context.reservation_authority_sha256:
        raise RuntimeError("Budget ledger changed its reservation authority.")
    return record


def _publication_safe_reconciliation(
    prepare_event: Callable[[Event], Event],
    *,
    reservation: BudgetStepReservation,
    reconciliation: BudgetReconciliation,
) -> BudgetReconciliation:
    """Normalize dynamic settlement evidence before it becomes ledger authority."""

    return _publication_safe_reconciliation_for_record(
        prepare_event,
        record=reservation.record,
        reconciliation=reconciliation,
    )


def _publication_safe_reconciliation_for_record(
    prepare_event: Callable[[Event], Event],
    *,
    record: BudgetReservationRecord,
    reconciliation: BudgetReconciliation,
) -> BudgetReconciliation:
    """Normalize settlement evidence using only durable reservation authority."""

    if type(reconciliation) is not BudgetReconciliation:
        raise TypeError("reconciliation must be a BudgetReconciliation.")
    raw_settlement = _budget_settlement_record(
        record,
        reconciliation,
    )
    prepared_event = prepare_event(raw_settlement.event)
    reconciliation_payload = budget_reconciliation_payload(reconciliation)
    prepared_payload = {key: prepared_event.payload[key] for key in reconciliation_payload}
    prepared_profile_fingerprint = prepared_event.payload.get("execution_profile_fingerprint")
    if prepared_profile_fingerprint != reconciliation.execution_profile_fingerprint:
        raise RuntimeError("Budget settlement publication changed its execution profile.")
    prepared_reconciliation = budget_reconciliation_from_payload(prepared_payload).model_copy(
        update={"execution_profile_fingerprint": prepared_profile_fingerprint},
        deep=True,
    )
    prepared_settlement = _budget_settlement_record(
        record,
        prepared_reconciliation,
    )
    if prepared_settlement.event != prepared_event:
        raise RuntimeError("Budget settlement publication changed immutable reservation authority.")
    return prepared_reconciliation


class BudgetReservationLeaseLost(RuntimeError):
    """Raised when a live model step can no longer prove its budget reservation."""


class BudgetReservationLeaseLostBeforeModelDispatch(BudgetReservationLeaseLost):
    """Raised when lease loss is detected before any provider attempt starts."""


class ModelCompletionBudgetSettlementPending(RuntimeError):
    """Raised while a model stage still owns reservations without durable settlement."""


class ProviderIteratorCleanupError(RuntimeError):
    """Credential-free evidence that a cancelled provider iterator did not stop cleanly."""


_PROVIDER_CLEANUP_FAILURE_ATTRIBUTE = "_cayu_budget_provider_cleanup_failure"
_PROVIDER_CLEANUP_FAILURE_TOKEN = object()


@dataclass(frozen=True, slots=True)
class _ProviderCleanupFailureHandoff:
    failure: BaseException
    token: object


def _record_provider_cleanup_failure(
    authoritative_failure: BaseException,
    provider_failure: BaseException,
) -> None:
    """Carry a secondary provider failure to the redacting terminal boundary."""

    handoff = _ProviderCleanupFailureHandoff(
        failure=provider_failure,
        token=_PROVIDER_CLEANUP_FAILURE_TOKEN,
    )
    set_exception_state(
        authoritative_failure,
        _PROVIDER_CLEANUP_FAILURE_ATTRIBUTE,
        handoff,
    )
    add_exception_note_safely(
        authoritative_failure,
        "Provider iterator cleanup also failed while budget lease loss was handled.",
    )


def budget_provider_cleanup_failure(
    authoritative_failure: BaseException,
) -> BaseException | None:
    """Return only provider cleanup evidence attached by this runtime boundary."""

    handoff = exception_state(
        authoritative_failure,
        _PROVIDER_CLEANUP_FAILURE_ATTRIBUTE,
    )
    if (
        type(handoff) is not _ProviderCleanupFailureHandoff
        or handoff.token is not _PROVIDER_CLEANUP_FAILURE_TOKEN
        or not isinstance(handoff.failure, BaseException)
    ):
        return None
    return handoff.failure


def _safe_provider_stream_cleanup_failure(
    cancellation: asyncio.CancelledError,
) -> ProviderIteratorCleanupError | None:
    """Reconstruct only authenticated, credential-free cleanup evidence."""

    failures = provider_cancellation_failures(cancellation)
    if not failures:
        return None
    return ProviderIteratorCleanupError(failures[0]["error"])


class BudgetDispatchReservationFailed(RuntimeError):
    def __init__(self, result: BudgetReservationResult) -> None:
        self.result = result
        super().__init__(result.message)


def add_budget_failure_note(
    authoritative_failure: BaseException,
    *,
    operation: str,
    accounting_failure: Exception,
) -> None:
    note = (
        f"Budget {operation} also failed: {type(accounting_failure).__name__}: {accounting_failure}"
    )
    if note not in getattr(authoritative_failure, "__notes__", ()):
        authoritative_failure.add_note(note)


def budget_heartbeat_task_failure(task: asyncio.Task[None]) -> BaseException:
    if task.cancelled():
        return BudgetReservationLeaseLost(
            "Budget reservation heartbeat was cancelled unexpectedly."
        )
    failure = task.exception()
    if failure is None:
        return BudgetReservationLeaseLost("Budget reservation heartbeat stopped unexpectedly.")
    return failure


def _preserve_completed_metadata(
    source: BaseException,
    target: BaseException,
) -> None:
    """Copy trustworthy completion evidence without replacing the primary signal."""

    completed_metadata = getattr(source, "completed_metadata", None)
    if type(completed_metadata) is not dict:
        return
    try:
        copied_metadata = copy_json_value(completed_metadata, "completed_metadata")
    except Exception as exc:
        target.add_note(
            "Budgeted operation completion evidence could not be preserved: "
            f"{type(exc).__name__}: {exc}"
        )
        return
    target.__dict__["completed_metadata"] = copied_metadata


async def _next_model_step_item(
    iterator: AsyncIterator[tuple[Event | None, _StreamResultT | None]],
) -> tuple[Event | None, _StreamResultT | None]:
    return await anext(iterator)


@dataclass(frozen=True)
class BudgetStepReservation:
    limit: _EffectiveBudgetLimit
    record: BudgetReservationRecord
    request_billing_identity: BillingIdentity | None = field(default=None, repr=False)


def _accounting_reservation_record(
    reservation: BudgetStepReservation,
) -> BudgetReservationRecord:
    """Restore transient request evidence while calculating an exact settlement."""

    return reservation.record.model_copy(
        update={"billing_identity": copy_billing_identity(reservation.request_billing_identity)},
        deep=True,
    )


@dataclass
class BudgetProviderDispatch:
    model_attempt_identity: ModelAttemptIdentity
    reservations: tuple[BudgetStepReservation, ...]
    completion: Event | None = None
    settled_reservation_ids: set[str] = field(default_factory=set)

    @property
    def settled(self) -> bool:
        return self.settled_reservation_ids == {
            reservation.record.reservation_id for reservation in self.reservations
        }


@dataclass
class BudgetModelStepLifecycle:
    dispatches: list[BudgetProviderDispatch] = field(default_factory=list)
    pending_reservations: tuple[BudgetStepReservation, ...] | None = None
    pending_model_attempt_identity: ModelAttemptIdentity | None = None
    observed_reservation_ids: set[str] = field(default_factory=set)
    reservation_transition_lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    @property
    def provider_dispatch_may_have_occurred(self) -> bool:
        return bool(self.dispatches)

    def prepare_provider_dispatch(
        self,
        model_attempt_identity: ModelAttemptIdentity,
        reservations: list[BudgetStepReservation],
    ) -> None:
        model_attempt_identity = copy_model_attempt_identity(model_attempt_identity)
        if self.pending_reservations is not None or self.pending_model_attempt_identity is not None:
            raise RuntimeError("Provider dispatch already has prepared budget reservations.")
        reservation_ids = {reservation.record.reservation_id for reservation in reservations}
        if len(reservation_ids) != len(reservations) or (
            reservation_ids & self.observed_reservation_ids
        ):
            raise RuntimeError("Budget ledger reused a reservation identity.")
        for reservation in reservations:
            if (
                reservation.record.model_step_id != model_attempt_identity.model_step_id
                or reservation.record.model_attempt_id != model_attempt_identity.model_attempt_id
            ):
                raise ValueError(
                    "Prepared budget reservation belongs to a different model attempt."
                )
        self.observed_reservation_ids.update(reservation_ids)
        self.pending_reservations = tuple(reservations)
        self.pending_model_attempt_identity = model_attempt_identity

    def mark_provider_dispatch(
        self,
        model_attempt_identity: ModelAttemptIdentity,
    ) -> None:
        model_attempt_identity = copy_model_attempt_identity(model_attempt_identity)
        if self.pending_reservations is None or self.pending_model_attempt_identity is None:
            raise RuntimeError("Provider dispatch has no prepared budget reservations.")
        if self.pending_model_attempt_identity != model_attempt_identity:
            raise ValueError("Provider dispatch identity differs from its budget reservations.")
        self.dispatches.append(
            BudgetProviderDispatch(
                model_attempt_identity=model_attempt_identity,
                reservations=self.pending_reservations,
            )
        )
        self.pending_reservations = None
        self.pending_model_attempt_identity = None

    def record_model_completion(
        self,
        event: Event,
        *,
        prepare_event: Callable[[Event], Event],
        settled_at: datetime,
    ) -> Event:
        if not self.dispatches:
            raise RuntimeError("Model completed before provider dispatch was recorded.")
        if self.dispatches[-1].completion is not None:
            raise RuntimeError("Provider dispatch produced more than one model completion.")
        identity = self.dispatches[-1].model_attempt_identity
        if (
            event.payload.get("model_step_id") != identity.model_step_id
            or event.payload.get("model_attempt_id") != identity.model_attempt_id
        ):
            raise ValueError("Model completion identity differs from its provider dispatch.")
        completion = _model_completion_with_budget_settlement_evidence(
            event,
            self.dispatches[-1].reservations,
            prepare_event=prepare_event,
            settled_at=settled_at,
        )
        self.dispatches[-1].completion = completion
        return completion.model_copy(deep=True)


def _model_completion_reconciliation(
    event: Event,
    reservation: BudgetStepReservation,
    *,
    settled_at: datetime,
) -> BudgetReconciliation:
    raw_identity = event.payload.get("billing_identity")
    completed_billing_identity = (
        BillingIdentity.model_validate(raw_identity) if type(raw_identity) is dict else None
    )
    pricing: BudgetReconciliationPricing | None = None
    try:
        priced_actual = budget_actual_cost_for_event(
            limit=reservation.limit,
            event=event,
        )
    except ValueError:
        actual_amount = reservation.record.reserved_amount
        settlement_kind: Literal["completed", "conservative"] = "conservative"
        reason = "model completed without priced usage; charged reserved amount"
    else:
        actual_amount = priced_actual.amount
        settlement_kind = "completed"
        reason = "model completed"
        completed_billing_identity = (
            priced_actual.line_item.billing_identity or completed_billing_identity
        )
        pricing = budget_reconciliation_pricing(priced_actual.line_item)
    return budget_reconciliation_preview(
        _accounting_reservation_record(reservation),
        actual_amount=actual_amount,
        settlement_kind=settlement_kind,
        reason=reason,
        occurred_at=settled_at,
        billing_identity=completed_billing_identity,
        pricing=pricing,
    )


def _reconciliation_pricing_evidence(
    reconciliation: BudgetReconciliation,
) -> BudgetReconciliationPricing | None:
    if reconciliation.pricing_provider_name is None:
        return None
    if (
        reconciliation.pricing_model is None
        or reconciliation.pricing_match is None
        or reconciliation.pricing_provenance is None
    ):
        raise ValueError("Budget reconciliation has incomplete pricing evidence.")
    return BudgetReconciliationPricing(
        provider_name=reconciliation.pricing_provider_name,
        model=reconciliation.pricing_model,
        match=reconciliation.pricing_match,
        provenance=reconciliation.pricing_provenance,
        effective_from=reconciliation.pricing_effective_from,
        effective_through=reconciliation.pricing_effective_through,
        tier_max_input_tokens=reconciliation.pricing_tier_max_input_tokens,
        billing_identity=reconciliation.billing_identity,
    )


def _model_completion_with_budget_settlement_evidence(
    event: Event,
    reservations: tuple[BudgetStepReservation, ...],
    *,
    prepare_event: Callable[[Event], Event],
    settled_at: datetime,
) -> Event:
    if type(event) is not Event or event.type != EventType.MODEL_COMPLETED:
        raise ValueError("Budget settlement evidence requires one model.completed event.")
    if MODEL_COMPLETION_BUDGET_SETTLEMENTS_KEY in event.payload:
        raise ValueError("Model completion already contains budget settlement evidence.")
    prepared_event = prepare_event(event)
    if type(prepared_event) is not Event:
        raise TypeError("Model completion preparation must return an Event.")
    if not reservations:
        return prepared_event

    reconciliations: list[BudgetReconciliation] = []
    for reservation in reservations:
        exact = _model_completion_reconciliation(
            event,
            reservation,
            settled_at=settled_at,
        )
        try:
            reconciliation = _publication_safe_reconciliation(
                prepare_event,
                reservation=reservation,
                reconciliation=exact,
            )
        except ValueError as exact_error:
            fallback = budget_reconciliation_preview(
                _accounting_reservation_record(reservation),
                actual_amount=reservation.record.reserved_amount,
                settlement_kind="conservative",
                reason=reservation.record.settlement_fallback.reconciliation_reason,
                occurred_at=settled_at,
                billing_identity=exact.billing_identity,
            )
            try:
                reconciliation = _publication_safe_reconciliation(
                    prepare_event,
                    reservation=reservation,
                    reconciliation=fallback,
                )
            except Exception as fallback_error:
                add_exception_note_safely(
                    exact_error,
                    "The prevalidated conservative model-budget publication fallback "
                    f"also failed: {type(fallback_error).__name__}.",
                )
                raise exact_error from fallback_error
        reconciliations.append(reconciliation)

    return _model_completion_with_reconciliation_evidence(
        prepared_event,
        reconciliations=tuple(reconciliations),
        reservation_ids=tuple(reservation.record.reservation_id for reservation in reservations),
        prepare_event=prepare_event,
    )


def _model_completion_with_reconciliation_evidence(
    prepared_event: Event,
    *,
    reconciliations: tuple[BudgetReconciliation, ...],
    reservation_ids: tuple[str, ...],
    prepare_event: Callable[[Event], Event],
) -> Event:
    payload = copy_json_value(prepared_event.payload, "model_completion_payload")
    payload[MODEL_COMPLETION_BUDGET_SETTLEMENTS_KEY] = [
        budget_reconciliation_payload(reconciliation) for reconciliation in reconciliations
    ]
    completion = event_with_runtime_nested_payload_authority(
        prepared_event.model_copy(update={"payload": payload}, deep=True),
        ("budget_settlements", "*", "reservation_id"),
        ("budget_settlements", "*", "settlement_id"),
        ("budget_settlements", "*", "budget_limit_id"),
        ("budget_settlements", "*", "model_step_id"),
        ("budget_settlements", "*", "model_attempt_id"),
    )
    completion = prepare_event(completion)
    if type(completion) is not Event:
        raise TypeError("Model completion preparation must return an Event.")
    parsed = model_completion_budget_settlements(
        completion,
        reservation_ids=reservation_ids,
    )
    if parsed != tuple(reconciliations):
        raise RuntimeError("Model completion publication changed budget settlement evidence.")
    return completion


@dataclass(frozen=True)
class LimitEvaluation:
    decision: StopDecision | None
    usage_summary: SessionUsageSummary
    cost_summary: SessionCostSummary | None
    events: tuple[Event, ...] = ()


@dataclass(frozen=True)
class BudgetEvaluation:
    check: BudgetCheck | None
    events: tuple[Event, ...] = ()


@dataclass(frozen=True)
class BudgetReservationSetup:
    reservations: tuple[BudgetStepReservation, ...]
    failure: BudgetReservationResult | None
    events: tuple[Event, ...]
    error: Exception | None


@dataclass(frozen=True)
class OperationReservationSetup:
    reservations: tuple[BudgetStepReservation, ...]
    results: tuple[BudgetReservationResult, ...]
    events: tuple[Event, ...]
    releases: tuple[BudgetReconciliation, ...]
    failure: BudgetReservationResult | None
    error: BaseException | None


@dataclass(frozen=True)
class OperationBudgetCheck:
    limit: BudgetLimit
    check: BudgetCheck


@dataclass(frozen=True)
class BudgetedOperationSucceeded(Generic[_OperationResultT]):
    result: _OperationResultT
    events: tuple[Event, ...]


@dataclass(frozen=True)
class BudgetedOperationRejected:
    failure: BudgetReservationResult
    events: tuple[Event, ...]


@dataclass(frozen=True)
class BudgetedOperationFailed:
    error: BaseException
    cause: BaseException | None
    events: tuple[Event, ...]


@dataclass
class _BudgetedOperationLifecycle:
    reservations: list[BudgetStepReservation] = field(default_factory=list)
    events: list[Event] = field(default_factory=list)
    provider_dispatch_started: bool = False
    settled: bool = False
    predispatch_release_reason: str = "context compaction reservation lost before provider dispatch"


@dataclass(frozen=True)
class _BudgetLimitOutcome:
    decision: StopDecision
    check: BudgetCheck


class SessionUsageTracker:
    """Incrementally accumulate usage with one ordered tail query per refresh."""

    def __init__(self, session_store: SessionStore, *, session_id: str) -> None:
        self._session_store = session_store
        self._session_id = require_clean_nonblank(session_id, "session_id")
        self._after_sequence: int | None = None
        self._events: list[Event] = []

    async def _new_usage_records(self) -> list[EventRecord]:
        # One multi-type query and one shared watermark are essential. Separate
        # per-type reads can skip spend appended between queries.
        return await query_all_event_records(
            self._session_store,
            EventQuery(
                session_id=self._session_id,
                event_types=USAGE_BEARING_EVENT_TYPES,
                after_sequence=self._after_sequence,
            ),
        )

    async def mark_current_position(self) -> None:
        new_records = await self._new_usage_records()
        if new_records:
            self._after_sequence = new_records[-1].sequence

    async def usage_events(self) -> list[Event]:
        new_records = await self._new_usage_records()
        if new_records:
            self._events.extend(record.event for record in new_records)
            self._after_sequence = new_records[-1].sequence
        return self._events


class BudgetReservationIdentityGuard:
    """Atomically bind reservation identities in both shared durability domains."""

    def __init__(self, session_store: SessionStore, budget_ledger: BudgetLedger) -> None:
        self._session_store = session_store
        self._budget_ledger = budget_ledger

    async def claim(
        self,
        reservation_id: str,
        *,
        publication_session_id: str,
        publication_id: str,
    ) -> None:
        """Claim one id before adding it to cleanup or provider-dispatch state."""
        reservation_id = require_clean_nonblank(reservation_id, "reservation_id")
        publication_session_id = require_clean_nonblank(
            publication_session_id,
            "publication_session_id",
        )
        publication_id = require_clean_nonblank(publication_id, "publication_id")
        claim = {
            "reservation_id": reservation_id,
            "publication_session_id": publication_session_id,
            "publication_id": publication_id,
        }
        # Claim the publication store first so a stale session owner cannot
        # poison the shared ledger registry. The ledger-domain claim then
        # closes split-store topologies before provider-controlled work starts.
        await self._session_store.claim_budget_reservation_identity(**claim)
        await self._budget_ledger.claim_reservation_identity(**claim)


class RunLimitController:
    """Evaluate run limits and own durable budget-accounting lifecycle."""

    def __init__(
        self,
        *,
        session_store: SessionStore,
        budget_store: BudgetStore,
        budget_ledger: BudgetLedger,
        event_writer: RuntimeEventWriter,
        clock: Callable[[], datetime],
    ) -> None:
        self._session_store = session_store
        self._budget_store = budget_store
        self._budget_ledger = budget_ledger
        self._event_writer = event_writer
        self._clock = clock
        self._reservation_identity_guard = BudgetReservationIdentityGuard(
            session_store,
            budget_ledger,
        )
        self._global_settlement_recovery_lock = asyncio.Lock()
        self._global_settlement_recovery_after: BudgetSettlementCursor | None = None

    def usage_tracker(self, session_id: str) -> SessionUsageTracker:
        return SessionUsageTracker(self._session_store, session_id=session_id)

    def reservation_identity_guard(self) -> BudgetReservationIdentityGuard:
        """Return the controller's store-wide live identity guard."""

        return self._reservation_identity_guard

    @property
    def reservation_ttl_seconds(self) -> int | None:
        return self._budget_ledger.reservation_ttl_seconds

    def budget_settlement_time(self) -> datetime:
        """Return the accounting clock used by the configured budget ledger."""

        return self._clock()

    async def session_usage_events(self, session_id: str) -> list[Event]:
        records = await query_all_event_records(
            self._session_store,
            EventQuery(
                session_id=session_id,
                event_types=USAGE_BEARING_EVENT_TYPES,
            ),
        )
        return [record.event for record in records]

    async def evaluate_operation_run_limit(
        self,
        *,
        session: Session,
        limits: RunLimits,
        operation_events: list[Event],
        operation_started_at: float,
    ) -> StopDecision | None:
        """Evaluate limits for a bounded operation such as explicit compaction."""

        if not has_run_limits(limits):
            return None
        usage_events = _merge_events_by_id(operation_events)
        if limits.scope == "session":
            usage_events = _merge_events_by_id(
                await self.session_usage_events(session.id),
                usage_events,
            )
            created_at = session.created_at
            if created_at.tzinfo is None:
                created_at = created_at.replace(tzinfo=UTC)
            elapsed_seconds = max(
                0,
                int((self._clock() - created_at.astimezone(UTC)).total_seconds()),
            )
        else:
            elapsed_seconds = max(0, int(time.monotonic() - operation_started_at))
        return first_reached_limit(
            limits=limits,
            usage=session_usage_summary(session.id, usage_events),
            elapsed_seconds=elapsed_seconds,
        )

    async def evaluate_operation_budgets(
        self,
        *,
        session: Session,
        budget_limits: tuple[BudgetLimit, ...],
        operation_events: list[Event],
        provider_name: str | None,
        model: str | None,
        billing_identity_state: BillingIdentityState = UNRESOLVED_BILLING_IDENTITY,
    ) -> tuple[OperationBudgetCheck, ...]:
        """Evaluate scopes while including an operation's uncommitted events."""

        budget_limits = _operation_budget_limits_for_session(
            limits=budget_limits,
            agent_name=session.agent_name,
            causal_budget_id=session.causal_budget_id,
        )
        checks: list[OperationBudgetCheck] = []
        for limit in budget_limits:
            if limit.scope in {"app", "agent", "causal"}:
                existing_events = await self._budget_store.load_events_for_budget(
                    scope=limit.scope,
                    key=limit.key,
                    window=limit.window,
                )
            elif limit.scope == "session":
                existing_events = await self.session_usage_events(session.id)
            elif limit.scope == "run":
                existing_events = []
            else:
                raise ValueError(f"Unsupported request budget scope: {limit.scope}")
            events = events_for_budget_window(
                _merge_events_by_id(existing_events, operation_events),
                limit.window,
                now=self._clock(),
            )
            event_provider_name, event_model = _latest_model_event_identity(operation_events)
            effective_provider_name = event_provider_name or provider_name
            effective_model = event_model or model
            if effective_provider_name is None or effective_model is None:
                summary = estimate_session_cost(
                    session_id=session.id,
                    events=events,
                    pricing=limit.pricing,
                    currency=limit.currency,
                )
                if (
                    summary.unpriced_model_steps == 0
                    and summary.total_cost < limit.max_estimated_cost
                ):
                    continue
            check = budget_check_from_events(
                limit=limit,
                events=events,
                provider_name=effective_provider_name,
                model=effective_model,
                billing_identity_state=billing_identity_state,
                effective_at=self._clock(),
            )
            checks.append(OperationBudgetCheck(limit=limit, check=check))
        return tuple(checks)

    async def evaluate_request_limits(
        self,
        *,
        session: Session,
        agent_name: str,
        environment_name: str | None,
        limits: RunLimits,
        budget_limits: tuple[BudgetLimit, ...],
        run_started_at: float,
        run_baseline: SessionUsageSummary | None = None,
        budget_baseline_events: list[Event] | None = None,
        run_budget_authorities: Mapping[str, RunBudgetAccountingAuthority] | None = None,
        pending_tool_calls: int = 0,
        budget_notify_events: list[Event] | None = None,
        usage_tracker: SessionUsageTracker | None = None,
        billing_identity_state: BillingIdentityState = UNRESOLVED_BILLING_IDENTITY,
        pricing_provider_name: str | None = None,
        model: str | None = None,
        additional_usage_events: list[Event] | None = None,
        execution_identity: ModelStepIdentity | ModelAttemptIdentity | None = None,
        execution_profile_fingerprint: str | None = None,
    ) -> LimitEvaluation:
        budget_limits = request_budget_limits_for_session(
            limits=budget_limits,
            agent_name=agent_name,
            causal_budget_id=session.causal_budget_id,
        )
        if not has_run_limits(limits) and not budget_limits:
            return LimitEvaluation(
                decision=None,
                usage_summary=SessionUsageSummary(session_id=session.id),
                cost_summary=None,
            )
        events = (
            await usage_tracker.usage_events()
            if usage_tracker is not None
            else await self.session_usage_events(session.id)
        )
        additional_events = [
            event.model_copy(deep=True) for event in (additional_usage_events or [])
        ]
        events = _merge_events_by_id(events, additional_events)
        usage_summary = session_usage_summary(session.id, events)
        usage_for_limits = usage_summary
        if limits.scope == "run" and run_baseline is not None:
            current, baseline = usage_summary.usage, run_baseline.usage
            usage_for_limits = SessionUsageSummary(
                session_id=session.id,
                tool_calls=max(0, usage_summary.tool_calls - run_baseline.tool_calls),
                usage=build_aggregate_usage_metrics(
                    input_tokens=max(0, current.input_tokens - baseline.input_tokens),
                    output_tokens=max(0, current.output_tokens - baseline.output_tokens),
                    total_tokens=max(0, current.total_tokens - baseline.total_tokens),
                ),
            )
        if limits.scope == "session":
            created_at = session.created_at
            if created_at.tzinfo is None:
                created_at = created_at.replace(tzinfo=UTC)
            elapsed_seconds = max(
                0,
                int((self._clock() - created_at.astimezone(UTC)).total_seconds()),
            )
        else:
            elapsed_seconds = max(0, int(time.monotonic() - run_started_at))
        decision = first_reached_limit(
            limits=limits,
            usage=usage_for_limits,
            elapsed_seconds=elapsed_seconds,
            pending_tool_calls=pending_tool_calls,
        )
        if decision is not None:
            return LimitEvaluation(
                decision=decision,
                usage_summary=usage_summary,
                cost_summary=None,
            )

        expected_run_budgets = {
            _effective_budget_limit_id(limit): limit
            for limit in budget_limits
            if limit.scope == "run"
        }
        if (
            run_budget_authorities is not None
            and set(run_budget_authorities) != expected_run_budgets.keys()
        ):
            raise ValueError("Run budget authorities do not match the effective budget limits.")
        if run_budget_authorities is not None:
            for budget_limit_id, authority in run_budget_authorities.items():
                if (
                    type(authority) is not RunBudgetAccountingAuthority
                    or authority.budget_limit_id != budget_limit_id
                    or authority.currency != expected_run_budgets[budget_limit_id].currency
                ):
                    raise ValueError(
                        "Run budget authority does not match its effective budget limit."
                    )

        cost_summary: SessionCostSummary | None = None
        emitted_events: list[Event] = []
        for budget_limit in budget_limits:
            budget_events = events
            budget_baseline: SessionCostSummary | None = None
            budget_window_now = self._clock()
            if budget_limit.scope in {"app", "agent", "causal"}:
                budget_events = await self._budget_store.load_events_for_budget(
                    scope=budget_limit.scope,
                    key=budget_limit.key,
                    window=budget_limit.window,
                )
                budget_events = _merge_events_by_id(
                    budget_events,
                    additional_events,
                )
            elif budget_limit.scope == "run":
                budget_events = events_for_budget_window(
                    events,
                    budget_limit.window,
                    now=budget_window_now,
                )
                if run_budget_authorities is not None:
                    authority = run_budget_authorities[_effective_budget_limit_id(budget_limit)]
                    budget_events = [
                        event for event in budget_events if event.timestamp >= authority.started_at
                    ]
                else:
                    budget_baseline = estimate_session_cost(
                        session_id=session.id,
                        events=events_for_budget_window(
                            budget_baseline_events or [],
                            budget_limit.window,
                            now=budget_window_now,
                        ),
                        pricing=budget_limit.pricing,
                        currency=budget_limit.currency,
                    )
            elif budget_limit.scope == "session":
                budget_events = events_for_budget_window(
                    events,
                    budget_limit.window,
                    now=budget_window_now,
                )
            else:
                raise ValueError(f"Unsupported request budget scope: {budget_limit.scope}")

            cost_summary = estimate_session_cost(
                session_id=session.id,
                events=budget_events,
                pricing=budget_limit.pricing,
                currency=budget_limit.currency,
            )
            budget_outcome = _first_budget_limit_outcome(
                session=session,
                limit=budget_limit,
                cost_summary=cost_summary,
                cost_baseline=budget_baseline,
                effective_at=budget_window_now,
                billing_identity_state=billing_identity_state,
                pricing_provider_name=pricing_provider_name,
                model=model,
            )
            if budget_outcome is None:
                continue
            if budget_limit.action == "notify":
                if not _budget_notify_already_emitted_in_invocation(
                    budget_notify_events or [],
                    check=budget_outcome.check,
                ):
                    event = await self._emit_budget_limit_reached(
                        session=session,
                        agent_name=agent_name,
                        environment_name=environment_name,
                        check=budget_outcome.check,
                        execution_identity=execution_identity,
                        execution_profile_fingerprint=execution_profile_fingerprint,
                    )
                    emitted_events.append(event)
                    if budget_notify_events is not None:
                        budget_notify_events.append(event)
                continue
            return LimitEvaluation(
                decision=budget_outcome.decision,
                usage_summary=usage_summary,
                cost_summary=cost_summary,
                events=tuple(emitted_events),
            )
        return LimitEvaluation(
            decision=None,
            usage_summary=usage_summary,
            cost_summary=cost_summary,
            events=tuple(emitted_events),
        )

    async def evaluate_policy_budgets(
        self,
        *,
        session: Session,
        agent_name: str,
        environment_name: str | None,
        budget_policy: BudgetPolicy | None,
        billing_identity_state: BillingIdentityState = UNRESOLVED_BILLING_IDENTITY,
        pricing_provider_name: str | None = None,
        model: str | None = None,
        additional_usage_events: list[Event] | None = None,
        execution_identity: ModelStepIdentity | ModelAttemptIdentity | None = None,
        execution_profile_fingerprint: str | None = None,
    ) -> BudgetEvaluation:
        limits = budget_limits_for_session(
            policy=budget_policy,
            agent_name=agent_name,
            causal_budget_id=session.causal_budget_id,
        )
        if not limits:
            return BudgetEvaluation(check=None)
        emitted_events: list[Event] = []
        effective_provider_name = pricing_provider_name or session.provider_name
        effective_model = model or session.model
        additional_events = [
            event.model_copy(deep=True) for event in (additional_usage_events or [])
        ]
        for limit in limits:
            events = await self._budget_store.load_events_for_budget(
                scope=limit.scope,
                key=limit.key,
                window=limit.window,
            )
            events = _merge_events_by_id(events, additional_events)
            check = budget_check_from_events(
                limit=limit,
                events=events,
                provider_name=effective_provider_name,
                model=effective_model,
                billing_identity_state=billing_identity_state,
                effective_at=self._clock(),
            )
            deferred_contextual_check = (
                not isinstance(billing_identity_state, ResolvedBillingIdentity)
                and not check.limit_reached
                and has_deferred_contextual_price(
                    limit.pricing,
                    provider_name=effective_provider_name,
                    model=effective_model,
                )
            )
            if not deferred_contextual_check:
                emitted_events.append(
                    await self._event_writer.emit(
                        _event_with_budget_authority(
                            Event(
                                type=EventType.BUDGET_CHECKED,
                                session_id=session.id,
                                agent_name=agent_name,
                                environment_name=environment_name,
                                payload={
                                    **budget_check_payload(check),
                                    **_execution_identity_payload(execution_identity),
                                },
                            ),
                            execution_identity=execution_identity,
                            execution_profile_fingerprint=execution_profile_fingerprint,
                            additional_fields=("budget_limit_id",),
                        )
                    )
                )
            if not check.limit_reached:
                continue
            if limit.action == "notify":
                if not await self._budget_notify_already_emitted(limit=limit, check=check):
                    emitted_events.append(
                        await self._emit_budget_limit_reached(
                            session=session,
                            agent_name=agent_name,
                            environment_name=environment_name,
                            check=check,
                            execution_identity=execution_identity,
                            execution_profile_fingerprint=execution_profile_fingerprint,
                        )
                    )
                continue
            return BudgetEvaluation(check=check, events=tuple(emitted_events))
        return BudgetEvaluation(check=None, events=tuple(emitted_events))

    async def _budget_notify_already_emitted(
        self,
        *,
        limit: BudgetLimit,
        check: BudgetCheck,
    ) -> bool:
        if type(limit) not in {BudgetLimit, _EffectiveBudgetLimit}:
            raise TypeError("limit must be a BudgetLimit instance.")
        if type(check) is not BudgetCheck:
            raise TypeError("check must be a BudgetCheck instance.")
        if limit.action != "notify":
            return False

        since, until = limit.window.bounds()
        agent_name: str | None = None
        causal_budget_id: str | None = None
        if limit.scope == "agent":
            agent_name = require_clean_nonblank(limit.key or "", "key")
        elif limit.scope == "causal":
            causal_budget_id = require_clean_nonblank(limit.key or "", "key")
        elif limit.scope != "app":
            return False

        records = await query_all_event_records(
            self._session_store,
            EventQuery(
                causal_budget_id=causal_budget_id,
                event_type=EventType.BUDGET_LIMIT_REACHED,
                agent_name=agent_name,
                since=since,
                until=until,
                limit=5000,
            ),
        )
        return any(
            _budget_limit_reached_payload_matches(record.event.payload, check=check)
            for record in records
        )

    async def _emit_budget_limit_reached(
        self,
        *,
        session: Session,
        agent_name: str,
        environment_name: str | None,
        check: BudgetCheck,
        execution_identity: ModelStepIdentity | ModelAttemptIdentity | None = None,
        execution_profile_fingerprint: str | None = None,
    ) -> Event:
        return await self._event_writer.emit(
            _event_with_budget_authority(
                Event(
                    type=EventType.BUDGET_LIMIT_REACHED,
                    session_id=session.id,
                    agent_name=agent_name,
                    environment_name=environment_name,
                    payload={
                        **budget_limit_reached_payload(check),
                        **_execution_identity_payload(execution_identity),
                    },
                ),
                execution_identity=execution_identity,
                execution_profile_fingerprint=execution_profile_fingerprint,
                additional_fields=("budget_limit_id",),
            )
        )

    async def reserve_for_model_step(
        self,
        *,
        session: Session,
        agent_name: str,
        provider_name: str,
        environment_name: str | None,
        model_attempt_identity: ModelAttemptIdentity,
        budget_policy: BudgetPolicy | None,
        request_budget_limits: tuple[BudgetLimit, ...] = (),
        billing_identity: BillingIdentity | None = None,
        execution_profile_fingerprint: str | None = None,
        existing_reservation_ids: Collection[str] = (),
        reservation_identity_guard: BudgetReservationIdentityGuard | None = None,
    ) -> BudgetReservationSetup:
        model_attempt_identity = copy_model_attempt_identity(model_attempt_identity)
        limits = self.provider_reservation_limits(
            session=session,
            agent_name=agent_name,
            budget_policy=budget_policy,
            request_budget_limits=request_budget_limits,
        )
        if not limits:
            return BudgetReservationSetup((), None, (), None)

        reservations: list[BudgetStepReservation] = []
        reservation_ids = set(existing_reservation_ids)
        emitted_events: list[Event] = []
        reservation_failure: BudgetReservationResult | None = None
        release_reason = "reservation setup failed"
        expected_billing_identity = copy_billing_identity(billing_identity)
        settlement_event_payload = _interaction_bound_settlement_event_payload(
            self._event_writer,
            session_id=session.id,
            agent_name=agent_name,
            environment_name=environment_name,
            payload=(
                {}
                if execution_profile_fingerprint is None
                else {"execution_profile_fingerprint": execution_profile_fingerprint}
            ),
        )
        identity_guard = reservation_identity_guard or self.reservation_identity_guard()
        try:
            for limit in limits:
                expected_limit = _copy_effective_budget_limit(limit)
                ledger_limit = _copy_effective_budget_limit(limit)
                ledger_model_attempt_identity = copy_model_attempt_identity(model_attempt_identity)
                reservation_effective_at = self._clock()
                expected_requested_amount = _budget_reservation_amount(
                    limit=expected_limit,
                    provider_name=provider_name,
                    model=session.model,
                    effective_at=reservation_effective_at,
                    billing_identity=expected_billing_identity,
                )
                authority = _new_publishable_budget_reservation_authority(
                    self._event_writer,
                    limit=expected_limit,
                    model_attempt_identity=model_attempt_identity,
                    session_id=session.id,
                    agent_name=agent_name,
                    environment_name=environment_name,
                    provider_name=provider_name,
                    model=session.model,
                    settlement_event_payload=settlement_event_payload,
                    billing_identity=expected_billing_identity,
                    reserved_amount=expected_requested_amount,
                    fallback_settled_at=reservation_effective_at,
                    reservation_ttl_seconds=self.reservation_ttl_seconds,
                )
                ledger_billing_identity = copy_billing_identity(authority.billing_identity)
                try:
                    if expected_billing_identity is None and ledger_billing_identity is None:
                        result = await self._budget_ledger.reserve(
                            reservation_id=authority.reservation_id,
                            limit=ledger_limit,
                            session_id=session.id,
                            agent_name=agent_name,
                            provider_name=provider_name,
                            model=session.model,
                            model_attempt_identity=ledger_model_attempt_identity,
                            environment_name=environment_name,
                            settlement_event_payload=copy_durable_json_object(
                                settlement_event_payload,
                                "settlement_event_payload",
                            ),
                            settlement_fallback=authority.settlement_fallback,
                            effective_at=reservation_effective_at,
                        )
                    else:
                        result = await self._budget_ledger.reserve(
                            reservation_id=authority.reservation_id,
                            limit=ledger_limit,
                            session_id=session.id,
                            agent_name=agent_name,
                            provider_name=provider_name,
                            model=session.model,
                            model_attempt_identity=ledger_model_attempt_identity,
                            environment_name=environment_name,
                            settlement_event_payload=copy_durable_json_object(
                                settlement_event_payload,
                                "settlement_event_payload",
                            ),
                            settlement_fallback=authority.settlement_fallback,
                            requested_amount=expected_requested_amount,
                            billing_identity=ledger_billing_identity,
                            effective_at=reservation_effective_at,
                        )
                finally:
                    _validate_ledger_limit_unchanged(
                        ledger_limit,
                        expected=expected_limit,
                    )
                result = _validate_ledger_reservation_result(
                    result,
                    expected_reservation_id=authority.reservation_id,
                    limit=expected_limit,
                    model_attempt_identity=model_attempt_identity,
                    session_id=session.id,
                    agent_name=agent_name,
                    provider_name=provider_name,
                    model=session.model,
                    environment_name=environment_name,
                    settlement_event_payload=settlement_event_payload,
                    settlement_fallback=authority.settlement_fallback,
                    billing_identity=authority.billing_identity,
                    expected_requested_amount=expected_requested_amount,
                )
                reservation: BudgetStepReservation | None = None
                if result.accepted:
                    accepted_record = result.record
                    if accepted_record is None:  # pragma: no cover - validated above
                        raise RuntimeError("Accepted budget reservation has no record.")
                    if accepted_record.reservation_id in reservation_ids:
                        raise RuntimeError("Budget ledger reused a reservation identity.")
                    reservation = BudgetStepReservation(
                        limit=expected_limit,
                        record=accepted_record,
                        request_billing_identity=copy_billing_identity(expected_billing_identity),
                    )
                    # Once the ledger has accepted a fully validated record,
                    # retain it for pre-dispatch cleanup before constructing or
                    # attesting any event that can still fail. A proven identity
                    # conflict below removes it because this run must not settle
                    # the winning reservation.
                    reservations.append(reservation)
                reservation_event = Event(
                    type=(
                        EventType.BUDGET_RESERVED
                        if result.accepted
                        else EventType.BUDGET_RESERVATION_FAILED
                    ),
                    session_id=session.id,
                    agent_name=agent_name,
                    environment_name=environment_name,
                    payload=budget_reservation_payload(result),
                )
                reservation_event = _event_with_budget_authority(
                    reservation_event,
                    execution_identity=model_attempt_identity,
                    execution_profile_fingerprint=execution_profile_fingerprint,
                    additional_fields=(
                        "budget_limit_id",
                        "reservation_id",
                        "session_id",
                    ),
                )
                if result.accepted:
                    assert reservation is not None
                    try:
                        await identity_guard.claim(
                            accepted_record.reservation_id,
                            publication_session_id=reservation_event.session_id,
                            publication_id=reservation_event.id,
                        )
                    except (BudgetReservationIdentityConflict, SessionRunFenced):
                        _remove_reservation(reservations, reservation)
                        raise
                    reservation_ids.add(accepted_record.reservation_id)
                emitted_events.append(await self._event_writer.emit(reservation_event))
                await self.recover_pending_budget_settlements()
                if not result.accepted:
                    reservation_failure = result
                    release_reason = "reservation failed"
                    async for event in self.release_reservations(
                        reservations,
                        session=session,
                        agent_name=agent_name,
                        environment_name=environment_name,
                        reason=release_reason,
                    ):
                        emitted_events.append(event)
                    return BudgetReservationSetup(
                        tuple(reservations),
                        result,
                        tuple(emitted_events),
                        None,
                    )
        except BaseException as reservation_exc:
            async for event in self.settlement_events_preserving_failure(
                self.release_reservations(
                    reservations,
                    session=session,
                    agent_name=agent_name,
                    environment_name=environment_name,
                    reason=release_reason,
                ),
                authoritative_failure=reservation_exc,
            ):
                emitted_events.append(event)
            if not isinstance(reservation_exc, Exception):
                raise
            return BudgetReservationSetup(
                tuple(reservations),
                reservation_failure,
                tuple(emitted_events),
                reservation_exc,
            )
        return BudgetReservationSetup(tuple(reservations), None, tuple(emitted_events), None)

    def provider_reservation_limits(
        self,
        *,
        session: Session,
        agent_name: str,
        budget_policy: BudgetPolicy | None,
        request_budget_limits: tuple[BudgetLimit, ...] = (),
    ) -> tuple[_EffectiveBudgetLimit, ...]:
        return tuple(
            limit
            for limit in self.provider_budget_limits(
                session=session,
                agent_name=agent_name,
                budget_policy=budget_policy,
                request_budget_limits=request_budget_limits,
            )
            if limit.reservation is not None
        )

    def provider_budget_limits(
        self,
        *,
        session: Session,
        agent_name: str,
        budget_policy: BudgetPolicy | None,
        request_budget_limits: tuple[BudgetLimit, ...] = (),
    ) -> tuple[_EffectiveBudgetLimit, ...]:
        effective_request_limits = request_budget_limits_for_session(
            limits=request_budget_limits,
            agent_name=agent_name,
            causal_budget_id=session.causal_budget_id,
        )
        return (
            *budget_limits_for_session(
                policy=budget_policy,
                agent_name=agent_name,
                causal_budget_id=session.causal_budget_id,
            ),
            *effective_request_limits,
        )

    async def _commit_expected_reconciliation(
        self,
        reservation: BudgetStepReservation,
        expected: BudgetReconciliation,
    ) -> BudgetReconciliation:
        """Commit one publication-safe reconciliation and verify its exact replay."""

        return await self._commit_expected_reconciliation_for_record(
            reservation.record,
            expected,
        )

    async def _commit_expected_reconciliation_for_record(
        self,
        record: BudgetReservationRecord,
        expected: BudgetReconciliation,
    ) -> BudgetReconciliation:
        """Commit settlement using immutable ledger authority retained by recovery."""

        try:
            expected = _publication_safe_reconciliation_for_record(
                self._event_writer.prepare,
                record=record,
                reconciliation=expected,
            )
        except ValueError as exact_error:
            fallback = budget_reconciliation_preview(
                record,
                actual_amount=record.reserved_amount,
                settlement_kind="conservative",
                reason=record.settlement_fallback.reconciliation_reason,
                occurred_at=expected.settled_at,
            )
            try:
                expected = _publication_safe_reconciliation_for_record(
                    self._event_writer.prepare,
                    record=record,
                    reconciliation=fallback,
                )
            except Exception as fallback_error:
                add_exception_note_safely(
                    exact_error,
                    "The prevalidated conservative budget-settlement fallback also failed: "
                    f"{type(fallback_error).__name__}.",
                )
                raise exact_error from fallback_error
        if expected.status != "reconciled" or expected.actual_amount is None:
            raise ValueError("Expected budget reconciliation must charge a reservation.")
        if expected.settlement_kind == "released":
            raise ValueError("Expected budget reconciliation cannot release a reservation.")
        pricing = _reconciliation_pricing_evidence(expected)
        billing_identity = copy_billing_identity(
            expected.billing_identity if record.billing_identity is not None else None
        )
        if billing_identity is None:
            reconciliation = await self._budget_ledger.reconcile(
                reservation_id=expected.reservation_id,
                actual_amount=expected.actual_amount,
                settlement_kind=expected.settlement_kind,
                reason=expected.reason,
                occurred_at=expected.settled_at,
                pricing=pricing,
            )
        else:
            reconciliation = await self._budget_ledger.reconcile(
                reservation_id=expected.reservation_id,
                actual_amount=expected.actual_amount,
                settlement_kind=expected.settlement_kind,
                reason=expected.reason,
                occurred_at=expected.settled_at,
                billing_identity=billing_identity,
                pricing=pricing,
            )
        reconciliation = _validate_ledger_reconciliation_against_reservation_record(
            reconciliation,
            record=record,
            expected_status="reconciled",
            expected_settlement_kind=expected.settlement_kind,
            expected_actual_amount=expected.actual_amount,
            expected_reason=expected.reason,
            expected_billing_identity=expected.billing_identity,
            expected_pricing=pricing,
        )
        if reconciliation != expected:
            raise RuntimeError("Budget ledger changed exact settlement evidence.")
        return reconciliation

    async def _commit_expected_release(
        self,
        reservation: BudgetStepReservation,
        expected: BudgetReconciliation,
    ) -> BudgetReconciliation:
        """Commit one publication-safe release and verify its exact replay."""

        try:
            expected = _publication_safe_reconciliation(
                self._event_writer.prepare,
                reservation=reservation,
                reconciliation=expected,
            )
        except ValueError as exact_error:
            fallback = budget_release_preview(
                reservation.record,
                reason=reservation.record.settlement_fallback.release_reason,
                occurred_at=reservation.record.settlement_fallback.settled_at,
            )
            try:
                expected = _publication_safe_reconciliation(
                    self._event_writer.prepare,
                    reservation=reservation,
                    reconciliation=fallback,
                )
            except Exception as fallback_error:
                add_exception_note_safely(
                    exact_error,
                    "The prevalidated budget-release fallback also failed: "
                    f"{type(fallback_error).__name__}.",
                )
                raise exact_error from fallback_error
        if (
            expected.status != "released"
            or expected.settlement_kind != "released"
            or expected.actual_amount is not None
        ):
            raise ValueError("Expected budget release must release one reservation.")
        reconciliation = await self._budget_ledger.release(
            reservation_id=expected.reservation_id,
            reason=expected.reason or "reservation released",
            occurred_at=expected.settled_at,
        )
        reconciliation = _validate_ledger_reconciliation(
            reconciliation,
            reservation=reservation,
            expected_status="released",
            expected_settlement_kind="released",
            expected_actual_amount=None,
            expected_reason=expected.reason,
            expected_billing_identity=expected.billing_identity,
        )
        if reconciliation != expected:
            expiration_reason = reservation.record.settlement_fallback.expiration_reason
            if (
                expiration_reason is None
                or reconciliation.reason != expiration_reason
                or reconciliation.settled_at != reservation.record.settlement_fallback.settled_at
                or _publication_safe_reconciliation(
                    self._event_writer.prepare,
                    reservation=reservation,
                    reconciliation=reconciliation,
                )
                != reconciliation
            ):
                raise RuntimeError("Budget ledger changed exact release evidence.")
        return reconciliation

    async def reconcile_dispatched_reservations(
        self,
        reservations: list[BudgetStepReservation],
        *,
        lifecycle: BudgetModelStepLifecycle,
        session: Session,
        agent_name: str,
        environment_name: str | None,
        unknown_reason: str,
    ) -> AsyncIterator[Event]:
        if not lifecycle.dispatches:
            raise ValueError("Cannot reconcile a model step with no provider dispatch.")
        active_lifecycle_reservation_ids = {
            reservation.record.reservation_id
            for dispatch in lifecycle.dispatches
            for reservation in dispatch.reservations
            if reservation.record.reservation_id not in dispatch.settled_reservation_ids
        }
        if lifecycle.pending_reservations is not None:
            active_lifecycle_reservation_ids.update(
                reservation.record.reservation_id for reservation in lifecycle.pending_reservations
            )
        if active_lifecycle_reservation_ids != {
            reservation.record.reservation_id for reservation in reservations
        }:
            raise RuntimeError("Budget dispatch lifecycle lost a reservation.")

        for dispatch in lifecycle.dispatches:
            if dispatch.settled:
                continue
            completed_settlements = (
                None
                if dispatch.completion is None
                else {
                    reconciliation.reservation_id: reconciliation
                    for reconciliation in model_completion_budget_settlements(
                        dispatch.completion,
                        reservation_ids=(
                            reservation.record.reservation_id
                            for reservation in dispatch.reservations
                        ),
                    )
                }
            )
            emitted_events: list[Event] = []
            settlement_failure: Exception | None = None
            async with lifecycle.reservation_transition_lock:
                for reservation in dispatch.reservations:
                    reservation_id = reservation.record.reservation_id
                    if reservation_id in dispatch.settled_reservation_ids:
                        continue
                    try:
                        expected = (
                            budget_reconciliation_preview(
                                reservation.record,
                                actual_amount=reservation.record.reserved_amount,
                                settlement_kind="conservative",
                                reason=unknown_reason,
                                occurred_at=self._clock(),
                            )
                            if completed_settlements is None
                            else completed_settlements[reservation_id]
                        )
                        assert expected.actual_amount is not None
                        if expected.settlement_kind == "released":
                            raise RuntimeError(
                                "Model completion cannot release a dispatched reservation."
                            )
                        reconciliation = await self._commit_expected_reconciliation(
                            reservation,
                            expected,
                        )
                        settlement = await self._load_committed_settlement(reconciliation)
                        if (
                            settlement.session_id != session.id
                            or settlement.agent_name != agent_name
                            or settlement.environment_name != environment_name
                        ):
                            raise RuntimeError(
                                "Budget settlement audit context conflicts with its model step."
                            )
                        emitted_events.append(await self.publish_budget_settlement(settlement))
                    except Exception as exc:
                        settlement_failure = exc
                        break
                    dispatch.settled_reservation_ids.add(reservation_id)
                    reservations[:] = [
                        active
                        for active in reservations
                        if active.record.reservation_id != reservation_id
                    ]

            for event in emitted_events:
                yield event
            if settlement_failure is not None:
                raise settlement_failure

    async def settle_after_model_failure(
        self,
        reservations: list[BudgetStepReservation],
        *,
        lifecycle: BudgetModelStepLifecycle,
        session: Session,
        agent_name: str,
        environment_name: str | None,
        release_reason: str,
        unknown_reason: str = UNKNOWN_POST_DISPATCH_BUDGET_REASON,
    ) -> AsyncIterator[Event]:
        if lifecycle.provider_dispatch_may_have_occurred:
            async for event in self.reconcile_dispatched_reservations(
                reservations,
                lifecycle=lifecycle,
                session=session,
                agent_name=agent_name,
                environment_name=environment_name,
                unknown_reason=unknown_reason,
            ):
                yield event

        if lifecycle.pending_reservations is not None:
            async for event in self.release_reservations(
                list(lifecycle.pending_reservations),
                session=session,
                agent_name=agent_name,
                environment_name=environment_name,
                reason=release_reason,
            ):
                yield event

    async def settlement_events_preserving_failure(
        self,
        settlement_events: AsyncIterator[Event],
        *,
        authoritative_failure: BaseException,
    ) -> AsyncIterator[Event]:
        try:
            async for event in settlement_events:
                yield event
        except Exception as accounting_exc:
            add_budget_failure_note(
                authoritative_failure,
                operation="settlement",
                accounting_failure=accounting_exc,
            )

    async def before_provider_dispatch(
        self,
        reservations: list[BudgetStepReservation],
        *,
        lifecycle: BudgetModelStepLifecycle,
        model_attempt_identity: ModelAttemptIdentity,
    ) -> None:
        model_attempt_identity = copy_model_attempt_identity(model_attempt_identity)
        async with lifecycle.reservation_transition_lock:
            if reservations and self._budget_ledger.reservation_ttl_seconds is not None:
                try:
                    await self.renew_reservations(reservations)
                except BudgetReservationLeaseLost as exc:
                    if not lifecycle.provider_dispatch_may_have_occurred:
                        raise BudgetReservationLeaseLostBeforeModelDispatch(
                            "Budget reservation lease was lost before model dispatch."
                        ) from exc
                    raise
            deferred_failure = await self.mark_reservations_dispatched(
                reservations,
                dispatch_id=model_attempt_identity.model_attempt_id,
            )
            lifecycle.mark_provider_dispatch(model_attempt_identity)
            if deferred_failure is not None:
                raise deferred_failure

    async def mark_reservations_dispatched(
        self,
        reservations: Collection[BudgetStepReservation],
        *,
        dispatch_id: str,
    ) -> BaseException | None:
        """Fence every reservation, exactly replaying an ambiguous acknowledgement."""

        dispatch_id = require_clean_nonblank(dispatch_id, "dispatch_id")
        expected_records = tuple(reservation.record for reservation in reservations)
        if not expected_records:
            return None
        dispatched_at = self._clock()
        reservation_ids = tuple(record.reservation_id for record in expected_records)

        async def mark_once() -> None:
            marked_records = await self._budget_ledger.mark_dispatched(
                reservation_ids=reservation_ids,
                dispatch_id=dispatch_id,
                dispatched_at=dispatched_at,
            )
            if type(marked_records) is not tuple or len(marked_records) != len(expected_records):
                raise RuntimeError("Budget ledger returned an incomplete dispatch fence.")
            for expected, marked in zip(expected_records, marked_records, strict=True):
                if type(marked) is not BudgetReservationRecord:
                    raise TypeError(
                        "Budget ledger dispatch fences must be BudgetReservationRecord instances."
                    )
                marked = BudgetReservationRecord.model_validate(marked.model_dump(mode="python"))
                expected_marked = expected.model_copy(
                    update={
                        "dispatch_id": dispatch_id,
                        "dispatched_at": dispatched_at,
                        # A lease heartbeat immediately before the fence may
                        # have advanced this ledger-owned timestamp.
                        "updated_at": marked.updated_at,
                    },
                    deep=True,
                )
                if marked.updated_at < expected.updated_at or marked != expected_marked:
                    raise RuntimeError("Budget ledger returned a conflicting dispatch fence.")

        try:
            await mark_once()
            return None
        except BaseException as first_error:
            cancellation = first_error if isinstance(first_error, asyncio.CancelledError) else None
            replay_task = asyncio.create_task(mark_once())
            outcome = await await_shielded_task_outcome(
                replay_task,
                cancellation=cancellation,
            )
            if outcome.error is not None:
                replay_error = outcome.error
                if isinstance(replay_error, asyncio.CancelledError):
                    replay_error = unexpected_child_cancellation_error(
                        replay_error,
                        operation="Budget dispatch-fence acknowledgement replay",
                    )
                if outcome.cancellation is not None:
                    add_exception_note_safely(
                        outcome.cancellation,
                        "Budget dispatch-fence acknowledgement replay also failed after "
                        f"{type(first_error).__name__}.",
                    )
                    raise outcome.cancellation from replay_error
                add_exception_note_safely(
                    first_error,
                    "Budget dispatch-fence acknowledgement replay also failed: "
                    f"{type(replay_error).__name__}.",
                )
                raise first_error from replay_error
            deferred_failure = outcome.cancellation
            if deferred_failure is None and not isinstance(first_error, Exception):
                deferred_failure = first_error
            return deferred_failure

    async def release_pre_provider_dispatch_reservations(
        self,
        *,
        reservation_ids: tuple[str, ...],
        recovery_contexts: tuple[BudgetReservationRecoveryContext, ...],
        dispatch_id: str,
    ) -> list[Event]:
        """Release one receipt-less stage's exact reservation batch and audit it."""

        if type(reservation_ids) is not tuple:
            raise TypeError("reservation_ids must be a tuple.")
        reservation_ids = tuple(
            require_clean_nonblank(reservation_id, "reservation_id")
            for reservation_id in reservation_ids
        )
        if len(set(reservation_ids)) != len(reservation_ids):
            raise ValueError("reservation_ids must be distinct.")
        if not reservation_ids:
            return []
        if tuple(context.reservation_id for context in recovery_contexts) != reservation_ids:
            raise ValueError("Model-completion recovery lost its reservation authority.")
        dispatch_id = require_clean_nonblank(dispatch_id, "dispatch_id")
        released_at = self._clock()
        original_records: list[BudgetReservationRecord] = []
        for reservation_id, context in zip(reservation_ids, recovery_contexts, strict=True):
            raw_record = await self._budget_ledger.load_reservation(reservation_id)
            if raw_record is None:
                raise KeyError(f"Budget reservation not found: {reservation_id}")
            record = _validate_ledger_reservation_against_recovery_context(
                raw_record,
                context=context,
                dispatch_id=dispatch_id,
            )
            if record.status not in {"active", "released"}:
                raise RuntimeError(
                    "Budget ledger pre-provider release changed its reservation authority."
                )
            original_records.append(record)

        async def release_once() -> tuple[BudgetReconciliation, ...]:
            raw = await self._budget_ledger.release_pre_provider_dispatch(
                reservation_ids=reservation_ids,
                dispatch_id=dispatch_id,
                reason=PRE_PROVIDER_DISPATCH_BUDGET_RELEASE_REASON,
                occurred_at=released_at,
            )
            if type(raw) is not tuple or len(raw) != len(reservation_ids):
                raise RuntimeError(
                    "Budget ledger returned an incomplete pre-provider release batch."
                )
            reconciliations: list[BudgetReconciliation] = []
            for record, item in zip(original_records, raw, strict=True):
                try:
                    reconciliation = _validate_ledger_reconciliation_against_reservation_record(
                        item,
                        record=record,
                        expected_status="released",
                        expected_settlement_kind="released",
                        expected_actual_amount=None,
                        expected_reason=PRE_PROVIDER_DISPATCH_BUDGET_RELEASE_REASON,
                        expected_billing_identity=record.billing_identity,
                    )
                except TypeError:
                    raise
                except Exception:
                    raise RuntimeError(
                        "Budget ledger pre-provider release changed its requested outcome."
                    ) from None
                reconciliations.append(reconciliation)
            return tuple(reconciliations)

        async def release_with_exact_replay() -> tuple[BudgetReconciliation, ...]:
            try:
                return await release_once()
            except (Exception, asyncio.CancelledError) as first_error:
                try:
                    return await release_once()
                except (Exception, asyncio.CancelledError) as replay_error:
                    add_exception_note_safely(
                        replay_error,
                        "Exact pre-provider budget release replay also failed after "
                        f"{type(first_error).__name__}.",
                    )
                    raise replay_error from first_error

        release_task = asyncio.create_task(release_with_exact_replay())
        outcome = await await_shielded_task_outcome(release_task)
        cancellation = outcome.cancellation
        error = outcome.error
        if isinstance(error, asyncio.CancelledError) and cancellation is None:
            error = unexpected_child_cancellation_error(
                error,
                operation="Pre-provider budget reservation release",
            )
        if error is not None:
            if cancellation is not None:
                add_exception_note_safely(
                    cancellation,
                    f"Pre-provider budget reservation release also failed: {type(error).__name__}.",
                )
                raise cancellation from error
            raise error
        if outcome.result is None:
            result_error = RuntimeError(
                "Pre-provider budget reservation release returned no acknowledgement."
            )
            if cancellation is not None:
                add_exception_note_safely(cancellation, str(result_error))
                raise cancellation from result_error
            raise result_error

        events: list[Event] = []
        try:
            for reconciliation in outcome.result:
                settlement = await self._load_committed_settlement(reconciliation)
                events.append(
                    settlement.event.model_copy(deep=True)
                    if settlement.event_published
                    else await self.publish_budget_settlement(settlement)
                )
        except BaseException as publication_error:
            if cancellation is not None:
                add_exception_note_safely(
                    cancellation,
                    "Pre-provider budget release publication also failed: "
                    f"{type(publication_error).__name__}.",
                )
                raise cancellation from publication_error
            raise
        if cancellation is not None:
            raise cancellation
        return events

    async def require_model_completion_reservation_settlements(
        self,
        *,
        reservation_ids: tuple[str, ...],
        recovery_contexts: tuple[BudgetReservationRecoveryContext, ...],
        dispatch_id: str,
    ) -> None:
        """Require durable settlement/outbox authority before a model stage is cleared."""

        if type(reservation_ids) is not tuple:
            raise TypeError("reservation_ids must be a tuple.")
        reservation_ids = tuple(
            require_clean_nonblank(reservation_id, "reservation_id")
            for reservation_id in reservation_ids
        )
        if len(set(reservation_ids)) != len(reservation_ids):
            raise ValueError("reservation_ids must be distinct.")
        if tuple(context.reservation_id for context in recovery_contexts) != reservation_ids:
            raise ValueError("Model-completion recovery lost its reservation authority.")
        dispatch_id = require_clean_nonblank(dispatch_id, "dispatch_id")
        for reservation_id, context in zip(reservation_ids, recovery_contexts, strict=True):
            raw_record = await self._budget_ledger.load_reservation(reservation_id)
            if raw_record is None or type(raw_record) is not BudgetReservationRecord:
                raise ModelCompletionBudgetSettlementPending(
                    "Model-completion budget reservation has no durable terminal authority: "
                    f"{reservation_id}"
                )
            record = _validate_ledger_reservation_against_recovery_context(
                raw_record,
                context=context,
                dispatch_id=dispatch_id,
            )
            raw_settlement = await self._budget_ledger.load_settlement(
                budget_settlement_id(reservation_id)
            )
            if record.status == "active" or raw_settlement is None:
                raise ModelCompletionBudgetSettlementPending(
                    f"Model-completion budget reservation remains unsettled: {reservation_id}"
                )
            settlement = _validate_ledger_settlement_record(raw_settlement)
            if record.status not in {"reconciled", "released"}:
                raise RuntimeError(
                    "Model-completion budget reservation has a conflicting terminal status."
                )
            _validate_ledger_reconciliation_against_reservation_record(
                settlement.reconciliation,
                record=record,
                expected_status=record.status,
                expected_settlement_kind=settlement.settlement_kind,
                expected_actual_amount=record.actual_amount,
                expected_reason=record.reason,
                expected_billing_identity=record.billing_identity,
                expected_pricing=_reconciliation_pricing_evidence(settlement.reconciliation),
            )
            if (
                settlement.reservation_id != reservation_id
                or settlement.session_id != record.session_id
                or settlement.agent_name != record.agent_name
                or settlement.environment_name != record.environment_name
            ):
                raise RuntimeError(
                    "Model-completion budget settlement conflicts with its reservation."
                )

    async def reconcile_manual_model_completion_reservations(
        self,
        *,
        reservation_ids: tuple[str, ...],
        recovery_contexts: tuple[BudgetReservationRecoveryContext, ...],
        session: Session,
        provider_name: str,
        model_attempt_identity: ModelAttemptIdentity,
        dispatch_id: str,
    ) -> list[Event]:
        """Conservatively settle an ambiguous synchronous model dispatch."""

        if type(reservation_ids) is not tuple:
            raise TypeError("reservation_ids must be a tuple.")
        reservation_ids = tuple(
            require_clean_nonblank(reservation_id, "reservation_id")
            for reservation_id in reservation_ids
        )
        if len(set(reservation_ids)) != len(reservation_ids):
            raise ValueError("reservation_ids must be distinct.")
        if tuple(context.reservation_id for context in recovery_contexts) != reservation_ids:
            raise ValueError("Model-completion recovery lost its reservation authority.")
        if type(session) is not Session:
            raise TypeError("session must be a Session.")
        provider_name = require_clean_nonblank(provider_name, "provider_name")
        model_attempt_identity = copy_model_attempt_identity(model_attempt_identity)
        dispatch_id = require_clean_nonblank(dispatch_id, "dispatch_id")

        records: list[BudgetReservationRecord] = []
        for reservation_id, context in zip(reservation_ids, recovery_contexts, strict=True):
            raw_record = await self._budget_ledger.load_reservation(reservation_id)
            if raw_record is None:
                raise KeyError(f"Budget reservation not found: {reservation_id}")
            record = _validate_ledger_reservation_against_recovery_context(
                raw_record,
                context=context,
                dispatch_id=dispatch_id,
            )
            if (
                record.budget_limit_id != context.budget_limit_id
                or record.model_step_id != model_attempt_identity.model_step_id
                or record.model_attempt_id != model_attempt_identity.model_attempt_id
                or record.session_id != session.id
                or record.agent_name != session.agent_name
                or record.environment_name != session.environment_name
                or record.provider_name != provider_name
                or record.model != session.model
                or record.dispatch_id != dispatch_id
                or record.status not in {"active", "reconciled"}
            ):
                raise RuntimeError(
                    "Manual model recovery found conflicting budget reservation authority."
                )
            records.append(record)

        settled_at = self._clock()
        events: list[Event] = []
        for record in records:
            if record.status == "active":
                expected = budget_reconciliation_preview(
                    record,
                    actual_amount=record.reserved_amount,
                    settlement_kind="conservative",
                    reason=UNKNOWN_POST_DISPATCH_BUDGET_REASON,
                    occurred_at=settled_at,
                )
                reconciliation = await self._commit_expected_reconciliation_for_record(
                    record,
                    expected,
                )
                settlement = await self._load_committed_settlement(reconciliation)
            else:
                raw_settlement = await self._budget_ledger.load_settlement(
                    budget_settlement_id(record.reservation_id)
                )
                settlement = (
                    None
                    if raw_settlement is None
                    else _validate_ledger_settlement_record(raw_settlement)
                )
                if settlement is None:
                    raise RuntimeError(
                        "Manual model recovery found a reconciled reservation without settlement."
                    )

            reconciliation = settlement.reconciliation
            if reconciliation.reason not in {
                UNKNOWN_POST_DISPATCH_BUDGET_REASON,
                record.settlement_fallback.reconciliation_reason,
            }:
                raise RuntimeError(
                    "Manual model recovery found a conflicting budget settlement reason."
                )
            _validate_ledger_reconciliation_against_reservation_record(
                reconciliation,
                record=record,
                expected_status="reconciled",
                expected_settlement_kind="conservative",
                expected_actual_amount=record.reserved_amount,
                expected_reason=reconciliation.reason,
                expected_billing_identity=record.billing_identity,
            )
            if (
                settlement.reservation_id != record.reservation_id
                or settlement.session_id != record.session_id
                or settlement.agent_name != record.agent_name
                or settlement.environment_name != record.environment_name
            ):
                raise RuntimeError(
                    "Manual model recovery found a settlement for another reservation owner."
                )
            events.append(
                settlement.event.model_copy(deep=True)
                if settlement.event_published
                else await self.publish_budget_settlement(settlement)
            )
        return events

    async def publish_budget_settlement(
        self,
        settlement: BudgetSettlementRecord,
    ) -> Event:
        """Publish one ledger-owned audit event and acknowledge its exact handoff."""

        settlement = _validate_ledger_settlement_record(settlement)
        prepared_event = self._event_writer.prepare_exact_replay(settlement.event)
        persisted = await self._event_writer.persist_exact_replay(prepared_event)
        if persisted != settlement.event:
            raise RuntimeError("Budget settlement publication changed its durable audit event.")
        acknowledged = _validate_ledger_settlement_record(
            await self._budget_ledger.mark_settlement_event_published(
                settlement_id=settlement.settlement_id,
                event_id=settlement.event.id,
            )
        )
        if not acknowledged.event_published or acknowledged.model_copy(
            update={"event_published": False}, deep=True
        ) != settlement.model_copy(update={"event_published": False}, deep=True):
            raise RuntimeError("Budget ledger acknowledged a conflicting settlement event.")
        await self._event_writer.fan_out_persisted([settlement.event])
        return settlement.event.model_copy(deep=True)

    async def recover_pending_budget_settlements(
        self,
        *,
        session_id: str | None = None,
    ) -> list[Event]:
        """Publish reachable settlement outbox rows without reapplying amounts."""

        if session_id is not None:
            session_id = require_clean_nonblank(session_id, "session_id")
            return await self._recover_pending_budget_settlement_pages(
                session_id=session_id,
                after=None,
                max_pages=None,
            )
        async with self._global_settlement_recovery_lock:
            return await self._recover_pending_budget_settlement_pages(
                session_id=None,
                after=self._global_settlement_recovery_after,
                max_pages=10,
            )

    async def _recover_pending_budget_settlement_pages(
        self,
        *,
        session_id: str | None,
        after: BudgetSettlementCursor | None,
        max_pages: int | None,
    ) -> list[Event]:
        recovered: list[Event] = []
        pages = 0
        while True:
            page_limit = 100
            pending = _validate_ledger_settlement_page(
                await self._budget_ledger.list_pending_settlements(
                    session_id=session_id,
                    after=after,
                    limit=page_limit,
                ),
                session_id=session_id,
                after=after,
                limit=page_limit,
            )
            if not pending:
                if session_id is None:
                    self._global_settlement_recovery_after = None
                return recovered
            pages += 1
            for settlement in pending:
                next_cursor = BudgetSettlementCursor(
                    settled_at=settlement.reconciliation.settled_at,
                    settlement_id=settlement.settlement_id,
                )
                if session_id is None:
                    try:
                        owner = await self._session_store.load(settlement.session_id)
                    except KeyError:
                        owner = None
                    if owner is None:
                        after = next_cursor
                        self._global_settlement_recovery_after = after
                        continue
                    try:
                        self._event_writer.prepare_exact_replay(settlement.event)
                    except ValueError:
                        # The owning publication domain may use a different
                        # workload-secret policy. Retain the outbox row for it.
                        after = next_cursor
                        self._global_settlement_recovery_after = after
                        continue
                try:
                    recovered.append(await self.publish_budget_settlement(settlement))
                except KeyError:
                    if session_id is not None:
                        raise
                    owner = await self._session_store.load(settlement.session_id)
                    if owner is not None:
                        raise
                after = next_cursor
                if session_id is None:
                    self._global_settlement_recovery_after = after
            if max_pages is not None and pages >= max_pages:
                return recovered

    async def reconcile_model_completion_settlements(
        self,
        completion_event: Event,
        *,
        reservation_ids: Collection[str],
    ) -> list[Event]:
        """Converge one durable model completion into its original reservations."""

        expected = model_completion_budget_settlements(
            completion_event,
            reservation_ids=reservation_ids,
        )
        events: list[Event] = []
        for evidence in expected:
            if evidence.actual_amount is None or evidence.settlement_kind == "released":
                raise ValueError("Model completion contains a non-reconcilable settlement.")
            raw_settlement = await self._budget_ledger.load_settlement(evidence.settlement_id)
            settlement = (
                None
                if raw_settlement is None
                else _validate_ledger_settlement_record(raw_settlement)
            )
            if settlement is None:
                reconciliation = await self._budget_ledger.reconcile(
                    reservation_id=evidence.reservation_id,
                    actual_amount=evidence.actual_amount,
                    settlement_kind=evidence.settlement_kind,
                    reason=evidence.reason,
                    occurred_at=evidence.settled_at,
                    billing_identity=copy_billing_identity(evidence.billing_identity),
                    pricing=_reconciliation_pricing_evidence(evidence),
                )
                if reconciliation != evidence:
                    raise RuntimeError(
                        "Recovered budget settlement conflicts with durable model completion."
                    )
                settlement = await self._load_committed_settlement(reconciliation)
            elif (
                settlement.reconciliation != evidence
                or settlement.reservation_id != evidence.reservation_id
            ):
                raise RuntimeError(
                    "Committed budget settlement conflicts with durable model completion."
                )
            if settlement.event_published:
                events.append(settlement.event.model_copy(deep=True))
            else:
                events.append(await self.publish_budget_settlement(settlement))
        return events

    async def recover_model_completion_budget_evidence(
        self,
        completion_event: Event,
        *,
        reservation_ids: tuple[str, ...],
        recovery_contexts: tuple[BudgetReservationRecoveryContext, ...],
        session: Session,
        provider_name: str,
        model_attempt_identity: ModelAttemptIdentity,
        dispatch_id: str,
        request_billing_identity: BillingIdentity | None,
    ) -> Event:
        """Attach the original reservation pricing to one recovered completion."""

        if (
            type(completion_event) is not Event
            or completion_event.type is not EventType.MODEL_COMPLETED
        ):
            raise ValueError("Recovered budget evidence requires one model.completed event.")
        reservations = await self._reconstruct_provider_operation_reservations(
            reservation_ids=reservation_ids,
            recovery_contexts=recovery_contexts,
            session=session,
            provider_name=provider_name,
            model_attempt_identity=model_attempt_identity,
            dispatch_id=dispatch_id,
            request_billing_identity=request_billing_identity,
        )
        prepared_event = self._event_writer.prepare(completion_event)
        reconciliations: list[BudgetReconciliation] = []
        settled_at = self._clock()
        for reservation in reservations:
            record = reservation.record
            if record.status == "active":
                expected = _model_completion_reconciliation(
                    prepared_event,
                    reservation,
                    settled_at=settled_at,
                )
                reconciliations.append(
                    _publication_safe_reconciliation(
                        self._event_writer.prepare,
                        reservation=reservation,
                        reconciliation=expected,
                    )
                )
                continue
            if record.status != "reconciled":
                raise ValueError(
                    "A dispatched provider-operation reservation was released unexpectedly."
                )
            settlement = await self._load_provider_operation_settlement(record)
            reconciliations.append(settlement.reconciliation)
        return _model_completion_with_reconciliation_evidence(
            prepared_event,
            reconciliations=tuple(reconciliations),
            reservation_ids=reservation_ids,
            prepare_event=self._event_writer.prepare,
        )

    async def reconcile_cancelled_provider_operation_reservations(
        self,
        *,
        reservation_ids: tuple[str, ...],
        recovery_contexts: tuple[BudgetReservationRecoveryContext, ...],
        session: Session,
        provider_name: str,
        model_attempt_identity: ModelAttemptIdentity,
        dispatch_id: str,
        request_billing_identity: BillingIdentity | None,
    ) -> list[Event]:
        """Conservatively settle the original reservations after confirmed cancellation."""

        return await self.reconcile_unavailable_provider_operation_reservations(
            reservation_ids=reservation_ids,
            recovery_contexts=recovery_contexts,
            session=session,
            provider_name=provider_name,
            model_attempt_identity=model_attempt_identity,
            dispatch_id=dispatch_id,
            request_billing_identity=request_billing_identity,
            reason="provider operation cancellation confirmed; charged reserved amount",
            occurred_at=self._clock(),
        )

    async def reconcile_unavailable_provider_operation_reservations(
        self,
        *,
        reservation_ids: tuple[str, ...],
        recovery_contexts: tuple[BudgetReservationRecoveryContext, ...],
        session: Session,
        provider_name: str,
        model_attempt_identity: ModelAttemptIdentity,
        dispatch_id: str,
        request_billing_identity: BillingIdentity | None,
        reason: str,
        occurred_at: datetime,
    ) -> list[Event]:
        """Conservatively settle reservations whose provider usage remains unknown."""

        reservations = await self._reconstruct_provider_operation_reservations(
            reservation_ids=reservation_ids,
            recovery_contexts=recovery_contexts,
            session=session,
            provider_name=provider_name,
            model_attempt_identity=model_attempt_identity,
            dispatch_id=dispatch_id,
            request_billing_identity=request_billing_identity,
        )
        events: list[Event] = []
        for reservation in reservations:
            record = reservation.record
            if record.status == "active":
                expected = budget_reconciliation_preview(
                    record,
                    actual_amount=record.reserved_amount,
                    settlement_kind="conservative",
                    reason=reason,
                    occurred_at=occurred_at,
                )
                reconciliation = await self._commit_expected_reconciliation(
                    reservation,
                    expected,
                )
                settlement = await self._load_committed_settlement(reconciliation)
            elif record.status == "reconciled":
                settlement = await self._load_provider_operation_settlement(record)
            else:
                raise ValueError(
                    "A dispatched provider-operation reservation was released unexpectedly."
                )
            events.append(
                settlement.event.model_copy(deep=True)
                if settlement.event_published
                else await self.publish_budget_settlement(settlement)
            )
        return events

    async def _reconstruct_provider_operation_reservations(
        self,
        *,
        reservation_ids: tuple[str, ...],
        recovery_contexts: tuple[BudgetReservationRecoveryContext, ...],
        session: Session,
        provider_name: str,
        model_attempt_identity: ModelAttemptIdentity,
        dispatch_id: str,
        request_billing_identity: BillingIdentity | None,
    ) -> tuple[BudgetStepReservation, ...]:
        if tuple(context.reservation_id for context in recovery_contexts) != reservation_ids:
            raise ValueError(
                "Provider-operation recovery lost its original budget reservation order."
            )
        model_attempt_identity = copy_model_attempt_identity(model_attempt_identity)
        reservations: list[BudgetStepReservation] = []
        for context in recovery_contexts:
            limit = context.limit
            if limit is None:
                raise ValueError("Provider-operation recovery lost its original budget pricing.")
            record = await self._budget_ledger.load_reservation(context.reservation_id)
            if record is None:
                raise KeyError(f"Budget reservation not found: {context.reservation_id}")
            record = _validate_ledger_reservation_against_recovery_context(
                record,
                context=context,
                dispatch_id=dispatch_id,
            )
            if (
                record.budget_limit_id != context.budget_limit_id
                or record.model_step_id != model_attempt_identity.model_step_id
                or record.model_attempt_id != model_attempt_identity.model_attempt_id
                or record.session_id != session.id
                or record.agent_name != session.agent_name
                or record.environment_name != session.environment_name
                or record.provider_name != provider_name
                or record.model != session.model
                or record.dispatch_id != dispatch_id
                or record.scope != limit.scope
                or record.key != limit.key
                or record.window != limit.window
                or record.currency != limit.currency
            ):
                raise ValueError(
                    "Provider-operation budget reservation conflicts with its recovery context."
                )
            reservations.append(
                BudgetStepReservation(
                    # The durable identity is checked against the ledger above;
                    # only this frozen limit's pricing fields are consumed.
                    limit=cast("_EffectiveBudgetLimit", limit),
                    record=record,
                    request_billing_identity=copy_billing_identity(request_billing_identity),
                )
            )
        return tuple(reservations)

    async def _load_provider_operation_settlement(
        self,
        record: BudgetReservationRecord,
    ) -> BudgetSettlementRecord:
        raw = await self._budget_ledger.load_settlement(budget_settlement_id(record.reservation_id))
        settlement = None if raw is None else _validate_ledger_settlement_record(raw)
        if settlement is None or settlement.reservation_id != record.reservation_id:
            raise RuntimeError("Reconciled provider-operation reservation has no exact settlement.")
        return settlement

    async def _load_committed_settlement(
        self,
        reconciliation: BudgetReconciliation,
    ) -> BudgetSettlementRecord:
        raw_settlement = await self._budget_ledger.load_settlement(reconciliation.settlement_id)
        settlement = (
            None if raw_settlement is None else _validate_ledger_settlement_record(raw_settlement)
        )
        if (
            settlement is None
            or settlement.reconciliation != reconciliation
            or settlement.reservation_id != reconciliation.reservation_id
        ):
            raise RuntimeError("Budget ledger did not retain its exact committed settlement.")
        return settlement

    async def budget_settlement_event(
        self,
        reconciliation: BudgetReconciliation,
    ) -> Event:
        """Return the exact audit event material committed beside a settlement."""

        settlement = await self._load_committed_settlement(reconciliation)
        return settlement.event.model_copy(deep=True)

    async def acknowledge_budget_settlement_events(
        self,
        events: Collection[Event],
    ) -> None:
        """Mark already-persisted exact settlement events as published."""

        for event in events:
            if event.type not in {
                EventType.BUDGET_RECONCILED,
                EventType.BUDGET_RESERVATION_RELEASED,
            }:
                continue
            raw_settlement_id = event.payload.get("settlement_id")
            if type(raw_settlement_id) is not str:
                continue
            raw_settlement = await self._budget_ledger.load_settlement(raw_settlement_id)
            settlement = (
                None
                if raw_settlement is None
                else _validate_ledger_settlement_record(raw_settlement)
            )
            if settlement is None or settlement.event != event:
                raise RuntimeError("Persisted budget event conflicts with its ledger settlement.")
            acknowledged = _validate_ledger_settlement_record(
                await self._budget_ledger.mark_settlement_event_published(
                    settlement_id=settlement.settlement_id,
                    event_id=event.id,
                )
            )
            if not acknowledged.event_published or acknowledged.model_copy(
                update={"event_published": False}, deep=True
            ) != settlement.model_copy(update={"event_published": False}, deep=True):
                raise RuntimeError("Budget ledger acknowledged a conflicting settlement event.")

    async def model_step_events_with_heartbeat(
        self,
        model_step_events: AsyncIterator[tuple[Event | None, _StreamResultT | None]],
        *,
        reservations: list[BudgetStepReservation],
        lifecycle: BudgetModelStepLifecycle,
    ) -> AsyncIterator[tuple[Event | None, _StreamResultT | None]]:
        ttl_seconds = self._budget_ledger.reservation_ttl_seconds
        if not reservations or ttl_seconds is None:
            async for item in model_step_events:
                yield item
            return

        heartbeat_task = asyncio.create_task(
            self.heartbeat_reservations(
                reservations,
                reservation_transition_lock=lifecycle.reservation_transition_lock,
                interval_seconds=ttl_seconds / 3,
            )
        )
        iterator = model_step_events.__aiter__()
        next_item_task: asyncio.Task[tuple[Event | None, _StreamResultT | None]] | None = None
        exhausted = False
        try:
            while True:
                if heartbeat_task.done():
                    heartbeat_failure = budget_heartbeat_task_failure(heartbeat_task)
                    if not lifecycle.provider_dispatch_may_have_occurred:
                        raise BudgetReservationLeaseLostBeforeModelDispatch(
                            "Budget reservation lease was lost before model dispatch."
                        ) from heartbeat_failure
                    raise heartbeat_failure

                next_item_task = asyncio.create_task(_next_model_step_item(iterator))
                done, _ = await asyncio.wait(
                    {next_item_task, heartbeat_task},
                    return_when=asyncio.FIRST_COMPLETED,
                )
                if heartbeat_task in done:
                    heartbeat_failure = budget_heartbeat_task_failure(heartbeat_task)
                    completed_item: tuple[Event | None, _StreamResultT | None] | None = None
                    provider_failure: Exception | None = None
                    if not next_item_task.done():
                        next_item_task.cancel()
                        try:
                            await next_item_task
                        except asyncio.CancelledError as cancellation:
                            provider_failure = _safe_provider_stream_cleanup_failure(cancellation)
                        except Exception as exc:
                            provider_failure = exc
                    else:
                        try:
                            completed_item = next_item_task.result()
                        except StopAsyncIteration:
                            pass
                        except asyncio.CancelledError as cancellation:
                            provider_failure = _safe_provider_stream_cleanup_failure(cancellation)
                        except Exception as exc:
                            provider_failure = exc
                    if provider_failure is not None:
                        _record_provider_cleanup_failure(
                            heartbeat_failure,
                            provider_failure,
                        )
                    if completed_item is not None:
                        next_item_task = None
                        yield completed_item
                    if not lifecycle.provider_dispatch_may_have_occurred:
                        raise BudgetReservationLeaseLostBeforeModelDispatch(
                            "Budget reservation lease was lost before model dispatch."
                        ) from heartbeat_failure
                    raise heartbeat_failure
                try:
                    item = next_item_task.result()
                except StopAsyncIteration:
                    exhausted = True
                    break
                next_item_task = None
                yield item

            heartbeat_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await heartbeat_task
            async with lifecycle.reservation_transition_lock:
                await self.renew_reservations(reservations)
        finally:
            if next_item_task is not None and not next_item_task.done():
                next_item_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await next_item_task
            if not heartbeat_task.done():
                heartbeat_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await heartbeat_task
            if not exhausted:
                close = getattr(iterator, "aclose", None)
                if close is not None:
                    with contextlib.suppress(Exception):
                        await close()

    async def heartbeat_reservations(
        self,
        reservations: list[BudgetStepReservation],
        *,
        reservation_transition_lock: asyncio.Lock | None,
        interval_seconds: float,
    ) -> None:
        while True:
            await asyncio.sleep(interval_seconds)
            if reservation_transition_lock is None:
                await self.renew_reservations(reservations)
            else:
                async with reservation_transition_lock:
                    await self.renew_reservations(reservations)

    async def renew_reservations(
        self,
        reservations: list[BudgetStepReservation],
    ) -> None:
        for reservation in reservations:
            reservation_id = reservation.record.reservation_id
            try:
                renewed = await self._budget_ledger.heartbeat(
                    reservation_id=reservation_id,
                )
            except Exception as exc:
                raise BudgetReservationLeaseLost(
                    f"Could not renew budget reservation: {reservation_id}"
                ) from exc
            if not renewed:
                raise BudgetReservationLeaseLost(
                    f"Budget reservation lease was lost: {reservation_id}"
                )

    async def run_operation_with_reservation_heartbeat(
        self,
        operation: Callable[[], Awaitable[_OperationResultT]],
        *,
        reservations: list[BudgetStepReservation],
        authoritative_failure_types: tuple[type[BaseException], ...],
        lease_lost_before_dispatch_message: str,
        authoritative_failure_note: str,
        concurrent_failure_note: str,
        completed_metadata_from_result: (
            Callable[[_OperationResultT], dict[str, object] | None] | None
        ) = None,
    ) -> tuple[_OperationResultT, BaseException | None]:
        if not reservations:
            return await operation(), None
        ttl_seconds = self._budget_ledger.reservation_ttl_seconds
        if ttl_seconds is None:
            return await operation(), None
        try:
            await self.renew_reservations(reservations)
        except BudgetReservationLeaseLost as exc:
            raise BudgetReservationLeaseLostBeforeModelDispatch(
                lease_lost_before_dispatch_message
            ) from exc

        async def await_operation() -> _OperationResultT:
            return await operation()

        operation_task = asyncio.create_task(await_operation())
        heartbeat_task = asyncio.create_task(
            self.heartbeat_reservations(
                reservations,
                reservation_transition_lock=None,
                interval_seconds=ttl_seconds / 3,
            )
        )
        try:
            done, _ = await asyncio.wait(
                {operation_task, heartbeat_task},
                return_when=asyncio.FIRST_COMPLETED,
            )
            if heartbeat_task in done:
                failure = budget_heartbeat_task_failure(heartbeat_task)
                if operation_task.done():
                    try:
                        return operation_task.result(), failure
                    except BaseException as operation_failure:
                        if isinstance(operation_failure, authoritative_failure_types):
                            operation_failure.add_note(f"{authoritative_failure_note}: {failure}")
                            raise operation_failure from failure
                        _preserve_completed_metadata(operation_failure, failure)
                        failure.add_note(
                            f"{concurrent_failure_note}: "
                            f"{type(operation_failure).__name__}: {operation_failure}"
                        )
                        raise failure from operation_failure
                operation_task.cancel()
                try:
                    return await operation_task, failure
                except asyncio.CancelledError as operation_cancellation:
                    _preserve_completed_metadata(operation_cancellation, failure)
                except BaseException as operation_failure:
                    if isinstance(operation_failure, authoritative_failure_types):
                        operation_failure.add_note(f"{authoritative_failure_note}: {failure}")
                        raise operation_failure from failure
                    _preserve_completed_metadata(operation_failure, failure)
                    failure.add_note(
                        f"{concurrent_failure_note}: "
                        f"{type(operation_failure).__name__}: {operation_failure}"
                    )
                raise failure
            result = operation_task.result()
            heartbeat_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await heartbeat_task
            try:
                await self.renew_reservations(reservations)
            except BudgetReservationLeaseLost as exc:
                return result, exc
            return result, None
        except asyncio.CancelledError as exc:
            if not operation_task.done():
                operation_task.cancel()
            try:
                completed_result = await operation_task
            except asyncio.CancelledError as operation_cancellation:
                _preserve_completed_metadata(operation_cancellation, exc)
            except BaseException as operation_failure:
                _preserve_completed_metadata(operation_failure, exc)
                exc.add_note(
                    "Budgeted operation also failed while caller cancellation was handled: "
                    f"{type(operation_failure).__name__}: {operation_failure}"
                )
            else:
                try:
                    completed_metadata = (
                        None
                        if completed_metadata_from_result is None
                        else completed_metadata_from_result(completed_result)
                    )
                    if completed_metadata is not None:
                        if type(completed_metadata) is not dict:
                            raise TypeError(
                                "completed_metadata_from_result must return a dictionary or None."
                            )
                        exc.__dict__["completed_metadata"] = copy_json_value(
                            completed_metadata,
                            "completed_metadata",
                        )
                except BaseException as evidence_failure:
                    exc.add_note(
                        "Completed operation evidence could not be preserved during "
                        f"caller cancellation: {type(evidence_failure).__name__}: "
                        f"{evidence_failure}"
                    )
            raise
        finally:
            if not operation_task.done():
                operation_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await operation_task
            if not heartbeat_task.done():
                heartbeat_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await heartbeat_task

    async def run_automatic_compaction_dispatch(
        self,
        operation: Callable[[], Awaitable[_OperationResultT]],
        *,
        completed_events: Callable[[], list[Event]],
        prior_completion_events: list[Event] | None = None,
        budget_limits: tuple[BudgetLimit, ...],
        session: Session,
        agent_name: str,
        environment_name: str | None,
        provider_name: str,
        model: str,
        model_attempt_identity: ModelAttemptIdentity,
        authoritative_failure_types: tuple[type[BaseException], ...],
        billing_identity: BillingIdentity | None = None,
        pricing_provider_name: str | None = None,
        execution_profile_fingerprint: str | None = None,
        reservation_identity_guard: BudgetReservationIdentityGuard | None = None,
        before_provider_dispatch: Callable[[], Awaitable[None]] | None = None,
    ) -> (
        BudgetedOperationSucceeded[_OperationResultT]
        | BudgetedOperationRejected
        | BudgetedOperationFailed
    ):
        """Run one observable compactor dispatch under strict budget accounting."""

        model_attempt_identity = copy_model_attempt_identity(model_attempt_identity)
        budget_limits = _operation_budget_limits_for_session(
            limits=budget_limits,
            agent_name=agent_name,
            causal_budget_id=session.causal_budget_id,
        )
        lifecycle = _BudgetedOperationLifecycle()
        identity_guard = reservation_identity_guard or self.reservation_identity_guard()
        prior_completion_events = (
            []
            if prior_completion_events is None
            else [event.model_copy(deep=True) for event in prior_completion_events]
        )
        effective_pricing_provider_name = pricing_provider_name or provider_name
        dispatch_preflight_limits = tuple(
            limit for limit in budget_limits if limit.action == "interrupt"
        )
        if dispatch_preflight_limits:
            resolved_checks = await self.evaluate_operation_budgets(
                session=session,
                budget_limits=dispatch_preflight_limits,
                operation_events=prior_completion_events,
                provider_name=effective_pricing_provider_name,
                model=model,
                billing_identity_state=resolved_billing_identity(billing_identity),
            )
            for resolved in resolved_checks:
                if not resolved.check.limit_reached:
                    continue
                failure = BudgetReservationResult(
                    accepted=False,
                    budget_limit_id=resolved.check.budget_limit_id,
                    model_step_id=model_attempt_identity.model_step_id,
                    model_attempt_id=model_attempt_identity.model_attempt_id,
                    scope=resolved.limit.scope,
                    key=resolved.limit.key,
                    window=resolved.limit.window,
                    currency=resolved.limit.currency,
                    maximum=resolved.limit.max_estimated_cost,
                    action=resolved.limit.action,
                    requested=Decimal("0"),
                    actual=resolved.check.actual,
                    message=resolved.check.message,
                )
                event = await self._event_writer.emit(
                    _event_with_budget_authority(
                        Event(
                            type=EventType.BUDGET_RESERVATION_FAILED,
                            session_id=session.id,
                            agent_name=agent_name,
                            environment_name=environment_name,
                            payload=budget_reservation_payload(failure),
                        ),
                        execution_identity=model_attempt_identity,
                        execution_profile_fingerprint=execution_profile_fingerprint,
                        additional_fields=("budget_limit_id",),
                    )
                )
                return BudgetedOperationRejected(failure=failure, events=(event,))
        result: _OperationResultT | None = None
        reservation_failure: BudgetReservationResult | None = None
        authoritative_failure: BaseException | None = None
        authoritative_cause: BaseException | None = None
        lease_failure: BaseException | None = None

        try:
            setup = await self.reserve_operation_budgets(
                budget_limits=budget_limits,
                session_id=session.id,
                agent_name=agent_name,
                provider_name=provider_name,
                model=model,
                model_attempt_identity=model_attempt_identity,
                environment_name=environment_name,
                billing_identity=billing_identity,
                execution_profile_fingerprint=execution_profile_fingerprint,
                reservation_identity_guard=identity_guard,
                rejection_release_reason="reservation failed",
                accepted_record_error=(
                    "Accepted automatic compaction budget reservation has no record."
                ),
                reservation_event_factory=lambda reservation_result: Event(
                    type=(
                        EventType.BUDGET_RESERVED
                        if reservation_result.accepted
                        else EventType.BUDGET_RESERVATION_FAILED
                    ),
                    session_id=session.id,
                    agent_name=agent_name,
                    environment_name=environment_name,
                    payload=budget_reservation_payload(reservation_result),
                ),
            )
            lifecycle.reservations.extend(setup.reservations)
            for reservation_event in setup.events:
                lifecycle.events.append(await self._event_writer.emit(reservation_event))
            for reconciliation in setup.releases:
                settlement = await self._load_committed_settlement(reconciliation)
                lifecycle.events.append(await self.publish_budget_settlement(settlement))
            if setup.error is not None:
                raise setup.error
            if setup.failure is not None:
                reservation_failure = setup.failure
            else:

                async def run_dispatched_operation() -> _OperationResultT:
                    if before_provider_dispatch is not None:
                        await before_provider_dispatch()
                    deferred_dispatch_failure = await self.mark_reservations_dispatched(
                        lifecycle.reservations,
                        dispatch_id=model_attempt_identity.model_attempt_id,
                    )
                    lifecycle.provider_dispatch_started = True
                    if deferred_dispatch_failure is not None:
                        raise deferred_dispatch_failure
                    return await operation()

                result, lease_failure = await self.run_operation_with_reservation_heartbeat(
                    run_dispatched_operation,
                    reservations=lifecycle.reservations,
                    authoritative_failure_types=authoritative_failure_types,
                    lease_lost_before_dispatch_message=(
                        "Compaction budget reservation lease was lost before provider dispatch."
                    ),
                    authoritative_failure_note=(
                        "Budget reservation lease was also lost as compaction failed"
                    ),
                    concurrent_failure_note=(
                        "Compactor also failed while reservation lease loss was handled"
                    ),
                )
        except BaseException as exc:
            authoritative_failure = exc
            authoritative_cause = exception_cause(exc)
            if not lifecycle.provider_dispatch_started:
                lifecycle.predispatch_release_reason = (
                    "reservation setup cancelled"
                    if isinstance(exc, asyncio.CancelledError)
                    else "reservation setup failed"
                )

        completion_events: list[Event] = []
        if lifecycle.provider_dispatch_started:
            try:
                completion_events = completed_events()
            except BaseException as evidence_failure:
                if authoritative_failure is None:
                    authoritative_failure = evidence_failure
                    authoritative_cause = exception_cause(evidence_failure)
                else:
                    authoritative_failure.add_note(
                        "Automatic compaction completion evidence also failed: "
                        f"{type(evidence_failure).__name__}: {evidence_failure}"
                    )

        (
            settlement_cancellation,
            settlement_failure,
        ) = await self._settle_budgeted_operation_resisting_cancellation(
            lifecycle=lifecycle,
            completed_events=completion_events,
            session=session,
            agent_name=agent_name,
            environment_name=environment_name,
        )

        if settlement_cancellation is not None:
            propagated_cancellation = (
                authoritative_failure
                if isinstance(authoritative_failure, asyncio.CancelledError)
                else settlement_cancellation
            )
            cancellation_cause: BaseException | None = None
            if settlement_failure is not None:
                propagated_cancellation.add_note(
                    "Automatic compaction budget settlement also failed: "
                    f"{type(settlement_failure).__name__}: {settlement_failure}"
                )
                cancellation_cause = settlement_failure
            elif (
                authoritative_failure is not None
                and authoritative_failure is not propagated_cancellation
            ):
                propagated_cancellation.add_note(
                    "Automatic compaction had already failed before cancellation: "
                    f"{type(authoritative_failure).__name__}: {authoritative_failure}"
                )
                cancellation_cause = authoritative_failure
            return BudgetedOperationFailed(
                error=propagated_cancellation,
                cause=cancellation_cause,
                events=tuple(lifecycle.events),
            )

        if settlement_failure is not None:
            if authoritative_failure is not None:
                authoritative_failure.__dict__["_cayu_compaction_budget_settlement_failed"] = True
                if isinstance(authoritative_failure, ModelProviderError):
                    authoritative_failure.retryable = False
                authoritative_failure.add_note(
                    "Automatic compaction budget settlement also failed: "
                    f"{type(settlement_failure).__name__}: {settlement_failure}"
                )
                return BudgetedOperationFailed(
                    error=authoritative_failure,
                    cause=settlement_failure,
                    events=tuple(lifecycle.events),
                )
            return BudgetedOperationFailed(
                error=settlement_failure,
                cause=None,
                events=tuple(lifecycle.events),
            )

        if authoritative_failure is not None:
            return BudgetedOperationFailed(
                error=authoritative_failure,
                cause=authoritative_cause,
                events=tuple(lifecycle.events),
            )
        if lease_failure is not None:
            return BudgetedOperationFailed(
                error=lease_failure,
                cause=exception_cause(lease_failure),
                events=tuple(lifecycle.events),
            )
        if reservation_failure is not None:
            return BudgetedOperationRejected(
                failure=reservation_failure,
                events=tuple(lifecycle.events),
            )
        if result is None:
            return BudgetedOperationFailed(
                error=RuntimeError("Automatic compaction completed without a result."),
                cause=None,
                events=tuple(lifecycle.events),
            )
        return BudgetedOperationSucceeded(result=result, events=tuple(lifecycle.events))

    async def _settle_budgeted_operation_resisting_cancellation(
        self,
        *,
        lifecycle: _BudgetedOperationLifecycle,
        completed_events: list[Event],
        session: Session,
        agent_name: str,
        environment_name: str | None,
    ) -> tuple[asyncio.CancelledError | None, BaseException | None]:
        settlement_task = asyncio.create_task(
            self._settle_budgeted_operation(
                lifecycle=lifecycle,
                completed_events=completed_events,
                session=session,
                agent_name=agent_name,
                environment_name=environment_name,
            )
        )
        cancellation: asyncio.CancelledError | None = None
        settlement_failure: BaseException | None = None
        while not settlement_task.done():
            try:
                await asyncio.shield(settlement_task)
            except asyncio.CancelledError as exc:
                if settlement_task.cancelled():
                    settlement_failure = exc
                    break
                if cancellation is None:
                    cancellation = exc
            except BaseException as exc:
                settlement_failure = exc
                break
        if settlement_failure is None:
            try:
                settlement_task.result()
            except BaseException as exc:
                settlement_failure = exc
        return cancellation, settlement_failure

    async def _settle_budgeted_operation(
        self,
        *,
        lifecycle: _BudgetedOperationLifecycle,
        completed_events: list[Event],
        session: Session,
        agent_name: str,
        environment_name: str | None,
    ) -> None:
        if lifecycle.settled:
            return
        if not lifecycle.reservations:
            lifecycle.settled = True
            return
        settlement_failures: list[tuple[str, Exception]] = []

        def raise_settlement_failure() -> None:
            if not settlement_failures:
                return
            first_reservation_id, first_failure = settlement_failures[0]
            first_failure.add_note(
                "Automatic compaction budget settlement failed for reservation "
                f"{first_reservation_id}."
            )
            for reservation_id, failure in settlement_failures[1:]:
                first_failure.add_note(
                    "Additional automatic compaction budget settlement failure for "
                    f"reservation {reservation_id}: {type(failure).__name__}: {failure}"
                )
            raise first_failure

        if not lifecycle.provider_dispatch_started:
            for reservation in list(lifecycle.reservations):
                try:
                    working_reservations = [reservation]
                    async for reconciliation in self.release_operation_reservations(
                        working_reservations,
                        reason=lifecycle.predispatch_release_reason,
                    ):
                        settlement = await self._load_committed_settlement(reconciliation)
                        lifecycle.events.append(await self.publish_budget_settlement(settlement))
                except Exception as exc:
                    settlement_failures.append((reservation.record.reservation_id, exc))
            lifecycle.settled = True
            raise_settlement_failure()
            return

        for reservation in list(lifecycle.reservations):
            try:
                priced_actuals = []
                uncertain_completion_count = 0
                for event in completed_events:
                    try:
                        priced_actuals.append(
                            budget_actual_cost_for_event(limit=reservation.limit, event=event)
                        )
                    except ValueError:
                        uncertain_completion_count += 1
                if not completed_events:
                    actual_amount = reservation.record.reserved_amount
                    reason = (
                        "automatic context compaction dispatch has uncertain usage; "
                        "charged reserved amount"
                    )
                else:
                    actual_amount = sum(
                        (priced.amount for priced in priced_actuals),
                        start=(reservation.record.reserved_amount * uncertain_completion_count),
                    )
                    if uncertain_completion_count:
                        reason = (
                            "automatic context compaction completed with partially uncertain "
                            "usage; charged known cost plus reserved amount per uncertain "
                            "completion"
                        )
                    else:
                        reason = "automatic context compaction model completed"
                completed_billing_identity = _single_completion_billing_identity(completed_events)
                if len(completed_events) == 1 and len(priced_actuals) == 1:
                    completed_billing_identity = (
                        priced_actuals[0].line_item.billing_identity or completed_billing_identity
                    )
                occurred_at = completed_events[-1].timestamp if completed_events else self._clock()
                settlement_kind: Literal["completed", "conservative"] = (
                    "completed"
                    if completed_events and not uncertain_completion_count
                    else "conservative"
                )
                pricing = (
                    budget_reconciliation_pricing(priced_actuals[0].line_item)
                    if len(completed_events) == 1 and len(priced_actuals) == 1
                    else None
                )
                expected = budget_reconciliation_preview(
                    _accounting_reservation_record(reservation),
                    actual_amount=actual_amount,
                    settlement_kind=settlement_kind,
                    reason=reason,
                    occurred_at=occurred_at,
                    billing_identity=completed_billing_identity,
                    pricing=pricing,
                )
                reconciliation = await self._commit_expected_reconciliation(
                    reservation,
                    expected,
                )
                settlement = await self._load_committed_settlement(reconciliation)
                lifecycle.events.append(await self.publish_budget_settlement(settlement))
            except Exception as exc:
                settlement_failures.append((reservation.record.reservation_id, exc))
        lifecycle.settled = True
        raise_settlement_failure()

    async def reserve_operation_budgets(
        self,
        *,
        budget_limits: tuple[BudgetLimit, ...],
        session_id: str,
        agent_name: str,
        provider_name: str | None,
        model: str | None,
        model_attempt_identity: ModelAttemptIdentity,
        environment_name: str | None = None,
        settlement_event_payload: dict[str, object] | None = None,
        execution_profile_fingerprint: str | None = None,
        rejection_release_reason: str,
        accepted_record_error: str,
        reservation_event_factory: Callable[[BudgetReservationResult], Event] | None = None,
        billing_identity: BillingIdentity | None = None,
        reservation_identity_guard: BudgetReservationIdentityGuard | None = None,
    ) -> OperationReservationSetup:
        model_attempt_identity = copy_model_attempt_identity(model_attempt_identity)
        effective_limits = _effective_budget_limits(
            budget_limits,
            identity_namespace="operation",
            preserve_effective=True,
        )
        limits = [limit for limit in effective_limits if limit.reservation is not None]
        if not limits or provider_name is None or model is None:
            return OperationReservationSetup((), (), (), (), None, None)

        reservations: list[BudgetStepReservation] = []
        reservation_ids: set[str] = set()
        results: list[BudgetReservationResult] = []
        events: list[Event] = []
        releases: list[BudgetReconciliation] = []
        expected_billing_identity = copy_billing_identity(billing_identity)
        profile_settlement_payload = copy_durable_json_object(
            settlement_event_payload or {},
            "settlement_event_payload",
        )
        if execution_profile_fingerprint is not None:
            existing_fingerprint = profile_settlement_payload.get("execution_profile_fingerprint")
            if existing_fingerprint not in {None, execution_profile_fingerprint}:
                raise ValueError("Settlement event payload conflicts with its execution profile.")
            profile_settlement_payload["execution_profile_fingerprint"] = (
                execution_profile_fingerprint
            )
        expected_settlement_event_payload = _interaction_bound_settlement_event_payload(
            self._event_writer,
            session_id=session_id,
            agent_name=agent_name,
            environment_name=environment_name,
            payload=profile_settlement_payload,
        )
        identity_guard = reservation_identity_guard or self.reservation_identity_guard()
        if reservation_event_factory is None:

            def default_reservation_event_factory(
                result: BudgetReservationResult,
            ) -> Event:
                return Event(
                    type=(
                        EventType.BUDGET_RESERVED
                        if result.accepted
                        else EventType.BUDGET_RESERVATION_FAILED
                    ),
                    session_id=session_id,
                    agent_name=agent_name,
                    payload=budget_reservation_payload(result),
                )

            event_factory = default_reservation_event_factory
        else:
            event_factory = reservation_event_factory
        for limit in limits:
            try:
                expected_limit = _copy_effective_budget_limit(limit)
                ledger_limit = _copy_effective_budget_limit(limit)
                ledger_model_attempt_identity = copy_model_attempt_identity(model_attempt_identity)
                ledger_settlement_event_payload = copy_durable_json_object(
                    expected_settlement_event_payload,
                    "settlement_event_payload",
                )
                reservation_effective_at = self._clock()
                expected_requested_amount = _budget_reservation_amount(
                    limit=expected_limit,
                    provider_name=provider_name,
                    model=model,
                    effective_at=reservation_effective_at,
                    billing_identity=expected_billing_identity,
                )
                authority = _new_publishable_budget_reservation_authority(
                    self._event_writer,
                    limit=expected_limit,
                    model_attempt_identity=model_attempt_identity,
                    session_id=session_id,
                    agent_name=agent_name,
                    environment_name=environment_name,
                    provider_name=provider_name,
                    model=model,
                    settlement_event_payload=expected_settlement_event_payload,
                    billing_identity=expected_billing_identity,
                    reserved_amount=expected_requested_amount,
                    fallback_settled_at=reservation_effective_at,
                    reservation_ttl_seconds=self.reservation_ttl_seconds,
                )
                ledger_billing_identity = copy_billing_identity(authority.billing_identity)
                try:
                    if expected_billing_identity is None and ledger_billing_identity is None:
                        result = await self._budget_ledger.reserve(
                            reservation_id=authority.reservation_id,
                            limit=ledger_limit,
                            session_id=session_id,
                            agent_name=agent_name,
                            provider_name=provider_name,
                            model=model,
                            model_attempt_identity=ledger_model_attempt_identity,
                            environment_name=environment_name,
                            settlement_event_payload=ledger_settlement_event_payload,
                            settlement_fallback=authority.settlement_fallback,
                            effective_at=reservation_effective_at,
                        )
                    else:
                        result = await self._budget_ledger.reserve(
                            reservation_id=authority.reservation_id,
                            limit=ledger_limit,
                            session_id=session_id,
                            agent_name=agent_name,
                            provider_name=provider_name,
                            model=model,
                            model_attempt_identity=ledger_model_attempt_identity,
                            environment_name=environment_name,
                            settlement_event_payload=ledger_settlement_event_payload,
                            settlement_fallback=authority.settlement_fallback,
                            requested_amount=expected_requested_amount,
                            billing_identity=ledger_billing_identity,
                            effective_at=reservation_effective_at,
                        )
                finally:
                    _validate_ledger_limit_unchanged(
                        ledger_limit,
                        expected=expected_limit,
                    )
                result = _validate_ledger_reservation_result(
                    result,
                    expected_reservation_id=authority.reservation_id,
                    limit=expected_limit,
                    model_attempt_identity=model_attempt_identity,
                    session_id=session_id,
                    agent_name=agent_name,
                    provider_name=provider_name,
                    model=model,
                    environment_name=environment_name,
                    settlement_event_payload=expected_settlement_event_payload,
                    settlement_fallback=authority.settlement_fallback,
                    billing_identity=authority.billing_identity,
                    expected_requested_amount=expected_requested_amount,
                )
                reservation: BudgetStepReservation | None = None
                if result.accepted:
                    accepted_record = result.record
                    if accepted_record is None:  # pragma: no cover - validated above
                        raise RuntimeError(accepted_record_error)
                    if accepted_record.reservation_id in reservation_ids:
                        raise RuntimeError("Budget ledger reused a reservation identity.")
                    reservation = BudgetStepReservation(
                        limit=expected_limit,
                        record=accepted_record,
                        request_billing_identity=copy_billing_identity(expected_billing_identity),
                    )
                    # An accepted, validated record must remain reachable by
                    # caller-owned cleanup even if custom event construction or
                    # authority attestation fails before publication.
                    reservations.append(reservation)
                reservation_event = event_factory(result)
                if type(reservation_event) is not Event:
                    raise TypeError("Reservation event factory must return an Event.")
                expected_event_type = (
                    EventType.BUDGET_RESERVED
                    if result.accepted
                    else EventType.BUDGET_RESERVATION_FAILED
                )
                if (
                    reservation_event.type != expected_event_type
                    or reservation_event.session_id != session_id
                ):
                    raise ValueError(
                        "Reservation event factory returned an event for a different result."
                    )
                expected_payload = budget_reservation_payload(result)
                if any(
                    key not in reservation_event.payload or reservation_event.payload[key] != value
                    for key, value in expected_payload.items()
                ):
                    raise ValueError(
                        "Reservation event factory changed runtime-owned budget evidence."
                    )
                reservation_event = _event_with_budget_authority(
                    reservation_event,
                    execution_identity=model_attempt_identity,
                    execution_profile_fingerprint=execution_profile_fingerprint,
                    additional_fields=(
                        "budget_limit_id",
                        "reservation_id",
                        "session_id",
                    ),
                )
                if not result.accepted:
                    results.append(result)
                    events.append(reservation_event)
                    await self.recover_pending_budget_settlements()
                    async for reconciliation in self.release_operation_reservations(
                        reservations,
                        reason=rejection_release_reason,
                    ):
                        releases.append(reconciliation)
                    return OperationReservationSetup(
                        reservations=(),
                        results=tuple(results),
                        events=tuple(events),
                        releases=tuple(releases),
                        failure=result,
                        error=None,
                    )
                assert reservation is not None
                try:
                    await identity_guard.claim(
                        accepted_record.reservation_id,
                        publication_session_id=reservation_event.session_id,
                        publication_id=reservation_event.id,
                    )
                except (BudgetReservationIdentityConflict, SessionRunFenced):
                    _remove_reservation(reservations, reservation)
                    raise
                reservation_ids.add(accepted_record.reservation_id)
                results.append(result)
                events.append(reservation_event)
                await self.recover_pending_budget_settlements()
            except BaseException as exc:
                return OperationReservationSetup(
                    reservations=tuple(reservations),
                    results=tuple(results),
                    events=tuple(events),
                    releases=tuple(releases),
                    failure=(results[-1] if results and not results[-1].accepted else None),
                    error=exc,
                )
        return OperationReservationSetup(
            reservations=tuple(reservations),
            results=tuple(results),
            events=tuple(events),
            releases=(),
            failure=None,
            error=None,
        )

    async def reconcile_operation_reservations(
        self,
        reservations: list[BudgetStepReservation],
        *,
        model_completed_events: list[Event],
        completed_reason: str,
        missing_usage_reason: str,
    ) -> AsyncIterator[BudgetReconciliation]:
        for reservation in list(reservations):
            reconciliation = await self._reconcile_operation_reservation(
                reservation,
                model_completed_events=model_completed_events,
                completed_reason=completed_reason,
                missing_usage_reason=missing_usage_reason,
            )
            _remove_reservation(reservations, reservation)
            yield reconciliation

    async def _reconcile_operation_reservation(
        self,
        reservation: BudgetStepReservation,
        *,
        model_completed_events: list[Event],
        completed_reason: str,
        missing_usage_reason: str,
    ) -> BudgetReconciliation:
        priced_actuals = []
        try:
            priced_actuals = [
                budget_actual_cost_for_event(limit=reservation.limit, event=event)
                for event in model_completed_events
            ]
            if not priced_actuals:
                raise ValueError("Operation completed without model usage.")
            actual_amount = sum(
                (priced.amount for priced in priced_actuals),
                start=Decimal("0"),
            )
            reason = completed_reason
        except ValueError:
            actual_amount = reservation.record.reserved_amount
            reason = missing_usage_reason
        completed_billing_identity = _single_completion_billing_identity(model_completed_events)
        if len(model_completed_events) == 1 and len(priced_actuals) == 1:
            completed_billing_identity = (
                priced_actuals[0].line_item.billing_identity or completed_billing_identity
            )
        if reservation.record.billing_identity is None:
            # A context-free price may reserve before the provider's optional
            # billing hook runs. Completion cannot introduce an identity that
            # was absent from the request-time reservation.
            completed_billing_identity = None
        occurred_at = (
            model_completed_events[-1].timestamp if model_completed_events else self._clock()
        )
        settlement_kind: Literal["completed", "conservative"] = (
            "completed" if priced_actuals else "conservative"
        )
        pricing = (
            budget_reconciliation_pricing(priced_actuals[0].line_item)
            if len(priced_actuals) == 1
            else None
        )
        expected = budget_reconciliation_preview(
            _accounting_reservation_record(reservation),
            actual_amount=actual_amount,
            settlement_kind=settlement_kind,
            reason=reason,
            occurred_at=occurred_at,
            billing_identity=completed_billing_identity,
            pricing=pricing,
        )
        reconciliation = await self._commit_expected_reconciliation(
            reservation,
            expected,
        )
        return reconciliation

    async def reconcile_uncertain_operation_reservations(
        self,
        reservations: list[BudgetStepReservation],
        *,
        reason: str,
    ) -> AsyncIterator[BudgetReconciliation]:
        for reservation in list(reservations):
            expected = budget_reconciliation_preview(
                reservation.record,
                actual_amount=reservation.record.reserved_amount,
                settlement_kind="conservative",
                reason=reason,
                occurred_at=self._clock(),
            )
            reconciliation = await self._commit_expected_reconciliation(
                reservation,
                expected,
            )
            _remove_reservation(reservations, reservation)
            yield reconciliation

    async def release_operation_reservations(
        self,
        reservations: list[BudgetStepReservation],
        *,
        reason: str,
    ) -> AsyncIterator[BudgetReconciliation]:
        for reservation in list(reservations):
            expected = budget_release_preview(
                reservation.record,
                reason=reason,
                occurred_at=self._clock(),
            )
            reconciliation = await self._commit_expected_release(
                reservation,
                expected,
            )
            _remove_reservation(reservations, reservation)
            yield reconciliation

    async def release_reservations(
        self,
        reservations: list[BudgetStepReservation],
        *,
        session: Session,
        agent_name: str,
        environment_name: str | None,
        reason: str,
    ) -> AsyncIterator[Event]:
        for reservation in list(reservations):
            expected = budget_release_preview(
                reservation.record,
                reason=reason,
                occurred_at=self._clock(),
            )
            reconciliation = await self._commit_expected_release(
                reservation,
                expected,
            )
            settlement = await self._load_committed_settlement(reconciliation)
            if (
                settlement.session_id != session.id
                or settlement.agent_name != agent_name
                or settlement.environment_name != environment_name
            ):
                raise RuntimeError("Budget release audit context conflicts with its reservation.")
            event = await self.publish_budget_settlement(settlement)
            _remove_reservation(reservations, reservation)
            yield event


class RunLimitGate:
    """Retain one run's limit inputs and incremental usage watermark."""

    def __init__(
        self,
        controller: RunLimitController,
        *,
        session: Session,
        agent_name: str,
        environment_name: str | None,
        limits: RunLimits,
        budget_limits: tuple[BudgetLimit, ...],
        run_started_at: float,
        run_baseline: SessionUsageSummary | None,
        budget_baseline_events: list[Event],
        budget_notify_events: list[Event],
        run_budget_authorities: Mapping[str, RunBudgetAccountingAuthority] | None = None,
        pricing_provider_name: str | None = None,
        execution_profile_fingerprint: str | None = None,
    ) -> None:
        self._controller = controller
        self._session = session
        self._agent_name = agent_name
        self._environment_name = environment_name
        self._limits = limits
        self._budget_limits = budget_limits
        self._run_started_at = run_started_at
        self._run_baseline = run_baseline
        self._budget_baseline_events = budget_baseline_events
        self._run_budget_authorities = run_budget_authorities
        self._budget_notify_events = budget_notify_events
        self._pricing_provider_name = pricing_provider_name
        self._execution_profile_fingerprint = execution_profile_fingerprint
        self._usage_tracker = controller.usage_tracker(session.id)

    async def evaluate_limits(
        self,
        *,
        pending_tool_calls: int = 0,
        billing_identity_state: BillingIdentityState = UNRESOLVED_BILLING_IDENTITY,
        pricing_provider_name: str | None = None,
        model: str | None = None,
        additional_usage_events: list[Event] | None = None,
        budget_limits: tuple[BudgetLimit, ...] | None = None,
        execution_identity: ModelStepIdentity | ModelAttemptIdentity | None = None,
    ) -> LimitEvaluation:
        return await self._controller.evaluate_request_limits(
            session=self._session,
            agent_name=self._agent_name,
            environment_name=self._environment_name,
            limits=self._limits,
            budget_limits=self._budget_limits if budget_limits is None else budget_limits,
            run_started_at=self._run_started_at,
            run_baseline=self._run_baseline,
            budget_baseline_events=self._budget_baseline_events,
            run_budget_authorities=self._run_budget_authorities,
            pending_tool_calls=pending_tool_calls,
            budget_notify_events=self._budget_notify_events,
            usage_tracker=self._usage_tracker,
            billing_identity_state=billing_identity_state,
            pricing_provider_name=pricing_provider_name or self._pricing_provider_name,
            model=model,
            additional_usage_events=additional_usage_events,
            execution_identity=execution_identity,
            execution_profile_fingerprint=self._execution_profile_fingerprint,
        )

    async def evaluate_budget(
        self,
        budget_policy: BudgetPolicy | None,
        *,
        billing_identity_state: BillingIdentityState = UNRESOLVED_BILLING_IDENTITY,
        pricing_provider_name: str | None = None,
        model: str | None = None,
        additional_usage_events: list[Event] | None = None,
        execution_identity: ModelStepIdentity | ModelAttemptIdentity | None = None,
    ) -> BudgetEvaluation:
        return await self._controller.evaluate_policy_budgets(
            session=self._session,
            agent_name=self._agent_name,
            environment_name=self._environment_name,
            budget_policy=budget_policy,
            billing_identity_state=billing_identity_state,
            pricing_provider_name=pricing_provider_name or self._pricing_provider_name,
            model=model,
            additional_usage_events=additional_usage_events,
            execution_identity=execution_identity,
            execution_profile_fingerprint=self._execution_profile_fingerprint,
        )

    def has_run_limits(self) -> bool:
        """Return whether provider dispatches need per-call run-limit admission."""

        return has_run_limits(self._limits)


def _latest_model_event_identity(events: list[Event]) -> tuple[str | None, str | None]:
    for event in reversed(events):
        if event.type != EventType.MODEL_COMPLETED:
            continue
        provider_name = event.payload.get("provider_name")
        model = event.payload.get("model") or event.payload.get("requested_model")
        return (
            provider_name if type(provider_name) is str else None,
            model if type(model) is str else None,
        )
    return None, None


def _single_completion_billing_identity(events: list[Event]) -> BillingIdentity | None:
    if len(events) != 1:
        return None
    raw_identity = events[0].payload.get("billing_identity")
    return BillingIdentity.model_validate(raw_identity) if type(raw_identity) is dict else None


def _remove_reservation(
    reservations: list[BudgetStepReservation],
    settled: BudgetStepReservation,
) -> None:
    reservation_id = settled.record.reservation_id
    reservations[:] = [
        reservation
        for reservation in reservations
        if reservation.record.reservation_id != reservation_id
    ]


def budget_limit_reached_payload(check: BudgetCheck) -> dict[str, object]:
    if type(check) is not BudgetCheck:
        raise TypeError("check must be a BudgetCheck.")
    return budget_check_payload(check)


def _budget_limit_reached_payload_matches(
    payload: dict[str, object],
    *,
    check: BudgetCheck,
) -> bool:
    if not isinstance(payload, dict):
        return False
    if type(check) is not BudgetCheck:
        raise TypeError("check must be a BudgetCheck.")
    return payload.get("budget_limit_id") == check.budget_limit_id


def _budget_notify_already_emitted_in_invocation(
    events: list[Event],
    *,
    check: BudgetCheck,
) -> bool:
    if type(check) is not BudgetCheck:
        raise TypeError("check must be a BudgetCheck.")
    return any(
        event.type == EventType.BUDGET_LIMIT_REACHED
        and _budget_limit_reached_payload_matches(event.payload, check=check)
        for event in events
    )


def _first_budget_limit_outcome(
    *,
    session: Session,
    limit: BudgetLimit,
    cost_summary: SessionCostSummary,
    cost_baseline: SessionCostSummary | None,
    effective_at: datetime,
    billing_identity_state: BillingIdentityState = UNRESOLVED_BILLING_IDENTITY,
    pricing_provider_name: str | None = None,
    model: str | None = None,
) -> _BudgetLimitOutcome | None:
    if type(session) is not Session:
        raise TypeError("session must be a Session instance.")
    if type(limit) is not _EffectiveBudgetLimit:
        raise TypeError("limit must be a BudgetLimit instance.")
    if type(cost_summary) is not SessionCostSummary:
        raise TypeError("cost_summary must be a SessionCostSummary.")
    if cost_baseline is not None and type(cost_baseline) is not SessionCostSummary:
        raise TypeError("cost_baseline must be a SessionCostSummary.")

    actual_cost = cost_summary.total_cost
    unpriced_model_steps = cost_summary.unpriced_model_steps
    if limit.scope == "run" and cost_baseline is not None:
        actual_cost = max(
            cost_summary.total_cost - cost_baseline.total_cost,
            Decimal("0"),
        )
        unpriced_model_steps = max(
            unpriced_model_steps - cost_baseline.unpriced_model_steps,
            0,
        )

    if unpriced_model_steps > 0 and not limit.allow_unpriced:
        decision = StopDecision(
            limit=StopLimit.ESTIMATED_COST,
            maximum=limit.max_estimated_cost,
            actual=actual_cost,
            message=(
                "Estimated cost budget cannot be verified because "
                f"{unpriced_model_steps} model step(s) have no matching pricing."
            ),
        )
        return _BudgetLimitOutcome(
            decision=decision,
            check=_budget_check_from_stop_decision(
                limit=limit,
                decision=decision,
                cost_summary=cost_summary,
                unpriced_model_steps=unpriced_model_steps,
            ),
        )
    preflight_error = _budget_limit_preflight_error(
        session=session,
        limit=limit,
        effective_at=effective_at,
        billing_identity_state=billing_identity_state,
        pricing_provider_name=pricing_provider_name,
        model=model,
    )
    if preflight_error is not None:
        decision = StopDecision(
            limit=StopLimit.ESTIMATED_COST,
            maximum=limit.max_estimated_cost,
            actual=actual_cost,
            message=preflight_error,
        )
        return _BudgetLimitOutcome(
            decision=decision,
            check=_budget_check_from_stop_decision(
                limit=limit,
                decision=decision,
                cost_summary=cost_summary,
                unpriced_model_steps=unpriced_model_steps,
            ),
        )
    if actual_cost >= limit.max_estimated_cost:
        decision = StopDecision(
            limit=StopLimit.ESTIMATED_COST,
            maximum=limit.max_estimated_cost,
            actual=actual_cost,
            message=(
                "Estimated cost budget reached: "
                f"{actual_cost} >= {limit.max_estimated_cost} {limit.currency}."
            ),
        )
        return _BudgetLimitOutcome(
            decision=decision,
            check=_budget_check_from_stop_decision(
                limit=limit,
                decision=decision,
                cost_summary=cost_summary,
                unpriced_model_steps=unpriced_model_steps,
            ),
        )
    return None


def _budget_check_from_stop_decision(
    *,
    limit: _EffectiveBudgetLimit,
    decision: StopDecision,
    cost_summary: SessionCostSummary,
    unpriced_model_steps: int,
) -> BudgetCheck:
    if decision.limit != StopLimit.ESTIMATED_COST:
        raise ValueError("Budget checks can only be created for estimated-cost decisions.")
    if type(decision.actual) is not Decimal:
        raise TypeError("Estimated-cost decisions must use Decimal actual values.")
    return BudgetCheck(
        budget_limit_id=limit.budget_limit_id,
        scope=limit.scope,
        key=limit.key,
        window=limit.window,
        currency=limit.currency,
        maximum=limit.max_estimated_cost,
        actual=decision.actual,
        action=limit.action,
        model_steps=cost_summary.model_steps,
        unpriced_model_steps=unpriced_model_steps,
        limit_reached=True,
        message=decision.message,
        cost_summary=cost_summary,
    )


def _budget_limit_preflight_error(
    *,
    session: Session,
    limit: BudgetLimit,
    effective_at: datetime,
    billing_identity_state: BillingIdentityState = UNRESOLVED_BILLING_IDENTITY,
    pricing_provider_name: str | None = None,
    model: str | None = None,
) -> str | None:
    effective_provider_name = pricing_provider_name or session.provider_name
    effective_model = require_clean_nonblank(model or session.model, "model")
    if not isinstance(billing_identity_state, ResolvedBillingIdentity):
        if limit.allow_unpriced:
            return None
        price = budget_price(
            limit,
            provider_name=effective_provider_name,
            model=effective_model,
            effective_at=effective_at,
        )
        if price is None:
            # Contextual rates cannot be selected until the provider resolves
            # request pricing dimensions. Recheck immediately before dispatch.
            deferred = budget_check_from_events(
                limit=limit,
                events=[],
                provider_name=effective_provider_name,
                model=effective_model,
                effective_at=effective_at,
            )
            if not deferred.limit_reached:
                return None
            return (
                "Estimated cost budget cannot be verified because "
                f"{effective_provider_name}/{effective_model} has no matching pricing."
            )
        if price.currency.upper() != limit.currency.upper():
            return (
                "Estimated cost budget cannot be verified because "
                f"{effective_provider_name}/{effective_model} pricing currency {price.currency} "
                f"does not match requested {limit.currency}."
            )
        return None
    check = budget_check_from_events(
        limit=limit,
        events=[],
        provider_name=effective_provider_name,
        model=effective_model,
        billing_identity_state=billing_identity_state,
        effective_at=effective_at,
    )
    return check.message if check.limit_reached else None
