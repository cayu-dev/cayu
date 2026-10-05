"""Pure external-wait record projections and reconstruction checks, not authority."""

from __future__ import annotations

from datetime import UTC, datetime
from hashlib import sha256
from typing import Any

from cayu._validation import canonical_durable_json_bytes
from cayu.collaboration._contracts import ObjectRef, OwnerRef
from cayu.sessions._session_continuation import (
    ContinuationLatch,
    ContinuationPreparation,
    ContinuationWait,
    continuation_digest,
)
from cayu.sessions.external_waits import (
    ExternalWaitConflict,
    ExternalWaitRecord,
    ExternalWaitRegistration,
    ExternalWaitUnavailable,
    external_wait_digest,
)


def external_wait_owner(registration: ExternalWaitRegistration) -> OwnerRef:
    scope = registration.correlation.request.scope
    return OwnerRef(
        application_scope=scope.application_scope,
        owner_id="external-waits",
        incarnation=str(scope.generation),
    )


def external_continuation_intent(registration: ExternalWaitRegistration) -> ContinuationWait:
    correlation = registration.correlation
    request = correlation.request
    return ContinuationWait(
        registration_key="external-wait:" + external_wait_digest(registration),
        targets=(
            ObjectRef(
                owner=external_wait_owner(registration),
                kind="external-event",
                object_id=request.correlation_key,
                incarnation=correlation.incarnation,
                revision=1,
            ),
        ),
        predicate_kind="ANY_SUCCESS",
        predicate_version=1,
        deadline=None if request.deadline is None else request.deadline.isoformat(),
        failure_policy="unavailable",
        service_policy="external-event-v1",
        wait_edge_revision=1,
        purpose="external-event-v1",
    )


def require_binding_identity(
    registration: ExternalWaitRegistration, preparation: ContinuationPreparation
) -> None:
    """Reconstruction consistency only; not authentication or writer authority."""
    ticket = preparation.intent
    actual = ContinuationWait.model_validate(
        {name: getattr(ticket, name) for name in ContinuationWait.model_fields}
    )
    if (
        actual != external_continuation_intent(registration)
        or ticket.owner.application_scope
        != registration.correlation.request.scope.application_scope
        or preparation.registration.child.destination != external_wait_owner(registration)
    ):
        raise PermissionError("External continuation does not bind the exact registered wait.")


def elected_external_latch(record: ExternalWaitRecord) -> ContinuationLatch:
    """Project retained evidence; callers must still use the registered receiver."""
    if (
        record.registration is None
        or record.continuation is None
        or record.outcome is None
        or record.outcome.kind not in {"event", "timeout"}
        or record.projection_json is None
        or record.handoff not in {"pending", "settled"}
    ):
        raise ExternalWaitUnavailable("External continuation has no retained ready outcome.")
    ticket = record.continuation.intent
    return ContinuationLatch(
        ticket=ticket,
        wait_receipt_digest=continuation_digest(record.continuation.registration),
        outcome_kind="success" if record.outcome.kind == "event" else "settled",
        selected_manifest=ticket.targets if record.outcome.kind == "event" else (),
        disclosure_digest=sha256(
            canonical_durable_json_bytes(
                {
                    "registration": record.registration.model_dump(mode="json"),
                    "projection": record.projection_json,
                },
                "external projection",
            )
        ).hexdigest(),
        latch_key="external-outcome:" + external_wait_digest(record.registration),
        outcome_digest=external_wait_digest(record.outcome),
        accepted_at=datetime.fromtimestamp(
            record.outcome.selected_at_ms / 1000, tz=UTC
        ).isoformat(),
    )


def validate_external_wait_row(row: Any) -> ExternalWaitRecord:
    """Every indexed column must agree with its reconstructed document."""
    record = ExternalWaitRecord.model_validate_json(row["record_json"])
    scope = record.correlation.request.scope
    expected = (
        scope.application_scope,
        scope.generation,
        record.correlation.request.correlation_key,
        record.correlation.request.source,
        record.correlation.incarnation,
        record.revision,
        record.reserved_bytes,
        record.handoff,
        int(record.pending_handoff),
        *external_wait_session_identity(record),
    )
    actual = tuple(
        row[key]
        for key in (
            "scope",
            "generation",
            "correlation_key",
            "source",
            "incarnation",
            "revision",
            "reserved_bytes",
            "handoff",
            "pending_handoff",
            "session_id",
            "session_instance_id",
        )
    )
    if actual != expected:
        raise ExternalWaitConflict("External wait indexed identity conflicts.")
    return record


def external_wait_session_identity(record: ExternalWaitRecord) -> tuple[Any, Any]:
    if record.continuation is not None:
        return record.continuation.intent.session_id, record.continuation.intent.session_instance_id
    if record.execution is not None:
        return record.execution.intent.session_id, record.execution.session_instance_id
    return None, None
