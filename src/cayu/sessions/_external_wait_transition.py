"""Pure election rules, executed within the native owner's transaction."""

from __future__ import annotations

from typing import Literal
from uuid import uuid4

from pydantic import Field, StrictStr

from cayu.sessions._session_continuation import ContinuationPreparation, ContinuationService
from cayu.sessions.external_waits import (
    EXTERNAL_WAIT_MAX_HORIZON_SECONDS,
    ExternalCorrelation,
    ExternalCorrelationRequest,
    ExternalDeliveryReceipt,
    ExternalEventDelivery,
    ExternalWaitCapacityExceeded,
    ExternalWaitConflict,
    ExternalWaitExecution,
    ExternalWaitExecutionIntent,
    ExternalWaitExecutionRetirement,
    ExternalWaitLimits,
    ExternalWaitOutcome,
    ExternalWaitRecord,
    ExternalWaitRegistration,
    ExternalWaitTimer,
    Identifier,
    _Value,
    canonical_payload,
    encode_record,
)


class ExternalWaitMutation(_Value):
    kind: Literal[
        "reserve",
        "register",
        "deliver",
        "cancel",
        "observe",
        "project",
        "bind",
        "settle",
        "prepare_execution",
        "exclude_execution",
        "prepare_service",
        "complete_retirement",
        "prepare_retirement",
        "reconcile_binding",
        "prepare_timer",
        "publish_timer",
    ]
    request: ExternalCorrelationRequest
    limits: ExternalWaitLimits
    expected: ExternalCorrelation | None = None
    registration: ExternalWaitRegistration | None = None
    delivery: ExternalEventDelivery | None = None
    operation_key: Identifier | None = None
    content_sha256: StrictStr | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    expected_outcome: ExternalWaitOutcome | None = None
    projection_json: StrictStr | None = None
    continuation: ContinuationPreparation | None = None
    service: ContinuationService | None = None
    service_stage_id: Identifier | None = None
    execution_intent: ExternalWaitExecutionIntent | None = None
    preparation_owner_id: Identifier | None = None
    handoff_disposition: Literal["settled", "excluded"] | None = None
    handoff_receipt_sha256: StrictStr | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    timer: ExternalWaitTimer | None = None


def transition(
    current: ExternalWaitRecord | None,
    command: ExternalWaitMutation,
    *,
    now_ms: int,
    count: int,
    reserved_bytes: int,
) -> ExternalWaitRecord:
    request = command.request
    if current is None:
        if command.kind != "reserve":
            raise ExternalWaitConflict("External correlation is unavailable.")
        if count >= command.limits.correlations or (
            reserved_bytes + command.limits.payload_bytes + command.limits.projection_bytes
            > command.limits.retained_bytes
        ):
            raise ExternalWaitCapacityExceeded("External correlation capacity is full.")
        if request.deadline is not None and (
            int(request.deadline.timestamp() * 1000) - now_ms
            > EXTERNAL_WAIT_MAX_HORIZON_SECONDS * 1000
        ):
            raise ValueError("External wait deadline exceeds its horizon.")
        return ExternalWaitRecord(
            correlation=ExternalCorrelation(
                request=request,
                incarnation=uuid4().hex,
                created_at_ms=now_ms,
                early_event_expires_at_ms=now_ms + request.early_event_retention_seconds * 1000,
                limits=command.limits,
            ),
            revision=1,
        )
    if current.correlation.request != request or current.correlation.limits != command.limits:
        raise ExternalWaitConflict("External correlation intent conflicts.")
    if command.kind == "reserve":
        return current
    if command.expected != current.correlation:
        raise ExternalWaitConflict("External correlation incarnation conflicts.")
    if command.kind in {"prepare_timer", "publish_timer"}:
        from cayu.runtime._external_wait_timer import require_timer_scope

        require_timer_scope(command)
        if command.registration != current.registration or command.timer is None:
            raise ExternalWaitConflict("External timer lacks its exact registration.")
        if current.timer is not None and current.timer != command.timer:
            raise ExternalWaitConflict("External timer was prepared differently.")
        if command.kind == "publish_timer" and current.timer is None:
            raise ExternalWaitConflict("External timer was not prepared.")
        published = current.timer_published or command.kind == "publish_timer"
        if current.timer == command.timer and current.timer_published == published:
            return current
        result = ExternalWaitRecord.model_validate(
            {
                **current.__dict__,
                "timer": command.timer,
                "timer_published": published,
                "revision": current.revision + 1,
            }
        )
        encode_record(result)
        return result
    if command.kind == "reconcile_binding":
        from cayu.runtime._external_wait_settlement import require_settlement_scope

        require_settlement_scope(command)
        if (
            current.registration != command.registration
            or current.execution is None
            or current.execution_excluded
        ):
            raise ExternalWaitConflict("External binding recovery lacks its execution preparation.")
        if current.continuation is not None:
            if current.continuation != command.continuation:
                raise ExternalWaitConflict("External binding recovery conflicts.")
            return current
        result = ExternalWaitRecord.model_validate(
            {
                **current.__dict__,
                "continuation": command.continuation,
                "handoff": "pending",
                "revision": current.revision + 1,
            }
        )
        encode_record(result)
        return result
    if command.kind == "prepare_retirement":
        from cayu.runtime._external_wait_settlement import require_settlement_scope

        require_settlement_scope(command)
        if (
            current.registration != command.registration
            or current.continuation is None
            or current.continuation != command.continuation
            or current.outcome is None
            or command.operation_key is None
        ):
            raise ExternalWaitConflict("External retirement request conflicts with its wait.")
        if current.execution_retirement is not None:
            if current.execution_retirement.operation_key != command.operation_key:
                raise ExternalWaitConflict(
                    "External execution was requested to retire differently."
                )
            return current
        if current.handoff != "pending":
            raise ExternalWaitConflict("External execution handoff is already terminal.")
        result = ExternalWaitRecord.model_validate(
            {
                **current.__dict__,
                "execution_retirement": ExternalWaitExecutionRetirement(
                    operation_key=command.operation_key, prepared_at_ms=now_ms
                ),
                "revision": current.revision + 1,
            }
        )
        encode_record(result)
        return result
    if command.kind == "exclude_execution":
        from cayu.runtime._external_wait_execution_scope import require_execution_preparation

        require_execution_preparation(command)
        if (
            current.registration != command.registration
            or current.execution is None
            or current.execution.intent != command.execution_intent
            or current.execution.preparation_owner_id != command.preparation_owner_id
            or current.continuation is not None
            or current.outcome is None
        ):
            raise ExternalWaitConflict("External execution exclusion identity conflicts.")
        # Explicit exclusion discharges execution responsibility, not election.
        # An accepted event/timeout remains inspectable and immutable. Native
        # stores independently require the exact unconsumed frontier or released
        # writer under the same lock that fences delayed CREATE/ADMIT.
        if current.execution_excluded:
            return current
        return ExternalWaitRecord.model_validate(
            {**current.__dict__, "execution_excluded": True, "revision": current.revision + 1}
        )
    if command.kind == "prepare_service":
        from cayu.runtime._external_wait_settlement import require_settlement_scope

        require_settlement_scope(command)
        if (
            current.registration != command.registration
            or current.continuation != command.continuation
            or command.service is None
            or command.service_stage_id is None
        ):
            raise ExternalWaitConflict("External service binding conflicts.")
        if current.service is not None:
            if (
                current.service != command.service
                or current.service_stage_id != command.service_stage_id
            ):
                raise ExternalWaitConflict("External service was prepared differently.")
            return current
        result = ExternalWaitRecord.model_validate(
            {
                **current.__dict__,
                "service": command.service,
                "service_stage_id": command.service_stage_id,
                "revision": current.revision + 1,
            }
        )
        encode_record(result)
        return result
    if command.kind == "prepare_execution":
        from cayu.runtime._external_wait_execution_scope import require_execution_preparation

        require_execution_preparation(command)
        if command.preparation_owner_id is None:
            raise ExternalWaitConflict("External execution requires a retained preparation owner.")
        if (
            command.registration != current.registration
            or current.registration is None
            or command.execution_intent is None
        ):
            raise ExternalWaitConflict("External execution has no exact registered wait.")
        if current.execution is not None:
            if current.execution.intent != command.execution_intent:
                raise ExternalWaitConflict(
                    "External execution intent conflicts with its retained operation."
                )
            return current
        if current.continuation is not None or (
            current.outcome is not None and current.outcome.kind in {"cancelled", "unavailable"}
        ):
            raise ExternalWaitConflict("External wait cannot acquire new execution responsibility.")
        intent = command.execution_intent
        interaction_id, interaction_event_id = str(uuid4()), str(uuid4())
        if intent.mode == "resume":
            from cayu.runtime._external_wait_admission import resume_identity

            admission = resume_identity(intent)
            interaction_id = admission.target_active_profile.interaction_id
            assert admission.interaction_started_event is not None
            interaction_event_id = admission.interaction_started_event.id
        result = ExternalWaitRecord.model_validate(
            {
                **current.__dict__,
                "execution": ExternalWaitExecution(
                    intent=intent,
                    preparation_owner_id=command.preparation_owner_id,
                    session_instance_id=intent.expected_session_instance_id or str(uuid4()),
                    interaction_id=interaction_id,
                    interaction_event_id=interaction_event_id,
                    prepared_at_ms=now_ms,
                ),
                "revision": current.revision + 1,
            }
        )
        encode_record(result)
        return result
    if command.kind in {"settle", "complete_retirement"}:
        from cayu.runtime._external_wait_settlement import require_settlement_scope

        require_settlement_scope(command)
        if (
            current.registration != command.registration
            or current.continuation is None
            or current.continuation != command.continuation
        ):
            raise ExternalWaitConflict("External settlement binding conflicts.")
        if current.handoff in {"settled", "excluded"}:
            if (current.handoff, current.handoff_receipt_sha256) != (
                command.handoff_disposition,
                command.handoff_receipt_sha256,
            ):
                raise ExternalWaitConflict("External settlement evidence conflicts.")
            if command.kind == "complete_retirement" and not current.retirement_complete:
                return ExternalWaitRecord.model_validate(
                    {
                        **current.__dict__,
                        "retirement_complete": True,
                        "revision": current.revision + 1,
                    }
                )
            return current
        if command.kind == "complete_retirement":
            raise ExternalWaitConflict("External exclusion has not committed.")
        result = ExternalWaitRecord.model_validate(
            {
                **current.__dict__,
                "handoff": command.handoff_disposition,
                "handoff_receipt_sha256": command.handoff_receipt_sha256,
                "revision": current.revision + 1,
            }
        )
        encode_record(result)
        return result
    if command.kind == "bind":
        from cayu.runtime._external_wait_binding import require_binding_scope

        require_binding_scope(command)
        if current.execution_excluded:
            raise ExternalWaitConflict("External execution was excluded before creation.")
        if (
            current.registration is None
            or current.registration != command.registration
            or command.continuation is None
        ):
            raise ExternalWaitConflict("External continuation registration conflicts.")
        if current.continuation is not None:
            if current.continuation != command.continuation:
                raise ExternalWaitConflict("External continuation is already bound differently.")
            return current
        if current.outcome is not None and current.outcome.kind in {"cancelled", "unavailable"}:
            raise ExternalWaitConflict(
                "External wait cannot acquire new continuation responsibility."
            )
        ticket = command.continuation.intent
        deadline = current.correlation.request.deadline
        if ticket.deadline != (None if deadline is None else deadline.isoformat()):
            raise ExternalWaitConflict("External continuation deadline conflicts.")
        result = ExternalWaitRecord.model_validate(
            {
                **current.__dict__,
                "continuation": command.continuation,
                "handoff": "pending",
                "revision": current.revision + 1,
            }
        )
        encode_record(result)
        return result
    if command.kind == "project":
        if (
            current.registration is None
            or current.registration != command.registration
            or current.outcome is None
            or current.outcome != command.expected_outcome
            or current.outcome.kind not in {"event", "timeout"}
            or command.projection_json is None
        ):
            raise ExternalWaitConflict("External projection lacks its exact registered outcome.")
        projected = canonical_payload(
            command.projection_json, current.correlation.limits.projection_bytes
        )
        if current.projection_json is not None:
            if current.projection_json != projected:
                raise ExternalWaitConflict("External projection was committed differently.")
            return current
        result = ExternalWaitRecord.model_validate(
            {
                **current.__dict__,
                "projection_json": projected,
                "revision": current.revision + 1,
            }
        )
        encode_record(result)
        return result
    # Validate exact operations BEFORE performing lazy expiry/election. A
    # rejected operation cannot leave partial mutations on any backend.
    if command.kind == "register":
        registration = command.registration
        if registration is None or registration.correlation != current.correlation:
            raise ExternalWaitConflict("External registration identity conflicts.")
        if current.registration is not None:
            if current.registration != registration:
                raise ExternalWaitConflict("External registration intent conflicts.")
            return current
    if command.kind == "cancel":
        if command.operation_key is None:
            raise ValueError("Cancellation requires an operation key.")
        if current.cancel_key is not None:
            if current.cancel_key != command.operation_key:
                raise ExternalWaitConflict("External cancellation intent conflicts.")
            return current
    if command.kind == "deliver":
        delivery = command.delivery
        if delivery is None or delivery.correlation != current.correlation:
            raise ExternalWaitConflict("External delivery identity conflicts.")
        if command.content_sha256 is None:
            raise ValueError("External delivery requires a content commitment.")
        canonical_payload(delivery.payload_json, command.limits.payload_bytes)
        for receipt in current.deliveries:
            if receipt.delivery_id == delivery.delivery_id:
                if receipt.content_sha256 != command.content_sha256:
                    raise ExternalWaitConflict("External delivery content conflicts.")
                return current
        if len(current.deliveries) >= command.limits.deliveries:
            raise ExternalWaitCapacityExceeded("External delivery receipt capacity is full.")

    update: dict[str, object] = {}
    early = current.early_event
    outcome = current.outcome
    deadline = None if request.deadline is None else int(request.deadline.timestamp() * 1000)
    # Expiry is never reinterpreted as an empty inbox eligible for replacement.
    if current.registration is None and now_ms >= current.correlation.early_event_expires_at_ms:
        if outcome is None:
            outcome = ExternalWaitOutcome(kind="unavailable", selected_at_ms=now_ms)
        early = None
    if command.kind == "register":
        update["registration"] = command.registration
    registered = current.registration is not None or command.kind == "register"
    if outcome is None and registered:
        if early is not None:
            outcome = early
            early = None
        elif deadline is not None and now_ms >= deadline:
            outcome = ExternalWaitOutcome(kind="timeout", selected_at_ms=deadline)
    if command.kind == "deliver":
        assert command.delivery is not None and command.content_sha256 is not None
        if outcome is not None:
            disposition = "unavailable" if outcome.kind == "unavailable" else "settled"
        elif deadline is not None and now_ms >= deadline:
            disposition = "late"
        elif early is not None:
            disposition = "additional"
        else:
            disposition = "accepted"
            accepted = ExternalWaitOutcome(
                kind="event",
                selected_at_ms=now_ms,
                delivery_id=command.delivery.delivery_id,
                payload_json=command.delivery.payload_json,
                content_sha256=command.content_sha256,
            )
            if registered:
                outcome = accepted
            else:
                early = accepted
        receipt = ExternalDeliveryReceipt(
            delivery_id=command.delivery.delivery_id,
            content_sha256=command.content_sha256,
            accepted_at_ms=now_ms,
            disposition=disposition,
        )
        update["deliveries"] = (*current.deliveries, receipt)
    if command.kind == "cancel":
        update["cancel_key"] = command.operation_key
        if outcome is None:
            if early is not None:
                outcome, early = early, None
            elif deadline is not None and now_ms >= deadline:
                outcome = ExternalWaitOutcome(kind="timeout", selected_at_ms=deadline)
            else:
                outcome = ExternalWaitOutcome(kind="cancelled", selected_at_ms=now_ms)
    update.update(early_event=early, outcome=outcome)
    if all(getattr(current, field) == value for field, value in update.items()):
        return current
    update["revision"] = current.revision + 1
    result = ExternalWaitRecord.model_validate({**current.__dict__, **update})
    encode_record(result)
    return result
