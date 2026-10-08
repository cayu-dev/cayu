"""Native continuation child publication, enclosed by the existing session lock.

Foreign reads and callbacks must finish before entering this synchronous owner.
The previous child is compared against its authenticated parent index; it is
never trusted merely because a caller supplies a structurally valid receipt.
"""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, Any

from cayu.collaboration._preparation import prepare_contract
from cayu.sessions._session_continuation import (
    CONTINUATION_MAX_SERVICES,
    ContinuationConflict,
    ContinuationRecord,
    continuation_operation_key,
    continuation_writer_frontier,
    require_ticket_identity,
)
from cayu.sessions._session_continuation_scope import current_publication_key
from cayu.sessions._session_continuation_store import (
    ROOT_KEY,
    ContinuationRoot,
    digest,
    require_history,
    service_receipt_epochs,
)
from cayu.sessions._temporary_continuation import (
    TemporaryServiceAdmission,
    TemporaryServiceExecution,
    TemporaryServiceRecord,
    advance_temporary_service_record,
    reference_for_service,
    require_temporary_service_capacity,
    temporary_service_key,
)
from cayu.vaults.redaction import SecretRedactor

if TYPE_CHECKING:
    from cayu.sessions.base import SessionOperationPublication
    from cayu.sessions.records import Session


def require_service_deadline(proposed: TemporaryServiceRecord, now: datetime) -> None:
    if int(now.timestamp() * 1000) >= proposed.intent.question.deadline_at_ms:
        raise ContinuationConflict("Temporary service question deadline has expired.")
    if proposed.intent.ticket.deadline is not None and now >= datetime.fromisoformat(
        proposed.intent.ticket.deadline
    ):
        raise ContinuationConflict("Temporary service original wait deadline has expired.")


def publish_service_record(
    session: Session,
    checkpoint: dict[str, Any] | None,
    current: dict[str, Any] | None,
    previous: TemporaryServiceRecord | None,
    proposed: TemporaryServiceRecord,
    now: datetime,
) -> SessionOperationPublication:
    """Build an atomic parent/index/child update, not a foreign admission grant."""
    from cayu.sessions.base import SessionOperationPublication, _continuation_writer_was_released

    redactor = SecretRedactor()
    proposed = prepare_contract(TemporaryServiceRecord, proposed, redactor=redactor)
    intent = proposed.intent
    key = continuation_operation_key(intent.ticket)
    child_key = temporary_service_key(intent.operation)
    if current_publication_key() != key or current is None or checkpoint is None:
        raise ContinuationConflict("Temporary service lacks its native continuation owner.")
    before = ContinuationRecord.model_validate(current)
    require_history(before)
    require_ticket_identity(intent.ticket, before.ticket)
    root = ContinuationRoot.model_validate(checkpoint.get(ROOT_KEY))
    entry = next((item for item in root.entries if item.ticket_key == key), None)
    if (
        root.namespace != before.namespace
        or session.id != before.ticket.session_id
        or session.instance_id != before.ticket.session_instance_id
        or entry is None
        or entry.record_sha256 != digest(current)
        or entry.originating_writer_generation != before.ticket.writer_generation
        or entry.purpose != before.ticket.purpose
        or entry.state != before.ticket.state
        or entry.service_receipt_epochs != service_receipt_epochs(before)
    ):
        raise ContinuationConflict("Temporary service has divergent native ownership evidence.")
    references = {item.key: item for item in before.services}
    reference = references.get(child_key)
    active = [item for item in before.services if item.state in {"reserved", "admitted"}]
    parent_key = (
        None if intent.parent_service is None else temporary_service_key(intent.parent_service)
    )
    if previous is None:
        if reference is not None:
            raise ContinuationConflict("Temporary service retry must reconcile its retained child.")
        require_service_deadline(proposed, now)
        require_temporary_service_capacity(proposed)
        if (
            proposed.state not in {"prepared", "reserved"}
            or before.ticket.service_policy == "none"
            or before.ticket != intent.ticket
            or before.ticket.state not in {"WAITING", "SERVICING"}
            or before.latch is not None
            or before.consumption is not None
            or before.retirement is not None
            or len(before.services) >= CONTINUATION_MAX_SERVICES
            or any(item.state == "prepared" for item in before.services)
            or intent.service_generation != len(before.services) + 1
            or parent_key != (active[-1].key if active else None)
            or intent.depth != len(active) + 1
            or (active and active[-1].state != "admitted")
        ):
            raise ContinuationConflict("Temporary service lost ticket, latch or stack arbitration.")
        writer, released = continuation_writer_frontier(before)
        source_epoch = writer + (not released)
        if (
            not _continuation_writer_was_released(session, checkpoint)
            or session.run_epoch != source_epoch
            or (
                intent.mode == "same_session"
                and proposed.admission.dispatch.expected_run_epoch != session.run_epoch
            )
        ):
            raise ContinuationConflict("Temporary service requires proven source writer release.")
    else:
        previous = prepare_contract(TemporaryServiceRecord, previous, redactor=redactor)
        # Another exact owner may already have committed this transition/ACK.
        # Validate the proposed transition and its complete indexed commitment;
        # an obsolete snapshot is not a conflict with identical admitted evidence.
        replays = (
            (proposed.model_copy(update={"settlement_acknowledged": True}), proposed)
            if proposed.state in {"returned", "excluded"}
            else (proposed,)
        )
        for replay in replays:
            if reference == reference_for_service(replay, before.services):
                advance_temporary_service_record(previous, proposed)
                return SessionOperationPublication(
                    checkpoint=checkpoint,
                    operation_records={key: current, child_key: replay.model_dump(mode="json")},
                )
        if reference is None or reference != reference_for_service(previous, before.services):
            raise ContinuationConflict(
                "Temporary service readback conflicts with its native index."
            )
        proposed = advance_temporary_service_record(previous, proposed)
        if previous == proposed:
            return SessionOperationPublication(
                checkpoint=checkpoint,
                operation_records={key: current, child_key: proposed.model_dump(mode="json")},
            )
        if previous.state == "prepared":
            if proposed.state == "reserved":
                require_service_deadline(proposed, now)
                writer, released = continuation_writer_frontier(before)
                if (
                    before.ticket.state not in {"WAITING", "SERVICING"}
                    or before.latch is not None
                    or before.consumption is not None
                    or before.retirement is not None
                    or not _continuation_writer_was_released(session, checkpoint)
                    or session.run_epoch != writer + (not released)
                    or parent_key != (active[-1].key if active else None)
                    or (active and active[-1].state != "admitted")
                ):
                    raise ContinuationConflict("Temporary preparation lost writer arbitration.")
        elif previous.state not in {"returned", "excluded"} and (
            not active or active[-1].key != child_key
        ):
            raise ContinuationConflict("Temporary service must settle in stack order.")
    references[child_key] = reference_for_service(proposed, before.services)
    services = tuple(sorted(references.values(), key=lambda item: item.generation))
    state = (
        "SERVICING"
        if any(item.state in {"reserved", "admitted"} for item in services)
        else ("WAITING" if before.ticket.state == "SERVICING" else before.ticket.state)
    )
    updated = before.model_copy(
        update={
            "services": services,
            "ticket": before.ticket.model_copy(
                update={"state": state, "revision": before.ticket.revision + 1}
            ),
        }
    )
    # Child receipts are the service history. Rebind the bounded parent's latest
    # digest without spending an ordinary final-continuation event per turn.
    last = updated.events[-1]
    updated = updated.model_copy(
        update={
            "events": (
                *updated.events[:-1],
                last.model_copy(
                    update={
                        "record_sha256": digest(updated.model_dump(mode="json", exclude={"events"}))
                    }
                ),
            )
        }
    )
    updated = ContinuationRecord.model_validate(updated.model_dump(mode="json"))
    require_history(updated)
    raw = updated.model_dump(mode="json")
    updated_entry = entry.model_copy(
        update={
            "state": state,
            "record_sha256": digest(raw),
            "service_receipt_epochs": service_receipt_epochs(updated),
        }
    )
    root = ContinuationRoot.model_validate(
        root.model_dump()
        | {
            "entries": tuple(
                updated_entry if item.ticket_key == key else item for item in root.entries
            )
        }
    )
    return SessionOperationPublication(
        checkpoint=checkpoint | {ROOT_KEY: root.model_dump(mode="json")},
        operation_records={key: raw, child_key: proposed.model_dump(mode="json")},
    )


def compose_temporary_service_admission(
    *,
    source_session,
    source_checkpoint,
    parent_record,
    child_record,
    admitted_session,
    admitted_checkpoint,
    admission,
    now,
):
    """Select the native composition without treating a handoff as atomic."""
    if admission.dispatch.intent.mode == "same_session":
        return compose_same_session_admission(
            source_session=source_session,
            source_checkpoint=source_checkpoint,
            parent_record=parent_record,
            child_record=child_record,
            admitted_session=admitted_session,
            admitted_checkpoint=admitted_checkpoint,
            admission=admission,
            now=now,
        )
    from cayu.sessions._temporary_service_target import (
        TemporaryServiceTarget,
        admit_side_target,
        publish_target_record,
    )

    if child_record is None or source_session.run_epoch != admission.dispatch.expected_run_epoch:
        raise ContinuationConflict("Side-session admission lacks its prepared target fence.")
    previous = TemporaryServiceTarget.model_validate(child_record)
    proposed = admit_side_target(
        previous,
        admission=admission,
        checkpoint=admitted_checkpoint,
        session_id=admitted_session.id,
        session_instance_id=admitted_session.instance_id,
        run_epoch=admitted_session.run_epoch,
        now=now,
    )
    return publish_target_record(
        admitted_session, admitted_checkpoint, child_record, previous, proposed, now
    )


def compose_same_session_admission(
    *,
    source_session: Session,
    source_checkpoint: dict[str, Any] | None,
    parent_record: dict[str, Any] | None,
    child_record: dict[str, Any] | None,
    admitted_session: Session,
    admitted_checkpoint: dict[str, Any] | None,
    admission: TemporaryServiceAdmission,
    now: datetime,
) -> SessionOperationPublication:
    """Compose service ownership with the native invocation transaction result."""
    from cayu.sessions._temporary_continuation_scope import require_temporary_transition
    from cayu.sessions.records import SessionStatus

    require_temporary_transition(admission)
    intent = admission.dispatch.intent
    expected_epoch = admission.dispatch.expected_run_epoch + 1
    if (
        intent.mode != "same_session"
        or source_session.id != intent.target.object_id
        or source_session.instance_id != intent.target.incarnation
        or source_session.run_epoch != admission.dispatch.expected_run_epoch
        or admitted_session.id != source_session.id
        or admitted_session.instance_id != source_session.instance_id
        or admitted_session.run_epoch != expected_epoch
        or admitted_session.status is not SessionStatus.RUNNING
        or admitted_checkpoint is None
    ):
        raise ContinuationConflict("Temporary service native admission has a different target.")
    execution = native_service_execution(admission, admitted_checkpoint)
    if execution is None:
        raise ContinuationConflict("Temporary service lacks its exact native permit consumption.")
    previous = (
        None
        if child_record is None
        else prepare_contract(TemporaryServiceRecord, child_record, redactor=SecretRedactor())
    )
    if previous is not None and previous.state != "prepared":
        raise ContinuationConflict("Temporary service admission was already decided.")
    reserved = TemporaryServiceRecord(admission=admission, state="reserved")
    claim = publish_service_record(
        source_session, source_checkpoint, parent_record, previous, reserved, now
    )
    admitted = TemporaryServiceRecord(admission=admission, state="admitted", execution=execution)
    return publish_service_record(
        admitted_session,
        admitted_checkpoint | {ROOT_KEY: claim.checkpoint[ROOT_KEY]},
        claim.operation_records[continuation_operation_key(intent.ticket)],
        reserved,
        admitted,
        now,
    )


def native_service_execution(
    admission: TemporaryServiceAdmission, checkpoint: dict[str, Any] | None
) -> TemporaryServiceExecution | None:
    """Project an exact receiving receipt; absence is never admission exclusion."""
    from cayu.sessions._invocation_lifecycle import (
        InvocationLifecycleCommandKind,
        _invocation_lifecycle_receipt_from_checkpoint,
    )

    intent = admission.dispatch.intent
    target = intent.target
    expected_epoch = admission.dispatch.expected_run_epoch + 1
    identity = f"admit:{target.object_id}:{target.incarnation}:{expected_epoch}"
    receipt = _invocation_lifecycle_receipt_from_checkpoint(checkpoint, command_identity=identity)
    if receipt is None:
        return None
    if (
        receipt.kind is not InvocationLifecycleCommandKind.ADMIT
        or receipt.session_id != target.object_id
        or receipt.session_instance_id != target.incarnation
        or receipt.active_profile.run_epoch != expected_epoch
        or receipt.command_sha256 != admission.admission_command_sha256
        or receipt.participant_permit_operation != admission.permit.operation.caller_key
        or receipt.participant_permit_commitment != admission.permit_receipt_sha256
        or receipt.temporary_service_operation_key != temporary_service_key(intent.operation)
        or receipt.active_profile.interaction_id != intent.invocation_id
        or receipt.active_profile.profile.fingerprint != intent.execution_profile_sha256
    ):
        raise ContinuationConflict("Temporary service lacks its exact native permit consumption.")
    return TemporaryServiceExecution(
        receipt_id=receipt.command_identity,
        receipt_sha256=receipt.record_sha256,
        admission_command_sha256=receipt.command_sha256,
        session_id=target.object_id,
        session_instance_id=target.incarnation,
        invocation_id=receipt.active_profile.interaction_id,
        run_epoch=receipt.active_profile.run_epoch,
    )


def native_service_outcome(
    admission: TemporaryServiceAdmission, checkpoint: dict[str, Any] | None
) -> TemporaryServiceRecord | None:
    """Reconstruct admission/return from native receipts, never exception inference."""
    from cayu.collaboration._permits import ReceivingSettlementReceipt
    from cayu.sessions._invocation_lifecycle import (
        InvocationLifecycleCommandKind,
        _invocation_lifecycle_receipt_from_checkpoint,
    )

    execution = native_service_execution(admission, checkpoint)
    if execution is None:
        return None
    release = _invocation_lifecycle_receipt_from_checkpoint(
        checkpoint,
        command_identity=f"release:{execution.session_id}:{execution.session_instance_id}:{execution.run_epoch}",
    )
    if release is None:
        return TemporaryServiceRecord(admission=admission, state="admitted", execution=execution)
    if (
        release.kind is not InvocationLifecycleCommandKind.RELEASE
        or release.session_id != execution.session_id
        or release.session_instance_id != execution.session_instance_id
        or release.active_profile.run_epoch != execution.run_epoch
        or release.active_profile.interaction_id != execution.invocation_id
        or release.active_profile.profile.fingerprint
        != admission.dispatch.intent.execution_profile_sha256
        or release.result_session.run_epoch != execution.run_epoch + 1
    ):
        raise ContinuationConflict("Temporary service release conflicts with native admission.")
    return TemporaryServiceRecord(
        admission=admission,
        state="returned",
        execution=execution,
        settlement=ReceivingSettlementReceipt(
            expected=admission.permit,
            receiving_owner=admission.dispatch.intent.target.owner,
            receipt_id=release.command_identity,
            outcome="quiescent",
        ),
        returned_writer_generation=release.result_session.run_epoch,
        released_session_status=release.result_session.status.value,
    )
