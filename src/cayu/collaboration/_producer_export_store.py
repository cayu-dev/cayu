"""Atomic intent/acknowledgement around the existing foreign export owner."""

from hashlib import sha256

from cayu.collaboration._capacity import require_capacity
from cayu.collaboration._contracts import MAX_ENVELOPE_BYTES, CollaborationConflict, OwnerRef
from cayu.collaboration._preparation import contract_bytes, prepare_contract, require_exact_contract
from cayu.collaboration._producer_contracts import (
    ProducerCompletionRecord,
    ProducerExportPublished,
    ProducerExportRecord,
    ProducerOutputRecord,
)
from cayu.collaboration._producer_store import completion_operation, read_output_registration
from cayu.collaboration._request_store import operation_key, retained_request
from cayu.collaboration.base import _Anchor
from cayu.collaboration.participants import CollaborationUnavailable


def export_operation(destination, redactor):
    return destination.operation.model_copy(
        update={
            "caller_key": "producer-export:"
            + sha256(contract_bytes(destination.operation, redactor=redactor)).hexdigest()
        }
    )


async def read_export(tx, command, destination, *, redactor):
    operation = export_operation(destination, redactor)
    raw = await tx.get("operations", operation_key(operation))
    if raw is None:
        return None
    record = prepare_contract(ProducerExportRecord, raw, redactor=redactor)
    if (record.operation, record.registration, record.destination) != (
        operation,
        command.operation,
        destination.operation,
    ):
        raise CollaborationConflict("Producer export identity conflicts.")
    if record.completion != completion_operation(command, redactor):
        raise CollaborationConflict("Producer export completion identity conflicts.")
    completion = prepare_contract(
        ProducerCompletionRecord,
        await tx.get("operations", operation_key(record.completion)),
        redactor=redactor,
    )
    require_exact_contract(command, completion.output.registration, redactor=redactor)
    prepared = command.admission.prepared
    assert prepared is not None
    request = record.request
    if (
        request.ref.session_id != prepared.target.session_id
        or request.ref.session_instance_id != prepared.target.session_instance_id
        or request.ref.operation.caller_key
        != "producer:"
        + sha256(contract_bytes(destination.operation, redactor=redactor)).hexdigest()
        or request.source_indices != completion.output.source_indices
        or completion.output.disposition != "answer"
        or request.source_selection != "assistant_visible_text_v1"
        or request.projector != destination.projector
        or request.policy != destination.disclosure_policy
        or record.initiator != command.initiator.model_copy(update={"mandate": destination.mandate})
        or request.audience
        != OwnerRef(
            application_scope=destination.recipient.owner.application_scope,
            owner_id=destination.recipient.participant_id,
            incarnation=destination.recipient.incarnation,
        )
    ):
        raise CollaborationConflict("Producer export retained selection conflicts.")
    if record.publication is not None:
        if record.publication != record.operation.model_copy(
            update={"caller_key": record.operation.caller_key + ":published"}
        ):
            raise CollaborationConflict("Producer export publication identity conflicts.")
        published = prepare_contract(
            ProducerExportPublished,
            await tx.get("operations", operation_key(record.publication)),
            redactor=redactor,
        )
        if (
            published.operation != record.publication
            or published.export != record.operation
            or published.receipt_commitment != record.receipt_commitment
            or published.sequence <= record.sequence
        ):
            raise CollaborationUnavailable("Producer export publication evidence conflicts.")
    if record.rejection is not None:
        from cayu.collaboration._producer_output_failure import read_output_failure

        await read_output_failure(tx, record, redactor=redactor)
    return record


async def _charge(
    store, tx, initialized, registration, updated, old, new, *, sequence, redactor, operations=1
):
    anchor = await store._anchor(tx, initialized, redactor)
    reservation = operations * 2 * MAX_ENVELOPE_BYTES
    updated = prepare_contract(
        ProducerOutputRecord,
        updated.model_copy(
            update={
                "reserved_operations": registration.reserved_operations - operations,
                "reserved_events": registration.reserved_events - 1,
                "reserved_bytes": registration.reserved_bytes - reservation,
            }
        ),
        redactor=redactor,
    )
    charge = sum(len(contract_bytes(value, redactor=redactor)) for value in (updated, *new)) - sum(
        len(contract_bytes(value, redactor=redactor)) for value in (registration, *old)
    )
    if charge > reservation:
        raise CollaborationUnavailable("Producer export exceeds reserved bookkeeping.")
    counted = prepare_contract(
        _Anchor,
        anchor.model_copy(
            update={
                "operation_count": anchor.operation_count + operations,
                "event_count": anchor.event_count + 1,
                "event_sequence": sequence,
                "reserved_operations": anchor.reserved_operations - operations,
                "reserved_events": anchor.reserved_events - 1,
                "reserved_bytes": anchor.reserved_bytes - reservation,
                "retained_bytes": anchor.retained_bytes + charge,
            }
        ),
        redactor=redactor,
    )
    require_capacity(counted, ordinary=False)
    await tx.put("operations", operation_key(registration.command.operation), updated, insert=False)
    await tx.put("anchors", (), counted, insert=False)


async def prepare_export(
    store, tx, initialized, command, destination, request, initiator, *, redactor
):
    from cayu.collaboration._producer_destination_exclusion import require_destination_live

    await require_destination_live(tx, command, destination, redactor=redactor)
    registration = await read_output_registration(tx, command, redactor=redactor)
    if registration is None or registration.completion is None:
        raise CollaborationUnavailable("Producer export lacks retained production.")
    completion = prepare_contract(
        ProducerCompletionRecord,
        await tx.get("operations", operation_key(registration.completion)),
        redactor=redactor,
    )
    prepared = command.admission.prepared
    assert prepared is not None
    output = completion.output
    if (
        destination not in command.destinations
        or output.disposition != "answer"
        or request.ref.session_id != prepared.target.session_id
        or request.ref.session_instance_id != prepared.target.session_instance_id
        or request.source_indices != output.source_indices
        or request.source_selection != "assistant_visible_text_v1"
        or request.projector != destination.projector
        or request.policy != destination.disclosure_policy
        or request.audience
        != OwnerRef(
            application_scope=destination.recipient.owner.application_scope,
            owner_id=destination.recipient.participant_id,
            incarnation=destination.recipient.incarnation,
        )
        or initiator != command.initiator.model_copy(update={"mandate": destination.mandate})
    ):
        raise CollaborationConflict("Producer export selection conflicts.")
    existing = await read_export(tx, command, destination, redactor=redactor)
    if existing is not None:
        require_exact_contract(request, existing.request, redactor=redactor)
        require_exact_contract(initiator, existing.initiator, redactor=redactor)
        if (
            existing.completion != completion.operation
            or existing.operation not in registration.exports
        ):
            raise CollaborationUnavailable("Producer export index conflicts.")
        return existing
    if await tx.now_ms() >= min(command.limits.deadline_at_ms, destination.attempt.deadline_at_ms):
        raise CollaborationConflict("Producer export admission expired.")
    expected = command.admission.expected
    request_state = await retained_request(
        store, tx, initialized, expected.intent.request, expected.initiator, redactor
    )
    if request_state is None or request_state.state not in {"open", "answered"}:
        raise CollaborationConflict("Request closure excludes new producer exports.")
    anchor = await store._anchor(tx, initialized, redactor)
    operation = export_operation(destination, redactor)
    record = ProducerExportRecord(
        operation=operation,
        registration=command.operation,
        completion=completion.operation,
        destination=destination.operation,
        request=request,
        initiator=initiator,
        sequence=anchor.event_sequence + 1,
    )
    updated = registration.model_copy(
        update={
            "exports": tuple(
                sorted((*registration.exports, operation), key=lambda item: item.caller_key)
            )
        }
    )
    await _charge(
        store,
        tx,
        initialized,
        registration,
        updated,
        (),
        (record,),
        sequence=record.sequence,
        redactor=redactor,
    )
    await tx.put("operations", operation_key(operation), record, insert=True)
    return record


async def require_export_interest(
    store, tx, initialized, command, destination, expected, *, redactor
):
    """Fresh dispatch eligibility, separate from historical export acknowledgement."""
    from cayu.collaboration._producer_destination_exclusion import require_destination_live

    await require_destination_live(tx, command, destination, redactor=redactor)
    registration = await read_output_registration(tx, command, redactor=redactor)
    retained = await read_export(tx, command, destination, redactor=redactor)
    if registration is None or retained is None or retained.operation not in registration.exports:
        raise CollaborationUnavailable("Producer export responsibility is unavailable.")
    require_exact_contract(retained.request, expected.request, redactor=redactor)
    request = command.admission.expected
    snapshot = await retained_request(
        store, tx, initialized, request.intent.request, request.initiator, redactor
    )
    if (
        snapshot is None
        or snapshot.state not in {"open", "answered"}
        or await tx.now_ms()
        >= min(command.limits.deadline_at_ms, destination.attempt.deadline_at_ms)
    ):
        raise CollaborationConflict("Producer export has no current request interest.")


async def acknowledge_export(
    store, tx, initialized, command, destination, expected, receipt, *, redactor
):
    registration = await read_output_registration(tx, command, redactor=redactor)
    record = await read_export(tx, command, destination, redactor=redactor)
    if record is not None and record.rejection is not None:
        raise CollaborationConflict("Rejected producer output cannot be published.")
    if registration is None or record is None or record.operation not in registration.exports:
        raise CollaborationUnavailable("Producer export responsibility is unavailable.")
    require_exact_contract(
        record.model_copy(
            update={
                "state": "prepared",
                "publication": None,
                "receipt_commitment": None,
                "output_commitment": None,
            }
        ),
        expected.model_copy(
            update={
                "state": "prepared",
                "publication": None,
                "receipt_commitment": None,
                "output_commitment": None,
            }
        ),
        redactor=redactor,
    )
    require_exact_contract(record.request, receipt.expected.intent.request, redactor=redactor)
    require_exact_contract(record.initiator, receipt.expected.initiator, redactor=redactor)
    completion = prepare_contract(
        ProducerCompletionRecord,
        await tx.get("operations", operation_key(record.completion)),
        redactor=redactor,
    )
    if completion.output.source_commitment != "sha256:" + receipt.expected.intent.source_commitment:
        raise CollaborationConflict("Producer export was not derived from retained output.")
    commitment = "sha256:" + sha256(contract_bytes(receipt, redactor=redactor)).hexdigest()
    output = "sha256:" + receipt.expected.intent.output_commitment
    if record.state == "published":
        if (record.receipt_commitment, record.output_commitment) != (commitment, output):
            raise CollaborationConflict("Producer export receipt changed.")
        return record
    anchor = await store._anchor(tx, initialized, redactor)
    operation = record.operation.model_copy(
        update={"caller_key": record.operation.caller_key + ":published"}
    )
    if await tx.get("operations", operation_key(operation)) is not None:
        raise CollaborationConflict("Producer export acknowledgement identity is occupied.")
    event = ProducerExportPublished(
        operation=operation,
        export=record.operation,
        receipt_commitment=commitment,
        sequence=anchor.event_sequence + 1,
    )
    updated = prepare_contract(
        ProducerExportRecord,
        record.model_copy(
            update={
                "state": "published",
                "publication": operation,
                "receipt_commitment": commitment,
                "output_commitment": output,
            }
        ),
        redactor=redactor,
    )
    await _charge(
        store,
        tx,
        initialized,
        registration,
        registration,
        (record,),
        (updated, event),
        sequence=event.sequence,
        redactor=redactor,
    )
    await tx.put("operations", operation_key(record.operation), updated, insert=False)
    await tx.put("operations", operation_key(operation), event, insert=True)
    return updated
