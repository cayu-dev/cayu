"""Mandatory producer delivery bookkeeping around the existing receiving queue."""

from hashlib import sha256

from cayu.collaboration._contracts import MAX_ID_BYTES, CollaborationConflict
from cayu.collaboration._preparation import contract_bytes, prepare_contract, require_exact_contract
from cayu.collaboration._producer_bounds import supports_peer_delivery
from cayu.collaboration._producer_contracts import (
    ProducerDeliveryAccepted,
    ProducerDeliveryIndex,
    ProducerDeliveryRecord,
)
from cayu.collaboration._producer_export_store import _charge, read_export, require_export_interest
from cayu.collaboration._producer_outcome_store import outcome_operation
from cayu.collaboration._producer_store import read_output_registration
from cayu.collaboration._request_store import operation_key, require_request_event
from cayu.collaboration._session_export_store import digest
from cayu.collaboration.participants import CollaborationUnavailable
from cayu.collaboration.peer_content import PeerContentAppendRequest, PeerContentReceipt
from cayu.collaboration.requests import ProducerOutcomeCommand, RequestOutcomeReceipt


def delivery_operation(destination, redactor):
    return destination.operation.model_copy(
        update={
            "caller_key": "producer-delivery:"
            + sha256(contract_bytes(destination.operation, redactor=redactor)).hexdigest()
        }
    )


def acceptance_index_operation(namespace, export_ref, redactor):
    return namespace.model_copy(
        update={
            "caller_key": "producer-acceptance:"
            + sha256(contract_bytes(export_ref, redactor=redactor)).hexdigest()
        }
    )


async def read_delivery(tx, command, destination, *, redactor):
    operation = delivery_operation(destination, redactor)
    raw = await tx.get("operations", operation_key(operation))
    if raw is None:
        return None
    record = prepare_contract(ProducerDeliveryRecord, raw, redactor=redactor)
    index_operation = acceptance_index_operation(
        command.operation, record.source_receipt.expected.intent.request.ref, redactor
    )
    index = prepare_contract(
        ProducerDeliveryIndex,
        await tx.get("operations", operation_key(index_operation)),
        redactor=redactor,
    )
    if (
        index.operation != index_operation
        or index.delivery != operation
        or index.registration != command.operation
        or index.destination != destination.operation
        or index.export_receipt_commitment
        != "sha256:" + sha256(contract_bytes(record.source_receipt, redactor=redactor)).hexdigest()
    ):
        raise CollaborationConflict("Producer receiving index conflicts.")
    elected = prepare_contract(
        RequestOutcomeReceipt,
        await tx.get("operations", operation_key(record.outcome)),
        redactor=redactor,
    )
    if (
        record.outcome != outcome_operation(command, redactor)
        or not isinstance(elected.command, ProducerOutcomeCommand)
        or elected.command.operation != record.outcome
        or elected.command.producer != command.operation
        or elected.command.outcome != "answered"
        or elected.command.publisher_generation != command.publisher_generation
    ):
        raise CollaborationConflict("Producer delivery lacks its elected answer.")
    require_exact_contract(command.admission.expected, elected.command.expected, redactor=redactor)
    await require_request_event(tx, elected.event, redactor)
    exported = await read_export(tx, command, destination, redactor=redactor)
    prepared = command.admission.prepared
    assert prepared is not None
    occurrence = record.append.occurrence
    if (
        destination not in command.destinations
        or (record.operation, record.registration, record.destination)
        != (operation, command.operation, destination.operation)
        or exported is None
        or exported.state != "published"
        or record.export != exported.operation
        or elected.command.completion != exported.completion
        or record.append.operation_key != operation.caller_key
        or record.append.attempt_key != destination.attempt
        or record.append.wake_policy != "none"
        or record.append.replaces_operation_key is not None
        or (occurrence.sender_participant_id, occurrence.sender_participant_incarnation)
        != (prepared.recipient.participant_id, prepared.recipient.incarnation)
        or (occurrence.sender_session_id, occurrence.sender_session_instance_id)
        != (prepared.target.session_id, prepared.target.session_instance_id)
        or occurrence.audience != (destination.recipient.participant_id,)
        or occurrence.payload.artifact_commitments
        or occurrence.producer_receipt_id != command.operation.caller_key
        or occurrence.source_export_receipt_id != record.source_receipt.event_id
        or "sha256:" + sha256(contract_bytes(record.source_receipt, redactor=redactor)).hexdigest()
        != exported.receipt_commitment
        or "sha256:" + digest({"text": occurrence.payload.text}) != exported.output_commitment
    ):
        raise CollaborationConflict("Producer delivery identity conflicts.")
    if record.acceptance is not None:
        assert record.receipt is not None
        expected = operation.model_copy(update={"caller_key": operation.caller_key + ":accepted"})
        event = prepare_contract(
            ProducerDeliveryAccepted,
            await tx.get("operations", operation_key(record.acceptance)),
            redactor=redactor,
        )
        if (
            record.acceptance != expected
            or event.operation != expected
            or event.delivery != operation
            or event.sequence <= record.sequence
            or event.receipt_commitment
            != "sha256:" + sha256(contract_bytes(record.receipt, redactor=redactor)).hexdigest()
        ):
            raise CollaborationUnavailable("Producer delivery acceptance evidence conflicts.")
    return record


async def prepare_delivery(
    store,
    tx,
    initialized,
    command,
    destination,
    append,
    source_receipt,
    *,
    authority_expires_at_ms,
    redactor,
):
    """Called only under the registered source owner's current disclosure guard."""
    from cayu.collaboration._producer_destination_exclusion import require_destination_live

    await require_destination_live(tx, command, destination, redactor=redactor)
    append = prepare_contract(PeerContentAppendRequest, append, redactor=redactor)
    registration = await read_output_registration(tx, command, redactor=redactor)
    exported = await read_export(tx, command, destination, redactor=redactor)
    if registration is None or exported is None or exported.state != "published":
        raise CollaborationUnavailable("Producer delivery lacks published source responsibility.")
    existing = await read_delivery(tx, command, destination, redactor=redactor)
    if existing is not None:
        require_exact_contract(existing.append, append, redactor=redactor)
        require_exact_contract(existing.source_receipt, source_receipt, redactor=redactor)
        return existing
    if type(authority_expires_at_ms) is not int or await tx.now_ms() >= authority_expires_at_ms:
        raise CollaborationConflict("Producer delivery disclosure expired.")
    await require_export_interest(
        store, tx, initialized, command, destination, exported, redactor=redactor
    )
    anchor = await store._anchor(tx, initialized, redactor)
    operation = delivery_operation(destination, redactor)
    record = ProducerDeliveryRecord(
        operation=operation,
        registration=command.operation,
        destination=destination.operation,
        export=exported.operation,
        outcome=outcome_operation(command, redactor),
        source_receipt=source_receipt,
        append=append,
        sequence=anchor.event_sequence + 1,
    )
    # Reserve the complete receiving outcome envelope before queue dispatch,
    # not merely its byte-accounting charge. This schema-only bound is never
    # persisted or treated as receiving acceptance.
    key = append.append_key
    maximum = "\x01" * MAX_ID_BYTES
    prepare_contract(
        ProducerDeliveryRecord,
        record.model_copy(
            update={
                "acceptance": operation.model_copy(
                    update={"caller_key": operation.caller_key + ":accepted"}
                ),
                "receipt": PeerContentReceipt(
                    operation_key=append.operation_key,
                    append_key=key,
                    attempt_generation=append.attempt_key.attempt_generation,
                    status="appended",
                    occurrence=append.occurrence,
                    queue_id=maximum,
                    transcript_event_id=maximum,
                    target_session_id=key.target_session_id or maximum,
                    target_session_instance_id=key.target_session_instance_id or maximum,
                ),
            }
        ),
        redactor=redactor,
    )
    updated = registration.model_copy(
        update={
            "deliveries": tuple(
                sorted((*registration.deliveries, operation), key=lambda item: item.caller_key)
            )
        }
    )
    await tx.put("operations", operation_key(operation), record, insert=True)
    index = ProducerDeliveryIndex(
        operation=acceptance_index_operation(
            command.operation, source_receipt.expected.intent.request.ref, redactor
        ),
        registration=command.operation,
        destination=destination.operation,
        delivery=operation,
        export_receipt_commitment="sha256:"
        + sha256(contract_bytes(source_receipt, redactor=redactor)).hexdigest(),
    )
    await tx.put("operations", operation_key(index.operation), index, insert=True)
    # Validate the exact same reconstructed tuple before this transaction commits.
    await read_delivery(tx, command, destination, redactor=redactor)
    await _charge(
        store,
        tx,
        initialized,
        registration,
        updated,
        (),
        (record, index),
        sequence=record.sequence,
        redactor=redactor,
        operations=2,
    )
    return record


async def reconcile_delivery(store, initialized, command, destination, receiving, *, redactor):
    """Fixed receiving readback only; absence never proves exclusion or exposure."""
    scope = initialized.owner.application_scope
    async with store._transaction(scope, write=False) as tx:
        current = await read_delivery(tx, command, destination, redactor=redactor)
    if current is None:
        raise CollaborationUnavailable("Producer delivery responsibility is unavailable.")
    if current.receipt is not None:
        return current
    if not supports_peer_delivery(receiving):
        raise CollaborationUnavailable("Receiving store lacks exact peer readback.")
    raw = await receiving.read_peer_content_attempt(current.append)
    if raw is None:
        return current
    receipt = prepare_contract(PeerContentReceipt, raw, redactor=redactor)
    if receipt.status == "pending":
        if (receipt.operation_key, receipt.append_key, receipt.attempt_generation) != (
            current.append.operation_key,
            current.append.append_key,
            current.append.attempt_key.attempt_generation,
        ):
            raise CollaborationUnavailable("Pending producer delivery readback conflicts.")
        return current
    receipt = receipt.model_copy(update={"replayed": False})
    acceptance = current.operation.model_copy(
        update={"caller_key": current.operation.caller_key + ":accepted"}
    )
    terminal = prepare_contract(
        ProducerDeliveryRecord,
        current.model_copy(
            update={
                "receipt": receipt,
                "acceptance": acceptance,
            }
        ),
        redactor=redactor,
    )
    async with store._transaction(scope, write=True) as tx:
        retained = await read_delivery(tx, command, destination, redactor=redactor)
        registration = await read_output_registration(tx, command, redactor=redactor)
        if (
            retained is None
            or registration is None
            or current.operation not in registration.deliveries
        ):
            raise CollaborationUnavailable("Producer delivery ownership disappeared.")
        require_exact_contract(current.append, retained.append, redactor=redactor)
        if retained.receipt is not None:
            require_exact_contract(terminal, retained, redactor=redactor)
            return retained
        anchor = await store._anchor(tx, initialized, redactor)
        event = ProducerDeliveryAccepted(
            operation=acceptance,
            delivery=current.operation,
            receipt_commitment="sha256:"
            + sha256(contract_bytes(receipt, redactor=redactor)).hexdigest(),
            sequence=anchor.event_sequence + 1,
        )
        await _charge(
            store,
            tx,
            initialized,
            registration,
            registration,
            (retained,),
            (terminal, event),
            sequence=event.sequence,
            redactor=redactor,
        )
        await tx.put("operations", operation_key(current.operation), terminal, insert=False)
        await tx.put("operations", operation_key(acceptance), event, insert=True)
    return terminal
