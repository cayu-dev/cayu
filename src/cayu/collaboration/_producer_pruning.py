"""Exact bounded inventory of settled producer records for request reclamation.

The request retirement owner captures this inventory before removing any source
evidence. A cursor authorizes only those exact bytes; it is not cleanup evidence
and cannot turn an unresolved producer into a reclaimable one.
"""

from hashlib import sha256

from pydantic import Field, StrictInt

from cayu.collaboration._contracts import ContractValue, Identifier
from cayu.collaboration._permit_store import prepare_permit_record, registered_receipt
from cayu.collaboration._permits import PermitSettlement, PermitSnapshot
from cayu.collaboration._preparation import contract_bytes, prepare_contract
from cayu.collaboration._producer_cleanup_finalization import ProducerCleanupFinalized
from cayu.collaboration._producer_contracts import (
    ProducerAdmittedCleanup,
    ProducerCleanupRecord,
    ProducerCompletionRecord,
    ProducerDeliveryAccepted,
    ProducerDeliveryIndex,
    ProducerDeliveryRecord,
    ProducerExportPublished,
    ProducerExportRecord,
    ProducerLaunchDecision,
    ProducerOutputRecord,
    ProducerRegistrationEvent,
    ProducerRequestIndex,
)
from cayu.collaboration._producer_delivery_store import acceptance_index_operation
from cayu.collaboration._producer_destination_exclusion import ProducerDestinationExclusion
from cayu.collaboration._producer_output_failure import ProducerExportRejected
from cayu.collaboration._producer_store import read_request_output, request_output_index
from cayu.collaboration._request_store import operation_key
from cayu.collaboration.participants import CollaborationUnavailable
from cayu.collaboration.prepared_admission import NativeCommitment

MAX_PRODUCER_PRUNING_RECORDS = 128

_SCHEMAS = {
    "producer_export_rejected": ProducerExportRejected,
    "producer_destination_exclusion": ProducerDestinationExclusion,
    "producer_output_record": ProducerOutputRecord,
    "producer_request_index": ProducerRequestIndex,
    "producer_output_registered": ProducerRegistrationEvent,
    "producer_launch_decision": ProducerLaunchDecision,
    "producer_cleanup": ProducerCleanupRecord,
    "producer_admitted_cleanup": ProducerAdmittedCleanup,
    "producer_completion": ProducerCompletionRecord,
    "producer_export": ProducerExportRecord,
    "producer_export_published": ProducerExportPublished,
    "producer_delivery": ProducerDeliveryRecord,
    "producer_delivery_accepted": ProducerDeliveryAccepted,
    "producer_delivery_index": ProducerDeliveryIndex,
    "producer_cleanup_finalized": ProducerCleanupFinalized,
}


def producer_record(raw, *, redactor):
    schema = _SCHEMAS.get(raw.get("mode")) if isinstance(raw, dict) else None
    return None if schema is None else prepare_contract(schema, raw, redactor=redactor)


class ProducerPruningItem(ContractValue):
    # All records share the request's complete namespace identity. Do not repeat
    # that large tuple for every destination in the bounded cursor.
    caller_key: Identifier
    commitment: NativeCommitment
    events: StrictInt = Field(ge=0, le=1)


def item_for(value, *, redactor):
    operation = (
        value.command.operation if isinstance(value, ProducerOutputRecord) else value.operation
    )
    return ProducerPruningItem(
        caller_key=operation.caller_key,
        commitment="sha256:" + sha256(contract_bytes(value, redactor=redactor)).hexdigest(),
        events=int(hasattr(value, "sequence")),
    )


async def producer_pruning_request(tx, raw, *, redactor):
    """Route any recognized child to its retained root; never prune it alone."""
    value = producer_record(raw, redactor=redactor)
    if value is None:
        return None
    if isinstance(value, ProducerOutputRecord):
        return value.command.admission.expected
    if isinstance(value, ProducerCompletionRecord):
        return value.output.registration.admission.expected
    if isinstance(value, (ProducerExportPublished, ProducerDeliveryAccepted)):
        parent = value.export if isinstance(value, ProducerExportPublished) else value.delivery
        parent_raw = await tx.get("operations", operation_key(parent))
        parent_value = producer_record(parent_raw, redactor=redactor)
        if not isinstance(parent_value, (ProducerExportRecord, ProducerDeliveryRecord)):
            raise CollaborationUnavailable("Producer pruning parent is unavailable.")
        registration = parent_value.registration
    else:
        registration = value.registration
    root = prepare_contract(
        ProducerOutputRecord,
        await tx.get("operations", operation_key(registration)),
        redactor=redactor,
    )
    if root.command.operation != registration:
        raise CollaborationUnavailable("Producer pruning root conflicts.")
    return root.command.admission.expected


async def producer_pruning_inventory(tx, snapshot, *, redactor):
    record = await read_request_output(tx, snapshot.receipt.expected, redactor=redactor)
    if record is None or record.command.operation != snapshot.producer_operation:
        raise CollaborationUnavailable("Producer pruning lacks its registered root.")
    if record.reserved_operations or record.reserved_events or record.reserved_bytes:
        raise CollaborationUnavailable("Producer pruning retains reserved responsibility.")
    settlement = record.cleanup_ack
    if record.cleanup is None or settlement is None or settlement != snapshot.producer_settlement:
        raise CollaborationUnavailable("Producer pruning requires exact final settlement.")
    # Both collaboration obligations must have positive durable settlement, not
    # merely an empty pending index or a native terminal status.
    for permit in (record.permit, snapshot.permit):
        if await registered_receipt(tx, permit, redactor) is None:
            raise CollaborationUnavailable("Producer pruning permit is unavailable.")
        settled = prepare_permit_record(
            await tx.get("operations", operation_key(permit.intent.request.settlement_operation)),
            redactor,
        )
        state = prepare_contract(
            PermitSnapshot,
            await tx.get("permits", operation_key(permit.operation)),
            redactor=redactor,
        )
        if (
            not isinstance(settled, PermitSettlement)
            or settled.expected != permit
            or state.state != "settled"
        ):
            raise CollaborationUnavailable("Producer pruning responsibility is unresolved.")
    operations = list(record.exclusions)
    # Dependent publication/acceptance records precede their parents so every
    # remaining row can still route to the root during a later pruning batch.
    for operation in record.deliveries:
        delivery = prepare_contract(
            ProducerDeliveryRecord,
            await tx.get("operations", operation_key(operation)),
            redactor=redactor,
        )
        if delivery.acceptance is not None:
            operations.append(delivery.acceptance)
        operations.append(
            acceptance_index_operation(
                record.command.operation,
                delivery.source_receipt.expected.intent.request.ref,
                redactor,
            )
        )
        operations.append(operation)
    for operation in record.exports:
        exported = prepare_contract(
            ProducerExportRecord,
            await tx.get("operations", operation_key(operation)),
            redactor=redactor,
        )
        if exported.publication is not None:
            operations.append(exported.publication)
        if exported.rejection is not None:
            operations.append(exported.rejection)
        operations.append(operation)
    operations.extend(
        item
        for item in (
            record.cleanup_ack,
            record.cleanup.operation,
            record.completion,
            record.launch.operation if record.launch is not None else None,
            record.event.operation,
            request_output_index(record.command, redactor),
            record.command.operation,
        )
        if item is not None
    )
    if len(operations) > MAX_PRODUCER_PRUNING_RECORDS or len(set(operations)) != len(operations):
        raise CollaborationUnavailable("Producer pruning inventory is not bounded and unique.")
    inventory = []
    namespace = snapshot.receipt.expected.operation
    for operation in operations:
        if operation != namespace.model_copy(update={"caller_key": operation.caller_key}):
            raise CollaborationUnavailable("Producer pruning crosses its namespace.")
        value = producer_record(
            await tx.get("operations", operation_key(operation)), redactor=redactor
        )
        if value is None:
            raise CollaborationUnavailable("Producer pruning inventory is incomplete.")
        item = item_for(value, redactor=redactor)
        if item.caller_key != operation.caller_key:
            raise CollaborationUnavailable("Producer pruning operation conflicts.")
        inventory.append(item)
    return tuple(inventory)


async def prune_producer_items(tx, command, items, *, redactor):
    from cayu.collaboration._history_references import history_references
    from cayu.collaboration._retention_store import release_unused_history

    released = events = 0
    references = []
    for item in items:
        operation = command.operation.model_copy(update={"caller_key": item.caller_key})
        value = producer_record(
            await tx.get("operations", operation_key(operation)), redactor=redactor
        )
        if value is None or item_for(value, redactor=redactor) != item:
            raise CollaborationUnavailable("Producer pruning inventory changed.")
        await tx.delete("operations", operation_key(operation))
        released += len(contract_bytes(value, redactor=redactor))
        events += item.events
        references.extend(history_references(value))
    released += await release_unused_history(tx, tuple(references), redactor)
    return released, events
