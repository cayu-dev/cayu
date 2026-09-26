"""Content-free source fence, elected atomically against first delivery preparation.

This is not receiving-owner evidence: it proves that this producer never handed
this destination to the receiver. Already prepared deliveries use peer exclusion.
"""

from hashlib import sha256
from typing import Literal

from cayu.collaboration._contracts import (
    CollaborationConflict,
    ContractValue,
    Generation,
    OperationRef,
)
from cayu.collaboration._preparation import contract_bytes, prepare_contract
from cayu.collaboration._request_store import operation_key
from cayu.collaboration.prepared_admission import NativeCommitment


class ProducerDestinationExclusion(ContractValue):
    mode: Literal["producer_destination_exclusion"] = "producer_destination_exclusion"
    operation: OperationRef
    registration: OperationRef
    registration_commitment: NativeCommitment
    destination: OperationRef
    sequence: Generation


def exclusion_operation(destination, redactor):
    return destination.operation.model_copy(
        update={
            "caller_key": "producer-excluded:"
            + sha256(contract_bytes(destination.operation, redactor=redactor)).hexdigest()
        }
    )


async def read_destination_exclusion(tx, command, destination, *, redactor):
    operation = exclusion_operation(destination, redactor)
    raw = await tx.get("operations", operation_key(operation))
    if raw is None:
        return None
    record = prepare_contract(ProducerDestinationExclusion, raw, redactor=redactor)
    if (
        record.operation != operation
        or record.registration != command.operation
        or record.destination != destination.operation
        or destination not in command.destinations
        or record.registration_commitment
        != "sha256:" + sha256(contract_bytes(command, redactor=redactor)).hexdigest()
    ):
        raise CollaborationConflict("Producer destination exclusion conflicts.")
    return record


async def require_destination_live(tx, command, destination, *, redactor):
    if await read_destination_exclusion(tx, command, destination, redactor=redactor) is not None:
        raise CollaborationConflict("Producer destination is durably excluded.")


async def exclude_unprepared_destination(store, tx, initialized, command, destination, *, redactor):
    from cayu.collaboration._producer_delivery_store import read_delivery
    from cayu.collaboration._producer_export_store import _charge
    from cayu.collaboration._producer_store import read_output_registration
    from cayu.collaboration.participants import CollaborationUnavailable

    registration = await read_output_registration(tx, command, redactor=redactor)
    if registration is None or registration.cleanup is not None:
        raise CollaborationUnavailable("Producer responsibility is unavailable for exclusion.")
    prior = await read_destination_exclusion(tx, command, destination, redactor=redactor)
    if prior is not None:
        return prior
    if await read_delivery(tx, command, destination, redactor=redactor) is not None:
        return None
    anchor = await store._anchor(tx, initialized, redactor)
    record = ProducerDestinationExclusion(
        operation=exclusion_operation(destination, redactor),
        registration=command.operation,
        registration_commitment="sha256:"
        + sha256(contract_bytes(command, redactor=redactor)).hexdigest(),
        destination=destination.operation,
        sequence=anchor.event_sequence + 1,
    )
    updated = registration.model_copy(
        update={"exclusions": (*registration.exclusions, record.operation)}
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
    await tx.put("operations", operation_key(record.operation), record, insert=True)
    return record
