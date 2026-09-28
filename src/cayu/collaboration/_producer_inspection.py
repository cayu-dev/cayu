"""Content-free source-owned producer evidence for host servicing.

References to prepared exports/deliveries do not prove publication, append, or
provider exposure. Launch election likewise does not prove active execution.
The receiving owners remain authoritative for those separate outcomes.
"""

from typing import Literal

from pydantic import Field

from cayu.collaboration._contracts import ContractValue, OperationRef
from cayu.collaboration._producer_bounds import MAX_OUTPUT_DESTINATIONS
from cayu.collaboration._producer_recovery import ProducerOutputRecovery


class ProducerDestinationInspection(ContractValue):
    destination: OperationRef
    export: Literal["prepared", "published", "rejected"] | None
    delivery: Literal["pending", "appended", "excluded"] | None
    export_cleanup: Literal["released", "retired", "excluded"] | None = None


class ProducerOutputInspection(ContractValue):
    recovery: ProducerOutputRecovery
    state: Literal["registered", "launch_claimed", "excluded"]
    completion: OperationRef | None
    cleanup: OperationRef | None
    cleanup_ack: OperationRef | None
    exports: tuple[OperationRef, ...] = Field(max_length=MAX_OUTPUT_DESTINATIONS)
    deliveries: tuple[OperationRef, ...] = Field(max_length=MAX_OUTPUT_DESTINATIONS)
    exclusions: tuple[OperationRef, ...] = Field(max_length=MAX_OUTPUT_DESTINATIONS)
    request_state: Literal["open", "answered", "failed", "declined", "cancelled", "expired"]
    completion_disposition: (
        Literal["answer", "oversized", "unsupported", "empty", "failed", "stopped"] | None
    )
    answer_destination: OperationRef | None
    destinations: tuple[ProducerDestinationInspection, ...] = Field(
        min_length=1, max_length=MAX_OUTPUT_DESTINATIONS
    )


async def inspect_record(tx, store, initialized, record, *, redactor):
    """Project one authenticated source snapshot, never call a foreign owner."""
    from hashlib import sha256

    from cayu.collaboration._preparation import contract_bytes, prepare_contract
    from cayu.collaboration._producer_contracts import ProducerCompletionRecord
    from cayu.collaboration._producer_delivery_store import read_delivery
    from cayu.collaboration._producer_destination_exclusion import read_destination_exclusion
    from cayu.collaboration._producer_export_store import read_export
    from cayu.collaboration._request_store import operation_key, retained_request
    from cayu.collaboration.participants import CollaborationUnavailable
    from cayu.collaboration.requests import ProducerOutcomeCommand

    command = record.command
    expected = command.admission.expected
    request = await retained_request(
        store, tx, initialized, expected.intent.request, expected.initiator, redactor
    )
    if request is None or request.producer_operation != command.operation:
        raise CollaborationUnavailable("Producer inspection lacks its exact request attachment.")
    completed = None
    if record.completion is not None:
        completed = prepare_contract(
            ProducerCompletionRecord,
            await tx.get("operations", operation_key(record.completion)),
            redactor=redactor,
        )
    answer_destination = None
    if request.outcome is not None:
        outcome = request.outcome.command
        if not isinstance(outcome, ProducerOutcomeCommand) or outcome.producer != command.operation:
            raise CollaborationUnavailable("Producer inspection outcome belongs to another owner.")
        answer_destination = outcome.destination
    destinations = []
    for destination in command.destinations:
        exported = await read_export(tx, command, destination, redactor=redactor)
        delivered = await read_delivery(tx, command, destination, redactor=redactor)
        excluded = await read_destination_exclusion(tx, command, destination, redactor=redactor)
        if excluded is not None and delivered is not None:
            raise CollaborationUnavailable("Producer destination decisions conflict.")
        destinations.append(
            ProducerDestinationInspection(
                destination=destination.operation,
                export=None
                if exported is None
                else "rejected"
                if exported.rejection is not None
                else exported.state,
                delivery="excluded"
                if excluded is not None
                else None
                if delivered is None
                else "pending"
                if delivered.receipt is None
                else delivered.receipt.status,
            )
        )
    return ProducerOutputInspection(
        recovery=ProducerOutputRecovery(
            registration=command.operation,
            registration_commitment="sha256:"
            + sha256(contract_bytes(command, redactor=redactor)).hexdigest(),
        ),
        state=record.state,
        completion=record.completion,
        cleanup=None if record.cleanup is None else record.cleanup.operation,
        cleanup_ack=record.cleanup_ack,
        exports=record.exports,
        deliveries=record.deliveries,
        exclusions=record.exclusions,
        request_state=request.state,
        completion_disposition=None if completed is None else completed.output.disposition,
        answer_destination=answer_destination,
        destinations=tuple(destinations),
    )


async def inspect_producer_output(app, expected, *, context, wait_for_settlement=False):
    """Exact authenticated readback, not a launch or disclosure grant."""
    from cayu.collaboration._contracts import (
        CollaborationConflict,
        CollaborationContractError,
        ExactConflict,
        ExactMatch,
        ExactUnavailable,
    )
    from cayu.collaboration._preparation import prepare_contract
    from cayu.collaboration._producer_readback import _lookup_producer
    from cayu.collaboration.access import CollaborationAccessContext
    from cayu.collaboration.exports import SessionExportUnavailable
    from cayu.collaboration.participants import CollaborationUnavailable

    context = prepare_contract(CollaborationAccessContext, context, redactor=app._secret_redactor)
    found = await _lookup_producer(
        app,
        expected,
        context=context,
        completion=False,
        inspection=True,
        wait_for_settlement=wait_for_settlement,
    )
    if not isinstance(found, ExactMatch) or found.receipt.cleanup_ack is not None:
        return found
    if not any(item.delivery in ("appended", "excluded") for item in found.receipt.destinations):
        return found
    try:
        return await _inspect_native_settlements(
            app, found.receipt, context=context, wait_for_settlement=wait_for_settlement
        )
    except CollaborationConflict:
        return ExactConflict()
    except (CollaborationContractError, CollaborationUnavailable, SessionExportUnavailable):
        return ExactUnavailable()


async def _inspect_native_settlements(app, snapshot, *, context, wait_for_settlement=False):
    """Foreign readback outside source transactions; never infer settlement from absence."""
    from cayu.collaboration._contracts import ExactMatch
    from cayu.collaboration._preparation import prepare_contract, require_exact_contract
    from cayu.collaboration._producer_delivery_store import read_delivery
    from cayu.collaboration._producer_destination_exclusion import read_destination_exclusion
    from cayu.collaboration._producer_export_store import read_export
    from cayu.collaboration._producer_readback import lookup_producer_registration
    from cayu.collaboration._producer_store import read_output_registration
    from cayu.collaboration._recipient_admission_receiver import RecipientAdmissionReceivingOwner
    from cayu.collaboration.participants import CollaborationUnavailable

    found = await lookup_producer_registration(
        app, snapshot.recovery, context=context, wait_for_settlement=wait_for_settlement
    )
    if not isinstance(found, ExactMatch):
        raise CollaborationUnavailable("Producer inspection registration is unavailable.")
    command = found.receipt
    configured = app._request_coordinator._registration
    receiver = None if configured is None else configured.receiving_owner
    if type(receiver) is not RecipientAdmissionReceivingOwner:
        raise CollaborationUnavailable("Producer inspection requires its native receiver.")
    redactor = app._secret_redactor
    require_exact_contract(command.receiver, receiver.ref, redactor=redactor)
    store, initialized = app._participant_coordinator._ready()
    entries = []
    async with store._transaction(initialized.owner.application_scope, write=False) as tx:
        record = await read_output_registration(tx, command, redactor=redactor)
        if record is None:
            raise CollaborationUnavailable("Producer inspection lost source responsibility.")
        for item, destination in zip(snapshot.destinations, command.destinations, strict=True):
            if item.destination != destination.operation:
                raise CollaborationUnavailable("Producer inspection destination order changed.")
            if item.delivery not in ("appended", "excluded"):
                entries.append((item, None, None, None))
                continue
            entries.append(
                (
                    item,
                    await read_delivery(tx, command, destination, redactor=redactor),
                    await read_destination_exclusion(tx, command, destination, redactor=redactor),
                    await read_export(tx, command, destination, redactor=redactor),
                )
            )
    projected = []
    for item, delivery, exclusion, exported in entries:
        state = None
        if delivery is not None:
            settled = await receiver._read_producer_export_settlement(
                record, delivery, allow_pending=True
            )
            if settled is not None:
                state = "released" if settled.request.mode == "release" else "retired"
        elif exclusion is not None:
            if exported is None:
                state = "excluded"
            else:
                retired = await receiver._read_producer_export_retirement(
                    record, exclusion, exported, allow_pending=True
                )
                if retired is not None:
                    state = retired.state
        projected.append(item.model_copy(update={"export_cleanup": state}))
    current = await lookup_producer_registration(
        app, snapshot.recovery, context=context, wait_for_settlement=wait_for_settlement
    )
    if not isinstance(current, ExactMatch) or current.receipt != command:
        raise CollaborationUnavailable("Producer inspection authority changed.")
    result = prepare_contract(
        ProducerOutputInspection,
        snapshot.model_copy(update={"destinations": tuple(projected)}),
        redactor=redactor,
    )
    return ExactMatch[ProducerOutputInspection](receipt=result)
