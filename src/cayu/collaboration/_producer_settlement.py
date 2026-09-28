"""Content-free owner evidence for cleanup; gathering evidence does not release pins."""

from hashlib import sha256

from cayu.collaboration._preparation import contract_bytes, prepare_contract, require_exact_contract
from cayu.collaboration._producer_contracts import (
    ProducerCompletionRecord,
    ProducerDestinationSettlement,
    ProducerOutputRegistration,
    ProducerSettlementEvidence,
)
from cayu.collaboration._producer_delivery_store import read_delivery
from cayu.collaboration._producer_export_store import read_export
from cayu.collaboration._producer_outcome_store import require_producer_outcome
from cayu.collaboration._producer_store import read_output_registration
from cayu.collaboration._producer_terminal import cleanup_destinations, producer_terminal
from cayu.collaboration._recipient_admission_receiver import RecipientAdmissionReceivingOwner
from cayu.collaboration._request_store import operation_key, retained_request
from cayu.collaboration.participants import CollaborationUnavailable


async def read_producer_settlement(app, command):
    """Internal registered-owner readback. No caller-supplied settlement receipts.

    Foreign reads run outside collaboration transactions. Every destination must
    already be durably settled; partial success leaves the whole owner pending.
    This does not yet establish descendant quiescence or mutate responsibility.
    """
    redactor = app._secret_redactor
    command = prepare_contract(ProducerOutputRegistration, command, redactor=redactor)
    coordinator = app._request_coordinator
    registered = coordinator._registration
    receiver = None if registered is None else registered.receiving_owner
    if type(receiver) is not RecipientAdmissionReceivingOwner:
        raise CollaborationUnavailable("Producer settlement requires its registered owner.")
    require_exact_contract(command.receiver, coordinator.prepared_receiver_ref(), redactor=redactor)
    require_exact_contract(command.receiver, receiver.ref, redactor=redactor)
    store, initialized = app._participant_coordinator._ready()
    async with store._transaction(initialized.owner.application_scope, write=False) as tx:
        record = await read_output_registration(tx, command, redactor=redactor)
        if record is None or record.completion is None:
            raise CollaborationUnavailable("Producer completion remains unresolved.")
        completion = prepare_contract(
            ProducerCompletionRecord,
            await tx.get("operations", operation_key(record.completion)),
            redactor=redactor,
        )
        expected = command.admission.expected
        request = await retained_request(
            store, tx, initialized, expected.intent.request, expected.initiator, redactor
        )
        terminal_kind, terminal = producer_terminal(request, command, redactor=redactor)
        if terminal_kind == "outcome":
            assert request is not None and request.outcome is not None
            await require_producer_outcome(tx, request, request.outcome.command, redactor=redactor)
        deliveries = []
        retirements = []
        excluded_destinations = []
        for destination in cleanup_destinations(record, completion, request, redactor=redactor):
            delivery = await read_delivery(tx, command, destination, redactor=redactor)
            from cayu.collaboration._producer_destination_exclusion import (
                read_destination_exclusion,
            )

            exclusion = await read_destination_exclusion(
                tx, command, destination, redactor=redactor
            )
            if exclusion is not None:
                if delivery is not None:
                    raise CollaborationUnavailable("Producer exclusion conflicts with delivery.")
                intent = await read_export(tx, command, destination, redactor=redactor)
                excluded_destinations.append((destination, exclusion, intent))
                continue
            if delivery is None and terminal_kind == "closure":
                intent = await read_export(tx, command, destination, redactor=redactor)
                if intent is None:
                    raise CollaborationUnavailable("Producer export responsibility is unavailable.")
                retirements.append((destination, intent))
                continue
            if delivery is None or delivery.receipt is None or delivery.acceptance is None:
                raise CollaborationUnavailable("Producer destination remains unresolved.")
            deliveries.append((destination, delivery))

    native = await receiver._read_producer_release(record)
    accounting = await receiver._read_producer_budget_settlement(record)
    require_exact_contract(command, native.registration, redactor=redactor)
    if (
        accounting.registration != command.operation
        or accounting.native_release_commitment != native.release_commitment
    ):
        raise CollaborationUnavailable("Producer accounting and native release conflict.")

    def commitment(value):
        return "sha256:" + sha256(contract_bytes(value, redactor=redactor)).hexdigest()

    destinations = []
    for destination, exclusion, intent in excluded_destinations:
        retired = (
            exclusion
            if intent is None
            else await receiver._read_producer_export_retirement(record, exclusion, intent)
        )
        destinations.append(
            ProducerDestinationSettlement(
                destination=destination.operation,
                kind="destination_exclusion",
                export_settlement_commitment=commitment(retired),
                export_state="excluded" if intent is None else retired.state,
            )
        )
    for destination, delivery in deliveries:
        settled = await receiver._read_producer_export_settlement(record, delivery)
        destinations.append(
            ProducerDestinationSettlement(
                destination=destination.operation,
                delivery=delivery.operation,
                receipt_commitment=commitment(delivery.receipt),
                export_settlement_commitment=commitment(settled),
                export_state="released" if settled.request.mode == "release" else "retired",
            )
        )
    for destination, intent in retirements:
        assert request is not None and request.terminal is not None
        settled = await receiver._read_producer_export_retirement(record, request.terminal, intent)
        destinations.append(
            ProducerDestinationSettlement(
                destination=destination.operation,
                kind="export_retirement",
                export_settlement_commitment=commitment(settled),
                export_state=settled.state,
            )
        )
    order = {item.operation: index for index, item in enumerate(command.destinations)}
    destinations.sort(key=lambda item: order[item.destination])
    return ProducerSettlementEvidence(
        registration=command.operation,
        registration_commitment=commitment(command),
        completion=completion.operation,
        completion_commitment=commitment(completion),
        terminal=terminal,
        terminal_kind=terminal_kind,
        native_release_commitment=native.release_commitment,
        native_release_run_epoch=native.run_epoch,
        budget_settlement_commitment=commitment(accounting),
        destinations=tuple(destinations),
    )


async def prepare_producer_cleanup(app, command, *, wait_for_settlement=False):
    """Retain authenticated cleanup evidence without releasing responsibility.

    Exact acknowledgement-loss recovery uses source-owned acceptance. Observation
    cancellation leaves the worker owned; it cannot discard a late source commit.
    """
    from cayu.collaboration._producer_cleanup_store import (
        _received_cleanup,
        accept_admitted_cleanup,
    )
    from cayu.collaboration._producer_contracts import ProducerAdmittedCleanup
    from cayu.collaboration._request_coordinator import _safe_request_failure
    from cayu.collaboration.base import REQUEST_FAMILY

    redactor = app._secret_redactor
    command = prepare_contract(ProducerOutputRegistration, command, redactor=redactor)
    coordinator = app._request_coordinator
    store, initialized = app._participant_coordinator._ready()
    app._participant_coordinator._capability(
        store, initialized, mutation=True, family=REQUEST_FAMILY
    )

    async def prepare():
        registered = coordinator._registration
        receiver = None if registered is None else registered.receiving_owner
        if type(receiver) is not RecipientAdmissionReceivingOwner:
            raise CollaborationUnavailable("Producer cleanup requires its registered owner.")
        require_exact_contract(command.receiver, receiver.ref, redactor=redactor)
        require_exact_contract(
            command.receiver, coordinator.prepared_receiver_ref(), redactor=redactor
        )
        async with store._transaction(initialized.owner.application_scope, write=False) as tx:
            record = await read_output_registration(tx, command, redactor=redactor)
            if record is not None and isinstance(record.cleanup, ProducerAdmittedCleanup):
                return record.cleanup
        evidence = await read_producer_settlement(app, command)
        authority = _received_cleanup(command, evidence, redactor)
        async with store._transaction(initialized.owner.application_scope, write=True) as tx:
            return await accept_admitted_cleanup(
                store, tx, initialized, command, evidence, authority=authority, redactor=redactor
            )

    async def owned():
        return await coordinator._dependency(prepare)

    return await coordinator._observe(
        coordinator._owners.run(
            owned,
            key=("producer_cleanup_preparation", object()),
            expectation=contract_bytes(command, redactor=redactor),
            redactor=redactor,
            failure_snapshot=lambda error: _safe_request_failure(error, redactor),
            wait_for_settlement=wait_for_settlement,
        )
    )
