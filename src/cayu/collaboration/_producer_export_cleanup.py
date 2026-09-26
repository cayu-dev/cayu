"""Owned export cleanup from durable request closure or exact delivery exclusion."""

from hashlib import sha256
from typing import Literal

from pydantic import model_validator

from cayu.collaboration._contracts import ContractValue, OperationRef
from cayu.collaboration._preparation import contract_bytes, prepare_contract, require_exact_contract
from cayu.collaboration._producer_contracts import ProducerOutputRegistration
from cayu.collaboration._producer_delivery_store import read_delivery
from cayu.collaboration._producer_export_retirement import (
    _received_delivery_retirement,
    _received_export_retirement,
)
from cayu.collaboration._producer_export_store import read_export
from cayu.collaboration._producer_store import read_output_registration
from cayu.collaboration._recipient_admission_receiver import RecipientAdmissionReceivingOwner
from cayu.collaboration._request_coordinator import _safe_request_failure
from cayu.collaboration._request_store import retained_request
from cayu.collaboration._session_export_store import (
    ExportFutureExclusion,
    ExportPreparation,
    ExportRecord,
)
from cayu.collaboration.access import CollaborationAccessContext
from cayu.collaboration.base import REQUEST_FAMILY
from cayu.collaboration.participants import CollaborationUnavailable
from cayu.collaboration.prepared_admission import NativeCommitment


class ProducerExportCleanupStatus(ContractValue):
    registration: OperationRef
    destination: OperationRef
    export: OperationRef | None
    closure: OperationRef | None = None
    delivery: OperationRef | None = None
    destination_exclusion: OperationRef | None = None
    state: Literal["excluded", "retired", "released"]
    native_commitment: NativeCommitment | None = None
    exclusion_commitment: NativeCommitment | None = None

    @model_validator(mode="after")
    def exact_basis(self):
        if (
            sum(
                value is not None
                for value in (self.closure, self.delivery, self.destination_exclusion)
            )
            != 1
        ):
            raise ValueError("Export cleanup requires one exact cleanup decision.")
        if (self.export is None) != (self.native_commitment is None):
            raise ValueError("Native export cleanup requires its exact native evidence.")
        if (self.export is None) != (self.exclusion_commitment is not None):
            raise ValueError("An absent export requires exact source exclusion evidence.")
        if self.export is None and (self.destination_exclusion is None or self.state != "excluded"):
            raise ValueError("Only source exclusion can settle an undispatched export.")
        return self


async def retire_unneeded_producer_export(app, command, destination_operation, *, context):
    """Never disclose content, append, rerun production or release source retention."""
    redactor = app._secret_redactor
    command = prepare_contract(ProducerOutputRegistration, command, redactor=redactor)
    destination_operation = prepare_contract(OperationRef, destination_operation, redactor=redactor)
    context = prepare_contract(CollaborationAccessContext, context, redactor=redactor)
    participants, requests = app._participant_coordinator, app._request_coordinator
    store, initialized = participants._ready()
    registered = requests._registration
    receiver = None if registered is None else registered.receiving_owner
    if type(receiver) is not RecipientAdmissionReceivingOwner:
        raise CollaborationUnavailable("Producer export cleanup requires its registered owner.")
    require_exact_contract(command.receiver, receiver.ref, redactor=redactor)
    require_exact_contract(command.receiver, requests.prepared_receiver_ref(), redactor=redactor)
    destination = next(
        (item for item in command.destinations if item.operation == destination_operation), None
    )
    if destination is None:
        raise CollaborationUnavailable("Producer export cleanup destination is not registered.")

    async def retire():
        participants._capability(store, initialized, mutation=True, family=REQUEST_FAMILY)
        _, grant = participants._authorize(context, "request_control")
        prepared = command.admission.prepared
        assert prepared is not None
        participants._require_refs(grant, (prepared.recipient, destination.recipient))
        async with store._transaction(initialized.owner.application_scope, write=True) as tx:
            record = await read_output_registration(tx, command, redactor=redactor)
            expected = command.admission.expected
            request = await retained_request(
                store, tx, initialized, expected.intent.request, expected.initiator, redactor
            )
            if record is None or request is None or request.producer_operation != command.operation:
                raise CollaborationUnavailable(
                    "Producer export cleanup lacks durable responsibility."
                )
            delivery = await read_delivery(tx, command, destination, redactor=redactor)
            from cayu.collaboration._producer_destination_exclusion import (
                exclude_unprepared_destination,
                read_destination_exclusion,
            )

            excluded = await read_destination_exclusion(tx, command, destination, redactor=redactor)
            if delivery is not None and (
                delivery.receipt is None
                or delivery.receipt.status != "excluded"
                or delivery.acceptance is None
            ):
                raise CollaborationUnavailable(
                    "Delivery cleanup requires exact receiving-owner exclusion."
                )
            if (
                delivery is None
                and excluded is None
                and (request.state not in ("cancelled", "expired") or request.terminal is None)
            ):
                raise CollaborationUnavailable("Producer export cleanup lacks durable closure.")
            intent = await read_export(tx, command, destination, redactor=redactor)
            if intent is None:
                if excluded is not None:
                    return ProducerExportCleanupStatus(
                        registration=command.operation,
                        destination=destination.operation,
                        export=None,
                        destination_exclusion=excluded.operation,
                        state="excluded",
                        exclusion_commitment="sha256:"
                        + sha256(contract_bytes(excluded, redactor=redactor)).hexdigest(),
                    )
                raise CollaborationUnavailable(
                    "Producer export intent is unavailable, not excluded."
                )
            if delivery is None and excluded is None:
                # Freeze the same per-destination decision used by reconciliation
                # before crossing the native-owner boundary. Request closure
                # authorizes cleanup, but must not become a competing tombstone
                # identity that a later destination exclusion could replace.
                excluded = await exclude_unprepared_destination(
                    store, tx, initialized, command, destination, redactor=redactor
                )
                assert excluded is not None
            closure_operation = None
            if delivery is None:
                if excluded is not None:
                    closure_operation = excluded.operation
                else:
                    assert request.terminal is not None
                    closure_operation = request.terminal.expected.operation
            closure = (excluded or request.terminal) if delivery is None else delivery
            if record.cleanup_ack is not None:
                from cayu.collaboration._producer_cleanup_finalization import read_finalization
                from cayu.collaboration._producer_contracts import ProducerAdmittedCleanup

                final = await read_finalization(tx, record, redactor=redactor)
                if final is None or not isinstance(record.cleanup, ProducerAdmittedCleanup):
                    raise CollaborationUnavailable("Producer cleanup finalization is unavailable.")
                evidence = record.cleanup.evidence
                retained = next(
                    (
                        item
                        for item in evidence.destinations
                        if item.destination == destination_operation
                    ),
                    None,
                )
                expected_kind = (
                    "destination_exclusion"
                    if excluded is not None
                    else "export_retirement"
                    if delivery is None
                    else "delivery"
                )
                if (
                    retained is None
                    or retained.kind != expected_kind
                    or (
                        delivery is None
                        and excluded is None
                        and (
                            evidence.terminal_kind != "closure"
                            or evidence.terminal != closure_operation
                        )
                    )
                    or (
                        delivery is not None
                        and (
                            retained.delivery != delivery.operation
                            or retained.receipt_commitment
                            != "sha256:"
                            + sha256(
                                contract_bytes(delivery.receipt, redactor=redactor)
                            ).hexdigest()
                        )
                    )
                ):
                    raise CollaborationUnavailable(
                        "Producer export cleanup finalization conflicts."
                    )
                return ProducerExportCleanupStatus(
                    registration=command.operation,
                    destination=destination.operation,
                    export=intent.operation,
                    closure=closure_operation if excluded is None else None,
                    delivery=None if delivery is None else delivery.operation,
                    destination_exclusion=None if excluded is None else excluded.operation,
                    state=retained.export_state,
                    native_commitment=retained.export_settlement_commitment,
                )
        exports = app._session_export_coordinator
        if (
            exports.mandate_ref != requests._resolver_ref
            or exports.policy_ref != destination.disclosure_policy
        ):
            raise CollaborationUnavailable("Producer export cleanup registration conflicts.")
        authority = (
            _received_export_retirement(exports, command, closure, intent)
            if delivery is None
            else _received_delivery_retirement(exports, command, delivery, intent)
        )
        native = await receiver._retire_producer_export(
            record, closure, intent, exports=exports, authority=authority
        )
        if isinstance(native, ExportFutureExclusion):
            from cayu.collaboration._producer_export_absence import expected_exclusion

            require_exact_contract(
                expected_exclusion(command, closure, intent, redactor=redactor),
                native,
                redactor=redactor,
            )
        elif isinstance(native, ExportPreparation):
            if native.state != "excluded" or not native.admission.settled:
                raise CollaborationUnavailable("Producer export preparation remains pending.")
            require_exact_contract(intent.request, native.admission.request, redactor=redactor)
        elif isinstance(native, ExportRecord):
            if (
                native.state not in ("retired", "released")
                or native.admission is None
                or not native.admission.settled
            ):
                raise CollaborationUnavailable("Producer export retirement remains pending.")
            require_exact_contract(
                intent.request, native.receipt.expected.intent.request, redactor=redactor
            )
        else:
            raise CollaborationUnavailable("Producer export cleanup evidence is not qualified.")
        return ProducerExportCleanupStatus(
            registration=command.operation,
            destination=destination.operation,
            export=intent.operation,
            closure=closure_operation if excluded is None else None,
            delivery=None if delivery is None else delivery.operation,
            destination_exclusion=None if excluded is None else excluded.operation,
            state=native.state,
            native_commitment="sha256:"
            + sha256(
                contract_bytes(native if delivery is None else native.settlement, redactor=redactor)
            ).hexdigest(),
        )

    async def owned():
        return await requests._dependency(retire)

    return await requests._observe(
        requests._owners.run(
            owned,
            key=("producer_export_cleanup", object()),
            expectation=contract_bytes(command, redactor=redactor)
            + contract_bytes(destination_operation, redactor=redactor)
            + contract_bytes(context, redactor=redactor),
            redactor=redactor,
            failure_snapshot=lambda error: _safe_request_failure(error, redactor),
        )
    )
