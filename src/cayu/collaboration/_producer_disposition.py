"""Service retained request closure without renewing producer execution authority."""

from typing import Literal

from pydantic import model_validator

from cayu.collaboration._contracts import ContractValue, OperationRef
from cayu.collaboration._preparation import contract_bytes, prepare_contract, require_exact_contract
from cayu.collaboration._producer_contracts import ProducerOutputRegistration
from cayu.collaboration._producer_store import read_output_registration
from cayu.collaboration._recipient_admission_receiver import RecipientAdmissionReceivingOwner
from cayu.collaboration._request_coordinator import _safe_request_failure
from cayu.collaboration._request_store import retained_request
from cayu.collaboration.access import CollaborationAccessContext
from cayu.collaboration.base import REQUEST_FAMILY
from cayu.collaboration.participants import CollaborationUnavailable
from cayu.collaboration.prepared_admission import NativeCommitment
from cayu.runtime._producer_release import ProducerNativeRelease
from cayu.runtime._producer_stop import NativeProducerStopReceipt, _received_producer_closure
from cayu.runtime.session_steering import SessionSteeringConflict


class ProducerDispositionStatus(ContractValue):
    """Closure servicing observation; no form claims effect quiescence or settlement."""

    registration: OperationRef
    control: OperationRef
    disposition: Literal["stop_accepted", "invocation_released"]
    native_stop: NativeProducerStopReceipt | None = None
    native_release_commitment: NativeCommitment | None = None

    @model_validator(mode="after")
    def exact_stop(self):
        if (self.disposition == "stop_accepted") != (self.native_stop is not None):
            raise ValueError("Producer disposition requires exact native stop evidence.")
        if (self.disposition == "invocation_released") != (
            self.native_release_commitment is not None
        ):
            raise ValueError("Producer disposition requires exact native release evidence.")
        if self.native_stop is not None and self.native_stop.registration != self.registration:
            raise ValueError("Producer stop belongs to another registration.")
        return self


async def _service_native_stop(receiver, record, closure, *, redactor):
    """Shared closure/recovery path; caller authenticates source closure first."""
    command = record.command
    authority = _received_producer_closure(record, closure)
    try:
        receipt = await receiver._request_producer_stop(record, closure, authority=authority)
    except SessionSteeringConflict:
        # Native completion can win the stop-acceptance transaction. Neither
        # terminal status nor this conflict proves release. The fixed native
        # owner must authenticate the original invocation's durable release
        # in one backend snapshot; a replacement epoch remains fenced.
        # A released human pause is still resumable. If it won this race,
        # retain closure responsibility for exact retry rather than acknowledging
        # release as completion without installing the paused-stop fence.
        if await receiver._read_producer_output(record) is None:
            raise CollaborationUnavailable("Producer stop remains unresolved.") from None
        released = prepare_contract(
            ProducerNativeRelease,
            await receiver._read_producer_release(record),
            redactor=redactor,
        )
        require_exact_contract(command, released.registration, redactor=redactor)
        return ProducerDispositionStatus(
            registration=command.operation,
            control=closure.expected.operation,
            disposition="invocation_released",
            native_release_commitment=released.release_commitment,
        )
    return ProducerDispositionStatus(
        registration=command.operation,
        control=closure.expected.operation,
        disposition="stop_accepted",
        native_stop=receipt,
    )


async def service_closed_producer(app, command, *, context):
    """Private owner entrance for already-closed, launch-claimed producers.

    The retained closure and frozen disposition are the durable stop intent;
    the output permit remains discoverable and capacity-counted throughout.
    The native owner records acceptance under one deterministic interaction key.
    Lost acknowledgement can therefore be reconciled without another stop scope.
    """
    redactor = app._secret_redactor
    command = prepare_contract(ProducerOutputRegistration, command, redactor=redactor)
    context = prepare_contract(CollaborationAccessContext, context, redactor=redactor)
    coordinator = app._request_coordinator
    participants = app._participant_coordinator
    registered = coordinator._registration
    receiver = None if registered is None else registered.receiving_owner
    if type(receiver) is not RecipientAdmissionReceivingOwner:
        raise CollaborationUnavailable("Producer disposition requires its registered native owner.")
    require_exact_contract(command.receiver, receiver.ref, redactor=redactor)
    require_exact_contract(command.receiver, coordinator.prepared_receiver_ref(), redactor=redactor)
    store, initialized = participants._ready()

    async def service():
        participants._capability(store, initialized, mutation=True, family=REQUEST_FAMILY)
        _, grant = participants._authorize(context, "request_control")
        prepared = command.admission.prepared
        assert prepared is not None
        participants._require_refs(grant, (prepared.recipient,))
        async with store._transaction(initialized.owner.application_scope, write=False) as tx:
            record = await read_output_registration(tx, command, redactor=redactor)
            expected = command.admission.expected
            request = await retained_request(
                store, tx, initialized, expected.intent.request, expected.initiator, redactor
            )
            if (
                record is None
                or record.state != "launch_claimed"
                or request is None
                or request.state not in ("cancelled", "expired")
                or request.terminal is None
                or request.producer_operation != command.operation
            ):
                raise CollaborationUnavailable("Producer closure responsibility is unavailable.")
            closure = request.terminal
            require_exact_contract(expected, closure.expected.intent.expected, redactor=redactor)
        return await _service_native_stop(receiver, record, closure, redactor=redactor)

    async def owned():
        return await coordinator._dependency(service)

    return await coordinator._observe(
        coordinator._owners.run(
            owned,
            key=("producer_disposition", object()),
            expectation=contract_bytes(command, redactor=redactor)
            + contract_bytes(context, redactor=redactor),
            redactor=redactor,
            failure_snapshot=lambda error: _safe_request_failure(error, redactor),
        )
    )
