"""Read-only preparation of producer identity; never registration or execution."""

from cayu.collaboration._contracts import CollaborationContractError, ExactMatch
from cayu.collaboration._preparation import contract_bytes, prepare_contract, require_exact_contract
from cayu.collaboration._producer_contracts import (
    ProducerOutputProposal,
    ProducerOutputRegistration,
)
from cayu.collaboration._producer_registration import (
    _native_execution_commitment,
    _snapshot_execution,
)
from cayu.collaboration._recipient_admission_receiver import RecipientAdmissionReceivingOwner
from cayu.collaboration._request_coordinator import _initiator, _safe_request_failure
from cayu.collaboration.access import CollaborationAccessDenied
from cayu.collaboration.mandates import MandateAccessContext
from cayu.collaboration.participants import CollaborationUnavailable
from cayu.collaboration.prepared_admission import FreshRecipientAdmissionTarget
from cayu.collaboration.request_access import RequestReceivingAuthorization
from cayu.runtime._producer_output_store import preflight_native_output


async def prepare_producer_output(app, proposal, execution, *, context, wait_for_settlement=False):
    """Derive the native commitment from authenticated preparation, not caller hashes.

    The returned immutable command is still untrusted at registration and launch.
    Those owners repeat their own current-authority and exact native checks.
    """
    redactor = app._secret_redactor
    proposal = prepare_contract(ProducerOutputProposal, proposal, redactor=redactor)
    context = prepare_contract(MandateAccessContext, context, redactor=redactor)
    execution = _snapshot_execution(execution)
    admission = proposal.admission
    if admission.expected.intent.request.cancellation != "stop":
        raise CollaborationContractError(
            "Request-only production requires a stop disposition; independent work is not qualified."
        )
    prepared = admission.prepared
    if (
        prepared is None
        or admission.decision != "fresh"
        or not isinstance(prepared.target, FreshRecipientAdmissionTarget)
        or prepared.target.resources
    ):
        raise CollaborationAccessDenied(
            "Producer preparation requires resource-free FRESH admission."
        )
    require_exact_contract(admission.initiator, _initiator(context), redactor=redactor)
    if context.participant != prepared.recipient:
        raise CollaborationAccessDenied("Producer preparation names another participant.")
    coordinator = app._request_coordinator
    registration = coordinator._registration
    receiver = None if registration is None else registration.receiving_owner
    if type(receiver) is not RecipientAdmissionReceivingOwner:
        raise CollaborationUnavailable("Producer preparation requires the native receiver.")
    receiver_ref = coordinator.prepared_receiver_ref()
    require_exact_contract(prepared.receiver, receiver_ref, redactor=redactor)
    if not app.session_store._supports_producer_attachment_protocol():
        raise CollaborationUnavailable("Native producer attachment is not qualified.")

    async def prepare():
        reader = app.collaboration_admission_reader()
        if wait_for_settlement:
            from cayu.collaboration._admission_reader import RegisteredRequestAdmissionReader

            if type(reader) is not RegisteredRequestAdmissionReader:
                raise CollaborationUnavailable(
                    "Retained preparation requires the native admission reader."
                )
            found = await reader._lookup_owned(admission, context=context)
        else:
            found = await reader.lookup(admission, context=context)
        if not isinstance(found, ExactMatch) or found.receipt.state != "admitted":
            raise CollaborationUnavailable("Producer admission evidence is unavailable.")
        require_exact_contract(admission, found.receipt.command, redactor=redactor)
        async with receiver.acquire(admission, context=context) as raw:
            authority = prepare_contract(RequestReceivingAuthorization, raw, redactor=redactor)
            require_exact_contract(authority.command, admission, redactor=redactor)
            require_exact_contract(authority.receiver, receiver_ref, redactor=redactor)
            commitment = await _native_execution_commitment(app, prepared, execution)
            command = prepare_contract(
                ProducerOutputRegistration,
                {
                    "operation": proposal.operation,
                    "admission": admission,
                    "initiator": _initiator(context),
                    "receiver": receiver_ref,
                    "binding_incarnation": proposal.binding_incarnation,
                    "execution_key": execution.execution_key,
                    "execution_commitment": commitment,
                    "publisher_generation": admission.generation,
                    "disposition": admission.expected.intent.request.cancellation,
                    "limits": proposal.limits,
                    "destinations": proposal.destinations,
                },
                redactor=redactor,
            )
            preflight_native_output(command)
            return command

    async def owned():
        return await coordinator._dependency(prepare)

    return await coordinator._observe(
        coordinator._owners.run(
            owned,
            key=("producer_preparation", object()),
            expectation=contract_bytes(proposal, redactor=redactor),
            redactor=redactor,
            failure_snapshot=lambda error: _safe_request_failure(error, redactor),
            wait_for_settlement=wait_for_settlement,
        )
    )
