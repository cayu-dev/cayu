"""Retain fixed native-owner completion under the original output responsibility."""

from cayu.collaboration._preparation import contract_bytes, prepare_contract, require_exact_contract
from cayu.collaboration._producer_contracts import (
    ProducerCompletionRecord,
    ProducerOutputRegistration,
)
from cayu.collaboration._producer_store import accept_native_completion, read_output_registration
from cayu.collaboration._recipient_admission_receiver import RecipientAdmissionReceivingOwner
from cayu.collaboration._request_coordinator import _safe_request_failure
from cayu.collaboration._request_store import operation_key
from cayu.collaboration.base import REQUEST_FAMILY
from cayu.collaboration.participants import CollaborationUnavailable


async def retain_producer_completion(app, command):
    """Internal mandatory handoff. Public callers cannot supply production receipts.

    This retains content-free references to already-owned output even after expiry
    or revocation. It neither renews disclosure/launch authority nor settles work.
    """
    coordinator = app._request_coordinator
    redactor = app._secret_redactor
    command = prepare_contract(ProducerOutputRegistration, command, redactor=redactor)
    registered = coordinator._registration
    receiver = None if registered is None else registered.receiving_owner
    if type(receiver) is not RecipientAdmissionReceivingOwner:
        raise CollaborationUnavailable("Producer completion requires its registered native owner.")
    require_exact_contract(command.receiver, coordinator.prepared_receiver_ref(), redactor=redactor)
    require_exact_contract(command.receiver, receiver.ref, redactor=redactor)
    store, initialized = app._participant_coordinator._ready()
    app._participant_coordinator._capability(
        store, initialized, mutation=True, family=REQUEST_FAMILY
    )

    async def accept():
        async with store._transaction(initialized.owner.application_scope, write=False) as tx:
            record = await read_output_registration(tx, command, redactor=redactor)
            if record is not None and record.completion is not None:
                # The source transaction already authenticated and retained this
                # immutable completion. Replay must survive native handoff and
                # pruning; it is not a new production or disclosure decision.
                # read_output_registration checks the complete expected command,
                # original responsibility, launch and completion commitment.
                return prepare_contract(
                    ProducerCompletionRecord,
                    await tx.get("operations", operation_key(record.completion)),
                    redactor=redactor,
                )
        if record is None:
            raise CollaborationUnavailable("Producer registration is unavailable.")
        # No native call while holding a collaboration transaction. The source
        # pin stays held until later authenticated export/delivery acceptance.
        output = await receiver._read_producer_output(record)
        if output is None:
            raise CollaborationUnavailable("Native producer output remains unresolved.")
        async with store._transaction(initialized.owner.application_scope, write=True) as tx:
            return await accept_native_completion(
                store, tx, initialized, command, output, redactor=redactor
            )

    async def owned():
        return await coordinator._dependency(accept)

    return await coordinator._observe(
        coordinator._owners.run(
            owned,
            key=("producer_completion", object()),
            expectation=contract_bytes(command, redactor=redactor),
            redactor=redactor,
            failure_snapshot=lambda error: _safe_request_failure(error, redactor),
        )
    )
