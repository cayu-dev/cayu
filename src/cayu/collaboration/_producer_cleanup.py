"""Owned source/native/source cleanup handshake, separate from answer election."""

from cayu.collaboration._preparation import contract_bytes, prepare_contract, require_exact_contract
from cayu.collaboration._producer_cleanup_finalization import (
    _received_native_cleanup,
    finalize_cleanup,
    read_finalization,
)
from cayu.collaboration._producer_contracts import ProducerCleanupRecord, ProducerOutputRegistration
from cayu.collaboration._producer_settlement import prepare_producer_cleanup
from cayu.collaboration._producer_store import read_output_registration
from cayu.collaboration._recipient_admission_receiver import RecipientAdmissionReceivingOwner
from cayu.collaboration._request_coordinator import _safe_request_failure
from cayu.collaboration.base import REQUEST_FAMILY
from cayu.collaboration.participants import CollaborationUnavailable
from cayu.runtime._producer_cleanup_receipt import _accepted_source_cleanup


async def settle_producer_output(app, command, *, wait_for_settlement=False):
    """Internal mandatory cleanup. No content read, new dispatch or sponsor renewal.

    Native acknowledgement is session-independent. If observer cancellation or
    acknowledgement loss follows native release, recovery can finish the same
    source settlement even after deletion or public-ID reuse.
    """
    redactor = app._secret_redactor
    command = prepare_contract(ProducerOutputRegistration, command, redactor=redactor)
    coordinator = app._request_coordinator
    registered = coordinator._registration
    receiver = None if registered is None else registered.receiving_owner
    if type(receiver) is not RecipientAdmissionReceivingOwner:
        raise CollaborationUnavailable("Producer cleanup requires its registered native owner.")
    require_exact_contract(command.receiver, receiver.ref, redactor=redactor)
    require_exact_contract(command.receiver, coordinator.prepared_receiver_ref(), redactor=redactor)
    store, initialized = app._participant_coordinator._ready()
    app._participant_coordinator._capability(
        store, initialized, mutation=True, family=REQUEST_FAMILY
    )

    async def settle():
        async with store._transaction(initialized.owner.application_scope, write=False) as tx:
            record = await read_output_registration(tx, command, redactor=redactor)
            if record is None:
                raise CollaborationUnavailable("Producer responsibility is unavailable.")
            final = await read_finalization(tx, record, redactor=redactor)
            if final is not None:
                return final
        if not isinstance(record.cleanup, ProducerCleanupRecord):
            await prepare_producer_cleanup(app, command, wait_for_settlement=wait_for_settlement)
        async with store._transaction(initialized.owner.application_scope, write=False) as tx:
            record = await read_output_registration(tx, command, redactor=redactor)
            if record is None:
                raise CollaborationUnavailable("Producer cleanup acceptance is unavailable.")
        authority = _accepted_source_cleanup(record)
        native = await receiver._complete_producer_cleanup(record, authority=authority)
        accepted = _received_native_cleanup(command, native, redactor)
        async with store._transaction(initialized.owner.application_scope, write=True) as tx:
            return await finalize_cleanup(
                store, tx, initialized, command, native, authority=accepted, redactor=redactor
            )

    async def owned():
        return await coordinator._dependency(settle)

    return await coordinator._observe(
        coordinator._owners.run(
            owned,
            key=("producer_final_cleanup", object()),
            expectation=contract_bytes(command, redactor=redactor),
            redactor=redactor,
            failure_snapshot=lambda error: _safe_request_failure(error, redactor),
            wait_for_settlement=wait_for_settlement,
        )
    )
