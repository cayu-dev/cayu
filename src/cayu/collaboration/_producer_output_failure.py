"""Native validation evidence bridged by the registered owner, never caller claims."""

from hashlib import sha256
from typing import Literal

from cayu.collaboration._contracts import (
    CollaborationConflict,
    ContractValue,
    Generation,
    OperationRef,
)
from cayu.collaboration._preparation import contract_bytes, prepare_contract, require_exact_contract
from cayu.collaboration._request_store import operation_key
from cayu.collaboration._session_export_store import (
    ExportPreparation,
    ExportValidationFailure,
    read_scope,
)
from cayu.collaboration.prepared_admission import NativeCommitment


class ProducerExportRejected(ContractValue):
    mode: Literal["producer_export_rejected"] = "producer_export_rejected"
    operation: OperationRef
    export: OperationRef
    registration: OperationRef
    native_commitment: NativeCommitment
    failure: ExportValidationFailure
    sequence: Generation


async def read_output_failure(tx, exported, *, redactor):
    if exported.rejection is None:
        return None
    record = prepare_contract(
        ProducerExportRejected,
        await tx.get("operations", operation_key(exported.rejection)),
        redactor=redactor,
    )
    if (
        record.operation != exported.rejection
        or record.sequence <= exported.sequence
        or record.operation
        != exported.operation.model_copy(
            update={"caller_key": exported.operation.caller_key + ":rejected"}
        )
        or record.export != exported.operation
        or record.registration != exported.registration
        or record.failure.request != exported.request
        or record.native_commitment
        != "sha256:" + sha256(contract_bytes(record.failure, redactor=redactor)).hexdigest()
    ):
        raise CollaborationConflict("Producer output rejection conflicts.")
    return record


async def retain_output_failure(app, command, destination, intent):
    from cayu.collaboration._producer_contracts import (
        ProducerCompletionRecord,
        ProducerExportRecord,
    )
    from cayu.collaboration._producer_export_store import _charge, read_export
    from cayu.collaboration._producer_store import read_output_registration

    redactor = app._secret_redactor
    from cayu.collaboration._recipient_admission_receiver import RecipientAdmissionReceivingOwner
    from cayu.collaboration.participants import CollaborationUnavailable

    registration_config = app._request_coordinator._registration
    receiver = None if registration_config is None else registration_config.receiving_owner
    if type(receiver) is not RecipientAdmissionReceivingOwner:
        raise CollaborationUnavailable("Producer validation requires its registered native owner.")
    failure = await receiver._read_producer_validation_failure(
        command, intent, exports=app._session_export_coordinator
    )
    if failure is None:
        return None
    store, initialized = app._participant_coordinator._ready()
    async with store._transaction(initialized.owner.application_scope, write=True) as tx:
        registration = await read_output_registration(tx, command, redactor=redactor)
        current = await read_export(tx, command, destination, redactor=redactor)
        if registration is None or current is None or current.state != "prepared":
            raise CollaborationConflict("Producer rejection has no matching pending export.")
        prior = await read_output_failure(tx, current, redactor=redactor)
        if prior is not None:
            require_exact_contract(prior.failure, failure, redactor=redactor)
            return prior
        completion = prepare_contract(
            ProducerCompletionRecord,
            await tx.get("operations", operation_key(current.completion)),
            redactor=redactor,
        )
        if completion.output.source_commitment != "sha256:" + failure.source_commitment:
            raise CollaborationConflict("Producer rejection source changed.")
        anchor = await store._anchor(tx, initialized, redactor)
        record = ProducerExportRejected(
            operation=current.operation.model_copy(
                update={"caller_key": current.operation.caller_key + ":rejected"}
            ),
            export=current.operation,
            registration=command.operation,
            failure=failure,
            native_commitment="sha256:"
            + sha256(contract_bytes(failure, redactor=redactor)).hexdigest(),
            sequence=anchor.event_sequence + 1,
        )
        updated = prepare_contract(
            ProducerExportRecord,
            current.model_copy(update={"rejection": record.operation}),
            redactor=redactor,
        )
        await _charge(
            store,
            tx,
            initialized,
            registration,
            registration,
            (current,),
            (updated, record),
            sequence=record.sequence,
            redactor=redactor,
        )
        await tx.put("operations", operation_key(record.operation), record, insert=True)
        await tx.put("operations", operation_key(updated.operation), updated, insert=False)
        return record


async def read_native_validation_failure(sessions, command, intent, *, redactor):
    from cayu.collaboration._session_export_store import operation_key as native_key

    request = intent.request
    prepared = command.admission.prepared
    if (
        prepared is None
        or intent.registration != command.operation
        or (request.ref.session_id, request.ref.session_instance_id)
        != (prepared.target.session_id, prepared.target.session_instance_id)
    ):
        raise CollaborationConflict("Producer validation target conflicts.")
    with read_scope(request.ref.session_id):
        raw = await sessions.load_session_operation(
            request.ref.session_id, native_key(request.ref.operation)
        )
    if type(raw) is not dict or "admission" not in raw or "receipt" in raw:
        return None
    preparation = prepare_contract(ExportPreparation, raw, redactor=redactor)
    failure = preparation.validation_failure
    if failure is None:
        return None
    require_exact_contract(request, failure.request, redactor=redactor)
    require_exact_contract(
        intent.initiator,
        preparation.admission.authorization.initiating_identity(),
        redactor=redactor,
    )
    if preparation.admission.permit.intent.request.participant != prepared.recipient:
        raise CollaborationConflict("Producer validation participant conflicts.")
    return failure
