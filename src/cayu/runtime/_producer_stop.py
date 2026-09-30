"""Exact producer closure handoff to the existing native safe-stop owner.

This is a private receiving seam. The source cleanup coordinator must retain
ownership of this operation across observer cancellation and acknowledgement
loss. Acceptance neither proves quiescence nor releases output or accounting.
"""

from dataclasses import dataclass
from hashlib import sha256

from cayu._validation import canonical_durable_json_bytes
from cayu.collaboration._contracts import ContractValue, Generation, Identifier, OperationRef
from cayu.collaboration._preparation import contract_bytes, prepare_contract, require_exact_contract
from cayu.collaboration._producer_contracts import ProducerOutputRecord
from cayu.collaboration.prepared_admission import FreshRecipientAdmissionTarget, NativeCommitment
from cayu.collaboration.requests import RequestControlReceipt
from cayu.runtime._producer_output_store import (
    ROOT_KEY,
    NativeProducerAttachment,
    NativeProducerIndex,
    attachment_operation_key,
)
from cayu.runtime._session_steering import accept_session_steering
from cayu.runtime.session_steering import StopAfterCurrentToolRoundRequest
from cayu.sessions.base import _invocation_lifecycle_authority_read_scope
from cayu.vaults.redaction import SecretRedactor

_SEAL = object()


def _bytes(value):
    return contract_bytes(value, redactor=SecretRedactor())


def _digest(value):
    return "sha256:" + sha256(_bytes(value)).hexdigest()


@dataclass(frozen=True)
class _ProducerStopAuthority:
    registration: bytes
    closure: bytes
    seal: object

    def require(self, registration, closure):
        if (
            self.seal is not _SEAL
            or self.registration != _bytes(registration.command)
            or self.closure != _bytes(closure)
        ):
            raise PermissionError("Producer stop requires authenticated source closure.")


def _received_producer_closure(registration, closure):
    """Only source-owner readback may mint this private receiving handoff."""
    return _ProducerStopAuthority(_bytes(registration.command), _bytes(closure), _SEAL)


class NativeProducerStopReceipt(ContractValue):
    """Safe-stop acceptance, not cancellation, effect exclusion or settlement."""

    registration: OperationRef
    registration_commitment: NativeCommitment
    closure_commitment: NativeCommitment
    session_id: Identifier
    session_instance_id: Identifier
    interaction_id: Identifier
    run_epoch: Generation
    stop_key: Identifier
    profile_commitment: NativeCommitment
    steering_commitment: NativeCommitment


async def accept_native_producer_stop(store, registration, closure, *, authority):
    """Fence only an already-admitted original invocation, never a newer run.

    Prepared/uncertain native admission needs the separate admission-exclusion
    decision. Missing native state cannot be interpreted as successful stop.
    """
    redactor = SecretRedactor()
    registration = prepare_contract(ProducerOutputRecord, registration, redactor=redactor)
    closure = prepare_contract(RequestControlReceipt, closure, redactor=redactor)
    if type(authority) is not _ProducerStopAuthority:
        raise PermissionError("Producer stop requires its registered source owner.")
    authority.require(registration, closure)
    command = registration.command
    require_exact_contract(
        command.admission.expected, closure.expected.intent.expected, redactor=redactor
    )
    if (
        command.disposition != "stop"
        or registration.state != "launch_claimed"
        or registration.launch is None
        or closure.state not in ("cancelled", "expired")
        or not store._supports_producer_attachment_protocol()
        or not store._supports_session_steering_protocol()
    ):
        raise PermissionError("Producer stop scope is not qualified.")
    prepared = command.admission.prepared
    assert prepared is not None and isinstance(prepared.target, FreshRecipientAdmissionTarget)
    target = prepared.target
    key = attachment_operation_key(command)
    with _invocation_lifecycle_authority_read_scope():
        checkpoint = await store.load_checkpoint(target.session_id)
    if checkpoint is None or ROOT_KEY not in checkpoint:
        raise ValueError("Producer native admission is unavailable, not stopped.")
    index = prepare_contract(NativeProducerIndex, checkpoint[ROOT_KEY], redactor=redactor)
    if (
        index.session_id != target.session_id
        or index.session_instance_id != target.session_instance_id
        or index.operation_key != key
        or index.state != "admitted"
        or index.invocation is None
    ):
        raise ValueError("Producer stop conflicts with its native invocation.")
    attachment = prepare_contract(
        NativeProducerAttachment,
        await store.load_session_operation(target.session_id, key),
        redactor=redactor,
    )
    require_exact_contract(
        NativeProducerAttachment.from_registration(registration), attachment, redactor=redactor
    )
    require_exact_contract(registration.launch, index.invocation.launch, redactor=redactor)
    if index.record_commitment != _digest(attachment):
        raise ValueError("Producer stop attachment commitment conflicts.")
    from cayu.runtime._producer_lineage import require_producer_epoch
    from cayu.runtime._session_steering import require_steering_receipt, steering_operation_key
    from cayu.runtime.execution_profiles import active_invocation_execution_profile_from_checkpoint
    from cayu.runtime.session_steering import SessionSteeringConflict

    def require_current(session, current_checkpoint):
        active = active_invocation_execution_profile_from_checkpoint(current_checkpoint)
        if active is None or active.run_epoch != session.run_epoch:
            raise SessionSteeringConflict()
        proven = require_producer_epoch(index, current_checkpoint, active.profile, active.run_epoch)
        if active != proven:
            raise SessionSteeringConflict()

    active = active_invocation_execution_profile_from_checkpoint(checkpoint)
    if active is None:
        raise SessionSteeringConflict()
    if active != require_producer_epoch(index, checkpoint, active.profile, active.run_epoch):
        raise SessionSteeringConflict()
    request = StopAfterCurrentToolRoundRequest(
        session_id=target.session_id,
        session_instance_id=target.session_instance_id,
        interaction_id=index.invocation.interaction_id,
        expected_run_epoch=active.run_epoch,
        idempotency_key="producer-stop:" + sha256(_bytes(command.operation)).hexdigest(),
    )
    existing = await store.load_session_operation(
        target.session_id,
        steering_operation_key(target.session_instance_id, index.invocation.interaction_id),
    )
    if existing is not None:
        prior = require_steering_receipt(existing)
        require_producer_epoch(index, checkpoint, active.profile, prior.request.expected_run_epoch)
        if (
            prior.request.expected_run_epoch > active.run_epoch
            or prior.request
            != request.model_copy(update={"expected_run_epoch": prior.request.expected_run_epoch})
        ):
            raise SessionSteeringConflict()
        request = prior.request
    # The native owner compares incarnation, active profile and epoch again
    # within the same transaction that publishes the stop fence. The preceding
    # reads alone never authorize stopping a replacement invocation.
    from cayu.runtime._producer_paused_stop import accept_paused_stop

    receipt = await accept_paused_stop(store, command, index, attachment, request, _digest(closure))
    if receipt is None:
        receipt = await accept_session_steering(
            request, session_store=store, redactor=redactor, invocation_guard=require_current
        )
    if receipt.request != request or (
        "sha256:" + receipt.execution_profile_fingerprint != index.invocation.profile_commitment
    ):
        raise ValueError("Native producer stop receipt conflicts.")
    return NativeProducerStopReceipt(
        registration=command.operation,
        registration_commitment=_digest(command),
        closure_commitment=_digest(closure),
        session_id=request.session_id,
        session_instance_id=request.session_instance_id,
        interaction_id=request.interaction_id,
        run_epoch=request.expected_run_epoch,
        stop_key=request.idempotency_key,
        profile_commitment=index.invocation.profile_commitment,
        steering_commitment="sha256:"
        + sha256(
            canonical_durable_json_bytes(receipt.model_dump(mode="json"), "producer stop receipt")
        ).hexdigest(),
    )
