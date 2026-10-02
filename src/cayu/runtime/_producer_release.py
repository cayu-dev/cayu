"""Read-only native release proof, distinct from producer/budget settlement."""

from cayu.collaboration._contracts import ContractValue, Generation, Identifier
from cayu.collaboration._preparation import prepare_contract, require_exact_contract
from cayu.collaboration._producer_contracts import ProducerOutputRegistration
from cayu.collaboration.prepared_admission import FreshRecipientAdmissionTarget, NativeCommitment
from cayu.sessions._producer_checkpoint import (
    ROOT_KEY,
    NativeProducerAttachment,
    NativeProducerIndex,
    attachment_index,
    attachment_operation_key,
)
from cayu.vaults.redaction import SecretRedactor


class ProducerNativeRelease(ContractValue):
    """Invocation release only; neither external quiescence nor spend settlement."""

    registration: ProducerOutputRegistration
    session_id: Identifier
    session_instance_id: Identifier
    interaction_id: Identifier
    run_epoch: Generation
    release_commitment: NativeCommitment


def release_read_target(command):
    command = prepare_contract(ProducerOutputRegistration, command, redactor=SecretRedactor())
    prepared = command.admission.prepared
    assert prepared is not None and isinstance(prepared.target, FreshRecipientAdmissionTarget)
    key = attachment_operation_key(command)
    return command, prepared.target.session_id, key


def release_from_snapshot(command, session, checkpoint, raw_attachment):
    """Caller must own a single backend snapshot containing all three inputs."""
    from cayu.execution_profiles import (
        ExecutionProfileIdentity,
    )
    from cayu.runtime._invocation_lifecycle import (
        _require_released_invocation_command_receipt,
        require_invocation_rebind_lineage,
    )
    from cayu.sessions._execution_profile_checkpoint import (
        ActiveInvocationExecutionProfile,
        active_invocation_execution_profile_from_checkpoint,
    )

    redactor = SecretRedactor()
    command, session_id, _ = release_read_target(command)
    if session is None or checkpoint is None:
        raise ValueError("Producer release evidence is unavailable.")
    attachment = prepare_contract(NativeProducerAttachment, raw_attachment, redactor=redactor)
    require_exact_contract(command, attachment.command, redactor=redactor)
    index = prepare_contract(NativeProducerIndex, checkpoint.get(ROOT_KEY), redactor=redactor)
    require_exact_contract(
        attachment_index(attachment),
        index.model_copy(
            update={
                "state": "prepared",
                "invocation": None,
                "output_commitment": None,
                "paused_stop": None,
                "exclusion_commitment": None,
                "cleanup_commitment": None,
                "cleanup_receipt": None,
            }
        ),
        redactor=redactor,
    )
    invocation = index.invocation
    if invocation is None or session.instance_id != index.session_instance_id:
        raise ValueError("Producer release lacks exact native admission.")
    prepared = command.admission.prepared
    assert prepared is not None and isinstance(prepared.target, FreshRecipientAdmissionTarget)
    profile = ExecutionProfileIdentity.model_validate_json(prepared.execution_profile_json)
    if invocation.profile_commitment != "sha256:" + profile.fingerprint:
        raise ValueError("Producer release profile conflicts.")
    if index.cleanup_receipt is not None:
        from hashlib import sha256

        from cayu.collaboration._preparation import contract_bytes

        settled = index.cleanup_receipt
        if (
            settled.registration != command.operation
            or settled.receiver != command.receiver
            or settled.registration_commitment
            != "sha256:" + sha256(contract_bytes(command, redactor=redactor)).hexdigest()
            or settled.run_epoch is None
        ):
            raise ValueError("Settled producer release conflicts with its original registration.")
        return ProducerNativeRelease(
            registration=command,
            session_id=session_id,
            session_instance_id=index.session_instance_id,
            interaction_id=invocation.interaction_id,
            run_epoch=settled.run_epoch,
            release_commitment=settled.native_release_commitment,
        )
    original = ActiveInvocationExecutionProfile(
        session_id=session_id,
        interaction_id=invocation.interaction_id,
        run_epoch=invocation.run_epoch,
        profile=profile,
    )
    active = active_invocation_execution_profile_from_checkpoint(checkpoint)
    if active is None:
        raise ValueError("Producer release lacks retained invocation authority.")
    # Recovery may replace the execution epoch, but cannot replace the original
    # interaction, profile, or native admission with an otherwise equal new run.
    # The store-owned lifecycle ledger positively proves every rebind hop.
    require_invocation_rebind_lineage(
        checkpoint,
        session_instance_id=index.session_instance_id,
        original=original,
        current=active,
    )
    receipt = _require_released_invocation_command_receipt(
        session,
        checkpoint,
        session_id=session_id,
        session_instance_id=index.session_instance_id,
        active_profile=active,
    )
    return ProducerNativeRelease(
        registration=command,
        session_id=session_id,
        session_instance_id=index.session_instance_id,
        interaction_id=invocation.interaction_id,
        run_epoch=active.run_epoch,
        release_commitment="sha256:" + receipt.record_sha256,
    )


async def producer_participant_settlement(store, checkpoint, expected, commitment):
    """Do not retire a producer's execution permit from terminal status alone.

    None means the ordinary non-admitted participant path owns classification.
    An admitted producer always requires its exact protected native release.
    """
    from cayu.collaboration._contracts import ExactMatch, ExactUnavailable
    from cayu.collaboration._permits import ReceivingSettlementReceipt
    from cayu.sessions.base import SessionRunFenced

    if checkpoint is None or ROOT_KEY not in checkpoint:
        return None
    redactor = SecretRedactor()
    index = prepare_contract(NativeProducerIndex, checkpoint[ROOT_KEY], redactor=redactor)
    if index.state != "admitted":
        return None
    invocation = index.invocation
    assert invocation is not None
    target = expected.intent.request.target
    if index.cleanup_receipt is not None and (
        invocation.participant_permit_operation != expected.operation.caller_key
        or invocation.participant_permit_commitment != "sha256:" + commitment
    ):
        # The old permit keeps its exact historical release; ordinary receiving
        # settlement owns a separately authenticated successor permit.
        return None
    if (
        invocation.participant_permit_operation != expected.operation.caller_key
        or invocation.participant_permit_commitment != "sha256:" + commitment
        or (index.session_id, index.session_instance_id) != (target.object_id, target.incarnation)
    ):
        raise ValueError("Producer participant execution authority conflicts.")
    attachment = prepare_contract(
        NativeProducerAttachment,
        await store.load_session_operation(index.session_id, index.operation_key),
        redactor=redactor,
    )
    try:
        release = await store._read_native_producer_release(attachment.command)
    except SessionRunFenced:
        return ExactUnavailable()
    return ExactMatch[ReceivingSettlementReceipt](
        receipt=ReceivingSettlementReceipt(
            expected=expected,
            receiving_owner=target.owner,
            receipt_id="producer-release:" + release.release_commitment.removeprefix("sha256:"),
            outcome="quiescent",
        )
    )
