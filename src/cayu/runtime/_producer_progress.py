"""Content-free native producer milestones from one store-owned read snapshot."""

from hashlib import sha256

from cayu.collaboration._preparation import contract_bytes, prepare_contract, require_exact_contract
from cayu.collaboration._producer_progress_contracts import ProducerProgressEvidence
from cayu.runtime._producer_release import release_read_target
from cayu.sessions._producer_checkpoint import (
    ROOT_KEY,
    NativeProducerAttachment,
    NativeProducerIndex,
    attachment_index,
)
from cayu.vaults.redaction import SecretRedactor


def progress_from_snapshot(command, kind, session, checkpoint, raw_attachment):
    """Caller owns a single backend snapshot; no separately loaded status inference."""
    from cayu.runtime.execution_profiles import (
        ExecutionProfileIdentity,
        active_invocation_execution_profile_from_checkpoint,
    )

    redactor = SecretRedactor()
    command, session_id, _ = release_read_target(command)
    if type(kind) is not str or kind not in {"prepared", "started", "producing", "published"}:
        raise ValueError("Producer progress kind is unavailable.")
    if session is None or checkpoint is None:
        raise ValueError("Producer progress lacks native evidence.")
    attachment = prepare_contract(NativeProducerAttachment, raw_attachment, redactor=redactor)
    require_exact_contract(command, attachment.command, redactor=redactor)
    index = prepare_contract(NativeProducerIndex, checkpoint.get(ROOT_KEY), redactor=redactor)
    active = (
        active_invocation_execution_profile_from_checkpoint(checkpoint)
        if kind == "producing"
        else None
    )
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
    if (
        session.id != session_id
        or session.instance_id != index.session_instance_id
        or index.state == "excluded"
        or index.cleanup_commitment is not None
        or (kind != "prepared" and index.invocation is None)
        or (
            kind == "producing"
            and (
                session.status != "running"
                or index.invocation is None
                or active is None
                or active.session_id != session_id
                or active.interaction_id != index.invocation.interaction_id
                or active.run_epoch != session.run_epoch
                or "sha256:" + active.profile.fingerprint != index.invocation.profile_commitment
            )
        )
        or (kind == "published" and index.output_commitment is None)
    ):
        raise ValueError("Producer progress lacks the requested native milestone.")
    prepared = command.admission.prepared
    assert prepared is not None
    profile = ExecutionProfileIdentity.model_validate_json(prepared.execution_profile_json)
    if kind == "producing":
        from cayu.runtime._producer_lineage import require_producer_epoch

        assert active is not None
        if active != require_producer_epoch(index, checkpoint, profile, active.run_epoch):
            raise ValueError("Producer progress invocation lineage conflicts.")
    registration_commitment = (
        "sha256:" + sha256(contract_bytes(command, redactor=redactor)).hexdigest()
    )
    if index.invocation is not None and (
        index.invocation.profile_commitment != "sha256:" + profile.fingerprint
        or index.invocation.launch.registration != command.operation
        or index.invocation.launch.command_commitment != registration_commitment
    ):
        raise ValueError("Producer progress native profile conflicts.")
    return ProducerProgressEvidence(
        registration=command.operation,
        registration_commitment=registration_commitment,
        kind=kind,
        session_id=index.session_id,
        session_instance_id=index.session_instance_id,
        interaction_id=None if index.invocation is None else index.invocation.interaction_id,
        run_epoch=active.run_epoch
        if active is not None
        else None
        if index.invocation is None
        else index.invocation.run_epoch,
        profile_commitment="sha256:" + profile.fingerprint,
        native_commitment="sha256:" + sha256(contract_bytes(index, redactor=redactor)).hexdigest(),
    )


async def published_progress(store, command):
    """A published milestone needs exact retained output or terminal failure evidence."""
    from cayu.runtime._producer_output_store import read_retained_native_output
    from cayu.runtime.execution_profiles import ExecutionProfileIdentity

    command, session_id, _ = release_read_target(command)
    output = await read_retained_native_output(store, command)
    if output is None:
        raise ValueError("Producer publication remains unresolved.")
    prepared = command.admission.prepared
    assert prepared is not None
    profile = ExecutionProfileIdentity.model_validate_json(prepared.execution_profile_json)
    redactor = SecretRedactor()
    return ProducerProgressEvidence(
        registration=command.operation,
        registration_commitment="sha256:"
        + sha256(contract_bytes(command, redactor=redactor)).hexdigest(),
        kind="published",
        session_id=session_id,
        session_instance_id=prepared.target.session_instance_id,
        interaction_id=output.interaction_id,
        run_epoch=output.run_epoch,
        profile_commitment="sha256:" + profile.fingerprint,
        native_commitment="sha256:" + sha256(contract_bytes(output, redactor=redactor)).hexdigest(),
    )
