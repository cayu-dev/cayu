"""Native producer attachment and its private checkpoint/operation fence.

Preparation is not launch authority. The native invocation owner must consume an
authenticated launch decision before changing this attachment to admitted.
"""

from __future__ import annotations

from hashlib import sha256
from uuid import NAMESPACE_URL, uuid5

from cayu.collaboration._preparation import contract_bytes, prepare_contract, require_exact_contract
from cayu.collaboration._producer_contracts import (
    ProducerCleanupRecord,
    ProducerCompletionRecord,
    ProducerNativeExclusion,
    ProducerNativeFailure,
    ProducerOutputRecord,
    ProducerOutputRegistration,
)
from cayu.collaboration._producer_contracts import ProducerNativeOutput as NativeProducerOutput
from cayu.collaboration.prepared_admission import FreshRecipientAdmissionTarget
from cayu.collaboration.requests import RequestControlReceipt
from cayu.sessions._producer_checkpoint import _PUBLICATION as _PUBLICATION
from cayu.sessions._producer_checkpoint import OPERATION_PREFIX as OPERATION_PREFIX
from cayu.sessions._producer_checkpoint import ROOT_KEY as ROOT_KEY
from cayu.sessions._producer_checkpoint import NativeProducerAttachment as NativeProducerAttachment
from cayu.sessions._producer_checkpoint import NativeProducerIndex as NativeProducerIndex
from cayu.sessions._producer_checkpoint import NativeProducerInvocation as NativeProducerInvocation
from cayu.sessions._producer_checkpoint import NativeProducerPausedStop as NativeProducerPausedStop
from cayu.sessions._producer_checkpoint import _publication_scope as _publication_scope
from cayu.sessions._producer_checkpoint import attachment_index as attachment_index
from cayu.sessions._producer_checkpoint import attachment_operation_key as attachment_operation_key
from cayu.sessions._producer_checkpoint import checkpoint_visible as checkpoint_visible
from cayu.sessions._producer_checkpoint import project_checkpoint_root as project_checkpoint_root
from cayu.sessions._producer_checkpoint import (
    require_operation_key_access as require_operation_key_access,
)
from cayu.sessions._producer_cleanup_contract import NativeProducerCleanupReceipt
from cayu.vaults.redaction import SecretRedactor


class NativeProducerAdmissionWon(ValueError):
    """Exact native admission won the exclusion transaction; no stop is implied."""

    def __init__(self):
        super().__init__("The exact producer invocation was already admitted.")


def preflight_native_output(command: ProducerOutputRegistration) -> None:
    """Reserve the complete native receipt envelope before admitting work."""
    from cayu.collaboration._contracts import MAX_ID_BYTES, ExactMatch

    output = NativeProducerOutput(
        registration=command,
        stage_id="s" * MAX_ID_BYTES,
        publication_id="p" * MAX_ID_BYTES,
        publication_commitment="sha256:" + "0" * 64,
        interaction_id="i" * MAX_ID_BYTES,
        run_epoch=2**53 - 1,
        source_indices=tuple(range(2**53 - 16, 2**53)),
        source_commitment="sha256:" + "0" * 64,
        disposition="answer",
    )
    completed = ProducerCompletionRecord(
        operation=command.operation.model_copy(update={"caller_key": "c" * MAX_ID_BYTES}),
        output=output,
        native_commitment="sha256:" + "0" * 64,
        sequence=2**53 - 1,
        recorded_at_ms=2**53 - 1,
    )
    failed = ProducerCompletionRecord(
        operation=command.operation.model_copy(update={"caller_key": "c" * MAX_ID_BYTES}),
        output=ProducerNativeFailure(
            registration=command,
            interaction_id="i" * MAX_ID_BYTES,
            run_epoch=2**53 - 1,
            event_id="e" * MAX_ID_BYTES,
            settlement_commitment="sha256:" + "0" * 64,
        ),
        native_commitment="sha256:" + "0" * 64,
        sequence=2**53 - 1,
        recorded_at_ms=2**53 - 1,
    )
    # Public exact readback adds its own envelope. Reserve it before native
    # admission rather than discovering an unreplayable receipt after completion.
    ExactMatch[ProducerCompletionRecord](receipt=completed)
    ExactMatch[ProducerCompletionRecord](receipt=failed)


def require_erasure_quiescence(*, session, checkpoint, records) -> None:
    redactor = SecretRedactor()
    raw = None if checkpoint is None else checkpoint.get(ROOT_KEY)
    if raw is None and not records:
        return
    index = prepare_contract(NativeProducerIndex, raw, redactor=redactor)
    if index.state == "admitted" and index.cleanup_commitment is not None:
        from cayu.runtime._producer_cleanup_receipt import require_cleanup_for_erasure

        return require_cleanup_for_erasure(session=session, checkpoint=checkpoint, records=records)
    if index.state != "excluded" or index.cleanup_commitment is None:
        raise ValueError("Session closure requires settled producer output responsibility.")
    if (index.session_id, index.session_instance_id) != (session.id, session.instance_id):
        raise ValueError("Producer cleanup belongs to another session.")
    if set(records) != {
        index.operation_key,
        index.operation_key + ":excluded",
        index.operation_key + ":cleanup",
        index.operation_key + ":cleanup-ack",
    }:
        raise ValueError("Producer cleanup evidence is incomplete.")
    attachment = prepare_contract(
        NativeProducerAttachment, records[index.operation_key], redactor=redactor
    )
    original = attachment_index(attachment)
    require_exact_contract(
        original,
        index.model_copy(
            update={
                "state": "prepared",
                "exclusion_commitment": None,
                "cleanup_commitment": None,
                "cleanup_receipt": None,
            }
        ),
        redactor=redactor,
    )
    exclusion = prepare_contract(
        ProducerNativeExclusion, records[index.operation_key + ":excluded"], redactor=redactor
    )
    cleanup = prepare_contract(
        ProducerCleanupRecord, records[index.operation_key + ":cleanup"], redactor=redactor
    )

    receipt = prepare_contract(
        NativeProducerCleanupReceipt,
        records[index.operation_key + ":cleanup-ack"],
        redactor=redactor,
    )
    require_exact_contract(cleanup.exclusion, exclusion, redactor=redactor)
    if (
        cleanup.registration != attachment.command.operation
        or exclusion.registration != attachment.command.operation
        or exclusion.session_id != session.id
        or exclusion.session_instance_id != session.instance_id
        or exclusion.attachment_commitment != index.record_commitment
        or index.exclusion_commitment
        != "sha256:" + sha256(contract_bytes(exclusion, redactor=redactor)).hexdigest()
        or index.cleanup_commitment
        != "sha256:" + sha256(contract_bytes(receipt, redactor=redactor)).hexdigest()
        or receipt.mode != "exclusion"
        or receipt.registration != attachment.command.operation
        or receipt.receiver != attachment.command.receiver
        or receipt.registration_commitment
        != "sha256:" + sha256(contract_bytes(attachment.command, redactor=redactor)).hexdigest()
        or receipt.source_cleanup_commitment
        != "sha256:" + sha256(contract_bytes(cleanup, redactor=redactor)).hexdigest()
        or receipt.session_id != session.id
        or receipt.session_instance_id != session.instance_id
        or receipt.native_release_commitment != index.exclusion_commitment
    ):
        raise ValueError("Producer cleanup identity conflicts.")


def require_native_admission(checkpoint, command, *, now):
    """An attachment cannot be bypassed by the ordinary lifecycle command entrance."""
    raw = None if checkpoint is None else checkpoint.get(ROOT_KEY)
    if raw is None:
        return
    redactor = SecretRedactor()
    prior = prepare_contract(NativeProducerIndex, raw, redactor=redactor)
    if prior.cleanup_receipt is not None:
        require_successor_invocation(prior, command.target_active_profile)
        if command.expected_session_instance_id != prior.session_instance_id:
            raise PermissionError("Successor invocation changed the session incarnation.")
        # The exact cleanup receipt is owner-published under the store lock.
        # It releases producer ownership, not execution authority: the enclosing
        # lifecycle transaction still authenticates the ordinary admission and
        # atomically records the successor's native ownership receipt.
        return
    current = _PUBLICATION.get()
    if (
        current is None
        or current.state != "admitted"
        or prior.state != "prepared"
        or current.session_id != command.session_id
        or current.session_instance_id != command.expected_session_instance_id
    ):
        raise PermissionError("Producer invocation requires its registered native handoff.")
    invocation = current.invocation
    assert invocation is not None
    # The source election is not native admission. An attachment read or lock
    # wait can outlive its authority even while the revocation guard is held.
    # This callback runs inside the receiving store's admission transaction.
    if int(now.timestamp() * 1000) >= invocation.launch.authority_expires_at_ms:
        raise PermissionError("Producer launch authority expired before native admission.")
    if (
        invocation.interaction_id != command.target_active_profile.interaction_id
        or invocation.run_epoch != command.target_active_profile.run_epoch
        or invocation.profile_commitment
        != "sha256:" + command.target_active_profile.profile.fingerprint
        or invocation.participant_permit_operation != command.participant_permit_operation
        or command.participant_permit_commitment is None
        or invocation.participant_permit_commitment
        != "sha256:" + command.participant_permit_commitment
    ):
        raise PermissionError("Producer invocation authority conflicts.")


def require_successor_invocation(index, active):
    receipt = index.cleanup_receipt
    if receipt is None or (
        active.session_id != index.session_id
        or active.interaction_id == receipt.interaction_id
        or active.run_epoch <= (receipt.run_epoch or 0)
    ):
        raise PermissionError("Invocation is not an authorized successor to settled production.")


async def admit_native_producer(store, registration: ProducerOutputRecord, command):
    """Consume a guarded launch election in the existing invocation transaction.

    Only the runtime-owned launch handoff calls this method while holding current
    execution authorization. The source decision alone is not a public grant.
    """
    from cayu.runtime._invocation_lifecycle import (
        AdmitInvocationCommand,
        InvocationCheckpointPatch,
        copy_invocation_lifecycle_command,
    )
    from cayu.sessions.base import (
        RuntimePublicationMutation,
        SessionStatus,
        runtime_publication_checkpoint_mutation,
    )

    redactor = SecretRedactor()
    registration = prepare_contract(ProducerOutputRecord, registration, redactor=redactor)
    command = copy_invocation_lifecycle_command(command)
    if type(command) is not AdmitInvocationCommand or registration.launch is None:
        raise ValueError("Producer admission requires its exact launch election.")
    attachment = NativeProducerAttachment.from_registration(registration)
    index = attachment_index(attachment)
    prepared = registration.command.admission.prepared
    assert prepared is not None and isinstance(prepared.target, FreshRecipientAdmissionTarget)
    from cayu.execution_profiles import (
        ExecutionProfileIdentity,
    )

    profile = ExecutionProfileIdentity.model_validate_json(prepared.execution_profile_json)
    if (
        command.session_id != index.session_id
        or command.expected_session_instance_id != index.session_instance_id
        or command.expected_run_epoch != 0
        or command.expected_statuses != (SessionStatus.PENDING,)
        or command.target_active_profile.run_epoch != 1
        or command.target_active_profile.profile != profile
        or command.target_active_profile.interaction_id
        != str(
            uuid5(
                NAMESPACE_URL,
                f"cayu-participant-session:{index.session_id}:{registration.command.execution_key}",
            )
        )
        or not command.allow_pending_initial_interaction
        or not command.defer_interaction_source
        or command.participant_permit_operation is None
        or command.participant_permit_commitment is None
        or any(item.key == ROOT_KEY for item in command.checkpoint_patch.mutation.operations)
    ):
        raise ValueError("Producer native admission conflicts with registered execution.")
    retained = await store.load_session_operation(index.session_id, index.operation_key)
    require_exact_contract(
        attachment,
        prepare_contract(NativeProducerAttachment, retained, redactor=redactor),
        redactor=redactor,
    )
    admitted = prepare_contract(
        NativeProducerIndex,
        index.model_copy(
            update={
                "state": "admitted",
                "invocation": NativeProducerInvocation(
                    launch=registration.launch,
                    interaction_id=command.target_active_profile.interaction_id,
                    run_epoch=command.target_active_profile.run_epoch,
                    profile_commitment="sha256:" + profile.fingerprint,
                    participant_permit_operation=command.participant_permit_operation,
                    participant_permit_commitment="sha256:" + command.participant_permit_commitment,
                ),
            }
        ),
        redactor=redactor,
    )
    patch = runtime_publication_checkpoint_mutation(
        {ROOT_KEY: index.model_dump(mode="json")}, {ROOT_KEY: admitted.model_dump(mode="json")}
    )
    bound = copy_invocation_lifecycle_command(
        command.model_copy(
            update={
                "checkpoint_patch": InvocationCheckpointPatch(
                    mutation=RuntimePublicationMutation(
                        operations=(
                            *command.checkpoint_patch.mutation.operations,
                            *patch.operations,
                        )
                    )
                )
            }
        )
    )
    # Scope covers only the qualified native transaction, never application code.
    with _publication_scope(admitted):
        return await store.apply_invocation_lifecycle_command(bound)


def attachment_from_snapshot(command, session, checkpoint, raw_attachment):
    """Read both attachment representations from one native-store snapshot.

    None proves absence only within the exact existing target incarnation. A
    partial record, replaced session, or conflicting index is never absence.
    This does not grant permission to execute or certify invocation settlement.
    """
    command = prepare_contract(ProducerOutputRegistration, command, redactor=SecretRedactor())
    prepared = command.admission.prepared
    assert prepared is not None and isinstance(prepared.target, FreshRecipientAdmissionTarget)
    if session is None or (session.id, session.instance_id) != (
        prepared.target.session_id,
        prepared.target.session_instance_id,
    ):
        raise ValueError("Producer attachment target is unavailable or replaced.")
    raw_index = None if checkpoint is None else checkpoint.get(ROOT_KEY)
    if raw_index is None and raw_attachment is None:
        return None
    if raw_index is None or raw_attachment is None:
        raise ValueError("Producer attachment has incomplete native evidence.")
    redactor = SecretRedactor()
    attachment = prepare_contract(NativeProducerAttachment, raw_attachment, redactor=redactor)
    require_exact_contract(command, attachment.command, redactor=redactor)
    index = prepare_contract(NativeProducerIndex, raw_index, redactor=redactor)
    require_exact_contract(
        attachment_index(attachment),
        index.model_copy(
            update={
                "state": "prepared",
                "invocation": None,
                "exclusion_commitment": None,
                "cleanup_commitment": None,
                "cleanup_receipt": None,
                "output_commitment": None,
                "paused_stop": None,
            }
        ),
        redactor=redactor,
    )
    return attachment


async def attach_native_producer(store, registration: ProducerOutputRecord):
    """Private registered-owner handoff, retaining both native representations atomically.

    The caller authenticates registration against CollaborationStore before this
    native transaction. No receiving callback executes inside the native scope.
    """
    from cayu.sessions.base import SessionOperationPublication, SessionStatus

    registration = prepare_contract(ProducerOutputRecord, registration, redactor=SecretRedactor())
    attachment = NativeProducerAttachment.from_registration(registration)
    index = attachment_index(attachment)

    class AlreadyAttached(Exception):
        """Abort the write after authenticating an existing immutable attachment."""

    def publish(session, checkpoint, current):
        if session.instance_id != index.session_instance_id:
            raise ValueError("Producer attachment targets another session incarnation.")
        existing_index = None if checkpoint is None else checkpoint.get(ROOT_KEY)
        if current is not None:
            retained = attachment_from_snapshot(attachment.command, session, checkpoint, current)
            require_exact_contract(attachment, retained, redactor=SecretRedactor())
            # Replay acknowledges attachment, not preparation or permission to
            # execute. Never replace an admitted/excluded/settled index.
            raise AlreadyAttached()
        elif existing_index is not None:
            raise ValueError("Producer session already has another attachment or lost evidence.")
        elif session.status != SessionStatus.PENDING or session.run_epoch != 0:
            raise ValueError("Producer attachment requires an inert native recipient.")
        return SessionOperationPublication(
            checkpoint={**(checkpoint or {}), ROOT_KEY: index.model_dump(mode="json")},
            operation_records={index.operation_key: attachment.model_dump(mode="json")},
        )

    try:
        with _publication_scope(index):
            await store.publish_session_operation(
                index.session_id,
                idempotency_key=index.operation_key,
                operation_transform=publish,
                events=[],
            )
    except AlreadyAttached:
        pass
    return attachment


def producer_owns_invocation(checkpoint):
    """An unsettled producer cannot lend its invocation to queued successor work."""
    raw = None if checkpoint is None else checkpoint.get(ROOT_KEY)
    if raw is None:
        return False
    index = prepare_contract(NativeProducerIndex, raw, redactor=SecretRedactor())
    return index.state == "admitted" and index.cleanup_commitment is None


def native_exclusion(registration: ProducerOutputRecord, control: RequestControlReceipt):
    redactor = SecretRedactor()
    registration = prepare_contract(ProducerOutputRecord, registration, redactor=redactor)
    control = prepare_contract(RequestControlReceipt, control, redactor=redactor)
    require_exact_contract(
        registration.command.admission.expected, control.expected.intent.expected, redactor=redactor
    )
    attachment = NativeProducerAttachment.from_registration(registration)
    index = attachment_index(attachment)
    receipt = ProducerNativeExclusion(
        registration=registration.command.operation,
        control_operation=control.expected.operation,
        control_commitment="sha256:"
        + sha256(contract_bytes(control, redactor=redactor)).hexdigest(),
        attachment_commitment=index.record_commitment,
        session_id=index.session_id,
        session_instance_id=index.session_instance_id,
        execution_commitment=registration.command.execution_commitment,
    )
    excluded = prepare_contract(
        NativeProducerIndex,
        index.model_copy(
            update={
                "state": "excluded",
                "exclusion_commitment": "sha256:"
                + sha256(contract_bytes(receipt, redactor=redactor)).hexdigest(),
            }
        ),
        redactor=redactor,
    )
    return attachment, index, excluded, receipt


async def exclude_native_producer(
    store, registration: ProducerOutputRecord, control: RequestControlReceipt
):
    """Registered cleanup handoff after authenticating exact source closure.

    This permanently fences a prepared invocation under the native session lock.
    An admitted invocation is not excluded by guessing from its session status.
    """
    from cayu.sessions.base import SessionOperationPublication, SessionStatus

    redactor = SecretRedactor()
    attachment, index, excluded, receipt = native_exclusion(registration, control)

    def exclude(session, checkpoint, current):
        if session.instance_id != index.session_instance_id:
            raise ValueError("Producer exclusion targets another session incarnation.")
        require_exact_contract(
            attachment,
            prepare_contract(NativeProducerAttachment, current, redactor=redactor),
            redactor=redactor,
        )
        retained = prepare_contract(
            NativeProducerIndex,
            None if checkpoint is None else checkpoint.get(ROOT_KEY),
            redactor=redactor,
        )
        if (
            retained.model_copy(update={"cleanup_commitment": None, "cleanup_receipt": None})
            == excluded
        ):
            pass  # Exact retry cannot reopen native admission.
        elif retained.state == "admitted" and retained.invocation is not None:
            if registration.launch is None:
                raise ValueError("Native producer admission lacks its source launch decision.")
            require_exact_contract(
                index,
                retained.model_copy(
                    update={
                        "state": "prepared",
                        "invocation": None,
                        "cleanup_commitment": None,
                        "cleanup_receipt": None,
                        "output_commitment": None,
                        "paused_stop": None,
                    }
                ),
                redactor=redactor,
            )
            require_exact_contract(
                registration.launch, retained.invocation.launch, redactor=redactor
            )
            raise NativeProducerAdmissionWon()
        elif retained != index or session.status != SessionStatus.PENDING or session.run_epoch != 0:
            raise ValueError("Producer exclusion lacks an unconsumed native attachment.")
        return SessionOperationPublication(
            checkpoint={
                **(checkpoint or {}),
                ROOT_KEY: (
                    retained if retained.cleanup_commitment is not None else excluded
                ).model_dump(mode="json"),
            },
            operation_records={
                index.operation_key: attachment.model_dump(mode="json"),
                index.operation_key + ":excluded": receipt.model_dump(mode="json"),
            },
        )

    with _publication_scope(excluded):
        await store.publish_session_operation(
            index.session_id,
            idempotency_key=index.operation_key,
            operation_transform=exclude,
            events=[],
        )
    return await read_native_exclusion(store, registration, control)


async def read_native_exclusion(
    store, registration: ProducerOutputRecord, control: RequestControlReceipt
):
    """Read the immutable no-start decision, never infer it from absence."""
    _, index, _, expected = native_exclusion(registration, control)
    raw = await store.load_session_operation(index.session_id, index.operation_key + ":excluded")
    receipt = prepare_contract(ProducerNativeExclusion, raw, redactor=SecretRedactor())
    require_exact_contract(expected, receipt, redactor=SecretRedactor())
    return receipt


async def retain_native_output(store, session_id, *, invocation, stage_id):
    """Runtime-selected terminal output, retained before terminal publication.

    Native completion/publication records and the transcript remain retained by
    the existing publication prune fence and this producer's session-erasure pin.
    This record contains only exact references/commitments, never private parts.
    """
    from cayu.collaboration._native_output import read_native_output
    from cayu.sessions.base import (
        SessionOperationPublication,
        _invocation_lifecycle_authority_read_scope,
    )

    with _invocation_lifecycle_authority_read_scope():
        checkpoint = await store.load_checkpoint(session_id)
    raw = None if checkpoint is None else checkpoint.get(ROOT_KEY)
    if raw is None:
        return None
    if not store._supports_producer_attachment_protocol():
        raise NotImplementedError("Native producer output retention is not qualified.")
    if invocation is None:
        raise PermissionError("Producer output requires runtime invocation authority.")
    invocation.require_runtime_authority()
    redactor = SecretRedactor()
    index = prepare_contract(NativeProducerIndex, raw, redactor=redactor)
    if index.cleanup_receipt is not None:
        require_successor_invocation(index, invocation.active_profile)
        if invocation.binding.session_instance_id != index.session_instance_id:
            raise PermissionError("Successor invocation belongs to another session incarnation.")
        return None
    binding = index.invocation
    if binding is None or (
        invocation.binding.session_id,
        invocation.binding.session_instance_id,
    ) != (index.session_id, index.session_instance_id):
        raise PermissionError("Producer output invocation conflicts.")
    if invocation.profile.fingerprint != binding.profile_commitment.removeprefix("sha256:"):
        raise PermissionError("Producer output profile conflicts.")
    from cayu.runtime._invocation_lifecycle import require_invocation_rebind_lineage
    from cayu.sessions._execution_profile_checkpoint import (
        ActiveInvocationExecutionProfile,
    )

    require_invocation_rebind_lineage(
        checkpoint,
        session_instance_id=index.session_instance_id,
        original=ActiveInvocationExecutionProfile(
            session_id=session_id,
            interaction_id=binding.interaction_id,
            run_epoch=binding.run_epoch,
            profile=invocation.profile,
        ),
        current=invocation.active_profile,
    )
    attachment = prepare_contract(
        NativeProducerAttachment,
        await store.load_session_operation(session_id, index.operation_key),
        redactor=redactor,
    )
    require_exact_contract(
        attachment_index(attachment),
        index.model_copy(
            update={"state": "prepared", "invocation": None, "output_commitment": None}
        ),
        redactor=redactor,
    )
    from cayu.sessions._model_completion_publication import model_step_publication_from_checkpoint
    from cayu.sessions.checkpoints import decode_runtime_checkpoint

    pointer = model_step_publication_from_checkpoint(
        decode_runtime_checkpoint(checkpoint, session_id=session_id)
    )
    stage = await store.load_model_completion_stage(session_id, stage_id)
    if (
        pointer is None
        or pointer.stage_id != stage_id
        or stage is None
        or stage.state != "completed"
        or stage.publication is None
        or stage.source_run_epoch > invocation.binding.run_epoch
        or stage.purpose != "assistant-turn"
    ):
        raise ValueError("Producer output lacks its current native completion.")
    from cayu.runtime._producer_lineage import require_producer_epoch

    require_producer_epoch(index, checkpoint, invocation.profile, stage.source_run_epoch)
    publication = stage.publication
    receipt = await store.load_runtime_publication_receipt(session_id, publication.publication_id)
    if (
        receipt is None
        or receipt.interaction_id != binding.interaction_id
        or receipt.source_run_epoch != stage.source_run_epoch
    ):
        raise ValueError("Producer output publication conflicts.")
    messages = publication.transcript_messages
    indices = tuple(range(receipt.transcript_start_cursor, receipt.transcript_end_cursor))
    supported = (
        0 < len(indices) <= 16
        and bool(messages)
        and all(
            message.role == "assistant"
            and all(part.type in {"text", "provider_state", "thinking"} for part in message.content)
            for message in messages
        )
    )
    text_bytes = sum(
        len(part.text.encode("utf-8"))
        for message in messages
        for part in message.content
        if part.type == "text"
    )
    disposition = (
        "empty"
        if not text_bytes
        else "unsupported"
        if not supported
        else "oversized"
        if text_bytes > attachment.command.limits.output_bytes
        else "answer"
    )
    evidence = None
    if disposition == "answer":
        evidence = await read_native_output(
            store,
            session_id=session_id,
            session_instance_id=index.session_instance_id,
            invocation_id=binding.interaction_id,
            run_epoch=stage.source_run_epoch,
            stage_id=stage_id,
            source_indices=indices,
        )
    output = NativeProducerOutput(
        registration=attachment.command,
        stage_id=stage_id,
        publication_id=publication.publication_id,
        publication_commitment="sha256:" + receipt.publication_digest,
        interaction_id=binding.interaction_id,
        run_epoch=stage.source_run_epoch,
        source_indices=indices if evidence is not None else (),
        source_commitment=None if evidence is None else "sha256:" + evidence.source_commitment,
        disposition=disposition,
    )
    updated = prepare_contract(
        NativeProducerIndex,
        index.model_copy(
            update={
                "output_commitment": "sha256:"
                + sha256(contract_bytes(output, redactor=redactor)).hexdigest()
            }
        ),
        redactor=redactor,
    )

    def publish(session, checkpoint, existing):
        if (
            session.instance_id != index.session_instance_id
            or session.run_epoch != invocation.binding.run_epoch
        ):
            raise ValueError("Producer output lost its native invocation.")
        prior = prepare_contract(
            NativeProducerIndex,
            None if checkpoint is None else checkpoint.get(ROOT_KEY),
            redactor=redactor,
        )
        if prior not in (index, updated):
            raise ValueError("Producer output retention conflicts.")
        if existing is not None:
            require_exact_contract(
                output,
                prepare_contract(NativeProducerOutput, existing, redactor=redactor),
                redactor=redactor,
            )
        return SessionOperationPublication(
            checkpoint={**(checkpoint or {}), ROOT_KEY: updated.model_dump(mode="json")},
            operation_records={index.operation_key + ":output": output.model_dump(mode="json")},
        )

    with _publication_scope(updated):
        await store.publish_session_operation(
            session_id,
            idempotency_key=index.operation_key + ":output",
            operation_transform=publish,
            events=[],
        )
    return output


async def read_retained_native_output(store, command):
    """Read evidence for an independently authenticated owner, not a disclosure grant.

    Missing output remains unresolved production, not permission to rerun it.
    """
    from cayu.collaboration._native_output import read_native_output
    from cayu.sessions.base import _invocation_lifecycle_authority_read_scope

    if not store._supports_producer_attachment_protocol():
        raise NotImplementedError("Native producer output readback is not qualified.")
    redactor = SecretRedactor()
    command = prepare_contract(ProducerOutputRegistration, command, redactor=redactor)
    prepared = command.admission.prepared
    assert prepared is not None and isinstance(prepared.target, FreshRecipientAdmissionTarget)
    session_id = prepared.target.session_id
    with _invocation_lifecycle_authority_read_scope():
        checkpoint = await store.load_checkpoint(session_id)
    index = prepare_contract(
        NativeProducerIndex,
        None if checkpoint is None else checkpoint.get(ROOT_KEY),
        redactor=redactor,
    )
    attachment = prepare_contract(
        NativeProducerAttachment,
        await store.load_session_operation(session_id, index.operation_key),
        redactor=redactor,
    )
    require_exact_contract(command, attachment.command, redactor=redactor)
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
    raw = await store.load_session_operation(session_id, index.operation_key + ":output")
    if index.output_commitment is None and raw is None:
        from cayu.runtime._producer_failure import read_native_failure

        return await read_native_failure(store, attachment, index)
    output = prepare_contract(NativeProducerOutput, raw, redactor=redactor)
    require_exact_contract(command, output.registration, redactor=redactor)
    if (
        index.output_commitment
        != "sha256:" + sha256(contract_bytes(output, redactor=redactor)).hexdigest()
    ):
        raise ValueError("Retained producer output commitment conflicts.")
    invocation = index.invocation
    if invocation is None or output.interaction_id != invocation.interaction_id:
        raise ValueError("Retained producer output invocation conflicts.")
    from cayu.execution_profiles import (
        ExecutionProfileIdentity,
    )
    from cayu.runtime._invocation_lifecycle import require_invocation_rebind_lineage
    from cayu.sessions._execution_profile_checkpoint import (
        ActiveInvocationExecutionProfile,
    )

    profile = ExecutionProfileIdentity.model_validate_json(prepared.execution_profile_json)
    if invocation.profile_commitment != "sha256:" + profile.fingerprint:
        raise ValueError("Retained producer output profile conflicts.")
    original = ActiveInvocationExecutionProfile(
        session_id=session_id,
        interaction_id=invocation.interaction_id,
        run_epoch=invocation.run_epoch,
        profile=profile,
    )
    require_invocation_rebind_lineage(
        checkpoint,
        session_instance_id=index.session_instance_id,
        original=original,
        current=original.model_copy(update={"run_epoch": output.run_epoch}),
    )
    session = await store.load(session_id)
    if session is None or session.instance_id != index.session_instance_id:
        raise ValueError("Retained producer output session is unavailable.")
    stage = await store.load_model_completion_stage(session_id, output.stage_id)
    dispatch = await store.load_model_completion_stage_dispatch(session_id, output.stage_id)
    if (
        stage is None
        or stage.state != "completed"
        or stage.purpose != "assistant-turn"
        or stage.source_run_epoch != output.run_epoch
        or stage.publication is None
        or stage.publication.publication_id != output.publication_id
        or dispatch is None
        or dispatch.preparation_digest != stage.preparation_digest
        or dispatch.interaction_id != output.interaction_id
        or dispatch.source_run_epoch != output.run_epoch
    ):
        raise ValueError("Retained producer completion conflicts.")
    receipt = await store.load_runtime_publication_receipt(session_id, output.publication_id)
    if receipt is None or (
        receipt.interaction_id,
        receipt.source_run_epoch,
        "sha256:" + receipt.publication_digest,
    ) != (output.interaction_id, output.run_epoch, output.publication_commitment):
        raise ValueError("Retained producer publication conflicts.")
    if output.disposition == "answer":
        evidence = await read_native_output(
            store,
            session_id=session_id,
            session_instance_id=index.session_instance_id,
            invocation_id=output.interaction_id,
            run_epoch=output.run_epoch,
            stage_id=output.stage_id,
            source_indices=output.source_indices,
        )
        if (
            evidence.publication_id != output.publication_id
            or "sha256:" + evidence.source_commitment != output.source_commitment
        ):
            raise ValueError("Retained producer source conflicts.")
    return output


async def acknowledge_native_cleanup(store, registration, control, cleanup):
    """Release excluded retention with the same durable ACK owner as admitted work."""
    from cayu.runtime._producer_cleanup_receipt import _accepted_source_cleanup

    redactor = SecretRedactor()
    _, _, _, exclusion = native_exclusion(registration, control)
    cleanup = prepare_contract(ProducerCleanupRecord, cleanup, redactor=redactor)
    require_exact_contract(cleanup.exclusion, exclusion, redactor=redactor)
    if cleanup.registration != registration.command.operation or registration.cleanup != cleanup:
        raise ValueError("Source cleanup acknowledgement conflicts.")
    return await store._complete_native_producer_cleanup(
        registration, authority=_accepted_source_cleanup(registration)
    )
