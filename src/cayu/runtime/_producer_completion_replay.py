"""Native-owner handoff for a committed producer model result after process loss."""

from dataclasses import dataclass

from cayu.collaboration._preparation import prepare_contract, require_exact_contract
from cayu.runtime._invocation_lifecycle import InvocationContext
from cayu.runtime._producer_output_store import (
    ROOT_KEY,
    NativeProducerAttachment,
    NativeProducerIndex,
    attachment_index,
)
from cayu.vaults.redaction import SecretRedactor

_SEAL = object()


def producer_completion_requires_execution(checkpoint):
    """Detect a completion candidate before recovery mutates its invocation.

    This is only a requirement for current authorization, not permission to
    replay. The native owner separately authenticates the complete handoff.
    """
    from cayu.runtime._model_completion_publication import model_step_publication_from_checkpoint

    raw = None if checkpoint is None else checkpoint.get(ROOT_KEY)
    if raw is None:
        return False
    index = prepare_contract(NativeProducerIndex, raw, redactor=SecretRedactor())
    pointer = model_step_publication_from_checkpoint(checkpoint)
    return (
        index.state == "admitted"
        and index.cleanup_receipt is None
        and index.output_commitment is None
        and pointer is not None
        and pointer.classification.get("type") == "final"
        and pointer.tool_round_id is None
    )


@dataclass(frozen=True, repr=False)
class _ProducerCompletionReplay:
    """One authenticated result, not authority for another model or tool dispatch."""

    index: NativeProducerIndex
    stage_id: str
    source_run_epoch: int
    invocation: InvocationContext
    seal: object

    def require(self, invocation, boundary):
        if self.seal is not _SEAL or invocation is not self.invocation:
            raise PermissionError("Producer completion replay lost its native owner.")
        invocation._validate()
        stage, pointer = boundary.completed_stage, boundary.pointer
        original = self.index.invocation
        if (
            original is None
            or stage is None
            or pointer is None
            or stage.stage_id != self.stage_id
            or pointer.stage_id != self.stage_id
            or pointer.classification.get("type") != "final"
            or pointer.tool_round_id is not None
            or boundary.pending_tool_round is not None
            or stage.source_run_epoch != self.source_run_epoch
            or stage.publication is None
            or stage.publication.interaction_id != original.interaction_id
            or boundary.transcript_cursor != pointer.transcript_end_cursor
        ):
            raise PermissionError("Producer completion replay boundary changed.")


async def prepare_producer_completion_replay(store, session, checkpoint, invocation, boundary):
    """Qualify the exact retained terminal model result without selecting newer text.

    The common native loop must still run post-model policy/finalization gates.
    This does not elect an answer or create a work-attempt/admission grant.
    """
    from cayu.runtime._invocation_lifecycle import require_invocation_rebind_lineage
    from cayu.runtime.execution_profiles import ActiveInvocationExecutionProfile

    raw = None if checkpoint is None else checkpoint.get(ROOT_KEY)
    if raw is None or invocation is None:
        return None
    if not store._supports_producer_attachment_protocol():
        raise NotImplementedError("Producer completion recovery is not qualified.")
    redactor = SecretRedactor()
    index = prepare_contract(NativeProducerIndex, raw, redactor=redactor)
    if (
        index.state != "admitted"
        or index.output_commitment is not None
        or index.cleanup_receipt is not None
    ):
        return None
    stage, pointer = boundary.completed_stage, boundary.pointer
    if (
        stage is None
        or pointer is None
        or pointer.classification.get("type") != "final"
        or pointer.tool_round_id is not None
        or boundary.pending_tool_round is not None
    ):
        return None
    invocation._validate()
    attachment = prepare_contract(
        NativeProducerAttachment,
        await store.load_session_operation(session.id, index.operation_key),
        redactor=redactor,
    )
    require_exact_contract(
        attachment_index(attachment),
        index.model_copy(update={"state": "prepared", "invocation": None}),
        redactor=redactor,
    )
    original = index.invocation
    assert original is not None
    if (
        session.id != index.session_id
        or session.instance_id != index.session_instance_id
        or invocation.binding.session_id != session.id
        or invocation.binding.session_instance_id != session.instance_id
        or invocation.binding.run_epoch != session.run_epoch
        or "sha256:" + invocation.profile.fingerprint != original.profile_commitment
    ):
        raise PermissionError("Producer recovery invocation conflicts.")
    require_invocation_rebind_lineage(
        checkpoint,
        session_instance_id=session.instance_id,
        original=ActiveInvocationExecutionProfile(
            session_id=session.id,
            interaction_id=original.interaction_id,
            run_epoch=original.run_epoch,
            profile=invocation.profile,
        ),
        current=invocation.active_profile,
    )
    from cayu.runtime._producer_lineage import require_producer_epoch

    if stage.source_run_epoch > invocation.binding.run_epoch:
        raise PermissionError("Producer completion belongs to a future invocation.")
    require_producer_epoch(index, checkpoint, invocation.profile, stage.source_run_epoch)
    replay = _ProducerCompletionReplay(
        index, stage.stage_id, stage.source_run_epoch, invocation, _SEAL
    )
    replay.require(invocation, boundary)
    return replay
