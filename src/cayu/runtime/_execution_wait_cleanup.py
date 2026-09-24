"""Administrative discharge of exact native wait debt; never renewed authority."""

from __future__ import annotations

from typing import TYPE_CHECKING

from cayu.collaboration._namespace_store import inspect_retirement
from cayu.collaboration._preparation import prepare_contract
from cayu.collaboration.access import CollaborationAccessContext
from cayu.collaboration.base import REQUEST_FAMILY
from cayu.collaboration.lifecycle import NamespaceRef
from cayu.collaboration.mandates import MandateAccessContext
from cayu.collaboration.participants import ParticipantRef
from cayu.collaboration.waits import CollaborationWait, ParticipantSessionWaitExclusionReceipt
from cayu.runtime._continuation_wait_settlement import acknowledge_retirement, retirement_receipt
from cayu.runtime._execution_to_wait import _build_execution_wait
from cayu.runtime._invocation_lifecycle import _invocation_lifecycle_receipt_from_checkpoint
from cayu.runtime._session_continuation import (
    ContinuationConflict,
    ContinuationReleasedRetirement,
    ContinuationRetirement,
    ContinuationUnavailable,
    ContinuationWait,
    continuation_digest,
)
from cayu.runtime._session_continuation_store import require_released_wait_invocation
from cayu.sessions.base import _invocation_lifecycle_authority_read_scope
from cayu.sessions.context_views import ParticipantSessionExecutionRequest

if TYPE_CHECKING:
    from cayu.applications import CayuApp


async def exclude_released_execution_wait(
    app: CayuApp,
    execution: ParticipantSessionExecutionRequest,
    wait: CollaborationWait,
    *,
    participant: ParticipantRef,
    context: CollaborationAccessContext,
    wait_context: MandateAccessContext,
) -> ParticipantSessionWaitExclusionReceipt:
    if type(execution) is not ParticipantSessionExecutionRequest:
        raise TypeError("Participant wait cleanup requires a typed execution request.")
    execution = ParticipantSessionExecutionRequest(
        request=execution.request,
        session_instance_id=execution.session_instance_id,
        execution_key=execution.execution_key,
    )
    if execution.request.session_id is None:
        raise ValueError("Execution wait cleanup requires an exact session.")
    context = prepare_contract(CollaborationAccessContext, context, redactor=app._secret_redactor)
    participants = app._participant_coordinator
    _, grant = participants._authorize(context, "request_control")
    participants._require_refs(grant, (), create=True)
    await participants.inspect(participant, context=context, action="administration")
    source, initialized = participants._ready()
    participants._capability(source, initialized, mutation=True, family=REQUEST_FAMILY)
    binding = await app.session_store.load_participant_session_binding(execution.request.session_id)
    if binding is None or binding.participant != participant:
        raise PermissionError("Execution wait is not owned by this participant.")
    # wait_context is immutable selection data here, not a disclosure grant.
    handoff = await _build_execution_wait(app, execution, wait, context=wait_context)
    native = await app.session_store.load_continuation_ticket(
        handoff.session_id,
        session_instance_id=handoff.session_instance_id,
        registration_key=handoff.intent.registration_key,
    )
    if native is None:
        raise ContinuationUnavailable("Execution wait has no receiving responsibility.")
    expected = ContinuationWait(
        **{name: getattr(native.preparation.intent, name) for name in ContinuationWait.model_fields}
    )
    if expected != handoff.intent or native.ticket.state not in {"ARMING", "WAITING", "RETIRED"}:
        raise ContinuationConflict("Cleanup requires the complete original execution and wait.")
    assert native.ticket.execution_admission_sha256 is not None
    execution_commitment = native.ticket.execution_admission_sha256
    assert handoff.registration is not None
    bound = handoff.registration.wait.model_copy(
        update={"delivery_ticket": native.preparation.intent}
    )
    release = None
    proof = native.released_retirement
    if proof is None:
        with _invocation_lifecycle_authority_read_scope():
            checkpoint = await app.session_store.load_checkpoint(handoff.session_id)
        identity = (
            f"{native.ticket.session_id}:{native.ticket.session_instance_id}:"
            f"{native.ticket.writer_generation}"
        )
        admission = _invocation_lifecycle_receipt_from_checkpoint(
            checkpoint, command_identity="admit:" + identity
        )
        release = _invocation_lifecycle_receipt_from_checkpoint(
            checkpoint, command_identity="release:" + identity
        )
        if (
            admission is None
            or admission.participant_permit_operation is None
            or admission.participant_permit_commitment is None
        ):
            raise ContinuationUnavailable("Execution wait lacks native permit consumption.")
        proof = require_released_wait_invocation(
            native.ticket,
            checkpoint,
            permit_operation=admission.participant_permit_operation,
            permit_commitment=admission.participant_permit_commitment,
        )
    identity = handoff.execution_identity
    if identity is None or proof.permit_operation != identity.operation_key:
        raise ContinuationConflict("Wait release belongs to another execution operation.")
    permit = await source._lookup_registered_permit(
        initialized, initialized.operation(identity.operation_key), redactor=app._secret_redactor
    )
    if permit is not None:
        if (
            permit.expected.intent.request.admission_commitment != execution_commitment
            or identity.permit_commitment(permit) != proof.permit_commitment
        ):
            raise ContinuationConflict("Wait cleanup conflicts with its execution permit.")
        from cayu.applications import _ParticipantExecutionSettlementReader

        await participants._store_result(
            source._settle_permit(
                initialized,
                permit.expected,
                reader=_ParticipantExecutionSettlementReader(
                    app, permit.expected, proof.permit_commitment
                ),
                redactor=app._secret_redactor,
            )
        )
    retired = await inspect_retirement(
        source,
        initialized,
        NamespaceRef(
            owner=bound.source_owner,
            namespace_incarnation=bound.operation.namespace_incarnation,
            generation=bound.operation.generation,
        ),
        app._secret_redactor,
    )
    if retired is not None:
        if native.retirement is None:
            assert release is not None
            # Fixed native release time makes competing exact cleanup attempts
            # agree. This is a quiescence anchor, not a new foreign wait decision.
            native = await handoff.owner.retire_released(
                ContinuationReleasedRetirement(
                    retirement=ContinuationRetirement(
                        ticket=native.ticket,
                        control_id="retired-wait:" + continuation_digest(bound.operation),
                        reason="unavailable",
                        retired_at=release.result_session.updated_at.isoformat(),
                    ),
                    permit_operation=proof.permit_operation,
                    permit_commitment=proof.permit_commitment,
                )
            )
        await acknowledge_retirement(handoff.owner, bound, native)
    else:
        await app._wait_coordinator._exclude_released_administrative(
            bound,
            context=context,
            continuation_owner=handoff.owner,
            permit_operation=proof.permit_operation,
            permit_commitment=proof.permit_commitment,
        )
    settled = await app.session_store.load_continuation_ticket(
        handoff.session_id,
        session_instance_id=handoff.session_instance_id,
        registration_key=handoff.intent.registration_key,
    )
    if settled is None or not settled.retirement_acknowledged:
        raise ContinuationUnavailable("Native cleanup acknowledgement remains pending.")
    return ParticipantSessionWaitExclusionReceipt(
        operation=bound.operation,
        session_id=handoff.session_id,
        session_instance_id=handoff.session_instance_id,
        registration_key=handoff.intent.registration_key,
        execution_admission_sha256=execution_commitment,
        retirement_sha256=continuation_digest(retirement_receipt(settled)),
    )
