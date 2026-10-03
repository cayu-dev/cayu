"""Explicit host-selected whole-turn stop, never a model/caller invocation grant."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from cayu.collaboration._preparation import prepare_contract
from cayu.collaboration.mandates import MandateAccessContext
from cayu.collaboration.waits import CollaborationWait, request_object_ref
from cayu.sessions._session_continuation import (
    ContinuationRecord,
    ContinuationWait,
    continuation_digest,
)

if TYPE_CHECKING:
    from cayu.applications import CayuApp
    from cayu.collaboration._wait_coordinator import WaitCoordinator
    from cayu.collaboration.access import CollaborationAccessContext
    from cayu.collaboration.participants import ParticipantRef
    from cayu.collaboration.waits import ParticipantSessionWaitExclusionReceipt
    from cayu.runtime._invocation_lifecycle import InvocationContext
    from cayu.runtime._session_continuation_owner import SessionContinuationOwner
    from cayu.sessions._participant_execution_identity import ParticipantExecutionIdentity
    from cayu.sessions.context_views import ParticipantSessionExecutionRequest


@dataclass(frozen=True, slots=True)
class _ExecutionWaitRegistration:
    coordinator: WaitCoordinator
    wait: CollaborationWait
    context: MandateAccessContext


@dataclass(frozen=True, slots=True)
class _ExecutionToWait:
    """Private registered-owner handoff passed explicitly through runtime calls.

    It does not reconstruct invocation authority from a checkpoint or accept it
    from a public request. The actual engine invokes park with its admitted
    context only after the complete assistant/tool turn permits stopping.
    """

    owner: SessionContinuationOwner
    intent: ContinuationWait
    session_id: str
    session_instance_id: str
    registration: _ExecutionWaitRegistration | None = None
    execution_identity: ParticipantExecutionIdentity | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "intent",
            prepare_contract(ContinuationWait, self.intent, redactor=self.owner.redactor),
        )

    @property
    def commitment(self) -> str:
        return continuation_digest(
            self.intent if self.registration is None else self.registration.wait
        )

    async def prepare(self, invocation: InvocationContext) -> ContinuationRecord:
        invocation.require_runtime_authority()
        if (
            invocation.binding.session_id != self.session_id
            or invocation.binding.session_instance_id != self.session_instance_id
        ):
            raise PermissionError("Execution wait belongs to another session incarnation.")
        prepared = await self.owner.prepare(self.intent, invocation=invocation)
        if self.registration is not None:
            wait = self.registration.wait.model_copy(
                update={"delivery_ticket": prepared.preparation.intent}
            )
            await self.registration.coordinator.register(wait, context=self.registration.context)
        return prepared

    async def park(self, invocation: InvocationContext) -> None:
        prepared = await self.prepare(invocation)
        await self.owner.park(prepared.ticket, invocation=invocation)

    async def parked_replay(self, *, permit_operation: str, permit_commitment: str) -> bool:
        """Recognize the exact released invocation, not a terminal status guess."""
        from cayu.runtime._session_continuation_store import require_released_wait_invocation
        from cayu.sessions._session_continuation import (
            ContinuationConflict,
            ContinuationUnavailable,
        )
        from cayu.sessions.base import _invocation_lifecycle_authority_read_scope

        record = await self.owner.store.load_continuation_ticket(
            self.session_id,
            session_instance_id=self.session_instance_id,
            registration_key=self.intent.registration_key,
        )
        if record is None:
            return False
        recorded_intent = ContinuationWait(
            **{
                name: getattr(record.preparation.intent, name)
                for name in ContinuationWait.model_fields
            }
        )
        if recorded_intent != self.intent:
            raise ContinuationConflict("Execution wait replay changed its retained intent.")
        if record.ticket.state != "WAITING":
            raise ContinuationUnavailable("Execution wait still requires boundary reconciliation.")
        with _invocation_lifecycle_authority_read_scope():
            checkpoint = await self.owner.store.load_checkpoint(self.session_id)
        require_released_wait_invocation(
            record.ticket,
            checkpoint,
            permit_operation=permit_operation,
            permit_commitment=permit_commitment,
        )
        if self.registration is not None:
            bound = self.registration.wait.model_copy(
                update={"delivery_ticket": record.preparation.intent}
            )
            retained = await self.registration.coordinator.inspect(
                bound, context=self.registration.context
            )
            if retained is None or retained.registration.wait != bound:
                raise ContinuationUnavailable("Execution wait source registration is unavailable.")
        return True


async def prepare_execution_wait(
    app: CayuApp,
    execution: ParticipantSessionExecutionRequest,
    wait: CollaborationWait,
    *,
    context: MandateAccessContext,
) -> _ExecutionToWait:
    """Build a registered receiving handoff after authenticating every source."""
    wait = prepare_contract(CollaborationWait, wait, redactor=app._secret_redactor)
    await app._wait_coordinator.authorize_registration(wait, context=context)
    return await _build_execution_wait(app, execution, wait, context=context)


async def _build_execution_wait(
    app: CayuApp,
    execution: ParticipantSessionExecutionRequest,
    wait: CollaborationWait,
    *,
    context: MandateAccessContext,
) -> _ExecutionToWait:
    """Construct exact data; callers separately authorize execution or cleanup."""
    from cayu.collaboration._capabilities import CapabilityDescriptor
    from cayu.collaboration._request_coordinator import _initiator
    from cayu.execution_profiles import (
        ExecutionProfileIdentity,
    )
    from cayu.runtime._session_continuation_owner import LATCH_FAMILY, SessionContinuationOwner
    from cayu.sessions._participant_execution_identity import participant_execution_identity

    wait = prepare_contract(CollaborationWait, wait, redactor=app._secret_redactor)
    context = prepare_contract(MandateAccessContext, context, redactor=app._secret_redactor)
    if wait.delivery_ticket is not None or execution.request.session_id is None:
        raise ValueError("Execution wait requires an unbound wait and an exact root session.")
    if wait.initiator != _initiator(context):
        raise PermissionError("Execution wait selection changed its original initiator.")
    binding = await app.session_store.load_participant_session_binding(execution.request.session_id)
    creation = await app.session_store.load_participant_session_creation_receipt(
        execution.request.session_id
    )
    if binding is None or creation is None or creation.binding != binding:
        raise PermissionError("Execution wait requires its exact participant binding.")
    profile = ExecutionProfileIdentity.model_validate_json(creation.execution_profile_json)
    identity = participant_execution_identity(
        execution,
        binding,
        execution_profile_fingerprint=profile.fingerprint,
        wait_commitment=continuation_digest(wait),
    )
    _, initialized = app._participant_coordinator._ready()
    owner = SessionContinuationOwner(
        store=app.session_store,
        owner=initialized.owner,
        receiver=app.collaboration_wait_latch_receiver(),
        receiver_capability=CapabilityDescriptor(
            owner=initialized.owner, mutations=(), readbacks=(LATCH_FAMILY,)
        ),
        redactor=app._secret_redactor,
        track=app._request_coordinator.owners.track,
    )
    intent = ContinuationWait(
        registration_key="execution-wait:" + continuation_digest(wait.operation),
        targets=tuple(
            request_object_ref(target.intent.selection.reference) for target in wait.targets
        ),
        predicate_kind=wait.predicate,
        predicate_version=wait.predicate_version,
        threshold=wait.threshold,
        deadline=wait.deadline,
        failure_policy=wait.failure_policy,
        service_policy=wait.service_policy,
        wait_edge_revision=wait.wait_edge_revision,
        purpose="explicit_execution_wait",
        execution_admission_sha256=identity.admission_commitment,
        collaboration_wait_sha256=continuation_digest(wait),
    )
    return _ExecutionToWait(
        owner=owner,
        intent=intent,
        session_id=execution.request.session_id,
        session_instance_id=execution.session_instance_id,
        registration=_ExecutionWaitRegistration(app._wait_coordinator, wait, context),
        execution_identity=identity,
    )


async def exclude_execution_wait(
    app: CayuApp,
    execution: ParticipantSessionExecutionRequest,
    wait: CollaborationWait,
    *,
    participant: ParticipantRef,
    context: CollaborationAccessContext,
    wait_context: MandateAccessContext,
) -> ParticipantSessionWaitExclusionReceipt:
    """Explicit return-and-report cleanup, never execution retry or status inference."""
    from cayu.runtime._execution_wait_cleanup import exclude_released_execution_wait

    return await exclude_released_execution_wait(
        app, execution, wait, participant=participant, context=context, wait_context=wait_context
    )
