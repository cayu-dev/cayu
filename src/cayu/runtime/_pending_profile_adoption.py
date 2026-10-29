"""Let a pending tool approval settle after the application's execution profile changed.

A paused invocation keeps the execution profile it started with. These parts let
a pending tool approval adopt the current profile after an application upgrade
(most often a new Cayu version):

- ``decide_pending_profile_adoption`` asks the application's
  ``ExecutionProfilePolicy`` (or the built-in rule for a Cayu-only change) whether
  the pending work may adopt the current profile, and records a refusal of an
  explicit request.
- ``require_paused_round_exposure`` refuses adoption when the paused round's
  frozen tool catalogue no longer matches the current agent's tools.
- ``revalidate_paused_tool_calls`` checks the paused calls again against the
  current tools and tool policy, so the approval still covers only the exact
  call and arguments it was requested for.
- ``prepare_adopting_continuation_command`` turns the claim into one typed
  admission that commits the adoption decision with the continued interaction.
"""

from __future__ import annotations

from collections.abc import Callable, Collection
from dataclasses import dataclass, replace
from hashlib import sha256
from typing import Any

from cayu._validation import (
    canonical_durable_json_bytes,
    copy_durable_json_object,
    copy_durable_metadata,
)
from cayu.approvals.tools import (
    PendingToolApproval,
    ToolApprovalProfileAdoptionRefused,
    ToolApprovalRequest,
    ToolPolicyEvidence,
)
from cayu.execution_profiles import (
    ExecutionProfileAdoptionIntent,
    ExecutionProfileComponentClass,
    ExecutionProfileDecision,
    ExecutionProfileDecisionKind,
    execution_profile_runtime_component,
)
from cayu.runtime import _approval_support as approval_support
from cayu.runtime import _execution_profile_admission as execution_profile_admission
from cayu.runtime import _execution_profile_continuation as execution_profile_continuation
from cayu.runtime import _runtime_records as runtime_records
from cayu.runtime._approved_tool_arguments import _approval_authority
from cayu.runtime._event_writer import RuntimeEventWriter
from cayu.runtime._execution_profile_classification import ExecutionProfileClassifier
from cayu.runtime._invocation_lifecycle import (
    InvocationContext,
    invocation_checkpoint_state_sha256,
)
from cayu.runtime._tool_invocation.admission import ToolInvocationAdmission
from cayu.runtime._tool_round_staging import _redactor_for_tool_calls
from cayu.runtime.execution_profiles import (
    ExecutionProfileAdoptionRejected,
    ExecutionProfileMigrationRequired,
)
from cayu.sessions import _pending_approval_reader as pending_approval_reader
from cayu.sessions._checkpoint_mutations import (
    RuntimePublicationMutation,
    runtime_publication_checkpoint_mutation,
)
from cayu.sessions._execution_profile_checkpoint import (
    ActiveInvocationExecutionProfile,
    active_invocation_execution_profile_from_checkpoint,
    execution_profile_from_session_metadata,
)
from cayu.sessions._invocation_lifecycle import (
    AdmitInvocationCommand,
    InvocationCheckpointPatch,
    RejectInvocationCommand,
)
from cayu.sessions.authority import SessionRunFenced
from cayu.sessions.checkpoints import (
    ACTIVE_INVOCATION_EXECUTION_PROFILE_CHECKPOINT_KEY,
    CHECKPOINT_SCHEMA_VERSION_KEY,
    INVOCATION_LIFECYCLE_RECEIPT_CHECKPOINT_KEY,
)
from cayu.sessions.records import Session, SessionRuntimeIdentity, SessionStatus, copy_session
from cayu.sessions.store import SessionStore
from cayu.sessions.transcript_input import DeferredInteractionInput
from cayu.tools.exposure import (
    ResolvedToolExposureAuthority,
    tool_capability_ceiling_from_session_metadata,
    validate_resolved_tool_exposure_authority,
)
from cayu.tools.policy import TOOL_POLICY_REAUTHORIZATION_METADATA_KEY, ToolPolicyDecision
from cayu.vaults.redaction import SecretRedactor

# Adoption moves the paused invocation to the current profile. A model target is
# changed only by an explicit ``ModelTarget`` resume at a clean boundary.
_NOT_ADOPTABLE_WHILE_PENDING = frozenset({ExecutionProfileComponentClass.PROVIDER_TARGET})


@dataclass(frozen=True)
class PendingProfileAdoption:
    """An accepted adoption for one paused invocation, not yet committed."""

    source: ActiveInvocationExecutionProfile
    target: ActiveInvocationExecutionProfile
    decision: ExecutionProfileDecision
    adopted_runtime_identity: SessionRuntimeIdentity | None

    def at_run_epoch(self, run_epoch: int) -> PendingProfileAdoption:
        return replace(
            self,
            source=self.source.model_copy(update={"run_epoch": run_epoch}),
            target=self.target.model_copy(update={"run_epoch": run_epoch}),
        )


def tool_approval_adoption_request_fingerprint(
    request: ToolApprovalRequest,
    *,
    redactor: SecretRedactor,
) -> str | None:
    """Bind an approval's adoption idempotency key to the exact decision it settles."""

    intent = request.profile_adoption
    if intent is None:
        return None
    document = intent.model_dump(mode="json")
    redactor.require_no_secret_keys(
        document, field_name="profile_adoption", match_short_substrings=True
    )
    if redactor.redact_json_values(document) != document:
        raise ValueError("Execution-profile adoption audit fields contain a workload secret.")
    material = {
        "kind": "tool-approval",
        "session_id": request.session_id,
        "approval_id": request.approval_id,
        "tool_round_id": request.tool_round_id,
        "tool_call_id": request.tool_call_id,
        "resolution_request_digest": approval_support.approval_resolution_request_digest(request),
        "profile_adoption": document,
    }
    return sha256(
        canonical_durable_json_bytes(material, "tool_approval_profile_adoption")
    ).hexdigest()


async def decide_pending_profile_adoption(
    *,
    session: Session,
    plan: execution_profile_admission.ExecutionProfileContinuationPlan,
    classifier: ExecutionProfileClassifier,
    intent: ExecutionProfileAdoptionIntent | None,
    adoption_request_fingerprint: str | None,
    session_store: SessionStore,
    event_writer: RuntimeEventWriter,
) -> PendingProfileAdoption | None:
    """Decide whether paused work may continue under the current profile.

    Returns ``None`` when nothing changed or adoption is not available, so the
    caller keeps its strict continuation check. Raises after durably recording
    the refusal when the policy rejects an explicit ``intent``.
    """

    changed = plan.changed_component_classes
    source = plan.snapshot
    if (
        not changed
        or _NOT_ADOPTABLE_WHILE_PENDING.intersection(changed)
        or source.session_id != session.id
        # Only a released pause can be admitted again, here under a new profile.
        or source.run_epoch != session.run_epoch - 1
        or execution_profile_from_session_metadata(session.metadata) != source.profile
        or await session_store.load_active_model_completion_stage(session.id) is not None
    ):
        return None
    decision = await classifier.classify(
        session=session,
        expected_profile=source.profile,
        candidate_profile=plan.candidate_profile,
        changed_component_classes=changed,
        intent=intent,
        adoption_request_fingerprint=adoption_request_fingerprint,
        target_changed=False,
        target_provider_name=session.provider_name,
        target_model=session.model,
    )
    if decision.kind is not ExecutionProfileDecisionKind.ADOPTED:
        # Compatible reuse can't move a paused invocation; the strict check decides.
        if intent is None or decision.kind not in {
            ExecutionProfileDecisionKind.MIGRATION_REQUIRED,
            ExecutionProfileDecisionKind.REJECTED,
        }:
            return None
        event_writer.prepare_exact_replay(decision.event)
        rejection = await session_store.apply_invocation_lifecycle_command(
            RejectInvocationCommand(
                session_id=session.id,
                expected_session_instance_id=session.instance_id,
                expected_statuses=(session.status,),
                expected_run_epoch=session.run_epoch,
                expected_profile=source.profile,
                candidate_profile=plan.candidate_profile,
                event=decision.event,
                decision=decision,
                expected_active_profile=source,
            )
        )
        if not rejection.replayed:
            await event_writer.fan_out_persisted([rejection.event])
        error_type = (
            ExecutionProfileMigrationRequired
            if decision.kind is ExecutionProfileDecisionKind.MIGRATION_REQUIRED
            else ExecutionProfileAdoptionRejected
        )
        raise error_type(
            session_id=session.id,
            expected_profile_fingerprint=source.profile.fingerprint,
            candidate_profile_fingerprint=plan.candidate_profile.fingerprint,
            changed_component_classes=changed,
            expected_profile=source.profile,
            candidate_profile=plan.candidate_profile,
            boundary="continuation",
            application_policy_configured=classifier.policy_identity is not None,
        )
    adopted_runtime_identity = None
    if ExecutionProfileComponentClass.RUNTIME in changed:
        # The same values the continuation used to build the candidate profile.
        adopted_runtime_identity = SessionRuntimeIdentity(
            runtime_name="cayu",
            runtime_version=execution_profile_continuation._runtime_version(),
            runtime_build_provenance=(
                execution_profile_continuation.current_runtime_build_provenance()
            ),
        )
        if plan.candidate_profile.component(
            ExecutionProfileComponentClass.RUNTIME
        ) != execution_profile_runtime_component(
            adopted_runtime_identity.runtime_name,
            adopted_runtime_identity.runtime_version,
            adopted_runtime_identity.runtime_build_provenance,
        ):
            return None
    event_writer.prepare_exact_replay(decision.event)
    return PendingProfileAdoption(
        source=source,
        target=source.model_copy(update={"profile": plan.candidate_profile}),
        decision=decision,
        adopted_runtime_identity=adopted_runtime_identity,
    )


def require_paused_round_exposure(
    tool_exposure: ResolvedToolExposureAuthority | None,
    registered_agent: runtime_records.RegisteredAgentState,
) -> None:
    """Refuse to settle a paused round whose tool catalogue no longer matches.

    The round froze the tools the model saw, and settling it either way (a
    denial still runs the round's allowed sibling calls) continues under that
    catalogue. After a tool was added, removed or redescribed it can only be
    settled by the application version that proposed it, so this is refused
    before any policy is asked or anything is claimed.
    """

    if tool_exposure is None:
        return
    try:
        validate_resolved_tool_exposure_authority(
            tool_exposure,
            registered_agent.tool_capabilities,
            catalogue_revision=registered_agent.tool_catalogue.revision,
        )
    except (TypeError, ValueError) as exc:
        raise ToolApprovalProfileAdoptionRefused(
            "The agent's tools were added, removed or redescribed since this call was "
            "proposed, so its paused round can't continue under the current application, "
            "whether approved or denied. Settle it with the application version that "
            "proposed it."
        ) from exc


async def revalidate_paused_tool_calls(
    *,
    approval: PendingToolApproval,
    request_metadata: dict[str, Any],
    recorded_call_ids: Collection[str],
    gating_arguments: dict[str, Any] | None,
    gating_reauthorized: bool,
    session: Session,
    invocation_context: InvocationContext,
    admission: ToolInvocationAdmission,
    redactor: SecretRedactor,
) -> None:
    """Refuse adoption unless every paused call would be authorized the same way now.

    Each call that would still run is checked against the current tool and tool
    policy, read-only, with the paused round's own run context. A call allowed
    before must still be allowed; a call sent to approval must be allowed or need
    exactly the approval it got. Anything else stays pending for a person to deny
    so the agent can propose it again.

    ``gating_arguments`` are the arguments the approved call will actually run
    with when they differ from the proposal (an edit, possibly pinned by an
    earlier claim); ``gating_reauthorized`` says they were already checked
    against the current tool and policy.
    """

    registered_agent = invocation_context.registered_agent
    pause_redactor = _redactor_for_tool_calls(
        redactor,
        registered_agent=registered_agent,
        tool_calls=[
            approval_support.tool_call_request_from_pending(pending)
            for pending in approval.tool_calls
        ],
    )
    for call in approval.tool_calls:
        original = approval_support.policy_result_from_pending_tool_call(call)
        if (
            call.tool_call_id in recorded_call_ids
            or pending_approval_reader.effective_tool_policy_evidence(call)
            is not ToolPolicyEvidence.AUTHORITATIVE
            or original is None
            or original.decision is ToolPolicyDecision.DENY
        ):
            continue
        name = call.tool_name
        if (
            call.targeted_tool_grant_id is not None
            or call.model_tool_name is not None
            or call.targeted_tool_invocation is not None
            or call.targeted_tool_rejection is not None
        ):
            raise ToolApprovalProfileAdoptionRefused(
                f"The paused call to {name} came through a targeted tool grant, which is bound "
                "to the previous execution profile."
            )
        registered_tool = registered_agent.executable_tool(name)
        if registered_tool is None:
            raise ToolApprovalProfileAdoptionRefused(
                f"The paused call's tool is no longer registered: {name}"
            )
        if registered_tool.child_session_recovery is not None:
            raise ToolApprovalProfileAdoptionRefused(
                f"The paused call to {name} runs a child session, which keeps its own profile."
            )
        arguments = call.arguments
        if call.tool_call_id == approval.tool_call_id:
            if gating_reauthorized:
                continue
            if gating_arguments is not None:
                arguments = gating_arguments
        result = await admission.authorize(
            session=session,
            registered_agent=registered_agent,
            registered_environment=invocation_context.registered_environment,
            tool_call=runtime_records.ToolCallRequest(
                id=call.tool_call_id, name=name, arguments=arguments
            ),
            request_metadata={**request_metadata, TOOL_POLICY_REAUTHORIZATION_METADATA_KEY: True},
            taint_labels=call.active_taint_labels,
            tool_round_id=approval.tool_round_id,
            recorded_scope=approval,
        )
        if result.decision is ToolPolicyDecision.ALLOW:
            continue
        if result.decision is ToolPolicyDecision.DENY:
            raise ToolApprovalProfileAdoptionRefused(
                f"The current tool or tool policy refuses the paused call to {name}."
            )
        reauthorized = copy_durable_metadata(pause_redactor.redact_json_values(result.metadata))
        if original.decision is not ToolPolicyDecision.REQUIRE_APPROVAL or _approval_authority(
            reauthorized
        ) != _approval_authority(original.metadata):
            raise ToolApprovalProfileAdoptionRefused(
                f"The current tool policy needs approval for the paused call to {name} that "
                "this approval did not cover."
            )


def prepare_adopting_continuation_command(
    session: Session,
    checkpoint: dict[str, Any] | None,
    *,
    expected_statuses: set[SessionStatus] | frozenset[SessionStatus],
    checkpoint_transform: Callable[[Session, dict[str, Any] | None], dict[str, Any] | None],
    adoption: PendingProfileAdoption,
    deferred_input: DeferredInteractionInput | None,
) -> AdmitInvocationCommand:
    """Prepare the claim of a paused invocation that adopts a new profile.

    It is the admission counterpart of a recovery rebind: the same exact
    checkpoint snapshot and ordinary checkpoint patch, but the store installs the
    adopted profile and commits the decision, and the session's recorded profile
    and Cayu runtime identity, in the same transaction.
    """

    source_checkpoint = (
        None if checkpoint is None else copy_durable_json_object(checkpoint, "checkpoint")
    )
    expected = active_invocation_execution_profile_from_checkpoint(source_checkpoint)
    if (
        expected is None
        or expected != adoption.source
        or expected.run_epoch != session.run_epoch - 1
    ):
        raise SessionRunFenced("The paused invocation changed before it adopted a new profile.")
    if deferred_input is not None and deferred_input.interaction_id != expected.interaction_id:
        raise SessionRunFenced("Deferred input belongs to another interaction.")
    transformed = checkpoint_transform(
        copy_session(session),
        None if source_checkpoint is None else copy_durable_json_object(source_checkpoint, "c"),
    )
    desired = None if transformed is None else copy_durable_json_object(transformed, "checkpoint")
    target = adoption.target.model_copy(update={"run_epoch": session.run_epoch + 1})
    if active_invocation_execution_profile_from_checkpoint(desired) != target:
        raise ValueError("Profile adoption transform produced conflicting authority.")
    for key in (CHECKPOINT_SCHEMA_VERSION_KEY, INVOCATION_LIFECYCLE_RECEIPT_CHECKPOINT_KEY):
        if canonical_durable_json_bytes(
            None if source_checkpoint is None else source_checkpoint.get(key), key
        ) != canonical_durable_json_bytes(None if desired is None else desired.get(key), key):
            raise ValueError("Profile adoption transform attempted to mutate lifecycle authority.")
    mutation = runtime_publication_checkpoint_mutation(source_checkpoint, desired)
    return AdmitInvocationCommand(
        session_id=session.id,
        expected_session_instance_id=session.instance_id,
        expected_statuses=tuple(sorted(expected_statuses, key=str)),
        expected_run_epoch=session.run_epoch,
        expected_checkpoint_sha256=invocation_checkpoint_state_sha256(source_checkpoint),
        target_active_profile=target,
        checkpoint_patch=InvocationCheckpointPatch(
            mutation=RuntimePublicationMutation(
                operations=tuple(
                    operation
                    for operation in mutation.operations
                    if operation.key != ACTIVE_INVOCATION_EXECUTION_PROFILE_CHECKPOINT_KEY
                )
            )
        ),
        # Input deferred with this interaction stays deferred, unchanged.
        interaction_source_messages=(
            () if deferred_input is None else tuple(deferred_input.source_messages)
        ),
        defer_interaction_source=deferred_input is not None,
        tool_capability_ceiling=tool_capability_ceiling_from_session_metadata(session.metadata),
        continued_interaction_id=expected.interaction_id,
        execution_profile_decision=adoption.decision,
        adopted_runtime_identity=adoption.adopted_runtime_identity,
        expected_active_profile=expected,
    )


__all__ = [
    "PendingProfileAdoption",
    "decide_pending_profile_adoption",
    "prepare_adopting_continuation_command",
    "require_paused_round_exposure",
    "revalidate_paused_tool_calls",
    "tool_approval_adoption_request_fingerprint",
]
