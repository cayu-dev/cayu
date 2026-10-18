"""Own independent session forks, prompt succession and exact publication replay."""

from __future__ import annotations

import asyncio
import hashlib
import traceback as traceback_module
from collections.abc import AsyncGenerator, Callable
from dataclasses import dataclass
from datetime import datetime
from typing import Any, NoReturn

from cayu._task_wait import (
    ShieldedTaskOutcome,
    await_shielded_task_outcome,
    capture_awaitable_outcome,
    restore_task_cancellation_requests,
    unexpected_child_cancellation_error,
)
from cayu._validation import (
    canonical_durable_json_bytes,
    copy_durable_metadata,
    copy_durable_record,
    copy_session_metadata,
    require_clean_nonblank,
)
from cayu.approvals.user_input import user_input_lifecycle_authority_from_checkpoint
from cayu.budgets.base import BudgetPolicy
from cayu.configuration import RunDefaults
from cayu.context.base import validate_context_messages
from cayu.deadlines import (
    EXECUTION_DEADLINE_METADATA_KEY,
    current_execution_deadline,
    effective_deadline,
)
from cayu.egress.transitions import require_forkable_egress_authority_transition
from cayu.events import (
    Event,
    EventType,
    copy_event,
    event_with_runtime_envelope_authority,
    event_with_runtime_generated_id,
    event_with_runtime_nested_payload_authority,
    event_with_runtime_payload_authority,
)
from cayu.execution_profiles import (
    ExecutionProfileComponentClass,
    ExecutionProfileDecision,
    ExecutionProfileDecisionKind,
    ExecutionProfileIdentity,
    changed_execution_profile_components,
    execution_profile_with_component,
    execution_profile_with_tool_capability_ceiling,
    inherited_execution_profile_component_changes,
    unavailable_execution_profile_components,
)
from cayu.fork_controllers import (
    ForkControllerDenied,
    VerifiedForkController,
    fork_controller_host_tool_ceiling,
    fork_overlay_agent_name,
    fork_overlay_tool_capability_ceiling,
    require_fork_controller_host_tools,
    require_fork_controller_hosted_tools,
    require_fork_controller_model_runs,
    require_requested_fork_controller_host_tools,
    require_verified_fork_controller,
)
from cayu.messages import Message, MessageRole, detach_message
from cayu.providers.retry_policy import RetryPolicy
from cayu.runtime import _approval_support as approval_support
from cayu.runtime import _execution_profile_admission as execution_profile_admission
from cayu.runtime import _execution_profile_continuation as execution_profile_continuation
from cayu.runtime import _model_target as model_target
from cayu.runtime import (
    _prompt_transition,
    _session_checkpoint_admission,
    _session_execution_profile,
    _session_operation_state,
)
from cayu.runtime import _runtime_records as runtime_records
from cayu.runtime import _session_request_boundary as session_request_boundary
from cayu.runtime import _transcript as transcript_helpers
from cayu.runtime._checkpoint_store import advance_task_retention_epoch
from cayu.runtime._child_session_identity import ChildSessionKind, generate_child_session_id
from cayu.runtime._durable_tool_round import _environment_name as _environment_name
from cayu.runtime._environment_allocation import (
    environment_allocation_owners_from_checkpoint,
    require_authorized_environment_allocation_owners,
)
from cayu.runtime._environment_lifecycle import (
    EnvironmentLifecycle,
    render_initial_system_prompt_with_contributions,
)
from cayu.runtime._event_writer import RuntimeEventWriter
from cayu.runtime._execution_profile_classification import ExecutionProfileClassifier
from cayu.runtime._execution_profile_continuation import ExecutionProfileContinuation
from cayu.runtime._fork_source_snapshot import (
    fork_source_checkpoint_sha256,
    strip_source_owned_fork_checkpoint_state,
)
from cayu.runtime._interruption_coordinator import (
    _PENDING_INTERRUPTION_CASCADE_CHECKPOINT_KEY,
    _PENDING_SESSION_INTERRUPT_CHECKPOINT_KEY,
)
from cayu.runtime._model_step_executor import (
    _model_request_tools,
    _tool_capability_ceiling_exposure,
)
from cayu.runtime._recovery_ownership import RecoveryOwnership
from cayu.runtime._session_tool_discovery import (
    _require_tool_discovery_view,
    _tool_discovery_operation_initializer,
)
from cayu.runtime._task_execution_admission import require_ordinary_session_execution
from cayu.runtime._tool_invocation.admission import ToolInvocationAdmission
from cayu.runtime.build_provenance import current_runtime_build_provenance
from cayu.runtime.execution_identity import ExecutionProfileBehaviorIdentity
from cayu.runtime.execution_profiles import (
    ExecutionProfileAdoptionRejected,
    ExecutionProfileMigrationRequired,
)
from cayu.runtime.loop_policies import LoopPolicy
from cayu.sessions import _pending_approval_reader as pending_approval_reader
from cayu.sessions import _pending_tool_round_reader as pending_round_reader
from cayu.sessions._checkpoint_publication import _initial_transcript_pending_interaction_id
from cayu.sessions._execution_profile_checkpoint import (
    EXECUTION_PROFILE_METADATA_KEY,
    execution_profile_session_metadata,
    model_failover_progress_for_session,
)
from cayu.sessions._execution_profile_rules import (
    _session_metadata_with_model_projection,
    session_model_projection_cursor,
)
from cayu.sessions._model_failover import ModelFailoverPolicy, ModelTarget
from cayu.sessions.authority import SessionRunFenced
from cayu.sessions.event_queries import EventQuery
from cayu.sessions.fork_overlays import (
    FORK_LINEAGE_METADATA_KEY,
    FORK_OVERLAY_METADATA_KEY,
    ForkOverlayRecord,
    fork_lineage_for_fork,
    resolve_fork_lineage_async,
)
from cayu.sessions.forks import (
    FORK_EXECUTION_PROFILE_EXACT_SOURCE_SCHEMA_VERSION,
    FORK_EXECUTION_PROFILE_METADATA_KEY,
    FORK_EXECUTION_PROFILE_ORDINARY_SCHEMA_VERSION,
    FORK_SOURCE_SNAPSHOT_METADATA_KEY,
    FORK_TRANSCRIPT_VALIDATION_ERROR,
    PROMPT_ANATOMY_TRANSITION_METADATA_KEY,
    ForkExecutionProfileDecisionRecord,
    ForkExecutionProfileSelection,
    ForkRequestConflict,
    ForkSystemPromptPolicy,
    ForkSystemPromptReplacement,
    ProfiledSessionForkResult,
    PromptAnatomyTransitionReceipt,
    SessionForkActiveModelStageConflict,
    SessionForkEnvironmentAllocationOwner,
    SessionForkProfileRelationship,
    SessionForkSourceNotFound,
    apply_fork_system_prompt_replacement,
    copy_profiled_session_fork_result,
    fork_session_invocation,
    fork_source_state_sha256,
    session_fork_profile_relationship,
    session_prompt_anatomy_transition,
    validate_profiled_fork_evidence,
)
from cayu.sessions.records import (
    RUNTIME_BUILD_PROVENANCE_METADATA_KEY,
    Session,
    SessionRuntimeIdentity,
    SessionStatus,
    _fork_source_session_instance_fingerprint,
)
from cayu.sessions.recovery import _incomplete_recovery_claim_from_checkpoint
from cayu.sessions.requests import (
    ForkSessionRequest,
    _fork_initial_invocation_request_sha256,
    copy_fork_session_request,
)
from cayu.sessions.store import SessionStore
from cayu.sessions.transcript_input import system_prompt_messages_sha256
from cayu.sessions.transcript_queries import TranscriptSnapshot, fork_source_transcript_sha256
from cayu.tasks.store import TaskStore
from cayu.tools.exposure import (
    ToolCapabilityCeiling,
    resolve_tool_capability_ceiling,
    session_metadata_with_tool_capability_ceiling,
    tool_capability_ceiling_from_session_metadata,
)
from cayu.tools.policy import metadata_with_taint_labels, taint_labels_from_metadata
from cayu.tools.targeted_projection import resolve_targeted_tool_projection
from cayu.vaults.redaction import SecretRedactor
from cayu.workspaces.observation_recovery import (
    retain_workspace_observation_pending_cancellation_requests,
)
from cayu.workspaces.revisions import WorkspaceForkLineage, WorkspaceForkLineageStatus

_FORKABLE_SESSION_STATUSES = _session_checkpoint_admission._RESUMABLE_SESSION_STATUSES


def _validate_fork_source_checkpoint_state(
    current_source: Session,
    source_checkpoint: dict[str, Any] | None,
    *,
    copy_checkpoint: bool,
    redactor: SecretRedactor,
) -> dict[str, Any] | None:
    """Reject source state that no direct session fork may inherit."""

    if _incomplete_recovery_claim_from_checkpoint(source_checkpoint) is not None:
        raise RuntimeError("Session has an active incomplete-session recovery operation.")
    if _initial_transcript_pending_interaction_id(source_checkpoint) is not None:
        raise RuntimeError(
            "Source session setup did not publish its authoritative initial "
            "transcript; fork fails closed. Start a new session."
        )
    if (
        source_checkpoint is not None
        and _session_operation_state._SESSION_OPERATIONS_CHECKPOINT_KEY in source_checkpoint
    ):
        operations = _session_operation_state._session_operation_state(source_checkpoint)
        source_checkpoint[_session_operation_state._SESSION_OPERATIONS_CHECKPOINT_KEY] = operations
    active_operation_id = _session_operation_state._active_session_operation_id(source_checkpoint)
    if active_operation_id is not None:
        raise RuntimeError(f"Session has an active durable operation: {active_operation_id}")
    _session_checkpoint_admission._reject_pending_provider_operation_disposition(source_checkpoint)
    _, source_profile, _ = session_request_boundary.prepare_fork_source_execution_profile(
        current_source,
        source_checkpoint,
    )
    require_forkable_egress_authority_transition(
        source_checkpoint,
        session_id=current_source.id,
        environment_name=current_source.environment_name,
        profile_authority=source_profile.egress_authority,
    )
    if (
        source_checkpoint is not None
        and _PENDING_SESSION_INTERRUPT_CHECKPOINT_KEY in source_checkpoint
    ):
        raise RuntimeError(
            "Session has incomplete terminal interruption evidence. "
            "Recover the session before forking it."
        )
    if current_source.status == SessionStatus.INTERRUPTED and not copy_checkpoint:
        raise ValueError("Interrupted sessions cannot be forked without checkpoint state.")
    if current_source.status == SessionStatus.INTERRUPTED and source_checkpoint is None:
        raise RuntimeError(
            "Interrupted session cannot be forked because checkpoint state is missing."
        )
    pending_user_input, _ = user_input_lifecycle_authority_from_checkpoint(
        source_checkpoint,
        redactor=redactor,
        consume_on_rejection=True,
        current_run_epoch=current_source.run_epoch,
        runtime_session=current_source,
    )
    if pending_user_input is not None:
        raise RuntimeError(
            "Session awaiting user input cannot be forked; answer it with "
            "resolve_user_input(...) first."
        )
    if (
        pending_approval_reader.pending_approval_from_checkpoint(
            source_checkpoint,
            redactor=redactor,
            consume_on_rejection=True,
        )
        is not None
    ):
        raise RuntimeError(
            "Session awaiting tool approval cannot be forked; resolve it with "
            "resolve_tool_approval(...) first."
        )
    if (
        pending_round_reader.pending_tool_round_from_checkpoint(
            source_checkpoint,
            redactor=redactor,
            consume_on_rejection=True,
        )
        is not None
    ):
        raise RuntimeError(
            "Session with a pending tool round cannot be forked; resume or "
            "recover the session first."
        )
    if (
        source_checkpoint is not None
        and _PENDING_INTERRUPTION_CASCADE_CHECKPOINT_KEY in source_checkpoint
    ):
        raise RuntimeError(
            "Session has an incomplete background interruption cascade. "
            "Retry the interruption before forking it."
        )
    return source_checkpoint


def _prompt_fork_matches_prepared(existing: Session, prepared: Session) -> bool:
    return (
        existing.id,
        existing.agent_name,
        existing.provider_name,
        existing.model,
        existing.parent_session_id,
        existing.causal_budget_id,
        existing.runtime_name,
        existing.runtime_version,
        existing.environment_name,
        existing.status,
        existing.labels,
        existing.metadata,
        existing.run_epoch,
    ) == (
        prepared.id,
        prepared.agent_name,
        prepared.provider_name,
        prepared.model,
        prepared.parent_session_id,
        prepared.causal_budget_id,
        prepared.runtime_name,
        prepared.runtime_version,
        prepared.environment_name,
        prepared.status,
        prepared.labels,
        prepared.metadata,
        prepared.run_epoch,
    )


@dataclass(frozen=True, slots=True)
class _PromptSuccessionWorkflow:
    """Cohesive durable state and event projection for one prompt succession."""

    request: ForkSessionRequest
    request_sha256: str
    receipt: PromptAnatomyTransitionReceipt

    @classmethod
    def recover(
        cls,
        *,
        request: ForkSessionRequest,
        accepted_request_sha256s: tuple[str, ...],
        descendant: Session,
    ) -> _PromptSuccessionWorkflow:
        receipt = session_prompt_anatomy_transition(descendant)
        if receipt is None or receipt.request_sha256 not in accepted_request_sha256s:
            raise ForkRequestConflict(
                "Existing prompt-anatomy descendant conflicts with the exact request."
            )
        return cls(
            request=request,
            request_sha256=receipt.request_sha256,
            receipt=receipt,
        )

    def prepared_intent(
        self,
        *,
        source_run_epoch: int,
        source_snapshot_cursor: int,
        requested_at: datetime,
    ) -> _prompt_transition._PromptTransitionIntent:
        return _prompt_transition._PromptTransitionIntent.prepared(
            receipt=self.receipt,
            source_run_epoch=source_run_epoch,
            source_snapshot_cursor=source_snapshot_cursor,
            requested_at=requested_at,
        )

    def raw_fork_event(self, descendant: Session) -> Event:
        receipt = self.receipt
        payload: dict[str, Any] = {
            "source_session_id": receipt.source_session_id,
            "source_status": receipt.source_status.value,
            "parent_session_id": descendant.parent_session_id,
            "causal_budget_id": descendant.causal_budget_id,
            "transcript_cursor": self.request.transcript_cursor,
            "copy_checkpoint": receipt.copy_checkpoint,
            "agent_name": descendant.agent_name,
            "provider_name": descendant.provider_name,
            "model": descendant.model,
            "environment_name": descendant.environment_name,
            "inherited_taint_labels": list(receipt.inherited_taint_labels),
            "prompt_anatomy_transition": {
                "transition_id": receipt.transition_id,
                "source_transcript_cursor": receipt.source_transcript_cursor,
                "source_prompt_sha256": receipt.source_prompt_sha256,
                "child_prompt_sha256": receipt.child_prompt_sha256,
                "source_system_prompt_retained": receipt.source_system_prompt_retained,
                "policy": receipt.policy.value,
            },
        }
        return event_with_runtime_generated_id(
            Event(
                id=f"prompt-transition-{receipt.transition_id}",
                type=EventType.SESSION_FORKED,
                session_id=descendant.id,
                timestamp=descendant.created_at,
                agent_name=descendant.agent_name,
                environment_name=descendant.environment_name,
                payload=payload,
            )
        )


@dataclass(slots=True)
class _ForkPromptWorkflow:
    """Own the prompt-policy branch of session forking behind one strategy seam."""

    request: ForkSessionRequest
    request_sha256: str
    prompt_replacement: ForkSystemPromptReplacement | None = None
    child_prompt_messages: tuple[Message, ...] = ()
    rendered_child_prompt: str | None = None
    prompt_environment_name: str | None = None
    succession: _PromptSuccessionWorkflow | None = None
    intent: _prompt_transition._PromptTransitionIntent | None = None

    @classmethod
    async def prepare(
        cls,
        *,
        request: ForkSessionRequest,
        request_sha256: str,
        registered_agent: runtime_records.RegisteredAgentState,
        registered_environment: runtime_records.RegisteredEnvironment | None,
        environment_lifecycle: EnvironmentLifecycle,
    ) -> _ForkPromptWorkflow:
        workflow = cls(request=request, request_sha256=request_sha256)
        if request.system_prompt_policy is not ForkSystemPromptPolicy.CURRENT_AGENT:
            return workflow
        if request.copy_checkpoint:
            raise ValueError(
                "Explicit prompt-anatomy succession cannot copy checkpoint state; "
                "set copy_checkpoint=False."
            )
        if registered_environment is None or registered_environment.factory is not None:
            raise ValueError(
                "Explicit prompt-anatomy succession requires a concrete registered "
                "environment so current workspace instructions can be rendered."
            )
        workflow.prompt_environment_name = registered_environment.spec.name
        workspace_instructions = await environment_lifecycle.load_workspace_instructions(
            registered_environment,
        )
        rendered_child_prompt, _prompt_contributions = (
            render_initial_system_prompt_with_contributions(
                agent_system_prompt=registered_agent.spec.system_prompt,
                workspace_instructions=workspace_instructions,
            )
        )
        workflow.rendered_child_prompt = rendered_child_prompt
        workflow.child_prompt_messages = tuple(
            transcript_helpers.initial_messages(
                system_prompt=rendered_child_prompt,
                request_messages=[],
            )
        )
        workflow.prompt_replacement = ForkSystemPromptReplacement(
            message=(workflow.child_prompt_messages[0] if workflow.child_prompt_messages else None),
        )
        return workflow

    def requires_transcript_snapshot(
        self,
        *,
        target_changed: bool,
        source_projection_cursor: int,
    ) -> bool:
        return (
            self.prompt_replacement is not None
            or self.request.transcript_cursor is not None
            or target_changed
            or bool(source_projection_cursor)
        )

    def require_inherited_system_projection(
        self,
        *,
        source_messages: list[Message],
        selected_messages: list[Message],
    ) -> None:
        """Keep an inherited profile coherent with the copied transcript."""

        if self.prompt_replacement is not None:
            return
        source_system = [
            message for message in source_messages if message.role is MessageRole.SYSTEM
        ]
        selected_system = [
            message for message in selected_messages if message.role is MessageRole.SYSTEM
        ]
        if selected_system != source_system:
            raise ValueError(
                "A partial fork that inherits the parent execution profile must retain "
                "the complete durable system projection. Increase transcript_cursor or "
                "select the current child prompt explicitly."
            )

    def project_selected_messages(
        self,
        messages: list[Message],
        interaction_ids: list[str | None],
    ) -> tuple[list[Message], str | None]:
        if self.prompt_replacement is None:
            return messages, None
        projected, projected_interaction_ids = apply_fork_system_prompt_replacement(
            messages,
            interaction_ids,
            self.prompt_replacement,
        )
        projected_interaction_ids.clear()
        return projected, system_prompt_messages_sha256(self.child_prompt_messages)

    def requires_portability_validation(self, *, target_changed: bool) -> bool:
        return target_changed or self.prompt_replacement is not None

    def attach_receipt(
        self,
        metadata: dict[str, Any],
        *,
        source_session: Session,
        destination_session_id: str,
        selected_source_cursor: int,
        source_prompt_sha256: str | None,
        child_prompt_sha256: str | None,
        child_agent_name: str,
        provider_name: str,
        model: str,
        model_changed: bool,
        inherited_taint_labels: set[str] | frozenset[str],
    ) -> None:
        if self.prompt_replacement is None:
            return
        if source_prompt_sha256 is None or child_prompt_sha256 is None:
            raise AssertionError("Prompt succession lost its prompt digests.")
        if self.prompt_environment_name is None:
            raise AssertionError("Prompt succession lost its concrete environment identity.")
        receipt = PromptAnatomyTransitionReceipt.create(
            request_sha256=self.request_sha256,
            source_session_id=source_session.id,
            descendant_session_id=destination_session_id,
            source_status=source_session.status,
            source_transcript_cursor=selected_source_cursor,
            source_prompt_sha256=source_prompt_sha256,
            child_prompt_sha256=child_prompt_sha256,
            child_agent_name=child_agent_name,
            child_environment_name=self.prompt_environment_name,
            copy_checkpoint=self.request.copy_checkpoint,
            source_provider_name=source_session.provider_name,
            source_model=source_session.model,
            provider_name=provider_name,
            model=model,
            model_target_changed=model_changed,
            portability_preflight=(
                "provider_portable_transcript_preflight"
                if model_changed
                else "context_messages_validated"
            ),
            inherited_taint_labels=tuple(sorted(inherited_taint_labels)),
        )
        self.succession = _PromptSuccessionWorkflow(
            request=self.request,
            request_sha256=self.request_sha256,
            receipt=receipt,
        )
        metadata[PROMPT_ANATOMY_TRANSITION_METADATA_KEY] = receipt.model_dump(mode="json")

    def require_prepared_intent(self, source_checkpoint: dict[str, Any] | None) -> None:
        if self.succession is None:
            return
        if self.intent is None or source_checkpoint is None:
            raise RuntimeError("Prompt-anatomy transition intent disappeared.")
        intent_ledger = _prompt_transition._PromptTransitionIntentLedger.from_checkpoint(
            source_checkpoint,
            required=True,
        )
        intent_ledger.require_exact(self.intent)

    async def prepare_effect(
        self,
        *,
        session_store: SessionStore,
        source_session: Session,
        source_snapshot_cursor: int,
        fork_session: Session,
        validate_source_checkpoint: Callable[
            [Session, dict[str, Any] | None], dict[str, Any] | None
        ],
        clock: Callable[[], datetime],
    ) -> Session:
        if self.succession is None:
            return fork_session
        self.intent = self.succession.prepared_intent(
            source_run_epoch=source_session.run_epoch,
            source_snapshot_cursor=source_snapshot_cursor,
            requested_at=clock(),
        )

        def persist_prompt_transition_intent(
            current_source: Session,
            source_checkpoint: dict[str, Any] | None,
        ) -> dict[str, Any]:
            validated = validate_source_checkpoint(current_source, source_checkpoint)
            updated = {} if validated is None else copy_durable_record(validated, "checkpoint")
            intent_ledger = _prompt_transition._PromptTransitionIntentLedger.from_checkpoint(
                updated,
                required=False,
            )
            if self.intent is None:
                raise AssertionError("Prompt succession lost its durable intent identity.")
            intent_ledger.prepare(self.intent)
            intent_ledger.store_in(updated)
            return updated

        await session_store.publish_checkpoint_and_events(
            source_session.id,
            checkpoint_transform=persist_prompt_transition_intent,
            events=[],
            expected_statuses=_FORKABLE_SESSION_STATUSES,
            expected_run_epoch=source_session.run_epoch,
            expected_transcript_cursor=source_snapshot_cursor,
        )
        descendant_created_at = clock()
        return fork_session.model_copy(
            update={
                "created_at": descendant_created_at,
                "updated_at": descendant_created_at,
            }
        )

    async def complete_effect(
        self,
        *,
        session_store: SessionStore,
        source_session_id: str,
        created: Session,
        clock: Callable[[], datetime],
    ) -> None:
        if self.succession is None:
            return
        if self.intent is None:
            raise AssertionError("Prompt succession lost its durable intent identity.")

        def complete_prompt_transition_intent(
            _current_source: Session,
            source_checkpoint: dict[str, Any] | None,
        ) -> dict[str, Any]:
            if source_checkpoint is None:
                raise RuntimeError("Prompt-anatomy transition intent disappeared.")
            updated = copy_durable_record(source_checkpoint, "checkpoint")
            intent_ledger = _prompt_transition._PromptTransitionIntentLedger.from_checkpoint(
                updated,
                required=True,
            )
            if self.intent is None:
                raise AssertionError("Prompt succession lost its durable intent identity.")
            intent_ledger.complete(
                self.intent,
                descendant_created_at=created.created_at,
                completed_at=clock(),
            )
            intent_ledger.store_in(updated)
            return updated

        await session_store.publish_checkpoint_and_events(
            source_session_id,
            checkpoint_transform=complete_prompt_transition_intent,
            events=[],
        )

    def raw_fork_event(
        self,
        *,
        created: Session,
        ordinary_event: Event,
    ) -> Event:
        if self.succession is None:
            return ordinary_event
        return self.succession.raw_fork_event(created)


def _fork_event_with_runtime_authority(event: Event) -> Event:
    payload_fields = [
        "source_session_id",
        "parent_session_id",
        "causal_budget_id",
        "execution_profile_selection",
        "selected_profile_fingerprint",
        "source_profile_fingerprint",
        "fork_request_sha256",
        "system_prompt_policy",
    ]
    if "fork_overlay_fingerprint" in event.payload:
        payload_fields.extend(("fork_overlay_controller", "fork_overlay_fingerprint"))
    if "initial_invocation_request_sha256" in event.payload:
        payload_fields.extend(
            (
                "initial_dispatch_id",
                "initial_invocation_request_sha256",
                "initial_invocation_profile_fingerprint",
            )
        )
    if "source_checkpoint_sha256" in event.payload:
        payload_fields.extend(
            (
                "source_instance_fingerprint",
                "source_transcript_sha256",
                "source_checkpoint_sha256",
                "source_execution_profile_fingerprint",
            )
        )
    event = event_with_runtime_payload_authority(
        event_with_runtime_envelope_authority(event, "session_id"),
        *payload_fields,
    )
    return event_with_runtime_nested_payload_authority(
        event,
        ("source_environment_allocation_owners", "*", "environment_name"),
        ("source_environment_allocation_owners", "*", "owner_session_id"),
    )


# Per-invocation profile parts a fork's first invocation may set freely.
_FORK_OVERLAY_INVOCATION_COMPONENTS = frozenset(
    {
        ExecutionProfileComponentClass.RUNTIME,
        ExecutionProfileComponentClass.INVOCATION_BUDGET_POLICY,
        ExecutionProfileComponentClass.STRUCTURED_OUTPUT,
        ExecutionProfileComponentClass.FINALIZATION,
    }
)


class SessionFork:
    """Own child preparation, durable publication, cancellation and exact replay."""

    def __init__(
        self,
        *,
        session_store: SessionStore,
        task_store: TaskStore | None,
        profile_classifier: ExecutionProfileClassifier,
        execution_profile_continuation: ExecutionProfileContinuation,
        environment_lifecycle: EnvironmentLifecycle,
        event_writer: RuntimeEventWriter,
        recovery_ownership: RecoveryOwnership,
        tool_admission: ToolInvocationAdmission,
        secret_redactor: SecretRedactor,
        clock: Callable[[], datetime],
        runtime_hooks: tuple[runtime_records.RegisteredRuntimeHook, ...],
        loop_policies: tuple[LoopPolicy, ...],
        loop_policy_execution_profile_identities: tuple[
            ExecutionProfileBehaviorIdentity | None, ...
        ],
        get_budget_policy: Callable[[], BudgetPolicy | None],
        get_registered_agent: Callable[[str], runtime_records.RegisteredAgentState],
        get_registered_provider: Callable[[str | None], runtime_records.RegisteredProvider],
        get_registered_environment_for_session: Callable[
            [str | None], runtime_records.RegisteredEnvironment | None
        ],
        effective_retry_policy: Callable[[RetryPolicy | None], RetryPolicy],
        application_run_defaults: RunDefaults,
        execution_profile_process_identity: str,
    ) -> None:
        self.session_store = session_store
        self._task_store = task_store
        self._profile_classifier = profile_classifier
        self._execution_profile_continuation = execution_profile_continuation
        self._environment_lifecycle = environment_lifecycle
        self._event_writer = event_writer
        self._recovery_ownership = recovery_ownership
        self._tool_admission = tool_admission
        self._secret_redactor = secret_redactor
        self._clock = clock
        self._runtime_hooks = runtime_hooks
        self._loop_policies = loop_policies
        if len(loop_policies) != len(loop_policy_execution_profile_identities):
            raise ValueError("Loop-policy execution-profile identities are inconsistent.")
        self._loop_policy_execution_profile_identities = loop_policy_execution_profile_identities
        self._get_budget_policy = get_budget_policy
        self._get_registered_agent = get_registered_agent
        self._get_registered_provider = get_registered_provider
        self._get_registered_environment_for_session = get_registered_environment_for_session
        self._effective_retry_policy = effective_retry_policy
        if type(application_run_defaults) is not RunDefaults:
            raise TypeError("application_run_defaults must be a RunDefaults.")
        self._application_run_defaults = application_run_defaults
        self._execution_profile_process_identity = require_clean_nonblank(
            execution_profile_process_identity,
            "execution_profile_process_identity",
        )

    async def preflight_source(self, session_id: str) -> None:
        """Inspect source task authority using the operation's configured store."""

        await require_ordinary_session_execution(
            session_id,
            admit_session=False,
            task_store=self._task_store,
            redactor=self._secret_redactor,
        )

    async def preflight_fork_source_checkpoint_state(
        self,
        source: Session,
        checkpoint: dict[str, Any] | None,
    ) -> None:
        """Apply source-state rules before an independent child is published."""

        if source.status not in _FORKABLE_SESSION_STATUSES:
            raise ValueError(
                f"Only completed, failed, or interrupted sessions can be forked: {source.status}"
            )
        active_model_stage = await self.session_store.load_active_model_completion_stage(source.id)
        if active_model_stage is not None:
            del active_model_stage
            raise session_request_boundary.ForkActiveModelStageError(
                "Fork source session has an active model-completion stage."
            ) from None
        await self._settle_fork_source_expiring_authority(source.id)
        checkpoint = await self.session_store.load_checkpoint(source.id)
        copied_checkpoint = (
            None if checkpoint is None else copy_durable_record(checkpoint, "checkpoint")
        )
        try:
            _validate_fork_source_checkpoint_state(
                source,
                copied_checkpoint,
                copy_checkpoint=True,
                redactor=self._secret_redactor,
            )
        finally:
            if copied_checkpoint is not None:
                copied_checkpoint.clear()

    async def _settle_fork_source_expiring_authority(self, session_id: str) -> None:
        expired_recovery_claim_id: str | None = None

        def settle(
            _session: Session,
            checkpoint: dict[str, Any] | None,
            store_now: datetime,
        ) -> dict[str, Any] | None:
            nonlocal expired_recovery_claim_id
            recovery_claim = _incomplete_recovery_claim_from_checkpoint(checkpoint)
            if recovery_claim is not None:
                if recovery_claim[1] > store_now:
                    raise RuntimeError(
                        "Session has an active incomplete-session recovery operation."
                    )
                expired_recovery_claim_id = recovery_claim[0]
                return None
            if (
                checkpoint is None
                or _session_operation_state._SESSION_OPERATIONS_CHECKPOINT_KEY not in checkpoint
            ):
                return None
            updated = copy_durable_record(checkpoint, "checkpoint")
            operations = _session_operation_state._session_operation_state(updated)
            active_before = operations.get("active_operation_id")
            _session_operation_state._abandon_expired_session_operation(operations, now=store_now)
            active_after = operations.get("active_operation_id")
            if active_after is not None:
                raise RuntimeError(f"Session has an active durable operation: {active_after}")
            if active_before is None:
                return None
            updated[_session_operation_state._SESSION_OPERATIONS_CHECKPOINT_KEY] = operations
            return updated

        await self.session_store.transform_checkpoint_with_store_time(session_id, settle)
        if expired_recovery_claim_id is not None:
            await self._fence_expired_incomplete_recovery_claim_for_fork(
                session_id,
                _session_checkpoint_admission._ExpiredIncompleteRecoveryClaim(
                    expired_recovery_claim_id
                ),
            )

    async def _fence_expired_incomplete_recovery_claim_for_fork(
        self,
        session_id: str,
        expired_claim: _session_checkpoint_admission._ExpiredIncompleteRecoveryClaim,
    ) -> NoReturn:
        current = await self.session_store.load(session_id)
        if current is None:
            raise KeyError(f"Session not found: {session_id}") from None
        fenced = await self._recovery_ownership.fence_expired_incomplete_recovery_claim(
            session=current,
            claim_id=expired_claim.claim_id,
        )
        if not fenced:
            raise RuntimeError(
                "Expired incomplete-session recovery ownership changed while the "
                "fork was fencing it; retry with current session state."
            ) from None
        raise ValueError(
            "Session fork fenced an expired incomplete-session recovery owner; "
            "retry with current session state."
        ) from None

    async def fork_session(
        self,
        request: ForkSessionRequest,
        *,
        request_sha256: str,
        accepted_request_sha256s: tuple[str, ...],
        store_resolved_source_session_id: str | None = None,
        fork_controller: VerifiedForkController | None = None,
    ) -> AsyncGenerator[Event, None]:
        request = copy_fork_session_request(request)
        overlay = request.overlay
        verified_controller: VerifiedForkController | None = None
        if overlay is not None:
            verified_controller = require_verified_fork_controller(fork_controller)
            if overlay.controller != verified_controller.controller.name:
                raise ForkControllerDenied("The overlay names a different controller.")
            if request.initial_invocation is not None:
                require_fork_controller_model_runs(verified_controller.controller)
            require_requested_fork_controller_host_tools(
                verified_controller.controller,
                request.tool_capability_ceiling,
                None
                if request.initial_invocation is None
                else request.initial_invocation.tool_capability_ceiling,
            )
        elif fork_controller is not None:
            raise ValueError("Controller forks require an overlay.")
        if (
            request.initial_invocation is not None
            and request.initial_invocation.failover is not None
        ):
            self.session_store._require_model_failover_stage_protocol()
        if (
            len(request_sha256) != 64
            or any(character not in "0123456789abcdef" for character in request_sha256)
            or not accepted_request_sha256s
            or accepted_request_sha256s[0] != request_sha256
            or any(
                len(candidate) != 64
                or any(character not in "0123456789abcdef" for character in candidate)
                for candidate in accepted_request_sha256s
            )
        ):
            raise ValueError("Fork request replay identity is invalid.")
        fork_request_sha256 = request_sha256
        runtime_generated_session_id: str | None = None
        destination_session_id = request.session_id
        if destination_session_id is None:
            logical_fork_id = hashlib.sha256(
                canonical_durable_json_bytes(
                    request.model_dump(mode="json", warnings=False),
                    "generated_session_fork_identity",
                )
            ).hexdigest()
            runtime_generated_session_id = generate_child_session_id(
                kind=ChildSessionKind.SESSION_FORK,
                parent_session_id=request.source_session_id,
                logical_spawn_id=logical_fork_id,
            )
            destination_session_id = runtime_generated_session_id
        if destination_session_id is not None:
            existing_descendant = await self.session_store.load(destination_session_id)
            if existing_descendant is not None and verified_controller is not None:
                # Replay returns durable evidence, so it needs current enrollment
                # for the child's own agent; the source may be gone by now.
                try:
                    child_registered_agent = self._get_registered_agent(
                        existing_descendant.agent_name
                    )
                except KeyError:
                    raise ForkControllerDenied(
                        "The fork is not an agent this controller may fork."
                    ) from None
                if (
                    child_registered_agent.policy_agent_name
                    not in verified_controller.controller.agents
                ):
                    raise ForkControllerDenied("The fork is not an agent this controller may fork.")
                require_fork_controller_host_tools(
                    verified_controller.controller,
                    tool_capability_ceiling_from_session_metadata(existing_descendant.metadata),
                    child_registered_agent,
                )
            if existing_descendant is not None:
                relationship: SessionForkProfileRelationship | None = None
                malformed_relationship = False
                try:
                    relationship = session_fork_profile_relationship(existing_descendant)
                except Exception as exc:
                    if exc.__traceback__ is not None:
                        traceback_module.clear_frames(exc.__traceback__)
                    malformed_relationship = True
                if (
                    malformed_relationship
                    or relationship is None
                    or relationship.source_session_id != request.source_session_id
                    or relationship.request_sha256 not in accepted_request_sha256s
                    or relationship.source_state_sha256
                    != (
                        None
                        if request.expected_source is None
                        else fork_source_state_sha256(request.expected_source)
                    )
                ):
                    existing_descendant = None
                    relationship = None
                    raise ForkRequestConflict(
                        "Existing fork destination conflicts with the exact profiled request."
                    ) from None
                evidence_ids = (
                    [relationship.decision.event_id] if relationship.decision is not None else []
                ) + [
                    relationship.fork_event_id,
                    f"{relationship.fork_event_id}:targeted-tool-grants-reset",
                ]
                persisted_events: list[Event] = []
                for evidence_id in evidence_ids:
                    records = await self.session_store.query_events(
                        EventQuery(
                            session_id=existing_descendant.id,
                            event_id=evidence_id,
                            limit=1,
                        )
                    )
                    if len(records) != 1:
                        raise RuntimeError(
                            "Existing profiled fork is missing its durable evidence."
                        )
                    persisted_events.append(records[0].event)
                validate_profiled_fork_evidence(
                    fork=existing_descendant,
                    relationship=relationship,
                    events=persisted_events,
                )
                succession = (
                    _PromptSuccessionWorkflow.recover(
                        request=request,
                        accepted_request_sha256s=accepted_request_sha256s,
                        descendant=existing_descendant,
                    )
                    if request.system_prompt_policy is ForkSystemPromptPolicy.CURRENT_AGENT
                    else None
                )
                if succession is not None:
                    surviving_source = await self.session_store.load(request.source_session_id)
                    source_checkpoint = (
                        None
                        if surviving_source is None
                        else await self.session_store.load_checkpoint(request.source_session_id)
                    )
                    intent_requires_completion = False
                    try:
                        if source_checkpoint is not None:
                            intent_ledger = (
                                _prompt_transition._PromptTransitionIntentLedger.from_checkpoint(
                                    source_checkpoint,
                                    required=False,
                                )
                            )
                            intent_requires_completion = intent_ledger.complete_from_receipt(
                                succession.receipt,
                                descendant_created_at=existing_descendant.created_at,
                                completed_at=self._clock(),
                            )
                    finally:
                        if type(source_checkpoint) is dict:
                            source_checkpoint.clear()
                        source_checkpoint = None
                        surviving_source = None

                    if intent_requires_completion:
                        # A crash may leave the source intent prepared after the
                        # child commit. Repair that source-owned state when the
                        # original parent survives; an already completed intent
                        # and a deleted parent require no source admission.
                        await require_ordinary_session_execution(
                            request.source_session_id,
                            admit_session=True,
                            task_store=self._task_store,
                            redactor=self._secret_redactor,
                        )

                        def reconcile_surviving_intent(
                            _current_source: Session,
                            current_checkpoint: dict[str, Any] | None,
                        ) -> dict[str, Any] | None:
                            if current_checkpoint is None:
                                return None
                            updated = copy_durable_record(current_checkpoint, "checkpoint")
                            current_ledger = (
                                _prompt_transition._PromptTransitionIntentLedger.from_checkpoint(
                                    updated,
                                    required=False,
                                )
                            )
                            if current_ledger.complete_from_receipt(
                                succession.receipt,
                                descendant_created_at=existing_descendant.created_at,
                                completed_at=self._clock(),
                            ):
                                current_ledger.store_in(updated)
                            return updated

                        try:
                            await self.session_store.publish_checkpoint_and_events(
                                request.source_session_id,
                                checkpoint_transform=reconcile_surviving_intent,
                                events=[],
                            )
                        except KeyError:
                            # Parent deletion is allowed to race exact child
                            # acknowledgement. Suppress only the proven deletion;
                            # other source-publication failures remain visible.
                            if await self.session_store.load(request.source_session_id) is not None:
                                raise
                await advance_task_retention_epoch(
                    existing_descendant,
                    session_store=self.session_store,
                    task_store=self._task_store,
                )
                delivered = await self._event_writer.fan_out_persisted(persisted_events)
                delivered_fork_event = next(
                    (event for event in delivered if event.type is EventType.SESSION_FORKED),
                    None,
                )
                if delivered_fork_event is None:
                    raise RuntimeError("Existing profiled fork lost its public fork event.")
                yield delivered_fork_event
                return
        source_session = await self.session_store.load(request.source_session_id)
        if source_session is None:
            raise session_request_boundary.ForkSourceNotFoundError(
                "Fork source session was not found."
            ) from None
        try:
            source_session = session_request_boundary.prepare_fork_source_session(
                source_session,
                expected_source_session_id=request.source_session_id,
                store_resolved_source_session_id=store_resolved_source_session_id,
                redactor=self._secret_redactor,
            )
        except ValueError:
            del source_session
            raise
        fork_deadline = effective_deadline(
            current_execution_deadline(), source_session.execution_deadline
        )
        fork_deadline.require_admission("fork")
        # Only first-time creation executes against source authority. Keep the
        # public preflight on this branch so reconciliation of an already
        # committed child does not depend on a live parent or its task store.
        await require_ordinary_session_execution(
            source_session.id,
            admit_session=False,
            task_store=self._task_store,
            redactor=self._secret_redactor,
        )
        expected_source_snapshot = request.expected_source
        if expected_source_snapshot is not None and (
            source_session.id != expected_source_snapshot.source_session_id
            or _fork_source_session_instance_fingerprint(source_session)
            != expected_source_snapshot.source_instance_fingerprint
            or source_session.status != expected_source_snapshot.status
            or source_session.run_epoch != expected_source_snapshot.run_epoch
            or source_session.causal_budget_id != expected_source_snapshot.causal_budget_id
        ):
            raise SessionRunFenced(
                "Fork source identity or lifecycle changed after its exact snapshot was captured."
            )
        if source_session.status not in _FORKABLE_SESSION_STATUSES:
            raise ValueError(
                "Only completed, failed, or interrupted sessions can be forked: "
                f"{source_session.status}"
            )
        if request.transcript_cursor is not None and request.copy_checkpoint:
            raise ValueError(
                "ForkSessionRequest.copy_checkpoint must be false when transcript_cursor is set."
            )
        if source_session.status == SessionStatus.INTERRUPTED and not request.copy_checkpoint:
            raise ValueError("Interrupted sessions cannot be forked without checkpoint state.")
        if not self.session_store.supports_profiled_forks:
            raise RuntimeError("Session store does not support atomic profiled session forks.")

        environment_name = (
            request.environment_name
            if request.environment_name is not None
            else source_session.environment_name
        )
        source_checkpoint_for_profile = await self.session_store.load_checkpoint(source_session.id)
        try:
            (
                source_profile_source,
                source_execution_profile,
                source_active_profile,
            ) = session_request_boundary.prepare_fork_source_execution_profile(
                source_session,
                source_checkpoint_for_profile,
            )
            source_model_selection = model_failover_progress_for_session(
                session=source_session,
                execution_profile=source_execution_profile,
                checkpoint=source_checkpoint_for_profile,
            )
            source_environment_allocation_owners_by_environment = (
                environment_allocation_owners_from_checkpoint(
                    source_checkpoint_for_profile,
                )
            )
            require_authorized_environment_allocation_owners(
                source_session,
                source_environment_allocation_owners_by_environment,
            )
        finally:
            source_checkpoint_for_profile = None
        source_environment_allocation_owners = (
            tuple(
                SessionForkEnvironmentAllocationOwner(
                    environment_name=environment_name,
                    owner_session_id=owner_session_id,
                )
                for environment_name, owner_session_id in (
                    source_environment_allocation_owners_by_environment.items()
                )
            )
            if request.copy_checkpoint
            else ()
        )
        source_environment_allocation_owner_payload = [
            owner.model_dump(mode="json") for owner in source_environment_allocation_owners
        ]
        if expected_source_snapshot is not None and (
            source_execution_profile.fingerprint
            != expected_source_snapshot.execution_profile_fingerprint
        ):
            raise SessionRunFenced(
                "Fork source execution profile changed after its exact snapshot was captured."
            )
        try:
            source_registered_agent = self._get_registered_agent(source_session.agent_name)
        except KeyError as exc:
            raise KeyError(
                "Source agent must be registered to derive inherited taint before forking: "
                f"{source_session.agent_name}"
            ) from exc
        # An overlay fork belongs to its controller whether or not it carries that
        # controller's body; its durable overlay record names the owner too, and
        # every session created under it inherits that owner in its lineage marker.
        # A source without a marker, such as one written before markers existed,
        # takes its owner from the parents that settle its lineage.
        source_lineage = await resolve_fork_lineage_async(source_session, self.session_store.load)
        if overlay is not None and not source_lineage.settled:
            raise ForkControllerDenied("The source is not an agent this controller may fork.")
        source_owners = {
            owner
            for owner in (
                source_registered_agent.fork_controller_name,
                source_lineage.owner() if source_lineage.settled else None,
            )
            if owner is not None
        }
        if source_owners and (overlay is None or source_owners != {overlay.controller}):
            raise ForkControllerDenied(
                "Only the enrolling controller can fork one of its overlay forks."
            )
        source_tool_capability_ceiling = tool_capability_ceiling_from_session_metadata(
            source_session.metadata,
        )
        if source_registered_agent.tool_discovery_mode is not None:
            if not self.session_store.supports_atomic_session_operation_initialization:
                raise RuntimeError(
                    "Tool discovery requires atomic session operation initialization."
                )
            await _require_tool_discovery_view(
                self.session_store,
                source_session,
                source_registered_agent,
                source_tool_capability_ceiling,
                require_empty=False,
            )
        agent_name = request.agent_name or source_session.agent_name
        if request.agent_name is not None and (
            self._get_registered_agent(request.agent_name).fork_controller_name is not None
        ):
            raise ValueError("A fork-controller body is selected only by an overlay.")
        if verified_controller is not None and overlay is not None:
            # Authorize the source itself: a controller may read only sessions of
            # agents it is enrolled for, and its forks keep the source's agent.
            source_host_agent = source_registered_agent.policy_agent_name
            if source_host_agent not in verified_controller.controller.agents:
                raise ForkControllerDenied("The source is not an agent this controller may fork.")
            if request.agent_name is not None and request.agent_name != source_host_agent:
                raise ForkControllerDenied("A controller fork keeps the source session's agent.")
            agent_name = fork_overlay_agent_name(
                verified_controller.controller,
                overlay,
                host_agent_name=source_host_agent,
            )
            if (
                agent_name != source_session.agent_name
                and request.execution_profile_selection
                is ForkExecutionProfileSelection.INHERIT_PARENT
            ):
                raise ValueError(
                    "This overlay changes the child's body and requires "
                    "execution_profile_selection='current_child' with profile_adoption."
                )
        registered_agent = self._get_registered_agent(agent_name)
        if (
            registered_agent.tool_discovery_mode is not None
            and not self.session_store.supports_atomic_session_operation_initialization
        ):
            raise RuntimeError("Tool discovery requires atomic session operation initialization.")
        effective_tool_capability_ceiling = resolve_tool_capability_ceiling(
            request.tool_capability_ceiling,
            registered_agent.tool_capabilities,
            maximum=source_tool_capability_ceiling,
        )
        # Controller tools are never inherited or requested through the host
        # ceiling; only this fork's overlay grants admit them.
        effective_tool_capability_ceiling = fork_overlay_tool_capability_ceiling(
            effective_tool_capability_ceiling,
            registered_agent,
            overlay,
        )
        if verified_controller is not None:
            # Hosted tools ride on the agent, not the ceiling, so they are refused.
            require_fork_controller_hosted_tools(verified_controller.controller, registered_agent)
            # An inherited ceiling keeps only the host tools this controller's
            # enrollment allows; an explicit request outside it was denied above.
            effective_tool_capability_ceiling = fork_controller_host_tool_ceiling(
                verified_controller.controller,
                effective_tool_capability_ceiling,
            )
        model = request.model or (
            registered_agent.spec.model if request.agent_name is not None else source_session.model
        )
        try:
            (
                agent_name,
                registered_agent_provider_name,
                model,
                environment_name,
            ) = session_request_boundary.prepare_fork_registered_authority(
                agent_name=agent_name,
                agent_provider_name=(
                    registered_agent.spec.provider_name if request.agent_name is not None else None
                ),
                model=model,
                environment_name=environment_name,
                redactor=self._secret_redactor,
            )
        except ValueError:
            del registered_agent
            agent_name = model = environment_name = ""
            raise
        registered_provider = self._get_registered_provider(
            registered_agent_provider_name or source_session.provider_name
        )
        registered_environment = self._get_registered_environment_for_session(environment_name)
        prompt_workflow = await _ForkPromptWorkflow.prepare(
            request=request,
            request_sha256=fork_request_sha256,
            registered_agent=registered_agent,
            registered_environment=registered_environment,
            environment_lifecycle=self._environment_lifecycle,
        )

        initial_invocation = request.initial_invocation
        if initial_invocation is not None and (
            initial_invocation.session_id != destination_session_id
        ):
            raise RuntimeError("Fork initial invocation authority is inconsistent.")
        initial_tool_capability_ceiling = effective_tool_capability_ceiling
        if initial_invocation is not None:
            initial_tool_capability_ceiling = resolve_tool_capability_ceiling(
                initial_invocation.tool_capability_ceiling,
                registered_agent.tool_capabilities,
                maximum=effective_tool_capability_ceiling,
            )
        current_child_runtime_identity = SessionRuntimeIdentity(
            runtime_name="cayu",
            runtime_version=execution_profile_continuation._runtime_version(),
            runtime_build_provenance=current_runtime_build_provenance(),
        )

        def ordinary_child_execution_profile(
            child_provider: runtime_records.RegisteredProvider,
            child_model: str,
            *,
            profile_agent: runtime_records.RegisteredAgentState | None = None,
            profile_ceiling: ToolCapabilityCeiling | None = None,
        ) -> ExecutionProfileIdentity:
            candidate = _session_execution_profile._execution_profile_identity(
                registered_agent=registered_agent if profile_agent is None else profile_agent,
                provider_name=child_provider.name,
                registered_provider=child_provider,
                model=child_model,
                durable_system_prompt=prompt_workflow.rendered_child_prompt,
                redactor=self._secret_redactor,
                registered_environment=registered_environment,
                process_identity=self._execution_profile_process_identity,
                runtime_hooks=self._runtime_hooks,
                loop_policies=self._loop_policies,
                loop_policy_execution_profile_identities=(
                    self._loop_policy_execution_profile_identities
                ),
                request_loop_policies=(
                    () if initial_invocation is None else initial_invocation.loop_policies
                ),
                request_loop_policy_instance_identities=(
                    ()
                    if initial_invocation is None
                    else self._execution_profile_continuation.request_loop_policy_instance_identities(
                        initial_invocation.loop_policies
                    )
                ),
                budget_policy=self._get_budget_policy(),
                request_budget_limits=(
                    () if initial_invocation is None else initial_invocation.budget_limits
                ),
                causal_budget_id=source_session.causal_budget_id,
                structured_output=(
                    None if initial_invocation is None else initial_invocation.structured_output
                ),
                thinking=(None if initial_invocation is None else initial_invocation.thinking),
                max_steps=(
                    self._application_run_defaults.max_steps
                    if initial_invocation is None
                    else initial_invocation.max_steps
                ),
                limits=(
                    self._application_run_defaults.copy_limits()
                    if initial_invocation is None
                    else initial_invocation.limits
                ),
                retry_policy=(
                    self._effective_retry_policy(None)
                    if initial_invocation is None
                    else self._effective_retry_policy(initial_invocation.retry_policy)
                ),
                tool_capability_ceiling=(
                    initial_tool_capability_ceiling if profile_ceiling is None else profile_ceiling
                ),
                runtime_identity=current_child_runtime_identity,
                tool_completion=None
                if initial_invocation is None
                else initial_invocation.tool_completion,
            )
            if (
                prompt_workflow.rendered_child_prompt is None
                and request.system_prompt_policy is ForkSystemPromptPolicy.INHERIT_SOURCE
            ):
                candidate = execution_profile_with_component(
                    candidate,
                    source_execution_profile.component(
                        ExecutionProfileComponentClass.DURABLE_SYSTEM_PROJECTION
                    ),
                )
            return candidate

        def current_child_execution_profile() -> ExecutionProfileIdentity:
            policy = None if initial_invocation is None else initial_invocation.failover
            inherited = source_execution_profile.model_failover
            if (
                policy is None
                and inherited is not None
                and request.execution_profile_selection
                is ForkExecutionProfileSelection.INHERIT_PARENT
            ):
                policy = ModelFailoverPolicy(
                    fallbacks=tuple(
                        ModelTarget(provider_name=item.provider_name, model=item.model)
                        for item in inherited.plan.candidates[1:]
                    ),
                    max_total_attempts=inherited.plan.max_total_attempts,
                )
            if policy is None:
                return ordinary_child_execution_profile(registered_provider, model)
            return execution_profile_admission.resolve_model_failover_execution_profile(
                policy=policy,
                primary=ModelTarget(provider_name=registered_provider.name, model=model),
                registered_primary=registered_provider,
                resolve_provider=self._get_registered_provider,
                resolve_candidate_profile=ordinary_child_execution_profile,
                redactor=self._secret_redactor,
            ).profile

        def overlay_changes_are_attributable() -> bool:
            """Whether only the overlay's grants explain the profile change.

            Rebuild the source's profile from today's registrations of its own
            agent and ceiling. Any difference outside per-invocation settings
            means a host or controller registration drifted since the source
            was created, which the built-in overlay adoption must not admit.
            """

            try:
                source_provider = self._get_registered_provider(source_session.provider_name)
            except KeyError:
                return False
            rebuilt_source = ordinary_child_execution_profile(
                source_provider,
                source_session.model,
                profile_agent=source_registered_agent,
                profile_ceiling=source_tool_capability_ceiling,
            )
            drift = set(
                changed_execution_profile_components(source_execution_profile, rebuilt_source)
            )
            return not (drift - _FORK_OVERLAY_INVOCATION_COMPONENTS)

        initial_invocation_profile: ExecutionProfileIdentity | None = None
        selected_execution_profile = source_execution_profile
        selected_runtime_identity = SessionRuntimeIdentity(
            runtime_name=source_session.runtime_name,
            runtime_version=source_session.runtime_version,
            runtime_build_provenance=source_session.runtime_build_provenance,
        )
        execution_profile_decision: ExecutionProfileDecision | None = None
        if request.execution_profile_selection is ForkExecutionProfileSelection.CURRENT_CHILD:
            candidate_execution_profile = current_child_execution_profile()
            unavailable_child_components = unavailable_execution_profile_components(
                candidate_execution_profile
            )
            if unavailable_child_components:
                raise RuntimeError(
                    "Current-child profile selection has unavailable required components: "
                    + ", ".join(component.value for component in unavailable_child_components)
                )
            decision_session = source_session.model_copy(
                update={
                    "id": destination_session_id,
                    "agent_name": agent_name,
                    "provider_name": registered_provider.name,
                    "model": model,
                    "parent_session_id": source_session.id,
                    "environment_name": environment_name,
                    "run_epoch": 0,
                },
                deep=True,
            )
            changed_profile_components = changed_execution_profile_components(
                source_execution_profile,
                candidate_execution_profile,
            )
            execution_profile_decision = await self._profile_classifier.classify(
                session=source_session,
                decision_session=decision_session,
                expected_profile=source_execution_profile,
                candidate_profile=candidate_execution_profile,
                changed_component_classes=changed_profile_components,
                intent=request.profile_adoption,
                adoption_request_fingerprint=fork_request_sha256,
                target_changed=(
                    registered_provider.name != source_session.provider_name
                    or model != source_session.model
                ),
                target_provider_name=registered_provider.name,
                target_model=model,
                source_provider_name=source_session.provider_name,
                source_model=source_session.model,
                force_authority_review=True,
                fork_controller_overlay=(
                    registered_agent.fork_controller_name is not None
                    and overlay_changes_are_attributable()
                ),
            )
            if execution_profile_decision.kind in {
                ExecutionProfileDecisionKind.MIGRATION_REQUIRED,
                ExecutionProfileDecisionKind.REJECTED,
            }:
                error_type = (
                    ExecutionProfileMigrationRequired
                    if execution_profile_decision.kind
                    is ExecutionProfileDecisionKind.MIGRATION_REQUIRED
                    else ExecutionProfileAdoptionRejected
                )
                raise error_type(
                    session_id=destination_session_id,
                    expected_profile_fingerprint=source_execution_profile.fingerprint,
                    candidate_profile_fingerprint=candidate_execution_profile.fingerprint,
                    changed_component_classes=changed_profile_components,
                    expected_profile=source_execution_profile,
                    candidate_profile=candidate_execution_profile,
                )
            selected_execution_profile = candidate_execution_profile
            selected_runtime_identity = current_child_runtime_identity
            initial_invocation_profile = (
                None if initial_invocation is None else candidate_execution_profile
            )
        else:
            selected_execution_profile = execution_profile_with_tool_capability_ceiling(
                source_execution_profile,
                initial_tool_capability_ceiling.tool_names,
            )
        if (
            request.execution_profile_selection is ForkExecutionProfileSelection.INHERIT_PARENT
            and initial_invocation is not None
        ):
            candidate_execution_profile = current_child_execution_profile()
            changed_profile_components = inherited_execution_profile_component_changes(
                source_execution_profile,
                candidate_execution_profile,
            )
            allowed_initial_changes = {
                ExecutionProfileComponentClass.INVOCATION_BUDGET_POLICY,
                ExecutionProfileComponentClass.STRUCTURED_OUTPUT,
                ExecutionProfileComponentClass.FINALIZATION,
                ExecutionProfileComponentClass.TOOL_VIEW_GRANTS,
            }
            unsupported_changes = tuple(
                component
                for component in changed_profile_components
                if component not in allowed_initial_changes
            )
            if unsupported_changes:
                raise RuntimeError(
                    "An inherited fork invocation changed execution authority outside its "
                    "declared initial budget, structured-output, finalization, or tool-ceiling "
                    "semantics: "
                    + ", ".join(component.value for component in unsupported_changes)
                    + ". Use an explicitly authorized current-child branch."
                )
            unavailable_child_components = unavailable_execution_profile_components(
                candidate_execution_profile
            )
            if unavailable_child_components:
                raise RuntimeError(
                    "Fork initial invocation has unavailable required components: "
                    + ", ".join(component.value for component in unavailable_child_components)
                )
            initial_invocation_profile = candidate_execution_profile

        if (
            prompt_workflow.prompt_replacement is not None
            and registered_provider.name != source_session.provider_name
        ):
            raise ValueError(
                "Prompt-anatomy succession to an agent with a different provider is not supported."
            )

        if (
            selected_execution_profile.model_failover is not None
            and not self.session_store._supports_model_failover_stage_protocol()
        ):
            raise NotImplementedError("Store does not attest atomic model failover forks.")

        source_projection_cursor = session_model_projection_cursor(source_session)
        if source_model_selection is not None:
            source_projection_cursor = max(
                source_projection_cursor, source_model_selection.projection_cursor
            )
        model_changed = model != source_session.model
        provider_changed = registered_provider.name != source_session.provider_name
        target_changed = model_changed or provider_changed
        expected_fork_transcript: tuple[Message, ...] | None = None
        fork_projection_cursor = 0
        selected_source_cursor = 0
        source_snapshot_cursor = 0
        source_prompt_sha256: str | None = None
        child_prompt_sha256: str | None = None
        if prompt_workflow.requires_transcript_snapshot(
            target_changed=target_changed,
            source_projection_cursor=source_projection_cursor,
        ):
            source_snapshot = await self.session_store.load_transcript_snapshot(source_session.id)
            source_snapshot_cursor = source_snapshot.cursor
            if source_projection_cursor > source_snapshot.cursor:
                del source_snapshot
                raise ValueError(
                    "Source session model-target projection cursor exceeds its transcript."
                )
            if request.transcript_cursor is not None:
                if request.transcript_cursor > source_snapshot.cursor:
                    del source_snapshot
                    raise ValueError("transcript_cursor is greater than source transcript length.")
                selected_source_records = [
                    record
                    for record in source_snapshot.records
                    if record.index < request.transcript_cursor
                ]
            else:
                selected_source_records = list(source_snapshot.records)
            selected_source_cursor = (
                source_snapshot.cursor
                if request.transcript_cursor is None
                else request.transcript_cursor
            )
            selected_messages = [
                detach_message(record.message) for record in selected_source_records
            ]
            source_messages = [detach_message(record.message) for record in source_snapshot.records]
            try:
                prompt_workflow.require_inherited_system_projection(
                    source_messages=source_messages,
                    selected_messages=selected_messages,
                )
            except BaseException:
                selected_source_records.clear()
                selected_messages.clear()
                del source_snapshot
                raise
            finally:
                source_messages.clear()
            source_prompt_sha256 = system_prompt_messages_sha256(selected_messages)
            selected_messages, child_prompt_sha256 = prompt_workflow.project_selected_messages(
                selected_messages,
                [record.interaction_id for record in selected_source_records],
            )
            expected_fork_transcript = tuple(selected_messages)
            if (
                self._secret_redactor.has_values
                and not session_request_boundary.fork_transcript_is_secret_free(
                    expected_fork_transcript,
                    redactor=self._secret_redactor,
                )
            ):
                selected_source_records.clear()
                selected_messages.clear()
                expected_fork_transcript = None
                del source_snapshot
                raise ValueError(FORK_TRANSCRIPT_VALIDATION_ERROR) from None
            if prompt_workflow.requires_portability_validation(target_changed=target_changed):
                portable_projection = None
                portable_messages: list[Message] = []
                try:
                    if target_changed:
                        portable_projection = model_target.project_portable_transcript(
                            list(expected_fork_transcript)
                        )
                        portable_messages = [
                            detach_message(message) for message in portable_projection.messages
                        ]
                    else:
                        portable_messages = [
                            detach_message(message) for message in expected_fork_transcript
                        ]
                    if portable_messages:
                        validate_context_messages(portable_messages)
                    if target_changed:
                        hook_tools: list[dict[str, Any]] = []
                        try:
                            hook_tools = _model_request_tools(
                                tool_exposure=_tool_capability_ceiling_exposure(
                                    registered_agent,
                                    effective_tool_capability_ceiling.tool_names,
                                ),
                                structured_output=None,
                                targeted_tool_projection=resolve_targeted_tool_projection(
                                    registered_agent.targeted_tool_mode,
                                    provider=registered_provider.provider,
                                    model=model,
                                ),
                                tool_discovery_mode=registered_agent.tool_discovery_mode,
                            )
                            registered_provider.provider.preflight_portable_messages(
                                model=model,
                                messages=portable_messages,
                                tools=hook_tools,
                            )
                        finally:
                            hook_tools.clear()
                    fork_projection_cursor = len(expected_fork_transcript)
                except BaseException:
                    expected_fork_transcript = None
                    portable_projection = None
                    portable_messages.clear()
                    selected_source_records.clear()
                    del source_snapshot
                    raise
                finally:
                    portable_messages.clear()
            else:
                fork_projection_cursor = sum(
                    record.index < source_projection_cursor for record in selected_source_records
                )
            selected_source_records.clear()
            del source_snapshot

        def validate_source_checkpoint(
            current_source: Session,
            source_checkpoint: dict[str, Any] | None,
        ) -> dict[str, Any] | None:
            return _validate_fork_source_checkpoint_state(
                current_source,
                source_checkpoint,
                copy_checkpoint=request.copy_checkpoint,
                redactor=self._secret_redactor,
            )

        def checkpoint_transform(
            current_source: Session,
            source_checkpoint: dict[str, Any] | None,
        ) -> dict[str, Any] | None:
            """Validate live checkpoint state even when the fork will discard it."""
            return prepare_fork_checkpoint(
                current_source,
                source_checkpoint,
                require_prepared_intent=True,
            )

        def prepare_fork_checkpoint(
            current_source: Session,
            source_checkpoint: dict[str, Any] | None,
            *,
            require_prepared_intent: bool,
        ) -> dict[str, Any] | None:
            current_allocation_owners_by_environment = (
                environment_allocation_owners_from_checkpoint(
                    source_checkpoint,
                )
            )
            require_authorized_environment_allocation_owners(
                current_source,
                current_allocation_owners_by_environment,
            )
            if (
                current_allocation_owners_by_environment
                != source_environment_allocation_owners_by_environment
            ):
                raise SessionRunFenced(
                    "Fork source environment-allocation ownership changed while the "
                    "fork was being prepared."
                )
            _, live_source_profile, _ = (
                session_request_boundary.prepare_fork_source_execution_profile(
                    current_source,
                    source_checkpoint,
                )
            )
            if expected_source_snapshot is not None:
                if (
                    _fork_source_session_instance_fingerprint(current_source)
                    != expected_source_snapshot.source_instance_fingerprint
                ):
                    raise SessionRunFenced(
                        "Fork source session incarnation changed after its exact snapshot "
                        "was captured."
                    )
                if (
                    live_source_profile.fingerprint
                    != expected_source_snapshot.execution_profile_fingerprint
                ):
                    raise SessionRunFenced(
                        "Fork source execution profile changed after its exact snapshot "
                        "was captured."
                    )
                if (
                    fork_source_checkpoint_sha256(source_checkpoint)
                    != expected_source_snapshot.checkpoint_sha256
                ):
                    raise SessionRunFenced(
                        "Fork source checkpoint changed after its exact snapshot was captured."
                    )
            source_checkpoint = validate_source_checkpoint(current_source, source_checkpoint)
            if require_prepared_intent:
                prompt_workflow.require_prepared_intent(source_checkpoint)
            fork_checkpoint = approval_support.checkpoint_for_fork(
                checkpoint=source_checkpoint,
            )
            if not request.copy_checkpoint:
                return None
            if fork_checkpoint is not None:
                # Snapshot assertions and the committed child must use one exact
                # definition of inheritable checkpoint state. Source-owned run,
                # publication, receipt, and recall records remain with the parent.
                strip_source_owned_fork_checkpoint_state(fork_checkpoint)
            if not session_request_boundary.fork_checkpoint_is_secret_free(
                fork_checkpoint,
                redactor=self._secret_redactor,
            ):
                if source_checkpoint is not None:
                    source_checkpoint.clear()
                if fork_checkpoint is not None:
                    fork_checkpoint.clear()
                source_checkpoint = fork_checkpoint = None
                raise ValueError(
                    "source_session.checkpoint contains a workload secret and cannot "
                    "be copied without changing resumable execution state."
                ) from None
            return fork_checkpoint

        def transcript_validator(
            messages: tuple[Message, ...],
            source_snapshot: TranscriptSnapshot,
        ) -> bool:
            return (
                (expected_fork_transcript is None or messages == expected_fork_transcript)
                and (
                    expected_source_snapshot is None
                    or (
                        source_snapshot.cursor == expected_source_snapshot.transcript_cursor
                        and fork_source_transcript_sha256(source_snapshot)
                        == expected_source_snapshot.transcript_sha256
                    )
                )
                and session_request_boundary.fork_transcript_is_secret_free(
                    messages,
                    redactor=self._secret_redactor,
                )
            )

        inherited_taint_labels = await self._tool_admission.prior_taint_labels(
            session_id=source_session.id,
            policy=source_registered_agent.tool_policy,
            request_metadata=source_session.metadata,
        )
        fork_metadata = copy_durable_metadata(request.metadata)
        if inherited_taint_labels:
            fork_metadata = metadata_with_taint_labels(
                fork_metadata,
                inherited_taint_labels | taint_labels_from_metadata(request.metadata),
            )
        fork_metadata = _session_metadata_with_model_projection(
            fork_metadata,
            target=ModelTarget(
                provider_name=registered_provider.name,
                model=model,
            ),
            transcript_cursor=fork_projection_cursor,
        )
        fork_metadata = session_metadata_with_tool_capability_ceiling(
            fork_metadata,
            effective_tool_capability_ceiling,
        )
        fork_owner: str | None = None
        if overlay is not None and verified_controller is not None:
            fork_owner = overlay.controller
            fork_metadata[FORK_OVERLAY_METADATA_KEY] = ForkOverlayRecord(
                overlay=overlay,
                fingerprint=overlay.fingerprint,
                controller_subject=verified_controller.subject,
            ).model_dump(mode="json")
        fork_lineage = fork_lineage_for_fork(source_lineage, controller=fork_owner)
        if fork_lineage is not None:
            fork_metadata[FORK_LINEAGE_METADATA_KEY] = fork_lineage
        prompt_workflow.attach_receipt(
            fork_metadata,
            source_session=source_session,
            destination_session_id=destination_session_id,
            selected_source_cursor=selected_source_cursor,
            source_prompt_sha256=source_prompt_sha256,
            child_prompt_sha256=child_prompt_sha256,
            child_agent_name=agent_name,
            provider_name=registered_provider.name,
            model=model,
            model_changed=model_changed,
            inherited_taint_labels=inherited_taint_labels,
        )
        fork_metadata[RUNTIME_BUILD_PROVENANCE_METADATA_KEY] = (
            selected_runtime_identity.runtime_build_provenance.model_dump(mode="json")
        )
        if fork_deadline.expires_at is not None:
            fork_metadata[EXECUTION_DEADLINE_METADATA_KEY] = fork_deadline.model_dump(mode="json")
        fork_session = Session(
            id=destination_session_id,
            agent_name=agent_name,
            provider_name=registered_provider.name,
            model=model,
            parent_session_id=source_session.id,
            causal_budget_id=source_session.causal_budget_id,
            runtime_name=selected_runtime_identity.runtime_name,
            runtime_version=selected_runtime_identity.runtime_version,
            environment_name=environment_name,
            status=source_session.status,
            invocation=fork_session_invocation(source_session),
            labels=source_session.labels,
            metadata=copy_session_metadata(fork_metadata),
        )
        try:
            fork_session = session_request_boundary.prepare_derived_fork_session(
                fork_session,
                source_session=source_session,
                runtime_generated_session_id=runtime_generated_session_id,
                store_resolved_source_session_id=store_resolved_source_session_id,
                redactor=self._secret_redactor,
            )
        except ValueError:
            del (
                fork_session,
                source_session,
                registered_agent,
                source_registered_agent,
                registered_environment,
                registered_provider,
            )
            if type(fork_metadata) is dict:
                fork_metadata.clear()
            inherited_taint_labels = frozenset()
            agent_name = model = environment_name = destination_session_id = ""
            runtime_generated_session_id = None
            raise
        if expected_source_snapshot is not None:
            authoritative_metadata = copy_session_metadata(fork_session.metadata)
            authoritative_metadata[FORK_SOURCE_SNAPSHOT_METADATA_KEY] = {
                "source_session_id": source_session.id,
                "source_instance_fingerprint": (
                    expected_source_snapshot.source_instance_fingerprint
                ),
                "status": source_session.status.value,
                "run_epoch": expected_source_snapshot.run_epoch,
                "transcript_cursor": expected_source_snapshot.transcript_cursor,
                "transcript_sha256": expected_source_snapshot.transcript_sha256,
                "checkpoint_sha256": expected_source_snapshot.checkpoint_sha256,
                "execution_profile_fingerprint": (
                    expected_source_snapshot.execution_profile_fingerprint
                ),
                "causal_budget_id": source_session.causal_budget_id,
            }
            fork_session = fork_session.model_copy(
                update={"metadata": authoritative_metadata},
                deep=True,
            )
        active_model_stage = await self.session_store.load_active_model_completion_stage(
            source_session.id
        )
        if active_model_stage is not None:
            del active_model_stage
            raise session_request_boundary.ForkActiveModelStageError(
                "Fork source session has an active model-completion stage."
            ) from None
        # Reject known-invalid checkpoint state before ordinary admission. The
        # store-owned transform below repeats the same validation after
        # admission so a concurrent checkpoint change still fails closed.
        await self._settle_fork_source_expiring_authority(source_session.id)
        source_checkpoint_preflight = await self.session_store.load_checkpoint(source_session.id)
        preflight_checkpoint: dict[str, Any] | None = None
        try:
            preflight_checkpoint = (
                None
                if source_checkpoint_preflight is None
                else copy_durable_record(source_checkpoint_preflight, "checkpoint")
            )
            prepare_fork_checkpoint(
                source_session,
                preflight_checkpoint,
                require_prepared_intent=False,
            )
        finally:
            if type(preflight_checkpoint) is dict:
                preflight_checkpoint.clear()
            preflight_checkpoint = None
            if type(source_checkpoint_preflight) is dict:
                source_checkpoint_preflight.clear()
            source_checkpoint_preflight = None

        # Prompt-succession preparation may publish a durable source intent.
        # Close the contract-attachment race immediately before that first
        # possible mutation; the admission also protects the later atomic fork.
        await require_ordinary_session_execution(
            source_session.id,
            admit_session=True,
            task_store=self._task_store,
            redactor=self._secret_redactor,
        )
        fork_session = await prompt_workflow.prepare_effect(
            session_store=self.session_store,
            source_session=source_session,
            source_snapshot_cursor=source_snapshot_cursor,
            fork_session=fork_session,
            validate_source_checkpoint=validate_source_checkpoint,
            clock=self._clock,
        )
        shared_live_workspace = (
            registered_environment is not None
            and registered_environment.environment.workspace is not None
            and source_session.environment_name is not None
            and source_session.environment_name == fork_session.environment_name
        )
        fork_model_candidate_index = (
            None
            if selected_execution_profile.model_failover is None
            else source_model_selection.candidate_index
            if request.execution_profile_selection is ForkExecutionProfileSelection.INHERIT_PARENT
            and source_model_selection is not None
            else 0
        )
        fork_event_payload: dict[str, Any] = {
            "source_session_id": source_session.id,
            "source_status": source_session.status.value,
            "parent_session_id": fork_session.parent_session_id,
            "causal_budget_id": fork_session.causal_budget_id,
            "transcript_cursor": request.transcript_cursor,
            "copy_checkpoint": request.copy_checkpoint,
            "system_prompt_policy": request.system_prompt_policy.value,
            "agent_name": fork_session.agent_name,
            "provider_name": fork_session.provider_name,
            "model": fork_session.model,
            "environment_name": fork_session.environment_name,
            "inherited_taint_labels": sorted(inherited_taint_labels),
            "execution_profile_selection": request.execution_profile_selection.value,
            "selected_profile_fingerprint": selected_execution_profile.fingerprint,
            "source_profile_fingerprint": source_execution_profile.fingerprint,
            "source_environment_allocation_owners": (source_environment_allocation_owner_payload),
            "fork_request_sha256": fork_request_sha256,
            **(
                {}
                if overlay is None
                else {
                    "fork_overlay_controller": overlay.controller,
                    "fork_overlay_fingerprint": overlay.fingerprint,
                    "fork_overlay_tool_names": list(overlay.tool_names),
                }
            ),
            "workspace_lineage": WorkspaceForkLineage(
                status=(
                    WorkspaceForkLineageStatus.SHARED_OR_AMBIGUOUS
                    if shared_live_workspace
                    else WorkspaceForkLineageStatus.UNPROVEN
                ),
                detail_code=(
                    "shared_live_workspace_not_isolated"
                    if shared_live_workspace
                    else "child_workspace_derivation_unproven"
                ),
            ).model_dump(mode="json"),
        }
        if fork_model_candidate_index is not None:
            fork_event_payload["model_failover_candidate_index"] = fork_model_candidate_index
        if expected_source_snapshot is not None:
            fork_event_payload.update(
                {
                    "source_instance_fingerprint": (
                        expected_source_snapshot.source_instance_fingerprint
                    ),
                    "source_run_epoch": expected_source_snapshot.run_epoch,
                    "source_transcript_cursor": (expected_source_snapshot.transcript_cursor),
                    "source_transcript_sha256": (expected_source_snapshot.transcript_sha256),
                    "source_checkpoint_sha256": (expected_source_snapshot.checkpoint_sha256),
                    "source_execution_profile_fingerprint": (
                        expected_source_snapshot.execution_profile_fingerprint
                    ),
                }
            )
        if initial_invocation is not None:
            if initial_invocation_profile is None:
                raise AssertionError("Fork initial invocation lost its frozen profile.")
            fork_event_payload.update(
                {
                    "initial_dispatch_id": request.initial_dispatch_id,
                    "initial_invocation_request_sha256": (
                        _fork_initial_invocation_request_sha256(initial_invocation)
                    ),
                    "initial_invocation_profile_fingerprint": (
                        initial_invocation_profile.fingerprint
                    ),
                }
            )
        ordinary_fork_event = event_with_runtime_generated_id(
            Event(
                id=(
                    "fork_"
                    + hashlib.sha256(
                        canonical_durable_json_bytes(
                            {
                                "child_session_id": fork_session.id,
                                "request_sha256": fork_request_sha256,
                            },
                            "profiled_fork_event",
                        )
                    ).hexdigest()
                ),
                type=EventType.SESSION_FORKED,
                session_id=fork_session.id,
                timestamp=fork_session.created_at,
                agent_name=registered_agent.spec.name,
                environment_name=_environment_name(registered_environment),
                payload=fork_event_payload,
            )
        )
        raw_fork_event = prompt_workflow.raw_fork_event(
            created=fork_session,
            ordinary_event=ordinary_fork_event,
        )
        raw_fork_event = raw_fork_event.model_copy(
            update={"payload": {**raw_fork_event.payload, **fork_event_payload}},
            deep=True,
        )
        fork_event = self._event_writer.prepare(_fork_event_with_runtime_authority(raw_fork_event))
        source_grant_interaction_id = (
            None if source_active_profile is None else source_active_profile.interaction_id
        )
        raw_grant_reset_event = event_with_runtime_envelope_authority(
            event_with_runtime_generated_id(
                Event(
                    id=f"{fork_event.id}:targeted-tool-grants-reset",
                    type=EventType.TARGETED_TOOL_GRANT_FORK_RESET,
                    session_id=fork_session.id,
                    timestamp=fork_session.created_at,
                    agent_name=registered_agent.spec.name,
                    environment_name=_environment_name(registered_environment),
                    payload={
                        "schema_version": 1,
                        "source_session_id": source_session.id,
                        "source_interaction_id": source_grant_interaction_id,
                        "inherited_grant_count": 0,
                        "inherited_reference_count": 0,
                    },
                )
            ),
            "session_id",
        )
        grant_reset_event = self._event_writer.prepare(
            event_with_runtime_payload_authority(
                raw_grant_reset_event,
                "source_session_id",
                *(("source_interaction_id",) if source_grant_interaction_id is not None else ()),
            )
        )
        decision_record: ForkExecutionProfileDecisionRecord | None = None
        prepared_events: list[Event] = []
        if execution_profile_decision is not None:
            if execution_profile_decision.actor is None:
                raise AssertionError("Current-child profile selection lost its actor.")
            decision_event = self._event_writer.prepare(execution_profile_decision.event)
            decision_payload = decision_event.payload
            decision_record = ForkExecutionProfileDecisionRecord.model_validate(
                {
                    "kind": execution_profile_decision.kind,
                    "policy_identity": decision_payload.get("policy_identity"),
                    "policy_reason": decision_payload.get("policy_reason"),
                    "authority_decision": execution_profile_decision.authority_decision,
                    "actor": decision_payload.get("actor"),
                    "reason": decision_payload.get("reason"),
                    "idempotency_identity": decision_payload.get("idempotency_identity"),
                    "adoption_request_fingerprint": fork_request_sha256,
                    "event_id": decision_event.id,
                }
            )
            prepared_events.append(decision_event)
        relationship = SessionForkProfileRelationship(
            schema_version=(
                FORK_EXECUTION_PROFILE_ORDINARY_SCHEMA_VERSION
                if expected_source_snapshot is None
                else FORK_EXECUTION_PROFILE_EXACT_SOURCE_SCHEMA_VERSION
            ),
            request_sha256=fork_request_sha256,
            source_state_sha256=(
                None
                if expected_source_snapshot is None
                else fork_source_state_sha256(expected_source_snapshot)
            ),
            source_session_id=source_session.id,
            child_session_id=fork_session.id,
            child_agent_name=fork_session.agent_name,
            child_provider_name=fork_session.provider_name,
            child_model=fork_session.model,
            child_environment_name=fork_session.environment_name,
            source_status=source_session.status,
            source_run_epoch=source_session.run_epoch,
            source_profile_source=source_profile_source,
            source_profile=source_execution_profile,
            source_active_interaction_id=(
                None if source_active_profile is None else source_active_profile.interaction_id
            ),
            source_active_run_epoch=(
                None if source_active_profile is None else source_active_profile.run_epoch
            ),
            transcript_cursor=request.transcript_cursor,
            copy_checkpoint=request.copy_checkpoint,
            system_prompt_policy=request.system_prompt_policy,
            selection=request.execution_profile_selection,
            selected_profile=selected_execution_profile,
            model_failover_candidate_index=fork_model_candidate_index,
            source_environment_allocation_owners=source_environment_allocation_owners,
            initial_invocation_request_sha256=(
                None
                if initial_invocation is None
                else _fork_initial_invocation_request_sha256(initial_invocation)
            ),
            initial_dispatch_id=request.initial_dispatch_id,
            initial_invocation_profile=initial_invocation_profile,
            decision=decision_record,
            fork_event_id=fork_event.id,
        )
        authoritative_metadata = copy_session_metadata(fork_session.metadata)
        authoritative_metadata[EXECUTION_PROFILE_METADATA_KEY] = execution_profile_session_metadata(
            selected_execution_profile
        )
        relationship_metadata = relationship.model_dump(mode="json")
        if relationship.source_state_sha256 is None:
            relationship_metadata.pop("source_state_sha256")
        authoritative_metadata[FORK_EXECUTION_PROFILE_METADATA_KEY] = relationship_metadata
        fork_session = fork_session.model_copy(
            update={"metadata": authoritative_metadata},
            deep=True,
        )
        prepared_events.append(fork_event)
        prepared_events.append(grant_reset_event)
        tool_discovery_initializer = _tool_discovery_operation_initializer(
            registered_agent,
            effective_tool_capability_ceiling,
        )

        async def validate_tool_discovery_view(created: Session) -> None:
            await _require_tool_discovery_view(
                self.session_store,
                created,
                registered_agent,
                effective_tool_capability_ceiling,
                # The child is visible after the store transaction commits. A
                # concurrent owner may legitimately discover a tool before this
                # acknowledgement or lost-ack reconciliation reads the record.
                require_empty=False,
            )

        async def reconcile_profiled_fork(
            initial_error: BaseException | None,
        ) -> tuple[ProfiledSessionForkResult, BaseException | None]:
            try:
                existing = await self.session_store.load(fork_session.id)
                if existing is None or not _prompt_fork_matches_prepared(
                    existing,
                    fork_session,
                ):
                    if initial_error is not None:
                        raise initial_error
                    raise RuntimeError(
                        "Profiled fork publication returned no authoritative result."
                    )
                existing_relationship = session_fork_profile_relationship(existing)
                if existing_relationship != relationship:
                    raise ForkRequestConflict(
                        "Existing profiled fork conflicts with the exact request."
                    )
                await validate_tool_discovery_view(existing)
                recovered_events: list[Event] = []
                for prepared_event in prepared_events:
                    records = await self.session_store.query_events(
                        EventQuery(
                            session_id=existing.id,
                            event_id=prepared_event.id,
                            limit=1,
                        )
                    )
                    if len(records) != 1 or records[0].event.model_dump(
                        mode="json",
                        exclude={"timestamp"},
                    ) != prepared_event.model_dump(
                        mode="json",
                        exclude={"timestamp"},
                    ):
                        raise RuntimeError(
                            "Existing profiled fork is missing exact durable evidence."
                        )
                    recovered_events.append(records[0].event)
                validate_profiled_fork_evidence(
                    fork=existing,
                    relationship=relationship,
                    events=recovered_events,
                )
                await advance_task_retention_epoch(
                    existing, session_store=self.session_store, task_store=self._task_store
                )
                recovered = ProfiledSessionForkResult(
                    session=existing,
                    events=tuple(recovered_events),
                )
                # Lost acknowledgements and malformed ordinary acknowledgements
                # are repaired by the immutable receipt. Process-control outcomes
                # remain authoritative even when their durable mutation committed.
                retained_error = (
                    initial_error
                    if initial_error is not None and not isinstance(initial_error, Exception)
                    else None
                )
                return recovered, retained_error
            except BaseException as reconciliation_error:
                if initial_error is None or reconciliation_error is initial_error:
                    raise
                raise BaseExceptionGroup(
                    "Profiled fork publication and exact reconciliation failed.",
                    [initial_error, reconciliation_error],
                ) from None

        async def publish_or_reconcile_profiled_fork() -> tuple[
            ProfiledSessionForkResult,
            BaseException | None,
        ]:
            try:
                fork_deadline.require_admission("fork_creation")
                raw_result = await self.session_store.create_profiled_fork(
                    source_session_id=source_session.id,
                    fork=fork_session.model_copy(deep=True),
                    source_statuses=set(_FORKABLE_SESSION_STATUSES),
                    transcript_cursor=request.transcript_cursor,
                    checkpoint_transform=checkpoint_transform,
                    system_prompt_replacement=prompt_workflow.prompt_replacement,
                    expected_source_run_epoch=source_session.run_epoch,
                    relationship=SessionForkProfileRelationship.model_validate(
                        relationship.model_dump(mode="json")
                    ),
                    events=[copy_event(event) for event in prepared_events],
                    transcript_validator=transcript_validator,
                    operation_initializer=tool_discovery_initializer,
                )
                try:
                    result = copy_profiled_session_fork_result(raw_result)
                    if not _prompt_fork_matches_prepared(
                        result.session,
                        fork_session,
                    ):
                        raise RuntimeError(
                            "Profiled fork publication returned a different session snapshot."
                        )
                    await validate_tool_discovery_view(result.session)
                    validate_profiled_fork_evidence(
                        fork=result.session,
                        relationship=relationship,
                        events=result.events,
                    )
                    await advance_task_retention_epoch(
                        result.session,
                        session_store=self.session_store,
                        task_store=self._task_store,
                    )
                    return result, None
                except Exception as invalid_result:
                    return await reconcile_profiled_fork(invalid_result)
            except asyncio.CancelledError as child_cancellation:
                return await reconcile_profiled_fork(
                    unexpected_child_cancellation_error(
                        child_cancellation,
                        operation="Profiled fork durable publication",
                    )
                )
            except BaseException as initial_error:
                return await reconcile_profiled_fork(initial_error)

        async def await_owned_fork_task(
            task: asyncio.Task[Any],
        ) -> tuple[ShieldedTaskOutcome[Any], list[BaseException]]:
            """Settle one dispatched fork mutation without losing supervisor control."""

            supervisory_controls: list[BaseException] = []
            while True:
                try:
                    return await await_shielded_task_outcome(task), supervisory_controls
                except asyncio.CancelledError:
                    # New Task.cancel() delivery is owned by the shield helper. An
                    # unowned cancellation must retain its normal caller meaning.
                    raise
                except BaseException as control:
                    # Generator/stream abandonment and process control cannot
                    # orphan a durable mutation. Retain the exact signal until
                    # the owned task reaches a terminal outcome.
                    supervisory_controls.append(control)

        publication_task = asyncio.create_task(
            capture_awaitable_outcome(publish_or_reconcile_profiled_fork),
            name="cayu-profiled-session-fork-publication",
        )
        publication_outcome, publication_controls = await await_owned_fork_task(publication_task)
        publication_error = publication_outcome.error
        publication_result = publication_outcome.result
        if publication_error is None and publication_result is not None:
            publication_error = publication_result.error
            publication_value = publication_result.result
        else:
            publication_value = None
        if (
            isinstance(publication_error, asyncio.CancelledError)
            and publication_outcome.cancellation is None
        ):
            publication_error = unexpected_child_cancellation_error(
                publication_error,
                operation="Profiled fork durable publication",
            )
        if publication_error is not None:
            if (
                publication_outcome.cancellation is not None
                and publication_outcome.cancellation_requests_consumed
            ):
                retain_workspace_observation_pending_cancellation_requests(
                    publication_outcome.cancellation,
                    publication_outcome.cancellation_requests_consumed,
                )
            restore_task_cancellation_requests(publication_outcome.cancellation_requests_consumed)
            if publication_outcome.cancellation is not None:
                if isinstance(publication_error, Exception) and not publication_controls:
                    raise publication_outcome.cancellation from publication_error
                concurrent_control = BaseExceptionGroup(
                    "Profiled fork publication failed concurrently with caller cancellation.",
                    [
                        publication_error,
                        *publication_controls,
                        publication_outcome.cancellation,
                    ],
                )
                retain_workspace_observation_pending_cancellation_requests(
                    concurrent_control,
                    publication_outcome.cancellation_requests_consumed,
                )
                raise concurrent_control from None
            if publication_controls:
                raise BaseExceptionGroup(
                    "Profiled fork publication failed with concurrent process control.",
                    [publication_error, *publication_controls],
                ) from None
            if isinstance(publication_error, SessionForkSourceNotFound):
                raise session_request_boundary.ForkSourceNotFoundError(
                    "Fork source session was not found."
                ) from None
            if isinstance(publication_error, SessionForkActiveModelStageConflict):
                raise session_request_boundary.ForkActiveModelStageError(
                    "Fork source session has an active model-completion stage."
                ) from None
            raise publication_error
        if publication_value is None:
            if (
                publication_outcome.cancellation is not None
                and publication_outcome.cancellation_requests_consumed
            ):
                retain_workspace_observation_pending_cancellation_requests(
                    publication_outcome.cancellation,
                    publication_outcome.cancellation_requests_consumed,
                )
            restore_task_cancellation_requests(publication_outcome.cancellation_requests_consumed)
            missing_result = RuntimeError(
                "Profiled fork publication returned no authoritative result."
            )
            if publication_outcome.cancellation is not None:
                if not publication_controls:
                    raise publication_outcome.cancellation from missing_result
                concurrent_control = BaseExceptionGroup(
                    "Profiled fork publication returned no result with concurrent control.",
                    [
                        missing_result,
                        *publication_controls,
                        publication_outcome.cancellation,
                    ],
                )
                if publication_outcome.cancellation_requests_consumed:
                    retain_workspace_observation_pending_cancellation_requests(
                        concurrent_control,
                        publication_outcome.cancellation_requests_consumed,
                    )
                raise concurrent_control from None
            if publication_controls:
                raise BaseExceptionGroup(
                    "Profiled fork publication returned no result with process control.",
                    [missing_result, *publication_controls],
                ) from None
            raise missing_result
        profiled_result, post_commit_control = publication_value
        created = profiled_result.session
        if (
            created.id != fork_session.id
            or created.parent_session_id != fork_session.parent_session_id
            or created.causal_budget_id != fork_session.causal_budget_id
            or session_fork_profile_relationship(created) != relationship
            or len(profiled_result.events) != len(prepared_events)
            or any(
                stored.model_dump(mode="json", exclude={"timestamp"})
                != prepared.model_dump(mode="json", exclude={"timestamp"})
                for stored, prepared in zip(
                    profiled_result.events,
                    prepared_events,
                    strict=True,
                )
            )
        ):
            raise RuntimeError("Session store changed prepared fork identity authority.")
        completion_task = asyncio.create_task(
            capture_awaitable_outcome(
                lambda: prompt_workflow.complete_effect(
                    session_store=self.session_store,
                    source_session_id=source_session.id,
                    created=created,
                    clock=self._clock,
                )
            ),
            name="cayu-profiled-session-fork-effect-completion",
        )
        completion_outcome, completion_controls = await await_owned_fork_task(completion_task)
        completion_error = completion_outcome.error
        if completion_error is None and completion_outcome.result is not None:
            completion_error = completion_outcome.result.error
        if (
            isinstance(completion_error, asyncio.CancelledError)
            and completion_outcome.cancellation is None
        ):
            completion_error = unexpected_child_cancellation_error(
                completion_error,
                operation="Profiled fork effect completion",
            )
        authoritative_cancellation = (
            publication_outcome.cancellation or completion_outcome.cancellation
        )
        cancellation_requests_consumed = (
            publication_outcome.cancellation_requests_consumed
            + completion_outcome.cancellation_requests_consumed
        )
        concurrent_failures: list[BaseException] = []
        for failure in (
            post_commit_control,
            completion_error,
            *publication_controls,
            *completion_controls,
        ):
            if isinstance(failure, BaseException):
                concurrent_failures.append(failure)
            elif failure is not None:
                concurrent_failures.append(
                    RuntimeError("Profiled fork completion returned invalid failure evidence.")
                )
        if authoritative_cancellation is not None:
            if cancellation_requests_consumed:
                retain_workspace_observation_pending_cancellation_requests(
                    authoritative_cancellation,
                    cancellation_requests_consumed,
                )
            restore_task_cancellation_requests(cancellation_requests_consumed)
            ordinary_failures = [
                failure for failure in concurrent_failures if isinstance(failure, Exception)
            ]
            if concurrent_failures and len(ordinary_failures) == len(concurrent_failures):
                cleanup_failure = (
                    ordinary_failures[0]
                    if len(ordinary_failures) == 1
                    else ExceptionGroup(
                        "Profiled fork post-commit completion failed.",
                        ordinary_failures,
                    )
                )
                raise authoritative_cancellation from cleanup_failure
            if concurrent_failures:
                concurrent_control = BaseExceptionGroup(
                    "Profiled fork committed with concurrent process control and cancellation.",
                    [*concurrent_failures, authoritative_cancellation],
                )
                if cancellation_requests_consumed:
                    retain_workspace_observation_pending_cancellation_requests(
                        concurrent_control,
                        cancellation_requests_consumed,
                    )
                raise concurrent_control from None
            raise authoritative_cancellation
        if concurrent_failures:
            if len(concurrent_failures) == 1:
                raise concurrent_failures[0]
            raise BaseExceptionGroup(
                "Profiled fork post-commit completion failed.",
                concurrent_failures,
            ) from None
        delivered_events = await self._event_writer.fan_out_persisted(list(profiled_result.events))
        delivered_fork_event = next(
            (event for event in delivered_events if event.type is EventType.SESSION_FORKED),
            None,
        )
        if delivered_fork_event is None:
            raise RuntimeError("Profiled fork publication lost its public fork event.")
        # Fork orchestration has always exposed exactly the session.forked
        # boundary. The adjacent grant-reset record remains durable and is
        # delivered to event sinks without changing that public stream contract.
        yield delivered_fork_event
