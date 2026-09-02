"""Tool-round policy, execution, pause, hook, and closure ownership.

This module is deliberately below :class:`CayuApp`.  It owns one complete
tool-round lifecycle without importing or accepting the application facade.
Session-level limit terminalization and interrupted-round recovery remain
orchestration boundaries supplied by the session engine and recovery coordinator
through narrow callbacks.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
from collections.abc import AsyncIterator, Awaitable, Callable, Iterable, Mapping
from contextlib import suppress
from copy import deepcopy
from dataclasses import dataclass, replace
from datetime import datetime, timedelta
from types import MappingProxyType
from typing import Any, Literal, Never, TypeVar, cast
from uuid import uuid4

from cayu._exception_groups import (
    exception_cause,
    exception_group_children,
    iter_exception_tree,
    set_exception_cause,
)
from cayu._task_wait import (
    CapturedAwaitableOutcome,
    await_shielded_task_outcome,
    capture_awaitable_outcome,
    consume_pending_task_cancellation,
    restore_task_cancellation_requests,
    unexpected_child_cancellation_error,
)
from cayu._validation import (
    copy_durable_json_object,
    copy_durable_json_value,
    copy_json_value,
    require_clean_nonblank,
    require_durable_text,
    require_nonblank,
    require_unicode_scalar_text,
)
from cayu._workspace_mutation import workspace_mutation_task_settlement_probe
from cayu.artifacts import ArtifactScope, LocalArtifactStore
from cayu.core.agents import AgentSpec
from cayu.core.events import (
    Event,
    EventType,
    copy_event,
    event_payload_authority_is_runtime_generated,
    event_retains_runtime_payload_authority,
    event_with_runtime_envelope_authority,
    event_with_runtime_generated_id,
    event_with_runtime_nested_payload_authority,
    event_with_runtime_payload_authority,
)
from cayu.core.messages import Message
from cayu.core.thinking import ThinkingConfig
from cayu.core.tools import (
    _TOOL_POLICY_DENIAL_SOURCE,
    DurableToolOperationConflict,
    ToolContext,
    ToolEffect,
    ToolResult,
    _bind_runtime_tool_invocation_authority,
    _bound_policy_denial_result,
    _bound_policy_denial_text,
)
from cayu.environments import BoundWorkspace
from cayu.environments.bindings import _runtime_owned_workspace_observer_name
from cayu.mcp import McpToolAdapter, McpToolset
from cayu.runners._cleanup import (
    attach_runner_cancellation_failure,
    pop_runner_cancellation_failure,
    runner_cancellation_failure,
    sanitize_runner_artifacts,
    transfer_runner_cancellation_failures,
)
from cayu.runners.base import attach_cancellation_artifacts
from cayu.runtime import _approval_publication as approval_publication
from cayu.runtime import _approval_support as approval_support
from cayu.runtime import _invocation_secrets as invocation_secrets
from cayu.runtime import _runtime_records as runtime_records
from cayu.runtime import _shared_artifact_results as shared_artifact_results
from cayu.runtime import _tool_argument_publication as tool_argument_publication
from cayu.runtime import _tool_execution as tool_execution
from cayu.runtime import _tool_results as tool_results
from cayu.runtime import _tool_round_publication as tool_round_publication
from cayu.runtime import _tool_round_recovery as tool_round_recovery
from cayu.runtime import _transcript as transcript_helpers
from cayu.runtime import _web_access_results as web_access_results
from cayu.runtime._assistant_tool_round_publication import (
    validate_tool_exposure_terminal_event,
)
from cayu.runtime._checkpoint_redaction import (
    require_secret_free_durable_object as _require_secret_free_durable_object,
)
from cayu.runtime._event_projection import PRIVATE_EVENT_AUTHORITY
from cayu.runtime._event_writer import RuntimeEventWriter, prepare_runtime_event
from cayu.runtime._interruption_coordinator import (
    _INTERRUPTION_TYPE_OPERATOR_REQUESTED,
    _PENDING_SESSION_INTERRUPT_CHECKPOINT_KEY,
)
from cayu.runtime._invocation_lifecycle import InvocationContext
from cayu.runtime._run_limits import (
    LimitEvaluation,
    RunLimitGate,
    SessionUsageTracker,
)
from cayu.runtime._session_control import (
    ActiveSessionRun,
    SessionControl,
    SessionInterruptedByRequest,
    clear_current_task_cancellation,
)
from cayu.runtime._session_queries import query_all_event_records
from cayu.runtime.approvals import (
    PendingToolApproval,
    PendingToolCallApproval,
    ToolPolicyEvidence,
    copy_pending_tool_approval,
)
from cayu.runtime.budgets import BudgetLimit, copy_request_budget_limits
from cayu.runtime.execution_profiles import (
    EXECUTION_PROFILE_FINGERPRINT_FIELD,
    ExecutionProfileIdentity,
    active_invocation_execution_profile_from_checkpoint,
    event_with_execution_profile_authority,
    event_with_execution_profile_fingerprint_authority,
)
from cayu.runtime.execution_units import (
    ModelAttemptIdentity,
    ToolRoundIdentity,
    copy_tool_round_identity,
)
from cayu.runtime.hooks import (
    AfterToolCallDecision,
    BeforeToolCallDecision,
    BeforeToolCallHookContext,
    RuntimeHookPhase,
    RuntimeHookRuntime,
    ToolCallHookContext,
    _runtime_hook_supports_phase,
)
from cayu.runtime.hooks import (
    _runtime_hook_event as _build_runtime_hook_event,
)
from cayu.runtime.mcp_manifest_policy import (
    McpManifestPolicy,
    McpManifestPolicyAction,
    McpManifestPolicyDecision,
    McpManifestPolicyError,
    mcp_manifest_policy_payload,
)
from cayu.runtime.public_authority import parse_public_authority_alias
from cayu.runtime.retry_policy import RetryPolicy, copy_retry_policy
from cayu.runtime.sessions import (
    _MCP_MANIFEST_BASELINE_MAX_TOOLS,
    INHERIT_INTERACTION,
    EventQuery,
    McpManifestBaseline,
    McpManifestBaselineLoadResult,
    McpManifestHistoryConflict,
    McpManifestPublicationResult,
    Session,
    SessionOperationPublication,
    SessionRunFenced,
    SessionStatus,
    SessionStore,
    _mcp_authoritative_manifest_hash,
    _mcp_manifest_session_ref,
    _McpManifestBaselineEvidenceInvalid,
    resolve_interaction_attribution,
    runtime_publication_checkpoint_value_digest,
)
from cayu.runtime.stop_policy import RunLimits, copy_run_limits
from cayu.runtime.structured_output import (
    StructuredOutputSpec,
    copy_structured_output_spec,
)
from cayu.runtime.tool_catalogue import (
    CALL_TOOL_NAME,
    SEARCH_TOOLS_NAME,
    ToolExecutionContract,
)
from cayu.runtime.tool_discovery import (
    TOOL_DISCOVERY_REFERENCE_PREFIX,
    TOOL_DISCOVERY_VIEW_OPERATION_KEY,
    _bind_runtime_tool_discovery_authority,
    current_tool_discovery_view,
    discovered_tool_rejection_event,
    resolved_discovered_tool_invocation,
    tool_discovery_generation_id,
    tool_discovery_record_matches_descriptor,
    tool_discovery_reference_rejection_reason,
)
from cayu.runtime.tool_exposure import (
    NOT_EXPOSED_IN_REQUEST_REASON,
    ResolvedToolExposureAuthority,
    copy_resolved_tool_exposure_authority,
    tool_capability_ceiling_from_session_metadata,
    unexposed_tool_result,
    validate_resolved_tool_exposure_authority,
)
from cayu.runtime.tool_gateway import (
    CallToolEnvelope,
    rejected_targeted_tool_invocation,
    resolved_targeted_tool_invocation,
    targeted_tool_rejection_content,
    unresolved_gateway_rejection_event,
    validate_effective_tool_arguments,
)
from cayu.runtime.tool_gateway import (
    arguments_sha256 as targeted_arguments_sha256,
)
from cayu.runtime.tool_grants import (
    TARGETED_TOOL_REFERENCE_FIELD_NAME,
    TargetedToolUseDisposition,
    TargetedToolUseRejectionReason,
    TargetedToolUseRequest,
    targeted_tool_use_rejection_event,
    targeted_tool_use_rejection_reason,
    targeted_tool_view_generation_id,
)
from cayu.runtime.tool_policy import (
    TAINT_LABELS_METADATA_KEY,
    TOOL_POLICY_REAUTHORIZATION_METADATA_KEY,
    TaintAwareToolPolicy,
    ToolPolicy,
    ToolPolicyDecision,
    ToolPolicyRequest,
    ToolPolicyResult,
    metadata_with_taint_labels,
    taint_labels_from_metadata,
)
from cayu.runtime.tool_result_projection import (
    _TOOL_RESULT_PROJECTION_PROVENANCE_PATH,
    ToolResultProjectionPolicy,
    ToolResultProjectionRequest,
    projection_failure,
    safe_projection_failure_type,
    validate_tool_result_projection,
)
from cayu.runtime.user_input import (
    PENDING_USER_INPUT_CHECKPOINT_KEY,
    PendingUserInput,
    copy_pending_user_input,
    event_with_pending_user_input_authority,
    pending_user_input_digest,
    pending_user_input_identity,
    public_pending_user_input_event_payload,
    public_pending_user_input_prompt,
    user_input_lifecycle_authority_from_checkpoint,
)
from cayu.runtime.workspace_mutation_attribution import (
    DirectWorkspaceMutationCollector,
    WorkspaceMutationWindow,
    begin_workspace_mutation_window,
    classify_workspace_mutation_attribution,
    direct_workspace_mutation_payload,
    observed_pre_window_change,
    reconcile_direct_workspace_mutations,
)
from cayu.runtime.workspace_observation_recovery import (
    WORKSPACE_OBSERVATION_TERMINAL_CONTROLS,
    WorkspaceObservationArtifact,
    WorkspaceObservationArtifactState,
    WorkspaceObservationEvidenceState,
    WorkspaceObservationLifecycle,
    WorkspaceObservationPhase,
    WorkspaceObservationTerminalStatus,
    _admit_workspace_observation_intent,
    _project_workspace_observation_authority,
    _WorkspaceObservationAuthorityProjection,
    await_workspace_observation_store_read,
    publish_workspace_observation_transition,
    raise_workspace_observation_concurrent_control,
    restore_workspace_observation_cancellation_requests,
    workspace_observation_artifact_metadata_matches,
    workspace_observation_event_digest,
    workspace_observation_terminal_from_delta_status,
    workspace_observations_from_checkpoint,
)
from cayu.tools._operation_boundary import (
    BoundedInvocationOperationRegistry,
    await_invocation_operation,
)
from cayu.tools._redaction import InvocationRedactorSnapshot
from cayu.tools._resources import (
    InvocationWorkspaceMutationOwner,
    WorkspaceMutationSettlementError,
    invocation_artifact_store_handle,
    invocation_workspace_handle,
)
from cayu.tools._runner import (
    invocation_runner_handle,
    is_current_runner_cancellation_group,
    sanitize_runner_failure,
    sanitize_runner_failure_group,
)
from cayu.vaults import SecretRedactor
from cayu.workspaces import (
    LocalWorkspace,
    WorkspaceDirectMutationReconciliation,
    WorkspaceIdentity,
    WorkspaceMutationAttribution,
    WorkspaceMutationAttributionConfidence,
    WorkspaceRevisionDelta,
    WorkspaceRevisionDeltaStatus,
    WorkspaceRevisionObservation,
    WorkspaceRevisionObservationLimits,
    WorkspaceRevisionObservationStatus,
    WorkspaceWriterIsolationEvidence,
    compare_workspace_revisions,
)
from cayu.workspaces.revisions import (
    _WORKSPACE_PATH_REVISION_AUTHORITY_FIELDS,
    _WORKSPACE_PATH_REVISION_DELTA_AUTHORITY_FIELDS,
    WorkspaceRevisionObservationLimitExceeded,
    copy_bounded_workspace_revision_observation,
    unsupported_workspace_revision,
)


def _event_with_tool_round_authority(
    event: Event,
    identity: ToolRoundIdentity,
    *additional_fields: str,
) -> Event:
    """Attest only runtime-owned linkage carried by a typed tool-round identity."""

    identity = copy_tool_round_identity(identity)
    fields = [
        field_name
        for field_name, value in identity.payload().items()
        if event.payload.get(field_name) == value
    ]
    for field_name in additional_fields:
        if field_name in event.payload:
            fields.append(field_name)
    event = event_with_runtime_envelope_authority(event, "session_id")
    return event_with_runtime_payload_authority(event, *fields) if fields else event


def _event_with_workspace_observation_authority(
    event: Event,
    identity: ToolRoundIdentity,
    interaction_id: str | None,
    *additional_fields: str,
) -> Event:
    """Attest an observation event only from its typed lifecycle owner."""

    event = _event_with_tool_round_authority(event, identity, *additional_fields)
    if (
        event.type
        in {
            EventType.WORKSPACE_REVISION_OBSERVED,
            EventType.WORKSPACE_MUTATION_RECORDED,
        }
        and "paths" in event.payload
    ):
        authority_fields = (
            _WORKSPACE_PATH_REVISION_AUTHORITY_FIELDS
            if event.type is EventType.WORKSPACE_REVISION_OBSERVED
            else _WORKSPACE_PATH_REVISION_DELTA_AUTHORITY_FIELDS
        )
        event = event_with_runtime_nested_payload_authority(
            event,
            *(("paths", "*", field_name) for field_name in sorted(authority_fields)),
        )
    if interaction_id is None:
        if event.interaction_id is not None:
            raise ValueError("Workspace observation event changed its interaction identity.")
        return event
    interaction_id = require_clean_nonblank(interaction_id, "interaction_id")
    if event.interaction_id != interaction_id:
        raise ValueError("Workspace observation event conflicts with its interaction owner.")
    return event_with_runtime_envelope_authority(event, "interaction_id")


_TOOL_RESULT_PROJECTION_TIMEOUT_SECONDS = 30.0
_WORKSPACE_RECEIPT_INLINE_PATH_LIMIT = 32
_WORKSPACE_RECEIPT_INLINE_BYTES = 8 * 1024
_WORKSPACE_OBSERVATION_RUNTIME_LIMITS = WorkspaceRevisionObservationLimits()
_WORKSPACE_OBSERVATION_MAX_TOTAL_PATHS = (1 << 63) - 1
_WORKSPACE_OBSERVATION_TIMEOUT_SECONDS = 30.0
_WORKSPACE_ARTIFACT_WRITE_TIMEOUT_SECONDS = 30.0
_MAX_RETAINED_WORKSPACE_CAPTURE_OPERATIONS = 64


@dataclass(frozen=True)
class _WorkspaceEvidenceProjection:
    paths: list[dict[str, Any]]
    status: str
    detail_code: str | None
    manifest_artifact_id: str | None = None
    manifest_artifact_sha256: str | None = None
    manifest_artifact_size_bytes: int | None = None
    evidence_available: bool = True


@dataclass(frozen=True)
class _WorkspaceCaptureResult:
    lifecycle: WorkspaceObservationLifecycle
    delta_lifecycle: WorkspaceObservationLifecycle
    events: tuple[Event, ...]
    receipt_event: Event
    terminal_status: WorkspaceObservationTerminalStatus
    terminal_detail_code: str | None


CheckpointTransform = Callable[
    [Session, dict[str, Any] | None],
    dict[str, Any],
]
CheckpointTransformFactory = Callable[[dict[str, Any]], CheckpointTransform]
_INTERRUPTION_TYPE_TOOL_APPROVAL_REQUIRED = "tool_approval_required"


@dataclass(frozen=True)
class ToolRoundLimitRequest:
    evaluation: LimitEvaluation
    session: Session
    registered_agent: runtime_records.RegisteredAgentState
    registered_environment: runtime_records.RegisteredEnvironment | None
    environment_name: str | None
    messages: list[Message]
    tool_calls: list[runtime_records.ToolCallRequest]
    completed_tool_outcomes: list[runtime_records.ToolCallOutcome]
    tool_round_identity: ToolRoundIdentity
    run_started_at: float
    turn_usage_tracker: SessionUsageTracker | None
    active_run: ActiveSessionRun[SessionUsageTracker] | None
    execution_profile: ExecutionProfileIdentity | None
    invocation_context: InvocationContext | None = None


@dataclass(frozen=True)
class InterruptedToolRoundRequest:
    session: Session
    registered_agent: runtime_records.RegisteredAgentState
    registered_environment: runtime_records.RegisteredEnvironment | None
    messages: list[Message]
    tool_calls: list[runtime_records.ToolCallRequest]
    tool_outcomes: list[runtime_records.ToolCallOutcome]
    tool_round_identity: ToolRoundIdentity
    cancellation_artifacts: list[dict[str, Any]] | None
    cancellation_artifacts_by_id: dict[str, list[dict[str, Any]]] | None
    cancellation_redactors_by_id: dict[str, SecretRedactor] | None = None
    execution_profile: ExecutionProfileIdentity | None = None
    invocation_context: InvocationContext | None = None


LimitEventStream = Callable[[ToolRoundLimitRequest], AsyncIterator[Event]]
InterruptedRoundEventStream = Callable[[InterruptedToolRoundRequest], AsyncIterator[Event]]
DeferredTerminalStager = Callable[
    [
        Event,
        runtime_records.ToolCallOutcome,
        bool,
        bool,
        invocation_secrets.InvocationPublicationSnapshot,
    ],
    Awaitable[Event],
]
DeferredTerminalFinalizer = Callable[[Event], Awaitable[Event]]
DeferredTerminalCaptureRecorder = Callable[[Event], Awaitable[Event]]

_AMBIGUOUS_POLICY_RECOVERY_REASON = (
    "Tool policy planning was interrupted without a durable outcome; "
    "explicit approval is required before execution."
)
_AMBIGUOUS_POLICY_RECOVERY_METADATA = {
    "recovered": True,
    "policy_evaluation": "ambiguous",
}
_TARGETED_TOOL_INVOCATION_PAYLOAD_FIELDS = (
    "dispatch_kind",
    "model_tool_name",
    "grant_id",
    "use_id",
    "effective_tool_id",
    "catalogue_revision",
    "descriptor_version",
    "schema_fingerprint",
    "arguments_sha256",
    "invocation_id",
)


def _targeted_tool_invocation_payload(
    tool_call: runtime_records.ToolCallRequest,
) -> dict[str, Any]:
    invocation = tool_call.targeted_tool_invocation
    if invocation is None:
        return {}
    payload = {
        "dispatch_kind": invocation.dispatch_kind,
        "model_tool_name": invocation.model_tool_name,
        "grant_id": invocation.grant_id,
        "use_id": invocation.use_id,
        "effective_tool_id": invocation.tool_id,
        "catalogue_revision": invocation.catalogue_revision,
        "descriptor_version": invocation.descriptor_version,
        "schema_fingerprint": invocation.schema_fingerprint,
        "arguments_sha256": invocation.arguments_sha256,
        "invocation_id": invocation.invocation_id,
    }
    if tuple(payload) != _TARGETED_TOOL_INVOCATION_PAYLOAD_FIELDS:
        raise AssertionError("Targeted tool invocation payload fields drifted.")
    return payload


def _event_with_targeted_tool_invocation_authority(
    event: Event,
    tool_call: runtime_records.ToolCallRequest,
) -> Event:
    """Attest exact targeted-tool linkage copied from one resolved invocation."""

    if tool_call.targeted_tool_invocation is None:
        return event
    return event_with_runtime_payload_authority(
        event,
        *_TARGETED_TOOL_INVOCATION_PAYLOAD_FIELDS,
    )


def _restore_targeted_tool_invocation_event_authority(
    event: Event,
    tool_call: runtime_records.ToolCallRequest,
    *,
    redactor: SecretRedactor | None = None,
) -> Event:
    """Rebind one terminal returned through a durable checkpoint callback."""

    invocation = tool_call.targeted_tool_invocation
    if invocation is None:
        return event
    if event.tool_name != tool_call.name or event.payload.get("tool_call_id") != tool_call.id:
        raise RuntimeError("Targeted tool terminal conflicts with its resolved invocation.")
    expected = _targeted_tool_invocation_payload(tool_call)
    for field_name, expected_value in expected.items():
        observed = event.payload.get(field_name)
        retained_authority = event_retains_runtime_payload_authority(
            event,
            field_name=field_name,
            value=expected_value,
        )
        redacted_attested_authority = retained_authority and (
            observed == PRIVATE_EVENT_AUTHORITY
            or (redactor is not None and observed == redactor.redact_json(expected_value))
        )
        if (
            field_name in event.payload
            and observed != expected_value
            and not redacted_attested_authority
        ):
            raise RuntimeError(
                "Targeted tool terminal conflicts with its durable invocation authority."
            )
    rebound = event.model_copy(update={"payload": {**event.payload, **expected}})
    return _event_with_targeted_tool_invocation_authority(rebound, tool_call)


def _require_matching_policy_round(
    *,
    pending_round: tool_round_recovery.PendingToolRound,
    tool_round_identity: ToolRoundIdentity,
    tool_calls: list[runtime_records.ToolCallRequest],
) -> None:
    if tool_round_recovery.pending_tool_round_identity(pending_round) != tool_round_identity:
        raise RuntimeError("Pending tool round identity changed before policy publication.")
    expected_calls = [runtime_records.copy_tool_call_request(call) for call in tool_calls]
    pending_calls = tool_round_recovery.pending_round_tool_calls(pending_round)
    if pending_calls != expected_calls:
        raise RuntimeError("Pending tool round calls changed before policy publication.")


def _planned_pending_tool_round(
    *,
    pending_round: tool_round_recovery.PendingToolRound,
    tool_calls: list[runtime_records.ToolCallRequest],
    policy_outcomes: list[runtime_records.ToolCallPolicyOutcome] | None,
    active_taint_by_id: Mapping[str, frozenset[str]],
    redactor: SecretRedactor,
    deferred_messages: list[Message] | None = None,
) -> tool_round_recovery.PendingToolRound:
    payload = pending_round.model_dump(mode="json")
    # Checkpoints written before policy-context versioning are valid while the
    # originating run is still evaluating their policy plan. Publishing a
    # freshly evaluated live-run plan, or a fail-closed ambiguous recovery
    # plan, is the authoritative migration boundary. Recovery preserves an
    # unversioned round only when every registered call already carries a
    # recognized durable policy decision.
    payload["policy_context_version"] = 1
    payload["policy_state"] = "planned"
    payload["tool_calls"] = [
        call.model_dump(mode="json")
        for call in approval_support.pending_tool_call_approvals(
            tool_calls=tool_calls,
            policy_outcomes=policy_outcomes,
            active_taint_by_id=active_taint_by_id,
            redactor=redactor,
        )
    ]
    if deferred_messages is not None:
        payload["deferred_messages"] = [
            message.model_dump(mode="json") for message in deferred_messages
        ]
    return tool_round_recovery.PendingToolRound.model_validate(payload)


class ToolApprovalRequired(Exception):
    """Internal control signal for a durably checkpointed approval pause."""

    def __init__(self, approval: PendingToolApproval) -> None:
        super().__init__(f"Tool call requires approval: {approval.tool_name}")
        self.approval = copy_pending_tool_approval(approval)


class UserInputRequired(Exception):
    """Internal control signal for a durably checkpointed user-input pause."""

    def __init__(self, pending: PendingUserInput) -> None:
        super().__init__(f"Tool call awaits user input: {pending.tool_name}")
        self.pending = copy_pending_user_input(pending)


def _consume_projection_task_outcome(task: asyncio.Task[Any]) -> None:
    """Observe a timed-out policy task after requesting cooperative cancellation."""

    with suppress(BaseException):
        task.result()


class _ToolRoundPublicationCoordinator:
    """Serialize actual secret discovery with private terminal staging."""

    def __init__(
        self,
        *,
        session_id: str,
        tool_round_identity: ToolRoundIdentity,
        session_store: SessionStore,
        redactor: SecretRedactor,
        execution_profile: ExecutionProfileIdentity | None,
        tool_exposure: ResolvedToolExposureAuthority | None = None,
    ) -> None:
        self._session_id = require_clean_nonblank(session_id, "session_id")
        self._tool_round_identity = copy_tool_round_identity(tool_round_identity)
        self._session_store = session_store
        self._redactor = redactor
        if (
            execution_profile is not None
            and type(execution_profile) is not ExecutionProfileIdentity
        ):
            raise TypeError("execution_profile must be an ExecutionProfileIdentity or None.")
        self._execution_profile_fingerprint = (
            None if execution_profile is None else execution_profile.fingerprint
        )
        self._tool_exposure = (
            None if tool_exposure is None else copy_resolved_tool_exposure_authority(tool_exposure)
        )
        self._unsafe_tool_call_ids: set[str] = set()
        self._lock = asyncio.Lock()

    @property
    def redactor(self) -> SecretRedactor:
        return self._redactor

    @property
    def tool_round_identity(self) -> ToolRoundIdentity:
        return copy_tool_round_identity(self._tool_round_identity)

    @property
    def argument_scope_finalized(self) -> bool:
        """Return whether every sealed call contributed complete secret evidence."""

        return not self._unsafe_tool_call_ids

    def restore_staged_event_authority(self, event: Event) -> Event:
        restored = restore_staged_terminal_authority(
            event,
            session_id=self._session_id,
            tool_round_identity=self._tool_round_identity,
            tool_exposure=self._tool_exposure,
        )
        restored = web_access_results.restore_persisted_web_access_result_authority(restored)
        restored = shared_artifact_results.restore_persisted_shared_artifact_result_authority(
            restored
        )
        observed_fingerprint = restored.payload.get(EXECUTION_PROFILE_FINGERPRINT_FIELD)
        if (
            observed_fingerprint is not None
            and observed_fingerprint != self._execution_profile_fingerprint
        ):
            raise RuntimeError("Staged terminal conflicts with its execution profile owner.")
        return event_with_execution_profile_fingerprint_authority(
            restored,
            self._execution_profile_fingerprint,
        )

    async def register_redactor(
        self,
        *,
        tool_call_id: str,
        redactor: SecretRedactor,
    ) -> None:
        """Persist one real invocation redactor before its secret is returned."""

        async with self._lock:
            self._redactor = self._redactor.merged_with(redactor)
            await self._session_store.transform_checkpoint(
                self._session_id,
                self._checkpoint_transform(
                    tool_call_id=tool_call_id,
                    cover_call=False,
                    staged_terminal=None,
                ),
            )

    async def seal_call(
        self,
        *,
        tool_call_id: str,
        snapshot: invocation_secrets.InvocationPublicationSnapshot,
    ) -> None:
        """Durably cover a call whose terminal outcome will be synthesized."""

        async with self._lock:
            self._redactor = self._redactor.merged_with(snapshot.redactor)
            if snapshot.secret_scope_incomplete:
                self._unsafe_tool_call_ids.add(tool_call_id)
            await self._session_store.transform_checkpoint(
                self._session_id,
                self._checkpoint_transform(
                    tool_call_id=tool_call_id,
                    cover_call=True,
                    unsafe_scope=snapshot.secret_scope_incomplete,
                    staged_terminal=None,
                ),
            )

    async def stage_terminal(
        self,
        *,
        tool_call_id: str,
        event: Event,
        snapshot: invocation_secrets.InvocationPublicationSnapshot,
        hooks_state: Literal["pending", "finalized", "observational", "completed"],
    ) -> Event:
        """Persist a stable terminal event after applying the cumulative scope."""

        if event.session_id != self._session_id:
            raise ValueError("Staged terminal event belongs to a different session.")
        async with self._lock:
            self._redactor = self._redactor.merged_with(snapshot.redactor)
            if snapshot.secret_scope_incomplete:
                self._unsafe_tool_call_ids.add(tool_call_id)
            staged = tool_round_recovery.StagedToolCallTerminal(
                tool_call_id=tool_call_id,
                event=_project_staged_terminal_event(event, redactor=self._redactor),
                hooks_state=hooks_state,
            )
            await self._session_store.transform_checkpoint(
                self._session_id,
                self._checkpoint_transform(
                    tool_call_id=tool_call_id,
                    cover_call=True,
                    unsafe_scope=snapshot.secret_scope_incomplete,
                    staged_terminal=staged,
                ),
            )
            checkpoint = await self._session_store.load_checkpoint(self._session_id)
            stored_stages = tool_round_recovery.checkpoint_staged_terminals(
                checkpoint,
                tool_round_identity=self._tool_round_identity,
            )
            stored = next(
                (item for item in stored_stages if item.tool_call_id == tool_call_id),
                None,
            )
            if stored is None or stored.event.id != event.id:
                raise RuntimeError("Staged terminal acknowledgement conflicts with its event.")
            return self.restore_staged_event_authority(stored.event)

    async def record_projected_terminal(self, event: Event) -> Event:
        """Persist a public projection while retaining its current hook state."""

        return await self._persist_projected_terminal(event, hooks_completed=False)

    async def record_workspace_capture(self, event: Event) -> Event:
        """Persist final workspace-capture controls on an owned terminal stage."""

        return await self._persist_projected_terminal(event, hooks_completed=False)

    async def complete_terminal_hooks(self, event: Event) -> Event:
        """Persist the final hook projection and mark its hooks complete."""

        return await self._persist_projected_terminal(event, hooks_completed=True)

    async def _persist_projected_terminal(
        self,
        event: Event,
        *,
        hooks_completed: bool,
    ) -> Event:
        if event.session_id != self._session_id:
            raise ValueError("Projected terminal belongs to a different session.")
        async with self._lock:
            # Hook execution and tool-result projection already applied the
            # finalized round redactor.  Preparing that event again validates
            # the boundary while preserving the runtime-owned projection
            # authority attached to externalized artifact references.
            projected = prepare_runtime_event(event, redactor=self._redactor)
            await self._session_store.transform_checkpoint(
                self._session_id,
                (
                    tool_round_recovery.completed_staged_terminal_transform
                    if hooks_completed
                    else tool_round_recovery.projected_staged_terminal_transform
                )(
                    tool_round_identity=self._tool_round_identity,
                    event=projected,
                ),
            )
            checkpoint = await self._session_store.load_checkpoint(self._session_id)
            stored_stages = tool_round_recovery.checkpoint_staged_terminals(
                checkpoint,
                tool_round_identity=self._tool_round_identity,
            )
            tool_call_id = projected.payload.get("tool_call_id")
            stored = next(
                (item for item in stored_stages if item.tool_call_id == tool_call_id),
                None,
            )
            if (
                stored is None
                or stored.event.id != projected.id
                or (hooks_completed and stored.hooks_state != "completed")
                or (not hooks_completed and stored.event != projected)
            ):
                raise RuntimeError("Projected terminal acknowledgement conflicts with its stage.")
            return self.restore_staged_event_authority(stored.event)

    def _checkpoint_transform(
        self,
        *,
        tool_call_id: str,
        cover_call: bool,
        staged_terminal: tool_round_recovery.StagedToolCallTerminal | None,
        unsafe_scope: bool = False,
    ) -> CheckpointTransform:
        identity = copy_tool_round_identity(self._tool_round_identity)
        redactor = self._redactor

        def transform(
            _session: Session,
            checkpoint: dict[str, Any] | None,
        ) -> dict[str, Any]:
            updated = (
                tool_round_recovery.checkpoint_with_assistant_publication_snapshot(
                    checkpoint,
                    tool_round_identity=identity,
                    tool_call_id=tool_call_id,
                    redactor=redactor,
                    unsafe_output=unsafe_scope,
                )
                if cover_call
                else tool_round_recovery.checkpoint_with_assistant_publication_redactor(
                    checkpoint,
                    tool_round_identity=identity,
                    tool_call_id=tool_call_id,
                    redactor=redactor,
                )
            )
            existing_stages = tool_round_recovery.checkpoint_staged_terminals(
                updated,
                tool_round_identity=identity,
            )
            projected = [
                item.model_copy(
                    update={
                        "event": _project_staged_terminal_event(
                            item.event,
                            redactor=redactor,
                            trust_persisted_tool_result_authority=True,
                        )
                    },
                    deep=True,
                )
                for item in existing_stages
            ]
            if staged_terminal is not None:
                existing = next(
                    (
                        item
                        for item in projected
                        if item.tool_call_id == staged_terminal.tool_call_id
                    ),
                    None,
                )
                if existing is None:
                    projected.append(staged_terminal)
                elif existing.event.id != staged_terminal.event.id:
                    raise RuntimeError(
                        "Tool call already has conflicting staged terminal evidence."
                    )
                else:
                    projected = [
                        staged_terminal
                        if item.tool_call_id == staged_terminal.tool_call_id
                        else item
                        for item in projected
                    ]
            return tool_round_recovery.checkpoint_with_staged_terminals(
                updated,
                tool_round_identity=identity,
                staged_terminals=projected,
            )

        return transform


class ToolRoundExecutor:
    """Execute tool calls and complete ordinary tool rounds.

    The executor owns policy planning, taint propagation, approval and input
    checkpoints, before/after hooks, reauthorization, tool execution, proxy
    telemetry, concurrency segmentation, and atomic result/checkpoint closure.
    It intentionally receives a narrow ``RuntimeHookRuntime`` rather than the
    complete application facade.
    """

    def __init__(
        self,
        *,
        session_store: SessionStore,
        event_writer: RuntimeEventWriter,
        session_control: SessionControl[SessionUsageTracker],
        hook_runtime: RuntimeHookRuntime,
        runtime_hooks: tuple[runtime_records.RegisteredRuntimeHook, ...],
        mcp_manifest_policy: McpManifestPolicy | None,
        tool_result_projection_policy: ToolResultProjectionPolicy | None,
        secret_redactor: SecretRedactor,
        tool_timeout_seconds: float | None,
        max_parallel_tool_calls: int,
        clock: Callable[[], datetime],
        checkpoint_transform: CheckpointTransformFactory,
        apply_limit_evaluation: LimitEventStream,
        close_interrupted_round: InterruptedRoundEventStream,
    ) -> None:
        self._session_store = session_store
        self._event_writer = event_writer
        self._session_control = session_control
        self._hook_runtime = hook_runtime
        self._runtime_hooks = runtime_hooks
        self._mcp_manifest_policy = mcp_manifest_policy
        self._tool_result_projection_policy = tool_result_projection_policy
        self._secret_redactor = secret_redactor
        self._tool_timeout_seconds = tool_timeout_seconds
        self._max_parallel_tool_calls = max_parallel_tool_calls
        self._clock = clock
        self._checkpoint_transform = checkpoint_transform
        self._apply_limit_evaluation = apply_limit_evaluation
        self._close_interrupted_round = close_interrupted_round
        self._workspace_capture_operations = BoundedInvocationOperationRegistry(
            max_operations=_MAX_RETAINED_WORKSPACE_CAPTURE_OPERATIONS
        )

    def create_run(
        self,
        *,
        session: Session,
        registered_agent: runtime_records.RegisteredAgentState,
        registered_environment: runtime_records.RegisteredEnvironment | None,
        environment_name: str | None,
        limit_gate: RunLimitGate,
        request_metadata: dict[str, Any],
        task_id: str | None,
        structured_output: StructuredOutputSpec | None,
        thinking: ThinkingConfig | None,
        max_steps: int,
        limits: RunLimits,
        budget_limits: tuple[BudgetLimit, ...],
        retry_policy: RetryPolicy,
        run_started_at: float,
        turn_usage_tracker: SessionUsageTracker | None,
        active_run: ActiveSessionRun[SessionUsageTracker] | None,
        execution_profile: ExecutionProfileIdentity | None = None,
        invocation_context: InvocationContext | None = None,
    ) -> ToolRoundRun:
        return ToolRoundRun(
            self,
            session=session,
            registered_agent=registered_agent,
            registered_environment=registered_environment,
            environment_name=environment_name,
            limit_gate=limit_gate,
            request_metadata=request_metadata,
            task_id=task_id,
            structured_output=structured_output,
            thinking=thinking,
            max_steps=max_steps,
            limits=limits,
            budget_limits=budget_limits,
            retry_policy=retry_policy,
            run_started_at=run_started_at,
            turn_usage_tracker=turn_usage_tracker,
            active_run=active_run,
            execution_profile=execution_profile,
            invocation_context=invocation_context,
        )

    def _targeted_tool_use_request(
        self,
        *,
        session: Session,
        registered_agent: runtime_records.RegisteredAgentState,
        registered_environment: runtime_records.RegisteredEnvironment | None,
        interaction_id: str,
        task_id: str | None,
        tool_ref: str,
        tool_id: str,
        tool_name: str,
        descriptor_version: str,
        schema_fingerprint: str,
        model_step_id: str,
        outer_tool_call_id: str,
        arguments_digest: str,
        invocation_id: str,
    ) -> TargetedToolUseRequest:
        return TargetedToolUseRequest(
            tool_ref=tool_ref,
            session_id=session.id,
            interaction_id=interaction_id,
            generation_id=targeted_tool_view_generation_id(
                session_id=session.id,
                root_invocation_id=session.invocation.root_invocation_id,
            ),
            agent_name=registered_agent.spec.name,
            task_id=task_id,
            environment_name=_environment_name(registered_environment),
            principal=session.invocation.origin.subject,
            tenant=session.invocation.origin.tenant,
            catalogue_revision=registered_agent.tool_catalogue.revision,
            descriptor_version=descriptor_version,
            schema_fingerprint=schema_fingerprint,
            tool_id=tool_id,
            tool_name=tool_name,
            model_step_id=model_step_id,
            outer_tool_call_id=outer_tool_call_id,
            arguments_sha256=arguments_digest,
            invocation_id=invocation_id,
            expected_run_epoch=session.run_epoch,
        )

    async def resolve_targeted_tool_calls(
        self,
        *,
        session: Session,
        registered_agent: runtime_records.RegisteredAgentState,
        registered_environment: runtime_records.RegisteredEnvironment | None,
        pending_round: tool_round_recovery.PendingToolRound,
        invocation_context: InvocationContext | None = None,
    ) -> tuple[
        tool_round_recovery.PendingToolRound,
        list[runtime_records.ToolCallRequest],
        tuple[Event, ...],
    ]:
        """Resolve and durably bind dynamic-tool calls before policy evaluation."""

        if invocation_context is not None and (
            invocation_context.binding.session_id != session.id
            or registered_agent is not invocation_context.registered_agent
            or registered_environment is not invocation_context.registered_environment
        ):
            raise RuntimeError("Targeted-tool resolution substituted frozen invocation authority.")

        source_calls = tool_round_recovery.pending_round_tool_calls(pending_round)
        if not any(
            call.name == CALL_TOOL_NAME
            or call.targeted_tool_grant_id is not None
            or call.targeted_tool_invocation is not None
            or call.targeted_tool_rejection is not None
            for call in source_calls
        ):
            return pending_round, source_calls, ()
        interaction_id = pending_round.interaction_id
        if interaction_id is None:
            raise RuntimeError("A pending dynamic-tool round has no interaction identity.")
        if pending_round.policy_state != "unplanned" and any(
            (call.name == CALL_TOOL_NAME or call.targeted_tool_grant_id is not None)
            and call.targeted_tool_invocation is None
            and call.targeted_tool_rejection is None
            for call in source_calls
        ):
            raise RuntimeError("An unresolved dynamic-tool call cannot carry a policy plan.")

        records = (
            ()
            if registered_agent.targeted_tool_mode is None
            else await self._session_store.list_targeted_tool_grants(
                session.id,
                interaction_id=interaction_id,
            )
        )
        records_by_ref = {record.tool_ref: record for record in records}
        records_by_id = {record.grant_id: record for record in records}
        if len(records_by_ref) != len(records) or len(records_by_id) != len(records):
            raise RuntimeError("Targeted grant state contains duplicate identities.")
        resolved_calls: list[runtime_records.ToolCallRequest] = []
        resolution_events: list[Event] = []
        identity = tool_round_recovery.pending_tool_round_identity(pending_round)
        observed_at = self._clock()
        capability_ceiling = tool_capability_ceiling_from_session_metadata(session.metadata)
        ceiling_names = frozenset(capability_ceiling.tool_names)
        discovery_view = None
        if registered_agent.tool_discovery_mode is not None:
            raw_discovery_view = await self._session_store.load_session_operation(
                session.id,
                TOOL_DISCOVERY_VIEW_OPERATION_KEY,
            )
            discovery_view = current_tool_discovery_view(
                raw_discovery_view,
                session_id=session.id,
                generation_id=tool_discovery_generation_id(
                    session_id=session.id,
                    root_invocation_id=session.invocation.root_invocation_id,
                ),
                agent_name=registered_agent.spec.name,
                catalogue=registered_agent.tool_catalogue,
                ceiling=capability_ceiling,
            )
        discovery_records_by_ref = (
            {}
            if discovery_view is None
            else {grant.tool_ref: grant for grant in discovery_view.grants}
        )
        discovery_records_by_id = (
            {}
            if discovery_view is None
            else {grant.grant_id: grant for grant in discovery_view.grants}
        )
        if len(discovery_records_by_id) != len(discovery_records_by_ref):
            raise RuntimeError("Tool discovery view contains duplicate grant identities.")

        async def publish_persisted(event: Event) -> None:
            delivered = await self._event_writer.fan_out_persisted([event])
            resolution_events.extend(delivered)

        async def publish_new(event: Event) -> None:
            resolution_events.append(await self._event_writer.emit(event))

        async def record_for_reference(tool_ref: str):
            record = records_by_ref.get(tool_ref)
            if record is not None:
                return record
            if registered_agent.targeted_tool_mode is None:
                return None
            try:
                parsed = parse_public_authority_alias(tool_ref)
            except (TypeError, ValueError):
                return None
            if parsed is None or parsed.field_name != TARGETED_TOOL_REFERENCE_FIELD_NAME:
                return None
            grant_id = await self._session_store.resolve_public_authority_alias(
                tool_ref,
                field_name=TARGETED_TOOL_REFERENCE_FIELD_NAME,
                scope_session_id=session.id,
            )
            return None if grant_id is None else records_by_id.get(grant_id)

        def invocation_id_for(call: runtime_records.ToolCallRequest) -> str:
            material = (
                f"{session.id}\x00{identity.tool_round_id}\x00{identity.model_step_id}\x00{call.id}"
            ).encode()
            return f"sha256:{hashlib.sha256(material).hexdigest()}"

        def rejected_call(
            call: runtime_records.ToolCallRequest,
            *,
            reason: TargetedToolUseRejectionReason,
            event: Event,
            dispatch_kind: Literal["gateway", "native"] = "gateway",
            model_tool_name: str = CALL_TOOL_NAME,
        ) -> runtime_records.ToolCallRequest:
            return runtime_records.ToolCallRequest(
                id=call.id,
                name=model_tool_name,
                arguments=copy_durable_json_value(call.transcript_arguments, "arguments"),
                targeted_tool_grant_id=call.targeted_tool_grant_id,
                model_tool_name=model_tool_name,
                targeted_tool_rejection=rejected_targeted_tool_invocation(
                    reason=reason,
                    event=event,
                    dispatch_kind=dispatch_kind,
                    model_tool_name=model_tool_name,
                ),
            )

        for call in source_calls:
            if call.targeted_tool_rejection is not None:
                resolved_calls.append(runtime_records.copy_tool_call_request(call))
                continue

            if call.targeted_tool_invocation is not None:
                invocation = call.targeted_tool_invocation
                record = (
                    await record_for_reference(invocation.tool_ref)
                    if invocation.dispatch_kind == "gateway" and invocation.tool_ref is not None
                    else records_by_id.get(invocation.grant_id)
                )
                discovered_record = (
                    discovery_records_by_ref.get(invocation.tool_ref)
                    if invocation.dispatch_kind == "gateway" and invocation.tool_ref is not None
                    else discovery_records_by_id.get(invocation.grant_id)
                )
                if record is None and discovered_record is not None:
                    try:
                        descriptor = registered_agent.tool_catalogue.descriptor_for_id(
                            discovered_record.tool_id
                        )
                    except KeyError:
                        descriptor = None
                    if (
                        discovered_record.grant_id != invocation.grant_id
                        or discovered_record.tool_id != invocation.tool_id
                        or discovered_record.tool_name != invocation.effective_tool_name
                        or discovered_record.catalogue_revision != invocation.catalogue_revision
                        or discovered_record.descriptor_version != invocation.descriptor_version
                        or discovered_record.schema_fingerprint != invocation.schema_fingerprint
                        or descriptor is None
                        or not tool_discovery_record_matches_descriptor(
                            discovered_record,
                            descriptor,
                        )
                        or not _tool_dispatch_authority_is_current(
                            registered_agent,
                            discovered_record.tool_name,
                        )
                        or descriptor.name not in ceiling_names
                        or invocation.session_id != session.id
                        or invocation.interaction_id != interaction_id
                        or invocation.model_step_id != identity.model_step_id
                        or invocation.outer_tool_call_id != call.id
                        or invocation.model_tool_name != call.model_tool_name
                        or (
                            invocation.dispatch_kind == "native"
                            and (
                                invocation.tool_ref is not None
                                or call.model_tool_name != discovered_record.tool_name
                            )
                        )
                        or targeted_arguments_sha256(call.arguments) != invocation.arguments_sha256
                    ):
                        raise RuntimeError(
                            "Resolved discovered-tool checkpoint conflicts with view state."
                        )
                    resolved_calls.append(runtime_records.copy_tool_call_request(call))
                    continue
                if (
                    record is None
                    or record.grant_id != invocation.grant_id
                    or record.tool_id != invocation.tool_id
                    or record.tool_name != invocation.effective_tool_name
                    or record.catalogue_revision != invocation.catalogue_revision
                    or record.descriptor_version != invocation.descriptor_version
                    or record.schema_fingerprint != invocation.schema_fingerprint
                    or invocation.session_id != session.id
                    or invocation.interaction_id != interaction_id
                    or invocation.model_step_id != identity.model_step_id
                    or invocation.outer_tool_call_id != call.id
                    or invocation.model_tool_name != call.model_tool_name
                    or targeted_arguments_sha256(call.arguments) != invocation.arguments_sha256
                ):
                    raise RuntimeError(
                        "Resolved targeted-tool checkpoint conflicts with grant state."
                    )
                if not _tool_dispatch_authority_is_current(
                    registered_agent,
                    invocation.effective_tool_name,
                ):
                    raise RuntimeError(
                        "Resolved targeted-tool checkpoint lost live dispatch authority."
                    )
                request = self._targeted_tool_use_request(
                    session=session,
                    registered_agent=registered_agent,
                    registered_environment=registered_environment,
                    interaction_id=interaction_id,
                    task_id=pending_round.task_id,
                    tool_ref=record.tool_ref,
                    tool_id=record.tool_id,
                    tool_name=record.tool_name,
                    descriptor_version=record.descriptor_version,
                    schema_fingerprint=record.schema_fingerprint,
                    model_step_id=identity.model_step_id,
                    outer_tool_call_id=call.id,
                    arguments_digest=invocation.arguments_sha256,
                    invocation_id=invocation.invocation_id,
                )
                result = await self._session_store.bind_targeted_tool_grant_use(
                    request,
                    observed_at=observed_at,
                )
                if result.event is None:
                    raise RuntimeError("Targeted tool rejoin returned no event evidence.")
                await publish_persisted(result.event)
                if result.disposition is TargetedToolUseDisposition.REJECTED:
                    if result.reason is None:
                        raise RuntimeError("Targeted tool rejoin rejection lost its reason.")
                    resolved_calls.append(
                        rejected_call(
                            call,
                            reason=result.reason,
                            event=result.event,
                            dispatch_kind=invocation.dispatch_kind,
                            model_tool_name=invocation.model_tool_name,
                        )
                    )
                    continue
                if (
                    result.disposition is not TargetedToolUseDisposition.REJOINED
                    or result.binding is None
                    or result.binding.use_id != invocation.use_id
                ):
                    raise RuntimeError(
                        "Resolved targeted-tool checkpoint did not rejoin its binding."
                    )
                resolved_calls.append(runtime_records.copy_tool_call_request(call))
                continue

            if call.name != CALL_TOOL_NAME and call.targeted_tool_grant_id is None:
                resolved_calls.append(runtime_records.copy_tool_call_request(call))
                continue

            invocation_id = invocation_id_for(call)
            dispatch_kind: Literal["gateway", "native"]
            model_tool_name = call.name
            tool_ref: str | None
            effective_arguments: dict[str, Any]
            discovered_record = None
            if call.name == CALL_TOOL_NAME:
                dispatch_kind = "gateway"
                raw_digest = targeted_arguments_sha256(call.arguments)
                try:
                    envelope = CallToolEnvelope.model_validate(call.arguments)
                except (TypeError, ValueError):
                    event = unresolved_gateway_rejection_event(
                        session_id=session.id,
                        interaction_id=interaction_id,
                        agent_name=registered_agent.spec.name,
                        environment_name=_environment_name(registered_environment),
                        model_step_id=identity.model_step_id,
                        outer_tool_call_id=call.id,
                        arguments_digest=raw_digest,
                        reason=TargetedToolUseRejectionReason.MALFORMED,
                        timestamp=observed_at,
                    )
                    await publish_new(event)
                    resolved_calls.append(
                        rejected_call(
                            call,
                            reason=TargetedToolUseRejectionReason.MALFORMED,
                            event=event,
                        )
                    )
                    continue
                tool_ref = envelope.tool_ref
                effective_arguments = envelope.arguments
                selected_grant_id = call.targeted_tool_grant_id
                if selected_grant_id is None:
                    record = await record_for_reference(envelope.tool_ref)
                    if record is None:
                        discovered_record = discovery_records_by_ref.get(envelope.tool_ref)
                else:
                    record = records_by_id.get(selected_grant_id)
                    resolved_grant_id = await self._session_store.resolve_public_authority_alias(
                        envelope.tool_ref,
                        field_name="tool_ref",
                        scope_session_id=session.id,
                    )
                    if record is None or resolved_grant_id != selected_grant_id:
                        raise RuntimeError(
                            "Runtime-selected call_tool reference conflicts with durable grant "
                            "state."
                        )
            else:
                dispatch_kind = "native"
                selected_grant_id = call.targeted_tool_grant_id
                if selected_grant_id is None:  # pragma: no cover - branch condition invariant
                    raise AssertionError("Native dynamic-tool call lost its grant selection.")
                record = records_by_id.get(selected_grant_id)
                if record is None:
                    discovered_record = discovery_records_by_id.get(selected_grant_id)
                if record is not None and record.tool_name != call.name:
                    raise RuntimeError(
                        "Runtime-selected native tool conflicts with durable grant state."
                    )
                if discovered_record is not None and discovered_record.tool_name != call.name:
                    raise RuntimeError(
                        "Runtime-selected native discovery tool conflicts with durable view state."
                    )
                tool_ref = None
                effective_arguments = copy_durable_json_value(call.arguments, "arguments")
            if discovered_record is not None:
                try:
                    descriptor = registered_agent.tool_catalogue.descriptor_for_id(
                        discovered_record.tool_id
                    )
                except KeyError:
                    descriptor = None
                rejection_reason = None
                if (
                    descriptor is None
                    or discovered_record.catalogue_revision
                    != registered_agent.tool_catalogue.revision
                    or not tool_discovery_record_matches_descriptor(
                        discovered_record,
                        descriptor,
                    )
                    or not _tool_dispatch_authority_is_current(
                        registered_agent,
                        discovered_record.tool_name,
                    )
                    or discovered_record.tool_name not in ceiling_names
                ):
                    rejection_reason = TargetedToolUseRejectionReason.CATALOGUE_DRIFT
                elif not validate_effective_tool_arguments(
                    effective_arguments,
                    descriptor.input_schema_copy(),
                ):
                    rejection_reason = TargetedToolUseRejectionReason.INVALID_ARGUMENTS
                arguments_digest = targeted_arguments_sha256(effective_arguments)
                if rejection_reason is not None:
                    event = discovered_tool_rejection_event(
                        session_id=session.id,
                        interaction_id=interaction_id,
                        agent_name=registered_agent.spec.name,
                        environment_name=_environment_name(registered_environment),
                        model_step_id=identity.model_step_id,
                        outer_tool_call_id=call.id,
                        arguments_sha256=arguments_digest,
                        reason=rejection_reason,
                        timestamp=observed_at,
                    )
                    await publish_new(event)
                    resolved_calls.append(
                        rejected_call(
                            call,
                            reason=rejection_reason,
                            event=event,
                            dispatch_kind=dispatch_kind,
                            model_tool_name=model_tool_name,
                        )
                    )
                    continue
                resolved_calls.append(
                    runtime_records.ToolCallRequest(
                        id=call.id,
                        name=discovered_record.tool_name,
                        arguments=copy_durable_json_value(effective_arguments, "arguments"),
                        model_tool_name=model_tool_name,
                        targeted_tool_invocation=resolved_discovered_tool_invocation(
                            record=discovered_record,
                            session_id=session.id,
                            interaction_id=interaction_id,
                            model_step_id=identity.model_step_id,
                            outer_tool_call_id=call.id,
                            arguments_sha256=arguments_digest,
                            invocation_id=invocation_id,
                            dispatch_kind=dispatch_kind,
                            model_tool_name=model_tool_name,
                        ),
                    )
                )
                continue
            if record is None:
                if dispatch_kind != "gateway" or tool_ref is None:
                    raise RuntimeError("Native dynamic-tool projection lost its durable grant.")
                if registered_agent.tool_discovery_mode is not None and (
                    registered_agent.targeted_tool_mode is None
                    or tool_ref.startswith(TOOL_DISCOVERY_REFERENCE_PREFIX)
                ):
                    arguments_digest = targeted_arguments_sha256(effective_arguments)
                    rejection_reason = tool_discovery_reference_rejection_reason(tool_ref)
                    event = discovered_tool_rejection_event(
                        session_id=session.id,
                        interaction_id=interaction_id,
                        agent_name=registered_agent.spec.name,
                        environment_name=_environment_name(registered_environment),
                        model_step_id=identity.model_step_id,
                        outer_tool_call_id=call.id,
                        arguments_sha256=arguments_digest,
                        reason=rejection_reason,
                        timestamp=observed_at,
                    )
                    await publish_new(event)
                    resolved_calls.append(
                        rejected_call(
                            call,
                            reason=rejection_reason,
                            event=event,
                        )
                    )
                    continue
                request = self._targeted_tool_use_request(
                    session=session,
                    registered_agent=registered_agent,
                    registered_environment=registered_environment,
                    interaction_id=interaction_id,
                    task_id=pending_round.task_id,
                    tool_ref=tool_ref,
                    tool_id="cayu:unresolved-targeted-reference",
                    tool_name=CALL_TOOL_NAME,
                    descriptor_version=f"sha256:{'0' * 64}",
                    schema_fingerprint=f"sha256:{'0' * 64}",
                    model_step_id=identity.model_step_id,
                    outer_tool_call_id=call.id,
                    arguments_digest=targeted_arguments_sha256(effective_arguments),
                    invocation_id=invocation_id,
                )
                result = await self._session_store.bind_targeted_tool_grant_use(
                    request,
                    observed_at=observed_at,
                )
                if (
                    result.disposition is not TargetedToolUseDisposition.REJECTED
                    or result.reason is None
                    or result.event is None
                ):
                    raise RuntimeError("Unknown call_tool reference unexpectedly resolved.")
                await publish_persisted(result.event)
                resolved_calls.append(rejected_call(call, reason=result.reason, event=result.event))
                continue

            try:
                descriptor = registered_agent.tool_catalogue.descriptor_for_id(record.tool_id)
            except KeyError:
                descriptor = None
            arguments_digest = targeted_arguments_sha256(effective_arguments)
            request = self._targeted_tool_use_request(
                session=session,
                registered_agent=registered_agent,
                registered_environment=registered_environment,
                interaction_id=interaction_id,
                task_id=pending_round.task_id,
                tool_ref=record.tool_ref,
                tool_id=(record.tool_id if descriptor is None else descriptor.tool_id),
                tool_name=(record.tool_name if descriptor is None else descriptor.name),
                descriptor_version=(
                    record.descriptor_version if descriptor is None else descriptor.version
                ),
                schema_fingerprint=(
                    record.schema_fingerprint
                    if descriptor is None
                    else descriptor.schema_fingerprint
                ),
                model_step_id=identity.model_step_id,
                outer_tool_call_id=call.id,
                arguments_digest=arguments_digest,
                invocation_id=invocation_id,
            )
            preflight_reason = targeted_tool_use_rejection_reason(
                record,
                request,
                observed_at=observed_at,
            )
            explicit_rejection = None
            if record.tool_name not in ceiling_names:
                explicit_rejection = TargetedToolUseRejectionReason.OUT_OF_CEILING
            elif descriptor is None or not _tool_dispatch_authority_is_current(
                registered_agent,
                descriptor.name,
            ):
                explicit_rejection = TargetedToolUseRejectionReason.CATALOGUE_DRIFT
            if explicit_rejection is not None:
                event = targeted_tool_use_rejection_event(
                    record,
                    request,
                    reason=explicit_rejection,
                    timestamp=observed_at,
                )
                await publish_new(event)
                resolved_calls.append(
                    rejected_call(
                        call,
                        reason=explicit_rejection,
                        event=event,
                        dispatch_kind=dispatch_kind,
                        model_tool_name=model_tool_name,
                    )
                )
                continue
            if preflight_reason is not None:
                result = await self._session_store.bind_targeted_tool_grant_use(
                    request,
                    observed_at=observed_at,
                )
                if (
                    result.disposition is not TargetedToolUseDisposition.REJECTED
                    or result.reason is None
                    or result.event is None
                ):
                    raise RuntimeError("Rejected targeted-tool preflight unexpectedly bound.")
                await publish_persisted(result.event)
                resolved_calls.append(
                    rejected_call(
                        call,
                        reason=result.reason,
                        event=result.event,
                        dispatch_kind=dispatch_kind,
                        model_tool_name=model_tool_name,
                    )
                )
                continue
            if descriptor is None:  # pragma: no cover - explicit rejection above
                raise AssertionError("Callable targeted grant lost its registered descriptor.")
            if not validate_effective_tool_arguments(
                effective_arguments,
                descriptor.input_schema_copy(),
            ):
                event = targeted_tool_use_rejection_event(
                    record,
                    request,
                    reason=TargetedToolUseRejectionReason.INVALID_ARGUMENTS,
                    timestamp=observed_at,
                )
                await publish_new(event)
                resolved_calls.append(
                    rejected_call(
                        call,
                        reason=TargetedToolUseRejectionReason.INVALID_ARGUMENTS,
                        event=event,
                        dispatch_kind=dispatch_kind,
                        model_tool_name=model_tool_name,
                    )
                )
                continue

            result = await self._session_store.bind_targeted_tool_grant_use(
                request,
                observed_at=observed_at,
            )
            if result.event is None:
                raise RuntimeError("Targeted tool binding returned no event evidence.")
            await publish_persisted(result.event)
            if result.disposition is TargetedToolUseDisposition.REJECTED:
                if result.reason is None:
                    raise RuntimeError("Targeted tool binding rejection lost its reason.")
                resolved_calls.append(
                    rejected_call(
                        call,
                        reason=result.reason,
                        event=result.event,
                        dispatch_kind=dispatch_kind,
                        model_tool_name=model_tool_name,
                    )
                )
                continue
            if result.grant is None or result.binding is None:
                raise RuntimeError("Targeted tool binding returned incomplete authority.")
            resolved_calls.append(
                runtime_records.ToolCallRequest(
                    id=call.id,
                    name=result.grant.tool_name,
                    arguments=copy_durable_json_value(effective_arguments, "arguments"),
                    targeted_tool_grant_id=call.targeted_tool_grant_id,
                    model_tool_name=model_tool_name,
                    targeted_tool_invocation=resolved_targeted_tool_invocation(
                        record=result.grant,
                        binding=result.binding,
                        tool_ref=tool_ref,
                        dispatch_kind=dispatch_kind,
                        model_tool_name=model_tool_name,
                    ),
                )
            )

        if resolved_calls == source_calls:
            return pending_round, resolved_calls, tuple(resolution_events)
        redactor = _redactor_for_tool_calls(
            self._secret_redactor,
            registered_agent=registered_agent,
            tool_calls=resolved_calls,
        )
        payload = pending_round.model_dump(mode="json")
        payload["tool_calls"] = [
            record.model_dump(mode="json")
            for record in approval_support.pending_tool_call_approvals(
                tool_calls=resolved_calls,
                policy_outcomes=None,
                default_policy_evidence=ToolPolicyEvidence.UNPLANNED,
                redactor=redactor,
            )
        ]
        resolved_round = tool_round_recovery.PendingToolRound.model_validate(payload)
        source_payload = pending_round.model_dump(mode="json")
        resolved_payload = _require_secret_free_durable_object(
            resolved_round.model_dump(mode="json"),
            redactor=redactor,
            field_name="pending_tool_round",
            schema_root=tool_round_recovery.PENDING_TOOL_ROUND_CHECKPOINT_KEY,
        )

        def publish_resolution(
            current_session: Session,
            current_checkpoint: dict[str, Any] | None,
        ) -> dict[str, Any]:
            if current_session.run_epoch != session.run_epoch:
                raise RuntimeError("Targeted-tool resolution lost its run fence.")
            current = (
                {}
                if current_checkpoint is None
                else copy_durable_json_object(current_checkpoint, "checkpoint")
            )
            if current.get(tool_round_recovery.PENDING_TOOL_ROUND_CHECKPOINT_KEY) != source_payload:
                raise RuntimeError("Pending tool round changed before targeted-tool resolution.")
            current[tool_round_recovery.PENDING_TOOL_ROUND_CHECKPOINT_KEY] = resolved_payload
            return copy_durable_json_object(current, "checkpoint")

        await self._session_store.transform_checkpoint(session.id, publish_resolution)
        return resolved_round, resolved_calls, tuple(resolution_events)

    async def rejoin_targeted_tool_call(
        self,
        *,
        session: Session,
        registered_agent: runtime_records.RegisteredAgentState,
        registered_environment: runtime_records.RegisteredEnvironment | None,
        tool_call: runtime_records.ToolCallRequest,
        task_id: str | None,
        invocation_context: InvocationContext | None = None,
    ) -> tuple[Event, ...]:
        """Rejoin a previously bound targeted use before paused execution resumes."""

        if invocation_context is not None and (
            invocation_context.binding.session_id != session.id
            or invocation_context.registered_agent is not registered_agent
            or invocation_context.registered_environment is not registered_environment
        ):
            raise RuntimeError("Targeted-tool rejoin lost frozen invocation authority.")

        invocation = tool_call.targeted_tool_invocation
        if invocation is None:
            return ()
        if (
            invocation.session_id != session.id
            or invocation.effective_tool_name != tool_call.name
            or invocation.model_tool_name != tool_call.model_tool_name
            or invocation.outer_tool_call_id != tool_call.id
            or targeted_arguments_sha256(tool_call.arguments) != invocation.arguments_sha256
        ):
            raise RuntimeError("Paused targeted invocation conflicts with its binding.")
        if not _tool_dispatch_authority_is_current(
            registered_agent,
            invocation.effective_tool_name,
        ):
            raise RuntimeError("Paused targeted invocation lost live dispatch authority.")
        tool_ref = invocation.tool_ref
        if registered_agent.tool_discovery_mode is not None:
            ceiling = tool_capability_ceiling_from_session_metadata(session.metadata)
            discovery_view = current_tool_discovery_view(
                await self._session_store.load_session_operation(
                    session.id,
                    TOOL_DISCOVERY_VIEW_OPERATION_KEY,
                ),
                session_id=session.id,
                generation_id=tool_discovery_generation_id(
                    session_id=session.id,
                    root_invocation_id=session.invocation.root_invocation_id,
                ),
                agent_name=registered_agent.spec.name,
                catalogue=registered_agent.tool_catalogue,
                ceiling=ceiling,
            )
            discovery_record = (
                discovery_view.record_for_reference(tool_ref)
                if tool_ref is not None
                else next(
                    (
                        record
                        for record in discovery_view.grants
                        if record.grant_id == invocation.grant_id
                    ),
                    None,
                )
            )
            if discovery_record is not None:
                try:
                    descriptor = registered_agent.tool_catalogue.descriptor_for_id(
                        discovery_record.tool_id
                    )
                except KeyError:
                    descriptor = None
                if (
                    discovery_record.grant_id != invocation.grant_id
                    or discovery_record.tool_id != invocation.tool_id
                    or discovery_record.tool_name != invocation.effective_tool_name
                    or discovery_record.catalogue_revision != invocation.catalogue_revision
                    or discovery_record.descriptor_version != invocation.descriptor_version
                    or discovery_record.schema_fingerprint != invocation.schema_fingerprint
                    or descriptor is None
                    or not tool_discovery_record_matches_descriptor(
                        discovery_record,
                        descriptor,
                    )
                    or descriptor.name not in ceiling.tool_names
                    or (
                        invocation.dispatch_kind == "gateway"
                        and tool_ref != discovery_record.tool_ref
                    )
                    or (
                        invocation.dispatch_kind == "native"
                        and (
                            tool_ref is not None
                            or invocation.model_tool_name != discovery_record.tool_name
                        )
                    )
                ):
                    raise RuntimeError(
                        "Paused discovered-tool invocation conflicts with its durable view."
                    )
                return ()
            if (
                tool_ref is not None and tool_ref.startswith(TOOL_DISCOVERY_REFERENCE_PREFIX)
            ) or registered_agent.targeted_tool_mode is None:
                raise RuntimeError("Paused discovered-tool invocation lost its durable view grant.")
        if tool_ref is None:
            records = await self._session_store.list_targeted_tool_grants(
                session.id,
                interaction_id=invocation.interaction_id,
            )
            matching = [record for record in records if record.grant_id == invocation.grant_id]
            if len(matching) != 1:
                raise RuntimeError("Paused native invocation lost its durable grant.")
            tool_ref = matching[0].tool_ref
        request = self._targeted_tool_use_request(
            session=session,
            registered_agent=registered_agent,
            registered_environment=registered_environment,
            interaction_id=invocation.interaction_id,
            task_id=task_id,
            tool_ref=tool_ref,
            tool_id=invocation.tool_id,
            tool_name=invocation.effective_tool_name,
            descriptor_version=invocation.descriptor_version,
            schema_fingerprint=invocation.schema_fingerprint,
            model_step_id=invocation.model_step_id,
            outer_tool_call_id=invocation.outer_tool_call_id,
            arguments_digest=invocation.arguments_sha256,
            invocation_id=invocation.invocation_id,
        )
        result = await self._session_store.bind_targeted_tool_grant_use(
            request,
            observed_at=self._clock(),
        )
        if (
            result.disposition is not TargetedToolUseDisposition.REJOINED
            or result.binding is None
            or result.binding.use_id != invocation.use_id
            or result.event is None
        ):
            raise RuntimeError("Paused targeted invocation did not rejoin its binding.")
        delivered = await self._event_writer.fan_out_persisted([result.event])
        return tuple(delivered)

    async def policy_plan(
        self,
        *,
        session: Session,
        registered_agent: runtime_records.RegisteredAgentState,
        registered_environment: runtime_records.RegisteredEnvironment | None,
        tool_calls: list[runtime_records.ToolCallRequest],
        request_metadata: dict[str, Any],
        tool_exposure: ResolvedToolExposureAuthority | None = None,
        invocation_context: InvocationContext | None = None,
    ) -> runtime_records.ToolRoundPolicyPlan:
        if invocation_context is not None and (
            invocation_context.binding.session_id != session.id
            or registered_agent is not invocation_context.registered_agent
            or registered_environment is not invocation_context.registered_environment
        ):
            raise RuntimeError("Tool policy substituted frozen invocation authority.")
        policy_outcomes: list[runtime_records.ToolCallPolicyOutcome] = []
        approval_policy_result: ToolPolicyResult | None = None
        approval_tool_call: runtime_records.ToolCallRequest | None = None
        executable_names = registered_agent.executable_tool_names
        exposed_names = (
            executable_names
            if tool_exposure is None
            else frozenset((*tool_exposure.tool_names, *registered_agent.runtime_tools))
        )
        has_authorizable_call = any(
            call.name in executable_names
            and (call.name in exposed_names or call.targeted_tool_invocation is not None)
            and _tool_dispatch_authority_is_current(registered_agent, call.name)
            for call in tool_calls
        )
        taint_labels = (
            await self.prior_taint_labels_for_policy(
                session_id=session.id,
                policy=registered_agent.tool_policy,
                request_metadata=request_metadata,
            )
            if has_authorizable_call
            else set()
        )
        active_taint_labels: dict[str, frozenset[str]] = {}
        for tool_call in tool_calls:
            active_taint_labels[tool_call.id] = frozenset(taint_labels)
            if tool_call.name not in executable_names or not _tool_dispatch_authority_is_current(
                registered_agent,
                tool_call.name,
            ):
                policy_outcomes.append(
                    runtime_records.ToolCallPolicyOutcome(
                        call=tool_call,
                        result=None,
                        evidence=ToolPolicyEvidence.UNREGISTERED,
                    )
                )
                continue
            if tool_call.name not in exposed_names and tool_call.targeted_tool_invocation is None:
                policy_outcomes.append(
                    runtime_records.ToolCallPolicyOutcome(
                        call=tool_call,
                        result=None,
                        evidence=ToolPolicyEvidence.UNEXPOSED,
                    )
                )
                continue

            policy_result = await self.authorize_tool_call(
                session=session,
                registered_agent=registered_agent,
                registered_environment=registered_environment,
                tool_call=tool_call,
                request_metadata=request_metadata,
                taint_labels=taint_labels,
            )
            policy_outcomes.append(
                runtime_records.ToolCallPolicyOutcome(
                    call=tool_call,
                    result=policy_result,
                    evidence=ToolPolicyEvidence.AUTHORITATIVE,
                )
            )
            if (
                approval_policy_result is None
                and policy_result.decision == ToolPolicyDecision.REQUIRE_APPROVAL
            ):
                approval_policy_result = policy_result
                approval_tool_call = tool_call
            taint_labels.update(
                _taint_labels_for_source_tool(
                    registered_agent.tool_policy,
                    tool_call.name,
                    policy_result=policy_result,
                )
            )

        if approval_policy_result is None or approval_tool_call is None:
            return runtime_records.ToolRoundPolicyPlan(
                outcomes=policy_outcomes,
                pending_approval=None,
                active_taint_labels=active_taint_labels,
            )
        return runtime_records.ToolRoundPolicyPlan(
            outcomes=policy_outcomes,
            active_taint_labels=active_taint_labels,
            pending_approval=runtime_records.PendingToolApprovalPlan(
                call=approval_tool_call,
                calls=[outcome.call for outcome in policy_outcomes],
                policy_outcomes=policy_outcomes,
                policy_result=approval_policy_result,
            ),
        )

    async def fail_closed_recovery_policy_plan(
        self,
        *,
        session: Session,
        registered_agent: runtime_records.RegisteredAgentState,
        tool_calls: list[runtime_records.ToolCallRequest],
        request_metadata: dict[str, Any],
        durable_tool_calls: list[PendingToolCallApproval] | None = None,
        tool_exposure: ResolvedToolExposureAuthority | None = None,
    ) -> runtime_records.ToolRoundPolicyPlan:
        """Build a manual gate without replaying an outcome-ambiguous policy.

        A process can stop after ``authorize()`` returns but before its result is
        durably published. Re-running a stateful or time-sensitive policy cannot
        prove what that earlier invocation decided. Ambiguous registered calls
        therefore carry explicit non-authoritative evidence and no policy
        decision. The round still pauses for operator acknowledgement, but that
        acknowledgement can only close the ambiguous calls as blocked; it can
        never authorize dispatch. Recognized durable legacy outcomes remain
        authoritative.
        """

        if durable_tool_calls is not None:
            if [
                (call.tool_call_id, call.tool_name, call.arguments) for call in durable_tool_calls
            ] != [(call.id, call.name, call.arguments) for call in tool_calls]:
                raise RuntimeError(
                    "Durable recovery policy calls do not match the pending tool round."
                )
            durable_by_id = {call.tool_call_id: call for call in durable_tool_calls}
        else:
            durable_by_id = {}
        recognized_decisions = {decision.value for decision in ToolPolicyDecision}
        policy_outcomes: list[runtime_records.ToolCallPolicyOutcome] = []
        authoritative_approval_tool_call: runtime_records.ToolCallRequest | None = None
        ambiguous_tool_call: runtime_records.ToolCallRequest | None = None
        executable_names = registered_agent.executable_tool_names
        exposed_names = (
            executable_names
            if tool_exposure is None
            else frozenset((*tool_exposure.tool_names, *registered_agent.runtime_tools))
        )
        has_authorizable_call = any(
            call.name in executable_names
            and (call.name in exposed_names or call.targeted_tool_invocation is not None)
            and _tool_dispatch_authority_is_current(registered_agent, call.name)
            for call in tool_calls
        )
        taint_labels = (
            await self.prior_taint_labels_for_policy(
                session_id=session.id,
                policy=registered_agent.tool_policy,
                request_metadata=request_metadata,
            )
            if has_authorizable_call
            else set()
        )
        active_taint_labels: dict[str, frozenset[str]] = {}
        for tool_call in tool_calls:
            active_taint_labels[tool_call.id] = frozenset(taint_labels)
            if tool_call.name not in executable_names or not _tool_dispatch_authority_is_current(
                registered_agent,
                tool_call.name,
            ):
                policy_outcomes.append(
                    runtime_records.ToolCallPolicyOutcome(
                        call=tool_call,
                        result=None,
                        evidence=ToolPolicyEvidence.UNREGISTERED,
                    )
                )
                continue
            if tool_call.name not in exposed_names and tool_call.targeted_tool_invocation is None:
                policy_outcomes.append(
                    runtime_records.ToolCallPolicyOutcome(
                        call=tool_call,
                        result=None,
                        evidence=ToolPolicyEvidence.UNEXPOSED,
                    )
                )
                continue
            durable_call = durable_by_id.get(tool_call.id)
            policy_result: ToolPolicyResult | None
            policy_evidence: ToolPolicyEvidence
            if durable_call is not None and durable_call.policy_decision in recognized_decisions:
                restored = approval_support.policy_result_from_pending_tool_call(durable_call)
                if restored is None:
                    raise AssertionError("Recognized durable policy decision was not restored.")
                policy_result = restored
                policy_evidence = ToolPolicyEvidence.AUTHORITATIVE
                active_taint_labels[tool_call.id] = frozenset(durable_call.active_taint_labels)
            else:
                policy_result = None
                policy_evidence = ToolPolicyEvidence.AMBIGUOUS
            policy_outcomes.append(
                runtime_records.ToolCallPolicyOutcome(
                    call=tool_call,
                    result=policy_result,
                    evidence=policy_evidence,
                )
            )
            if (
                authoritative_approval_tool_call is None
                and policy_result is not None
                and policy_result.decision == ToolPolicyDecision.REQUIRE_APPROVAL
            ):
                authoritative_approval_tool_call = tool_call
            if ambiguous_tool_call is None and policy_evidence is ToolPolicyEvidence.AMBIGUOUS:
                ambiguous_tool_call = tool_call
            if policy_result is not None:
                taint_labels.update(
                    _taint_labels_for_source_tool(
                        registered_agent.tool_policy,
                        tool_call.name,
                        policy_result=policy_result,
                    )
                )

        # A durable REQUIRE_APPROVAL decision must own the visible gate even
        # when an earlier sibling is ambiguous. Otherwise a "continue safely"
        # acknowledgement for the ambiguous call could also execute the
        # authoritatively approval-gated sibling without presenting that
        # approval as the operator's decision.
        approval_tool_call = authoritative_approval_tool_call or ambiguous_tool_call
        if approval_tool_call is None:
            return runtime_records.ToolRoundPolicyPlan(
                outcomes=policy_outcomes,
                pending_approval=None,
                active_taint_labels=active_taint_labels,
            )
        approval_outcome = next(
            outcome for outcome in policy_outcomes if outcome.call.id == approval_tool_call.id
        )
        approval_result = approval_outcome.result
        if approval_outcome.evidence is ToolPolicyEvidence.AMBIGUOUS:
            approval_result = ToolPolicyResult(
                decision=ToolPolicyDecision.REQUIRE_APPROVAL,
                reason=_AMBIGUOUS_POLICY_RECOVERY_REASON,
                metadata=_AMBIGUOUS_POLICY_RECOVERY_METADATA,
            )
        if approval_result is None:
            raise AssertionError("Recovery approval lost its display policy result.")
        return runtime_records.ToolRoundPolicyPlan(
            outcomes=policy_outcomes,
            active_taint_labels=active_taint_labels,
            pending_approval=runtime_records.PendingToolApprovalPlan(
                call=approval_tool_call,
                calls=[outcome.call for outcome in policy_outcomes],
                policy_outcomes=policy_outcomes,
                policy_result=approval_result,
            ),
        )

    async def authorize_tool_call(
        self,
        *,
        session: Session,
        registered_agent: runtime_records.RegisteredAgentState,
        registered_environment: runtime_records.RegisteredEnvironment | None,
        tool_call: runtime_records.ToolCallRequest,
        request_metadata: dict[str, Any],
        taint_labels: Iterable[str] | None = None,
    ) -> ToolPolicyResult:
        policy_metadata = request_metadata
        if taint_labels:
            policy_metadata = metadata_with_taint_labels(request_metadata, taint_labels)
        policy_result = await registered_agent.tool_policy.authorize(
            ToolPolicyRequest(
                session=session.model_copy(deep=True),
                agent=_copy_agent_spec(registered_agent.spec),
                tool_name=tool_call.name,
                tool_call_id=tool_call.id,
                tool_effect=_tool_effect(registered_agent, tool_call),
                arguments=tool_call.arguments,
                environment_name=_environment_name(registered_environment),
                workspace_id=_workspace_id(registered_environment),
                metadata=policy_metadata,
            )
        )
        return tool_execution.validate_tool_policy_result(policy_result)

    async def prior_taint_labels_for_policy(
        self,
        *,
        session_id: str,
        policy: ToolPolicy,
        request_metadata: dict[str, Any],
    ) -> set[str]:
        labels: set[str] = set(taint_labels_from_metadata(request_metadata))
        session = await self._session_store.load(session_id)
        if session is not None:
            labels.update(taint_labels_from_metadata(session.metadata))
        if not isinstance(policy, TaintAwareToolPolicy):
            return labels
        for event_type in (EventType.TOOL_CALL_COMPLETED, EventType.TOOL_CALL_FAILED):
            records = await query_all_event_records(
                self._session_store,
                EventQuery(
                    session_id=session_id,
                    event_type=event_type,
                    limit=5000,
                ),
            )
            for record in records:
                if record.event.tool_name is not None:
                    labels.update(policy.labels_for_source_tool(record.event.tool_name))
        return labels

    async def checkpoint_pending_tool_approval(
        self,
        *,
        session: Session,
        registered_agent: runtime_records.RegisteredAgentState,
        registered_environment: runtime_records.RegisteredEnvironment | None,
        tool_call: runtime_records.ToolCallRequest,
        tool_calls: list[runtime_records.ToolCallRequest],
        policy_outcomes: list[runtime_records.ToolCallPolicyOutcome] | None,
        active_taint_by_id: Mapping[str, frozenset[str]],
        task_id: str | None,
        policy_result: ToolPolicyResult,
        structured_output: StructuredOutputSpec | None,
        thinking: ThinkingConfig | None,
        max_steps: int | None,
        limits: RunLimits | None,
        budget_limits: tuple[BudgetLimit, ...] | None,
        retry_policy: RetryPolicy | None,
        tool_round_identity: ToolRoundIdentity,
        deferred_messages: list[Message] | None = None,
        recovered: bool = False,
    ) -> tuple[PendingToolApproval, list[Event]]:
        tool_round_identity = copy_tool_round_identity(tool_round_identity)
        redactor = _redactor_for_tool_calls(
            self._secret_redactor,
            registered_agent=registered_agent,
            tool_calls=tool_calls,
        )
        checkpoint = await self._session_store.load_checkpoint(session.id)
        checkpoint = {} if checkpoint is None else copy_json_value(checkpoint, "checkpoint")
        pending_round = tool_round_recovery.pending_tool_round_from_checkpoint(
            checkpoint,
            redactor=self._secret_redactor,
            consume_on_rejection=True,
        )
        if pending_round is None:
            raise RuntimeError("Session has no pending tool round for its approval.")
        _require_matching_policy_round(
            pending_round=pending_round,
            tool_round_identity=tool_round_identity,
            tool_calls=tool_calls,
        )
        if (
            approval_support.pending_approval_from_checkpoint(
                checkpoint,
                redactor=self._secret_redactor,
                consume_on_rejection=True,
            )
            is not None
        ):
            raise RuntimeError("Session already has a pending tool approval.")

        round_ttls = [policy_result.approval_expires_in_seconds]
        if policy_outcomes is not None:
            round_ttls.extend(
                outcome.result.approval_expires_in_seconds
                for outcome in policy_outcomes
                if outcome.result is not None
                and outcome.result.decision == ToolPolicyDecision.REQUIRE_APPROVAL
            )
        bounded_ttls = [ttl for ttl in round_ttls if ttl is not None]
        expires_at: datetime | None = None
        if bounded_ttls:
            expires_at = self._clock() + timedelta(seconds=min(bounded_ttls))
        approval = PendingToolApproval(
            approval_id=str(uuid4()),
            **tool_round_identity.payload(),
            tool_call_id=tool_call.id,
            tool_name=tool_call.name,
            arguments=copy_json_value(tool_call.arguments, "arguments"),
            agent_name=registered_agent.spec.name,
            environment_name=_environment_name(registered_environment),
            workspace_id=_workspace_id(registered_environment),
            task_id=task_id,
            execution_profile_fingerprint=pending_round.execution_profile_fingerprint,
            publish_arguments=_tool_round_publishes_arguments(
                registered_agent,
                tool_calls,
            ),
            secret_resolution_scope=(
                approval_support.tool_round_secret_resolution_scope(pending_round)
            ),
            reason=(
                None if policy_result.reason is None else redactor.redact_text(policy_result.reason)
            ),
            metadata=copy_json_value(
                redactor.redact_json_values(policy_result.metadata),
                "metadata",
            ),
            tool_calls=approval_support.pending_tool_call_approvals(
                tool_calls=tool_calls,
                policy_outcomes=policy_outcomes,
                active_taint_by_id=active_taint_by_id,
                redactor=redactor,
            ),
            structured_output=copy_structured_output_spec(structured_output),
            thinking=thinking,
            max_steps=max_steps,
            limits=copy_run_limits(limits) if limits is not None else None,
            run_limit_accounting=pending_round.run_limit_accounting,
            budget_limits=(
                copy_request_budget_limits(budget_limits) if budget_limits is not None else None
            ),
            retry_policy=copy_retry_policy(retry_policy) if retry_policy is not None else None,
            expires_at=expires_at,
        )
        approval_payload = _require_secret_free_durable_object(
            approval.model_dump(mode="json"),
            redactor=redactor,
            field_name="pending_tool_approval",
            schema_root=approval_support.PENDING_TOOL_APPROVAL_CHECKPOINT_KEY,
        )
        planned_round = _planned_pending_tool_round(
            pending_round=pending_round,
            tool_calls=tool_calls,
            policy_outcomes=policy_outcomes,
            active_taint_by_id=active_taint_by_id,
            redactor=redactor,
            deferred_messages=deferred_messages,
        )
        planned_round_payload = _require_secret_free_durable_object(
            planned_round.model_dump(mode="json"),
            redactor=redactor,
            field_name="pending_tool_round",
            schema_root=tool_round_recovery.PENDING_TOOL_ROUND_CHECKPOINT_KEY,
        )
        # Retain the exact serialized source for the compare-and-swap. Parsing
        # an older checkpoint fills model defaults (including policy version
        # fields), which is useful for behavior but must not change the value
        # we require the atomic publication to replace.
        source_round_payload = copy_json_value(
            checkpoint[tool_round_recovery.PENDING_TOOL_ROUND_CHECKPOINT_KEY],
            "pending_tool_round",
        )
        # Recovery can own a fenced session in a non-running status (for
        # example an incomplete-session claim over INTERRUPTED). Bind the
        # publication to the exact claimed status and epoch rather than
        # broadening recovery to a fixed set of statuses.
        eligible_statuses = {session.status} if recovered else {SessionStatus.RUNNING}

        def publish_policy_and_approval(
            current_session: Session,
            current_checkpoint: dict[str, Any] | None,
        ) -> dict[str, Any]:
            if (
                current_session.status not in eligible_statuses
                or current_session.run_epoch != session.run_epoch
            ):
                raise RuntimeError("Tool approval publication lost its run fence.")
            current = (
                {}
                if current_checkpoint is None
                else copy_json_value(current_checkpoint, "checkpoint")
            )
            if (
                current.get(tool_round_recovery.PENDING_TOOL_ROUND_CHECKPOINT_KEY)
                != source_round_payload
            ):
                raise RuntimeError("Pending tool round changed before approval publication.")
            if approval_support.PENDING_TOOL_APPROVAL_CHECKPOINT_KEY in current:
                raise RuntimeError("Session already has a pending tool approval.")
            if approval_support.APPROVAL_RESOLUTION_INTENT_CHECKPOINT_KEY in current:
                raise RuntimeError("Session has an orphaned approval resolution intent.")
            current[tool_round_recovery.PENDING_TOOL_ROUND_CHECKPOINT_KEY] = planned_round_payload
            current[approval_support.PENDING_TOOL_APPROVAL_CHECKPOINT_KEY] = approval_payload
            if recovered and _PENDING_SESSION_INTERRUPT_CHECKPOINT_KEY in current:
                interrupt_payload = current[_PENDING_SESSION_INTERRUPT_CHECKPOINT_KEY]
                if type(interrupt_payload) is not dict:
                    raise RuntimeError(
                        "Recovered approval has an invalid pending interruption payload."
                    )
                interrupt_payload = copy_json_value(
                    interrupt_payload,
                    "pending_session_interrupt",
                )
                interrupt_payload.update(
                    {
                        "interruption_type": (_INTERRUPTION_TYPE_TOOL_APPROVAL_REQUIRED),
                        **tool_round_identity.payload(),
                        "approval_id": approval.approval_id,
                        "tool_call_id": approval.tool_call_id,
                        **approval_support.bounded_pending_approval_event_payload(
                            approval,
                            redactor=redactor,
                        ),
                        "recovered": True,
                    }
                )
                current[_PENDING_SESSION_INTERRUPT_CHECKPOINT_KEY] = interrupt_payload
            # Existing checkpoint roots were validated by their owning
            # publication boundary. Re-scanning the whole checkpoint with this
            # invocation's secrets would misclassify unrelated protocol values
            # (for example a finish reason containing "tool") as leaked data.
            return copy_durable_json_object(current, "checkpoint")

        checkpoint_event = _redact_event_for_invocation(
            _event_with_tool_round_authority(
                Event(
                    type=EventType.SESSION_CHECKPOINTED,
                    session_id=session.id,
                    agent_name=registered_agent.spec.name,
                    environment_name=_environment_name(registered_environment),
                    payload={
                        "checkpoint": approval_support.PENDING_TOOL_APPROVAL_CHECKPOINT_KEY,
                        "approval_id": approval.approval_id,
                        "tool_call_id": approval.tool_call_id,
                        **tool_round_identity.payload(),
                    },
                ),
                tool_round_identity,
                "approval_id",
            ),
            redactor=redactor,
        )
        requested_payload = {
            **tool_round_identity.payload(),
            "approval_id": approval.approval_id,
            "tool_call_id": approval.tool_call_id,
            **approval_support.bounded_pending_approval_event_payload(
                approval,
                redactor=redactor,
            ),
        }
        if recovered:
            requested_payload["recovered"] = True
        requested_event = _redact_event_for_invocation(
            approval_support.event_with_pending_approval_authority(
                _event_with_tool_round_authority(
                    Event(
                        type=EventType.TOOL_CALL_APPROVAL_REQUESTED,
                        session_id=session.id,
                        agent_name=registered_agent.spec.name,
                        environment_name=_environment_name(registered_environment),
                        tool_name=approval.tool_name,
                        payload=requested_payload,
                    ),
                    tool_round_identity,
                    "approval_id",
                ),
                approval,
            ),
            redactor=redactor,
        )
        events = [checkpoint_event, requested_event]
        target_checkpoint = publish_policy_and_approval(session, checkpoint)
        transcript_cursor = await self._session_store.load_transcript_cursor(session.id)
        prepared = approval_publication.prepare_approval_publication(
            session_id=session.id,
            publication_id=f"approval-open:{approval.approval_id}",
            kind="approval-open",
            intent={
                "schema_version": 1,
                "approval_id": approval.approval_id,
                "tool_call_id": approval.tool_call_id,
                **tool_round_identity.payload(),
                "tool_call_ids": [call.tool_call_id for call in planned_round.tool_calls],
                "source_round_digest": runtime_publication_checkpoint_value_digest(
                    source_round_payload
                ),
                "planned_round_digest": runtime_publication_checkpoint_value_digest(
                    planned_round_payload
                ),
                "approval_digest": runtime_publication_checkpoint_value_digest(approval_payload),
                "event_ids": [event.id for event in events],
            },
            source_checkpoint=checkpoint,
            target_checkpoint=target_checkpoint,
            events=events,
            expected_statuses=eligible_statuses,
            expected_run_epoch=session.run_epoch,
            expected_transcript_cursor=transcript_cursor,
        )
        events = list(prepared.request.events)
        cancellation = await approval_publication.publish_approval_with_exact_replay(
            prepared,
            session_store=self._session_store,
            event_writer=self._event_writer,
        )
        if cancellation is not None:
            raise cancellation
        return approval, events

    async def checkpoint_tool_round_policy_plan(
        self,
        *,
        session: Session,
        registered_agent: runtime_records.RegisteredAgentState,
        tool_calls: list[runtime_records.ToolCallRequest],
        policy_outcomes: list[runtime_records.ToolCallPolicyOutcome],
        active_taint_by_id: Mapping[str, frozenset[str]],
        tool_round_identity: ToolRoundIdentity,
        recovered: bool = False,
    ) -> tool_round_recovery.PendingToolRound:
        """Atomically replace an unplanned round with its durable policy plan."""

        tool_round_identity = copy_tool_round_identity(tool_round_identity)
        redactor = _redactor_for_tool_calls(
            self._secret_redactor,
            registered_agent=registered_agent,
            tool_calls=tool_calls,
        )
        checkpoint = await self._session_store.load_checkpoint(session.id)
        checkpoint = {} if checkpoint is None else copy_json_value(checkpoint, "checkpoint")
        pending_round = tool_round_recovery.pending_tool_round_from_checkpoint(
            checkpoint,
            redactor=self._secret_redactor,
            consume_on_rejection=True,
        )
        if pending_round is None:
            raise RuntimeError("Session has no pending tool round for its policy plan.")
        _require_matching_policy_round(
            pending_round=pending_round,
            tool_round_identity=tool_round_identity,
            tool_calls=tool_calls,
        )
        if (
            approval_support.pending_approval_from_checkpoint(
                checkpoint,
                redactor=self._secret_redactor,
                consume_on_rejection=True,
            )
            is not None
        ):
            raise RuntimeError("Session already has a pending tool approval.")
        planned_round = _planned_pending_tool_round(
            pending_round=pending_round,
            tool_calls=tool_calls,
            policy_outcomes=policy_outcomes,
            active_taint_by_id=active_taint_by_id,
            redactor=redactor,
        )
        # Retain the exact durable source for the compare-and-swap. Additive
        # model defaults, including assistant publication evidence introduced
        # after an older v2 stage was written, must not fabricate a mismatch.
        source_round_payload = copy_json_value(
            checkpoint[tool_round_recovery.PENDING_TOOL_ROUND_CHECKPOINT_KEY],
            "pending_tool_round",
        )
        eligible_statuses = {session.status} if recovered else {SessionStatus.RUNNING}
        planned_round_payload = _require_secret_free_durable_object(
            planned_round.model_dump(mode="json"),
            redactor=redactor,
            field_name="pending_tool_round",
            schema_root=tool_round_recovery.PENDING_TOOL_ROUND_CHECKPOINT_KEY,
        )

        def publish_policy_plan(
            current_session: Session,
            current_checkpoint: dict[str, Any] | None,
        ) -> dict[str, Any]:
            if (
                current_session.status not in eligible_statuses
                or current_session.run_epoch != session.run_epoch
            ):
                raise RuntimeError("Tool policy publication lost its run fence.")
            current = (
                {}
                if current_checkpoint is None
                else copy_json_value(current_checkpoint, "checkpoint")
            )
            if (
                current.get(tool_round_recovery.PENDING_TOOL_ROUND_CHECKPOINT_KEY)
                != source_round_payload
            ):
                raise RuntimeError("Pending tool round changed before policy publication.")
            if approval_support.PENDING_TOOL_APPROVAL_CHECKPOINT_KEY in current:
                raise RuntimeError("Session already has a pending tool approval.")
            if approval_support.APPROVAL_RESOLUTION_INTENT_CHECKPOINT_KEY in current:
                raise RuntimeError("Session has an orphaned approval resolution intent.")
            current[tool_round_recovery.PENDING_TOOL_ROUND_CHECKPOINT_KEY] = planned_round_payload
            return copy_durable_json_object(current, "checkpoint")

        await self._session_store.transform_checkpoint(session.id, publish_policy_plan)
        return planned_round

    async def checkpoint_pending_user_input(
        self,
        *,
        session: Session,
        registered_agent: runtime_records.RegisteredAgentState,
        registered_environment: runtime_records.RegisteredEnvironment | None,
        tool_call: runtime_records.ToolCallRequest,
        tool_calls: list[runtime_records.ToolCallRequest],
        policy_outcomes: list[runtime_records.ToolCallPolicyOutcome] | None,
        active_taint_by_id: Mapping[str, frozenset[str]],
        task_id: str | None,
        structured_output: StructuredOutputSpec | None,
        thinking: ThinkingConfig | None,
        max_steps: int | None,
        limits: RunLimits | None,
        budget_limits: tuple[BudgetLimit, ...] | None,
        retry_policy: RetryPolicy | None,
        question: str,
        options: list[str],
        tool_round_identity: ToolRoundIdentity,
    ) -> tuple[PendingUserInput, list[Event]]:
        tool_round_identity = copy_tool_round_identity(tool_round_identity)
        redactor = _redactor_for_tool_calls(
            self._secret_redactor,
            registered_agent=registered_agent,
            tool_calls=tool_calls,
        )
        checkpoint = await self._session_store.load_checkpoint(session.id)
        checkpoint = {} if checkpoint is None else copy_json_value(checkpoint, "checkpoint")
        pending_round = tool_round_recovery.pending_tool_round_from_checkpoint(
            checkpoint,
            redactor=self._secret_redactor,
            consume_on_rejection=True,
        )
        if pending_round is None:
            raise RuntimeError("Session has no pending tool round for its user-input pause.")
        _require_matching_policy_round(
            pending_round=pending_round,
            tool_round_identity=tool_round_identity,
            tool_calls=tool_calls,
        )
        if (
            approval_support.pending_approval_from_checkpoint(
                checkpoint,
                redactor=self._secret_redactor,
                consume_on_rejection=True,
            )
            is not None
        ):
            raise RuntimeError("Session already has a pending tool approval.")
        pending_user_input, _ = user_input_lifecycle_authority_from_checkpoint(
            checkpoint,
            redactor=self._secret_redactor,
            consume_on_rejection=True,
            current_run_epoch=session.run_epoch,
        )
        if pending_user_input is not None:
            raise RuntimeError("Session already has a pending user input.")
        active_profile = active_invocation_execution_profile_from_checkpoint(checkpoint)
        if (
            active_profile is None
            or active_profile.session_id != session.id
            or active_profile.run_epoch != session.run_epoch
            or pending_round.source_run_epoch != session.run_epoch
            or pending_round.execution_profile_fingerprint != active_profile.profile.fingerprint
        ):
            raise RuntimeError("Pending user input has no exact active invocation authority.")
        execution_profile_fingerprint = pending_round.execution_profile_fingerprint
        if execution_profile_fingerprint is None:
            raise RuntimeError("Pending user input has no execution-profile authority.")

        pending = PendingUserInput(
            session_id=session.id,
            session_instance_id=session.instance_id,
            source_interaction_id=active_profile.interaction_id,
            source_run_epoch=session.run_epoch,
            input_id=str(uuid4()),
            tool_round_id=tool_round_identity.tool_round_id,
            model_step_id=tool_round_identity.model_step_id,
            model_attempt_id=tool_round_identity.model_attempt_id,
            model_step=pending_round.model_step,
            tool_call_id=tool_call.id,
            tool_name=tool_call.name,
            question=question,
            options=list(options),
            arguments=copy_json_value(tool_call.arguments, "arguments"),
            agent_name=registered_agent.spec.name,
            environment_name=_environment_name(registered_environment),
            workspace_id=_workspace_id(registered_environment),
            task_id=task_id,
            interaction_id=pending_round.interaction_id,
            execution_profile_fingerprint=execution_profile_fingerprint,
            tool_exposure=pending_round.tool_exposure,
            tool_calls=approval_support.pending_tool_call_approvals(
                tool_calls=tool_calls,
                policy_outcomes=policy_outcomes,
                active_taint_by_id=active_taint_by_id,
                redactor=redactor,
            ),
            assistant_message_state=pending_round.assistant_message_state,
            quarantined_assistant_message=pending_round.quarantined_assistant_message,
            assistant_publication=pending_round.assistant_publication,
            structured_output=copy_structured_output_spec(structured_output),
            thinking=thinking,
            max_steps=max_steps,
            limits=copy_run_limits(limits) if limits is not None else None,
            run_limit_accounting=pending_round.run_limit_accounting,
            budget_limits=(
                copy_request_budget_limits(budget_limits) if budget_limits is not None else None
            ),
            retry_policy=copy_retry_policy(retry_policy) if retry_policy is not None else None,
        )
        pending_payload = _require_secret_free_durable_object(
            pending.model_dump(mode="json"),
            redactor=redactor,
            field_name="pending_user_input",
            schema_root=PENDING_USER_INPUT_CHECKPOINT_KEY,
        )
        source_round_payload = copy_json_value(
            checkpoint[tool_round_recovery.PENDING_TOOL_ROUND_CHECKPOINT_KEY],
            "pending_tool_round",
        )
        target_checkpoint = copy_json_value(checkpoint, "checkpoint")
        target_checkpoint.pop(tool_round_recovery.PENDING_TOOL_ROUND_CHECKPOINT_KEY)
        target_checkpoint[PENDING_USER_INPUT_CHECKPOINT_KEY] = pending_payload
        target_checkpoint = _require_secret_free_durable_object(
            target_checkpoint,
            redactor=redactor,
            field_name="checkpoint",
        )
        pause_digest = pending_user_input_digest(pending)
        checkpoint_event = _redact_event_for_invocation(
            event_with_pending_user_input_authority(
                _event_with_tool_round_authority(
                    Event(
                        type=EventType.SESSION_CHECKPOINTED,
                        session_id=session.id,
                        interaction_id=pending.source_interaction_id,
                        agent_name=registered_agent.spec.name,
                        environment_name=_environment_name(registered_environment),
                        payload={
                            "checkpoint": PENDING_USER_INPUT_CHECKPOINT_KEY,
                            "input_id": pending.input_id,
                            "tool_call_id": pending.tool_call_id,
                            "source_run_epoch": pending.source_run_epoch,
                            "pause_digest": pause_digest,
                            **tool_round_identity.payload(),
                        },
                    ),
                    tool_round_identity,
                    "input_id",
                ),
                pending,
            ),
            redactor=redactor,
        )
        public_question, public_options = public_pending_user_input_prompt(pending)
        public_prompt_payload = (
            {}
            if public_question is None
            else {"question": public_question, "options": public_options}
        )
        public_pending_input = public_pending_user_input_event_payload(pending)
        awaiting_event = _redact_event_for_invocation(
            event_with_pending_user_input_authority(
                _event_with_tool_round_authority(
                    event_with_execution_profile_authority(
                        Event(
                            type=EventType.SESSION_AWAITING_USER_INPUT,
                            session_id=session.id,
                            interaction_id=pending.source_interaction_id,
                            agent_name=registered_agent.spec.name,
                            environment_name=_environment_name(registered_environment),
                            tool_name=tool_call.name,
                            payload={
                                **tool_round_identity.payload(),
                                "input_id": pending.input_id,
                                "tool_call_id": pending.tool_call_id,
                                "source_run_epoch": pending.source_run_epoch,
                                "pause_digest": pause_digest,
                                **public_prompt_payload,
                                "tool_calls": public_pending_input["tool_calls"],
                            },
                        ),
                        active_profile.profile,
                    ),
                    tool_round_identity,
                    "input_id",
                ),
                pending,
            ),
            redactor=redactor,
        )
        events = [checkpoint_event, awaiting_event]
        transcript_cursor = await self._session_store.load_transcript_cursor(session.id)
        prepared = approval_publication.prepare_pending_action_publication(
            session_id=session.id,
            publication_id=f"user-input-open:{pending.input_id}",
            kind="user-input-open",
            intent={
                **pending_user_input_identity(pending),
                "source_round_digest": runtime_publication_checkpoint_value_digest(
                    source_round_payload
                ),
                "event_ids": [event.id for event in events],
            },
            source_checkpoint=checkpoint,
            target_checkpoint=target_checkpoint,
            events=events,
            expected_statuses={SessionStatus.RUNNING},
            expected_run_epoch=session.run_epoch,
            expected_transcript_cursor=transcript_cursor,
        )
        cancellation = await approval_publication.publish_pending_action_with_exact_replay(
            prepared,
            session_store=self._session_store,
            event_writer=self._event_writer,
        )
        if cancellation is not None:
            raise cancellation
        return pending, list(prepared.request.events)

    async def checkpoint_with_pending_tool_round(
        self,
        *,
        session: Session,
        registered_agent: runtime_records.RegisteredAgentState,
        registered_environment: runtime_records.RegisteredEnvironment | None,
        tool_calls: list[runtime_records.ToolCallRequest],
        policy_outcomes: list[runtime_records.ToolCallPolicyOutcome] | None,
        task_id: str | None,
        structured_output: StructuredOutputSpec | None,
        tool_round_identity: ToolRoundIdentity,
    ) -> tuple[dict[str, Any], tool_round_recovery.PendingToolRound]:
        checkpoint = await self._session_store.load_checkpoint(session.id)
        redactor = _redactor_for_tool_calls(
            self._secret_redactor,
            registered_agent=registered_agent,
            tool_calls=tool_calls,
        )
        return tool_round_recovery.checkpoint_with_pending_tool_round(
            checkpoint,
            agent_name=registered_agent.spec.name,
            environment_name=_environment_name(registered_environment),
            task_id=task_id,
            source_run_epoch=session.run_epoch,
            tool_calls=tool_calls,
            policy_outcomes=policy_outcomes,
            structured_output=structured_output,
            tool_round_identity=tool_round_identity,
            redactor=redactor,
        )

    def redactor_for_tool_calls(
        self,
        *,
        registered_agent: runtime_records.RegisteredAgentState,
        tool_calls: list[runtime_records.ToolCallRequest],
    ) -> SecretRedactor:
        """Compose app and adapter-owned secrets for a tool publication boundary."""

        return _redactor_for_tool_calls(
            self._secret_redactor,
            registered_agent=registered_agent,
            tool_calls=tool_calls,
        )

    async def checkpoint_without_pending_tool_round(
        self,
        session_id: str,
    ) -> dict[str, Any]:
        checkpoint = await self._session_store.load_checkpoint(session_id)
        return tool_round_recovery.checkpoint_without_pending_tool_round(checkpoint)

    async def clear_pending_tool_approval_for_tool_round(
        self,
        session_id: str,
        tool_calls: list[runtime_records.ToolCallRequest],
    ) -> None:
        expected_ids = {tool_call.id for tool_call in tool_calls}
        if not expected_ids:
            return
        checkpoint = await self._session_store.load_checkpoint(session_id)
        if checkpoint is None:
            return
        copied_checkpoint = copy_json_value(checkpoint, "checkpoint")
        pending_approval = approval_support.pending_approval_from_checkpoint(
            copied_checkpoint,
            redactor=self._secret_redactor,
            consume_on_rejection=True,
        )
        if pending_approval is None or pending_approval.tool_call_id not in expected_ids:
            return
        await self._session_store.transform_checkpoint(
            session_id,
            lambda current_session, current_checkpoint: (
                self._checkpoint_with_approval_interrupt_close_intent(
                    current_session=current_session,
                    checkpoint=current_checkpoint,
                    approval=pending_approval,
                )
            ),
        )

    def _checkpoint_with_approval_interrupt_close_intent(
        self,
        *,
        current_session: Session,
        checkpoint: dict[str, Any] | None,
        approval: PendingToolApproval,
    ) -> dict[str, Any]:
        """Clear one approval only after its exact interrupt-close owner is durable."""

        if current_session.status != SessionStatus.INTERRUPTING:
            raise RuntimeError(
                "Pending approval can be cleared for interruption only while interrupting."
            )
        copied = approval_support.checkpoint_without_exact_pending_approval(
            checkpoint,
            approval=approval,
            redactor=self._secret_redactor,
        )
        interrupt_payload = copied.get(_PENDING_SESSION_INTERRUPT_CHECKPOINT_KEY)
        if (
            type(interrupt_payload) is not dict
            or interrupt_payload.get("interruption_type") != _INTERRUPTION_TYPE_OPERATOR_REQUESTED
            or type(interrupt_payload.get("interruption_request_id")) is not str
            or not interrupt_payload["interruption_request_id"].strip()
            or interrupt_payload["interruption_request_id"].strip()
            != interrupt_payload["interruption_request_id"]
        ):
            raise RuntimeError("Pending approval interruption has no authoritative close intent.")
        interrupt_payload = copy_json_value(
            interrupt_payload,
            "pending_session_interrupt",
        )
        interrupt_payload[approval_support.APPROVAL_INTERRUPT_CLOSE_INTENT_KEY] = {
            "approval_id": approval.approval_id,
            "tool_call_id": approval.tool_call_id,
            "tool_round_id": approval.tool_round_id,
            "model_step_id": approval.model_step_id,
            "model_attempt_id": approval.model_attempt_id,
        }
        copied[_PENDING_SESSION_INTERRUPT_CHECKPOINT_KEY] = interrupt_payload
        return copied

    async def execute_tool_call(
        self,
        *,
        session: Session,
        registered_agent: runtime_records.RegisteredAgentState,
        registered_environment: runtime_records.RegisteredEnvironment | None,
        tool_call: runtime_records.ToolCallRequest,
        request_metadata: dict[str, Any],
        task_id: str | None,
        model_step: int | None = None,
        execution_profile: ExecutionProfileIdentity | None = None,
        invocation_context: InvocationContext | None = None,
        check_policy: bool = True,
        emit_started: bool = True,
        policy_result: ToolPolicyResult | None = None,
        policy_evidence: ToolPolicyEvidence = ToolPolicyEvidence.AUTHORITATIVE,
        tool_exposure: ResolvedToolExposureAuthority | None = None,
        policy_output_secret_resolution_scope: Literal["static", "dynamic", "unknown"] = "unknown",
        approval_id: str | None = None,
        tool_round_identity: ToolRoundIdentity,
        input_id: str | None = None,
        taint_labels: frozenset[str] | None = None,
        publish_arguments_as_unavailable: bool = False,
        deferred_terminal_stager: DeferredTerminalStager | None = None,
        deferred_terminal_capture_recorder: DeferredTerminalCaptureRecorder | None = None,
        resolved_redactor_observer: (
            Callable[[str, InvocationRedactorSnapshot], Awaitable[None]] | None
        ) = None,
        publication_snapshot_observer: (
            Callable[
                [str, invocation_secrets.InvocationPublicationSnapshot],
                Awaitable[None],
            ]
            | None
        ) = None,
        rejoin_targeted_invocation: bool = False,
    ) -> AsyncIterator[tuple[Event, runtime_records.ToolCallOutcome | None]]:
        if invocation_context is not None and (
            invocation_context.binding.session_id != session.id
            or invocation_context.registered_agent is not registered_agent
            or invocation_context.registered_environment is not registered_environment
            or invocation_context.profile is not execution_profile
        ):
            raise RuntimeError("Tool execution lost frozen invocation authority.")
        tool_round_identity = copy_tool_round_identity(tool_round_identity)
        identity_payload = tool_round_identity.payload()
        tool_round_id = tool_round_identity.tool_round_id
        environment_name = _environment_name(registered_environment)
        registered_tool = registered_agent.executable_tool(tool_call.name)
        if rejoin_targeted_invocation:
            rejoined_events = await self.rejoin_targeted_tool_call(
                session=session,
                registered_agent=registered_agent,
                registered_environment=registered_environment,
                tool_call=tool_call,
                task_id=task_id,
                invocation_context=invocation_context,
            )
            for rejoined_event in rejoined_events:
                yield rejoined_event, None
        if tool_call.targeted_tool_rejection is not None:
            rejection = tool_call.targeted_tool_rejection
            result = ToolResult(
                content=targeted_tool_rejection_content(
                    rejection.reason,
                    dispatch_kind=rejection.dispatch_kind,
                ),
                structured={
                    "status": "rejected",
                    "reason": rejection.reason.value,
                },
                is_error=True,
            )
            idempotency_key = tool_execution.tool_idempotency_key(
                session_id=session.id,
                tool_call_id=tool_call.id,
                tool_round_id=tool_round_id,
                approval_id=approval_id,
                pause_id=input_id,
            )
            payload: dict[str, Any] = {
                "tool_call_id": tool_call.id,
                "idempotency_key": idempotency_key,
                "blocked_by": (
                    "targeted_tool_gateway"
                    if rejection.dispatch_kind == "gateway"
                    else "targeted_tool_native"
                ),
                "reason": rejection.reason.value,
                "rejection_event_id": rejection.rejection_event_id,
                "dispatch_kind": rejection.dispatch_kind,
                "model_tool_name": rejection.model_tool_name,
                "result": result.model_dump(mode="json"),
                **tool_argument_publication.unavailable_argument_projection().payload_fields(),
                **identity_payload,
            }
            event = event_with_runtime_payload_authority(
                _event_with_tool_round_authority(
                    event_with_execution_profile_authority(
                        Event(
                            type=EventType.TOOL_CALL_FAILED,
                            session_id=session.id,
                            agent_name=registered_agent.spec.name,
                            environment_name=environment_name,
                            tool_name=rejection.model_tool_name,
                            payload=payload,
                        ),
                        execution_profile,
                    ),
                    tool_round_identity,
                ),
                "dispatch_kind",
                "model_tool_name",
            )
            outcome = runtime_records.ToolCallOutcome(
                call=replace(tool_call, arguments={}),
                result=result,
            )
            publication_snapshot = invocation_secrets.InvocationPublicationSnapshot(
                redactor=self._secret_redactor,
                unsafe_output=False,
            )
            if publication_snapshot_observer is not None:
                await publication_snapshot_observer(tool_call.id, publication_snapshot)
            if deferred_terminal_stager is not None:
                await deferred_terminal_stager(
                    event,
                    outcome,
                    False,
                    False,
                    publication_snapshot,
                )
                return
            yield await self._event_writer.emit(event), outcome
            return
        mcp_policy_authority_rejected = (
            policy_evidence is ToolPolicyEvidence.UNREGISTERED
            and registered_tool is not None
            and isinstance(registered_tool.tool, McpToolAdapter)
        )
        if mcp_policy_authority_rejected or _registered_mcp_tool_authority_is_unavailable(
            registered_tool
        ):
            result = ToolResult(
                content="MCP tool unavailable because its catalogue authority is not current.",
                structured={
                    "status": "rejected",
                    "reason": "mcp_catalogue_authority_unavailable",
                },
                is_error=True,
            )
            idempotency_key = tool_execution.tool_idempotency_key(
                session_id=session.id,
                tool_call_id=tool_call.id,
                tool_round_id=tool_round_id,
                approval_id=approval_id,
                pause_id=input_id,
            )
            payload: dict[str, Any] = {
                "tool_call_id": tool_call.id,
                "idempotency_key": idempotency_key,
                "blocked_by": "mcp_catalogue_authority",
                "reason": "mcp_catalogue_authority_unavailable",
                "result": result.model_dump(mode="json"),
                **_targeted_tool_invocation_payload(tool_call),
                **tool_argument_publication.unavailable_argument_projection().payload_fields(),
                **identity_payload,
            }
            if approval_id is not None:
                payload["approval_id"] = approval_id
            if input_id is not None:
                payload["input_id"] = input_id
            event = _event_with_targeted_tool_invocation_authority(
                _event_with_tool_round_authority(
                    event_with_execution_profile_authority(
                        Event(
                            type=EventType.TOOL_CALL_FAILED,
                            session_id=session.id,
                            agent_name=registered_agent.spec.name,
                            environment_name=environment_name,
                            tool_name=tool_call.model_tool_name or tool_call.name,
                            payload=payload,
                        ),
                        execution_profile,
                    ),
                    tool_round_identity,
                    *(field for field in ("approval_id", "input_id") if field in payload),
                ),
                tool_call,
            )
            outcome = runtime_records.ToolCallOutcome(
                call=replace(tool_call, arguments={}),
                result=result,
            )
            publication_snapshot = invocation_secrets.InvocationPublicationSnapshot(
                redactor=self._secret_redactor,
                unsafe_output=False,
            )
            if publication_snapshot_observer is not None:
                await publication_snapshot_observer(tool_call.id, publication_snapshot)
            if deferred_terminal_stager is not None:
                await deferred_terminal_stager(
                    event,
                    outcome,
                    False,
                    False,
                    publication_snapshot,
                )
                return
            yield await self._event_writer.emit(event), outcome
            return
        if policy_evidence is ToolPolicyEvidence.UNEXPOSED:
            if tool_exposure is None:
                raise RuntimeError("Unexposed tool call lost its frozen exposure snapshot.")
            tool_exposure = validate_resolved_tool_exposure_authority(
                tool_exposure,
                registered_agent.tool_capabilities,
                catalogue_revision=registered_agent.tool_catalogue.revision,
            )
            if tool_call.name not in registered_agent.executable_tool_names:
                raise RuntimeError("Unexposed tool call is not registered.")
            if tool_call.name in tool_exposure.tool_names:
                raise RuntimeError("Unexposed tool call is present in its exposure snapshot.")
            result = unexposed_tool_result()
            idempotency_key = tool_execution.tool_idempotency_key(
                session_id=session.id,
                tool_call_id=tool_call.id,
                tool_round_id=tool_round_id,
                approval_id=approval_id,
                pause_id=input_id,
            )
            payload: dict[str, Any] = {
                "tool_call_id": tool_call.id,
                "idempotency_key": idempotency_key,
                "blocked_by": "tool_exposure",
                "reason": NOT_EXPOSED_IN_REQUEST_REASON,
                "profile_id": tool_exposure.profile_id,
                "exposure_fingerprint": tool_exposure.fingerprint,
                "result": result.model_dump(mode="json"),
                **tool_argument_publication.unavailable_argument_projection().payload_fields(),
                **identity_payload,
            }
            if approval_id is not None:
                payload["approval_id"] = approval_id
            if input_id is not None:
                payload["input_id"] = input_id
            event = _event_with_tool_round_authority(
                event_with_execution_profile_authority(
                    Event(
                        type=EventType.TOOL_CALL_BLOCKED,
                        session_id=session.id,
                        agent_name=registered_agent.spec.name,
                        environment_name=environment_name,
                        tool_name=tool_call.name,
                        payload=payload,
                    ),
                    execution_profile,
                ),
                tool_round_identity,
                *(
                    field
                    for field in (
                        "approval_id",
                        "input_id",
                        "profile_id",
                        "exposure_fingerprint",
                    )
                    if field in payload
                ),
            )
            projected_call = replace(tool_call, arguments={})
            outcome = runtime_records.ToolCallOutcome(
                call=projected_call,
                result=result,
            )
            publication_snapshot = invocation_secrets.InvocationPublicationSnapshot(
                redactor=self._secret_redactor,
                unsafe_output=False,
            )
            if publication_snapshot_observer is not None:
                await publication_snapshot_observer(tool_call.id, publication_snapshot)
            if deferred_terminal_stager is not None:
                await deferred_terminal_stager(
                    event,
                    outcome,
                    False,
                    False,
                    publication_snapshot,
                )
                return
            yield await self._event_writer.emit(event), outcome
            return
        model_arguments = copy_json_value(tool_call.arguments, "tool_call.arguments")
        invocation_redactor = _redactor_for_tool_calls(
            self._secret_redactor,
            registered_agent=registered_agent,
            tool_calls=[tool_call],
        )
        provisional_output_redactor = invocation_redactor
        publication_snapshot_recorded = False

        async def record_publication_snapshot(
            snapshot: invocation_secrets.InvocationPublicationSnapshot,
        ) -> None:
            nonlocal publication_snapshot_recorded
            if publication_snapshot_recorded:
                return
            if publication_snapshot_observer is not None:
                await publication_snapshot_observer(tool_call.id, snapshot)
            publication_snapshot_recorded = True

        async def record_static_publication_scope() -> (
            invocation_secrets.InvocationPublicationSnapshot
        ):
            snapshot = invocation_secrets.InvocationPublicationSnapshot(
                redactor=invocation_redactor,
                unsafe_output=False,
            )
            await record_publication_snapshot(snapshot)
            return snapshot

        async def record_interrupted_publication_scope(
            interrupt: BaseException,
        ) -> bool:
            """Persist the final scope without replacing an owned interrupt."""

            def record_abandoned_publication_diagnostic(error: BaseException) -> None:
                if type(error) is GeneratorExit:
                    error_type = "GeneratorExit"
                elif isinstance(error, BaseExceptionGroup):
                    error_type = "BaseExceptionGroup"
                elif isinstance(error, asyncio.CancelledError):
                    error_type = "CancelledError"
                else:
                    # Exception subclass names and messages are extension-owned
                    # and can contain workload-derived values. Keep only a fixed
                    # classification at this public cancellation boundary.
                    error_type = "BaseException"
                note = (
                    "Assistant publication projection terminated with "
                    f"{error_type} while preserving cancellation."
                )
                interrupt.add_note(note)
                if isinstance(interrupt, BaseExceptionGroup):
                    for candidate in iter_exception_tree(interrupt):
                        if isinstance(candidate, asyncio.CancelledError):
                            candidate.add_note(note)

            publication_task = asyncio.create_task(
                persist_sealed_invocation_evidence(invocation_secret_scope.seal_for_publication())
            )
            publication_outcome = await await_shielded_task_outcome(publication_task)
            publication_error = publication_outcome.error
            if publication_error is not None and not isinstance(publication_error, Exception):
                if type(publication_error) in (KeyboardInterrupt, SystemExit):
                    # Process-level interpreter signals retain their ordinary
                    # semantics; this issue does not redefine process shutdown.
                    raise publication_error
                # Caller cancellation remains authoritative over task-contained
                # snapshot abandonment. Retain only a fixed diagnostic because
                # raw extension-owned exception state is unsafe to publish.
                record_abandoned_publication_diagnostic(publication_error)
            if publication_error is not None and isinstance(publication_error, Exception):
                interrupt.add_note(
                    "Failed to persist the assistant publication projection while "
                    "preserving cancellation."
                )
            later_cancellation = publication_outcome.cancellation
            if later_cancellation is None:
                return False
            current_task = asyncio.current_task()
            if current_task is None:  # pragma: no cover - coroutine execution invariant
                interrupt.add_note(
                    "A later cancellation could not be redelivered after publication."
                )
                return False
            cancellation_args = later_cancellation.args
            if not cancellation_args:
                current_task.cancel()
            else:
                current_task.cancel(cancellation_args[0])
            return True

        started_event: Event | None = None
        if registered_tool is not None and not registered_tool.publish_arguments:
            publish_arguments_as_unavailable = True
        if taint_labels is None:
            taint_labels = frozenset(
                await self.prior_taint_labels_for_policy(
                    session_id=session.id,
                    policy=registered_agent.tool_policy,
                    request_metadata=request_metadata,
                )
            )
        idempotency_key = tool_execution.tool_idempotency_key(
            session_id=session.id,
            tool_call_id=tool_call.id,
            tool_round_id=tool_round_id,
            approval_id=approval_id,
            pause_id=input_id,
        )
        if emit_started:
            payload: dict[str, Any] = {
                "tool_call_id": tool_call.id,
                "idempotency_key": idempotency_key,
                **_targeted_tool_invocation_payload(tool_call),
                **tool_argument_publication.quarantined_argument_fields(),
                **identity_payload,
            }
            if registered_tool is not None:
                payload["effect"] = registered_tool.effect.value
            if approval_id is not None:
                payload["approval_id"] = approval_id
            if input_id is not None:
                payload["input_id"] = input_id
            started = event_with_execution_profile_authority(
                Event(
                    type=EventType.TOOL_CALL_STARTED,
                    session_id=session.id,
                    agent_name=registered_agent.spec.name,
                    environment_name=environment_name,
                    tool_name=tool_call.name,
                    payload=payload,
                ),
                execution_profile,
            )
            started = _event_with_targeted_tool_invocation_authority(started, tool_call)
            started_event = await self._event_writer.emit(
                prepare_runtime_event(
                    _event_with_tool_round_authority(
                        started,
                        tool_round_identity,
                        *(field for field in ("approval_id", "input_id") if field in payload),
                    ),
                    redactor=invocation_redactor,
                )
            )
            yield started_event, None

        if registered_tool is None:
            publication_snapshot = await record_static_publication_scope()
            result = ToolResult(
                content=f"Tool not registered: {tool_call.name}",
                is_error=True,
            )
            payload = {
                "tool_call_id": tool_call.id,
                "idempotency_key": idempotency_key,
                "result": result.model_dump(),
                **identity_payload,
            }
            if approval_id is not None:
                payload["approval_id"] = approval_id
            if input_id is not None:
                payload["input_id"] = input_id
            async for event in self.emit_tool_call_result_with_hooks(
                event=Event(
                    type=EventType.TOOL_CALL_FAILED,
                    session_id=session.id,
                    agent_name=registered_agent.spec.name,
                    environment_name=environment_name,
                    tool_name=tool_call.name,
                    payload=payload,
                ),
                session=session,
                registered_agent=registered_agent,
                registered_environment=registered_environment,
                tool_call=tool_call,
                result=result,
                task_id=task_id,
                execution_profile=execution_profile,
                invocation_context=invocation_context,
                redactor=invocation_redactor,
                output_redactor=provisional_output_redactor,
                deferred_terminal_stager=deferred_terminal_stager,
                publication_snapshot=publication_snapshot,
            ):
                yield event
            return

        if check_policy:
            if policy_result is None:
                resolved_policy_result = await self.authorize_tool_call(
                    session=session,
                    registered_agent=registered_agent,
                    registered_environment=registered_environment,
                    tool_call=tool_call,
                    request_metadata=request_metadata,
                    taint_labels=taint_labels,
                )
            else:
                resolved_policy_result = tool_execution.validate_tool_policy_result(policy_result)
            if resolved_policy_result.decision == ToolPolicyDecision.DENY:
                publication_snapshot = await record_static_publication_scope()
                public_policy_result = approval_support.public_policy_denial_result(
                    secret_resolution_scope=policy_output_secret_resolution_scope,
                    policy_result=resolved_policy_result,
                    publish_arguments=registered_tool.publish_arguments,
                )
                reason = tool_execution.policy_denial_reason(public_policy_result)
                result = tool_execution.blocked_tool_result(public_policy_result, reason=reason)
                payload = {
                    "tool_call_id": tool_call.id,
                    "idempotency_key": idempotency_key,
                    **policy_denial_payload_fields(
                        tool_name=tool_call.name,
                        denied_by=_TOOL_POLICY_DENIAL_SOURCE,
                        decision=public_policy_result.decision.value,
                        reason=reason,
                        metadata=public_policy_result.metadata,
                    ),
                    "result": result.model_dump(),
                    **identity_payload,
                }
                if approval_id is not None:
                    payload["approval_id"] = approval_id
                if input_id is not None:
                    payload["input_id"] = input_id
                async for event in self.emit_tool_call_result_with_hooks(
                    event=Event(
                        type=EventType.TOOL_CALL_BLOCKED,
                        session_id=session.id,
                        agent_name=registered_agent.spec.name,
                        environment_name=environment_name,
                        tool_name=tool_call.name,
                        payload=payload,
                    ),
                    session=session,
                    registered_agent=registered_agent,
                    registered_environment=registered_environment,
                    tool_call=tool_call,
                    result=result,
                    task_id=task_id,
                    execution_profile=execution_profile,
                    invocation_context=invocation_context,
                    redactor=invocation_redactor,
                    output_redactor=provisional_output_redactor,
                    deferred_terminal_stager=deferred_terminal_stager,
                    publication_snapshot=publication_snapshot,
                ):
                    yield event
                return
            if resolved_policy_result.decision == ToolPolicyDecision.REQUIRE_APPROVAL:
                approval, approval_events = await self.checkpoint_pending_tool_approval(
                    session=session,
                    registered_agent=registered_agent,
                    registered_environment=registered_environment,
                    tool_call=tool_call,
                    tool_calls=[tool_call],
                    policy_outcomes=None,
                    active_taint_by_id={tool_call.id: taint_labels},
                    task_id=task_id,
                    policy_result=resolved_policy_result,
                    structured_output=None,
                    thinking=None,
                    max_steps=None,
                    limits=None,
                    budget_limits=None,
                    retry_policy=None,
                    tool_round_identity=tool_round_identity,
                )
                for approval_event in approval_events:
                    yield approval_event, None
                raise ToolApprovalRequired(approval)
            if resolved_policy_result.decision != ToolPolicyDecision.ALLOW:
                raise ValueError(
                    f"Unsupported tool policy decision: {resolved_policy_result.decision}"
                )

        anchor_event = started_event or Event(
            type=EventType.TOOL_CALL_STARTED,
            session_id=session.id,
            agent_name=registered_agent.spec.name,
            environment_name=environment_name,
            tool_name=tool_call.name,
            payload={"tool_call_id": tool_call.id, **identity_payload},
        )
        before_resolution = _BeforeToolCallResolution(arguments=deepcopy(tool_call.arguments))
        quarantine_pre_execution_publication = (
            policy_output_secret_resolution_scope != "static"
            or not registered_tool.publish_arguments
        )
        async for hook_event in self._run_before_tool_call_hooks(
            session=session,
            registered_agent=registered_agent,
            registered_environment=registered_environment,
            tool_call=tool_call,
            anchor_event=anchor_event,
            task_id=task_id,
            execution_profile=execution_profile,
            invocation_context=invocation_context,
            resolution=before_resolution,
            redactor=invocation_redactor,
            output_redactor=provisional_output_redactor,
            quarantine_output=quarantine_pre_execution_publication,
        ):
            yield hook_event, None
        effective_tool_call = (
            tool_call
            if before_resolution.arguments == tool_call.arguments
            else replace(tool_call, arguments=before_resolution.arguments)
        )
        effective_arguments_payload = (
            {"effective_arguments": effective_tool_call.arguments}
            if effective_tool_call is not tool_call
            else {}
        )
        if before_resolution.block_reason is not None:
            publication_snapshot = await record_static_publication_scope()
            block_reason = (
                before_resolution.block_reason
                if registered_tool.publish_arguments
                else "Tool call blocked by a before_tool_call hook."
            )
            async for event in self._emit_terminal_tool_result(
                session=session,
                registered_agent=registered_agent,
                registered_environment=registered_environment,
                tool_call=effective_tool_call,
                event_type=EventType.TOOL_CALL_BLOCKED,
                result=ToolResult(content=block_reason, is_error=True),
                extra_payload={
                    "reason": block_reason,
                    "blocked_by": "before_tool_call_hook",
                    "idempotency_key": idempotency_key,
                    **effective_arguments_payload,
                },
                task_id=task_id,
                execution_profile=execution_profile,
                invocation_context=invocation_context,
                tool_round_identity=tool_round_identity,
                approval_id=approval_id,
                input_id=input_id,
                allow_modification=False,
                redactor=invocation_redactor,
                output_redactor=provisional_output_redactor,
                deferred_terminal_stager=deferred_terminal_stager,
                publication_snapshot=publication_snapshot,
            ):
                yield event
            return
        if before_resolution.short_circuit_result is not None:
            publication_snapshot = await record_static_publication_scope()
            short_result = (
                before_resolution.short_circuit_result
                if registered_tool.publish_arguments
                else _private_argument_short_circuit_result(before_resolution.short_circuit_result)
            )
            async for event in self._emit_terminal_tool_result(
                session=session,
                registered_agent=registered_agent,
                registered_environment=registered_environment,
                tool_call=effective_tool_call,
                event_type=(
                    EventType.TOOL_CALL_FAILED
                    if short_result.is_error
                    else EventType.TOOL_CALL_COMPLETED
                ),
                result=short_result,
                extra_payload={
                    "short_circuited_by": "before_tool_call_hook",
                    "idempotency_key": idempotency_key,
                    **effective_arguments_payload,
                },
                task_id=task_id,
                execution_profile=execution_profile,
                invocation_context=invocation_context,
                tool_round_identity=tool_round_identity,
                approval_id=approval_id,
                input_id=input_id,
                allow_modification=registered_tool.publish_arguments,
                redactor=invocation_redactor,
                output_redactor=provisional_output_redactor,
                deferred_terminal_stager=deferred_terminal_stager,
                publication_snapshot=publication_snapshot,
            ):
                yield event
            return

        if effective_tool_call is not tool_call:
            reauthorization = await self.authorize_tool_call(
                session=session,
                registered_agent=registered_agent,
                registered_environment=registered_environment,
                tool_call=effective_tool_call,
                request_metadata={
                    **request_metadata,
                    TOOL_POLICY_REAUTHORIZATION_METADATA_KEY: True,
                },
                taint_labels=taint_labels,
            )
            if reauthorization.decision != ToolPolicyDecision.ALLOW:
                publication_snapshot = await record_static_publication_scope()
                if reauthorization.decision == ToolPolicyDecision.DENY:
                    public_reauthorization = approval_support.public_policy_denial_result(
                        secret_resolution_scope=policy_output_secret_resolution_scope,
                        policy_result=reauthorization,
                        publish_arguments=registered_tool.publish_arguments,
                    )
                    reason = tool_execution.policy_denial_reason(public_reauthorization)
                    metadata = public_reauthorization.metadata
                else:
                    metadata = (
                        reauthorization.metadata
                        if policy_output_secret_resolution_scope == "static"
                        and registered_tool.publish_arguments
                        else {}
                    )
                    reason = (
                        (
                            reauthorization.reason
                            if policy_output_secret_resolution_scope == "static"
                            and registered_tool.publish_arguments
                            else None
                        )
                        or "Modified tool arguments require approval, which before_tool_call "
                        "hook modifications do not support."
                    )
                async for event in self._emit_terminal_tool_result(
                    session=session,
                    registered_agent=registered_agent,
                    registered_environment=registered_environment,
                    tool_call=effective_tool_call,
                    event_type=EventType.TOOL_CALL_BLOCKED,
                    result=ToolResult(content=reason, is_error=True),
                    extra_payload={
                        **policy_denial_payload_fields(
                            tool_name=effective_tool_call.name,
                            denied_by=_TOOL_POLICY_DENIAL_SOURCE,
                            decision=reauthorization.decision.value,
                            reason=reason,
                            metadata=metadata,
                        ),
                        "blocked_by": "tool_policy_reauthorization",
                        "idempotency_key": idempotency_key,
                    },
                    task_id=task_id,
                    execution_profile=execution_profile,
                    invocation_context=invocation_context,
                    tool_round_identity=tool_round_identity,
                    approval_id=approval_id,
                    input_id=input_id,
                    allow_modification=False,
                    redactor=invocation_redactor,
                    output_redactor=provisional_output_redactor,
                    deferred_terminal_stager=deferred_terminal_stager,
                    publication_snapshot=publication_snapshot,
                ):
                    yield event
                return

        invocation_secret_scope = invocation_secrets.InvocationSecretTracker(invocation_redactor)
        proxy_authorizations: list[invocation_secrets.ProxyAuthorizationRecord] = []
        staged_runner_completion_events: list[tuple[Event, int]] = []
        runner_events: list[Event] = []
        ctx_metadata = tool_execution.context_metadata(
            request_metadata=request_metadata,
            tool_call_id=tool_call.id,
            approval_id=approval_id,
            idempotency_key=idempotency_key,
            tool_effect=registered_tool.effect,
            input_id=input_id,
        )
        if taint_labels:
            ctx_metadata[TAINT_LABELS_METADATA_KEY] = sorted(taint_labels)
        if execution_profile is not None:
            ctx_metadata[EXECUTION_PROFILE_FINGERPRINT_FIELD] = execution_profile.fingerprint

        def redactor_provider():
            return invocation_secret_scope.redactor

        async def observe_runner_execution(
            phase: Literal["started", "completed"],
            payload: dict[str, Any],
            command_evidence_revision: int,
        ) -> None:
            if type(command_evidence_revision) is not int or command_evidence_revision < 0:
                raise TypeError("Runner command evidence revision must be a non-negative integer.")
            event = Event(
                type=(
                    EventType.RUNNER_EXEC_STARTED
                    if phase == "started"
                    else EventType.RUNNER_EXEC_COMPLETED
                ),
                session_id=session.id,
                agent_name=registered_agent.spec.name,
                environment_name=environment_name,
                tool_name=tool_call.name,
                payload={
                    **payload,
                    "tool_call_id": tool_call.id,
                    "idempotency_key": idempotency_key,
                    **identity_payload,
                    **({"approval_id": approval_id} if approval_id is not None else {}),
                    **({"input_id": input_id} if input_id is not None else {}),
                },
            )
            event = event_with_execution_profile_authority(event, execution_profile)
            event = _event_with_tool_round_authority(
                event,
                tool_round_identity,
                *(field for field in ("approval_id", "input_id") if field in event.payload),
            )

            command = event.payload.get("command")
            if not isinstance(command, dict) or command.get("kind") not in {
                "process",
                "shell",
            }:
                raise RuntimeError("Runner command evidence is malformed.")
            if phase == "started":
                # A command must not cross the runner boundary until its exact
                # invocation/profile linkage is durable. Command arguments can
                # become secret only after dispatch, so the pre-dispatch record
                # is deliberately content-free instead of being held in memory.
                event = event.model_copy(
                    update={
                        "payload": {
                            **event.payload,
                            "command": {
                                "kind": command["kind"],
                                "arguments_state": "unavailable",
                            },
                        }
                    },
                    deep=True,
                )
                runner_events.append(
                    await self._event_writer.emit(
                        prepare_runtime_event(
                            event,
                            redactor=invocation_secret_scope.redactor,
                        )
                    )
                )
                return
            staged_runner_completion_events.append((event, command_evidence_revision))

        async def publish_staged_runner_events(
            snapshot: invocation_secrets.InvocationPublicationSnapshot,
        ) -> None:
            """Publish runner completion detail after late secret registration closes."""

            nonlocal staged_runner_completion_events
            if not staged_runner_completion_events:
                return
            final_revision = invocation_secret_scope.snapshot().revision
            captured_events = staged_runner_completion_events
            staged_runner_completion_events = []
            prepared_events: list[Event] = []
            for event, command_evidence_revision in captured_events:
                if snapshot.secret_scope_incomplete or command_evidence_revision != final_revision:
                    command = event.payload.get("command")
                    if not isinstance(command, dict) or command.get("kind") not in {
                        "process",
                        "shell",
                    }:
                        raise RuntimeError("Runner command evidence is malformed.")
                    event = event.model_copy(
                        update={
                            "payload": {
                                **event.payload,
                                "command": {
                                    "kind": command["kind"],
                                    "arguments_state": "unavailable",
                                },
                            }
                        },
                        deep=True,
                    )
                prepared_events.append(prepare_runtime_event(event, redactor=snapshot.redactor))
            captured_events.clear()
            for event in prepared_events:
                runner_events.append(await self._event_writer.emit(event))

        async def persist_sealed_invocation_evidence(
            snapshot: invocation_secrets.InvocationPublicationSnapshot,
        ) -> None:
            await record_publication_snapshot(snapshot)
            await publish_staged_runner_events(snapshot)

        async def persist_resolved_secret_projection(
            snapshot: InvocationRedactorSnapshot,
        ) -> None:
            if resolved_redactor_observer is not None:
                await resolved_redactor_observer(tool_call.id, snapshot)
                return
            await self._session_store.transform_checkpoint(
                session.id,
                tool_round_recovery.assistant_publication_redactor_transform(
                    tool_round_identity=tool_round_identity,
                    tool_call_id=tool_call.id,
                    redactor=snapshot.redactor,
                ),
            )

        raw_workspace = _workspace(registered_environment)
        raw_artifact_store = _artifact_store(registered_environment)
        direct_workspace_mutations = DirectWorkspaceMutationCollector()
        workspace_mutation_owner = (
            InvocationWorkspaceMutationOwner(
                on_settlement_unproven=(
                    registered_environment.workspace_mutation_fence.fail_closed
                ),
            )
            if (
                registered_environment is not None
                and registered_tool.workspace_mutation
                and raw_workspace is not None
            )
            else None
        )
        (
            workspace_receipt_artifact_store,
            workspace_receipt_artifact_unavailable_detail,
        ) = _workspace_receipt_artifact_store(registered_environment)
        tool_context = ToolContext(
            session_id=session.id,
            agent_name=registered_agent.spec.name,
            environment_name=environment_name,
            causal_budget_id=session.causal_budget_id,
            workspace_id=_workspace_id(registered_environment),
            artifact_store_id=_artifact_store_id(registered_environment),
            idempotency_key=idempotency_key,
            workspace=invocation_workspace_handle(
                raw_workspace,
                redactor_snapshot_provider=invocation_secret_scope.snapshot,
                capture_observer=invocation_secret_scope.record_ambiguous_output_capture,
                mutation_owner=workspace_mutation_owner,
                direct_mutation_observer=direct_workspace_mutations.record,
            ),
            artifact_store=invocation_artifact_store_handle(
                raw_artifact_store,
                redactor_snapshot_provider=invocation_secret_scope.snapshot,
                capture_observer=invocation_secret_scope.record_ambiguous_output_capture,
            ),
            runner=invocation_runner_handle(
                _runner(registered_environment),
                redactor_snapshot_provider=invocation_secret_scope.snapshot,
                ambiguous_capture_observer=(
                    invocation_secret_scope.record_ambiguous_output_capture
                ),
                mutation_owner=workspace_mutation_owner,
                execution_observer=observe_runner_execution,
                publish_execution_arguments=registered_tool.publish_arguments,
            ),
            invocation_secret_redactor=redactor_provider,
            invocation_secret_snapshot_provider=invocation_secret_scope.snapshot,
            invocation_secret_capture_observer=(
                invocation_secret_scope.record_ambiguous_output_capture
            ),
            vault=invocation_secrets.vault_for_environment(
                registered_environment,
                tracker=invocation_secret_scope,
                on_redactor_change=persist_resolved_secret_projection,
            ),
            proxy=invocation_secrets.proxy_for_environment(
                registered_environment,
                tracker=invocation_secret_scope,
                on_authorize=proxy_authorizations.append,
                on_redactor_change=persist_resolved_secret_projection,
            ),
            knowledge_store=_knowledge_store(registered_environment),
            knowledge_access_scope=_knowledge_access_scope(registered_environment),
            mcp_servers=_mcp_servers(registered_environment),
            metadata=ctx_metadata,
        )
        tool_context._bind_runtime_resource_authorities(
            workspace=raw_workspace,
            artifact_store=raw_artifact_store,
        )
        if effective_tool_call.name == SEARCH_TOOLS_NAME:
            if registered_agent.tool_discovery_mode is None:
                raise RuntimeError("search_tools execution requires enabled tool discovery.")
            _bind_runtime_tool_discovery_authority(
                tool_context,
                generation_id=tool_discovery_generation_id(
                    session_id=session.id,
                    root_invocation_id=session.invocation.root_invocation_id,
                ),
                catalogue=registered_agent.tool_catalogue,
                ceiling=tool_capability_ceiling_from_session_metadata(session.metadata),
                directly_exposed_names=(
                    registered_agent.tools if tool_exposure is None else tool_exposure.tool_names
                ),
                model_step_id=tool_round_identity.model_step_id,
                created_at=self._clock(),
            )
        if execution_profile is not None:

            async def load_durable_operation(storage_key: str) -> dict[str, Any] | None:
                return await self._session_store.load_session_operation(
                    session.id,
                    storage_key,
                )

            async def authorize_shared_artifact(
                reference: dict[str, Any],
                policy_fingerprint: str,
                observed_at: str,
            ) -> dict[str, Any]:
                from cayu.tools.shared_artifacts import authorize_shared_artifact_materialization

                return await authorize_shared_artifact_materialization(
                    session_store=self._session_store,
                    caller_session_id=session.id,
                    caller_session_instance_id=session.instance_id,
                    reference=reference,
                    policy_fingerprint=policy_fingerprint,
                    observed_at=observed_at,
                )

            async def compare_and_set_durable_operation(
                storage_key: str,
                expected: dict[str, Any] | None,
                desired: dict[str, Any],
                secondary_records: Mapping[str, dict[str, Any]],
            ) -> dict[str, Any]:
                expected_copy = (
                    None
                    if expected is None
                    else copy_durable_json_object(expected, "durable_tool_operation.expected")
                )
                desired_copy = copy_durable_json_object(
                    desired,
                    "durable_tool_operation.desired",
                )
                secondary_copy = {
                    key: copy_durable_json_object(
                        value,
                        f"durable_tool_operation.secondary[{key!r}]",
                    )
                    for key, value in secondary_records.items()
                }
                if storage_key in secondary_copy:
                    raise ValueError("A durable tool operation cannot duplicate its primary key.")

                def publish(
                    current_session: Session,
                    checkpoint: dict[str, Any] | None,
                    current: dict[str, Any] | None,
                ) -> SessionOperationPublication:
                    if (
                        current_session.id != session.id
                        or current_session.run_epoch != session.run_epoch
                    ):
                        raise SessionRunFenced(
                            "Durable tool operation lost its parent run authority."
                        )
                    if current != expected_copy:
                        raise DurableToolOperationConflict(
                            "Durable tool operation changed before publication."
                        )
                    records = {storage_key: desired_copy, **secondary_copy}
                    return SessionOperationPublication(
                        checkpoint={} if checkpoint is None else checkpoint,
                        operation_records=records,
                    )

                publication = asyncio.create_task(
                    self._session_store.publish_session_operation(
                        session.id,
                        idempotency_key=storage_key,
                        operation_transform=publish,
                        events=[],
                        expected_statuses={SessionStatus.RUNNING},
                        expected_run_epoch=session.run_epoch,
                    )
                )
                outcome = await await_shielded_task_outcome(publication)
                publication_error = outcome.error
                if publication_error is not None:
                    persisted = await self._session_store.load_session_operation(
                        session.id,
                        storage_key,
                    )
                    if persisted != desired_copy:
                        if outcome.cancellation is not None:
                            outcome.cancellation.add_note(
                                "Durable tool operation publication also failed: "
                                f"{type(publication_error).__name__}."
                            )
                            restore_task_cancellation_requests(
                                outcome.cancellation_requests_consumed,
                                cancellation=outcome.cancellation,
                            )
                            raise outcome.cancellation from publication_error
                        raise publication_error
                    for key, value in secondary_copy.items():
                        if (
                            await self._session_store.load_session_operation(
                                session.id,
                                key,
                            )
                            != value
                        ):
                            inconsistency = RuntimeError(
                                "Durable tool operation acknowledgement is inconsistent."
                            )
                            if outcome.cancellation is not None:
                                restore_task_cancellation_requests(
                                    outcome.cancellation_requests_consumed,
                                    cancellation=outcome.cancellation,
                                )
                                raise outcome.cancellation from inconsistency
                            raise inconsistency from publication_error
                if outcome.cancellation is not None:
                    restore_task_cancellation_requests(
                        outcome.cancellation_requests_consumed,
                        cancellation=outcome.cancellation,
                    )
                    raise outcome.cancellation
                return copy_durable_json_object(
                    desired_copy,
                    "durable_tool_operation.result",
                )

            def seal_durable_output(record: dict[str, Any]) -> dict[str, Any]:
                snapshot = invocation_secret_scope.seal_for_publication()
                if snapshot.unsafe_output or snapshot.secret_scope_incomplete:
                    raise RuntimeError(
                        "Durable tool output requires a complete invocation secret scope."
                    )
                redacted = snapshot.redactor.redact_json_values(
                    copy_durable_json_object(record, "durable_tool_output")
                )
                if type(redacted) is not dict:
                    raise TypeError("Durable tool output must remain an object after redaction.")
                return redacted

            _bind_runtime_tool_invocation_authority(
                tool_context,
                parent_task_id=task_id,
                parent_run_epoch=session.run_epoch,
                model_step_id=tool_round_identity.model_step_id,
                model_attempt_id=tool_round_identity.model_attempt_id,
                tool_round_id=tool_round_identity.tool_round_id,
                tool_call_id=effective_tool_call.id,
                tool_name=effective_tool_call.name,
                idempotency_key=idempotency_key,
                effective_arguments=effective_tool_call.arguments,
                execution_profile_fingerprint=execution_profile.fingerprint,
                environment_allocation_fingerprint=(
                    None
                    if registered_environment is None
                    else registered_environment.live_allocation_fingerprint
                ),
                current_session_lineage={
                    "session_id": session.id,
                    "session_instance_id": session.instance_id,
                    "parent_session_id": session.parent_session_id,
                    "causal_budget_id": session.causal_budget_id,
                    "invocation": session.invocation.model_dump(mode="json"),
                },
                load_durable_operation=load_durable_operation,
                authorize_shared_artifact=authorize_shared_artifact,
                compare_and_set_durable_operation=(compare_and_set_durable_operation),
                seal_durable_output=seal_durable_output,
                secret_publication_sealer=invocation_secret_scope.seal_for_publication,
            )
        workspace_window_id: str | None = None
        workspace_attribution_window: WorkspaceMutationWindow | None = None
        writer_isolation_before = WorkspaceWriterIsolationEvidence()
        before_workspace_observation: WorkspaceRevisionObservation | None = None
        workspace_lifecycle: WorkspaceObservationLifecycle | None = None
        if registered_tool.workspace_mutation and _workspace(registered_environment) is not None:
            if registered_tool.parallel_safe:
                raise RuntimeError("Workspace-mutating tools must execute exclusively.")
            if deferred_terminal_stager is None or deferred_terminal_capture_recorder is None:
                raise RuntimeError("Workspace-mutating tools require durable terminal staging.")
            if registered_environment is None:  # pragma: no cover - narrowed by workspace lookup
                raise AssertionError("Workspace mutation lost its registered environment.")
            workspace_window_id = _workspace_mutation_window_id(
                session_id=session.id,
                session_run_epoch=session.run_epoch,
                tool_round_id=tool_round_identity.tool_round_id,
                tool_call_id=tool_call.id,
            )
            configured_workspace_id = _workspace_id(registered_environment)
            configured_observer = _workspace_revision_observer_name(registered_environment)
            configured_artifact_store_id = _artifact_store_id(registered_environment)
            workspace_authority = _project_workspace_observation_authority(
                session_id=session.id,
                configured_workspace_id=configured_workspace_id,
                configured_observer=configured_observer,
                configured_artifact_store_id=configured_artifact_store_id,
                observer_is_runtime_owned=(
                    _workspace_revision_observer_is_runtime_owned(registered_environment)
                ),
                secret_resolution_scope=(
                    invocation_secrets.registered_environment_secret_resolution_scope(
                        registered_environment
                    )
                ),
                redactor=self._secret_redactor,
                public_authority_alias_codec=(self._session_store.public_authority_alias_codec),
            )
            workspace_lifecycle = WorkspaceObservationLifecycle(
                session_id=session.id,
                interaction_id=(None if started_event is None else started_event.interaction_id),
                window_id=workspace_window_id,
                source_run_epoch=session.run_epoch,
                binding_generation_id=_workspace_binding_generation_id(registered_environment),
                workspace_id=workspace_authority.workspace_id,
                observer=workspace_authority.observer,
                observer_authority=workspace_authority.observer_authority,
                artifact_store_id=workspace_authority.artifact_store_id,
                agent_name=registered_agent.spec.name,
                environment_name=environment_name,
                tool_name=tool_call.name,
                tool_call_id=tool_call.id,
                model_step_id=tool_round_identity.model_step_id,
                model_attempt_id=tool_round_identity.model_attempt_id,
                tool_round_id=tool_round_identity.tool_round_id,
                model_step=model_step,
            )
            await publish_workspace_observation_transition(
                session_store=self._session_store,
                event_writer=self._event_writer,
                session=session,
                previous=None,
                current=workspace_lifecycle,
                phase="intent",
                intent_admission=_admit_workspace_observation_intent(
                    workspace_lifecycle,
                    redactor=self._secret_redactor,
                    configured_workspace_id=configured_workspace_id,
                    configured_artifact_store_id=configured_artifact_store_id,
                    authority_projection=workspace_authority,
                ),
            )
            workspace_attribution_window = begin_workspace_mutation_window(
                raw_workspace,
                window_id=workspace_window_id,
            )
            writer_isolation_before = _workspace_writer_isolation(registered_environment)
            try:
                before_workspace_observation = await _observe_workspace_revision(
                    registered_environment,
                    operation_registry=self._workspace_capture_operations,
                    require_quiescence_before_return=True,
                    authority_projection=workspace_authority,
                )
                captured_before_state = (
                    WorkspaceObservationEvidenceState.FAILED
                    if before_workspace_observation.status
                    is WorkspaceRevisionObservationStatus.FAILED
                    else WorkspaceObservationEvidenceState.CAPTURED_PRIVATE
                )
                before_captured_lifecycle = WorkspaceObservationLifecycle.model_validate(
                    {
                        **workspace_lifecycle.model_dump(mode="json"),
                        "phase": WorkspaceObservationPhase.BEFORE_CAPTURED.value,
                        "before_state": captured_before_state.value,
                    }
                )
                await publish_workspace_observation_transition(
                    session_store=self._session_store,
                    event_writer=self._event_writer,
                    session=session,
                    previous=workspace_lifecycle,
                    current=before_captured_lifecycle,
                    phase="before-capture",
                )
            except BaseException:
                workspace_attribution_window.close()
                raise
            workspace_lifecycle = before_captured_lifecycle

        async def close_workspace_mutation_window() -> tuple[Event, ...]:
            if workspace_window_id is None or before_workspace_observation is None:
                return ()
            if workspace_mutation_owner is not None:
                await workspace_mutation_owner.seal_and_wait()
            return ()

        def retain_workspace_settlement_failure(interrupt: BaseException) -> None:
            safe_failure = RuntimeError(
                "Workspace mutation settlement failed after caller cancellation."
            )
            prior_cause = exception_cause(interrupt)
            cause: BaseException = safe_failure
            if prior_cause is not None:
                cause = BaseExceptionGroup(
                    "Caller cancellation and workspace mutation settlement failures.",
                    [prior_cause, safe_failure],
                )
            if isinstance(interrupt, asyncio.CancelledError):
                attach_runner_cancellation_failure(interrupt, cause)
            elif isinstance(interrupt, BaseExceptionGroup):
                for candidate in iter_exception_tree(interrupt):
                    if isinstance(candidate, asyncio.CancelledError):
                        attach_runner_cancellation_failure(candidate, cause)
                        set_exception_cause(candidate, cause)
            set_exception_cause(interrupt, cause)

        async def close_workspace_mutation_window_after_interrupt(
            interrupt: BaseException,
            cancellation: asyncio.CancelledError,
        ) -> None:
            nonlocal workspace_capture_failure_detail, workspace_settlement_failure
            outcome = await await_invocation_operation(
                close_workspace_mutation_window,
                request_child_cancellation=False,
                cancellation=cancellation,
                on_unsettled_supervisory_exit=(
                    workspace_mutation_owner.fail_closed
                    if workspace_mutation_owner is not None
                    else None
                ),
            )
            if _contains_process_signal(outcome.error):
                if outcome.error is None:  # pragma: no cover - narrowed by the helper
                    raise AssertionError("Process-signal classification lost its error.")
                raise outcome.error
            if outcome.error is not None:
                if any(
                    isinstance(candidate, WorkspaceMutationSettlementError)
                    for candidate in iter_exception_tree(outcome.error)
                ):
                    workspace_settlement_failure = WorkspaceMutationSettlementError(
                        "Workspace mutation settlement could not be proven."
                    )
                else:
                    workspace_capture_failure_detail = "receipt_publication_failed"
                retain_workspace_settlement_failure(interrupt)

        workspace_events: tuple[Event, ...] = ()
        post_tool_cancellation: asyncio.CancelledError | None = None
        post_tool_cancellation_requests_consumed = 0
        workspace_settlement_failure: WorkspaceMutationSettlementError | None = None
        workspace_capture_failure_detail: str | None = None
        workspace_capture_payload: dict[str, str] = {}

        def consume_post_tool_cancellation(
            cancellation: asyncio.CancelledError,
        ) -> asyncio.CancelledError | None:
            nonlocal post_tool_cancellation_requests_consumed
            current_task = asyncio.current_task()
            requests_before = 0 if current_task is None else current_task.cancelling()
            observed = consume_pending_task_cancellation(cancellation)
            if current_task is not None:
                post_tool_cancellation_requests_consumed += max(
                    requests_before - current_task.cancelling(),
                    0,
                )
            return observed

        effective_terminal_stager = deferred_terminal_stager
        if workspace_lifecycle is not None:
            if (
                deferred_terminal_stager is None
                or deferred_terminal_capture_recorder is None
                or workspace_window_id is None
                or workspace_attribution_window is None
                or before_workspace_observation is None
            ):
                raise RuntimeError("Workspace observation lost its durable staging boundary.")

            async def stage_workspace_terminal(
                event: Event,
                outcome: runtime_records.ToolCallOutcome,
                allow_modification: bool,
                publish_before_hooks: bool,
                snapshot: invocation_secrets.InvocationPublicationSnapshot,
            ) -> Event:
                nonlocal post_tool_cancellation
                nonlocal workspace_events, workspace_lifecycle, workspace_capture_payload

                staged_event = await deferred_terminal_stager(
                    event,
                    outcome,
                    allow_modification,
                    publish_before_hooks,
                    snapshot,
                )
                current_workspace_lifecycle = workspace_lifecycle
                if current_workspace_lifecycle is None:
                    raise RuntimeError(
                        "Workspace observation lifecycle disappeared before staging."
                    )
                staged_lifecycle = WorkspaceObservationLifecycle.model_validate(
                    {
                        **current_workspace_lifecycle.model_dump(mode="json"),
                        "phase": WorkspaceObservationPhase.TOOL_OUTCOME_STAGED.value,
                        "tool_outcome_event_id": staged_event.id,
                        "tool_outcome_event_digest": workspace_observation_event_digest(
                            staged_event
                        ),
                    }
                )
                await publish_workspace_observation_transition(
                    session_store=self._session_store,
                    event_writer=self._event_writer,
                    session=session,
                    previous=current_workspace_lifecycle,
                    current=staged_lifecycle,
                    phase="tool-outcome",
                )
                workspace_lifecycle = staged_lifecycle

                capture: _WorkspaceCaptureResult | None = None
                capture_failure_detail: str | None = None
                workspace_evidence_available = (
                    not snapshot.unsafe_output and not publish_arguments_as_unavailable
                )
                if workspace_settlement_failure is not None:
                    capture_failure_detail = "mutation_settlement_unproven"
                elif workspace_capture_failure_detail is not None:
                    capture_failure_detail = workspace_capture_failure_detail
                else:
                    try:
                        capture = await _record_workspace_mutation_after(
                            session_store=self._session_store,
                            event_writer=self._event_writer,
                            registered_environment=registered_environment,
                            artifact_store=workspace_receipt_artifact_store,
                            artifact_unavailable_detail_code=(
                                workspace_receipt_artifact_unavailable_detail
                            ),
                            session=session,
                            registered_agent=registered_agent,
                            environment_name=environment_name,
                            tool_call=tool_call,
                            tool_round_identity=tool_round_identity,
                            execution_profile=execution_profile,
                            model_step=model_step,
                            window_id=workspace_window_id,
                            lifecycle=workspace_lifecycle,
                            before_observation=before_workspace_observation,
                            authority_projection=workspace_authority,
                            attribution_window=workspace_attribution_window,
                            writer_isolation_before=writer_isolation_before,
                            direct_mutations=direct_workspace_mutations,
                            redactor=snapshot.redactor,
                            evidence_available=workspace_evidence_available,
                            operation_registry=self._workspace_capture_operations,
                        )
                    except (KeyboardInterrupt, SystemExit, GeneratorExit):
                        raise
                    except asyncio.CancelledError as exc:
                        observed_cancellation = consume_post_tool_cancellation(exc)
                        if observed_cancellation is None:
                            raise
                        publication_failure = exception_cause(observed_cancellation)
                        if publication_failure is not None:
                            # The parallel tool-round boundary deliberately
                            # severs ordinary exception causes before it
                            # rebuilds the caller cancellation. Carry this
                            # runtime-owned publication failure through its
                            # authenticated cleanup channel so both the
                            # initial and reconciliation failures survive in
                            # bounded form on the final cancellation.
                            prior_failure = runner_cancellation_failure(observed_cancellation)
                            cancellation_failure = publication_failure
                            if (
                                prior_failure is not None
                                and prior_failure is not publication_failure
                            ):
                                cancellation_failure = BaseExceptionGroup(
                                    "Tool cancellation and workspace publication failures.",
                                    [prior_failure, publication_failure],
                                )
                            attach_runner_cancellation_failure(
                                observed_cancellation,
                                cancellation_failure,
                            )
                        invocation_secrets.initialize_cancellation_evidence(observed_cancellation)
                        invocation_secrets.set_cancellation_redactor(
                            observed_cancellation,
                            snapshot.redactor,
                        )
                        invocation_secrets.set_cancellation_tool_call_id(
                            observed_cancellation,
                            tool_call.id,
                        )
                        post_tool_cancellation = observed_cancellation
                        capture_failure_detail = "receipt_publication_interrupted"
                    except Exception:
                        capture_failure_detail = "receipt_publication_failed"
                    finally:
                        if workspace_attribution_window is not None:
                            workspace_attribution_window.close(
                                discard_history=not workspace_evidence_available
                            )

                if workspace_attribution_window is not None:
                    workspace_attribution_window.close(
                        discard_history=not workspace_evidence_available
                    )

                if capture is None:
                    if capture_failure_detail == "receipt_publication_interrupted":
                        workspace_capture_payload = {
                            "workspace_mutation_capture_status": "interrupted",
                            "workspace_mutation_capture_detail_code": capture_failure_detail,
                        }
                    else:
                        workspace_capture_payload = {
                            "workspace_mutation_capture_status": "failed",
                            "workspace_mutation_capture_detail_code": (
                                capture_failure_detail or "receipt_publication_failed"
                            ),
                        }
                else:
                    workspace_capture_payload = {
                        "workspace_mutation_capture_status": "recorded",
                    }

                if capture is None:
                    final_payload = dict(staged_event.payload)
                    final_payload.update(workspace_capture_payload)
                    finalized_stage = await deferred_terminal_capture_recorder(
                        staged_event.model_copy(update={"payload": final_payload}, deep=True)
                    )
                    if (
                        finalized_stage.id != staged_event.id
                        or workspace_observation_event_digest(finalized_stage)
                        != staged_lifecycle.tool_outcome_event_digest
                    ):
                        raise RuntimeError(
                            "Workspace capture changed the authoritative tool outcome."
                        )
                    checkpoint = await await_workspace_observation_store_read(
                        lambda: self._session_store.load_checkpoint(session.id),
                        operation="Workspace observation failure-closure checkpoint read",
                    )
                    current_lifecycle = workspace_observations_from_checkpoint(checkpoint).get(
                        workspace_window_id
                    )
                    if current_lifecycle is None:
                        raise RuntimeError(
                            "Workspace observation lifecycle disappeared before failure closure."
                        )
                    terminal_lifecycle = _workspace_observation_terminal_view(current_lifecycle)
                    incomplete_event = prepare_runtime_event(
                        _workspace_mutation_incomplete_event(
                            lifecycle=terminal_lifecycle,
                            session=session,
                            execution_profile=execution_profile,
                            status=(
                                WorkspaceObservationTerminalStatus.INCOMPLETE
                                if capture_failure_detail == "receipt_publication_interrupted"
                                else WorkspaceObservationTerminalStatus.FAILED
                            ),
                            detail_code=(capture_failure_detail or "receipt_publication_failed"),
                        ),
                        redactor=snapshot.redactor,
                    )
                    published = await publish_workspace_observation_transition(
                        session_store=self._session_store,
                        event_writer=self._event_writer,
                        session=session,
                        previous=current_lifecycle,
                        current=None,
                        phase="terminal",
                        terminal_status=(
                            WorkspaceObservationTerminalStatus.INCOMPLETE
                            if capture_failure_detail == "receipt_publication_interrupted"
                            else WorkspaceObservationTerminalStatus.FAILED
                        ),
                        terminal_detail_code=(
                            capture_failure_detail or "receipt_publication_failed"
                        ),
                        terminal_artifacts=terminal_lifecycle.artifacts,
                        events=(incomplete_event,),
                    )
                    workspace_events = published
                    workspace_lifecycle = current_lifecycle
                    return finalized_stage

                workspace_lifecycle = capture.lifecycle
                delta_state = (
                    WorkspaceObservationEvidenceState.PUBLISHED
                    if capture.terminal_status is not WorkspaceObservationTerminalStatus.FAILED
                    else WorkspaceObservationEvidenceState.FAILED
                )
                delta_published = WorkspaceObservationLifecycle.model_validate(
                    {
                        **capture.delta_lifecycle.model_dump(mode="json"),
                        "phase": WorkspaceObservationPhase.DELTA_PUBLISHED.value,
                        "delta_state": delta_state.value,
                        "mutation_event_id": capture.receipt_event.id,
                        "mutation_event_digest": workspace_observation_event_digest(
                            capture.receipt_event
                        ),
                    }
                )
                (receipt_event,) = await publish_workspace_observation_transition(
                    session_store=self._session_store,
                    event_writer=self._event_writer,
                    session=session,
                    previous=capture.lifecycle,
                    current=delta_published,
                    phase="delta-publication",
                    events=(capture.receipt_event,),
                )
                final_payload = dict(staged_event.payload)
                final_payload.update(workspace_capture_payload)
                finalized_stage = await deferred_terminal_capture_recorder(
                    staged_event.model_copy(update={"payload": final_payload}, deep=True)
                )
                if (
                    finalized_stage.id != staged_event.id
                    or workspace_observation_event_digest(finalized_stage)
                    != staged_lifecycle.tool_outcome_event_digest
                ):
                    raise RuntimeError("Workspace capture changed the authoritative tool outcome.")
                terminal_lifecycle = _workspace_observation_terminal_view(delta_published)
                finalized_event = prepare_runtime_event(
                    _workspace_mutation_incomplete_event(
                        lifecycle=terminal_lifecycle,
                        session=session,
                        execution_profile=execution_profile,
                        status=capture.terminal_status,
                        detail_code=capture.terminal_detail_code,
                    ),
                    redactor=snapshot.redactor,
                )
                (finalized_event,) = await publish_workspace_observation_transition(
                    session_store=self._session_store,
                    event_writer=self._event_writer,
                    session=session,
                    previous=delta_published,
                    current=None,
                    phase="terminal",
                    terminal_status=capture.terminal_status,
                    terminal_detail_code=capture.terminal_detail_code,
                    terminal_artifacts=terminal_lifecycle.artifacts,
                    events=(finalized_event,),
                )
                workspace_lifecycle = delta_published
                workspace_events = (*capture.events, receipt_event, finalized_event)
                return finalized_stage

            effective_terminal_stager = stage_workspace_terminal

        async def stage_interrupted_workspace_outcome(
            interrupt: BaseException,
            *,
            artifacts: list[dict[str, Any]] | None,
        ) -> None:
            """Stage and close exact workspace evidence before re-raising interruption."""

            if workspace_lifecycle is None:
                return
            if effective_terminal_stager is None:
                raise RuntimeError("Workspace interruption lost its terminal staging boundary.")
            interrupted_outcome = _interrupted_tool_call_outcome(
                tool_call=tool_call,
                tool_round_identity=tool_round_identity,
                registered_tool=registered_agent.executable_tool(tool_call.name),
                execution_started=True,
                artifacts=artifacts,
            )
            interrupted_event = _interrupted_tool_call_event(
                session=session,
                registered_agent=registered_agent,
                registered_environment=registered_environment,
                tool_call_outcome=interrupted_outcome,
                tool_round_identity=tool_round_identity,
            )
            try:
                await effective_terminal_stager(
                    interrupted_event,
                    interrupted_outcome,
                    False,
                    False,
                    invocation_secret_scope.seal_for_publication(),
                )
            except BaseException as closure_error:
                if _contains_process_signal(closure_error):
                    interrupt_nodes = {
                        id(candidate) for candidate in iter_exception_tree(interrupt)
                    }
                    closure_nodes = {
                        id(candidate) for candidate in iter_exception_tree(closure_error)
                    }
                    if interrupt_nodes & closure_nodes:
                        raise
                    raise BaseExceptionGroup(
                        "Tool interruption and workspace observation process control.",
                        [interrupt, closure_error],
                    ) from None
                # The original interruption remains authoritative. Recovery
                # can reconcile an intent or partial phase without rerunning
                # the tool, so attach only a fixed bounded diagnostic here.
                failure: BaseException = RuntimeError(
                    "Workspace observation closure failed during tool interruption."
                )
                prior = exception_cause(interrupt)
                if prior is not None:
                    failure = BaseExceptionGroup(
                        "Tool interruption and workspace observation closure failures.",
                        [prior, failure],
                    )
                set_exception_cause(interrupt, failure)

        try:
            execution_outcome = await tool_execution.run_tool(
                tool=registered_tool.tool,
                effect=registered_tool.effect,
                ctx=tool_context,
                arguments=effective_tool_call.arguments,
                redactor=lambda: invocation_secret_scope.redactor,
                registered_schema=registered_tool.schema,
                registered_execution_contract=registered_tool.execution_contract,
                finalize_publication=invocation_secret_scope.seal_for_publication,
                timeout_seconds=self._tool_timeout_seconds,
            )
        except BaseExceptionGroup as exc:
            if any(
                isinstance(candidate, (KeyboardInterrupt, SystemExit, GeneratorExit))
                for candidate in iter_exception_tree(exc)
            ):
                if workspace_mutation_owner is not None:
                    workspace_mutation_owner.seal_and_transfer_pending()
            elif is_current_runner_cancellation_group(exc):
                group_cancellation: asyncio.CancelledError | None = None
                for candidate in iter_exception_tree(exc):
                    if not isinstance(candidate, asyncio.CancelledError):
                        continue
                    if group_cancellation is None:
                        group_cancellation = candidate
                    invocation_secrets.initialize_cancellation_evidence(candidate)
                    invocation_secrets.set_cancellation_redactor(
                        candidate,
                        invocation_secret_scope.redactor,
                    )
                    invocation_secrets.set_cancellation_tool_call_id(
                        candidate,
                        tool_call.id,
                    )
                if group_cancellation is None:  # pragma: no cover - classification invariant
                    raise AssertionError(
                        "Current cancellation group has no cancellation leaf."
                    ) from None
                await close_workspace_mutation_window_after_interrupt(
                    exc,
                    group_cancellation,
                )
                await record_interrupted_publication_scope(exc)
                (
                    grouped_artifacts,
                    grouped_artifacts_by_id,
                    _grouped_redactors,
                ) = _grouped_cancellation_evidence(exc, tool_calls=[tool_call])
                await stage_interrupted_workspace_outcome(
                    exc,
                    artifacts=(
                        grouped_artifacts_by_id.get(tool_call.id, [])
                        if grouped_artifacts_by_id is not None
                        else grouped_artifacts
                    ),
                )
            raise
        except asyncio.CancelledError as exc:
            invocation_secrets.initialize_cancellation_evidence(exc)
            invocation_secrets.set_cancellation_redactor(
                exc,
                invocation_secret_scope.redactor,
            )
            invocation_secrets.set_cancellation_tool_call_id(exc, tool_call.id)
            await close_workspace_mutation_window_after_interrupt(exc, exc)
            cancellation_redelivered = await record_interrupted_publication_scope(exc)
            await stage_interrupted_workspace_outcome(
                exc,
                artifacts=invocation_secrets.cancellation_artifacts(exc),
            )
            if cancellation_redelivered:
                raise
            if proxy_authorizations and await self._session_control.interrupt_requested(session.id):
                clear_current_task_cancellation()
                async for event in self._emit_proxy_authorization_events(
                    session=session,
                    registered_agent=registered_agent,
                    registered_environment=registered_environment,
                    tool_call=tool_call,
                    records=proxy_authorizations,
                    tool_round_identity=tool_round_identity,
                    execution_profile=execution_profile,
                    approval_id=approval_id,
                    input_id=input_id,
                    idempotency_key=idempotency_key,
                    redactor=invocation_secret_scope.redactor,
                    output_redactor=invocation_secret_scope.redactor,
                    quarantine_output=deferred_terminal_stager is not None,
                ):
                    yield event, None
            raise
        except (KeyboardInterrupt, SystemExit, GeneratorExit):
            if workspace_mutation_owner is not None:
                workspace_mutation_owner.seal_and_transfer_pending()
            raise

        result = execution_outcome.result
        redactor = invocation_secret_scope.redactor
        output_redactor = redactor
        if workspace_window_id is not None:
            try:
                await close_workspace_mutation_window()
            except BaseExceptionGroup as exc:
                if _contains_process_signal(exc) or is_current_runner_cancellation_group(exc):
                    raise
                workspace_capture_failure_detail = "receipt_publication_failed"
                workspace_capture_payload = {
                    "workspace_mutation_capture_status": "failed",
                    "workspace_mutation_capture_detail_code": workspace_capture_failure_detail,
                }
            except asyncio.CancelledError as exc:
                post_tool_cancellation = consume_post_tool_cancellation(exc)
                if post_tool_cancellation is not None:
                    invocation_secrets.initialize_cancellation_evidence(post_tool_cancellation)
                    invocation_secrets.set_cancellation_redactor(
                        post_tool_cancellation,
                        invocation_secret_scope.redactor,
                    )
                    invocation_secrets.set_cancellation_tool_call_id(
                        post_tool_cancellation,
                        tool_call.id,
                    )
                workspace_capture_payload = {
                    "workspace_mutation_capture_status": "interrupted",
                    "workspace_mutation_capture_detail_code": "receipt_publication_interrupted",
                }
            except Exception as exc:
                if any(
                    isinstance(candidate, WorkspaceMutationSettlementError)
                    for candidate in iter_exception_tree(exc)
                ):
                    workspace_settlement_failure = WorkspaceMutationSettlementError(
                        "Workspace mutation settlement could not be proven."
                    )
                    workspace_capture_payload = {
                        "workspace_mutation_capture_status": "failed",
                        "workspace_mutation_capture_detail_code": "mutation_settlement_unproven",
                    }
                else:
                    workspace_capture_failure_detail = "receipt_publication_failed"
                    workspace_capture_payload = {
                        "workspace_mutation_capture_status": "failed",
                        "workspace_mutation_capture_detail_code": workspace_capture_failure_detail,
                    }
            else:
                workspace_capture_payload = {
                    "workspace_mutation_capture_status": "pending",
                }
        publication_snapshot = invocation_secret_scope.seal_for_publication()
        await _await_post_tool_operation(
            persist_sealed_invocation_evidence(publication_snapshot),
            cancellation=post_tool_cancellation,
            restore_cancellation_requests=post_tool_cancellation_requests_consumed,
        )

        hook_argument_projection = (
            tool_argument_publication.unavailable_argument_projection()
            if publish_arguments_as_unavailable
            else tool_argument_publication.finalized_argument_projection(
                effective_tool_call.arguments,
                redactor=publication_snapshot.redactor,
                scope_finalized=not publication_snapshot.unsafe_output,
            )
        )
        argument_projection = (
            tool_argument_publication.unavailable_argument_projection()
            if publish_arguments_as_unavailable
            else tool_argument_publication.finalized_argument_projection(
                model_arguments,
                redactor=publication_snapshot.redactor,
                scope_finalized=not publication_snapshot.unsafe_output,
            )
        )
        policy_denial = tool_context._policy_denial_for(registered_tool.tool)
        if policy_denial is not None:
            # Command-policy refusal occurs before the tool owns a publishable
            # result boundary. Command arguments can contain credentials even
            # when they were not resolved through the invocation secret
            # registry, so retain only explicit unavailability.
            argument_projection = tool_argument_publication.unavailable_argument_projection()
            hook_argument_projection = argument_projection
        result_event: Event | None = None
        published_terminal_event: Event | None = None
        projection_cancellation: asyncio.CancelledError | None = None
        if policy_denial is None:
            event_type = (
                EventType.TOOL_CALL_FAILED if result.is_error else EventType.TOOL_CALL_COMPLETED
            )
            payload = {
                "tool_call_id": tool_call.id,
                "idempotency_key": idempotency_key,
                **argument_projection.payload_fields(),
                tool_argument_publication.ARGUMENTS_EXACT_FIELD: (
                    tool_argument_publication.argument_projection_is_exact(
                        argument_projection,
                        private_arguments=model_arguments,
                    )
                ),
                "result": result.model_dump(),
                **effective_arguments_payload,
                **execution_outcome.terminal_payload_fields(),
                **workspace_capture_payload,
                **identity_payload,
            }
            if approval_id is not None:
                payload["approval_id"] = approval_id
            if input_id is not None:
                payload["input_id"] = input_id
            result_event = Event(
                type=event_type,
                session_id=session.id,
                agent_name=registered_agent.spec.name,
                environment_name=environment_name,
                tool_name=tool_call.name,
                payload=payload,
            )
            result_event = _restore_targeted_tool_invocation_event_authority(
                result_event,
                effective_tool_call,
                redactor=publication_snapshot.redactor,
            )
            if execution_outcome.publish_before_hooks and deferred_terminal_stager is None:
                result_event, result = _prepare_tool_result_event(
                    event=result_event,
                    result=result,
                    redactor=output_redactor,
                    runtime_tool=registered_tool.tool,
                )
                if not output_redactor.has_same_registry(redactor):
                    result_event, result = _prepare_tool_result_event(
                        event=result_event,
                        result=result,
                        redactor=redactor,
                    )
                (
                    result_event,
                    result,
                    projection_cancellation,
                ) = await _await_post_tool_operation(
                    self._project_terminal_tool_result(
                        event=result_event,
                        result=result,
                        session=session,
                        registered_environment=registered_environment,
                        tool_call=effective_tool_call,
                    ),
                    cancellation=post_tool_cancellation,
                    restore_cancellation_requests=post_tool_cancellation_requests_consumed,
                )
                result_event = _restore_targeted_tool_invocation_event_authority(
                    result_event,
                    effective_tool_call,
                    redactor=publication_snapshot.redactor,
                )
                published_terminal_event = await _await_post_tool_operation(
                    self._event_writer.emit(result_event),
                    cancellation=post_tool_cancellation,
                    restore_cancellation_requests=post_tool_cancellation_requests_consumed,
                )
                published_terminal_outcome = runtime_records.ToolCallOutcome(
                    call=replace(
                        effective_tool_call,
                        arguments=argument_projection.transcript_arguments(),
                    ),
                    result=result,
                )
                # This terminal is already durable. Expose it before proxy
                # telemetry so a later telemetry failure cannot erase the
                # authoritative tool outcome from the public stream. Receipt
                # events remain ordered immediately before that terminal.
                for runner_event in runner_events:
                    yield runner_event, None
                for workspace_event in workspace_events:
                    yield workspace_event, None
                yield published_terminal_event, published_terminal_outcome
        proxy_events: list[Event] = []
        async for event in _iterate_post_tool_events(
            self._emit_proxy_authorization_events(
                session=session,
                registered_agent=registered_agent,
                registered_environment=registered_environment,
                tool_call=tool_call,
                records=proxy_authorizations,
                tool_round_identity=tool_round_identity,
                execution_profile=execution_profile,
                approval_id=approval_id,
                input_id=input_id,
                idempotency_key=idempotency_key,
                redactor=redactor,
                output_redactor=output_redactor,
                quarantine_output=deferred_terminal_stager is not None,
            ),
            cancellation=post_tool_cancellation,
            restore_cancellation_requests=post_tool_cancellation_requests_consumed,
        ):
            if published_terminal_event is not None:
                yield event, None
            else:
                proxy_events.append(event)
        if projection_cancellation is not None:
            _raise_preserved_post_tool_cancellation(
                post_tool_cancellation,
                projection_cancellation,
                restore_cancellation_requests=post_tool_cancellation_requests_consumed,
            )
        current_task = asyncio.current_task()
        tool_swallowed_cancellation = current_task is not None and current_task.cancelling() > 0
        if tool_swallowed_cancellation and await _await_post_tool_operation(
            self._session_control.is_interrupting(session.id),
            cancellation=post_tool_cancellation,
            restore_cancellation_requests=post_tool_cancellation_requests_consumed,
        ):
            raise SessionInterruptedByRequest(session.id)
        if policy_denial is not None:
            public_policy_denial_reason = output_redactor.redact_text(policy_denial.reason)
            terminal_events: list[tuple[Event, runtime_records.ToolCallOutcome | None]] = []
            terminal_recorded = False
            async for event in _iterate_post_tool_events(
                self._emit_terminal_tool_result(
                    session=session,
                    registered_agent=registered_agent,
                    registered_environment=registered_environment,
                    tool_call=effective_tool_call,
                    event_type=EventType.TOOL_CALL_BLOCKED,
                    result=policy_denial.result,
                    extra_payload={
                        "idempotency_key": idempotency_key,
                        **workspace_capture_payload,
                        **policy_denial_payload_fields(
                            tool_name=effective_tool_call.name,
                            denied_by=policy_denial.denied_by,
                            decision=policy_denial.decision,
                            reason=public_policy_denial_reason,
                            metadata={},
                        ),
                    },
                    task_id=task_id,
                    execution_profile=execution_profile,
                    invocation_context=invocation_context,
                    tool_round_identity=tool_round_identity,
                    approval_id=approval_id,
                    input_id=input_id,
                    allow_modification=False,
                    redactor=redactor,
                    output_redactor=output_redactor,
                    argument_projection=argument_projection,
                    deferred_terminal_stager=effective_terminal_stager,
                    publication_snapshot=publication_snapshot,
                ),
                cancellation=post_tool_cancellation,
                restore_cancellation_requests=post_tool_cancellation_requests_consumed,
            ):
                if terminal_recorded:
                    yield event
                    continue
                terminal_events.append(event)
                if event[1] is None:
                    continue
                terminal_recorded = True
                for runner_event in runner_events:
                    yield runner_event, None
                for workspace_event in workspace_events:
                    yield workspace_event, None
                for proxy_event in proxy_events:
                    yield proxy_event, None
                for terminal_event in terminal_events:
                    yield terminal_event
                terminal_events.clear()
            if not terminal_recorded:
                for runner_event in runner_events:
                    yield runner_event, None
                for workspace_event in workspace_events:
                    yield workspace_event, None
                for proxy_event in proxy_events:
                    yield proxy_event, None
                for terminal_event in terminal_events:
                    yield terminal_event
            if post_tool_cancellation is not None:
                _raise_restored_post_tool_cancellation(
                    post_tool_cancellation,
                    restore_cancellation_requests=post_tool_cancellation_requests_consumed,
                )
            if workspace_settlement_failure is not None:
                raise workspace_settlement_failure from None
            return
        if published_terminal_event is not None:
            hook_tool_call = _project_tool_call_for_hook(
                effective_tool_call,
                argument_projection=hook_argument_projection,
                redactor=redactor,
            )
            async for hook_event, modified in _iterate_post_tool_events(
                self.run_tool_call_hooks(
                    session=session,
                    tool_event=published_terminal_event,
                    registered_agent=registered_agent,
                    registered_environment=registered_environment,
                    tool_call=hook_tool_call,
                    result=result,
                    task_id=task_id,
                    execution_profile=execution_profile,
                    invocation_context=invocation_context,
                    redactor=redactor,
                    output_redactor=output_redactor,
                    allow_modification=False,
                    quarantine_output=not registered_tool.publish_arguments,
                ),
                cancellation=post_tool_cancellation,
                restore_cancellation_requests=post_tool_cancellation_requests_consumed,
            ):
                if modified is not None:
                    raise AssertionError(
                        "Observational after-tool hook modified terminal evidence."
                    )
                yield hook_event, None
            if await _await_post_tool_operation(
                self._session_control.is_interrupting(session.id),
                cancellation=post_tool_cancellation,
                restore_cancellation_requests=post_tool_cancellation_requests_consumed,
            ):
                raise SessionInterruptedByRequest(session.id)
            if post_tool_cancellation is not None:
                _raise_restored_post_tool_cancellation(
                    post_tool_cancellation,
                    restore_cancellation_requests=post_tool_cancellation_requests_consumed,
                )
            if workspace_settlement_failure is not None:
                raise workspace_settlement_failure from None
            return
        if result_event is None:
            raise AssertionError("Ordinary tool result event was not constructed.")
        terminal_events: list[tuple[Event, runtime_records.ToolCallOutcome | None]] = []
        terminal_recorded = False
        async for event in _iterate_post_tool_events(
            self.emit_tool_call_result_with_hooks(
                event=result_event,
                session=session,
                registered_agent=registered_agent,
                registered_environment=registered_environment,
                tool_call=effective_tool_call,
                result=result,
                task_id=task_id,
                execution_profile=execution_profile,
                invocation_context=invocation_context,
                redactor=redactor,
                output_redactor=output_redactor,
                argument_projection=argument_projection,
                hook_argument_projection=hook_argument_projection,
                allow_modification=execution_outcome.allows_hook_modification,
                publish_before_hooks=execution_outcome.publish_before_hooks,
                deferred_terminal_stager=effective_terminal_stager,
                publication_snapshot=publication_snapshot,
                executed_runtime_tool=registered_tool.tool,
            ),
            cancellation=post_tool_cancellation,
            restore_cancellation_requests=post_tool_cancellation_requests_consumed,
        ):
            if terminal_recorded:
                yield event
                continue
            terminal_events.append(event)
            if event[1] is None:
                continue
            terminal_recorded = True
            for runner_event in runner_events:
                yield runner_event, None
            for workspace_event in workspace_events:
                yield workspace_event, None
            for proxy_event in proxy_events:
                yield proxy_event, None
            for terminal_event in terminal_events:
                yield terminal_event
            terminal_events.clear()
        if not terminal_recorded:
            for runner_event in runner_events:
                yield runner_event, None
            for workspace_event in workspace_events:
                yield workspace_event, None
            for proxy_event in proxy_events:
                yield proxy_event, None
            for terminal_event in terminal_events:
                yield terminal_event
        if post_tool_cancellation is not None:
            _raise_restored_post_tool_cancellation(
                post_tool_cancellation,
                restore_cancellation_requests=post_tool_cancellation_requests_consumed,
            )
        if await _await_post_tool_operation(
            self._session_control.is_interrupting(session.id),
            cancellation=post_tool_cancellation,
            restore_cancellation_requests=post_tool_cancellation_requests_consumed,
        ):
            raise SessionInterruptedByRequest(session.id)
        if workspace_settlement_failure is not None:
            raise workspace_settlement_failure from None

    async def emit_mcp_manifest_checks(
        self,
        *,
        session: Session,
        registered_agent: runtime_records.RegisteredAgentState,
        environment_name: str | None,
        invocation_context: InvocationContext | None = None,
    ) -> AsyncIterator[Event]:
        if invocation_context is not None and (
            invocation_context.binding.session_id != session.id
            or invocation_context.registered_agent is not registered_agent
            or environment_name
            != (
                None
                if invocation_context.registered_environment is None
                else invocation_context.registered_environment.spec.name
            )
        ):
            raise RuntimeError("MCP manifest validation lost frozen invocation authority.")
        candidates = _mcp_manifest_candidates_for_agent(
            registered_agent,
            environment_name=environment_name,
        )
        if not candidates:
            return

        async def publish_rejection_batch(events: list[Event]) -> list[Event]:
            return await _await_mcp_manifest_operation(
                lambda: self._event_writer.emit_many(session.id, events),
                operation="MCP manifest rejection publication",
            )

        def invalid_baseline_events() -> list[Event]:
            return [
                _mcp_manifest_history_blocked_event(
                    session=session,
                    registered_agent=registered_agent,
                    environment_name=environment_name,
                    history_key=candidate.history_key,
                    snapshot=candidate.snapshot,
                    status="history_conflict",
                    reason="authoritative_baseline_invalid",
                )
                for candidate in candidates
            ]

        if not self._session_store.supports_mcp_manifest_history:
            events = [
                _mcp_manifest_history_blocked_event(
                    session=session,
                    registered_agent=registered_agent,
                    environment_name=environment_name,
                    history_key=candidate.history_key,
                    snapshot=candidate.snapshot,
                    status="history_unavailable",
                    reason="session_store_manifest_history_unsupported",
                )
                for candidate in candidates
            ]
            for event in await publish_rejection_batch(events):
                yield event
            raise McpManifestHistoryConflict(
                f"{type(self._session_store).__name__} does not support durable MCP "
                "manifest history."
            )

        oversized_toolsets = [
            candidate
            for candidate in candidates
            if (
                candidate.snapshot.advertised_tool_count > _MCP_MANIFEST_BASELINE_MAX_TOOLS
                or candidate.snapshot.tool_count > _MCP_MANIFEST_BASELINE_MAX_TOOLS
            )
        ]
        if oversized_toolsets:
            oversized_ids = {id(candidate.toolset) for candidate in oversized_toolsets}
            events = [
                (
                    _mcp_manifest_history_blocked_event(
                        session=session,
                        registered_agent=registered_agent,
                        environment_name=environment_name,
                        history_key=candidate.history_key,
                        snapshot=candidate.snapshot,
                        status="history_conflict",
                        reason="manifest_tool_limit_exceeded",
                    )
                    if id(candidate.toolset) in oversized_ids
                    else _mcp_manifest_batch_blocked_event(
                        session=session,
                        registered_agent=registered_agent,
                        environment_name=environment_name,
                        history_key=candidate.history_key,
                        snapshot=candidate.snapshot,
                        reason="sibling_manifest_tool_limit_exceeded",
                    )
                )
                for candidate in candidates
            ]
            for event in await publish_rejection_batch(events):
                yield event
            raise McpManifestHistoryConflict(
                "MCP toolsets exposed to a run cannot contain more than "
                f"{_MCP_MANIFEST_BASELINE_MAX_TOOLS} manifest tools."
            )

        missing_identity_toolsets = [
            candidate for candidate in candidates if not candidate.snapshot.identity_is_explicit
        ]
        if missing_identity_toolsets:
            missing_ids = {id(candidate.toolset) for candidate in missing_identity_toolsets}
            events = [
                (
                    _mcp_manifest_history_blocked_event(
                        session=session,
                        registered_agent=registered_agent,
                        environment_name=environment_name,
                        history_key=candidate.history_key,
                        snapshot=candidate.snapshot,
                        status="history_conflict",
                        reason="connection_identity_required",
                    )
                    if id(candidate.toolset) in missing_ids
                    else _mcp_manifest_batch_blocked_event(
                        session=session,
                        registered_agent=registered_agent,
                        environment_name=environment_name,
                        history_key=candidate.history_key,
                        snapshot=candidate.snapshot,
                        reason="sibling_connection_identity_missing",
                    )
                )
                for candidate in candidates
            ]
            for event in await publish_rejection_batch(events):
                yield event
            raise McpManifestHistoryConflict(
                "MCP toolsets exposed to a run require an explicit, stable "
                "McpServerSpec.connection_id."
            )

        history_keys = tuple(candidate.history_key for candidate in candidates)
        if len(history_keys) != len(set(history_keys)):
            events = [
                _mcp_manifest_history_blocked_event(
                    session=session,
                    registered_agent=registered_agent,
                    environment_name=environment_name,
                    history_key=candidate.history_key,
                    snapshot=candidate.snapshot,
                    status="history_conflict",
                    reason="duplicate_connection_identity",
                )
                for candidate in candidates
            ]
            for event in await publish_rejection_batch(events):
                yield event
            raise McpManifestHistoryConflict(
                "Multiple MCP toolsets use the same environment and connection identity; "
                "configure distinct McpServerSpec.connection_id values."
            )

        for _ in range(_MCP_MANIFEST_PUBLICATION_MAX_ATTEMPTS):
            try:
                loaded = await _await_mcp_manifest_operation(
                    lambda: self._session_store.load_mcp_manifest_baselines(history_keys),
                    operation="MCP manifest baseline loading",
                )
            except _McpManifestBaselineEvidenceInvalid:
                for event in await publish_rejection_batch(invalid_baseline_events()):
                    yield event
                raise McpManifestHistoryConflict(
                    "The session store returned invalid MCP manifest baseline evidence."
                ) from None
            try:
                baseline_load = _validated_mcp_manifest_baseline_load(
                    loaded,
                    candidates=candidates,
                    environment_name=environment_name,
                )
            except Exception:
                for event in await publish_rejection_batch(invalid_baseline_events()):
                    yield event
                raise McpManifestHistoryConflict(
                    "The session store returned invalid MCP manifest baseline evidence."
                ) from None

            stored_baselines = baseline_load.baselines
            expected_generations = {
                history_key: (
                    None
                    if (baseline := stored_baselines.get(history_key)) is None
                    else baseline.generation
                )
                for history_key in history_keys
            }
            evaluations: list[_McpManifestEvaluation] = []
            for candidate in candidates:
                previous = stored_baselines.get(candidate.history_key)
                status, previous_payload, full_diff = _mcp_manifest_status(
                    snapshot=candidate.snapshot,
                    previous=previous,
                )
                decision = None
                if self._mcp_manifest_policy is not None:
                    decision = self._mcp_manifest_policy.decide(
                        status=status,
                        diff=full_diff,
                    )
                blocked = decision is not None and decision.action == McpManifestPolicyAction.BLOCK
                payload = _mcp_manifest_event_payload(
                    history_key=candidate.history_key,
                    snapshot=candidate.snapshot,
                    status=status,
                    previous=previous_payload,
                    diff=full_diff,
                    decision=decision,
                    outcome="blocked" if blocked else "accepted",
                )
                event = Event(
                    type=(
                        EventType.MCP_MANIFEST_BLOCKED
                        if blocked
                        else EventType.MCP_MANIFEST_CHECKED
                    ),
                    session_id=session.id,
                    agent_name=registered_agent.spec.name,
                    environment_name=environment_name,
                    payload=payload,
                )
                evaluations.append(
                    _McpManifestEvaluation(
                        candidate=candidate,
                        status=status,
                        decision=decision,
                        event=event,
                    )
                )

            blocked = [
                evaluation
                for evaluation in evaluations
                if evaluation.decision is not None
                and evaluation.decision.action == McpManifestPolicyAction.BLOCK
            ]
            publication_events = [evaluation.event for evaluation in evaluations]
            if blocked:
                publication_events = [
                    (
                        evaluation.event
                        if evaluation.decision is not None
                        and evaluation.decision.action == McpManifestPolicyAction.BLOCK
                        else evaluation.event.model_copy(
                            update={
                                "payload": {
                                    **evaluation.event.payload,
                                    "outcome": "batch_blocked",
                                }
                            },
                            deep=True,
                        )
                    )
                    for evaluation in evaluations
                ]
            publication_events = self._event_writer.prepare_many(publication_events)
            baseline_updates: dict[str, McpManifestBaseline] = {}
            if not blocked:
                for evaluation in evaluations:
                    candidate = evaluation.candidate
                    stored = stored_baselines.get(candidate.history_key)
                    if stored is None or evaluation.status == "changed":
                        baseline_updates[candidate.history_key] = _mcp_manifest_baseline(
                            history_key=candidate.history_key,
                            snapshot=candidate.snapshot,
                            generation=1 if stored is None else stored.generation + 1,
                            event=evaluation.event,
                        )

            try:
                publication_result = await _await_mcp_manifest_operation(
                    lambda expected=expected_generations, updates=baseline_updates, events=publication_events: (
                        self._session_store.compare_and_publish_mcp_manifest_checks(
                            session.id,
                            expected_generations=expected,
                            baseline_updates=updates,
                            events=events,
                        )
                    ),
                    operation="MCP manifest decision publication",
                )
            except _McpManifestBaselineEvidenceInvalid:
                for event in await publish_rejection_batch(invalid_baseline_events()):
                    yield event
                raise McpManifestHistoryConflict(
                    "The session store returned invalid MCP manifest baseline evidence."
                ) from None
            expected_published_baselines = {
                **stored_baselines,
                **baseline_updates,
            }
            try:
                publication = _validated_mcp_manifest_publication_result(
                    publication_result,
                    candidates=candidates,
                    environment_name=environment_name,
                    expected_published_baselines=expected_published_baselines,
                )
            except Exception:
                raise McpManifestHistoryConflict(
                    "The session store returned invalid MCP manifest publication evidence."
                ) from None
            if not publication.published:
                continue
            await _await_mcp_manifest_operation(
                lambda events=publication_events: self._event_writer.fan_out_persisted(events),
                operation="MCP manifest event side-effect delivery",
            )
            for event in publication_events:
                yield event.model_copy(deep=True)
            if blocked:
                reasons = "; ".join(
                    evaluation.decision.reason
                    for evaluation in blocked
                    if evaluation.decision is not None
                )
                raise McpManifestPolicyError(reasons)
            return

        conflict_events = [
            Event(
                type=EventType.MCP_MANIFEST_BLOCKED,
                session_id=session.id,
                agent_name=registered_agent.spec.name,
                environment_name=environment_name,
                payload={
                    "history_key": candidate.history_key,
                    "manifest_identity": candidate.snapshot.identity,
                    "manifest_hash": candidate.snapshot.manifest_hash,
                    "source_manifest_hash": candidate.snapshot.source_manifest_hash,
                    "status": "fenced",
                    "change_classes": [],
                    "outcome": "fenced",
                    "reason": "authoritative_baseline_changed_repeatedly",
                },
            )
            for candidate in candidates
        ]
        for event in await publish_rejection_batch(conflict_events):
            yield event
        raise McpManifestHistoryConflict(
            "MCP manifest history changed repeatedly while the run was being admitted."
        )

    async def _emit_proxy_authorization_events(
        self,
        *,
        session: Session,
        registered_agent: runtime_records.RegisteredAgentState,
        registered_environment: runtime_records.RegisteredEnvironment | None,
        tool_call: runtime_records.ToolCallRequest,
        records: list[invocation_secrets.ProxyAuthorizationRecord],
        tool_round_identity: ToolRoundIdentity,
        execution_profile: ExecutionProfileIdentity | None,
        approval_id: str | None,
        input_id: str | None,
        redactor: SecretRedactor,
        output_redactor: SecretRedactor,
        quarantine_output: bool = False,
        idempotency_key: str | None = None,
    ) -> AsyncIterator[Event]:
        identity_payload = copy_tool_round_identity(tool_round_identity).payload()
        for record in records:
            payload: dict[str, Any] = {
                "tool_call_id": tool_call.id,
                "destination": (
                    "unavailable"
                    if quarantine_output
                    else output_redactor.redact_text(record.destination)
                ),
                "credential": (
                    None
                    if quarantine_output or record.credential is None
                    else output_redactor.redact_text(record.credential.name)
                ),
                "action": (
                    None
                    if quarantine_output or record.action is None
                    else output_redactor.redact_text(record.action)
                ),
                "metadata": (
                    {}
                    if quarantine_output
                    else output_redactor.redact_json(copy_json_value(record.metadata, "metadata"))
                ),
                "allowed": record.result.allowed,
                "reason": (
                    None
                    if quarantine_output or record.result.reason is None
                    else output_redactor.redact_text(record.result.reason)
                ),
                "result_metadata": (
                    {}
                    if quarantine_output
                    else output_redactor.redact_json(
                        copy_json_value(record.result.metadata, "result_metadata")
                    )
                ),
                **identity_payload,
            }
            if idempotency_key is not None:
                payload["idempotency_key"] = idempotency_key
            if approval_id is not None:
                payload["approval_id"] = approval_id
            if input_id is not None:
                payload["input_id"] = input_id
            yield await self._event_writer.emit(
                prepare_runtime_event(
                    _event_with_tool_round_authority(
                        event_with_execution_profile_authority(
                            Event(
                                type=EventType.CREDENTIAL_PROXY_CHECKED,
                                session_id=session.id,
                                agent_name=registered_agent.spec.name,
                                environment_name=_environment_name(registered_environment),
                                tool_name=tool_call.name,
                                payload=payload,
                            ),
                            execution_profile,
                        ),
                        tool_round_identity,
                        *(
                            field_name
                            for field_name in ("approval_id", "input_id")
                            if field_name in payload
                        ),
                    ),
                    redactor=redactor,
                )
            )

    async def _emit_terminal_tool_result(
        self,
        *,
        session: Session,
        registered_agent: runtime_records.RegisteredAgentState,
        registered_environment: runtime_records.RegisteredEnvironment | None,
        tool_call: runtime_records.ToolCallRequest,
        event_type: EventType,
        result: ToolResult,
        extra_payload: dict[str, Any],
        task_id: str | None,
        execution_profile: ExecutionProfileIdentity | None,
        invocation_context: InvocationContext | None,
        tool_round_identity: ToolRoundIdentity,
        approval_id: str | None,
        input_id: str | None,
        allow_modification: bool,
        redactor: SecretRedactor | None = None,
        output_redactor: SecretRedactor | None = None,
        argument_projection: tool_argument_publication.ToolArgumentProjection | None = None,
        deferred_terminal_stager: DeferredTerminalStager | None = None,
        publication_snapshot: invocation_secrets.InvocationPublicationSnapshot | None = None,
    ) -> AsyncIterator[tuple[Event, runtime_records.ToolCallOutcome | None]]:
        payload: dict[str, Any] = {
            "tool_call_id": tool_call.id,
            **extra_payload,
            "result": result.model_dump(),
            **copy_tool_round_identity(tool_round_identity).payload(),
        }
        if approval_id is not None:
            payload["approval_id"] = approval_id
        if input_id is not None:
            payload["input_id"] = input_id
        async for event in self.emit_tool_call_result_with_hooks(
            event=Event(
                type=event_type,
                session_id=session.id,
                agent_name=registered_agent.spec.name,
                environment_name=_environment_name(registered_environment),
                tool_name=tool_call.name,
                payload=payload,
            ),
            session=session,
            registered_agent=registered_agent,
            registered_environment=registered_environment,
            tool_call=tool_call,
            result=result,
            task_id=task_id,
            execution_profile=execution_profile,
            invocation_context=invocation_context,
            redactor=redactor,
            output_redactor=output_redactor,
            argument_projection=argument_projection,
            allow_modification=allow_modification,
            deferred_terminal_stager=deferred_terminal_stager,
            publication_snapshot=publication_snapshot,
        ):
            yield event

    async def _run_before_tool_call_hooks(
        self,
        *,
        session: Session,
        registered_agent: runtime_records.RegisteredAgentState,
        registered_environment: runtime_records.RegisteredEnvironment | None,
        tool_call: runtime_records.ToolCallRequest,
        anchor_event: Event,
        task_id: str | None,
        resolution: _BeforeToolCallResolution,
        redactor: SecretRedactor,
        output_redactor: SecretRedactor,
        execution_profile: ExecutionProfileIdentity | None,
        invocation_context: InvocationContext | None = None,
        quarantine_output: bool = False,
    ) -> AsyncIterator[Event]:
        if invocation_context is not None and (
            invocation_context.binding.session_id != session.id
            or invocation_context.registered_agent is not registered_agent
            or invocation_context.registered_environment is not registered_environment
            or invocation_context.profile is not execution_profile
        ):
            raise RuntimeError("Before-tool hooks lost frozen invocation authority.")
        runtime_hooks = (
            self._runtime_hooks if invocation_context is None else invocation_context.runtime_hooks
        )
        for hooks, scope in (
            (runtime_hooks, "app"),
            (registered_agent.runtime_hooks, "agent"),
        ):
            for registered_hook in hooks:
                hook = registered_hook.hook
                if not _runtime_hook_supports_phase(
                    hook=hook,
                    phase=RuntimeHookPhase.BEFORE_TOOL_CALL,
                ):
                    continue
                hook_name = registered_hook.name
                yield await self._event_writer.emit(
                    _redact_event_for_invocation(
                        _runtime_hook_event(
                            event_type=EventType.HOOK_STARTED,
                            hook_name=hook_name,
                            scope=scope,
                            phase=RuntimeHookPhase.BEFORE_TOOL_CALL,
                            session=session,
                            registered_agent=registered_agent,
                            registered_environment=registered_environment,
                            terminal_event=anchor_event,
                            execution_profile=execution_profile,
                            payload={
                                "tool_name": tool_call.name,
                                "tool_call_id": tool_call.id,
                            },
                        ),
                        redactor=redactor,
                    )
                )
                context = BeforeToolCallHookContext(
                    runtime=self._hook_runtime,
                    hook_name=hook_name,
                    phase=RuntimeHookPhase.BEFORE_TOOL_CALL,
                    session=session,
                    tool_name=tool_call.name,
                    tool_call_id=tool_call.id,
                    arguments=resolution.arguments,
                    task_id=task_id,
                    publication_actions_allowed=not quarantine_output,
                    execution_profile=execution_profile,
                )
                hook_failure_payload: dict[str, Any] | None = None
                try:
                    decision = await hook.before_tool_call(context)
                    try:
                        stop = _resolve_before_tool_call_decision(
                            decision,
                            resolution,
                            redactor=output_redactor,
                        )
                    finally:
                        # Hook-owned objects can retain application secrets. Do
                        # not carry the raw decision across durable publication.
                        decision = None
                except Exception as exc:
                    hook_failure_payload = (
                        {
                            "error_type": "runtime_hook_failure",
                            "actions": [],
                            "actions_omitted": True,
                        }
                        if quarantine_output
                        else {
                            **_hook_failure_payload(
                                exc,
                                redactor=output_redactor,
                            ),
                            **_hook_actions_payload(
                                context,
                                redactor=output_redactor,
                            ),
                        }
                    )
                # Publish only after the raw hook exception has left the active
                # handler so a store failure cannot inherit its traceback.
                if hook_failure_payload is not None:
                    yield await self._event_writer.emit(
                        _redact_event_for_invocation(
                            _runtime_hook_event(
                                event_type=EventType.HOOK_FAILED,
                                hook_name=hook_name,
                                scope=scope,
                                phase=RuntimeHookPhase.BEFORE_TOOL_CALL,
                                session=session,
                                registered_agent=registered_agent,
                                registered_environment=registered_environment,
                                terminal_event=anchor_event,
                                execution_profile=execution_profile,
                                payload={
                                    "tool_name": tool_call.name,
                                    "tool_call_id": tool_call.id,
                                    **hook_failure_payload,
                                },
                            ),
                            redactor=redactor,
                        )
                    )
                    continue
                yield await self._event_writer.emit(
                    _redact_event_for_invocation(
                        _runtime_hook_event(
                            event_type=EventType.HOOK_COMPLETED,
                            hook_name=hook_name,
                            scope=scope,
                            phase=RuntimeHookPhase.BEFORE_TOOL_CALL,
                            session=session,
                            registered_agent=registered_agent,
                            registered_environment=registered_environment,
                            terminal_event=anchor_event,
                            execution_profile=execution_profile,
                            payload={
                                "tool_name": tool_call.name,
                                "tool_call_id": tool_call.id,
                                **(
                                    {"actions": [], "actions_omitted": True}
                                    if quarantine_output
                                    else _hook_actions_payload(
                                        context,
                                        redactor=output_redactor,
                                    )
                                ),
                            },
                        ),
                        redactor=redactor,
                    )
                )
                if stop:
                    return

    async def emit_tool_call_result_with_hooks(
        self,
        *,
        event: Event,
        session: Session,
        registered_agent: runtime_records.RegisteredAgentState,
        registered_environment: runtime_records.RegisteredEnvironment | None,
        tool_call: runtime_records.ToolCallRequest,
        result: ToolResult,
        task_id: str | None,
        redactor: SecretRedactor | None = None,
        output_redactor: SecretRedactor | None = None,
        argument_projection: tool_argument_publication.ToolArgumentProjection | None = None,
        hook_argument_projection: tool_argument_publication.ToolArgumentProjection | None = None,
        allow_modification: bool = False,
        publish_before_hooks: bool = False,
        deferred_terminal_stager: DeferredTerminalStager | None = None,
        deferred_terminal_projection_recorder: DeferredTerminalFinalizer | None = None,
        deferred_terminal_finalizer: DeferredTerminalFinalizer | None = None,
        hooks_already_completed: bool = False,
        publication_snapshot: invocation_secrets.InvocationPublicationSnapshot | None = None,
        execution_profile: ExecutionProfileIdentity | None = None,
        invocation_context: InvocationContext | None = None,
        executed_runtime_tool: object | None = None,
    ) -> AsyncIterator[tuple[Event, runtime_records.ToolCallOutcome | None]]:
        if invocation_context is not None and (
            invocation_context.binding.session_id != session.id
            or invocation_context.registered_agent is not registered_agent
            or invocation_context.registered_environment is not registered_environment
            or invocation_context.profile is not execution_profile
        ):
            raise RuntimeError("Tool-result publication lost frozen invocation authority.")
        if publish_before_hooks and allow_modification:
            raise ValueError("Pre-hook tool-result publication cannot allow hook modification.")
        if hooks_already_completed and (publish_before_hooks or allow_modification):
            raise ValueError("Completed terminal hooks cannot be scheduled again.")
        if publish_before_hooks and "terminal_outcome" not in event.payload:
            raise ValueError("Pre-hook tool-result publication requires terminal controls.")
        if deferred_terminal_stager is not None and publication_snapshot is None:
            raise ValueError("Deferred terminal staging requires a publication snapshot.")
        if deferred_terminal_projection_recorder is not None and not publish_before_hooks:
            raise ValueError("Deferred terminal projection recording requires observational hooks.")
        registered_tool = registered_agent.executable_tool(tool_call.name)
        quarantine_hook_output = (
            registered_tool is not None and not registered_tool.publish_arguments
        )
        if quarantine_hook_output:
            allow_modification = False
        resolved_redactor = redactor if redactor is not None else self._secret_redactor
        resolved_output_redactor = resolved_redactor if output_redactor is None else output_redactor
        resolved_argument_projection = (
            tool_argument_publication.unavailable_argument_projection()
            if argument_projection is None
            else argument_projection
        )
        resolved_hook_argument_projection = (
            resolved_argument_projection
            if hook_argument_projection is None
            else hook_argument_projection
        )
        projected_tool_call = replace(
            tool_call,
            arguments=resolved_argument_projection.transcript_arguments(),
        )
        event = _restore_targeted_tool_invocation_event_authority(
            event,
            tool_call,
            redactor=resolved_redactor,
        )
        event_payload = dict(event.payload)
        event_payload.pop(tool_argument_publication.ARGUMENTS_FIELD, None)
        event_payload.pop(tool_argument_publication.ARGUMENTS_STATE_FIELD, None)
        if resolved_argument_projection.state == "unavailable":
            event_payload.pop("effective_arguments", None)
            event_payload[tool_argument_publication.ARGUMENTS_EXACT_FIELD] = False
        event_payload.update(resolved_argument_projection.payload_fields())
        event = event.model_copy(update={"payload": event_payload})
        event = _event_with_targeted_tool_invocation_authority(event, tool_call)
        event = event_with_execution_profile_authority(event, execution_profile)
        hook_tool_call = _project_tool_call_for_hook(
            tool_call,
            argument_projection=resolved_hook_argument_projection,
            redactor=resolved_output_redactor,
        )
        event_identity_values = {
            field_name: event.payload.get(field_name)
            for field_name in ("model_step_id", "model_attempt_id", "tool_round_id")
        }
        if all(type(value) is str for value in event_identity_values.values()):
            event_identity = ToolRoundIdentity.model_validate(event_identity_values)
            event = _event_with_tool_round_authority(
                event,
                event_identity,
                *(
                    field_name
                    for field_name in (
                        "approval_id",
                        "input_id",
                        "profile_id",
                        "exposure_fingerprint",
                    )
                    if field_name in event.payload
                ),
            )
        event, result = _prepare_tool_result_event(
            event=event,
            result=result,
            redactor=resolved_output_redactor,
            runtime_tool=executed_runtime_tool,
        )
        if not resolved_output_redactor.has_same_registry(resolved_redactor):
            event, result = _prepare_tool_result_event(
                event=event,
                result=result,
                redactor=resolved_redactor,
            )
        event = _restore_targeted_tool_invocation_event_authority(
            event,
            tool_call,
            redactor=resolved_redactor,
        )
        if deferred_terminal_stager is not None:
            await deferred_terminal_stager(
                event,
                runtime_records.ToolCallOutcome(
                    call=projected_tool_call,
                    result=result,
                ),
                allow_modification,
                publish_before_hooks,
                cast(
                    "invocation_secrets.InvocationPublicationSnapshot",
                    publication_snapshot,
                ),
            )
            return
        if hooks_already_completed:
            tool_event = await self._event_writer.emit(event)
            yield (
                tool_event,
                runtime_records.ToolCallOutcome(
                    call=projected_tool_call,
                    result=result,
                ),
            )
            return
        if publish_before_hooks:
            event, result, projection_cancellation = await self._project_terminal_tool_result(
                event=event,
                result=result,
                session=session,
                registered_environment=registered_environment,
                tool_call=hook_tool_call,
            )
            event = _restore_targeted_tool_invocation_event_authority(
                event,
                tool_call,
                redactor=resolved_redactor,
            )
            if deferred_terminal_projection_recorder is not None:
                event = await deferred_terminal_projection_recorder(event)
                event = _restore_targeted_tool_invocation_event_authority(
                    event,
                    tool_call,
                    redactor=resolved_redactor,
                )
                stored_result = event.payload.get("result")
                if type(stored_result) is not dict:
                    raise RuntimeError("Projected staged terminal lost its tool result.")
                result = tool_results.tool_result_from_payload(stored_result)
            tool_event = await self._event_writer.emit(event)
            yield (
                tool_event,
                runtime_records.ToolCallOutcome(
                    call=projected_tool_call,
                    result=result,
                ),
            )
            if projection_cancellation is not None:
                raise projection_cancellation
            async for hook_event, modified in self.run_tool_call_hooks(
                session=session,
                tool_event=tool_event,
                registered_agent=registered_agent,
                registered_environment=registered_environment,
                tool_call=hook_tool_call,
                result=result,
                task_id=task_id,
                execution_profile=execution_profile,
                invocation_context=invocation_context,
                redactor=resolved_redactor,
                output_redactor=resolved_output_redactor,
                allow_modification=False,
                quarantine_output=quarantine_hook_output,
            ):
                if modified is not None:
                    raise AssertionError(
                        "Observational after-tool hook modified terminal evidence."
                    )
                yield hook_event, None
            if deferred_terminal_finalizer is not None:
                await deferred_terminal_finalizer(event)
            return
        final_result = result
        async for hook_event, modified in self.run_tool_call_hooks(
            session=session,
            tool_event=event,
            registered_agent=registered_agent,
            registered_environment=registered_environment,
            tool_call=hook_tool_call,
            result=final_result,
            task_id=task_id,
            execution_profile=execution_profile,
            invocation_context=invocation_context,
            redactor=resolved_redactor,
            output_redactor=resolved_output_redactor,
            allow_modification=allow_modification,
            quarantine_output=quarantine_hook_output,
        ):
            yield hook_event, None
            if modified is not None:
                final_result = modified
        if final_result is not result:
            payload = dict(event.payload)
            payload["result"] = final_result.model_dump()
            event = event.model_copy(update={"payload": payload})
            event, final_result = _prepare_tool_result_event(
                event=event,
                result=final_result,
                redactor=resolved_output_redactor,
                restore_terminal_result_controls=False,
            )
            if not resolved_output_redactor.has_same_registry(resolved_redactor):
                event, final_result = _prepare_tool_result_event(
                    event=event,
                    result=final_result,
                    redactor=resolved_redactor,
                    restore_terminal_result_controls=False,
                )
        event, final_result, projection_cancellation = await self._project_terminal_tool_result(
            event=event,
            result=final_result,
            session=session,
            registered_environment=registered_environment,
            tool_call=tool_call,
        )
        event = _restore_targeted_tool_invocation_event_authority(
            event,
            tool_call,
            redactor=resolved_redactor,
        )
        if deferred_terminal_finalizer is not None:
            event = await deferred_terminal_finalizer(event)
            event = _restore_targeted_tool_invocation_event_authority(
                event,
                tool_call,
                redactor=resolved_redactor,
            )
            stored_result = event.payload.get("result")
            if type(stored_result) is not dict:
                raise RuntimeError("Finalized staged terminal lost its tool result.")
            final_result = tool_results.tool_result_from_payload(stored_result)
        tool_event = await self._event_writer.emit(event)
        yield (
            tool_event,
            runtime_records.ToolCallOutcome(
                call=projected_tool_call,
                result=final_result,
            ),
        )
        if projection_cancellation is not None:
            raise projection_cancellation

    async def _project_terminal_tool_result(
        self,
        *,
        event: Event,
        result: ToolResult,
        session: Session,
        registered_environment: runtime_records.RegisteredEnvironment | None,
        tool_call: runtime_records.ToolCallRequest,
    ) -> tuple[Event, ToolResult, asyncio.CancelledError | None]:
        policy = self._tool_result_projection_policy
        if policy is None:
            return event, result, None
        request = ToolResultProjectionRequest(
            result=result,
            session_id=session.id,
            agent_name=session.agent_name,
            environment_name=_environment_name(registered_environment),
            tool_call_id=tool_call.id,
            artifact_store=_artifact_store(registered_environment),
        )
        policy_task = asyncio.create_task(policy.project(request))
        outcome = await await_shielded_task_outcome(
            policy_task,
            timeout_s=_TOOL_RESULT_PROJECTION_TIMEOUT_SECONDS,
        )
        error = outcome.error
        if outcome.timed_out:
            policy_task.cancel()
            policy_task.add_done_callback(_consume_projection_task_outcome)
            projection = projection_failure(
                policy=policy,
                request=request,
                failure_type="projection_timeout",
            )
        elif error is not None:
            if isinstance(error, asyncio.CancelledError):
                error = unexpected_child_cancellation_error(
                    error,
                    operation="Tool-result projection policy",
                )
            if not isinstance(error, Exception):
                raise error
            projection = projection_failure(
                policy=policy,
                request=request,
                failure_type=safe_projection_failure_type(
                    error,
                    fallback="projection_policy_failure",
                ),
            )
        elif outcome.result is None:
            projection = projection_failure(
                policy=policy,
                request=request,
                failure_type="missing_projection_result",
            )
        else:
            try:
                projection = validate_tool_result_projection(
                    outcome.result,
                    request=request,
                    policy=policy,
                )
            except Exception as exc:
                projection = projection_failure(
                    policy=policy,
                    request=request,
                    failure_type=safe_projection_failure_type(
                        exc,
                        fallback="projection_policy_failure",
                    ),
                )
        payload = dict(event.payload)
        payload["result"] = projection.result.model_dump()
        payload["tool_result_projection"] = projection.record.model_dump(
            mode="json",
            exclude_none=True,
        )
        projected_event = event.model_copy(update={"payload": payload})
        # The policy receives the final redacted result. Reapplying generic
        # redaction here would rewrite runtime-owned artifact identities when
        # a registered secret happens to overlap an id, hash, type, or status.
        projected_event, projected_result = _validate_and_synchronize_tool_result_event(
            event=projected_event,
            result=projection.result,
        )
        projected_event, projected_result = web_access_results.restore_attested_tool_result(
            projected_event,
            original=projected_result,
            redacted=projected_result,
        )
        projected_event, projected_result = shared_artifact_results.restore_attested_tool_result(
            projected_event,
            original=projected_result,
            redacted=projected_result,
        )
        projected_event = event_with_runtime_nested_payload_authority(
            projected_event,
            _TOOL_RESULT_PROJECTION_PROVENANCE_PATH,
        )
        return projected_event, projected_result, outcome.cancellation

    async def run_tool_call_hooks(
        self,
        *,
        session: Session,
        tool_event: Event,
        registered_agent: runtime_records.RegisteredAgentState,
        registered_environment: runtime_records.RegisteredEnvironment | None,
        tool_call: runtime_records.ToolCallRequest,
        result: ToolResult,
        task_id: str | None,
        redactor: SecretRedactor,
        output_redactor: SecretRedactor,
        execution_profile: ExecutionProfileIdentity | None = None,
        invocation_context: InvocationContext | None = None,
        allow_modification: bool = False,
        quarantine_output: bool = False,
    ) -> AsyncIterator[tuple[Event, ToolResult | None]]:
        if invocation_context is not None and (
            invocation_context.binding.session_id != session.id
            or invocation_context.registered_agent is not registered_agent
            or invocation_context.registered_environment is not registered_environment
            or invocation_context.profile is not execution_profile
        ):
            raise RuntimeError("After-tool hooks lost frozen invocation authority.")
        current_result = result
        runtime_hooks = (
            self._runtime_hooks if invocation_context is None else invocation_context.runtime_hooks
        )
        for hooks, scope in (
            (runtime_hooks, "app"),
            (registered_agent.runtime_hooks, "agent"),
        ):
            async for hook_event, modified in self._run_scoped_tool_call_hooks(
                session=session,
                tool_event=tool_event,
                registered_agent=registered_agent,
                registered_environment=registered_environment,
                tool_call=tool_call,
                result=current_result,
                task_id=task_id,
                execution_profile=execution_profile,
                hooks=hooks,
                scope=scope,
                redactor=redactor,
                output_redactor=output_redactor,
                allow_modification=allow_modification,
                quarantine_output=quarantine_output,
                attested_result=result,
            ):
                yield hook_event, modified
                if modified is not None:
                    current_result = modified

    async def _run_scoped_tool_call_hooks(
        self,
        *,
        session: Session,
        tool_event: Event,
        registered_agent: runtime_records.RegisteredAgentState,
        registered_environment: runtime_records.RegisteredEnvironment | None,
        tool_call: runtime_records.ToolCallRequest,
        result: ToolResult,
        task_id: str | None,
        hooks: tuple[runtime_records.RegisteredRuntimeHook, ...],
        scope: str,
        redactor: SecretRedactor,
        output_redactor: SecretRedactor,
        execution_profile: ExecutionProfileIdentity | None = None,
        allow_modification: bool = False,
        quarantine_output: bool = False,
        attested_result: ToolResult | None = None,
    ) -> AsyncIterator[tuple[Event, ToolResult | None]]:
        current_result = result
        for registered_hook in hooks:
            hook = registered_hook.hook
            if not _runtime_hook_supports_phase(
                hook=hook,
                phase=RuntimeHookPhase.AFTER_TOOL_CALL,
            ):
                continue
            hook_name = registered_hook.name
            yield (
                await self._event_writer.emit(
                    _redact_event_for_invocation(
                        _runtime_hook_event(
                            event_type=EventType.HOOK_STARTED,
                            hook_name=hook_name,
                            scope=scope,
                            phase=RuntimeHookPhase.AFTER_TOOL_CALL,
                            session=session,
                            registered_agent=registered_agent,
                            registered_environment=registered_environment,
                            terminal_event=tool_event,
                            execution_profile=execution_profile,
                            payload={
                                "tool_name": tool_call.name,
                                "tool_call_id": tool_call.id,
                            },
                        ),
                        redactor=redactor,
                    )
                ),
                None,
            )
            context = ToolCallHookContext(
                runtime=self._hook_runtime,
                hook_name=hook_name,
                phase=RuntimeHookPhase.AFTER_TOOL_CALL,
                session=session,
                tool_event=tool_event,
                tool_name=tool_call.name,
                tool_call_id=tool_call.id,
                arguments=output_redactor.redact_json(tool_call.arguments),
                result=(
                    current_result
                    if _is_policy_denial_event(tool_event) and not allow_modification
                    else _redact_tool_result_for_event(
                        event=tool_event,
                        result=current_result,
                        redactor=output_redactor,
                    )
                ),
                task_id=task_id,
                publication_actions_allowed=not quarantine_output,
                execution_profile=execution_profile,
            )
            hook_failure_payload: dict[str, Any] | None = None
            try:
                decision = await hook.after_tool_call(context)
                try:
                    resolved = _resolve_after_tool_call_decision(
                        decision,
                        redactor=output_redactor,
                    )
                finally:
                    # The copied, sanitized resolution is the only hook result
                    # permitted to cross the following durable event await.
                    decision = None
                modified = resolved if allow_modification else None
            except Exception as exc:
                hook_failure_payload = (
                    {
                        "error_type": "runtime_hook_failure",
                        "actions": [],
                        "actions_omitted": True,
                    }
                    if quarantine_output
                    else {
                        **_hook_failure_payload(
                            exc,
                            redactor=output_redactor,
                        ),
                        **_hook_actions_payload(
                            context,
                            redactor=output_redactor,
                        ),
                    }
                )
            # Publish only after the raw hook exception has left the active
            # handler so a store failure cannot inherit its traceback.
            if hook_failure_payload is not None:
                yield (
                    await self._event_writer.emit(
                        _redact_event_for_invocation(
                            _runtime_hook_event(
                                event_type=EventType.HOOK_FAILED,
                                hook_name=hook_name,
                                scope=scope,
                                phase=RuntimeHookPhase.AFTER_TOOL_CALL,
                                session=session,
                                registered_agent=registered_agent,
                                registered_environment=registered_environment,
                                terminal_event=tool_event,
                                execution_profile=execution_profile,
                                payload={
                                    "tool_name": tool_call.name,
                                    "tool_call_id": tool_call.id,
                                    **hook_failure_payload,
                                },
                            ),
                            redactor=redactor,
                        )
                    ),
                    None,
                )
                continue
            if modified is not None:
                if attested_result is not None:
                    modified = web_access_results.preserve_attested_controls_across_hook(
                        tool_event,
                        original=attested_result,
                        replacement=modified,
                    )
                    modified = shared_artifact_results.preserve_attested_controls_across_hook(
                        tool_event,
                        original=attested_result,
                        replacement=modified,
                    )
                current_result = modified
            yield (
                await self._event_writer.emit(
                    _redact_event_for_invocation(
                        _runtime_hook_event(
                            event_type=EventType.HOOK_COMPLETED,
                            hook_name=hook_name,
                            scope=scope,
                            phase=RuntimeHookPhase.AFTER_TOOL_CALL,
                            session=session,
                            registered_agent=registered_agent,
                            registered_environment=registered_environment,
                            terminal_event=tool_event,
                            execution_profile=execution_profile,
                            payload={
                                "tool_name": tool_call.name,
                                "tool_call_id": tool_call.id,
                                **(
                                    {"actions": [], "actions_omitted": True}
                                    if quarantine_output
                                    else _hook_actions_payload(
                                        context,
                                        redactor=output_redactor,
                                    )
                                ),
                            },
                        ),
                        redactor=redactor,
                    )
                ),
                modified,
            )


class ToolRoundRun:
    """Per-session state for one or more ordinary tool rounds in a run."""

    def __init__(
        self,
        executor: ToolRoundExecutor,
        *,
        session: Session,
        registered_agent: runtime_records.RegisteredAgentState,
        registered_environment: runtime_records.RegisteredEnvironment | None,
        environment_name: str | None,
        limit_gate: RunLimitGate,
        request_metadata: dict[str, Any],
        task_id: str | None,
        structured_output: StructuredOutputSpec | None,
        thinking: ThinkingConfig | None,
        max_steps: int,
        limits: RunLimits,
        budget_limits: tuple[BudgetLimit, ...],
        retry_policy: RetryPolicy,
        run_started_at: float,
        turn_usage_tracker: SessionUsageTracker | None,
        active_run: ActiveSessionRun[SessionUsageTracker] | None,
        execution_profile: ExecutionProfileIdentity | None,
        invocation_context: InvocationContext | None,
    ) -> None:
        self._executor = executor
        self._session = session
        self._registered_agent = registered_agent
        self._registered_environment = registered_environment
        self._environment_name = environment_name
        self._limit_gate = limit_gate
        self._request_metadata = request_metadata
        self._task_id = task_id
        self._structured_output = structured_output
        self._thinking = thinking
        self._max_steps = max_steps
        self._limits = limits
        self._budget_limits = budget_limits
        self._retry_policy = retry_policy
        self._run_started_at = run_started_at
        self._turn_usage_tracker = turn_usage_tracker
        self._active_run = active_run
        self._execution_profile = execution_profile
        if invocation_context is not None and (
            invocation_context.binding.session_id != session.id
            or invocation_context.registered_agent is not registered_agent
            or invocation_context.registered_environment is not registered_environment
            or invocation_context.profile is not execution_profile
        ):
            raise ValueError("Tool-round execution lost frozen invocation authority.")
        self._invocation_context = invocation_context
        self.stopped_for_limit = False
        self.user_input_pause_superseded = False

    @property
    def execution_profile(self) -> ExecutionProfileIdentity | None:
        """Return the exact immutable profile resolved for this invocation."""

        return self._execution_profile

    async def run(
        self,
        *,
        messages: list[Message],
        tool_calls: list[runtime_records.ToolCallRequest],
        tool_round_identity: ToolRoundIdentity,
        model_step: int | None = None,
    ) -> AsyncIterator[Event]:
        tool_round_identity = copy_tool_round_identity(tool_round_identity)
        model_attempt_identity = ModelAttemptIdentity(
            model_step_id=tool_round_identity.model_step_id,
            model_attempt_id=tool_round_identity.model_attempt_id,
        )
        self.stopped_for_limit = False
        self.user_input_pause_superseded = False
        executor = self._executor
        session = self._session
        tool_outcomes: list[runtime_records.ToolCallOutcome] = []
        durable_lifecycle_events: list[Event] = []
        source_checkpoint = await executor._session_store.load_checkpoint(session.id)
        source_pending_round = tool_round_recovery.pending_tool_round_from_checkpoint(
            source_checkpoint,
            redactor=executor._secret_redactor,
            consume_on_rejection=True,
        )
        if source_pending_round is None:
            raise RuntimeError("Tool round has no durable pending exposure authority.")
        _require_matching_policy_round(
            pending_round=source_pending_round,
            tool_round_identity=tool_round_identity,
            tool_calls=tool_calls,
        )
        (
            source_pending_round,
            tool_calls,
            targeted_resolution_events,
        ) = await executor.resolve_targeted_tool_calls(
            session=session,
            registered_agent=self._registered_agent,
            registered_environment=self._registered_environment,
            pending_round=source_pending_round,
            invocation_context=self._invocation_context,
        )
        for targeted_resolution_event in targeted_resolution_events:
            yield targeted_resolution_event
        tool_exposure = source_pending_round.tool_exposure
        if tool_exposure is not None:
            tool_exposure = validate_resolved_tool_exposure_authority(
                tool_exposure,
                self._registered_agent.tool_capabilities,
                catalogue_revision=self._registered_agent.tool_catalogue.revision,
            )
        try:
            await executor._session_control.raise_if_interrupted(session.id)
            policy_plan = await executor.policy_plan(
                session=session,
                registered_agent=self._registered_agent,
                registered_environment=self._registered_environment,
                tool_calls=tool_calls,
                request_metadata=self._request_metadata,
                tool_exposure=tool_exposure,
                invocation_context=self._invocation_context,
            )
            await executor._session_control.raise_if_interrupted(session.id)
        except (SessionInterruptedByRequest, asyncio.CancelledError) as exc:
            async for event in self.close_after_interrupt(
                exc,
                messages=messages,
                tool_calls=tool_calls,
                tool_outcomes=tool_outcomes,
                tool_round_identity=tool_round_identity,
            ):
                yield event
            raise

        planned_round: tool_round_recovery.PendingToolRound | None = None
        if policy_plan.pending_approval is None:
            try:
                try:
                    planned_round = await executor.checkpoint_tool_round_policy_plan(
                        session=session,
                        registered_agent=self._registered_agent,
                        tool_calls=tool_calls,
                        policy_outcomes=policy_plan.outcomes,
                        active_taint_by_id=policy_plan.active_taint_labels,
                        tool_round_identity=tool_round_identity,
                    )
                except (RuntimeError, ValueError):
                    # A remote worker may commit interruption after the policy
                    # pre-check but before the fenced checkpoint transform.
                    # Reclassify only when durable status positively proves
                    # interruption; otherwise preserve the publication error.
                    await executor._session_control.raise_if_interrupted(session.id)
                    raise
            except (SessionInterruptedByRequest, asyncio.CancelledError) as exc:
                async for event in self.close_after_interrupt(
                    exc,
                    messages=messages,
                    tool_calls=tool_calls,
                    tool_outcomes=tool_outcomes,
                    tool_round_identity=tool_round_identity,
                ):
                    yield event
                raise

        limit_evaluation = await self._limit_gate.evaluate_limits(
            pending_tool_calls=len(tool_calls),
            execution_identity=model_attempt_identity,
        )
        async for event in self._apply_limit_evaluation(
            limit_evaluation,
            messages=messages,
            tool_calls=tool_calls,
            tool_round_identity=tool_round_identity,
        ):
            yield event
        if limit_evaluation.decision is not None:
            self.stopped_for_limit = True
            return

        if policy_plan.pending_approval is not None:
            approval_plan = policy_plan.pending_approval
            try:
                try:
                    approval, approval_events = await executor.checkpoint_pending_tool_approval(
                        session=session,
                        registered_agent=self._registered_agent,
                        registered_environment=self._registered_environment,
                        tool_call=approval_plan.call,
                        tool_calls=approval_plan.calls,
                        policy_outcomes=approval_plan.policy_outcomes,
                        active_taint_by_id=policy_plan.active_taint_labels,
                        task_id=self._task_id,
                        policy_result=approval_plan.policy_result,
                        structured_output=self._structured_output,
                        thinking=self._thinking,
                        max_steps=self._max_steps,
                        limits=self._limits,
                        budget_limits=self._budget_limits,
                        retry_policy=self._retry_policy,
                        tool_round_identity=tool_round_identity,
                    )
                except (RuntimeError, ValueError):
                    await executor._session_control.raise_if_interrupted(session.id)
                    raise
                for approval_event in approval_events:
                    yield approval_event
            except (SessionInterruptedByRequest, asyncio.CancelledError) as exc:
                async for event in self.close_after_interrupt(
                    exc,
                    messages=messages,
                    tool_calls=tool_calls,
                    tool_outcomes=tool_outcomes,
                    tool_round_identity=tool_round_identity,
                    clear_pending_approval=True,
                ):
                    yield event
                raise
            raise ToolApprovalRequired(approval)

        policy_results_by_id = {outcome.call.id: outcome.result for outcome in policy_plan.outcomes}
        policy_evidence_by_id = {
            outcome.call.id: outcome.evidence for outcome in policy_plan.outcomes
        }
        user_input_pause = _first_user_input_tool_call(
            self._registered_agent,
            tool_calls,
            policy_results_by_id,
            policy_evidence_by_id,
        )
        if user_input_pause is not None:
            user_input_call, question, options = user_input_pause
            try:
                pending_input, pause_events = await executor.checkpoint_pending_user_input(
                    session=session,
                    registered_agent=self._registered_agent,
                    registered_environment=self._registered_environment,
                    tool_call=user_input_call,
                    tool_calls=tool_calls,
                    policy_outcomes=policy_plan.outcomes,
                    active_taint_by_id=policy_plan.active_taint_labels,
                    task_id=self._task_id,
                    structured_output=self._structured_output,
                    thinking=self._thinking,
                    max_steps=self._max_steps,
                    limits=self._limits,
                    budget_limits=self._budget_limits,
                    retry_policy=self._retry_policy,
                    question=question,
                    options=options,
                    tool_round_identity=tool_round_identity,
                )
            except (RuntimeError, ValueError):
                # A remote worker may claim interruption after the policy
                # pre-check but before the atomic pause publication. Translate
                # only positive durable interruption evidence; otherwise keep
                # the original publication failure authoritative.
                await executor._session_control.raise_if_interrupted(session.id)
                raise
            for pause_event in pause_events:
                yield pause_event
            try:
                await executor._session_control.raise_if_interrupted(session.id)
            except SessionInterruptedByRequest:
                # The pause committed, but a remote interruption atomically
                # superseded it while persisted side effects were delivered.
                # The pending round no longer exists, so let the outer run owner
                # finish the operator interruption without treating the pause as
                # an interrupted ordinary round.
                self.user_input_pause_superseded = True
                return
            raise UserInputRequired(pending_input)

        if planned_round is None:
            raise RuntimeError("Executable tool round has no durable policy plan.")
        policy_output_secret_resolution_scope = approval_support.tool_round_secret_resolution_scope(
            planned_round
        )
        defer_round_terminals = (
            len(tool_calls) > 1 and policy_output_secret_resolution_scope != "static"
        ) or any(
            registered is not None and registered.workspace_mutation
            for registered in (
                self._registered_agent.executable_tool(tool_call.name) for tool_call in tool_calls
            )
        )
        publication_coordinator = (
            _ToolRoundPublicationCoordinator(
                session_id=session.id,
                tool_round_identity=tool_round_identity,
                session_store=executor._session_store,
                redactor=_redactor_for_tool_calls(
                    executor._secret_redactor,
                    registered_agent=self._registered_agent,
                    tool_calls=tool_calls,
                ),
                execution_profile=self._execution_profile,
                tool_exposure=tool_exposure,
            )
            if defer_round_terminals
            else None
        )
        staged_hook_modes: dict[str, tuple[bool, bool]] = {}
        staged_private_outcomes: dict[str, runtime_records.ToolCallOutcome] = {}
        segments = self._tool_round_segments(tool_calls)

        async def synchronize_staged_outcomes() -> None:
            """Refresh private limit/interruption bookkeeping from durable stages."""

            if publication_coordinator is None:
                return
            checkpoint = await executor._session_store.load_checkpoint(session.id)
            pending = tool_round_recovery.pending_tool_round_from_checkpoint(checkpoint)
            if (
                pending is None
                or tool_round_recovery.pending_tool_round_identity(pending) != tool_round_identity
            ):
                raise RuntimeError("Staged outcomes lost their pending tool-round owner.")
            calls_by_id = {call.id: call for call in tool_calls}
            staged_private_outcomes.clear()
            for staged in pending.staged_terminals:
                call = calls_by_id.get(staged.tool_call_id)
                result_payload = staged.event.payload.get("result")
                if call is None or type(result_payload) is not dict:
                    raise RuntimeError("Staged outcome conflicts with its tool-round call.")
                staged_private_outcomes[staged.tool_call_id] = runtime_records.ToolCallOutcome(
                    call=replace(call, arguments={}),
                    result=tool_results.tool_result_from_payload(result_payload),
                )
            tool_outcomes[:] = [
                staged_private_outcomes[call.id]
                for call in tool_calls
                if call.id in staged_private_outcomes
            ]

        async def record_round_publication_snapshot(
            tool_call_id: str,
            snapshot: invocation_secrets.InvocationPublicationSnapshot,
        ) -> None:
            if publication_coordinator is not None:
                await publication_coordinator.seal_call(
                    tool_call_id=tool_call_id,
                    snapshot=snapshot,
                )
                return
            await executor._session_store.transform_checkpoint(
                session.id,
                tool_round_recovery.assistant_publication_snapshot_transform(
                    tool_round_identity=tool_round_identity,
                    tool_call_id=tool_call_id,
                    redactor=snapshot.redactor,
                    unsafe_output=snapshot.secret_scope_incomplete,
                ),
            )

        async def record_round_redactor(
            tool_call_id: str,
            snapshot: InvocationRedactorSnapshot,
        ) -> None:
            if publication_coordinator is None:
                raise AssertionError("Round redactor observer requires a publication coordinator.")
            await publication_coordinator.register_redactor(
                tool_call_id=tool_call_id,
                redactor=snapshot.redactor,
            )
            await synchronize_staged_outcomes()

        async def stage_round_terminal(
            event: Event,
            outcome: runtime_records.ToolCallOutcome,
            allow_modification: bool,
            publish_before_hooks: bool,
            snapshot: invocation_secrets.InvocationPublicationSnapshot,
        ) -> Event:
            if publication_coordinator is None:
                raise AssertionError("Terminal staging requires a publication coordinator.")
            prepared_event = executor._event_writer.prepare(event)
            interrupted_terminal = prepared_event.payload.get("interrupted") is True
            exposure_blocked = (
                prepared_event.type is EventType.TOOL_CALL_BLOCKED
                and prepared_event.payload.get("blocked_by") == "tool_exposure"
            )
            staged_event = await publication_coordinator.stage_terminal(
                tool_call_id=outcome.call.id,
                event=prepared_event,
                snapshot=snapshot,
                hooks_state=(
                    "completed"
                    if exposure_blocked
                    else (
                        "pending"
                        if interrupted_terminal
                        else (
                            "observational"
                            if publish_before_hooks
                            else ("pending" if allow_modification else "finalized")
                        )
                    )
                ),
            )
            staged_hook_modes[outcome.call.id] = (
                (False if interrupted_terminal else allow_modification),
                (False if interrupted_terminal else publish_before_hooks),
            )
            await synchronize_staged_outcomes()
            return staged_event

        async def complete_round_terminal_hooks(event: Event) -> Event:
            if publication_coordinator is None:
                raise AssertionError("Hook finalization requires a publication coordinator.")
            return await publication_coordinator.complete_terminal_hooks(event)

        async def record_round_terminal_projection(event: Event) -> Event:
            if publication_coordinator is None:
                raise AssertionError("Projection recording requires a publication coordinator.")
            return await publication_coordinator.record_projected_terminal(event)

        async def record_round_workspace_capture(event: Event) -> Event:
            if publication_coordinator is None:
                raise AssertionError(
                    "Workspace capture recording requires a publication coordinator."
                )
            recorded = await publication_coordinator.record_workspace_capture(event)
            await synchronize_staged_outcomes()
            return recorded

        async def publish_staged_round_terminals(
            expected_stage_ids: set[str],
        ) -> AsyncIterator[Event]:
            if publication_coordinator is None:
                if expected_stage_ids:
                    raise AssertionError("Staged publication requires a coordinator.")
                return
            staged_checkpoint = await executor._session_store.load_checkpoint(session.id)
            staged_round = tool_round_recovery.pending_tool_round_from_checkpoint(staged_checkpoint)
            if (
                staged_round is None
                or tool_round_recovery.pending_tool_round_identity(staged_round)
                != tool_round_identity
            ):
                raise RuntimeError("Staged terminal publication lost its pending tool round.")
            staged_by_id = {item.tool_call_id: item for item in staged_round.staged_terminals}
            if set(staged_by_id) != expected_stage_ids:
                raise RuntimeError(
                    "Dynamic multi-call publication has an unexpected staged-terminal set."
                )
            calls_by_id = {call.id: call for call in tool_calls}
            if not expected_stage_ids.issubset(calls_by_id):
                raise RuntimeError("Staged terminal publication names an unknown tool call.")
            final_outcomes: dict[str, runtime_records.ToolCallOutcome] = {}
            for tool_call in tool_calls:
                staged = staged_by_id.get(tool_call.id)
                if staged is None:
                    continue
                staged_event = publication_coordinator.restore_staged_event_authority(staged.event)
                result_payload = staged_event.payload.get("result")
                if type(result_payload) is not dict:
                    raise RuntimeError("Staged terminal publication lost its tool result.")
                result = tool_results.tool_result_from_payload(result_payload)
                argument_projection, hook_argument_projection = (
                    _staged_terminal_argument_projections(staged_event)
                )
                registered_tool = self._registered_agent.executable_tool(tool_call.name)
                if (
                    len(tool_calls) > 1
                    and registered_tool is not None
                    and registered_tool.publish_arguments
                    and staged_event.type
                    in {EventType.TOOL_CALL_COMPLETED, EventType.TOOL_CALL_FAILED}
                ):
                    argument_projection = tool_argument_publication.finalized_argument_projection(
                        tool_call.arguments,
                        redactor=publication_coordinator.redactor,
                        scope_finalized=publication_coordinator.argument_scope_finalized,
                    )
                    staged_payload = dict(staged_event.payload)
                    staged_payload[tool_argument_publication.ARGUMENTS_EXACT_FIELD] = (
                        tool_argument_publication.argument_projection_is_exact(
                            argument_projection,
                            private_arguments=tool_call.arguments,
                        )
                    )
                    staged_event = staged_event.model_copy(update={"payload": staged_payload})
                hooks_already_completed = staged.hooks_state == "completed"
                allow_modification, publish_before_hooks = (
                    (False, False)
                    if hooks_already_completed
                    else staged_hook_modes.get(
                        tool_call.id,
                        (
                            staged.hooks_state == "pending",
                            staged.hooks_state == "observational",
                        ),
                    )
                )
                async for event, outcome in executor.emit_tool_call_result_with_hooks(
                    event=staged_event,
                    session=session,
                    registered_agent=self._registered_agent,
                    registered_environment=self._registered_environment,
                    tool_call=tool_call,
                    result=result,
                    task_id=self._task_id,
                    execution_profile=self._execution_profile,
                    invocation_context=self._invocation_context,
                    redactor=publication_coordinator.redactor,
                    output_redactor=publication_coordinator.redactor,
                    argument_projection=argument_projection,
                    hook_argument_projection=hook_argument_projection,
                    allow_modification=allow_modification,
                    publish_before_hooks=publish_before_hooks,
                    deferred_terminal_projection_recorder=(
                        record_round_terminal_projection
                        if publish_before_hooks and not hooks_already_completed
                        else None
                    ),
                    deferred_terminal_finalizer=(
                        None if hooks_already_completed else complete_round_terminal_hooks
                    ),
                    hooks_already_completed=hooks_already_completed,
                ):
                    yield event
                    if event.type in tool_round_recovery._TOOL_ROUND_TERMINAL_EVENT_TYPES:
                        durable_lifecycle_events.append(copy_event(event))
                    if outcome is not None:
                        final_outcomes[outcome.call.id] = outcome
            if set(final_outcomes) != expected_stage_ids:
                raise RuntimeError("Staged terminal publication lost a public outcome.")
            current_by_id = {outcome.call.id: outcome for outcome in tool_outcomes}
            current_by_id.update(final_outcomes)
            tool_outcomes[:] = [
                current_by_id[call.id] for call in tool_calls if call.id in current_by_id
            ]

        async def publish_staged_terminals_before_limit() -> AsyncIterator[Event]:
            expected_stage_ids = {outcome.call.id for outcome in tool_outcomes}
            async for event in publish_staged_round_terminals(expected_stage_ids):
                yield event

        async def publish_staged_terminals_before_interrupt() -> AsyncIterator[Event]:
            """Make already completed effects authoritative before round interruption."""

            if publication_coordinator is None or not staged_private_outcomes:
                return
            async for event in publish_staged_round_terminals(set(staged_private_outcomes)):
                yield event

        # Static rounds have no late secret-resolution capability and can use
        # each invocation's sealed projection directly. Dynamic multi-call
        # rounds stay quarantined until the coordinator has merged every sealed
        # scope, then publish through that cumulative redactor.
        publish_arguments_as_unavailable = len(tool_calls) > 1 and (
            publication_coordinator is not None
        )
        round_task = asyncio.current_task()
        round_cancellation_baseline = 0 if round_task is None else round_task.cancelling()
        try:
            for run_parallel, segment_calls in segments:
                if run_parallel:
                    call_stream = self._run_tool_calls_parallel(
                        tool_calls=segment_calls,
                        tool_outcomes=tool_outcomes,
                        policy_results_by_id=policy_results_by_id,
                        policy_evidence_by_id=policy_evidence_by_id,
                        tool_exposure=tool_exposure,
                        tool_round_identity=tool_round_identity,
                        model_step=model_step,
                        taint_labels_by_id=policy_plan.active_taint_labels,
                        publish_arguments_as_unavailable=publish_arguments_as_unavailable,
                        policy_output_secret_resolution_scope=(
                            policy_output_secret_resolution_scope
                        ),
                        deferred_terminal_stager=(
                            stage_round_terminal if publication_coordinator is not None else None
                        ),
                        deferred_terminal_capture_recorder=(
                            record_round_workspace_capture
                            if publication_coordinator is not None
                            else None
                        ),
                        resolved_redactor_observer=(
                            record_round_redactor if publication_coordinator is not None else None
                        ),
                        publication_snapshot_observer=record_round_publication_snapshot,
                    )
                else:
                    call_stream = self._run_tool_calls_sequential(
                        messages=messages,
                        tool_calls=segment_calls,
                        round_tool_calls=tool_calls,
                        tool_outcomes=tool_outcomes,
                        policy_results_by_id=policy_results_by_id,
                        policy_evidence_by_id=policy_evidence_by_id,
                        tool_exposure=tool_exposure,
                        tool_round_identity=tool_round_identity,
                        model_step=model_step,
                        taint_labels_by_id=policy_plan.active_taint_labels,
                        publish_arguments_as_unavailable=publish_arguments_as_unavailable,
                        policy_output_secret_resolution_scope=(
                            policy_output_secret_resolution_scope
                        ),
                        deferred_terminal_stager=(
                            stage_round_terminal if publication_coordinator is not None else None
                        ),
                        deferred_terminal_capture_recorder=(
                            record_round_workspace_capture
                            if publication_coordinator is not None
                            else None
                        ),
                        resolved_redactor_observer=(
                            record_round_redactor if publication_coordinator is not None else None
                        ),
                        publication_snapshot_observer=record_round_publication_snapshot,
                        publish_staged_terminals=(
                            publish_staged_terminals_before_limit
                            if publication_coordinator is not None
                            else None
                        ),
                    )
                async for event, outcome in call_stream:
                    yield event
                    if event.type == EventType.TOOL_CALL_STARTED or (
                        event.type in tool_round_recovery._TOOL_ROUND_TERMINAL_EVENT_TYPES
                    ):
                        durable_lifecycle_events.append(copy_event(event))
                    if outcome is not None:
                        tool_outcomes.append(outcome)
                if self.stopped_for_limit:
                    break
            if self.stopped_for_limit:
                return
        except WorkspaceMutationSettlementError as settlement_failure:
            if publication_coordinator is not None:
                expected_stage_ids = set(staged_private_outcomes)
                try:
                    async for event in publish_staged_round_terminals(expected_stage_ids):
                        yield event
                        if event.type in tool_round_recovery._TOOL_ROUND_TERMINAL_EVENT_TYPES:
                            durable_lifecycle_events.append(copy_event(event))
                except asyncio.CancelledError as cancellation:
                    raise cancellation from settlement_failure
                except BaseException as publication_failure:
                    if any(
                        isinstance(
                            candidate,
                            (KeyboardInterrupt, SystemExit, GeneratorExit),
                        )
                        for candidate in iter_exception_tree(publication_failure)
                    ):
                        raise
                    raise settlement_failure from publication_failure
            raise settlement_failure
        except BaseExceptionGroup as exc:
            current_task = asyncio.current_task()
            minimum_cancellation_requests = 0 if current_task is None else current_task.cancelling()
            if not is_current_runner_cancellation_group(exc):
                raise
            interrupt: asyncio.CancelledError | BaseExceptionGroup = exc
            interrupt_cause: BaseException | None = None
            if round_task is not None and round_task.cancelling() > round_cancellation_baseline:
                cancellation_sources = _ordered_cancellation_sources(
                    exc,
                    child_cancellations=[],
                    tool_calls=tool_calls,
                )
                cancellation = _authoritative_task_cancellation(
                    round_task,
                    cancellation_sources=cancellation_sources,
                )
                _transfer_cancellation_evidence(
                    cancellation,
                    cancellation_sources,
                    tool_calls=tool_calls,
                )
                interrupt_cause = _cancellation_failure_cause(
                    sanitize_runner_failure_group(
                        exc,
                        caller_cancelled=True,
                    ),
                    runner_cleanup_failures=[],
                )
                if interrupt_cause is not None:
                    attach_runner_cancellation_failure(
                        cancellation,
                        interrupt_cause,
                    )
                    set_exception_cause(cancellation, interrupt_cause)
                interrupt = cancellation
            try:
                async for event in publish_staged_terminals_before_interrupt():
                    yield event
                async for event in self.close_after_interrupt(
                    interrupt,
                    messages=messages,
                    tool_calls=tool_calls,
                    tool_outcomes=tool_outcomes,
                    tool_round_identity=tool_round_identity,
                ):
                    yield event
            except BaseException as closure_error:
                restore_cancellation_requests = (
                    _consume_current_task_cancellation_requests(closure_error)
                    if isinstance(closure_error, asyncio.CancelledError)
                    else 0
                )
                _raise_after_interrupted_round_closure_failure(
                    interrupt,
                    closure_error=closure_error,
                    restore_cancellation_requests=restore_cancellation_requests,
                    minimum_cancellation_requests=minimum_cancellation_requests,
                )
            if isinstance(interrupt, asyncio.CancelledError):
                invocation_secrets.sanitize_external_cancellation(interrupt)
                _restore_current_task_cancellation_requests(
                    minimum_requests=minimum_cancellation_requests,
                )
                raise interrupt from exception_cause(interrupt)
            raise
        except asyncio.CancelledError as exc:
            current_task = asyncio.current_task()
            minimum_cancellation_requests = 0 if current_task is None else current_task.cancelling()
            try:
                async for event in publish_staged_terminals_before_interrupt():
                    yield event
                async for event in self.close_after_interrupt(
                    exc,
                    messages=messages,
                    tool_calls=tool_calls,
                    tool_outcomes=tool_outcomes,
                    tool_round_identity=tool_round_identity,
                ):
                    yield event
            except BaseException as closure_error:
                restore_cancellation_requests = (
                    _consume_current_task_cancellation_requests(closure_error)
                    if isinstance(closure_error, asyncio.CancelledError)
                    else 0
                )
                _raise_after_interrupted_round_closure_failure(
                    exc,
                    closure_error=closure_error,
                    restore_cancellation_requests=restore_cancellation_requests,
                    minimum_cancellation_requests=minimum_cancellation_requests,
                )
            invocation_secrets.sanitize_external_cancellation(exc)
            _restore_current_task_cancellation_requests(
                minimum_requests=minimum_cancellation_requests,
            )
            raise
        except SessionInterruptedByRequest as exc:
            async for event in publish_staged_terminals_before_interrupt():
                yield event
            async for event in self.close_after_interrupt(
                exc,
                messages=messages,
                tool_calls=tool_calls,
                tool_outcomes=tool_outcomes,
                tool_round_identity=tool_round_identity,
            ):
                yield event
            raise
        except (KeyboardInterrupt, SystemExit, GeneratorExit):
            # An async-generator close cannot yield, but a tool outcome that is
            # already durably staged must still become authoritative before
            # the invocation owner disappears.
            async for _event in publish_staged_terminals_before_interrupt():
                pass
            raise

        if publication_coordinator is not None:
            # Private stages were retained in ``tool_outcomes`` only so limit
            # and interruption closure could account for completed effects.
            # Rebuild the normal successful outcome list from the final public
            # events in model order.
            async for event in publish_staged_round_terminals({call.id for call in tool_calls}):
                yield event

        source_checkpoint = await executor._session_store.load_checkpoint(session.id)
        pending_round = tool_round_recovery.pending_tool_round_from_checkpoint(source_checkpoint)
        if (
            pending_round is None
            or tool_round_recovery.pending_tool_round_identity(pending_round) != tool_round_identity
        ):
            raise RuntimeError("The durable pending tool round changed before publication.")
        prepared_publication = tool_round_publication.prepare_tool_round_publication(
            session_id=session.id,
            pending_round=pending_round,
            source_checkpoint=source_checkpoint,
            durable_events=durable_lifecycle_events,
            expected_statuses={
                SessionStatus.RUNNING,
                SessionStatus.INTERRUPTING,
            },
            expected_run_epoch=session.run_epoch,
            expected_transcript_cursor=(
                await executor._session_store.load_transcript_cursor(session.id)
            ),
        )
        cancellation = await tool_round_publication.publish_tool_round_with_exact_replay(
            prepared_publication,
            session_store=executor._session_store,
            event_writer=executor._event_writer,
        )
        messages.extend(prepared_publication.request.transcript_messages)
        if cancellation is not None:
            raise cancellation
        await executor._session_control.raise_if_interrupted(session.id)

    async def _apply_limit_evaluation(
        self,
        evaluation: LimitEvaluation,
        *,
        messages: list[Message],
        tool_calls: list[runtime_records.ToolCallRequest],
        completed_tool_outcomes: list[runtime_records.ToolCallOutcome] | None = None,
        tool_round_identity: ToolRoundIdentity,
        publish_staged_terminals: Callable[[], AsyncIterator[Event]] | None = None,
    ) -> AsyncIterator[Event]:
        if evaluation.decision is not None and publish_staged_terminals is not None:
            async for event in publish_staged_terminals():
                yield event
        request = ToolRoundLimitRequest(
            evaluation=evaluation,
            session=self._session,
            registered_agent=self._registered_agent,
            registered_environment=self._registered_environment,
            environment_name=self._environment_name,
            messages=messages,
            tool_calls=tool_calls,
            completed_tool_outcomes=(
                [] if completed_tool_outcomes is None else completed_tool_outcomes
            ),
            tool_round_identity=copy_tool_round_identity(tool_round_identity),
            run_started_at=self._run_started_at,
            turn_usage_tracker=self._turn_usage_tracker,
            active_run=self._active_run,
            execution_profile=self._execution_profile,
            invocation_context=self._invocation_context,
        )
        async for event in self._executor._apply_limit_evaluation(request):
            yield event

    async def close_after_interrupt(
        self,
        exc: BaseException,
        *,
        messages: list[Message],
        tool_calls: list[runtime_records.ToolCallRequest],
        tool_outcomes: list[runtime_records.ToolCallOutcome],
        tool_round_identity: ToolRoundIdentity,
        clear_pending_approval: bool = False,
    ) -> AsyncIterator[Event]:
        cancellation_artifacts: list[dict[str, Any]] | None = None
        cancellation_artifacts_by_id: dict[str, list[dict[str, Any]]] | None = None
        cancellation_redactors_by_id: dict[str, SecretRedactor] | None = None
        if isinstance(exc, SessionInterruptedByRequest):
            pass
        elif isinstance(exc, BaseExceptionGroup):
            if not is_current_runner_cancellation_group(exc):
                raise TypeError("Unsupported interrupt exception group.")
            (
                cancellation_artifacts,
                cancellation_artifacts_by_id,
                cancellation_redactors_by_id,
            ) = _grouped_cancellation_evidence(
                exc,
                tool_calls=tool_calls,
            )
        elif isinstance(exc, asyncio.CancelledError):
            interrupt_requested = await self._executor._session_control.interrupt_requested(
                self._session.id
            )
            has_tool_evidence = invocation_secrets.has_cancellation_evidence(exc)
            if not interrupt_requested and not has_tool_evidence:
                invocation_secrets.sanitize_external_cancellation(exc)
                return
            if interrupt_requested:
                clear_current_task_cancellation()
            cancellation_artifacts = invocation_secrets.cancellation_artifacts(exc)
            cancellation_artifacts_by_id = invocation_secrets.cancellation_artifacts_by_id(exc)
            cancellation_redactors_by_id = invocation_secrets.cancellation_redactors_by_id(exc)
            if cancellation_artifacts_by_id is not None:
                cancellation_artifacts = None
            producer_id = invocation_secrets.cancellation_tool_call_id(exc)
            cancellation_redactor = invocation_secrets.cancellation_redactor(exc)
            if (
                producer_id is not None
                and cancellation_artifacts
                and cancellation_artifacts_by_id is None
            ):
                cancellation_artifacts_by_id = {producer_id: cancellation_artifacts}
                cancellation_artifacts = None
            if (
                producer_id is not None
                and cancellation_redactor is not None
                and cancellation_redactors_by_id is None
            ):
                cancellation_redactors_by_id = {producer_id: cancellation_redactor}
        else:
            raise TypeError(f"Unsupported interrupt exception: {type(exc).__name__}")
        if clear_pending_approval:
            await self._executor.clear_pending_tool_approval_for_tool_round(
                self._session.id,
                tool_calls,
            )
        request = InterruptedToolRoundRequest(
            session=self._session,
            registered_agent=self._registered_agent,
            registered_environment=self._registered_environment,
            messages=messages,
            tool_calls=tool_calls,
            tool_outcomes=tool_outcomes,
            tool_round_identity=copy_tool_round_identity(tool_round_identity),
            cancellation_artifacts=cancellation_artifacts,
            cancellation_artifacts_by_id=cancellation_artifacts_by_id,
            cancellation_redactors_by_id=cancellation_redactors_by_id,
            execution_profile=self._execution_profile,
            invocation_context=self._invocation_context,
        )
        async for event in self._executor._close_interrupted_round(request):
            yield event

    def _tool_round_segments(
        self,
        tool_calls: list[runtime_records.ToolCallRequest],
    ) -> list[tuple[bool, list[runtime_records.ToolCallRequest]]]:
        if self._executor._max_parallel_tool_calls <= 1:
            return [(False, tool_calls)]
        segments: list[tuple[bool, list[runtime_records.ToolCallRequest]]] = []
        safe_run: list[runtime_records.ToolCallRequest] = []
        for tool_call in tool_calls:
            if _tool_call_is_parallel_safe(self._registered_agent, tool_call):
                safe_run.append(tool_call)
                continue
            if safe_run:
                segments.append((len(safe_run) >= 2, safe_run))
                safe_run = []
            segments.append((False, [tool_call]))
        if safe_run:
            segments.append((len(safe_run) >= 2, safe_run))
        return segments

    async def _run_tool_calls_sequential(
        self,
        *,
        messages: list[Message],
        tool_calls: list[runtime_records.ToolCallRequest],
        tool_outcomes: list[runtime_records.ToolCallOutcome],
        policy_results_by_id: dict[str, ToolPolicyResult | None],
        policy_evidence_by_id: dict[str, ToolPolicyEvidence],
        tool_exposure: ResolvedToolExposureAuthority | None,
        tool_round_identity: ToolRoundIdentity,
        model_step: int | None,
        round_tool_calls: list[runtime_records.ToolCallRequest] | None = None,
        taint_labels_by_id: Mapping[str, frozenset[str]] = MappingProxyType({}),
        publish_arguments_as_unavailable: bool = False,
        policy_output_secret_resolution_scope: Literal["static", "dynamic", "unknown"] = "unknown",
        deferred_terminal_stager: DeferredTerminalStager | None = None,
        deferred_terminal_capture_recorder: DeferredTerminalCaptureRecorder | None = None,
        resolved_redactor_observer: (
            Callable[[str, InvocationRedactorSnapshot], Awaitable[None]] | None
        ) = None,
        publication_snapshot_observer: (
            Callable[
                [str, invocation_secrets.InvocationPublicationSnapshot],
                Awaitable[None],
            ]
            | None
        ) = None,
        publish_staged_terminals: Callable[[], AsyncIterator[Event]] | None = None,
    ) -> AsyncIterator[tuple[Event, runtime_records.ToolCallOutcome | None]]:
        if round_tool_calls is None:
            round_tool_calls = tool_calls
        model_attempt_identity = ModelAttemptIdentity(
            model_step_id=tool_round_identity.model_step_id,
            model_attempt_id=tool_round_identity.model_attempt_id,
        )
        for tool_call in tool_calls:
            await self._executor._session_control.raise_if_interrupted(self._session.id)
            limit_evaluation = await self._limit_gate.evaluate_limits(
                pending_tool_calls=1,
                execution_identity=model_attempt_identity,
            )
            async for event in self._apply_limit_evaluation(
                limit_evaluation,
                messages=messages,
                tool_calls=round_tool_calls,
                completed_tool_outcomes=tool_outcomes,
                tool_round_identity=tool_round_identity,
                publish_staged_terminals=publish_staged_terminals,
            ):
                yield event, None
            if limit_evaluation.decision is not None:
                self.stopped_for_limit = True
                return
            async for event, outcome in self._executor.execute_tool_call(
                session=self._session,
                registered_agent=self._registered_agent,
                registered_environment=self._registered_environment,
                tool_call=tool_call,
                request_metadata=self._request_metadata,
                task_id=self._task_id,
                model_step=model_step,
                execution_profile=self._execution_profile,
                invocation_context=self._invocation_context,
                policy_result=policy_results_by_id.get(tool_call.id),
                policy_evidence=policy_evidence_by_id.get(
                    tool_call.id,
                    ToolPolicyEvidence.UNPLANNED,
                ),
                tool_exposure=tool_exposure,
                tool_round_identity=tool_round_identity,
                taint_labels=taint_labels_by_id.get(tool_call.id, frozenset()),
                publish_arguments_as_unavailable=publish_arguments_as_unavailable,
                policy_output_secret_resolution_scope=(policy_output_secret_resolution_scope),
                deferred_terminal_stager=deferred_terminal_stager,
                deferred_terminal_capture_recorder=deferred_terminal_capture_recorder,
                resolved_redactor_observer=resolved_redactor_observer,
                publication_snapshot_observer=publication_snapshot_observer,
            ):
                yield event, outcome
            await self._executor._session_control.raise_if_interrupted(self._session.id)

    async def _run_tool_calls_parallel(
        self,
        *,
        tool_calls: list[runtime_records.ToolCallRequest],
        tool_outcomes: list[runtime_records.ToolCallOutcome],
        policy_results_by_id: dict[str, ToolPolicyResult | None],
        policy_evidence_by_id: dict[str, ToolPolicyEvidence],
        tool_exposure: ResolvedToolExposureAuthority | None,
        tool_round_identity: ToolRoundIdentity,
        model_step: int | None,
        taint_labels_by_id: Mapping[str, frozenset[str]] = MappingProxyType({}),
        publish_arguments_as_unavailable: bool = False,
        policy_output_secret_resolution_scope: Literal["static", "dynamic", "unknown"] = "unknown",
        deferred_terminal_stager: DeferredTerminalStager | None = None,
        deferred_terminal_capture_recorder: DeferredTerminalCaptureRecorder | None = None,
        resolved_redactor_observer: (
            Callable[[str, InvocationRedactorSnapshot], Awaitable[None]] | None
        ) = None,
        publication_snapshot_observer: (
            Callable[
                [str, invocation_secrets.InvocationPublicationSnapshot],
                Awaitable[None],
            ]
            | None
        ) = None,
    ) -> AsyncIterator[tuple[Event, runtime_records.ToolCallOutcome | None]]:
        semaphore = asyncio.Semaphore(self._executor._max_parallel_tool_calls)
        buffers: list[list[tuple[Event, runtime_records.ToolCallOutcome | None]]] = [
            [] for _ in tool_calls
        ]
        child_cancellations: list[asyncio.CancelledError | None] = [None] * len(tool_calls)

        async def execute_call(index: int, tool_call: runtime_records.ToolCallRequest) -> None:
            async with semaphore:
                await self._executor._session_control.raise_if_interrupted(self._session.id)
                try:
                    async for item in self._executor.execute_tool_call(
                        session=self._session,
                        registered_agent=self._registered_agent,
                        registered_environment=self._registered_environment,
                        tool_call=tool_call,
                        request_metadata=self._request_metadata,
                        task_id=self._task_id,
                        model_step=model_step,
                        execution_profile=self._execution_profile,
                        invocation_context=self._invocation_context,
                        policy_result=policy_results_by_id.get(tool_call.id),
                        policy_evidence=policy_evidence_by_id.get(
                            tool_call.id,
                            ToolPolicyEvidence.UNPLANNED,
                        ),
                        tool_exposure=tool_exposure,
                        tool_round_identity=tool_round_identity,
                        taint_labels=taint_labels_by_id.get(tool_call.id, frozenset()),
                        publish_arguments_as_unavailable=publish_arguments_as_unavailable,
                        policy_output_secret_resolution_scope=(
                            policy_output_secret_resolution_scope
                        ),
                        deferred_terminal_stager=deferred_terminal_stager,
                        deferred_terminal_capture_recorder=deferred_terminal_capture_recorder,
                        resolved_redactor_observer=resolved_redactor_observer,
                        publication_snapshot_observer=publication_snapshot_observer,
                    ):
                        buffers[index].append(item)
                except asyncio.CancelledError as exc:
                    child_cancellations[index] = exc
                    raise

        def flush_completed_outcomes() -> None:
            for buffer in buffers:
                for _, outcome in buffer:
                    if outcome is not None:
                        tool_outcomes.append(outcome)

        current_cancellation_failure: BaseExceptionGroup | None = None
        current_cancellation: asyncio.CancelledError | None = None
        current_cancellation_cause: BaseException | None = None
        parent_task = asyncio.current_task()
        cancellation_baseline = 0 if parent_task is None else parent_task.cancelling()

        async def execute_parallel_calls() -> None:
            # Keep TaskGroup's internal parent cancellation on a dedicated task.
            # Python 3.11 and 3.12 can leave that bookkeeping visible after a
            # child failure, which would otherwise look like caller cancellation.
            async with asyncio.TaskGroup() as task_group:
                for index, tool_call in enumerate(tool_calls):
                    task_group.create_task(execute_call(index, tool_call))

        try:
            await asyncio.create_task(execute_parallel_calls())
        except BaseExceptionGroup as exc_group:
            flush_completed_outcomes()
            child_runner_failures: list[BaseException] = []
            observed_runner_failures: set[int] = set()

            def preserve_runner_failure(failure: BaseException | None) -> None:
                if failure is None or id(failure) in observed_runner_failures:
                    return
                observed_runner_failures.add(id(failure))
                child_runner_failures.append(sanitize_runner_failure(failure))

            for child_cancellation in child_cancellations:
                if child_cancellation is not None:
                    preserve_runner_failure(runner_cancellation_failure(child_cancellation))
            for candidate in iter_exception_tree(exc_group):
                if not isinstance(candidate, asyncio.CancelledError):
                    continue
                preserve_runner_failure(pop_runner_cancellation_failure(candidate))
                set_exception_cause(candidate, None)
            cancellation_sources = _ordered_cancellation_sources(
                exc_group,
                child_cancellations=child_cancellations,
                tool_calls=tool_calls,
            )
            if parent_task is not None and parent_task.cancelling() > cancellation_baseline:
                current_cancellation = _authoritative_task_cancellation(
                    parent_task,
                    cancellation_sources=cancellation_sources,
                )
                _transfer_cancellation_evidence(
                    current_cancellation,
                    cancellation_sources,
                    tool_calls=tool_calls,
                )
                if not invocation_secrets.has_cancellation_evidence(current_cancellation):
                    invocation_secrets.initialize_cancellation_evidence(current_cancellation)
                sanitized_group = sanitize_runner_failure_group(
                    exc_group,
                    caller_cancelled=True,
                )
                current_cancellation_cause = _cancellation_failure_cause(
                    sanitized_group,
                    runner_cleanup_failures=child_runner_failures,
                )
                if current_cancellation_cause is not None:
                    attach_runner_cancellation_failure(
                        current_cancellation,
                        current_cancellation_cause,
                    )
                    set_exception_cause(
                        current_cancellation,
                        current_cancellation_cause,
                    )
            elif any(
                isinstance(candidate, BaseExceptionGroup)
                and is_current_runner_cancellation_group(candidate)
                for candidate in iter_exception_tree(exc_group)
            ):
                sanitized_group = sanitize_runner_failure_group(
                    exc_group,
                    caller_cancelled=True,
                )
                current_cancellation_failure = sanitized_group
                if child_runner_failures:
                    current_cancellation_failure = sanitize_runner_failure_group(
                        BaseExceptionGroup(
                            "Parallel tool execution and runner cleanup failures.",
                            [sanitized_group, *child_runner_failures],
                        ),
                        caller_cancelled=True,
                    )
            else:
                parallel_failure = _parallel_tool_round_exception(exc_group)
                if child_runner_failures:
                    failure_cause = _parallel_tool_round_cause(
                        exc_group,
                        primary=parallel_failure,
                        runner_cleanup_failures=child_runner_failures,
                    )
                    raise parallel_failure from failure_cause
                raise parallel_failure from exc_group
        except asyncio.CancelledError as cancellation:
            flush_completed_outcomes()
            transfer_runner_cancellation_failures(
                cancellation,
                child_cancellations,
            )
            _transfer_cancellation_evidence(
                cancellation,
                [
                    (child_exc, tool_calls[index].id)
                    for index, child_exc in enumerate(child_cancellations)
                    if child_exc is not None
                ],
                tool_calls=tool_calls,
            )
            raise
        if current_cancellation is not None:
            raise current_cancellation from current_cancellation_cause
        if current_cancellation_failure is not None:
            raise current_cancellation_failure
        for index, tool_call in enumerate(tool_calls):
            if deferred_terminal_stager is None and all(
                outcome is None for _, outcome in buffers[index]
            ):
                buffers[index].append(
                    await self._abnormal_tool_termination_item(
                        tool_call=tool_call,
                        tool_round_identity=tool_round_identity,
                    )
                )
        for buffer in buffers:
            for item in buffer:
                yield item

    async def _abnormal_tool_termination_item(
        self,
        *,
        tool_call: runtime_records.ToolCallRequest,
        tool_round_identity: ToolRoundIdentity,
    ) -> tuple[Event, runtime_records.ToolCallOutcome]:
        result = ToolResult(
            content="Tool call did not complete: the parallel task terminated abnormally.",
            structured={
                "tool_call_id": tool_call.id,
                "tool_name": tool_call.name,
                "abnormal_termination": True,
            },
            is_error=True,
        )
        payload: dict[str, Any] = {
            "tool_call_id": tool_call.id,
            "idempotency_key": tool_execution.tool_idempotency_key(
                session_id=self._session.id,
                tool_call_id=tool_call.id,
                tool_round_id=tool_round_identity.tool_round_id,
            ),
            "abnormal_termination": True,
            "result": result.model_dump(),
            **tool_argument_publication.unavailable_argument_projection().payload_fields(),
            **copy_tool_round_identity(tool_round_identity).payload(),
        }
        event = await self._executor._event_writer.emit(
            _event_with_tool_round_authority(
                event_with_execution_profile_authority(
                    Event(
                        type=EventType.TOOL_CALL_FAILED,
                        session_id=self._session.id,
                        agent_name=self._registered_agent.spec.name,
                        environment_name=self._environment_name,
                        tool_name=tool_call.name,
                        payload=payload,
                    ),
                    self._execution_profile,
                ),
                tool_round_identity,
            )
        )
        return (
            event,
            runtime_records.ToolCallOutcome(
                call=replace(tool_call, arguments={}),
                result=result,
            ),
        )


def ordered_tool_result_messages(
    tool_calls: list[runtime_records.ToolCallRequest],
    outcomes: list[runtime_records.ToolCallOutcome],
    *,
    parallel: bool,
    tool_round_identity: ToolRoundIdentity,
) -> list[Message]:
    if parallel:
        order = {tool_call.id: index for index, tool_call in enumerate(tool_calls)}
        outcomes = sorted(outcomes, key=lambda outcome: order.get(outcome.call.id, len(order)))
    return transcript_helpers.tool_result_messages(
        outcomes,
        tool_round_identity=tool_round_identity,
    )


def _first_user_input_tool_call(
    registered_agent: runtime_records.RegisteredAgentState,
    tool_calls: list[runtime_records.ToolCallRequest],
    policy_results_by_id: dict[str, ToolPolicyResult | None],
    policy_evidence_by_id: Mapping[str, ToolPolicyEvidence],
) -> tuple[runtime_records.ToolCallRequest, str, list[str]] | None:
    for tool_call in tool_calls:
        registered_tool = registered_agent.executable_tool(tool_call.name)
        if registered_tool is None or not getattr(registered_tool.tool, "pauses_session", False):
            continue
        if policy_evidence_by_id.get(tool_call.id) is not ToolPolicyEvidence.AUTHORITATIVE:
            continue
        policy_result = policy_results_by_id.get(tool_call.id)
        if policy_result is not None and policy_result.decision == ToolPolicyDecision.DENY:
            continue
        question, options = _user_input_prompt(tool_call)
        if question:
            return tool_call, question, options
    return None


def _user_input_prompt(
    tool_call: runtime_records.ToolCallRequest,
) -> tuple[str, list[str]]:
    raw_question = tool_call.arguments.get("question")
    question = raw_question.strip() if isinstance(raw_question, str) else ""
    raw_options = tool_call.arguments.get("options")
    options: list[str] = []
    if isinstance(raw_options, list):
        options = [
            option.strip() for option in raw_options if isinstance(option, str) and option.strip()
        ]
    return question, options


def _tool_call_is_parallel_safe(
    registered_agent: runtime_records.RegisteredAgentState,
    tool_call: runtime_records.ToolCallRequest,
) -> bool:
    registered_tool = registered_agent.executable_tool(tool_call.name)
    return True if registered_tool is None else registered_tool.parallel_safe


def _parallel_tool_round_exception(group: BaseExceptionGroup) -> BaseException:
    flattened = [
        candidate
        for candidate in iter_exception_tree(group)
        if not isinstance(candidate, BaseExceptionGroup)
        or exception_group_children(candidate) is None
    ]
    for exc in flattened:
        if isinstance(exc, SessionInterruptedByRequest | ToolApprovalRequired):
            return exc
    for exc in flattened:
        if not isinstance(exc, asyncio.CancelledError):
            return exc
    return flattened[0]


def _parallel_tool_round_cause(
    group: BaseExceptionGroup,
    *,
    primary: BaseException,
    runner_cleanup_failures: list[BaseException],
) -> BaseException:
    """Preserve non-primary siblings plus detached runner cleanup evidence."""

    remaining: list[BaseException] = []
    for candidate in iter_exception_tree(group):
        if isinstance(candidate, BaseExceptionGroup) or candidate is primary:
            continue
        if any(existing is candidate for existing in remaining):
            continue
        remaining.append(candidate)
    for failure in runner_cleanup_failures:
        if any(existing is failure for existing in remaining):
            continue
        remaining.append(failure)
    if not remaining:
        return RuntimeError("Parallel tool execution failed.")
    if len(remaining) == 1:
        return remaining[0]
    return BaseExceptionGroup(
        "Parallel tool execution and runner cleanup failures.",
        remaining,
    )


def _ordered_cancellation_sources(
    group: BaseExceptionGroup,
    *,
    child_cancellations: list[asyncio.CancelledError | None],
    tool_calls: list[runtime_records.ToolCallRequest],
) -> list[tuple[asyncio.CancelledError, str | None]]:
    """Return authenticated cancellation evidence in tool-call order."""

    sources: list[tuple[asyncio.CancelledError, str | None]] = []
    observed: set[int] = set()
    for index, cancellation in enumerate(child_cancellations):
        if cancellation is None or id(cancellation) in observed:
            continue
        observed.add(id(cancellation))
        sources.append((cancellation, tool_calls[index].id))
    for candidate in iter_exception_tree(group):
        if not isinstance(candidate, asyncio.CancelledError) or id(candidate) in observed:
            continue
        observed.add(id(candidate))
        sources.append(
            (
                candidate,
                invocation_secrets.cancellation_tool_call_id(candidate),
            )
        )
    return sources


def _authoritative_task_cancellation(
    task: asyncio.Task[Any],
    *,
    cancellation_sources: list[tuple[asyncio.CancelledError, str | None]],
) -> asyncio.CancelledError:
    """Detach the parent task's cancellation from child TaskGroup tracebacks."""

    args: tuple[object, ...] = ()
    cancel_message = getattr(task, "_cancel_message", None)
    if cancel_message is not None:
        args = (cancel_message,)
    if not args and cancellation_sources:
        try:
            candidate_args = BaseException.__dict__["args"].__get__(
                cancellation_sources[0][0],
                BaseException,
            )
        except BaseException:
            candidate_args = ()
        if type(candidate_args) is tuple:
            args = candidate_args
    return asyncio.CancelledError(*args)


def _transfer_cancellation_evidence(
    target: asyncio.CancelledError,
    sources: list[tuple[asyncio.CancelledError, str | None]],
    *,
    tool_calls: list[runtime_records.ToolCallRequest],
) -> None:
    """Move authenticated cleanup evidence onto one authoritative cancellation."""

    artifacts_by_id: dict[str, list[dict[str, Any]]] = {}
    redactors_by_id: dict[str, SecretRedactor] = {}
    unassigned_artifacts: list[dict[str, Any]] = []
    public_artifacts: list[dict[str, Any]] = []
    producer_ids: list[str] = []

    def extend_artifacts(
        destination: list[dict[str, Any]],
        artifacts: list[dict[str, Any]],
    ) -> None:
        copied = copy_json_value(artifacts, "cancellation_artifacts")
        if type(copied) is not list:
            return
        for artifact in copied:
            if type(artifact) is not dict:
                continue
            if artifact not in destination:
                destination.append(artifact)
            for public_artifact in sanitize_runner_artifacts([artifact]):
                if public_artifact not in public_artifacts:
                    public_artifacts.append(public_artifact)

    for source, fallback_tool_call_id in sources:
        source_artifacts_by_id = invocation_secrets.cancellation_artifacts_by_id(source)
        if source_artifacts_by_id is not None:
            for tool_call_id, artifacts in source_artifacts_by_id.items():
                extend_artifacts(
                    artifacts_by_id.setdefault(tool_call_id, []),
                    artifacts,
                )
        source_redactors_by_id = invocation_secrets.cancellation_redactors_by_id(source)
        if source_redactors_by_id is not None:
            redactors_by_id.update(source_redactors_by_id)

        source_artifacts = invocation_secrets.cancellation_artifacts(source)
        producer_id = invocation_secrets.cancellation_tool_call_id(source) or fallback_tool_call_id
        if producer_id is not None and producer_id not in producer_ids:
            producer_ids.append(producer_id)
        if source_artifacts:
            if producer_id is not None:
                extend_artifacts(
                    artifacts_by_id.setdefault(producer_id, []),
                    source_artifacts,
                )
            else:
                extend_artifacts(unassigned_artifacts, source_artifacts)
        source_redactor = invocation_secrets.cancellation_redactor(source)
        if source_redactor is not None and producer_id is not None:
            redactors_by_id[producer_id] = source_redactor

    if unassigned_artifacts and len(tool_calls) == 1:
        extend_artifacts(
            artifacts_by_id.setdefault(tool_calls[0].id, []),
            unassigned_artifacts,
        )
        unassigned_artifacts = []
    if not sources:
        return

    invocation_secrets.initialize_cancellation_evidence(target)
    if len(producer_ids) == 1:
        invocation_secrets.set_cancellation_tool_call_id(
            target,
            producer_ids[0],
        )
    if artifacts_by_id:
        invocation_secrets.set_cancellation_artifacts_by_id(
            target,
            artifacts_by_id,
        )
    if redactors_by_id:
        invocation_secrets.set_cancellation_redactors_by_id(
            target,
            redactors_by_id,
        )
    if public_artifacts:
        attach_cancellation_artifacts(target, public_artifacts)


def _grouped_cancellation_evidence(
    group: BaseExceptionGroup,
    *,
    tool_calls: list[runtime_records.ToolCallRequest],
) -> tuple[
    list[dict[str, Any]] | None,
    dict[str, list[dict[str, Any]]] | None,
    dict[str, SecretRedactor] | None,
]:
    """Project grouped cancellation leaves onto interrupted-round evidence."""

    sources = [
        (
            candidate,
            invocation_secrets.cancellation_tool_call_id(candidate),
        )
        for candidate in iter_exception_tree(group)
        if isinstance(candidate, asyncio.CancelledError)
    ]
    if not sources:
        return None, None, None
    cancellation = asyncio.CancelledError()
    _transfer_cancellation_evidence(
        cancellation,
        sources,
        tool_calls=tool_calls,
    )
    artifacts_by_id = invocation_secrets.cancellation_artifacts_by_id(cancellation)
    redactors_by_id = invocation_secrets.cancellation_redactors_by_id(cancellation)
    artifacts = invocation_secrets.cancellation_artifacts(cancellation)
    if artifacts_by_id is not None:
        artifacts = []
    return (
        artifacts or None,
        artifacts_by_id,
        redactors_by_id,
    )


def _cancellation_failure_cause(
    group: BaseExceptionGroup,
    *,
    runner_cleanup_failures: list[BaseException],
) -> BaseException | None:
    """Retain sanitized non-cancellation failures beside parent cancellation."""

    failures: list[BaseException] = []
    for candidate in iter_exception_tree(group):
        if isinstance(candidate, (BaseExceptionGroup, asyncio.CancelledError)):
            continue
        if any(existing is candidate for existing in failures):
            continue
        failures.append(candidate)
    for failure in runner_cleanup_failures:
        if any(existing is failure for existing in failures):
            continue
        failures.append(failure)
    if not failures:
        return None
    if len(failures) == 1:
        return failures[0]
    return BaseExceptionGroup(
        "Parallel tool execution and runner cleanup failures.",
        failures,
    )


def _raise_after_interrupted_round_closure_failure(
    interrupt: asyncio.CancelledError | BaseExceptionGroup,
    *,
    closure_error: BaseException,
    restore_cancellation_requests: int = 0,
    minimum_cancellation_requests: int = 0,
) -> Never:
    """Keep interruption authoritative when its durable closure also fails."""

    if _contains_process_signal(closure_error):
        _restore_current_task_cancellation_requests(
            consumed_requests=restore_cancellation_requests,
            minimum_requests=minimum_cancellation_requests,
        )
        interrupt_nodes = {id(candidate) for candidate in iter_exception_tree(interrupt)}
        closure_nodes = {id(candidate) for candidate in iter_exception_tree(closure_error)}
        if interrupt_nodes & closure_nodes:
            raise closure_error
        raise BaseExceptionGroup(
            "Tool-round interruption and closure process control.",
            [interrupt, closure_error],
        ) from None
    if isinstance(interrupt, asyncio.CancelledError):
        invocation_secrets.sanitize_external_cancellation(interrupt)
    closure_failure = RuntimeError("Interrupted tool-round closure failed.")
    prior_cause = exception_cause(interrupt)
    if prior_cause is None:
        combined_cause: BaseException = closure_failure
    else:
        combined_cause = BaseExceptionGroup(
            "Interrupted tool-round cleanup and closure failures.",
            [prior_cause, closure_failure],
        )
    set_exception_cause(interrupt, combined_cause)
    _restore_current_task_cancellation_requests(
        consumed_requests=restore_cancellation_requests,
        minimum_requests=minimum_cancellation_requests,
    )
    raise interrupt from combined_cause


def _restore_current_task_cancellation_requests(
    *,
    consumed_requests: int = 0,
    minimum_requests: int = 0,
) -> None:
    """Restore consumed requests without duplicating requests still pending."""

    if type(consumed_requests) is not int or consumed_requests < 0:
        raise ValueError("Consumed cancellation requests must be a non-negative int.")
    if type(minimum_requests) is not int or minimum_requests < 0:
        raise ValueError("Minimum cancellation requests must be a non-negative int.")
    current_task = asyncio.current_task()
    if current_task is None:
        return
    for _request in range(consumed_requests):
        current_task.cancel()
    for _request in range(max(minimum_requests - current_task.cancelling(), 0)):
        current_task.cancel()


def _consume_current_task_cancellation_requests(
    cancellation: asyncio.CancelledError,
) -> int:
    """Consume requests for owned cleanup and return the exact removed count."""

    current_task = asyncio.current_task()
    requests_before = 0 if current_task is None else current_task.cancelling()
    consume_pending_task_cancellation(cancellation)
    if current_task is None:
        return 0
    return max(requests_before - current_task.cancelling(), 0)


def _interrupted_tool_call_outcome(
    *,
    tool_call: runtime_records.ToolCallRequest,
    tool_round_identity: ToolRoundIdentity,
    registered_tool: runtime_records.RegisteredTool | None,
    execution_started: bool,
    artifacts: list[dict[str, Any]] | None = None,
) -> runtime_records.ToolCallOutcome:
    """Build the canonical bounded result for one interrupted tool call."""

    structured = {
        "interrupted": True,
        "tool_call_id": tool_call.id,
        "tool_name": tool_call.name,
        **copy_tool_round_identity(tool_round_identity).payload(),
    }
    if execution_started and registered_tool is not None:
        try:
            execution_contract = ToolExecutionContract.model_validate(
                registered_tool.execution_contract
            )
        except (TypeError, ValueError):
            execution_contract = None
        if execution_contract is not None and execution_contract.boundary == "posix_process":
            process_controls = {
                "terminal_outcome": "tool_execution_error",
                "tool_effect": registered_tool.effect.value,
                "outcome_unknown": registered_tool.effect is not ToolEffect.NONE,
                "manual_reconciliation_required": (registered_tool.effect is ToolEffect.EXTERNAL),
                "isolated_tool_failure_code": "process_interrupted",
                "tool_execution_boundary": "posix_process",
                "tool_timeout_strength": "hard_process_deadline",
            }
            tool_results.runtime_terminal_controls(process_controls)
            tool_results.runtime_tool_execution_boundary_controls(process_controls)
            structured.update(process_controls)
    return runtime_records.ToolCallOutcome(
        call=tool_call,
        result=ToolResult(
            content="Tool call interrupted before completion.",
            structured=structured,
            artifacts=[] if artifacts is None else artifacts,
            is_error=True,
        ),
    )


def _interrupted_tool_call_event(
    *,
    session: Session,
    registered_agent: runtime_records.RegisteredAgentState,
    registered_environment: runtime_records.RegisteredEnvironment | None,
    tool_call_outcome: runtime_records.ToolCallOutcome,
    tool_round_identity: ToolRoundIdentity,
) -> Event:
    """Build the terminal event paired with the canonical interrupted result."""

    structured = dict(tool_call_outcome.result.structured or {})
    terminal_controls = tool_results.runtime_terminal_controls(structured)
    terminal_controls.update(tool_results.runtime_tool_execution_boundary_controls(structured))
    return Event(
        type=(
            EventType.TOOL_CALL_FAILED
            if tool_call_outcome.result.is_error
            else EventType.TOOL_CALL_COMPLETED
        ),
        session_id=session.id,
        agent_name=registered_agent.spec.name,
        environment_name=_environment_name(registered_environment),
        tool_name=tool_call_outcome.call.name,
        payload={
            "tool_call_id": tool_call_outcome.call.id,
            "idempotency_key": tool_execution.tool_idempotency_key(
                session_id=session.id,
                tool_round_id=tool_round_identity.tool_round_id,
                tool_call_id=tool_call_outcome.call.id,
            ),
            "interrupted": True,
            "result": tool_call_outcome.result.model_dump(),
            **terminal_controls,
            **copy_tool_round_identity(tool_round_identity).payload(),
        },
    )


def _copy_agent_spec(spec: AgentSpec) -> AgentSpec:
    if type(spec) is not AgentSpec:
        raise TypeError("Agent registration requires an AgentSpec.")
    return AgentSpec(
        name=spec.name,
        model=spec.model,
        provider_name=spec.provider_name,
        system_prompt=spec.system_prompt,
        workflow_tool_names=spec.workflow_tool_names,
        authoring_state=spec.authoring_state,
        metadata=copy_json_value(spec.metadata, "metadata"),
        provider_options=copy_json_value(spec.provider_options, "provider_options"),
        thinking=spec.thinking,
    )


def _environment_name(
    registered_environment: runtime_records.RegisteredEnvironment | None,
) -> str | None:
    return None if registered_environment is None else registered_environment.spec.name


def _workspace_id(
    registered_environment: runtime_records.RegisteredEnvironment | None,
) -> str | None:
    if registered_environment is None or registered_environment.environment.workspace is None:
        return None
    workspace_id = getattr(registered_environment.environment.workspace, "id", None)
    if workspace_id is None:
        return None
    return require_clean_nonblank(workspace_id, "workspace.id")


def _workspace_binding_generation_id(
    registered_environment: runtime_records.RegisteredEnvironment,
) -> str:
    value = registered_environment.binding_generation_id
    value = require_clean_nonblank(value, "binding_generation_id")
    if not value.startswith("wbind_") or len(value) != 38:
        raise ValueError("Workspace binding generation identity is malformed.")
    return value


def _workspace_revision_observer_name(
    registered_environment: runtime_records.RegisteredEnvironment | None,
) -> str:
    if registered_environment is None:
        return "UnconfiguredEnvironment"
    binding = registered_environment.environment.binding
    if binding is None:
        return "UnconfiguredWorkspaceBinding"
    return type(binding).__name__


def _workspace_revision_observer_is_runtime_owned(
    registered_environment: runtime_records.RegisteredEnvironment | None,
) -> bool:
    if registered_environment is None:
        return True
    binding = registered_environment.environment.binding
    if binding is None:
        return True
    return _runtime_owned_workspace_observer_name(binding) is not None


def _workspace_writer_isolation(
    registered_environment: runtime_records.RegisteredEnvironment | None,
) -> WorkspaceWriterIsolationEvidence:
    """Copy adapter isolation evidence without treating malformed claims as proof."""

    if registered_environment is None:
        return WorkspaceWriterIsolationEvidence()
    binding = registered_environment.environment.binding
    if binding is None:
        return WorkspaceWriterIsolationEvidence()
    bound = registered_environment.bound_workspace
    if bound is None:
        workspace = registered_environment.environment.workspace
        bound = BoundWorkspace(
            workspace=workspace,
            source_workspace=workspace,
            runner=registered_environment.environment.runner,
        )
    try:
        evidence = binding.observe_writer_isolation(bound)
        if type(evidence) is not WorkspaceWriterIsolationEvidence:
            raise TypeError("Workspace binding returned invalid writer-isolation evidence.")
        return WorkspaceWriterIsolationEvidence.model_validate(evidence.model_dump(mode="python"))
    except Exception:
        return WorkspaceWriterIsolationEvidence(detail_code="writer_isolation_observer_failed")


async def _observe_workspace_revision(
    registered_environment: runtime_records.RegisteredEnvironment | None,
    *,
    operation_registry: BoundedInvocationOperationRegistry,
    require_quiescence_before_return: bool = False,
    authority_projection: _WorkspaceObservationAuthorityProjection | None = None,
) -> WorkspaceRevisionObservation:
    if type(require_quiescence_before_return) is not bool:
        raise TypeError("require_quiescence_before_return must be a boolean.")
    workspace_id = _workspace_id(registered_environment) or "workspace-unavailable"
    observer = _workspace_revision_observer_name(registered_environment)
    expected_identity = WorkspaceIdentity(workspace_id=workspace_id, observer=observer)
    durable_identity = (
        expected_identity
        if authority_projection is None
        else WorkspaceIdentity(
            workspace_id=authority_projection.workspace_id,
            observer=authority_projection.observer,
        )
    )
    if authority_projection is not None and (
        authority_projection.configured_identity
        != (expected_identity.workspace_id, expected_identity.observer)
    ):
        raise RuntimeError("Workspace observation authority changed before capture.")
    if registered_environment is None:
        return unsupported_workspace_revision(
            workspace_id=durable_identity.workspace_id,
            observer=durable_identity.observer,
        )
    binding = registered_environment.environment.binding
    bound = registered_environment.bound_workspace
    if binding is None:
        return unsupported_workspace_revision(
            workspace_id=durable_identity.workspace_id,
            observer=durable_identity.observer,
        )
    if bound is None:
        workspace = registered_environment.environment.workspace
        bound = BoundWorkspace(
            workspace=workspace,
            source_workspace=workspace,
            runner=registered_environment.environment.runner,
        )
    if not operation_registry.reserve():
        return WorkspaceRevisionObservation(
            identity=durable_identity,
            status=WorkspaceRevisionObservationStatus.FAILED,
            detail_code="revision_observer_capacity_exhausted",
        )

    try:
        observation_task = asyncio.create_task(
            capture_awaitable_outcome(lambda: binding.observe_revision(bound))
        )
    except BaseException:
        operation_registry.release_reservation()
        raise
    operation_registry.track(observation_task)
    try:
        outcome = await await_shielded_task_outcome(
            observation_task,
            timeout_s=_WORKSPACE_OBSERVATION_TIMEOUT_SECONDS,
            timeout_after_cancellation_s=0.0,
        )
    except BaseException:
        if not observation_task.done():
            registered_environment.workspace_mutation_fence.fail_closed(
                workspace_mutation_task_settlement_probe(observation_task)
            )
        raise
    if observation_task.done():
        operation_registry.release(observation_task)
    observed: object = None
    observer_error = outcome.error
    if observer_error is None:
        captured = outcome.result
        if type(captured) is not CapturedAwaitableOutcome:
            observer_error = RuntimeError(
                "Workspace revision observer returned an invalid owned outcome."
            )
        else:
            observed = captured.result
            observer_error = captured.error
    observer_settlement_unproven = observer_error is not None and any(
        isinstance(candidate, asyncio.CancelledError)
        for candidate in iter_exception_tree(observer_error)
    )
    if observer_settlement_unproven:
        # A child-originated cancellation proves only that the observer
        # coroutine stopped.  It does not prove that thread-, executor-, SDK-,
        # subprocess-, or remote-backed work dispatched by that observer also
        # stopped.  Retain a fail-closed environment owner even when an
        # independent caller cancellation arrived in the same scheduling turn.
        registered_environment.workspace_mutation_fence.fail_closed(
            workspace_mutation_task_settlement_probe(observation_task)
        )
    if outcome.cancellation is not None and not observation_task.done():
        registered_environment.workspace_mutation_fence.fail_closed(
            workspace_mutation_task_settlement_probe(observation_task)
        )
    if outcome.cancellation is not None:
        restore_workspace_observation_cancellation_requests(outcome.cancellation_requests_consumed)
    raise_workspace_observation_concurrent_control(
        cancellation=outcome.cancellation,
        error=observer_error,
        operation="Workspace revision observation",
        cancellation_requests_pending=outcome.cancellation_requests_consumed,
    )
    if outcome.timed_out:
        if not observation_task.done():
            registered_environment.workspace_mutation_fence.fail_closed(
                workspace_mutation_task_settlement_probe(observation_task)
            )
            # The tool must not begin mutating while a cancellation-opaque
            # before-observer may still be reading the same workspace.  The
            # retained fence permits a later invocation only after the exact
            # observer task settles.
            if require_quiescence_before_return:
                raise WorkspaceMutationSettlementError(
                    "Workspace revision observation did not settle after its deadline."
                ) from None
        return WorkspaceRevisionObservation(
            identity=durable_identity,
            status=WorkspaceRevisionObservationStatus.FAILED,
            detail_code="revision_observer_timeout",
        )
    if observer_error is not None and any(
        isinstance(candidate, (GeneratorExit, KeyboardInterrupt, SystemExit))
        for candidate in iter_exception_tree(observer_error)
    ):
        if observer_error is None:  # pragma: no cover - narrowed by the helper
            raise AssertionError("Process-signal classification lost its error.")
        raise observer_error
    if observer_settlement_unproven and require_quiescence_before_return:
        raise WorkspaceMutationSettlementError(
            "Workspace revision observation did not prove mutation quiescence."
        ) from None
    if observer_error is not None:
        return WorkspaceRevisionObservation(
            identity=durable_identity,
            status=WorkspaceRevisionObservationStatus.FAILED,
            detail_code="revision_observer_failed",
        )
    try:
        validated = _bounded_workspace_observation_copy(
            observed,
            expected_identity=expected_identity,
        )
    except WorkspaceRevisionObservationLimitExceeded:
        return WorkspaceRevisionObservation(
            identity=durable_identity,
            status=WorkspaceRevisionObservationStatus.TRUNCATED,
            detail_code="revision_observer_limit_exceeded",
        )
    except Exception:
        return WorkspaceRevisionObservation(
            identity=durable_identity,
            status=WorkspaceRevisionObservationStatus.FAILED,
            detail_code="revision_observer_failed",
        )
    return validated.model_copy(update={"identity": durable_identity})


def _contains_process_signal(error: BaseException | None) -> bool:
    if error is None:
        return False
    return any(
        isinstance(candidate, (GeneratorExit, KeyboardInterrupt, SystemExit))
        for candidate in iter_exception_tree(error)
    )


_PostToolResultT = TypeVar("_PostToolResultT")


def _raise_preserved_post_tool_cancellation(
    cancellation: asyncio.CancelledError | None,
    failure: BaseException,
    *,
    restore_cancellation_requests: int,
) -> Never:
    """Keep an earlier caller cancellation authoritative over terminalization failure."""

    if cancellation is None or type(failure) is GeneratorExit or _contains_process_signal(failure):
        raise failure
    # Terminalization crosses stores, hooks, projections, and other extension
    # seams. Preserve the fact of the secondary failure without retaining its
    # potentially workload-derived message, traceback, or mutable state on the
    # public cancellation object.
    safe_failure = RuntimeError("Tool terminalization failed after caller cancellation.")
    del failure
    prior_cause = exception_cause(cancellation)
    if prior_cause is None:
        cause: BaseException = safe_failure
    else:
        cause = BaseExceptionGroup(
            "Post-tool cancellation and terminalization failures.",
            [prior_cause, safe_failure],
        )
    attach_runner_cancellation_failure(cancellation, cause)
    set_exception_cause(cancellation, cause)
    _raise_restored_post_tool_cancellation(
        cancellation,
        restore_cancellation_requests=restore_cancellation_requests,
        cause=cause,
    )


def _raise_restored_post_tool_cancellation(
    cancellation: asyncio.CancelledError,
    *,
    restore_cancellation_requests: int,
    cause: BaseException | None = None,
) -> Never:
    """Redeliver owned cancellation without erasing Task.cancelling() evidence."""

    current_task = asyncio.current_task()
    if current_task is not None:
        for _request in range(restore_cancellation_requests):
            current_task.cancel()
    raise cancellation from cause


async def _await_post_tool_operation(
    operation: Awaitable[_PostToolResultT],
    *,
    cancellation: asyncio.CancelledError | None,
    restore_cancellation_requests: int,
) -> _PostToolResultT:
    """Await terminalization without allowing it to replace owned cancellation."""

    if cancellation is None:
        return await operation
    try:
        return await operation
    except BaseException as failure:
        _raise_preserved_post_tool_cancellation(
            cancellation,
            failure,
            restore_cancellation_requests=restore_cancellation_requests,
        )


async def _iterate_post_tool_events(
    events: AsyncIterator[_PostToolResultT],
    *,
    cancellation: asyncio.CancelledError | None,
    restore_cancellation_requests: int,
) -> AsyncIterator[_PostToolResultT]:
    """Iterate terminalization events under the same cancellation authority."""

    if cancellation is None:
        async for event in events:
            yield event
        return
    try:
        async for event in events:
            yield event
    except BaseException as failure:
        _raise_preserved_post_tool_cancellation(
            cancellation,
            failure,
            restore_cancellation_requests=restore_cancellation_requests,
        )


def _bounded_workspace_observation_copy(
    observed: object,
    *,
    expected_identity: WorkspaceIdentity,
) -> WorkspaceRevisionObservation:
    """Detach public observer output only after enforcing runtime hard limits."""
    return copy_bounded_workspace_revision_observation(
        observed,
        expected_identity=expected_identity,
        limits=_WORKSPACE_OBSERVATION_RUNTIME_LIMITS,
        max_total_paths=_WORKSPACE_OBSERVATION_MAX_TOTAL_PATHS,
    )


async def _record_workspace_mutation_after(
    *,
    session_store: SessionStore,
    event_writer: RuntimeEventWriter,
    registered_environment: runtime_records.RegisteredEnvironment | None,
    artifact_store: Any,
    artifact_unavailable_detail_code: str,
    session: Session,
    registered_agent: runtime_records.RegisteredAgentState,
    environment_name: str | None,
    tool_call: runtime_records.ToolCallRequest,
    tool_round_identity: ToolRoundIdentity,
    execution_profile: ExecutionProfileIdentity | None,
    model_step: int | None,
    window_id: str,
    lifecycle: WorkspaceObservationLifecycle,
    before_observation: WorkspaceRevisionObservation,
    authority_projection: _WorkspaceObservationAuthorityProjection,
    attribution_window: WorkspaceMutationWindow,
    writer_isolation_before: WorkspaceWriterIsolationEvidence,
    direct_mutations: DirectWorkspaceMutationCollector,
    redactor: SecretRedactor,
    evidence_available: bool,
    operation_registry: BoundedInvocationOperationRegistry,
) -> _WorkspaceCaptureResult:
    """Persist reconstructible evidence up to, but not including, terminal closure."""

    if registered_environment is None:
        raise RuntimeError("Workspace capture requires its registered environment owner.")
    if lifecycle.phase is not WorkspaceObservationPhase.TOOL_OUTCOME_STAGED:
        raise RuntimeError("Workspace capture requires a durably staged tool outcome.")
    if lifecycle.window_id != window_id or lifecycle.session_id != session.id:
        raise RuntimeError("Workspace capture lifecycle conflicts with its invocation.")
    if authority_projection.configured_artifact_store_id != _artifact_store_id(
        registered_environment
    ):
        raise RuntimeError("Workspace artifact-store authority changed before capture.")
    if (
        before_observation.identity.workspace_id != lifecycle.workspace_id
        or before_observation.identity.observer != lifecycle.observer
    ):
        raise RuntimeError("Workspace capture before evidence conflicts with its authority.")

    active_lifecycle = lifecycle

    async def record_artifact_state(artifact: WorkspaceObservationArtifact) -> None:
        nonlocal active_lifecycle
        existing_by_kind = {item.evidence_kind: item for item in active_lifecycle.artifacts}
        existing = existing_by_kind.get(artifact.evidence_kind)
        if existing is not None and (
            existing.artifact_id != artifact.artifact_id
            or existing.sha256 != artifact.sha256
            or existing.size_bytes != artifact.size_bytes
        ):
            raise RuntimeError("Workspace artifact identity changed across publication.")
        existing_by_kind[artifact.evidence_kind] = artifact
        updated = WorkspaceObservationLifecycle.model_validate(
            {
                **active_lifecycle.model_dump(mode="json"),
                "artifacts": [
                    existing_by_kind[key].model_dump(mode="json")
                    for key in sorted(existing_by_kind)
                ],
            }
        )
        await publish_workspace_observation_transition(
            session_store=session_store,
            event_writer=event_writer,
            session=session,
            previous=active_lifecycle,
            current=updated,
            phase=f"artifact-{artifact.evidence_kind}-{artifact.state.value}",
        )
        active_lifecycle = updated

    def lifecycle_with_referenced_artifact(
        value: WorkspaceObservationLifecycle,
        projection: _WorkspaceEvidenceProjection,
    ) -> WorkspaceObservationLifecycle:
        if projection.manifest_artifact_id is None:
            return value
        artifacts = []
        matched = False
        for artifact in value.artifacts:
            if artifact.artifact_id != projection.manifest_artifact_id:
                artifacts.append(artifact)
                continue
            matched = True
            if artifact.state is not WorkspaceObservationArtifactState.PUBLISHED:
                raise RuntimeError("Workspace event references an unpublished artifact.")
            artifacts.append(
                artifact.model_copy(update={"state": WorkspaceObservationArtifactState.REFERENCED})
            )
        if not matched:
            raise RuntimeError("Workspace event references an unknown artifact.")
        return WorkspaceObservationLifecycle.model_validate(
            {
                **value.model_dump(mode="json"),
                "artifacts": [artifact.model_dump(mode="json") for artifact in artifacts],
            }
        )

    before_projection = await _workspace_evidence_projection(
        paths=[path.model_dump(mode="json") for path in before_observation.paths],
        status=before_observation.status.value,
        detail_code=before_observation.detail_code,
        artifact_store=artifact_store,
        artifact_unavailable_detail_code=artifact_unavailable_detail_code,
        session=session,
        registered_agent=registered_agent,
        environment_name=environment_name,
        window_id=window_id,
        evidence_kind="revision-before",
        redactor=redactor,
        evidence_available=evidence_available,
        operation_registry=operation_registry,
        artifact_state_recorder=record_artifact_state,
        registered_environment=registered_environment,
    )
    before_event = prepare_runtime_event(
        _workspace_revision_observed_event(
            session=session,
            registered_agent=registered_agent,
            environment_name=environment_name,
            tool_call=tool_call,
            tool_round_identity=tool_round_identity,
            execution_profile=execution_profile,
            model_step=model_step,
            window_id=window_id,
            binding_generation_id=lifecycle.binding_generation_id,
            artifact_store_id=lifecycle.artifact_store_id,
            observer_is_runtime_owned=(lifecycle.observer_authority == "runtime_builtin"),
            interaction_id=lifecycle.interaction_id,
            phase="before",
            observation=before_observation,
            projection=before_projection,
        ),
        redactor=redactor,
    )
    before_state = (
        WorkspaceObservationEvidenceState.QUARANTINED
        if not before_projection.evidence_available
        else (
            WorkspaceObservationEvidenceState.FAILED
            if before_observation.status is WorkspaceRevisionObservationStatus.FAILED
            or before_projection.status != before_observation.status.value
            else WorkspaceObservationEvidenceState.PUBLISHED
        )
    )
    before_published = WorkspaceObservationLifecycle.model_validate(
        {
            **lifecycle_with_referenced_artifact(
                active_lifecycle,
                before_projection,
            ).model_dump(mode="json"),
            "before_state": before_state.value,
            "before_observation_id": before_event.id,
        }
    )
    (before_event,) = await publish_workspace_observation_transition(
        session_store=session_store,
        event_writer=event_writer,
        session=session,
        previous=active_lifecycle,
        current=before_published,
        phase="before-evidence",
        events=(before_event,),
    )
    active_lifecycle = before_published
    try:
        after_observation = await _observe_workspace_revision(
            registered_environment,
            operation_registry=operation_registry,
            authority_projection=authority_projection,
        )
        writer_isolation_after = _workspace_writer_isolation(registered_environment)
    except BaseException:
        attribution_window.close(discard_history=not evidence_available)
        raise
    if (
        after_observation.identity.workspace_id != lifecycle.workspace_id
        or after_observation.identity.observer != lifecycle.observer
    ):
        attribution_window.close(discard_history=not evidence_available)
        raise RuntimeError("Workspace capture after evidence conflicts with its authority.")
    try:
        try:
            delta = compare_workspace_revisions(before_observation, after_observation)
        except Exception:
            delta = WorkspaceRevisionDelta(
                identity=before_observation.identity,
                status=WorkspaceRevisionDeltaStatus.FAILED,
                before_revision=before_observation.revision,
                after_revision=after_observation.revision,
                detail_code="revision_comparison_failed",
            )
        direct_reconciliation = (
            reconcile_direct_workspace_mutations(
                before=before_observation,
                after=after_observation,
                collector=direct_mutations,
            )
            if evidence_available
            else WorkspaceDirectMutationReconciliation.NOT_OBSERVED
        )
        attribution = classify_workspace_mutation_attribution(
            window=attribution_window,
            isolation_before=writer_isolation_before,
            isolation_after=writer_isolation_after,
            direct_reconciliation=direct_reconciliation,
        )
        if not evidence_available:
            attribution = WorkspaceMutationAttribution(
                confidence=(
                    WorkspaceMutationAttributionConfidence.CONCURRENT_AMBIGUITY
                    if attribution_window.overlap_detected
                    else WorkspaceMutationAttributionConfidence.EXTERNAL_OR_UNKNOWN
                ),
                writer_isolation=attribution.writer_isolation,
                overlap_detected=attribution_window.overlap_detected,
                direct_reconciliation=WorkspaceDirectMutationReconciliation.NOT_OBSERVED,
                detail_code=(
                    "overlapping_workspace_mutation_windows"
                    if attribution_window.overlap_detected
                    else "workspace_evidence_quarantined"
                ),
            )
        pre_window_change = (
            observed_pre_window_change(
                attribution_window,
                before_observation,
            )
            if evidence_available
            else None
        )
    finally:
        attribution_window.close(
            after_observation if evidence_available else None,
            discard_history=not evidence_available,
        )
    after_projection = await _workspace_evidence_projection(
        paths=[path.model_dump(mode="json") for path in after_observation.paths],
        status=after_observation.status.value,
        detail_code=after_observation.detail_code,
        artifact_store=artifact_store,
        artifact_unavailable_detail_code=artifact_unavailable_detail_code,
        session=session,
        registered_agent=registered_agent,
        environment_name=environment_name,
        window_id=window_id,
        evidence_kind="revision-after",
        redactor=redactor,
        evidence_available=evidence_available,
        operation_registry=operation_registry,
        artifact_state_recorder=record_artifact_state,
        registered_environment=registered_environment,
    )
    after_event = prepare_runtime_event(
        _workspace_revision_observed_event(
            session=session,
            registered_agent=registered_agent,
            environment_name=environment_name,
            tool_call=tool_call,
            tool_round_identity=tool_round_identity,
            execution_profile=execution_profile,
            model_step=model_step,
            window_id=window_id,
            binding_generation_id=lifecycle.binding_generation_id,
            artifact_store_id=lifecycle.artifact_store_id,
            observer_is_runtime_owned=(lifecycle.observer_authority == "runtime_builtin"),
            interaction_id=lifecycle.interaction_id,
            phase="after",
            observation=after_observation,
            projection=after_projection,
        ),
        redactor=redactor,
    )
    after_state = (
        WorkspaceObservationEvidenceState.QUARANTINED
        if not after_projection.evidence_available
        else (
            WorkspaceObservationEvidenceState.FAILED
            if after_observation.status is WorkspaceRevisionObservationStatus.FAILED
            or after_projection.status != after_observation.status.value
            else WorkspaceObservationEvidenceState.PUBLISHED
        )
    )
    after_captured = WorkspaceObservationLifecycle.model_validate(
        {
            **lifecycle_with_referenced_artifact(
                active_lifecycle,
                after_projection,
            ).model_dump(mode="json"),
            "phase": WorkspaceObservationPhase.AFTER_CAPTURED.value,
            "after_state": after_state.value,
            "after_observation_id": after_event.id,
        }
    )
    (after_event,) = await publish_workspace_observation_transition(
        session_store=session_store,
        event_writer=event_writer,
        session=session,
        previous=active_lifecycle,
        current=after_captured,
        phase="after-capture",
        events=(after_event,),
    )
    active_lifecycle = after_captured
    delta_projection = await _workspace_evidence_projection(
        paths=[path.model_dump(mode="json") for path in delta.paths],
        status=delta.status.value,
        detail_code=delta.detail_code,
        artifact_store=artifact_store,
        artifact_unavailable_detail_code=artifact_unavailable_detail_code,
        session=session,
        registered_agent=registered_agent,
        environment_name=environment_name,
        window_id=window_id,
        evidence_kind="revision-delta",
        redactor=redactor,
        evidence_available=evidence_available,
        operation_registry=operation_registry,
        artifact_state_recorder=record_artifact_state,
        registered_environment=registered_environment,
    )
    receipt_event = prepare_runtime_event(
        _workspace_mutation_recorded_event(
            session=session,
            registered_agent=registered_agent,
            environment_name=environment_name,
            tool_call=tool_call,
            tool_round_identity=tool_round_identity,
            execution_profile=execution_profile,
            model_step=model_step,
            window_id=window_id,
            binding_generation_id=lifecycle.binding_generation_id,
            artifact_store_id=lifecycle.artifact_store_id,
            observer_is_runtime_owned=(lifecycle.observer_authority == "runtime_builtin"),
            interaction_id=lifecycle.interaction_id,
            tool_outcome_event_id=(lifecycle.tool_outcome_event_id or ""),
            tool_outcome_event_digest=(lifecycle.tool_outcome_event_digest or ""),
            before_observation_id=before_event.id,
            after_observation_id=after_event.id,
            delta=delta,
            projection=delta_projection,
            attribution=attribution,
            writer_isolation_before=writer_isolation_before,
            writer_isolation_after=writer_isolation_after,
            direct_mutations=direct_workspace_mutation_payload(
                direct_mutations,
                window_id=window_id,
                evidence_available=evidence_available,
            ),
            pre_window_change=pre_window_change,
        ),
        redactor=redactor,
    )
    terminal_status, terminal_detail_code = workspace_observation_terminal_from_delta_status(
        delta_projection.status,
        detail_code=delta_projection.detail_code,
    )
    projection_changed_evidence = (
        before_projection.status != before_observation.status.value
        or after_projection.status != after_observation.status.value
        or delta_projection.status != delta.status.value
        or not before_projection.evidence_available
        or not after_projection.evidence_available
        or not delta_projection.evidence_available
    )
    if projection_changed_evidence:
        terminal_status = WorkspaceObservationTerminalStatus.INCOMPLETE
        terminal_detail_code = "workspace_revision_evidence_incomplete"
    return _WorkspaceCaptureResult(
        lifecycle=active_lifecycle,
        delta_lifecycle=lifecycle_with_referenced_artifact(
            active_lifecycle,
            delta_projection,
        ),
        events=(before_event, after_event),
        receipt_event=receipt_event,
        terminal_status=terminal_status,
        terminal_detail_code=terminal_detail_code,
    )


def _workspace_mutation_window_id(
    *,
    session_id: str,
    session_run_epoch: int,
    tool_round_id: str,
    tool_call_id: str,
) -> str:
    material = json.dumps(
        [session_id, session_run_epoch, tool_round_id, tool_call_id],
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return "wmut_" + hashlib.sha256(material).hexdigest()


def _workspace_observation_event_id(*, window_id: str, phase: str) -> str:
    material = json.dumps(
        [window_id, phase],
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return "wevt_" + hashlib.sha256(material).hexdigest()


def _workspace_revision_observed_event(
    *,
    session: Session,
    registered_agent: runtime_records.RegisteredAgentState,
    environment_name: str | None,
    tool_call: runtime_records.ToolCallRequest,
    tool_round_identity: ToolRoundIdentity,
    execution_profile: ExecutionProfileIdentity | None,
    model_step: int | None,
    window_id: str,
    binding_generation_id: str,
    artifact_store_id: str | None,
    observer_is_runtime_owned: bool,
    interaction_id: str | None,
    phase: Literal["before", "after"],
    observation: WorkspaceRevisionObservation,
    projection: _WorkspaceEvidenceProjection,
) -> Event:
    payload: dict[str, Any] = {
        "phase": phase,
        "window_id": window_id,
        "session_run_epoch": session.run_epoch,
        "binding_generation_id": binding_generation_id,
        "artifact_store_id": artifact_store_id,
        "tool_call_id": tool_call.id,
        "workspace_id": observation.identity.workspace_id,
        "observer": observation.identity.observer,
        "status": projection.status,
        "revision": observation.revision if projection.evidence_available else None,
        "head_revision": (observation.head_revision if projection.evidence_available else None),
        "branch": observation.branch if projection.evidence_available else None,
        "path_scope": observation.path_scope,
        "paths": projection.paths,
        "total_paths": observation.total_paths if projection.evidence_available else 0,
        "detail_code": projection.detail_code,
        **tool_round_identity.payload(),
    }
    if model_step is not None:
        payload["model_step"] = model_step
    _add_workspace_manifest_artifact_fields(payload, projection)
    event = event_with_runtime_generated_id(
        Event(
            id=_workspace_observation_event_id(window_id=window_id, phase=phase),
            type=EventType.WORKSPACE_REVISION_OBSERVED,
            session_id=session.id,
            interaction_id=interaction_id,
            agent_name=registered_agent.spec.name,
            environment_name=environment_name,
            tool_name=tool_call.name,
            payload=payload,
        )
    )
    return _event_with_workspace_observation_authority(
        event_with_execution_profile_authority(event, execution_profile),
        tool_round_identity,
        interaction_id,
        "tool_call_id",
        "window_id",
        "binding_generation_id",
        "workspace_id",
        *(("observer",) if observer_is_runtime_owned else ()),
        *(() if artifact_store_id is None else ("artifact_store_id",)),
        "manifest_artifact_id",
        "manifest_artifact_sha256",
    )


def _workspace_mutation_recorded_event(
    *,
    session: Session,
    registered_agent: runtime_records.RegisteredAgentState,
    environment_name: str | None,
    tool_call: runtime_records.ToolCallRequest,
    tool_round_identity: ToolRoundIdentity,
    execution_profile: ExecutionProfileIdentity | None,
    model_step: int | None,
    window_id: str,
    binding_generation_id: str,
    artifact_store_id: str | None,
    observer_is_runtime_owned: bool,
    interaction_id: str | None,
    tool_outcome_event_id: str,
    tool_outcome_event_digest: str,
    before_observation_id: str,
    after_observation_id: str,
    delta: WorkspaceRevisionDelta,
    projection: _WorkspaceEvidenceProjection,
    attribution: WorkspaceMutationAttribution,
    writer_isolation_before: WorkspaceWriterIsolationEvidence,
    writer_isolation_after: WorkspaceWriterIsolationEvidence,
    direct_mutations: dict[str, Any],
    pre_window_change: WorkspaceRevisionDelta | None,
) -> Event:
    payload: dict[str, Any] = {
        "window_id": window_id,
        "before_observation_id": before_observation_id,
        "after_observation_id": after_observation_id,
        "session_run_epoch": session.run_epoch,
        "binding_generation_id": binding_generation_id,
        "artifact_store_id": artifact_store_id,
        "tool_call_id": tool_call.id,
        "tool_outcome_event_id": tool_outcome_event_id,
        "tool_outcome_event_digest": tool_outcome_event_digest,
        "workspace_id": delta.identity.workspace_id,
        "observer": delta.identity.observer,
        "status": projection.status,
        "before_revision": delta.before_revision if projection.evidence_available else None,
        "after_revision": delta.after_revision if projection.evidence_available else None,
        "paths": projection.paths,
        "total_paths": delta.total_paths if projection.evidence_available else 0,
        "head_changed": delta.head_changed if projection.evidence_available else False,
        "branch_changed": delta.branch_changed if projection.evidence_available else False,
        "detail_code": projection.detail_code,
        "attribution": attribution.model_dump(mode="json"),
        "writer_isolation": {
            "before": writer_isolation_before.model_dump(mode="json"),
            "after": writer_isolation_after.model_dump(mode="json"),
        },
        "direct_mutations": direct_mutations,
        **tool_round_identity.payload(),
    }
    if pre_window_change is not None:
        payload["pre_window_change"] = _pre_window_change_payload(
            pre_window_change,
            window_id=window_id,
        )
    if model_step is not None:
        payload["model_step"] = model_step
    _add_workspace_manifest_artifact_fields(payload, projection)
    event = event_with_runtime_generated_id(
        Event(
            id=_workspace_observation_event_id(window_id=window_id, phase="delta"),
            type=EventType.WORKSPACE_MUTATION_RECORDED,
            session_id=session.id,
            interaction_id=interaction_id,
            agent_name=registered_agent.spec.name,
            environment_name=environment_name,
            tool_name=tool_call.name,
            payload=payload,
        )
    )
    event = _event_with_workspace_observation_authority(
        event_with_execution_profile_authority(event, execution_profile),
        tool_round_identity,
        interaction_id,
        "tool_call_id",
        "window_id",
        "binding_generation_id",
        "tool_outcome_event_id",
        "tool_outcome_event_digest",
        "workspace_id",
        *(("observer",) if observer_is_runtime_owned else ()),
        *(() if artifact_store_id is None else ("artifact_store_id",)),
        "before_observation_id",
        "after_observation_id",
        "manifest_artifact_id",
        "manifest_artifact_sha256",
    )
    nested_authority_paths: list[tuple[str, ...]] = [
        ("attribution", "confidence"),
        ("attribution", "detail_code"),
        ("attribution", "direct_reconciliation"),
        ("attribution", "writer_isolation"),
        ("writer_isolation", "before", "status"),
        ("writer_isolation", "after", "status"),
        ("direct_mutations", "operations", "*", "method"),
        ("direct_mutations", "operations", "*", "path_sha256"),
        ("direct_mutations", "operations", "*", "result_operation"),
        ("direct_mutations", "operations", "*", "result_evidence_sha256"),
    ]
    if pre_window_change is not None:
        nested_authority_paths.extend(
            (
                ("pre_window_change", "attribution_confidence"),
                ("pre_window_change", "status"),
                ("pre_window_change", "paths", "*", "change"),
            )
        )
    return event_with_runtime_nested_payload_authority(
        event,
        *nested_authority_paths,
    )


def _pre_window_change_payload(
    delta: WorkspaceRevisionDelta,
    *,
    window_id: str,
) -> dict[str, Any]:
    """Project a bounded gap delta without exposing an unredacted path."""

    retained_paths = delta.paths[:_MAX_RETAINED_WORKSPACE_CAPTURE_OPERATIONS]
    return {
        "attribution_confidence": "external_or_unknown",
        "status": delta.status.value,
        "before_revision": delta.before_revision,
        "after_revision": delta.after_revision,
        "paths": [
            {
                "path_sha256": hashlib.sha256(f"{window_id}\0{path.path}".encode()).hexdigest(),
                "change": path.change,
            }
            for path in retained_paths
        ],
        "retained_paths": len(retained_paths),
        "total_paths": delta.total_paths,
        "truncated": len(retained_paths) < delta.total_paths,
        "head_changed": delta.head_changed,
        "branch_changed": delta.branch_changed,
        "detail_code": delta.detail_code,
    }


def _workspace_mutation_incomplete_event(
    *,
    lifecycle: WorkspaceObservationLifecycle,
    session: Session,
    execution_profile: ExecutionProfileIdentity | None,
    status: WorkspaceObservationTerminalStatus,
    detail_code: str | None,
) -> Event:
    normalized_detail_code = (
        None if detail_code is None else require_clean_nonblank(detail_code, "detail_code")
    )
    if (status.value, normalized_detail_code) not in WORKSPACE_OBSERVATION_TERMINAL_CONTROLS:
        raise ValueError("Workspace observation terminal classification is invalid.")
    identity = ToolRoundIdentity(
        model_step_id=lifecycle.model_step_id,
        model_attempt_id=lifecycle.model_attempt_id,
        tool_round_id=lifecycle.tool_round_id,
    )
    artifact_payload: dict[str, Any] = {}
    artifact_authority_fields: list[str] = []
    for artifact in lifecycle.artifacts:
        prefix = artifact.evidence_kind.replace("-", "_")
        id_field = f"{prefix}_artifact_id"
        artifact_payload[id_field] = artifact.artifact_id
        artifact_payload[f"{prefix}_artifact_sha256"] = artifact.sha256
        artifact_payload[f"{prefix}_artifact_size_bytes"] = artifact.size_bytes
        artifact_payload[f"{prefix}_artifact_state"] = artifact.state.value
        artifact_authority_fields.extend(
            (
                id_field,
                f"{prefix}_artifact_sha256",
            )
        )
    payload: dict[str, Any] = {
        "window_id": lifecycle.window_id,
        "session_run_epoch": lifecycle.source_run_epoch,
        "recovery_run_epoch": session.run_epoch,
        "binding_generation_id": lifecycle.binding_generation_id,
        "workspace_id": lifecycle.workspace_id,
        "observer": lifecycle.observer,
        "artifact_store_id": lifecycle.artifact_store_id,
        "tool_call_id": lifecycle.tool_call_id,
        "status": status.value,
        "paths": [],
        "total_paths": 0,
        "head_changed": False,
        "branch_changed": False,
        "detail_code": normalized_detail_code,
        "referenced_artifact_count": sum(
            artifact.state is WorkspaceObservationArtifactState.REFERENCED
            for artifact in lifecycle.artifacts
        ),
        "failed_artifact_count": sum(
            artifact.state is WorkspaceObservationArtifactState.FAILED
            for artifact in lifecycle.artifacts
        ),
        **artifact_payload,
        **identity.payload(),
    }
    if lifecycle.mutation_event_id is None:
        payload["attribution"] = {
            "confidence": (
                "concurrent_ambiguity"
                if status is WorkspaceObservationTerminalStatus.AMBIGUOUS
                else "external_or_unknown"
            ),
            "writer_isolation": "unknown",
            "overlap_detected": False,
            "direct_reconciliation": "not_observed",
            "detail_code": "workspace_attribution_recovery_incomplete",
        }
    if lifecycle.model_step is not None:
        payload["model_step"] = lifecycle.model_step
    if lifecycle.tool_outcome_event_id is not None:
        payload["tool_outcome_event_id"] = lifecycle.tool_outcome_event_id
    if lifecycle.tool_outcome_event_digest is not None:
        payload["tool_outcome_event_digest"] = lifecycle.tool_outcome_event_digest
    if lifecycle.before_observation_id is not None:
        payload["before_observation_id"] = lifecycle.before_observation_id
    if lifecycle.after_observation_id is not None:
        payload["after_observation_id"] = lifecycle.after_observation_id
    if lifecycle.mutation_event_id is not None:
        payload["mutation_event_id"] = lifecycle.mutation_event_id
    if lifecycle.mutation_event_digest is not None:
        payload["mutation_event_digest"] = lifecycle.mutation_event_digest
    event = event_with_runtime_generated_id(
        Event(
            id=_workspace_observation_event_id(
                window_id=lifecycle.window_id,
                phase="terminal",
            ),
            type=EventType.WORKSPACE_OBSERVATION_FINALIZED,
            session_id=lifecycle.session_id,
            interaction_id=lifecycle.interaction_id,
            agent_name=lifecycle.agent_name,
            environment_name=lifecycle.environment_name,
            tool_name=lifecycle.tool_name,
            payload=payload,
        )
    )
    return _event_with_workspace_observation_authority(
        event_with_execution_profile_authority(event, execution_profile),
        identity,
        lifecycle.interaction_id,
        "tool_call_id",
        "window_id",
        "binding_generation_id",
        "workspace_id",
        *(("observer",) if lifecycle.observer_authority == "runtime_builtin" else ()),
        *(() if lifecycle.artifact_store_id is None else ("artifact_store_id",)),
        "tool_outcome_event_id",
        "tool_outcome_event_digest",
        "before_observation_id",
        "after_observation_id",
        "mutation_event_id",
        "mutation_event_digest",
        *artifact_authority_fields,
    )


def _workspace_observation_terminal_view(
    lifecycle: WorkspaceObservationLifecycle,
) -> WorkspaceObservationLifecycle:
    """Classify successfully written but unreferenced terminal artifacts."""

    artifacts = tuple(
        artifact.model_copy(update={"state": WorkspaceObservationArtifactState.ORPHANED})
        if artifact.state is WorkspaceObservationArtifactState.PUBLISHED
        else artifact
        for artifact in lifecycle.artifacts
    )
    if artifacts == lifecycle.artifacts:
        return lifecycle
    return WorkspaceObservationLifecycle.model_validate(
        {
            **lifecycle.model_dump(mode="json"),
            "artifacts": [artifact.model_dump(mode="json") for artifact in artifacts],
        }
    )


def _add_workspace_manifest_artifact_fields(
    payload: dict[str, Any],
    projection: _WorkspaceEvidenceProjection,
) -> None:
    if projection.manifest_artifact_id is None:
        return
    payload.update(
        {
            "manifest_artifact_id": projection.manifest_artifact_id,
            "manifest_artifact_sha256": projection.manifest_artifact_sha256,
            "manifest_artifact_size_bytes": projection.manifest_artifact_size_bytes,
        }
    )


async def _workspace_evidence_projection(
    *,
    paths: list[dict[str, Any]],
    status: str,
    detail_code: str | None,
    artifact_store: Any,
    artifact_unavailable_detail_code: str = "manifest_artifact_store_unavailable",
    session: Session,
    registered_agent: runtime_records.RegisteredAgentState,
    environment_name: str | None,
    window_id: str,
    evidence_kind: Literal["revision-before", "revision-after", "revision-delta"],
    redactor: SecretRedactor,
    evidence_available: bool = True,
    operation_registry: BoundedInvocationOperationRegistry,
    registered_environment: runtime_records.RegisteredEnvironment,
    artifact_state_recorder: (
        Callable[[WorkspaceObservationArtifact], Awaitable[None]] | None
    ) = None,
) -> _WorkspaceEvidenceProjection:
    if not evidence_available:
        return _WorkspaceEvidenceProjection(
            paths=[],
            status="truncated",
            detail_code="workspace_evidence_quarantined",
            evidence_available=False,
        )
    redacted_paths = redactor.redact_json_values(paths)
    if type(redacted_paths) is not list or any(type(path) is not dict for path in redacted_paths):
        return _WorkspaceEvidenceProjection(
            paths=[],
            status="failed",
            detail_code="manifest_redaction_failed",
        )
    content = json.dumps(
        {
            "schema_version": 1,
            "kind": evidence_kind,
            "paths": redacted_paths,
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    if (
        len(redacted_paths) <= _WORKSPACE_RECEIPT_INLINE_PATH_LIMIT
        and len(content) <= _WORKSPACE_RECEIPT_INLINE_BYTES
    ):
        return _WorkspaceEvidenceProjection(
            paths=redacted_paths,
            status=status,
            detail_code=detail_code,
        )
    if artifact_store is None:
        return _WorkspaceEvidenceProjection(
            paths=[],
            status="truncated",
            detail_code=artifact_unavailable_detail_code,
        )

    digest = hashlib.sha256(content).hexdigest()
    artifact_identity = json.dumps(
        [session.id, window_id, evidence_kind, digest],
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    artifact_id = f"art_{hashlib.sha256(artifact_identity).hexdigest()[:32]}"
    filename = f"workspace-{evidence_kind}.json"
    artifact_record = WorkspaceObservationArtifact(
        evidence_kind=evidence_kind,
        artifact_id=artifact_id,
        sha256=digest,
        size_bytes=len(content),
        state=WorkspaceObservationArtifactState.INTENT,
    )
    if artifact_state_recorder is not None:
        await artifact_state_recorder(artifact_record)
    try:
        async with asyncio.timeout(_WORKSPACE_ARTIFACT_WRITE_TIMEOUT_SECONDS):
            artifact_outcome = await await_invocation_operation(
                lambda: artifact_store.put_bytes(
                    content,
                    artifact_id=artifact_id,
                    filename=filename,
                    content_type="application/json",
                    scope=ArtifactScope.SESSION,
                    session_id=session.id,
                    agent_name=registered_agent.spec.name,
                    environment_name=environment_name,
                    metadata={
                        "schema_version": 1,
                        "kind": evidence_kind,
                        "sha256": digest,
                        "window_id": window_id,
                    },
                ),
                request_child_cancellation=False,
                abandon_on_caller_cancellation=True,
                operation_registry=operation_registry,
                on_abandoned_caller_cancellation=(
                    registered_environment.workspace_mutation_fence.fail_closed
                ),
            )
            raise_workspace_observation_concurrent_control(
                cancellation=artifact_outcome.cancellation,
                error=artifact_outcome.error,
                operation="Workspace observation artifact publication",
            )
    except TimeoutError:
        # The store call was dispatched with cancellation-opaque semantics and
        # remains retained by ``operation_registry``.  Timeout proves only
        # that Cayu stopped waiting; it does not prove that publication failed.
        # Keep the durable intent so a late successful write remains an
        # explicitly reconstructible possible orphan rather than being
        # misreported as a settled failure.
        return _WorkspaceEvidenceProjection(
            paths=[],
            status="truncated",
            detail_code="manifest_artifact_write_unsettled",
        )
    if _contains_process_signal(artifact_outcome.error):
        if artifact_outcome.error is None:  # pragma: no cover - narrowed by the helper
            raise AssertionError("Process-signal classification lost its error.")
        raise artifact_outcome.error
    if artifact_outcome.error is not None:
        # Once the extension call started, an exception is not authoritative
        # evidence that the content-bound write did not commit.  Preserve the
        # durable intent so restart reconciliation can identify a successful
        # late or acknowledgement-lost write as an orphan.  Only positive
        # no-dispatch evidence may close the intent as failed.
        if not artifact_outcome.operation_started and artifact_state_recorder is not None:
            await artifact_state_recorder(
                artifact_record.model_copy(
                    update={"state": WorkspaceObservationArtifactState.FAILED}
                )
            )
        return _WorkspaceEvidenceProjection(
            paths=[],
            status="truncated",
            detail_code=(
                "manifest_artifact_write_failed"
                if not artifact_outcome.operation_started
                else "manifest_artifact_write_unsettled"
            ),
        )
    metadata = artifact_outcome.result
    if not workspace_observation_artifact_metadata_matches(
        metadata,
        artifact=artifact_record,
        session_id=session.id,
        agent_name=registered_agent.spec.name,
        environment_name=environment_name,
        window_id=window_id,
    ):
        return _WorkspaceEvidenceProjection(
            paths=[],
            status="failed",
            detail_code="manifest_artifact_reference_invalid",
        )
    if artifact_state_recorder is not None:
        await artifact_state_recorder(
            artifact_record.model_copy(
                update={"state": WorkspaceObservationArtifactState.PUBLISHED}
            )
        )
    return _WorkspaceEvidenceProjection(
        paths=[],
        status=status,
        detail_code=detail_code,
        manifest_artifact_id=artifact_id,
        manifest_artifact_sha256=digest,
        manifest_artifact_size_bytes=len(content),
    )


def _workspace(registered_environment: runtime_records.RegisteredEnvironment | None) -> Any:
    if registered_environment is None:
        return None
    return registered_environment.environment.workspace


def _artifact_store_id(
    registered_environment: runtime_records.RegisteredEnvironment | None,
) -> str | None:
    if registered_environment is None or registered_environment.environment.artifact_store is None:
        return None
    artifact_store_id = getattr(registered_environment.environment.artifact_store, "id", None)
    if artifact_store_id is None:
        return None
    artifact_store_id = require_clean_nonblank(artifact_store_id, "artifact_store.id")
    return require_unicode_scalar_text(artifact_store_id, "artifact_store.id")


def _artifact_store(registered_environment: runtime_records.RegisteredEnvironment | None) -> Any:
    if registered_environment is None:
        return None
    return registered_environment.environment.artifact_store


def _workspace_receipt_artifact_store(
    registered_environment: runtime_records.RegisteredEnvironment | None,
) -> tuple[Any, str]:
    """Exclude local stores whose receipt writes would mutate the observed workspace."""

    artifact_store = _artifact_store(registered_environment)
    workspace = _workspace(registered_environment)
    if registered_environment is not None and registered_environment.bound_workspace is not None:
        workspace = registered_environment.bound_workspace.workspace
    if isinstance(artifact_store, LocalArtifactStore) and isinstance(workspace, LocalWorkspace):
        try:
            artifact_store.root.relative_to(workspace.root)
        except ValueError:
            pass
        else:
            return None, "manifest_artifact_store_inside_workspace"
    return artifact_store, "manifest_artifact_store_unavailable"


def _runner(registered_environment: runtime_records.RegisteredEnvironment | None) -> Any:
    if registered_environment is None:
        return None
    return registered_environment.environment.runner


def _knowledge_store(
    registered_environment: runtime_records.RegisteredEnvironment | None,
) -> Any:
    if registered_environment is None:
        return None
    return registered_environment.environment.knowledge_store


def _knowledge_access_scope(
    registered_environment: runtime_records.RegisteredEnvironment | None,
) -> Any:
    if registered_environment is None:
        return None
    return registered_environment.environment.knowledge_access_scope


def _mcp_servers(
    registered_environment: runtime_records.RegisteredEnvironment | None,
) -> tuple[Any, ...]:
    if registered_environment is None:
        return ()
    return registered_environment.environment.mcp_servers


def _registered_mcp_tool_authority_is_unavailable(
    registered_tool: runtime_records.RegisteredTool | None,
) -> bool:
    if registered_tool is None or not isinstance(registered_tool.tool, McpToolAdapter):
        return False
    return not registered_tool.tool._dispatch_authority_is_current()


def _tool_dispatch_authority_is_current(
    registered_agent: runtime_records.RegisteredAgentState,
    tool_name: str,
) -> bool:
    return not _registered_mcp_tool_authority_is_unavailable(
        registered_agent.executable_tool(tool_name)
    )


def _mcp_manifest_candidates_for_agent(
    registered_agent: runtime_records.RegisteredAgentState,
    *,
    environment_name: str | None,
) -> tuple[_McpManifestCandidate, ...]:
    grouped: dict[int, tuple[McpToolset, list[runtime_records.RegisteredTool]]] = {}
    for registered_tool in registered_agent.tools.values():
        tool = registered_tool.tool
        if isinstance(tool, McpToolAdapter):
            binding = tool._manifest_binding
            toolset_key = id(binding.toolset)
            existing = grouped.get(toolset_key)
            if existing is None:
                grouped[toolset_key] = (binding.toolset, [registered_tool])
            else:
                if existing[0] is not binding.toolset:
                    raise McpManifestHistoryConflict(
                        "MCP adapter ownership changed during manifest admission."
                    )
                existing[1].append(registered_tool)

    candidates: list[_McpManifestCandidate] = []
    for toolset, registered_tools in grouped.values():
        source = toolset._manifest_snapshot
        exposed_tools = tuple(
            sorted(
                (
                    _mcp_manifest_exposed_tool_evidence(registered_tool)
                    for registered_tool in registered_tools
                ),
                key=lambda entry: entry.tool_id,
            )
        )
        if len(exposed_tools) != len({entry.tool_id for entry in exposed_tools}):
            raise McpManifestHistoryConflict(
                "MCP provider exposure contains ambiguous duplicate tool identities."
            )
        durable_tools = _durable_mcp_manifest_source_tools(source.tools)
        durable_exposed_tools = _durable_mcp_manifest_exposed_tool_evidence(exposed_tools)
        snapshot = _McpManifestAuditSnapshot(
            identity_is_explicit=source.identity_is_explicit,
            identity=source.identity,
            manifest_hash=_mcp_authoritative_manifest_hash(
                source_manifest_hash=source.manifest_hash,
                server_hash=source.server_hash,
                tools=durable_tools,
                exposed_tools=durable_exposed_tools,
            ),
            source_manifest_hash=source.manifest_hash,
            server_hash=source.server_hash,
            tools=source.tools,
            exposed_tools=exposed_tools,
            advertised_tool_count=source.tool_count,
            tool_count=len(exposed_tools),
        )
        candidates.append(
            _McpManifestCandidate(
                history_key=_mcp_manifest_history_key(
                    environment_name=environment_name,
                    manifest_identity=snapshot.identity,
                ),
                toolset=toolset,
                snapshot=snapshot,
            )
        )
    return tuple(candidates)


_MCP_MANIFEST_PUBLICATION_MAX_ATTEMPTS = 8
_MCP_MANIFEST_EVENT_TOOL_SAMPLE_LIMIT = 100
_McpManifestResultT = TypeVar("_McpManifestResultT")


@dataclass(frozen=True, slots=True)
class _McpManifestExposedToolEvidence:
    tool_id: str
    contract_hash: str


@dataclass(frozen=True, slots=True)
class _McpManifestAuditSnapshot:
    identity_is_explicit: bool
    identity: str
    manifest_hash: str
    source_manifest_hash: str
    server_hash: str
    tools: tuple[Any, ...]
    exposed_tools: tuple[_McpManifestExposedToolEvidence, ...]
    advertised_tool_count: int
    tool_count: int


@dataclass(frozen=True, slots=True)
class _McpManifestCandidate:
    history_key: str
    toolset: McpToolset
    snapshot: _McpManifestAuditSnapshot


@dataclass(frozen=True)
class _McpManifestEvaluation:
    candidate: _McpManifestCandidate
    status: str
    decision: McpManifestPolicyDecision | None
    event: Event


def _mcp_manifest_exposed_tool_evidence(
    registered_tool: runtime_records.RegisteredTool,
) -> _McpManifestExposedToolEvidence:
    tool = registered_tool.tool
    if not isinstance(tool, McpToolAdapter):
        raise TypeError("MCP exposure evidence requires an McpToolAdapter.")
    binding = tool._manifest_binding
    source = binding.toolset._manifest_snapshot
    matching_source_entries = [
        entry
        for entry in source.tools
        if entry.mcp_name == binding.manifest_mcp_name
        and entry.contract_hash == binding.manifest_contract_hash
    ]
    if len(matching_source_entries) != 1:
        raise McpManifestHistoryConflict(
            "An MCP adapter is not backed by exactly one advertised tool definition."
        )
    tool_id = _mcp_manifest_tool_identity(
        cayu_name=registered_tool.name,
        mcp_name=binding.manifest_mcp_name,
    )
    contract = json.dumps(
        {
            "schema": "cayu.mcp.exposed_tool_contract.v1",
            "tool_id": tool_id,
            "source_contract_hash": binding.manifest_contract_hash,
            "name": registered_tool.name,
            "description": registered_tool.description,
            "input_schema": registered_tool.schema,
            "parallel_safe": registered_tool.parallel_safe,
            "effect": registered_tool.effect.value,
            "mcp_name": binding.manifest_mcp_name,
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return _McpManifestExposedToolEvidence(
        tool_id=tool_id,
        contract_hash=f"sha256:{hashlib.sha256(contract).hexdigest()}",
    )


async def _await_mcp_manifest_operation(
    operation_factory: Callable[[], Awaitable[_McpManifestResultT]],
    *,
    operation: str,
) -> _McpManifestResultT:
    """Preserve new caller cancellation while rejecting child-only cancellation."""

    current_task = asyncio.current_task()
    # Deliver a request already pending at this boundary before creating the
    # operation. If no cancellation is delivered, the resulting count contains
    # only historical requests and is the provenance baseline for the await.
    await asyncio.sleep(0)
    cancellation_requests = 0 if current_task is None else current_task.cancelling()
    try:
        result = await operation_factory()
    except asyncio.CancelledError as cancellation:
        if current_task is not None and current_task.cancelling() > cancellation_requests:
            raise
        raise unexpected_child_cancellation_error(
            cancellation,
            operation=operation,
        ) from cancellation
    if current_task is not None and current_task.cancelling() > cancellation_requests:
        cancellation = consume_pending_task_cancellation(
            preserve_requests=cancellation_requests,
        )
        if cancellation is None:
            raise RuntimeError("MCP manifest caller cancellation could not be recovered.")
        raise cancellation
    return result


def _mcp_manifest_history_key(
    *,
    manifest_identity: str,
    environment_name: str | None,
) -> str:
    encoded = json.dumps(
        {
            "schema": "cayu.mcp.manifest_history.v1",
            "environment_name": environment_name,
            "manifest_identity": manifest_identity,
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return f"sha256:{hashlib.sha256(encoded).hexdigest()}"


def _validated_mcp_manifest_baseline_load(
    value: object,
    *,
    candidates: tuple[_McpManifestCandidate, ...],
    environment_name: str | None,
) -> McpManifestBaselineLoadResult:
    if not isinstance(value, McpManifestBaselineLoadResult):
        raise TypeError("load_mcp_manifest_baselines must return McpManifestBaselineLoadResult.")
    validated = McpManifestBaselineLoadResult.model_validate(value.model_dump(mode="python"))
    expected_keys = {candidate.history_key for candidate in candidates}
    unexpected_keys = set(validated.baselines) - expected_keys
    if unexpected_keys:
        raise ValueError("MCP baseline load returned unrequested history identities.")
    for candidate in candidates:
        baseline = validated.baselines.get(candidate.history_key)
        if baseline is None:
            continue
        expected_history_key = _mcp_manifest_history_key(
            environment_name=environment_name,
            manifest_identity=candidate.snapshot.identity,
        )
        if (
            baseline.manifest_identity != candidate.snapshot.identity
            or baseline.history_key != expected_history_key
            or candidate.history_key != expected_history_key
        ):
            raise ValueError("MCP baseline identity does not match the exposed toolset.")
    return validated


def _validated_mcp_manifest_publication_result(
    value: object,
    *,
    candidates: tuple[_McpManifestCandidate, ...],
    environment_name: str | None,
    expected_published_baselines: dict[str, McpManifestBaseline],
) -> McpManifestPublicationResult:
    if not isinstance(value, McpManifestPublicationResult):
        raise TypeError(
            "compare_and_publish_mcp_manifest_checks must return McpManifestPublicationResult."
        )
    validated = McpManifestPublicationResult.model_validate(value.model_dump(mode="python"))
    _validated_mcp_manifest_baseline_load(
        McpManifestBaselineLoadResult(baselines=validated.baselines),
        candidates=candidates,
        environment_name=environment_name,
    )
    expected = McpManifestBaselineLoadResult(
        baselines=expected_published_baselines,
    ).baselines
    if validated.published and validated.baselines != expected:
        raise ValueError("Published MCP manifest baselines do not match the accepted publication.")
    return validated


def _mcp_manifest_status(
    *,
    snapshot: _McpManifestAuditSnapshot,
    previous: McpManifestBaseline | None,
) -> tuple[str, dict[str, Any] | None, dict[str, Any]]:
    current_source_tools = _mcp_manifest_tool_hashes(_durable_mcp_manifest_tools(snapshot))
    current_exposed_tools = _mcp_manifest_tool_hashes(_durable_mcp_manifest_exposed_tools(snapshot))
    empty_diff = {
        "server_changed": False,
        "added_tools": [],
        "removed_tools": [],
        "changed_tools": [],
    }
    if previous is None:
        return "first_seen", None, empty_diff

    previous_summary = {
        "event_id": previous.accepted_event_id,
        "session_ref": previous.accepted_session_ref,
        "generation": previous.generation,
        "manifest_identity": previous.manifest_identity,
        "manifest_hash": previous.manifest_hash,
        "source_manifest_hash": previous.source_manifest_hash,
        "server_hash": previous.server_hash,
    }
    if previous.manifest_hash == snapshot.manifest_hash:
        return "unchanged", previous_summary, empty_diff

    added: set[str] = set()
    removed: set[str] = set()
    changed: set[str] = set()
    for current_tools, previous_tools in (
        (current_source_tools, _mcp_manifest_tool_hashes(previous.tools)),
        (
            current_exposed_tools,
            _mcp_manifest_tool_hashes(previous.exposed_tools),
        ),
    ):
        added.update(name for name in current_tools if name not in previous_tools)
        removed.update(name for name in previous_tools if name not in current_tools)
        changed.update(
            name
            for name, tool_hash in current_tools.items()
            if name in previous_tools and previous_tools[name] != tool_hash
        )
    return (
        "changed",
        previous_summary,
        {
            "server_changed": previous.server_hash != snapshot.server_hash,
            "added_tools": sorted(added),
            "removed_tools": sorted(removed),
            "changed_tools": sorted(changed),
        },
    )


def _mcp_manifest_history_blocked_event(
    *,
    session: Session,
    registered_agent: runtime_records.RegisteredAgentState,
    environment_name: str | None,
    history_key: str,
    snapshot: _McpManifestAuditSnapshot,
    status: str,
    reason: str,
) -> Event:
    return Event(
        type=EventType.MCP_MANIFEST_BLOCKED,
        session_id=session.id,
        agent_name=registered_agent.spec.name,
        environment_name=environment_name,
        payload={
            "history_key": history_key,
            "manifest_identity": snapshot.identity,
            "manifest_hash": snapshot.manifest_hash,
            "source_manifest_hash": snapshot.source_manifest_hash,
            "status": status,
            "change_classes": [],
            "outcome": "blocked",
            "reason": reason,
        },
    )


def _mcp_manifest_batch_blocked_event(
    *,
    session: Session,
    registered_agent: runtime_records.RegisteredAgentState,
    environment_name: str | None,
    history_key: str,
    snapshot: _McpManifestAuditSnapshot,
    reason: str,
) -> Event:
    return Event(
        type=EventType.MCP_MANIFEST_CHECKED,
        session_id=session.id,
        agent_name=registered_agent.spec.name,
        environment_name=environment_name,
        payload={
            "history_key": history_key,
            "manifest_identity": snapshot.identity,
            "manifest_hash": snapshot.manifest_hash,
            "source_manifest_hash": snapshot.source_manifest_hash,
            "server_hash": snapshot.server_hash,
            "status": "not_evaluated",
            "change_classes": [],
            "outcome": "batch_blocked",
            "reason": reason,
        },
    )


def _mcp_manifest_event_payload(
    *,
    history_key: str,
    snapshot: _McpManifestAuditSnapshot,
    status: str,
    previous: dict[str, Any] | None,
    diff: dict[str, Any],
    decision: McpManifestPolicyDecision | None,
    outcome: str,
) -> dict[str, Any]:
    projected_diff = _bounded_mcp_manifest_diff(diff)
    change_classes = []
    if diff.get("server_changed") is True:
        change_classes.append("server_changed")
    for field, change_class in (
        ("added_tools", "tools_added"),
        ("removed_tools", "tools_removed"),
        ("changed_tools", "tools_changed"),
    ):
        if isinstance(diff.get(field), list) and diff[field]:
            change_classes.append(change_class)
    payload: dict[str, Any] = {
        "history_key": history_key,
        "manifest_identity": snapshot.identity,
        "manifest_hash": snapshot.manifest_hash,
        "source_manifest_hash": snapshot.source_manifest_hash,
        "server_hash": snapshot.server_hash,
        "status": status,
        "tool_count": snapshot.tool_count,
        "advertised_tool_count": snapshot.advertised_tool_count,
        "previous": previous,
        "diff": projected_diff,
        "change_classes": change_classes,
        "outcome": outcome,
    }
    if decision is not None:
        payload["policy"] = mcp_manifest_policy_payload(decision)
    return payload


def _bounded_mcp_manifest_diff(diff: Mapping[str, Any]) -> dict[str, Any]:
    projected: dict[str, Any] = {
        "server_changed": diff.get("server_changed") is True,
    }
    truncated = False
    for field in ("added_tools", "removed_tools", "changed_tools"):
        value = diff.get(field)
        items = [item for item in value if isinstance(item, str)] if isinstance(value, list) else []
        projected[field] = items[:_MCP_MANIFEST_EVENT_TOOL_SAMPLE_LIMIT]
        projected[f"{field}_count"] = len(items)
        truncated = truncated or len(items) > _MCP_MANIFEST_EVENT_TOOL_SAMPLE_LIMIT
    projected["truncated"] = truncated
    return projected


def _mcp_manifest_baseline(
    *,
    history_key: str,
    snapshot: _McpManifestAuditSnapshot,
    generation: int,
    event: Event,
) -> McpManifestBaseline:
    return McpManifestBaseline(
        history_key=history_key,
        generation=generation,
        manifest_identity=snapshot.identity,
        manifest_hash=snapshot.manifest_hash,
        source_manifest_hash=snapshot.source_manifest_hash,
        server_hash=snapshot.server_hash,
        tools=_durable_mcp_manifest_tools(snapshot),
        exposed_tools=_durable_mcp_manifest_exposed_tools(snapshot),
        accepted_session_ref=_mcp_manifest_session_ref(event.session_id),
        accepted_event_id=event.id,
        accepted_at=event.timestamp,
    )


def _mcp_manifest_tool_hashes(value: object) -> dict[str, str]:
    if not isinstance(value, list | tuple):
        return {}
    result: dict[str, str] = {}
    for item in value:
        if not isinstance(item, Mapping):
            continue
        entry = cast("Mapping[str, object]", item)
        tool_id = entry.get("tool_id")
        contract_hash = entry.get("contract_hash")
        if isinstance(tool_id, str) and isinstance(contract_hash, str):
            result[tool_id] = contract_hash
    return result


def _durable_mcp_manifest_tools(
    snapshot: _McpManifestAuditSnapshot,
) -> tuple[dict[str, str], ...]:
    """Project immutable manifest entries into bounded, non-identifying evidence."""

    return _durable_mcp_manifest_source_tools(snapshot.tools)


def _durable_mcp_manifest_source_tools(
    tools: Iterable[Any],
) -> tuple[dict[str, str], ...]:
    projected: list[dict[str, str]] = []
    for entry in tools:
        projected.append(
            {
                "tool_id": _mcp_manifest_tool_identity(
                    cayu_name=entry.cayu_name,
                    mcp_name=entry.mcp_name,
                ),
                "contract_hash": entry.contract_hash,
            }
        )
    projected.sort(key=lambda item: item["tool_id"])
    return tuple(projected)


def _durable_mcp_manifest_exposed_tools(
    snapshot: _McpManifestAuditSnapshot,
) -> tuple[dict[str, str], ...]:
    return _durable_mcp_manifest_exposed_tool_evidence(snapshot.exposed_tools)


def _durable_mcp_manifest_exposed_tool_evidence(
    exposed_tools: Iterable[_McpManifestExposedToolEvidence],
) -> tuple[dict[str, str], ...]:
    return tuple(
        {
            "tool_id": entry.tool_id,
            "contract_hash": entry.contract_hash,
        }
        for entry in exposed_tools
    )


def _mcp_manifest_tool_identity(*, cayu_name: str, mcp_name: str) -> str:
    identity = json.dumps(
        {
            "schema": "cayu.mcp.audit_tool_identity.v1",
            "cayu_name": cayu_name,
            "mcp_name": mcp_name,
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return f"sha256:{hashlib.sha256(identity).hexdigest()}"


def _tool_effect(
    registered_agent: runtime_records.RegisteredAgentState,
    tool_call: runtime_records.ToolCallRequest,
) -> ToolEffect:
    registered_tool = registered_agent.executable_tool(tool_call.name)
    if registered_tool is None:
        return ToolEffect.EXTERNAL
    return registered_tool.effect


def _tool_round_publishes_arguments(
    registered_agent: runtime_records.RegisteredAgentState,
    tool_calls: list[runtime_records.ToolCallRequest],
) -> bool:
    """Require every paused call to permit one shared argument projection."""

    for tool_call in tool_calls:
        registered_tool = registered_agent.executable_tool(tool_call.name)
        if registered_tool is None or not registered_tool.publish_arguments:
            return False
    return True


def _taint_labels_for_source_tool(
    policy: ToolPolicy,
    tool_name: str,
    *,
    policy_result: ToolPolicyResult | None,
) -> set[str]:
    if not isinstance(policy, TaintAwareToolPolicy):
        return set()
    if policy_result is not None and policy_result.decision != ToolPolicyDecision.ALLOW:
        return set()
    return set(policy.labels_for_source_tool(tool_name))


@dataclass
class _BeforeToolCallResolution:
    arguments: dict[str, Any]
    short_circuit_result: ToolResult | None = None
    block_reason: str | None = None


def _private_argument_short_circuit_result(result: ToolResult) -> ToolResult:
    """Project one hook-controlled result without argument-derived content."""

    return ToolResult(
        content="Tool execution was short-circuited by a before_tool_call hook.",
        structured={"outcome": "short_circuited"},
        is_error=result.is_error,
    )


def _resolve_before_tool_call_decision(
    decision: BeforeToolCallDecision | None,
    resolution: _BeforeToolCallResolution,
    *,
    redactor: SecretRedactor,
) -> bool:
    if decision is None:
        return False
    if type(decision) is not BeforeToolCallDecision:
        raise TypeError("before_tool_call must return a BeforeToolCallDecision or None.")
    if decision.action == "proceed":
        return False
    if decision.action == "proceed_modified":
        modified_arguments = decision.modified_arguments
        if modified_arguments is None:
            raise TypeError("A proceed_modified decision must carry modified_arguments.")
        copied_arguments = copy_durable_json_value(
            modified_arguments,
            "modified_arguments",
        )
        if type(copied_arguments) is not dict:
            raise TypeError("A proceed_modified decision must carry object arguments.")
        resolution.arguments = copied_arguments
        return False
    if decision.action == "short_circuit":
        synthetic = decision.synthetic_result
        if synthetic is None:
            raise TypeError("A short_circuit decision must carry a synthetic_result.")
        resolution.short_circuit_result = _prepare_hook_authored_tool_result(
            synthetic,
            redactor=redactor,
        )
        return True
    if decision.action != "block":
        raise ValueError("Unsupported before_tool_call decision action.")
    reason = decision.block_reason
    if reason is None:
        raise TypeError("A block decision must carry a block_reason.")
    resolution.block_reason = require_durable_text(reason, "block_reason")
    return True


def _resolve_after_tool_call_decision(
    decision: AfterToolCallDecision | None,
    *,
    redactor: SecretRedactor,
) -> ToolResult | None:
    if decision is None:
        return None
    if type(decision) is not AfterToolCallDecision:
        raise TypeError("after_tool_call must return an AfterToolCallDecision or None.")
    if decision.action == "modify":
        modified = decision.modified_result
        if modified is None:
            raise TypeError("An after_tool_call modify decision must carry a modified_result.")
        return _prepare_hook_authored_tool_result(modified, redactor=redactor)
    if decision.action == "pass_through":
        if decision.modified_result is not None:
            raise TypeError("A pass_through decision must not carry a modified_result.")
        return None
    raise ValueError("Unsupported after_tool_call decision action.")


def _prepare_hook_authored_tool_result(
    result: ToolResult,
    *,
    redactor: SecretRedactor,
) -> ToolResult:
    """Detach untrusted hook output before the next runtime-owned await."""

    validated = tool_results.normalize_tool_result(tool_results.validate_tool_result(result))
    sanitized = ToolResult(
        content=validated.content,
        structured=tool_results.strip_untrusted_runtime_tool_result_control_authority(
            validated.structured
        ),
        artifacts=tool_results.strip_runtime_tool_result_projection_authority(validated.artifacts),
        is_error=validated.is_error,
    )
    return tool_results.redact_tool_result(sanitized, redactor)


def _runtime_hook_event(
    *,
    event_type: EventType,
    hook_name: str,
    scope: str,
    phase: RuntimeHookPhase,
    session: Session,
    registered_agent: runtime_records.RegisteredAgentState,
    registered_environment: runtime_records.RegisteredEnvironment | None,
    terminal_event: Event,
    payload: dict[str, Any],
    execution_profile: ExecutionProfileIdentity | None = None,
) -> Event:
    event = _build_runtime_hook_event(
        event_type=event_type,
        hook_name=hook_name,
        scope=scope,
        phase=phase,
        session=session,
        terminal_event=terminal_event,
        agent_name=registered_agent.spec.name,
        environment_name=_environment_name(registered_environment),
        payload=payload,
    )
    # Tool hooks belong to their tool event's interaction. A result event can
    # still be unpublished here, so fall back to the active tool-round owner.
    # Keep this scoped: terminal hooks use deterministic replay identities and
    # must preserve the terminal event's explicit attribution instead.
    interaction_id = terminal_event.interaction_id
    if interaction_id is None:
        interaction_id = resolve_interaction_attribution(session.id, INHERIT_INTERACTION)
    if interaction_id is not None:
        event = event_with_runtime_envelope_authority(
            event.model_copy(update={"interaction_id": interaction_id}),
            "interaction_id",
        )
    return event_with_execution_profile_authority(
        event,
        execution_profile,
    )


def _redact_event_for_invocation(
    event: Event,
    *,
    redactor: SecretRedactor,
) -> Event:
    """Scrub adapter/invocation secrets before the app-level writer sees them."""

    return prepare_runtime_event(event, redactor=redactor)


_POLICY_DENIAL_CONTROL_PAYLOAD_FIELDS = frozenset(
    {
        "approval_id",
        "blocked_by",
        "decision",
        "denied_by",
        "idempotency_key",
        "input_id",
        "model_attempt_id",
        "model_step_id",
        "tool_call_id",
        "tool_name",
        "tool_round_id",
    }
)
_POLICY_DENIAL_CONTROL_RESULT_FIELDS = frozenset({"decision", "error"})


def _prepare_tool_result_event(
    *,
    event: Event,
    result: ToolResult,
    redactor: SecretRedactor,
    runtime_tool: object | None = None,
    restore_terminal_result_controls: bool = True,
) -> tuple[Event, ToolResult]:
    argument_state = event.payload.get(tool_argument_publication.ARGUMENTS_STATE_FIELD)
    if argument_state is not None and (
        type(argument_state) is not str
        or argument_state not in tool_argument_publication.TERMINAL_ARGUMENT_STATES
    ):
        raise ValueError("Terminal tool event has an invalid argument publication state.")
    argument_projection: tool_argument_publication.ToolArgumentProjection | None = None
    effective_arguments: dict[str, Any] | None = None
    if argument_state is not None:
        argument_projection = tool_argument_publication.terminal_argument_projection(
            event.payload,
            legacy_arguments={},
        )
        raw_effective_arguments = event.payload.get("effective_arguments")
        if raw_effective_arguments is not None:
            if argument_projection.state == "unavailable":
                raw_effective_arguments = None
            elif type(raw_effective_arguments) is not dict:
                raise TypeError("Terminal effective_arguments must be an object.")
            else:
                projected_effective_arguments = redactor.redact_json(raw_effective_arguments)
                if type(projected_effective_arguments) is not dict:
                    raise AssertionError("Effective argument projection returned a non-object.")
                effective_arguments = projected_effective_arguments
        payload_without_argument_projection = dict(event.payload)
        payload_without_argument_projection.pop(tool_argument_publication.ARGUMENTS_FIELD, None)
        payload_without_argument_projection.pop(
            tool_argument_publication.ARGUMENTS_STATE_FIELD,
            None,
        )
        payload_without_argument_projection.pop("effective_arguments", None)
        event = event.model_copy(update={"payload": payload_without_argument_projection})
    result = ToolResult(
        content=result.content,
        structured=tool_results.restore_runtime_tool_result_control_authority(
            result.structured,
            event.payload,
            include_terminal_controls=restore_terminal_result_controls,
        ),
        artifacts=tool_results.strip_runtime_tool_result_projection_authority(result.artifacts),
        is_error=result.is_error,
    )
    event = web_access_results.attest_runtime_web_access_result(
        event,
        result,
        tool=runtime_tool,
    )
    event = shared_artifact_results.attest_runtime_shared_artifact_result(
        event,
        result,
        tool=runtime_tool,
    )
    event, result = _validate_and_synchronize_tool_result_event(
        event=event,
        result=result,
    )
    if _is_policy_denial_event(event):
        event, result = _redact_policy_denial_event(
            event=event,
            result=result,
            redactor=redactor,
        )
    else:
        event, result = tool_results.redact_tool_result_event(
            event=event,
            result=result,
            redactor=redactor,
            include_terminal_controls=restore_terminal_result_controls,
        )
    if argument_projection is not None:
        payload = dict(event.payload)
        payload.update(argument_projection.payload_fields())
        if effective_arguments is not None:
            payload["effective_arguments"] = effective_arguments
        event = event.model_copy(update={"payload": payload})
    event, result = _bound_policy_denial_event(event=event, result=result)
    return _validate_and_synchronize_tool_result_event(event=event, result=result)


def _project_tool_call_for_hook(
    tool_call: runtime_records.ToolCallRequest,
    *,
    argument_projection: tool_argument_publication.ToolArgumentProjection,
    redactor: SecretRedactor,
) -> runtime_records.ToolCallRequest:
    """Expose effective arguments only after the invocation scope is complete."""

    if argument_projection.state == "unavailable":
        arguments: dict[str, Any] = {}
    else:
        projected = redactor.redact_json(tool_call.arguments)
        if type(projected) is not dict:
            raise AssertionError("Hook argument projection returned a non-object.")
        arguments = projected
    return replace(tool_call, arguments=arguments)


def _validate_and_synchronize_tool_result_event(
    *,
    event: Event,
    result: ToolResult,
) -> tuple[Event, ToolResult]:
    validated_result = tool_results.normalize_tool_result(tool_results.validate_tool_result(result))
    payload = copy_durable_json_object(event.payload, "tool_result_event.payload")
    payload["result"] = copy_durable_json_value(
        validated_result.model_dump(mode="python"),
        "tool_result_event.result",
    )
    synchronized = copy_event(event.model_copy(update={"payload": payload}))
    return synchronized, validated_result


def _hook_failure_payload(
    exc: Exception,
    *,
    redactor: SecretRedactor,
) -> dict[str, Any]:
    diagnostic = tool_results.exception_diagnostic(
        exc,
        empty_message="runtime hook failed",
        nonportable_message="Runtime hook failed with a non-portable diagnostic.",
        redactor=redactor,
    )
    return copy_durable_json_object(diagnostic.payload_fields(), "hook_failure")


def _hook_actions_payload(
    context: BeforeToolCallHookContext | ToolCallHookContext,
    *,
    redactor: SecretRedactor,
) -> dict[str, Any]:
    try:
        actions = copy_durable_json_value(context.actions, "hook_actions")
        actions = redactor.redact_json(actions)
        actions = copy_durable_json_value(actions, "hook_actions")
    except Exception:
        return {"actions": [], "actions_omitted": True}
    if type(actions) is not list:
        return {"actions": [], "actions_omitted": True}
    return {"actions": actions}


def _redact_tool_result_for_event(
    *,
    event: Event,
    result: ToolResult,
    redactor: SecretRedactor,
) -> ToolResult:
    if _is_policy_denial_event(event):
        return _redact_policy_denial_result(result, redactor)
    _, redacted_result = tool_results.redact_tool_result_event(
        event=event,
        result=result,
        redactor=redactor,
    )
    return redacted_result


def _is_policy_denial_event(event: Event) -> bool:
    return event.type == EventType.TOOL_CALL_BLOCKED and "denied_by" in event.payload


def _redact_policy_denial_event(
    *,
    event: Event,
    result: ToolResult,
    redactor: SecretRedactor,
) -> tuple[Event, ToolResult]:
    redacted_result = _redact_tool_result_for_event(
        event=event,
        result=result,
        redactor=redactor,
    )
    if not redactor.has_values:
        return event, redacted_result
    payload: dict[str, Any] = {}
    for key, value in event.payload.items():
        if key == "result":
            continue
        if (
            key == EXECUTION_PROFILE_FINGERPRINT_FIELD
            and type(value) is str
            and event_payload_authority_is_runtime_generated(
                event,
                field_name=key,
                value=value,
            )
        ):
            payload[key] = value
        elif key in _POLICY_DENIAL_CONTROL_PAYLOAD_FIELDS:
            payload[key] = copy_json_value(value, key)
        else:
            payload[key] = redactor.redact_json(value)
    payload["result"] = redacted_result.model_dump()
    return event.model_copy(update={"payload": payload}), redacted_result


def _redact_policy_denial_result(
    result: ToolResult,
    redactor: SecretRedactor,
) -> ToolResult:
    if type(result) is not ToolResult:
        raise TypeError("Policy denial results must be ToolResult instances.")
    if not isinstance(redactor, SecretRedactor):
        raise TypeError("redactor must be a SecretRedactor.")
    if not redactor.has_values:
        return result
    structured = result.structured
    if structured is not None:
        structured = {
            key: (
                copy_json_value(value, key)
                if key in _POLICY_DENIAL_CONTROL_RESULT_FIELDS
                else redactor.redact_json(value)
            )
            for key, value in structured.items()
        }
    return ToolResult(
        content=redactor.redact_text(result.content),
        structured=structured,
        artifacts=redactor.redact_json(result.artifacts),
        is_error=result.is_error,
    )


def _bound_policy_denial_event(*, event: Event, result: ToolResult) -> tuple[Event, ToolResult]:
    if event.type != EventType.TOOL_CALL_BLOCKED or "denied_by" not in event.payload:
        return event, result
    bounded_result = _bound_policy_denial_result(result)
    payload = dict(event.payload)
    reason = payload.get("reason")
    if type(reason) is not str:
        raise ValueError("`reason` must be a string.")
    payload["reason"] = _bound_policy_denial_text(require_nonblank(reason, "reason"))
    payload["result"] = bounded_result.model_dump()
    return event.model_copy(update={"payload": payload}), bounded_result


def _project_staged_terminal_event(
    event: Event,
    *,
    redactor: SecretRedactor,
    trust_persisted_tool_result_authority: bool = False,
) -> Event:
    """Progressively sanitize one private terminal without changing its identity."""

    if trust_persisted_tool_result_authority:
        event = web_access_results.restore_persisted_web_access_result_authority(event)
        event = shared_artifact_results.restore_persisted_shared_artifact_result_authority(event)
    raw_result = event.payload.get("result")
    if type(raw_result) is not dict:
        raise ValueError("Staged terminal event requires a tool result object.")
    result = tool_results.tool_result_from_payload(raw_result)
    projected, _ = _prepare_tool_result_event(
        event=event,
        result=result,
        redactor=redactor,
    )
    return copy_event(projected)


def _staged_terminal_argument_projections(
    event: Event,
) -> tuple[
    tool_argument_publication.ToolArgumentProjection,
    tool_argument_publication.ToolArgumentProjection,
]:
    """Recover sealed public and hook arguments from a durable terminal stage."""

    projection = tool_argument_publication.terminal_argument_projection(
        event.payload,
        legacy_arguments={},
    )
    effective_arguments = event.payload.get("effective_arguments")
    if projection.state == "unavailable":
        if effective_arguments is not None:
            raise ValueError("Unavailable staged arguments cannot carry effective arguments.")
        return projection, projection
    if effective_arguments is None:
        return projection, projection
    if type(effective_arguments) is not dict:
        raise TypeError("Staged effective arguments must be an object.")
    return (
        projection,
        tool_argument_publication.ToolArgumentProjection(
            state="finalized",
            arguments=effective_arguments,
        ),
    )


def restore_staged_terminal_authority(
    event: Event,
    *,
    session_id: str,
    tool_round_identity: ToolRoundIdentity,
    tool_exposure: ResolvedToolExposureAuthority | None = None,
) -> Event:
    """Restore only typed authority erased by checkpoint serialization."""

    session_id = require_clean_nonblank(session_id, "session_id")
    identity = copy_tool_round_identity(tool_round_identity)
    if event.session_id != session_id or not identity.matches_payload(event.payload):
        raise RuntimeError("Staged terminal conflicts with its durable round owner.")
    tool_call_id = event.payload.get("tool_call_id")
    if type(tool_call_id) is not str or not tool_call_id:
        raise ValueError("Staged terminal lost its tool-call identity.")
    restored = event_with_runtime_generated_id(copy_event(event))
    additional_fields = ["tool_call_id"]
    if (
        restored.type is EventType.TOOL_CALL_BLOCKED
        and restored.payload.get("blocked_by") == "tool_exposure"
    ):
        if tool_exposure is None:
            raise RuntimeError("Staged tool-exposure terminal has no durable exposure owner.")
        if type(tool_exposure) is not ResolvedToolExposureAuthority:
            raise TypeError("tool_exposure must be a ResolvedToolExposureAuthority.")
        exposure = tool_exposure
        validate_tool_exposure_terminal_event(restored, tool_exposure=exposure)
        payload = dict(restored.payload)
        payload["profile_id"] = exposure.profile_id
        payload["exposure_fingerprint"] = exposure.fingerprint
        restored = restored.model_copy(update={"payload": payload})
        additional_fields.extend(("profile_id", "exposure_fingerprint"))
    restored = _event_with_tool_round_authority(restored, identity, *additional_fields)
    if restored.interaction_id is not None:
        restored = event_with_runtime_envelope_authority(restored, "interaction_id")
    controls, _references = tool_results.runtime_tool_event_boundary_controls(
        restored.payload,
        include_terminal_controls=restored.type
        in {EventType.TOOL_CALL_COMPLETED, EventType.TOOL_CALL_FAILED},
    )
    if "tool_result_projection" in controls:
        restored = event_with_runtime_nested_payload_authority(
            restored,
            _TOOL_RESULT_PROJECTION_PROVENANCE_PATH,
        )
    return restored


def policy_denial_payload_fields(
    *,
    tool_name: str,
    denied_by: str,
    decision: str,
    reason: str,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    return {
        "tool_name": require_clean_nonblank(tool_name, "tool_name"),
        "denied_by": require_clean_nonblank(denied_by, "denied_by"),
        "decision": require_clean_nonblank(decision, "decision"),
        "reason": require_nonblank(reason, "reason"),
        "metadata": copy_json_value(metadata, "metadata"),
    }


def _redactor_for_tool_calls(
    base: SecretRedactor,
    *,
    registered_agent: runtime_records.RegisteredAgentState,
    tool_calls: list[runtime_records.ToolCallRequest],
) -> SecretRedactor:
    redactor = base
    for tool_call in tool_calls:
        registered_tool = registered_agent.executable_tool(tool_call.name)
        if registered_tool is None:
            continue
        tool = registered_tool.tool
        if isinstance(tool, McpToolAdapter):
            redactor = redactor.merged_with(tool.toolset.secret_redactor)
    return redactor
