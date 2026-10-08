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
from collections.abc import AsyncGenerator, AsyncIterator, Awaitable, Callable, Iterable, Mapping
from contextlib import aclosing
from copy import deepcopy
from dataclasses import dataclass, replace
from datetime import datetime
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
    await_shielded_task_outcome,
    consume_pending_task_cancellation,
    unexpected_child_cancellation_error,
)
from cayu._validation import (
    canonical_durable_json_bytes,
    copy_durable_json_object,
    copy_durable_metadata,
    copy_durable_record,
    copy_json_value,
    require_clean_nonblank,
    require_nonblank,
)
from cayu.approvals.tools import (
    PendingToolApproval,
    PendingToolCallApproval,
    ToolPolicyEvidence,
)
from cayu.approvals.user_input import (
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
from cayu.artifacts._images import ImageDecodePolicy
from cayu.budgets._run_limit_accounting import (
    RunLimitAccountingContext,
    pause_run_limit_accounting_context,
)
from cayu.budgets.base import (
    BudgetLimit,
    copy_request_budget_limits,
)
from cayu.budgets.run_limits import RunLimits, copy_run_limits
from cayu.context.structured_output import (
    StructuredOutputSpec,
    copy_structured_output_spec,
)
from cayu.context.thinking import ThinkingConfig
from cayu.events import (
    Event,
    EventType,
    event_with_runtime_payload_authority,
)
from cayu.execution_profiles import (
    EXECUTION_PROFILE_FINGERPRINT_FIELD,
    ExecutionProfileIdentity,
    event_with_execution_profile_authority,
)
from cayu.execution_units import (
    ModelAttemptIdentity,
    ToolRoundIdentity,
    copy_tool_round_identity,
)
from cayu.knowledge._publication import KnowledgePublicationScope
from cayu.mcp.tools import McpToolAdapter, McpToolset
from cayu.messages import Message
from cayu.observability.hooks import (
    RuntimeHookRuntime,
)
from cayu.providers.retry_policy import RetryPolicy, copy_retry_policy
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
from cayu.runtime import _tool_execution as tool_execution
from cayu.runtime import _tool_round_recovery as tool_round_recovery
from cayu.runtime import _transcript as transcript_helpers
from cayu.runtime._auxiliary_inference import AuxiliaryInferenceOwner
from cayu.runtime._auxiliary_invocation import AuxiliaryInvocationPolicy
from cayu.runtime._browser_control_service import BrowserControlService
from cayu.runtime._durable_tool_round import DurableToolRound
from cayu.runtime._durable_tool_round import (
    InterruptedToolRoundRequest as InterruptedToolRoundRequest,
)
from cayu.runtime._durable_tool_round import (
    _interrupted_tool_call_event as _interrupted_tool_call_event,
)
from cayu.runtime._durable_tool_round import (
    _interrupted_tool_call_outcome as _interrupted_tool_call_outcome,
)
from cayu.runtime._environment_exposure import (
    require_environment_exposed,
)
from cayu.runtime._event_writer import RuntimeEventWriter, prepare_runtime_event
from cayu.runtime._interruption_coordinator import (
    _INTERRUPTION_TYPE_OPERATOR_REQUESTED,
    _PENDING_SESSION_INTERRUPT_CHECKPOINT_KEY,
)
from cayu.runtime._invocation_lifecycle import InvocationContext
from cayu.runtime._phase_timing import timed_phase, timed_tool_round
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
from cayu.runtime._tool_effect_state import (
    ToolEffectReconciliationCleanupFailure,
    ToolEffectReconciliationRequired,
    ToolEffectStateOwner,
    ToolEffectTerminal,
)
from cayu.runtime._tool_invocation.admission import (
    ToolApprovalRequired,
    ToolInvocationAdmission,
    _planned_pending_tool_round,
    _registered_mcp_tool_authority_is_unavailable,
    _registered_tool_argument_error,
    _require_matching_policy_round,
    _taint_policy,
    _tool_dispatch_authority_is_current,
)
from cayu.runtime._tool_invocation.cancellation import (
    _await_post_tool_operation,
    _contains_process_signal,
    _iterate_post_tool_events,
    _raise_preserved_post_tool_cancellation,
    _raise_restored_post_tool_cancellation,
    _receive_restored_post_tool_cancellation,
)
from cayu.runtime._tool_invocation.context import (
    _artifact_store_id,
    _environment_name,
    _event_with_targeted_tool_invocation_authority,
    _published_argument_presence,
    _restore_targeted_tool_invocation_event_authority,
    _targeted_tool_invocation_payload,
    _workspace,
    _workspace_id,
)
from cayu.runtime._tool_invocation.dispatch import ToolInvocationDispatch
from cayu.runtime._tool_invocation.evidence import InvocationEvidence, InvocationPublication
from cayu.runtime._tool_invocation.hooks import (
    ToolInvocationHooks,
    _BeforeToolCallResolution,
    _private_argument_short_circuit_result,
    _project_tool_call_for_hook,
    _redact_event_for_invocation,
)
from cayu.runtime._tool_invocation.resources import ToolInvocationCall, ToolInvocationResources
from cayu.runtime._tool_invocation.terminal import (
    DeferredTerminalCaptureRecorder,
    DeferredTerminalStager,
    ToolTerminalPublisher,
)
from cayu.runtime._tool_invocation.workspace_capture import (
    _MAX_RETAINED_WORKSPACE_CAPTURE_OPERATIONS,
    _observe_workspace_revision,
    _record_workspace_mutation_after,
    _workspace_binding_generation_id,
    _workspace_mutation_incomplete_event,
    _workspace_mutation_window_id,
    _workspace_observation_terminal_view,
    _workspace_revision_observer_is_runtime_owned,
    _workspace_revision_observer_name,
    _workspace_writer_isolation,
    _WorkspaceCaptureResult,
)
from cayu.runtime._tool_round_staging import (
    CheckpointTransform,
    _event_with_tool_round_authority,
    _prepare_tool_result_event,
    _redactor_for_tool_calls,
    _tool_terminal_payload_limits,
)
from cayu.runtime.mcp_manifest_policy import (
    McpManifestPolicy,
    McpManifestPolicyAction,
    McpManifestPolicyDecision,
    McpManifestPolicyError,
    mcp_manifest_policy_payload,
)
from cayu.sessions import _pending_approval_reader as pending_approval_reader
from cayu.sessions import _pending_tool_round as pending_rounds
from cayu.sessions import _pending_tool_round_reader as pending_round_reader
from cayu.sessions._checkpoint_secret_validation import (
    require_secret_free_durable_object as _require_secret_free_durable_object,
)
from cayu.sessions._execution_profile_checkpoint import (
    active_invocation_execution_profile_from_checkpoint,
)
from cayu.sessions.base import (
    _MCP_MANIFEST_BASELINE_MAX_TOOLS,
    McpManifestBaseline,
    McpManifestBaselineLoadResult,
    McpManifestHistoryConflict,
    McpManifestPublicationResult,
    SessionStore,
    _mcp_authoritative_manifest_hash,
    _mcp_manifest_session_ref,
    _McpManifestBaselineEvidenceInvalid,
    runtime_publication_checkpoint_value_digest,
)
from cayu.sessions.records import Session, SessionStatus
from cayu.tools import _argument_publication as tool_argument_publication
from cayu.tools._operation_boundary import (
    BoundedInvocationOperationRegistry,
    await_invocation_operation,
)
from cayu.tools._redaction import InvocationRedactorSnapshot
from cayu.tools._resources import (
    WorkspaceMutationSettlementError,
)
from cayu.tools._runner import (
    is_current_runner_cancellation_group,
    sanitize_runner_failure,
    sanitize_runner_failure_group,
)
from cayu.tools.base import (
    _TOOL_POLICY_DENIAL_SOURCE,
    ToolEffect,
    ToolResult,
)
from cayu.tools.discovery import (
    ToolDiscoveryProjectionKind,
    resolve_tool_discovery_projection,
)
from cayu.tools.exposure import (
    NOT_EXPOSED_IN_REQUEST_REASON,
    ResolvedToolExposureAuthority,
    unexposed_tool_result,
    validate_resolved_tool_exposure_authority,
)
from cayu.tools.gateway import (
    dynamic_tool_reference_rejection,
    targeted_tool_rejection_content,
)
from cayu.tools.policy import (
    TAINT_LABELS_METADATA_KEY,
    TOOL_POLICY_REAUTHORIZATION_METADATA_KEY,
    ToolPolicy,
    ToolPolicyDecision,
    ToolPolicyResult,
)
from cayu.tools.result_projection import (
    ToolResultProjectionPolicy,
)
from cayu.vaults.redaction import SecretRedactor
from cayu.workspaces.checkpoint_lifecycle import (
    begin_workspace_checkpoint_mutation,
    complete_workspace_checkpoint_mutation,
    ensure_workspace_checkpoint,
)
from cayu.workspaces.mutation_attribution import (
    WorkspaceMutationWindow,
    begin_workspace_mutation_window,
)
from cayu.workspaces.observation_recovery import (
    WorkspaceObservationEvidenceState,
    WorkspaceObservationLifecycle,
    WorkspaceObservationPhase,
    WorkspaceObservationTerminalStatus,
    _admit_workspace_observation_intent,
    _project_workspace_observation_authority,
    await_workspace_observation_store_read,
    publish_workspace_observation_transition,
    workspace_observation_event_digest,
    workspace_observations_from_checkpoint,
)
from cayu.workspaces.revisions import (
    WorkspaceRevisionObservation,
    WorkspaceRevisionObservationStatus,
    WorkspaceWriterIsolationEvidence,
)

CheckpointTransformFactory = Callable[[dict[str, Any]], CheckpointTransform]


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


LimitEventStream = Callable[[ToolRoundLimitRequest], AsyncIterator[Event]]
InterruptedRoundEventStream = Callable[[InterruptedToolRoundRequest], AsyncGenerator[Event, None]]

_AMBIGUOUS_POLICY_RECOVERY_REASON = (
    "Tool policy planning was interrupted without a durable outcome; "
    "explicit approval is required before execution."
)
_AMBIGUOUS_POLICY_RECOVERY_METADATA = {
    "recovered": True,
    "policy_evaluation": "ambiguous",
}


class UserInputRequired(Exception):
    """Internal control signal for a durably checkpointed user-input pause."""

    def __init__(self, pending: PendingUserInput) -> None:
        super().__init__(f"Tool call awaits user input: {pending.tool_name}")
        self.pending = copy_pending_user_input(pending)


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
        auxiliary_inference: AuxiliaryInferenceOwner,
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
        knowledge_publication_scope: KnowledgePublicationScope,
        browser_control_service: BrowserControlService | None = None,
        image_decode_policy: ImageDecodePolicy | None = None,
        strict_common_budget_admission: bool = False,
    ) -> None:
        self._session_store = session_store
        self._event_writer = event_writer
        # Projections that outlived their timeout may still write artifacts,
        # and abandoned secret resolutions may still use a vault or proxy:
        # waited for by environment cleanup, never cancelled again.
        self._auxiliary_inference = auxiliary_inference
        self._session_control = session_control
        self._mcp_manifest_policy = mcp_manifest_policy
        self._secret_redactor = secret_redactor
        self._tool_timeout_seconds = tool_timeout_seconds
        self._max_parallel_tool_calls = max_parallel_tool_calls
        self._clock = clock
        self.resources = ToolInvocationResources(
            session_store=session_store,
            image_decode_policy=image_decode_policy,
            knowledge_publication_scope=knowledge_publication_scope,
            browser_control_service=browser_control_service,
            clock=clock,
        )
        self.admission = ToolInvocationAdmission(
            session_store=session_store,
            event_writer=event_writer,
            secret_redactor=secret_redactor,
            clock=clock,
        )
        self.hooks = ToolInvocationHooks(
            event_writer=event_writer, hook_runtime=hook_runtime, runtime_hooks=runtime_hooks
        )
        self.terminals = ToolTerminalPublisher(
            event_writer=event_writer,
            hooks=self.hooks,
            secret_redactor=secret_redactor,
            projection_policy=tool_result_projection_policy,
            clock=clock,
        )
        self._checkpoint_transform = checkpoint_transform
        self._apply_limit_evaluation = apply_limit_evaluation
        self._close_interrupted_round = close_interrupted_round
        self._strict_common_budget_admission = strict_common_budget_admission
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
        run_limit_accounting: RunLimitAccountingContext | None = None,
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
            run_limit_accounting=run_limit_accounting,
        )

    @timed_phase("authorization")
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
            await self.admission.prior_taint_labels(
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

            policy_result = await self.admission.authorize(
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
            await self.admission.prior_taint_labels(
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

    @timed_phase("admission")
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
    ) -> pending_rounds.PendingToolRound:
        """Atomically replace an unplanned round with its durable policy plan."""

        tool_round_identity = copy_tool_round_identity(tool_round_identity)
        redactor = _redactor_for_tool_calls(
            self._secret_redactor,
            registered_agent=registered_agent,
            tool_calls=tool_calls,
        )
        checkpoint = await self._session_store.load_checkpoint(session.id)
        checkpoint = {} if checkpoint is None else copy_durable_record(checkpoint, "checkpoint")
        pending_round = pending_round_reader.pending_tool_round_from_checkpoint(
            checkpoint,
            redactor=self._secret_redactor,
            consume_on_rejection=True,
            runtime_session=session,
        )
        if pending_round is None:
            raise RuntimeError("Session has no pending tool round for its policy plan.")
        _require_matching_policy_round(
            pending_round=pending_round,
            tool_round_identity=tool_round_identity,
            tool_calls=tool_calls,
        )
        if (
            pending_approval_reader.pending_approval_from_checkpoint(
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
            checkpoint[pending_rounds.PENDING_TOOL_ROUND_CHECKPOINT_KEY],
            "pending_tool_round",
        )
        eligible_statuses = {session.status} if recovered else {SessionStatus.RUNNING}
        planned_round_payload = _require_secret_free_durable_object(
            planned_round.model_dump(mode="json"),
            redactor=redactor,
            field_name="pending_tool_round",
            schema_root=pending_rounds.PENDING_TOOL_ROUND_CHECKPOINT_KEY,
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
                else copy_durable_record(current_checkpoint, "checkpoint")
            )
            if (
                current.get(pending_rounds.PENDING_TOOL_ROUND_CHECKPOINT_KEY)
                != source_round_payload
            ):
                raise RuntimeError("Pending tool round changed before policy publication.")
            if pending_approval_reader.PENDING_TOOL_APPROVAL_CHECKPOINT_KEY in current:
                raise RuntimeError("Session already has a pending tool approval.")
            if pending_approval_reader.APPROVAL_RESOLUTION_INTENT_CHECKPOINT_KEY in current:
                raise RuntimeError("Session has an orphaned approval resolution intent.")
            current[pending_rounds.PENDING_TOOL_ROUND_CHECKPOINT_KEY] = planned_round_payload
            return copy_durable_json_object(current, "checkpoint")

        await self._session_store.transform_checkpoint(session.id, publish_policy_plan)
        return planned_round

    @timed_phase("round_commit", shared=True)
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
        checkpoint = {} if checkpoint is None else copy_durable_record(checkpoint, "checkpoint")
        pending_round = pending_round_reader.pending_tool_round_from_checkpoint(
            checkpoint,
            redactor=self._secret_redactor,
            consume_on_rejection=True,
            runtime_session=session,
        )
        if pending_round is None:
            raise RuntimeError("Session has no pending tool round for its user-input pause.")
        _require_matching_policy_round(
            pending_round=pending_round,
            tool_round_identity=tool_round_identity,
            tool_calls=tool_calls,
        )
        if (
            pending_approval_reader.pending_approval_from_checkpoint(
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
            runtime_session=session,
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
            run_limit_accounting=pause_run_limit_accounting_context(
                pending_round.run_limit_accounting, now=self._clock()
            ),
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
            runtime_session=session,
        )
        source_round_payload = copy_json_value(
            checkpoint[pending_rounds.PENDING_TOOL_ROUND_CHECKPOINT_KEY],
            "pending_tool_round",
        )
        target_checkpoint = copy_durable_record(checkpoint, "checkpoint")
        target_checkpoint.pop(pending_rounds.PENDING_TOOL_ROUND_CHECKPOINT_KEY)
        target_checkpoint[PENDING_USER_INPUT_CHECKPOINT_KEY] = pending_payload
        target_checkpoint = _require_secret_free_durable_object(
            target_checkpoint,
            redactor=redactor,
            field_name="checkpoint",
            runtime_session=session,
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

    @timed_phase("admission")
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
    ) -> tuple[dict[str, Any], pending_rounds.PendingToolRound]:
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
            runtime_session=session,
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
        copied_checkpoint = copy_durable_record(checkpoint, "checkpoint")
        pending_approval = pending_approval_reader.pending_approval_from_checkpoint(
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
            runtime_session=current_session,
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
        interrupt_payload[approval_support.APPROVAL_INTERRUPT_CLOSE_INTENT_KEY] = (
            approval_support.approval_interrupt_close_intent(approval)
        )
        copied[_PENDING_SESSION_INTERRUPT_CHECKPOINT_KEY] = interrupt_payload
        return copied

    @timed_phase("unattributed")
    async def execute_tool_call(
        self,
        *,
        session: Session,
        registered_agent: runtime_records.RegisteredAgentState,
        registered_environment: runtime_records.RegisteredEnvironment | None,
        tool_call: runtime_records.ToolCallRequest,
        request_metadata: dict[str, Any],
        budget_limits: tuple[BudgetLimit, ...],
        task_id: str | None,
        model_step: int | None = None,
        execution_profile: ExecutionProfileIdentity | None = None,
        invocation_context: InvocationContext | None = None,
        check_policy: bool = True,
        auxiliary_invocation_policy: AuxiliaryInvocationPolicy | None = None,
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
    ) -> AsyncGenerator[tuple[Event, runtime_records.ToolCallOutcome | None], None]:
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
        if registered_environment is not None:
            if invocation_context is None or execution_profile is None:
                raise RuntimeError(
                    "Environment-backed tool execution requires frozen exposure authority."
                )
            require_environment_exposed(
                registered_environment,
                session=session,
                invocation_context=invocation_context,
                registered_agent=registered_agent,
                execution_profile=execution_profile,
            )
            await ensure_workspace_checkpoint(self._session_store, session, registered_environment)
        registered_tool = registered_agent.executable_tool(tool_call.name)
        if rejoin_targeted_invocation:
            rejoined_events = await self.admission.rejoin_targeted_call(
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
            if (
                rejection.dispatch_kind == "gateway"
                and invocation_context is not None
                and resolve_tool_discovery_projection(
                    registered_agent.tool_discovery_mode,
                    provider=invocation_context.registered_provider.provider,
                    model=invocation_context.binding.model,
                )
                is ToolDiscoveryProjectionKind.SEARCH_TOOLS
            ):
                corrective = dynamic_tool_reference_rejection(rejection.reason)
                result = ToolResult(
                    content=corrective.message,
                    structured=corrective.model_dump(mode="json"),
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
                call=replace(tool_call, arguments={}, arguments_state="unavailable"),
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
                call=replace(tool_call, arguments={}, arguments_state="unavailable"),
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
            projected_call = replace(tool_call, arguments={}, arguments_state="unavailable")
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
        publication = InvocationPublication(
            tool_call=tool_call,
            redactor=invocation_redactor,
            observer=publication_snapshot_observer,
        )

        started_event: Event | None = None
        if registered_tool is not None and not registered_tool.publish_arguments:
            publish_arguments_as_unavailable = True
        if taint_labels is None:
            taint_labels = frozenset(
                await self.admission.prior_taint_labels(
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
        argument_error = _registered_tool_argument_error(registered_tool, tool_call.arguments)
        if emit_started and argument_error is None:
            payload: dict[str, Any] = {
                "tool_call_id": tool_call.id,
                "idempotency_key": idempotency_key,
                **_targeted_tool_invocation_payload(tool_call),
                **tool_argument_publication.quarantined_argument_fields(),
                **_published_argument_presence(
                    tool_call, registered_agent.executable_tool(tool_call.name)
                ),
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
            publication_snapshot = await publication.static_scope()
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
            async for event in self.terminals.publish_result(
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

        if argument_error is not None:
            # Approval of an older malformed call cannot bypass current input validation.
            check_policy = True
            policy_result = ToolPolicyResult(
                decision=ToolPolicyDecision.DENY,
                reason=argument_error,
                metadata={"reason": "invalid_arguments"},
            )
        if check_policy:
            if policy_result is None:
                resolved_policy_result = await self.admission.authorize(
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
                publication_snapshot = await publication.static_scope()
                public_policy_result = (
                    resolved_policy_result
                    if argument_error is not None
                    else approval_support.public_policy_denial_result(
                        secret_resolution_scope=policy_output_secret_resolution_scope,
                        policy_result=resolved_policy_result,
                        publish_arguments=registered_tool.publish_arguments,
                    )
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
                async for event in self.terminals.publish_result(
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
                    argument_projection=(
                        tool_argument_publication.finalized_argument_projection(
                            tool_call.arguments,
                            redactor=invocation_redactor,
                            scope_finalized=True,
                        )
                        if argument_error is not None
                        and policy_output_secret_resolution_scope == "static"
                        and registered_tool.publish_arguments
                        else None
                    ),
                    publication_snapshot=publication_snapshot,
                ):
                    yield event
                return
            if resolved_policy_result.decision == ToolPolicyDecision.REQUIRE_APPROVAL:
                approval, approval_events = await self.admission.pause_for_approval(
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
        async for hook_event in self.hooks.before_call(
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
            publication_snapshot = await publication.static_scope()
            block_reason = (
                before_resolution.block_reason
                if registered_tool.publish_arguments
                else "Tool call blocked by a before_tool_call hook."
            )
            async for event in self.terminals.emit_result(
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
            publication_snapshot = await publication.static_scope()
            short_result = (
                before_resolution.short_circuit_result
                if registered_tool.publish_arguments
                else _private_argument_short_circuit_result(before_resolution.short_circuit_result)
            )
            async for event in self.terminals.emit_result(
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
            reauthorization = await self.admission.authorize(
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
                publication_snapshot = await publication.static_scope()
                if reauthorization.decision == ToolPolicyDecision.DENY:
                    public_reauthorization = (
                        reauthorization
                        if _registered_tool_argument_error(
                            registered_tool, effective_tool_call.arguments
                        )
                        is not None
                        else approval_support.public_policy_denial_result(
                            secret_resolution_scope=policy_output_secret_resolution_scope,
                            policy_result=reauthorization,
                            publish_arguments=registered_tool.publish_arguments,
                        )
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
                async for event in self.terminals.emit_result(
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

        invocation_call = ToolInvocationCall(
            session=session,
            registered_agent=registered_agent,
            registered_environment=registered_environment,
            registered_tool=registered_tool,
            tool_call=tool_call,
            effective_tool_call=effective_tool_call,
            tool_round_identity=tool_round_identity,
            execution_profile=execution_profile,
            invocation_context=invocation_context,
            task_id=task_id,
            budget_limits=budget_limits,
            tool_exposure=tool_exposure,
            environment_name=environment_name,
            idempotency_key=idempotency_key,
            model_step=model_step,
            approval_id=approval_id,
            input_id=input_id,
        )
        evidence = InvocationEvidence(
            call=invocation_call,
            secret_scope=invocation_secret_scope,
            publication=publication,
            session_store=self._session_store,
            event_writer=self._event_writer,
            resolved_redactor_observer=resolved_redactor_observer,
        )
        runner_events = evidence.runner_events
        invocation_resources = self.resources.bind(
            invocation_call,
            ctx_metadata=ctx_metadata,
            invocation_secret_scope=invocation_secret_scope,
            proxy_authorizations=proxy_authorizations,
            redactor_provider=evidence.redactor,
            observe_runner_execution=evidence.observe_runner,
            persist_resolved_secret_projection=evidence.persist_resolved,
        )
        tool_context = invocation_resources.context
        raw_workspace = invocation_resources.raw_workspace
        direct_workspace_mutations = invocation_resources.direct_workspace_mutations
        workspace_mutation_owner = invocation_resources.workspace_mutation_owner
        workspace_receipt_artifact_store = invocation_resources.workspace_receipt_artifact_store
        workspace_receipt_artifact_unavailable_detail = (
            invocation_resources.workspace_receipt_artifact_unavailable_detail
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
            await begin_workspace_checkpoint_mutation(
                self._session_store,
                session,
                registered_environment,
                window_id=workspace_window_id,
                tool_call_id=tool_call.id,
                interaction_id=None if started_event is None else started_event.interaction_id,
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

        async def consume_post_tool_cancellation(
            cancellation: asyncio.CancelledError,
        ) -> asyncio.CancelledError | None:

            nonlocal post_tool_cancellation_requests_consumed
            current_task = asyncio.current_task()
            await _receive_restored_post_tool_cancellation()
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
                uncertainty_error: ToolEffectReconciliationRequired | None = None
                try:
                    if registered_environment is not None and workspace_window_id is not None:
                        await complete_workspace_checkpoint_mutation(
                            self._session_store,
                            session,
                            registered_environment,
                            window_id=workspace_window_id,
                            window_exclusive=lambda: (
                                not workspace_attribution_window.overlap_detected
                            ),
                            successful=(
                                event.type is EventType.TOOL_CALL_COMPLETED
                                and not outcome.result.is_error
                                and not snapshot.unsafe_output
                                and workspace_settlement_failure is None
                            ),
                        )
                    try:
                        staged_event = await deferred_terminal_stager(
                            event,
                            outcome,
                            allow_modification,
                            publish_before_hooks,
                            snapshot,
                        )
                    except ToolEffectReconciliationRequired as exc:
                        uncertainty_error = exc
                        effect_owner = ToolEffectStateOwner(self._session_store)
                        effect = await effect_owner.resolve_call(
                            session,
                            tool_round_id=tool_round_identity.tool_round_id,
                            tool_call_id=tool_call.id,
                        )
                        if effect is not None and effect.state == "prepared":
                            # No dispatch was consumed, hence no uncertainty event
                            # exists to bind to this observation. Keep the durable
                            # lifecycle for recovery, which owns prepared settlement.
                            workspace_attribution_window.close(discard_history=True)
                            raise
                        uncertain_events = await effect_owner.load_uncertainty_events(
                            session,
                            tool_round_id=tool_round_identity.tool_round_id,
                            tool_call_ids=(tool_call.id,),
                        )
                        if len(uncertain_events) != 1:
                            raise RuntimeError(
                                "Workspace capture lost exact uncertainty evidence."
                            ) from exc
                        staged_event = uncertain_events[0]
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
                            observed_cancellation = await consume_post_tool_cancellation(exc)
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
                            invocation_secrets.initialize_cancellation_evidence(
                                observed_cancellation
                            )
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
                        finalized_stage = (
                            staged_event
                            if uncertainty_error is not None
                            else await deferred_terminal_capture_recorder(
                                staged_event.model_copy(
                                    update={"payload": final_payload}, deep=True
                                )
                            )
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
                                detail_code=(
                                    capture_failure_detail or "receipt_publication_failed"
                                ),
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
                        if uncertainty_error is not None:
                            raise uncertainty_error
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
                    finalized_stage = (
                        staged_event
                        if uncertainty_error is not None
                        else await deferred_terminal_capture_recorder(
                            staged_event.model_copy(update={"payload": final_payload}, deep=True)
                        )
                    )
                    if (
                        finalized_stage.id != staged_event.id
                        or workspace_observation_event_digest(finalized_stage)
                        != staged_lifecycle.tool_outcome_event_digest
                    ):
                        raise RuntimeError(
                            "Workspace capture changed the authoritative tool outcome."
                        )
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
                    if uncertainty_error is not None:
                        raise uncertainty_error
                    return finalized_stage
                except Exception as cleanup_error:
                    if uncertainty_error is None or cleanup_error is uncertainty_error:
                        raise
                    raise ToolEffectReconciliationCleanupFailure(
                        uncertainty_error,
                        cast("Exception", sanitize_runner_failure(cleanup_error)),
                    ) from None

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
                snapshot = invocation_secret_scope.seal_for_publication()
                projection = (
                    tool_argument_publication.unavailable_argument_projection()
                    if publish_arguments_as_unavailable
                    else tool_argument_publication.finalized_argument_projection(
                        model_arguments,
                        redactor=snapshot.redactor,
                        scope_finalized=not snapshot.unsafe_output,
                    )
                )
                interrupted_payload = {
                    **interrupted_event.payload,
                    **projection.payload_fields(),
                    tool_argument_publication.ARGUMENTS_EXACT_FIELD: (
                        tool_argument_publication.argument_projection_is_exact(
                            projection,
                            private_arguments=model_arguments,
                        )
                    ),
                }
                if projection.state == "finalized" and effective_arguments_payload:
                    interrupted_payload["effective_arguments"] = snapshot.redactor.redact_json(
                        effective_tool_call.arguments
                    )
                interrupted_event = interrupted_event.model_copy(
                    update={"payload": interrupted_payload}
                )
                await effective_terminal_stager(
                    interrupted_event,
                    interrupted_outcome,
                    False,
                    False,
                    snapshot,
                )
            except BaseException as closure_error:
                if isinstance(closure_error, ToolEffectReconciliationCleanupFailure):
                    closure_error = closure_error.cleanup
                elif isinstance(closure_error, ToolEffectReconciliationRequired):
                    # Closure completed or prepared observation awaits recovery;
                    # neither case adds a failure to the original cancellation.
                    return
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
                # the tool. Retain bounded typed failure evidence, never raw
                # extension messages, tracebacks, or mutable exception state.
                failure: BaseException = sanitize_runner_failure(closure_error)
                prior = exception_cause(interrupt)
                if prior is not None:
                    failure = BaseExceptionGroup(
                        "Tool interruption and workspace observation closure failures.",
                        [prior, failure],
                    )
                # TaskGroup reconstruction clears unauthenticated causal
                # chains. Carry this fixed, runtime-owned diagnostic through
                # the same channel used by mutation settlement cleanup.
                for candidate in iter_exception_tree(interrupt):
                    if isinstance(candidate, asyncio.CancelledError):
                        attach_runner_cancellation_failure(candidate, failure)
                set_exception_cause(interrupt, failure)

        dispatch = ToolInvocationDispatch(
            call=invocation_call,
            context=tool_context,
            secret_scope=invocation_secret_scope,
            session_store=self._session_store,
            auxiliary_inference=self._auxiliary_inference,
            auxiliary_policy=auxiliary_invocation_policy,
            tool_timeout_seconds=self._tool_timeout_seconds,
            strict_common_budget_admission=self._strict_common_budget_admission,
            runner_events=runner_events,
            settle_workspace=(
                close_workspace_mutation_window if workspace_window_id is not None else None
            ),
        )
        try:
            execution_outcome = await dispatch.run()
            for auxiliary_event in dispatch.events:
                yield auxiliary_event, None
        except tool_execution.ToolDispatchAdmissionRefusal as refused:
            refusal = refused.refusal
            if dispatch.effect is not None:
                try:
                    # This proof comes from the runtime callback, never a
                    # tool-authored result or exception. Selection and its
                    # evidence share the existing effect transaction.
                    intent = dispatch.effect.intent
                    event = event_with_execution_profile_authority(
                        Event(
                            type=EventType.TOOL_CALL_FAILED,
                            session_id=intent.session_id,
                            interaction_id=intent.interaction_id,
                            agent_name=intent.agent_name,
                            environment_name=intent.environment_name,
                            tool_name=intent.tool_name,
                            payload={
                                **{
                                    name: getattr(intent, name)
                                    for name in (
                                        "model_step_id",
                                        "model_attempt_id",
                                        "tool_round_id",
                                        "tool_call_id",
                                        "idempotency_key",
                                    )
                                },
                                **(
                                    {"approval_id": intent.approval_id}
                                    if intent.approval_id is not None
                                    else {}
                                ),
                                **(
                                    {"input_id": intent.pause_id}
                                    if intent.pause_id is not None
                                    else {}
                                ),
                                "result": ToolResult(
                                    content="Tool was not invoked because execution admission was refused.",
                                    is_error=True,
                                ).model_dump(mode="json"),
                            },
                        ),
                        execution_profile,
                    )
                    event = self._event_writer.prepare(
                        _event_with_tool_round_authority(
                            event,
                            tool_round_identity,
                            *(
                                field
                                for field in ("approval_id", "input_id")
                                if field in event.payload
                            ),
                        )
                    )
                    await ToolEffectStateOwner(self._session_store).transition(
                        dispatch.effect,
                        state="failed",
                        run_epoch=session.run_epoch,
                        terminal=ToolEffectTerminal(
                            event_id=event.id,
                            result_digest=hashlib.sha256(
                                canonical_durable_json_bytes(
                                    event.payload["result"], "effect_terminal_result"
                                )
                            ).hexdigest(),
                        ),
                        events=(event,),
                    )
                    await self._event_writer.fan_out_persisted([event])
                except asyncio.CancelledError as cancellation:
                    secondary = exception_cause(cancellation)
                    invocation_secrets.retain_admission_refusal(
                        cancellation, refusal, settlement_failure=secondary
                    )
                    # Approval and user-input continuations do not cross the
                    # ordinary round's final sanitizer. Publish the same safe
                    # diagnostic snapshot at this shared refusal boundary.
                    invocation_secrets.sanitize_external_cancellation(cancellation)
                    raise cancellation from exception_cause(cancellation)
                except BaseException as settlement_failure:
                    raise BaseExceptionGroup(
                        "Execution admission refusal and settlement failure.",
                        [refusal, settlement_failure],
                    ) from None
            raise refusal from None
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
                await evidence.persist_interrupted(exc)
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
            cancellation_redelivered = await evidence.persist_interrupted(exc)
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
                    quarantine_output=publish_arguments_as_unavailable,
                ):
                    yield event, None
            raise
        except (KeyboardInterrupt, SystemExit, GeneratorExit):
            if workspace_mutation_owner is not None:
                workspace_mutation_owner.seal_and_transfer_pending()
            raise

        result_event: Event | None = None
        try:
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
                    post_tool_cancellation = await consume_post_tool_cancellation(exc)
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
            # The tool has returned an exact result. Sealed invocation/runner
            # evidence precedes terminal staging, so cancellation here must not
            # discard that result and turn a completed effect into ambiguity.
            publication_outcome = await await_shielded_task_outcome(
                asyncio.create_task(evidence.persist_sealed(publication_snapshot))
            )
            post_tool_cancellation_requests_consumed += (
                publication_outcome.cancellation_requests_consumed
            )
            if publication_outcome.cancellation is not None:
                observed_cancellation = await consume_post_tool_cancellation(
                    publication_outcome.cancellation
                )
                post_tool_cancellation = post_tool_cancellation or observed_cancellation
                if post_tool_cancellation is not None:
                    invocation_secrets.initialize_cancellation_evidence(post_tool_cancellation)
                    invocation_secrets.set_cancellation_redactor(
                        post_tool_cancellation, invocation_secret_scope.redactor
                    )
                    invocation_secrets.set_cancellation_tool_call_id(
                        post_tool_cancellation, tool_call.id
                    )
            if publication_outcome.error is not None:
                if post_tool_cancellation is not None:
                    _raise_preserved_post_tool_cancellation(
                        post_tool_cancellation,
                        publication_outcome.error,
                        restore_cancellation_requests=post_tool_cancellation_requests_consumed,
                    )
                raise publication_outcome.error

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
                    payload={},
                )
                result_event = result_event.model_copy(update={"payload": payload})
                (
                    result_event,
                    result,
                    size_failure,
                    initial_projection_cancellation,
                    initial_projection_requests,
                ) = await _await_post_tool_operation(
                    self.terminals.prepare_bounded_result(
                        event=result_event,
                        result=result,
                        registered_tool=registered_tool,
                        session=session,
                        registered_environment=registered_environment,
                        tool_call=effective_tool_call,
                        redactor=publication_snapshot.redactor,
                    ),
                    cancellation=post_tool_cancellation,
                    restore_cancellation_requests=post_tool_cancellation_requests_consumed,
                )
                if size_failure is not None:
                    execution_outcome = size_failure
                if initial_projection_cancellation is not None:
                    post_tool_cancellation_requests_consumed += initial_projection_requests
                    post_tool_cancellation = await consume_post_tool_cancellation(
                        initial_projection_cancellation
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
                        _projection_failure,
                        projection_cancellation,
                        projection_requests,
                    ) = await _await_post_tool_operation(
                        self.terminals.project_result(
                            event=result_event,
                            result=result,
                            session=session,
                            registered_environment=registered_environment,
                            tool_call=effective_tool_call,
                            effect=registered_tool.effect,
                            redactor=redactor,
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
                            arguments_state=argument_projection.state,
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
                    quarantine_output=publish_arguments_as_unavailable,
                ),
                cancellation=post_tool_cancellation,
                restore_cancellation_requests=post_tool_cancellation_requests_consumed,
            ):
                if published_terminal_event is not None:
                    yield event, None
                else:
                    proxy_events.append(event)
            if projection_cancellation is not None:
                _raise_restored_post_tool_cancellation(
                    post_tool_cancellation or projection_cancellation,
                    restore_cancellation_requests=(
                        post_tool_cancellation_requests_consumed + projection_requests
                    ),
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
                    self.terminals.emit_result(
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
                    self.hooks.after_call(
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
            try:
                async for event in _iterate_post_tool_events(
                    self.terminals.publish_result(
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
            except ToolEffectReconciliationRequired:
                # These diagnostic events were already persisted after sealing
                # the invocation's secret scope. Uncertainty replaces the tool
                # terminal, not delivery of its preceding runner/proxy evidence.
                if not terminal_recorded:
                    for diagnostics in (runner_events, workspace_events, proxy_events):
                        for diagnostic_event in diagnostics:
                            yield diagnostic_event, None
                raise
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
        except BaseException as publication_failure:
            if dispatch.effect is not None:
                from cayu.runtime._tool_effect_conflicts import raise_after_tool_dispatch_audit

                await raise_after_tool_dispatch_audit(
                    self._session_store,
                    dispatch.effect,
                    publication_failure,
                    candidate_event_id=None if result_event is None else result_event.id,
                )
            raise

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

    def detached_environment_work(self) -> set[asyncio.Future[Any]]:
        """Timed-out projections and abandoned secret resolutions still running."""

        return (
            self.terminals.detached_environment_work() | self.resources.detached_environment_work()
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
        run_limit_accounting: RunLimitAccountingContext | None = None,
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
        self._auxiliary_invocation_policy = AuxiliaryInvocationPolicy(
            limits=limits, retry_policy=retry_policy, accounting=run_limit_accounting
        )
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

    def rebind_queued_interaction(self, invocation_context: InvocationContext) -> None:
        """Install the store-authenticated same-epoch context before the next round."""

        current = self._invocation_context
        if (
            current is None
            or type(invocation_context) is not InvocationContext
            or current.binding.session_id != invocation_context.binding.session_id
            or current.binding.session_instance_id != invocation_context.binding.session_instance_id
            or current.binding.run_epoch != invocation_context.binding.run_epoch
            or current.profile is not invocation_context.profile
            or current.tool_capability_ceiling != invocation_context.tool_capability_ceiling
            or current.binding.interaction_id == invocation_context.binding.interaction_id
        ):
            raise ValueError("Tool-round queued handoff lost frozen invocation authority.")
        self._invocation_context = invocation_context

    @timed_tool_round
    async def run(
        self,
        *,
        messages: list[Message],
        tool_calls: list[runtime_records.ToolCallRequest],
        tool_round_identity: ToolRoundIdentity,
        model_step: int | None = None,
    ) -> AsyncGenerator[Event, None]:
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
        (
            _source_checkpoint,
            source_pending_round,
        ) = await pending_round_reader.load_pending_tool_round(
            executor._session_store,
            session.id,
            redactor=executor._secret_redactor,
            consume_on_rejection=True,
            runtime_session=session,
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
        ) = await executor.admission.resolve_targeted_calls(
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
            stream = self.close_after_interrupt(
                exc,
                messages=messages,
                tool_calls=tool_calls,
                tool_outcomes=tool_outcomes,
                tool_round_identity=tool_round_identity,
            )
            async with aclosing(stream) as owned_stream:
                async for event in owned_stream:
                    yield event
            raise

        planned_round: pending_rounds.PendingToolRound | None = None
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
                stream = self.close_after_interrupt(
                    exc,
                    messages=messages,
                    tool_calls=tool_calls,
                    tool_outcomes=tool_outcomes,
                    tool_round_identity=tool_round_identity,
                )
                async with aclosing(stream) as owned_stream:
                    async for event in owned_stream:
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
                    approval, approval_events = await executor.admission.pause_for_approval(
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
                stream = self.close_after_interrupt(
                    exc,
                    messages=messages,
                    tool_calls=tool_calls,
                    tool_outcomes=tool_outcomes,
                    tool_round_identity=tool_round_identity,
                    clear_pending_approval=True,
                )
                async with aclosing(stream) as owned_stream:
                    async for event in owned_stream:
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
        policy_output_secret_resolution_scope = (
            pending_approval_reader.tool_round_secret_resolution_scope(planned_round)
        )
        defer_round_terminals = (
            len(tool_calls) > 1 and policy_output_secret_resolution_scope != "static"
        ) or any(
            registered is not None
            and (registered.workspace_mutation or registered.effect is ToolEffect.EXTERNAL)
            for registered in (
                self._registered_agent.executable_tool(tool_call.name) for tool_call in tool_calls
            )
        )
        round_owner = DurableToolRound.for_execution(
            session=session,
            tool_round_identity=tool_round_identity,
            tool_calls=tool_calls,
            outcomes=tool_outcomes,
            session_store=executor._session_store,
            event_writer=executor._event_writer,
            registered_agent=self._registered_agent,
            registered_environment=self._registered_environment,
            task_id=self._task_id,
            execution_profile=self._execution_profile,
            invocation_context=self._invocation_context,
            redactor=(
                _redactor_for_tool_calls(
                    executor._secret_redactor,
                    registered_agent=self._registered_agent,
                    tool_calls=tool_calls,
                )
                if defer_round_terminals
                else executor._secret_redactor
            ),
            tool_exposure=tool_exposure,
            publication_governor=executor.terminals.governor,
            clock=executor._clock,
            emit_result=executor.terminals.publish_result,
            emit_terminal=executor.terminals.emit_staged,
            defer_terminals=defer_round_terminals,
            terminal_payload_limits=(
                await _tool_terminal_payload_limits(
                    self._registered_agent,
                    tool_calls,
                    publication_governor=executor.terminals.governor,
                    runtime_hooks=(
                        executor.hooks.registrations
                        if self._invocation_context is None
                        else self._invocation_context.runtime_hooks
                    ),
                )
                if defer_round_terminals
                else None
            ),
        )
        segments = self._tool_round_segments(tool_calls)

        # Static rounds have no late secret-resolution capability and can use
        # each invocation's sealed projection directly. Dynamic multi-call
        # rounds stay quarantined until the coordinator has merged every sealed
        # scope, then publish through that cumulative redactor.
        publish_arguments_as_unavailable = len(tool_calls) > 1 and (
            policy_output_secret_resolution_scope != "static"
        )
        round_task = asyncio.current_task()
        round_cancellation_baseline = 0 if round_task is None else round_task.cancelling()
        try:
            await round_owner.admit()
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
                            round_owner.stage_terminal if round_owner.defers_terminals else None
                        ),
                        deferred_terminal_capture_recorder=(
                            round_owner.record_workspace_capture
                            if round_owner.defers_terminals
                            else None
                        ),
                        resolved_redactor_observer=(
                            round_owner.record_redactor if round_owner.defers_terminals else None
                        ),
                        publication_snapshot_observer=round_owner.record_publication_snapshot,
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
                            round_owner.stage_terminal if round_owner.defers_terminals else None
                        ),
                        deferred_terminal_capture_recorder=(
                            round_owner.record_workspace_capture
                            if round_owner.defers_terminals
                            else None
                        ),
                        resolved_redactor_observer=(
                            round_owner.record_redactor if round_owner.defers_terminals else None
                        ),
                        publication_snapshot_observer=round_owner.record_publication_snapshot,
                        publish_staged_terminals=(
                            round_owner.publish_before_limit
                            if round_owner.defers_terminals
                            else None
                        ),
                    )
                # Keep closure in the owning task: an extra cancellation
                # checkpoint here would disturb restored post-tool cancellation.
                async with aclosing(call_stream) as owned_stream:
                    async for event, outcome in owned_stream:
                        yield event
                        round_owner.observe_execution(event, outcome)
                if self.stopped_for_limit:
                    break
            if self.stopped_for_limit:
                round_owner.finish_dispatch()
                return
        except WorkspaceMutationSettlementError as settlement_failure:
            if round_owner.defers_terminals:
                round_owner.finish_dispatch()
                try:
                    async for event in round_owner.publish_completed_effects():
                        yield event
                        if event.type in tool_round_recovery._TOOL_ROUND_TERMINAL_EVENT_TYPES:
                            round_owner.observe_execution(event)
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
            round_owner.finish_dispatch()
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
            # Persist the bounded round's entire closure before exposing outcomes.
            # GeneratorExit at an outward yield is consumer abandonment, not a
            # failure of durable tool cleanup; the session owner must handle it.
            interruption_events: list[Event] = []
            try:
                async for event in round_owner.publish_before_interrupt():
                    interruption_events.append(event)
                stream = self.close_after_interrupt(
                    interrupt,
                    messages=messages,
                    tool_calls=tool_calls,
                    tool_outcomes=tool_outcomes,
                    tool_round_identity=tool_round_identity,
                )
                async with aclosing(stream) as owned_stream:
                    async for event in owned_stream:
                        interruption_events.append(event)
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
            for event in interruption_events:
                yield event
            if isinstance(interrupt, asyncio.CancelledError):
                invocation_secrets.sanitize_external_cancellation(interrupt)
                _restore_current_task_cancellation_requests(
                    minimum_requests=minimum_cancellation_requests,
                )
                raise interrupt from exception_cause(interrupt)
            raise
        except asyncio.CancelledError as exc:
            round_owner.finish_dispatch()
            await _receive_restored_post_tool_cancellation()
            current_task = asyncio.current_task()
            minimum_cancellation_requests = 0 if current_task is None else current_task.cancelling()
            # Persist the bounded round's entire closure before exposing outcomes.
            # GeneratorExit at an outward yield is consumer abandonment, not a
            # failure of durable tool cleanup; the session owner must handle it.
            interruption_events: list[Event] = []
            try:
                async for event in round_owner.publish_before_interrupt():
                    interruption_events.append(event)
                stream = self.close_after_interrupt(
                    exc,
                    messages=messages,
                    tool_calls=tool_calls,
                    tool_outcomes=tool_outcomes,
                    tool_round_identity=tool_round_identity,
                )
                async with aclosing(stream) as owned_stream:
                    async for event in owned_stream:
                        interruption_events.append(event)
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
            for event in interruption_events:
                yield event
            invocation_secrets.sanitize_external_cancellation(exc)
            _restore_current_task_cancellation_requests(
                minimum_requests=minimum_cancellation_requests,
            )
            raise
        except SessionInterruptedByRequest as exc:
            round_owner.finish_dispatch()
            interruption_events = [event async for event in round_owner.publish_before_interrupt()]
            stream = self.close_after_interrupt(
                exc,
                messages=messages,
                tool_calls=tool_calls,
                tool_outcomes=tool_outcomes,
                tool_round_identity=tool_round_identity,
            )
            async with aclosing(stream) as owned_stream:
                async for event in owned_stream:
                    interruption_events.append(event)
            for event in interruption_events:
                yield event
            raise
        except Exception:
            round_owner.finish_dispatch()
            raise
        except (KeyboardInterrupt, SystemExit, GeneratorExit) as interrupt:
            # An async-generator close cannot yield. If workspace settlement
            # still owns the stage, retain it for recovery without replacing
            # the original supervisory signal with a publication failure.
            try:
                async for _event in round_owner.publish_before_interrupt():
                    pass
            except Exception as publication_failure:
                raise interrupt from publication_failure
            raise

        async with aclosing(round_owner.publish(messages)) as publication:
            async for event in publication:
                yield event
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
    ) -> AsyncGenerator[Event, None]:
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
        stream = self._executor._close_interrupted_round(request)
        async with aclosing(stream) as owned_stream:
            async for event in owned_stream:
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
    ) -> AsyncGenerator[tuple[Event, runtime_records.ToolCallOutcome | None], None]:
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
            stream = self._executor.execute_tool_call(
                session=self._session,
                registered_agent=self._registered_agent,
                registered_environment=self._registered_environment,
                tool_call=tool_call,
                request_metadata=self._request_metadata,
                budget_limits=self._budget_limits,
                task_id=self._task_id,
                model_step=model_step,
                auxiliary_invocation_policy=self._auxiliary_invocation_policy,
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
            )
            async with aclosing(stream) as owned_stream:
                async for event, outcome in owned_stream:
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
    ) -> AsyncGenerator[tuple[Event, runtime_records.ToolCallOutcome | None], None]:
        semaphore = asyncio.Semaphore(self._executor._max_parallel_tool_calls)
        buffers: list[list[tuple[Event, runtime_records.ToolCallOutcome | None]]] = [
            [] for _ in tool_calls
        ]
        child_cancellations: list[asyncio.CancelledError | None] = [None] * len(tool_calls)

        async def execute_call(index: int, tool_call: runtime_records.ToolCallRequest) -> None:
            async with semaphore:
                await self._executor._session_control.raise_if_interrupted(self._session.id)
                try:
                    stream = self._executor.execute_tool_call(
                        session=self._session,
                        registered_agent=self._registered_agent,
                        registered_environment=self._registered_environment,
                        tool_call=tool_call,
                        request_metadata=self._request_metadata,
                        budget_limits=self._budget_limits,
                        task_id=self._task_id,
                        model_step=model_step,
                        auxiliary_invocation_policy=self._auxiliary_invocation_policy,
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
                    )
                    async with aclosing(stream) as owned_stream:
                        async for item in owned_stream:
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
                call=replace(tool_call, arguments={}, arguments_state="unavailable"),
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

    invocation_secrets.transfer_admission_refusals(
        target, [source for source, _tool_call_id in sources]
    )

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


def _taint_labels_for_source_tool(
    policy: ToolPolicy,
    tool_name: str,
    *,
    policy_result: ToolPolicyResult | None,
) -> set[str]:
    taint_policy = _taint_policy(policy)
    if taint_policy is None:
        return set()
    if policy_result is not None and policy_result.decision != ToolPolicyDecision.ALLOW:
        return set()
    return set(taint_policy.labels_for_source_tool(tool_name))


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
        "metadata": copy_durable_metadata(metadata),
    }
