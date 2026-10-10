"""Tool-round planning, scheduling, pause and closure ownership.

This module is deliberately below :class:`CayuApp`.  It owns one complete
tool-round lifecycle without importing or accepting the application facade.
Session-level limit terminalization and interrupted-round recovery remain
operations supplied by the independent session-finalization and pending-round
recovery components.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
from collections.abc import AsyncGenerator, AsyncIterator, Awaitable, Callable, Iterable, Mapping
from contextlib import aclosing
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
    consume_pending_task_cancellation,
    unexpected_child_cancellation_error,
)
from cayu._validation import (
    copy_durable_record,
    copy_json_value,
)
from cayu.approvals.tools import (
    PendingToolApproval,
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
)
from cayu.execution_profiles import (
    ExecutionProfileIdentity,
    event_with_execution_profile_authority,
)
from cayu.execution_units import (
    ModelAttemptIdentity,
    ToolRoundIdentity,
    copy_tool_round_identity,
)
from cayu.mcp.tools import McpToolAdapter, McpToolset
from cayu.messages import Message
from cayu.providers.retry_policy import RetryPolicy, copy_retry_policy
from cayu.runners._cleanup import (
    attach_runner_cancellation_failure,
    pop_runner_cancellation_failure,
    runner_cancellation_failure,
    transfer_runner_cancellation_failures,
)
from cayu.runtime import _approval_publication as approval_publication
from cayu.runtime import _approval_support as approval_support
from cayu.runtime import _invocation_secrets as invocation_secrets
from cayu.runtime import _runtime_records as runtime_records
from cayu.runtime import _tool_execution as tool_execution
from cayu.runtime import _tool_round_recovery as tool_round_recovery
from cayu.runtime import _transcript as transcript_helpers
from cayu.runtime._auxiliary_invocation import AuxiliaryInvocationPolicy
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
from cayu.runtime._event_writer import RuntimeEventWriter
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
from cayu.runtime._tool_invocation.admission import (
    ToolApprovalRequired,
    _require_matching_policy_round,
    _taint_labels_for_source_tool,
    _tool_dispatch_authority_is_current,
)
from cayu.runtime._tool_invocation.cancellation import (
    _contains_process_signal,
    _grouped_cancellation_evidence,
    _receive_restored_post_tool_cancellation,
    _transfer_cancellation_evidence,
)
from cayu.runtime._tool_invocation.context import (
    _environment_name,
    _workspace_id,
)
from cayu.runtime._tool_invocation.hooks import (
    _redact_event_for_invocation,
)
from cayu.runtime._tool_invocation.invocation import ToolInvocation
from cayu.runtime._tool_invocation.terminal import (
    DeferredTerminalCaptureRecorder,
    DeferredTerminalStager,
)
from cayu.runtime._tool_round_staging import (
    CheckpointTransform,
    _event_with_tool_round_authority,
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
    SessionStore,
    runtime_publication_checkpoint_value_digest,
)
from cayu.sessions.mcp_manifest_history import (
    _MCP_MANIFEST_BASELINE_MAX_TOOLS,
    McpManifestBaseline,
    McpManifestBaselineLoadResult,
    McpManifestHistoryConflict,
    McpManifestPublicationResult,
    _mcp_authoritative_manifest_hash,
    _mcp_manifest_session_ref,
    _McpManifestBaselineEvidenceInvalid,
)
from cayu.sessions.records import Session, SessionStatus
from cayu.tools import _argument_publication as tool_argument_publication
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
    ToolEffect,
    ToolResult,
)
from cayu.tools.exposure import (
    ResolvedToolExposureAuthority,
    validate_resolved_tool_exposure_authority,
)
from cayu.tools.policy import (
    ToolPolicyDecision,
    ToolPolicyResult,
)
from cayu.vaults.redaction import SecretRedactor

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


class UserInputRequired(Exception):
    """Internal control signal for a durably checkpointed user-input pause."""

    def __init__(self, pending: PendingUserInput) -> None:
        super().__init__(f"Tool call awaits user input: {pending.tool_name}")
        self.pending = copy_pending_user_input(pending)


class ToolRoundExecutor:
    """Plan, schedule and close ordinary tool rounds.

    The executor composes a shared single-call invocation owner and retains
    round policy planning, input checkpoints, concurrency segmentation and
    durable round coordination. Recovery drives the same invocation owner.
    """

    def __init__(
        self,
        *,
        invocation: ToolInvocation,
        session_store: SessionStore,
        event_writer: RuntimeEventWriter,
        session_control: SessionControl[SessionUsageTracker],
        mcp_manifest_policy: McpManifestPolicy | None,
        secret_redactor: SecretRedactor,
        max_parallel_tool_calls: int,
        clock: Callable[[], datetime],
        checkpoint_transform: CheckpointTransformFactory,
        apply_limit_evaluation: LimitEventStream,
        close_interrupted_round: InterruptedRoundEventStream,
    ) -> None:
        self.invocation = invocation
        self._session_store = session_store
        self._event_writer = event_writer
        self._session_control = session_control
        self._mcp_manifest_policy = mcp_manifest_policy
        self._secret_redactor = secret_redactor
        self._max_parallel_tool_calls = max_parallel_tool_calls
        self._clock = clock
        self._checkpoint_transform = checkpoint_transform
        self._apply_limit_evaluation = apply_limit_evaluation
        self._close_interrupted_round = close_interrupted_round

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
            await self.invocation.admission.prior_taint_labels(
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

            policy_result = await self.invocation.admission.authorize(
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
        ) = await executor.invocation.admission.resolve_targeted_calls(
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
                    planned_round = (
                        await executor.invocation.admission.checkpoint_tool_round_policy_plan(
                            session=session,
                            registered_agent=self._registered_agent,
                            tool_calls=tool_calls,
                            policy_outcomes=policy_plan.outcomes,
                            active_taint_by_id=policy_plan.active_taint_labels,
                            tool_round_identity=tool_round_identity,
                        )
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
                    (
                        approval,
                        approval_events,
                    ) = await executor.invocation.admission.pause_for_approval(
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
            publication_governor=executor.invocation.terminals.governor,
            clock=executor._clock,
            emit_result=executor.invocation.terminals.publish_result,
            emit_terminal=executor.invocation.terminals.emit_staged,
            defer_terminals=defer_round_terminals,
            terminal_payload_limits=(
                await _tool_terminal_payload_limits(
                    self._registered_agent,
                    tool_calls,
                    publication_governor=executor.invocation.terminals.governor,
                    runtime_hooks=(
                        executor.invocation.hooks.registrations
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
            stream = self._executor.invocation.execute(
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
                    stream = self._executor.invocation.execute(
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
