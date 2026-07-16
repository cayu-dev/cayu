"""Tool-round policy, execution, pause, hook, and closure ownership.

This module is deliberately below :class:`CayuApp`.  It owns one complete
tool-round lifecycle without importing or accepting the application facade.
Session-level limit terminalization and interrupted-round recovery remain
orchestration boundaries supplied as narrow callbacks until their owning
runtime modules are extracted.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Callable, Iterable, Mapping
from copy import deepcopy
from dataclasses import dataclass, replace
from datetime import datetime, timedelta
from types import MappingProxyType
from typing import Any, cast
from uuid import uuid4

from cayu._validation import (
    copy_json_object,
    copy_json_value,
    require_clean_nonblank,
    require_nonblank,
    require_unicode_scalar_text,
)
from cayu.core.agents import AgentSpec
from cayu.core.events import Event, EventType
from cayu.core.messages import Message
from cayu.core.thinking import ThinkingConfig
from cayu.core.tools import (
    _TOOL_POLICY_DENIAL_SOURCE,
    ToolContext,
    ToolEffect,
    ToolResult,
    _bound_policy_denial_result,
    _bound_policy_denial_text,
)
from cayu.mcp import McpToolAdapter, McpToolset
from cayu.proxies import (
    CredentialProxy,
    ProxyAuthorizationResult,
    copy_proxy_authorization_result,
)
from cayu.runners import RunnerCancelledError
from cayu.runtime import _approval_support as approval_support
from cayu.runtime import _runtime_records as runtime_records
from cayu.runtime import _tool_execution as tool_execution
from cayu.runtime import _tool_results as tool_results
from cayu.runtime import _tool_round_recovery as tool_round_recovery
from cayu.runtime import _transcript as transcript_helpers
from cayu.runtime._event_writer import RuntimeEventWriter
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
from cayu.runtime.approvals import PendingToolApproval, copy_pending_tool_approval
from cayu.runtime.budgets import BudgetLimit, copy_request_budget_limits
from cayu.runtime.hooks import (
    AfterToolCallDecision,
    BeforeToolCallDecision,
    BeforeToolCallHookContext,
    RuntimeHook,
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
from cayu.runtime.retry_policy import RetryPolicy, copy_retry_policy
from cayu.runtime.sessions import EventQuery, EventRecord, Session, SessionStore
from cayu.runtime.stop_policy import RunLimits, copy_run_limits
from cayu.runtime.structured_output import (
    StructuredOutputSpec,
    copy_structured_output_spec,
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
from cayu.runtime.user_input import (
    PENDING_USER_INPUT_CHECKPOINT_KEY,
    PendingUserInput,
    copy_pending_user_input,
    pending_user_input_from_checkpoint,
)
from cayu.vaults import (
    ResolvedSecret,
    SecretRedactor,
    SecretRef,
    copy_resolved_secret,
    copy_secret_ref,
)

CheckpointTransform = Callable[
    [Session, dict[str, Any] | None],
    dict[str, Any],
]
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
    tool_round_id: str | None
    run_started_at: float
    turn_usage_tracker: SessionUsageTracker | None
    active_run: ActiveSessionRun[SessionUsageTracker] | None


@dataclass(frozen=True)
class InterruptedToolRoundRequest:
    session: Session
    registered_agent: runtime_records.RegisteredAgentState
    registered_environment: runtime_records.RegisteredEnvironment | None
    messages: list[Message]
    tool_calls: list[runtime_records.ToolCallRequest]
    tool_outcomes: list[runtime_records.ToolCallOutcome]
    tool_round_id: str | None
    cancellation_artifacts: list[dict[str, Any]] | None
    cancellation_artifacts_by_id: dict[str, list[dict[str, Any]]] | None


LimitEventStream = Callable[[ToolRoundLimitRequest], AsyncIterator[Event]]
InterruptedRoundEventStream = Callable[[InterruptedToolRoundRequest], AsyncIterator[Event]]


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
        runtime_hooks: tuple[RuntimeHook, ...],
        mcp_manifest_policy: McpManifestPolicy | None,
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
        self._secret_redactor = secret_redactor
        self._tool_timeout_seconds = tool_timeout_seconds
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
        )

    async def policy_plan(
        self,
        *,
        session: Session,
        registered_agent: runtime_records.RegisteredAgentState,
        registered_environment: runtime_records.RegisteredEnvironment | None,
        tool_calls: list[runtime_records.ToolCallRequest],
        request_metadata: dict[str, Any],
    ) -> runtime_records.ToolRoundPolicyPlan:
        policy_outcomes: list[runtime_records.ToolCallPolicyOutcome] = []
        approval_policy_result: ToolPolicyResult | None = None
        approval_tool_call: runtime_records.ToolCallRequest | None = None
        taint_labels = await self.prior_taint_labels_for_policy(
            session_id=session.id,
            policy=registered_agent.tool_policy,
            request_metadata=request_metadata,
        )
        active_taint_labels: dict[str, frozenset[str]] = {}
        for tool_call in tool_calls:
            active_taint_labels[tool_call.id] = frozenset(taint_labels)
            if tool_call.name not in registered_agent.tools:
                policy_outcomes.append(
                    runtime_records.ToolCallPolicyOutcome(call=tool_call, result=None)
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
                runtime_records.ToolCallPolicyOutcome(call=tool_call, result=policy_result)
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
    ) -> tuple[PendingToolApproval, Event]:
        checkpoint = await self._session_store.load_checkpoint(session.id)
        checkpoint = {} if checkpoint is None else copy_json_value(checkpoint, "checkpoint")
        if approval_support.pending_approval_from_checkpoint(checkpoint) is not None:
            raise RuntimeError("Session already has a pending tool approval.")
        checkpoint.pop(tool_round_recovery.PENDING_TOOL_ROUND_CHECKPOINT_KEY, None)

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
            tool_call_id=tool_call.id,
            tool_name=tool_call.name,
            arguments=copy_json_value(tool_call.arguments, "arguments"),
            agent_name=registered_agent.spec.name,
            environment_name=_environment_name(registered_environment),
            workspace_id=_workspace_id(registered_environment),
            task_id=task_id,
            reason=policy_result.reason,
            metadata=copy_json_value(policy_result.metadata, "metadata"),
            tool_calls=approval_support.pending_tool_call_approvals(
                tool_calls=tool_calls,
                policy_outcomes=policy_outcomes,
                active_taint_by_id=active_taint_by_id,
                redactor=self._secret_redactor,
            ),
            structured_output=copy_structured_output_spec(structured_output),
            thinking=thinking,
            max_steps=max_steps,
            limits=copy_run_limits(limits) if limits is not None else None,
            budget_limits=(
                copy_request_budget_limits(budget_limits) if budget_limits is not None else None
            ),
            retry_policy=copy_retry_policy(retry_policy) if retry_policy is not None else None,
            expires_at=expires_at,
        )
        checkpoint[approval_support.PENDING_TOOL_APPROVAL_CHECKPOINT_KEY] = approval.model_dump(
            mode="json"
        )
        await self._session_store.transform_checkpoint(
            session.id,
            self._checkpoint_transform(checkpoint),
        )
        return (
            approval,
            Event(
                type=EventType.SESSION_CHECKPOINTED,
                session_id=session.id,
                agent_name=registered_agent.spec.name,
                environment_name=_environment_name(registered_environment),
                payload={
                    "checkpoint": approval_support.PENDING_TOOL_APPROVAL_CHECKPOINT_KEY,
                    "approval_id": approval.approval_id,
                    "tool_call_id": approval.tool_call_id,
                },
            ),
        )

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
    ) -> tuple[PendingUserInput, Event]:
        checkpoint = await self._session_store.load_checkpoint(session.id)
        checkpoint = {} if checkpoint is None else copy_json_value(checkpoint, "checkpoint")
        if approval_support.pending_approval_from_checkpoint(checkpoint) is not None:
            raise RuntimeError("Session already has a pending tool approval.")
        if pending_user_input_from_checkpoint(checkpoint) is not None:
            raise RuntimeError("Session already has a pending user input.")
        checkpoint.pop(tool_round_recovery.PENDING_TOOL_ROUND_CHECKPOINT_KEY, None)

        pending = PendingUserInput(
            input_id=str(uuid4()),
            tool_call_id=tool_call.id,
            tool_name=tool_call.name,
            question=question,
            options=list(options),
            arguments=copy_json_value(tool_call.arguments, "arguments"),
            agent_name=registered_agent.spec.name,
            environment_name=_environment_name(registered_environment),
            workspace_id=_workspace_id(registered_environment),
            task_id=task_id,
            tool_calls=approval_support.pending_tool_call_approvals(
                tool_calls=tool_calls,
                policy_outcomes=policy_outcomes,
                active_taint_by_id=active_taint_by_id,
                redactor=self._secret_redactor,
            ),
            structured_output=copy_structured_output_spec(structured_output),
            thinking=thinking,
            max_steps=max_steps,
            limits=copy_run_limits(limits) if limits is not None else None,
            budget_limits=(
                copy_request_budget_limits(budget_limits) if budget_limits is not None else None
            ),
            retry_policy=copy_retry_policy(retry_policy) if retry_policy is not None else None,
        )
        checkpoint[PENDING_USER_INPUT_CHECKPOINT_KEY] = pending.model_dump(mode="json")
        await self._session_store.transform_checkpoint(
            session.id,
            self._checkpoint_transform(checkpoint),
        )
        return (
            pending,
            Event(
                type=EventType.SESSION_CHECKPOINTED,
                session_id=session.id,
                agent_name=registered_agent.spec.name,
                environment_name=_environment_name(registered_environment),
                payload={
                    "checkpoint": PENDING_USER_INPUT_CHECKPOINT_KEY,
                    "input_id": pending.input_id,
                    "tool_call_id": pending.tool_call_id,
                },
            ),
        )

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
    ) -> tuple[dict[str, Any], tool_round_recovery.PendingToolRound]:
        checkpoint = await self._session_store.load_checkpoint(session.id)
        return tool_round_recovery.checkpoint_with_pending_tool_round(
            checkpoint,
            agent_name=registered_agent.spec.name,
            environment_name=_environment_name(registered_environment),
            task_id=task_id,
            tool_calls=tool_calls,
            policy_outcomes=policy_outcomes,
            structured_output=structured_output,
            redactor=self._secret_redactor,
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
        pending_approval = approval_support.pending_approval_from_checkpoint(copied_checkpoint)
        if pending_approval is None or pending_approval.tool_call_id not in expected_ids:
            return
        copied_checkpoint.pop(approval_support.PENDING_TOOL_APPROVAL_CHECKPOINT_KEY, None)
        await self._session_store.transform_checkpoint(
            session_id,
            self._checkpoint_transform(copied_checkpoint),
        )

    async def execute_tool_call(
        self,
        *,
        session: Session,
        registered_agent: runtime_records.RegisteredAgentState,
        registered_environment: runtime_records.RegisteredEnvironment | None,
        tool_call: runtime_records.ToolCallRequest,
        request_metadata: dict[str, Any],
        task_id: str | None,
        check_policy: bool = True,
        emit_started: bool = True,
        policy_result: ToolPolicyResult | None = None,
        approval_id: str | None = None,
        tool_round_id: str | None = None,
        input_id: str | None = None,
        taint_labels: frozenset[str] | None = None,
    ) -> AsyncIterator[tuple[Event, runtime_records.ToolCallOutcome | None]]:
        environment_name = _environment_name(registered_environment)
        started_event: Event | None = None
        registered_tool = registered_agent.tools.get(tool_call.name)
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
                "arguments": deepcopy(tool_call.arguments),
            }
            if registered_tool is not None:
                payload["effect"] = registered_tool.effect.value
            if tool_round_id is not None:
                payload["tool_round_id"] = tool_round_id
            if approval_id is not None:
                payload["approval_id"] = approval_id
            if input_id is not None:
                payload["input_id"] = input_id
            started_event = await self._event_writer.emit(
                Event(
                    type=EventType.TOOL_CALL_STARTED,
                    session_id=session.id,
                    agent_name=registered_agent.spec.name,
                    environment_name=environment_name,
                    tool_name=tool_call.name,
                    payload=payload,
                )
            )
            yield started_event, None

        if registered_tool is None:
            result = ToolResult(
                content=f"Tool not registered: {tool_call.name}",
                is_error=True,
            )
            payload = {
                "tool_call_id": tool_call.id,
                "idempotency_key": idempotency_key,
                "result": result.model_dump(),
            }
            if tool_round_id is not None:
                payload["tool_round_id"] = tool_round_id
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
                reason = tool_execution.policy_denial_reason(resolved_policy_result)
                result = tool_execution.blocked_tool_result(resolved_policy_result, reason=reason)
                payload = {
                    "tool_call_id": tool_call.id,
                    "idempotency_key": idempotency_key,
                    **policy_denial_payload_fields(
                        tool_name=tool_call.name,
                        denied_by=_TOOL_POLICY_DENIAL_SOURCE,
                        decision=resolved_policy_result.decision.value,
                        reason=reason,
                        metadata=resolved_policy_result.metadata,
                    ),
                    "result": result.model_dump(),
                }
                if tool_round_id is not None:
                    payload["tool_round_id"] = tool_round_id
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
                ):
                    yield event
                return
            if resolved_policy_result.decision == ToolPolicyDecision.REQUIRE_APPROVAL:
                approval, checkpoint_event = await self.checkpoint_pending_tool_approval(
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
                )
                yield (await self._event_writer.emit(checkpoint_event), None)
                yield (
                    await self._event_writer.emit(
                        Event(
                            type=EventType.TOOL_CALL_APPROVAL_REQUESTED,
                            session_id=session.id,
                            agent_name=registered_agent.spec.name,
                            environment_name=environment_name,
                            tool_name=tool_call.name,
                            payload={"approval": approval.model_dump(mode="json")},
                        )
                    ),
                    None,
                )
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
            payload={"tool_call_id": tool_call.id},
        )
        before_resolution = _BeforeToolCallResolution(arguments=deepcopy(tool_call.arguments))
        async for hook_event in self._run_before_tool_call_hooks(
            session=session,
            registered_agent=registered_agent,
            registered_environment=registered_environment,
            tool_call=tool_call,
            anchor_event=anchor_event,
            task_id=task_id,
            resolution=before_resolution,
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
            async for event in self._emit_terminal_tool_result(
                session=session,
                registered_agent=registered_agent,
                registered_environment=registered_environment,
                tool_call=effective_tool_call,
                event_type=EventType.TOOL_CALL_BLOCKED,
                result=ToolResult(content=before_resolution.block_reason, is_error=True),
                extra_payload={
                    "reason": before_resolution.block_reason,
                    "blocked_by": "before_tool_call_hook",
                    "idempotency_key": idempotency_key,
                    **effective_arguments_payload,
                },
                task_id=task_id,
                tool_round_id=tool_round_id,
                approval_id=approval_id,
                input_id=input_id,
                allow_modification=False,
            ):
                yield event
            return
        if before_resolution.short_circuit_result is not None:
            short_result = before_resolution.short_circuit_result
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
                tool_round_id=tool_round_id,
                approval_id=approval_id,
                input_id=input_id,
                allow_modification=True,
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
                if reauthorization.decision == ToolPolicyDecision.DENY:
                    reason = tool_execution.policy_denial_reason(reauthorization)
                else:
                    reason = (
                        reauthorization.reason
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
                            metadata=reauthorization.metadata,
                        ),
                        "blocked_by": "tool_policy_reauthorization",
                        "idempotency_key": idempotency_key,
                    },
                    task_id=task_id,
                    tool_round_id=tool_round_id,
                    approval_id=approval_id,
                    input_id=input_id,
                    allow_modification=False,
                ):
                    yield event
                return

        resolved_proxy_secrets: list[ResolvedSecret] = []
        proxy_authorizations: list[_ProxyAuthorizationRecord] = []
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
        tool_context = ToolContext(
            session_id=session.id,
            agent_name=registered_agent.spec.name,
            environment_name=environment_name,
            causal_budget_id=session.causal_budget_id,
            workspace_id=_workspace_id(registered_environment),
            artifact_store_id=_artifact_store_id(registered_environment),
            idempotency_key=idempotency_key,
            workspace=_workspace(registered_environment),
            artifact_store=_artifact_store(registered_environment),
            runner=_runner(registered_environment),
            vault=_vault(registered_environment),
            proxy=_proxy(
                registered_environment,
                on_resolve=resolved_proxy_secrets.append,
                on_authorize=proxy_authorizations.append,
            ),
            knowledge_store=_knowledge_store(registered_environment),
            mcp_servers=_mcp_servers(registered_environment),
            metadata=ctx_metadata,
        )
        try:
            result = await tool_execution.run_tool(
                tool=registered_tool.tool,
                ctx=tool_context,
                arguments=effective_tool_call.arguments,
                timeout_seconds=self._tool_timeout_seconds,
            )
        except asyncio.CancelledError:
            if proxy_authorizations and await self._session_control.interrupt_requested(session.id):
                clear_current_task_cancellation()
                redactor = _redactor_with_resolved_secrets(
                    self._secret_redactor,
                    resolved_proxy_secrets,
                )
                async for event in self._emit_proxy_authorization_events(
                    session=session,
                    registered_agent=registered_agent,
                    registered_environment=registered_environment,
                    tool_call=tool_call,
                    records=proxy_authorizations,
                    tool_round_id=tool_round_id,
                    approval_id=approval_id,
                    input_id=input_id,
                    idempotency_key=idempotency_key,
                    redactor=redactor,
                ):
                    yield event, None
            raise
        redactor = _redactor_with_resolved_secrets(
            self._secret_redactor,
            resolved_proxy_secrets,
        )
        async for event in self._emit_proxy_authorization_events(
            session=session,
            registered_agent=registered_agent,
            registered_environment=registered_environment,
            tool_call=tool_call,
            records=proxy_authorizations,
            tool_round_id=tool_round_id,
            approval_id=approval_id,
            input_id=input_id,
            idempotency_key=idempotency_key,
            redactor=redactor,
        ):
            yield event, None
        current_task = asyncio.current_task()
        tool_swallowed_cancellation = current_task is not None and current_task.cancelling() > 0
        if tool_swallowed_cancellation and await self._session_control.is_interrupting(session.id):
            raise SessionInterruptedByRequest(session.id)
        policy_denial = tool_context._policy_denial_for(registered_tool.tool)
        if policy_denial is not None:
            async for event in self._emit_terminal_tool_result(
                session=session,
                registered_agent=registered_agent,
                registered_environment=registered_environment,
                tool_call=effective_tool_call,
                event_type=EventType.TOOL_CALL_BLOCKED,
                result=policy_denial.result,
                extra_payload={
                    "idempotency_key": idempotency_key,
                    **policy_denial_payload_fields(
                        tool_name=effective_tool_call.name,
                        denied_by=policy_denial.denied_by,
                        decision=policy_denial.decision,
                        reason=policy_denial.reason,
                        metadata={},
                    ),
                },
                task_id=task_id,
                tool_round_id=tool_round_id,
                approval_id=approval_id,
                input_id=input_id,
                allow_modification=False,
                redactor=redactor,
            ):
                yield event
            return
        event_type = (
            EventType.TOOL_CALL_FAILED if result.is_error else EventType.TOOL_CALL_COMPLETED
        )
        payload = {
            "tool_call_id": tool_call.id,
            "idempotency_key": idempotency_key,
            "result": result.model_dump(),
            **effective_arguments_payload,
        }
        if tool_round_id is not None:
            payload["tool_round_id"] = tool_round_id
        if approval_id is not None:
            payload["approval_id"] = approval_id
        if input_id is not None:
            payload["input_id"] = input_id
        async for event in self.emit_tool_call_result_with_hooks(
            event=Event(
                type=event_type,
                session_id=session.id,
                agent_name=registered_agent.spec.name,
                environment_name=environment_name,
                tool_name=tool_call.name,
                payload=payload,
            ),
            session=session,
            registered_agent=registered_agent,
            registered_environment=registered_environment,
            tool_call=effective_tool_call,
            result=result,
            task_id=task_id,
            redactor=redactor,
            allow_modification=True,
        ):
            yield event
        if await self._session_control.is_interrupting(session.id):
            raise SessionInterruptedByRequest(session.id)

    async def emit_mcp_manifest_checks(
        self,
        *,
        session: Session,
        registered_agent: runtime_records.RegisteredAgentState,
        environment_name: str | None,
    ) -> AsyncIterator[Event]:
        seen_toolsets: set[int] = set()
        prior_records = await query_all_event_records(
            self._session_store,
            EventQuery(
                event_type=EventType.MCP_MANIFEST_CHECKED,
                environment_name=environment_name,
            ),
        )
        toolsets = _mcp_toolsets_for_agent(registered_agent)
        current_server_counts = _mcp_current_server_counts(toolsets)
        prior_server_counts = _mcp_prior_server_counts(
            prior_records,
            environment_name=environment_name,
        )
        checks: list[tuple[dict[str, Any], McpManifestPolicyDecision | None]] = []
        for toolset in toolsets:
            toolset_key = id(toolset)
            if toolset_key in seen_toolsets:
                continue
            seen_toolsets.add(toolset_key)
            previous = _latest_mcp_manifest_event(
                prior_records,
                manifest_identity=toolset.manifest_identity,
                environment_name=environment_name,
            )
            if (
                previous is None
                and current_server_counts.get(toolset.server.name) == 1
                and prior_server_counts.get(toolset.server.name) == 1
            ):
                previous = _latest_mcp_manifest_event_for_server(
                    prior_records,
                    server_name=toolset.server.name,
                    environment_name=environment_name,
                )
            status, previous_payload, diff = _mcp_manifest_status(
                toolset=toolset,
                previous=previous,
            )
            payload: dict[str, Any] = {
                "server_name": toolset.server.name,
                "manifest_identity": toolset.manifest_identity,
                "manifest_hash": toolset.manifest_hash,
                "server_hash": toolset.manifest_server_hash,
                "status": status,
                "tool_count": len(toolset.definitions),
                "tools": copy_json_value(list(toolset.manifest_tools), "tools"),
                "server": {
                    "protocol_version": toolset.initialize_result.protocol_version,
                    "server_name": toolset.initialize_result.server_name,
                    "server_version": toolset.initialize_result.server_version,
                },
                "previous": previous_payload,
                "diff": diff,
            }
            decision = None
            if self._mcp_manifest_policy is not None:
                decision = self._mcp_manifest_policy.decide(status=status, diff=diff)
                payload["policy"] = mcp_manifest_policy_payload(decision)
            checks.append((payload, decision))

        blocked_checks = [
            (payload, decision)
            for payload, decision in checks
            if decision is not None and decision.action == McpManifestPolicyAction.BLOCK
        ]
        if blocked_checks:
            for payload, _ in blocked_checks:
                yield await self._event_writer.emit(
                    Event(
                        type=EventType.MCP_MANIFEST_BLOCKED,
                        session_id=session.id,
                        agent_name=registered_agent.spec.name,
                        environment_name=environment_name,
                        payload=copy_json_value(payload, "payload"),
                    )
                )
            reasons = "; ".join(decision.reason for _, decision in blocked_checks)
            raise McpManifestPolicyError(reasons)

        for payload, _ in checks:
            yield await self._event_writer.emit(
                Event(
                    type=EventType.MCP_MANIFEST_CHECKED,
                    session_id=session.id,
                    agent_name=registered_agent.spec.name,
                    environment_name=environment_name,
                    payload=payload,
                )
            )

    async def _emit_proxy_authorization_events(
        self,
        *,
        session: Session,
        registered_agent: runtime_records.RegisteredAgentState,
        registered_environment: runtime_records.RegisteredEnvironment | None,
        tool_call: runtime_records.ToolCallRequest,
        records: list[_ProxyAuthorizationRecord],
        tool_round_id: str | None,
        approval_id: str | None,
        input_id: str | None,
        redactor: SecretRedactor,
        idempotency_key: str | None = None,
    ) -> AsyncIterator[Event]:
        for record in records:
            payload: dict[str, Any] = {
                "tool_call_id": tool_call.id,
                "destination": record.destination,
                "credential": None if record.credential is None else record.credential.name,
                "action": record.action,
                "metadata": copy_json_value(record.metadata, "metadata"),
                "allowed": record.result.allowed,
                "reason": record.result.reason,
                "result_metadata": copy_json_value(record.result.metadata, "result_metadata"),
            }
            if idempotency_key is not None:
                payload["idempotency_key"] = idempotency_key
            if tool_round_id is not None:
                payload["tool_round_id"] = tool_round_id
            if approval_id is not None:
                payload["approval_id"] = approval_id
            if input_id is not None:
                payload["input_id"] = input_id
            if redactor.has_values:
                redacted_payload = redactor.redact_json(payload)
                if type(redacted_payload) is not dict:
                    raise AssertionError(
                        "Proxy authorization redaction returned non-object payload."
                    )
                payload = redacted_payload
            yield await self._event_writer.emit(
                Event(
                    type=EventType.CREDENTIAL_PROXY_CHECKED,
                    session_id=session.id,
                    agent_name=registered_agent.spec.name,
                    environment_name=_environment_name(registered_environment),
                    tool_name=tool_call.name,
                    payload=payload,
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
        tool_round_id: str | None,
        approval_id: str | None,
        input_id: str | None,
        allow_modification: bool,
        redactor: SecretRedactor | None = None,
    ) -> AsyncIterator[tuple[Event, runtime_records.ToolCallOutcome | None]]:
        payload: dict[str, Any] = {
            "tool_call_id": tool_call.id,
            **extra_payload,
            "result": result.model_dump(),
        }
        if tool_round_id is not None:
            payload["tool_round_id"] = tool_round_id
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
            redactor=redactor,
            allow_modification=allow_modification,
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
    ) -> AsyncIterator[Event]:
        for hooks, scope in (
            (self._runtime_hooks, "app"),
            (registered_agent.runtime_hooks, "agent"),
        ):
            for hook in hooks:
                if not _runtime_hook_supports_phase(
                    hook=hook,
                    phase=RuntimeHookPhase.BEFORE_TOOL_CALL,
                ):
                    continue
                hook_name = require_clean_nonblank(hook.name, "runtime_hook.name")
                yield await self._event_writer.emit(
                    _runtime_hook_event(
                        event_type=EventType.HOOK_STARTED,
                        hook_name=hook_name,
                        scope=scope,
                        phase=RuntimeHookPhase.BEFORE_TOOL_CALL,
                        session=session,
                        registered_agent=registered_agent,
                        registered_environment=registered_environment,
                        terminal_event=anchor_event,
                        payload={
                            "tool_name": tool_call.name,
                            "tool_call_id": tool_call.id,
                        },
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
                )
                try:
                    decision = await hook.before_tool_call(context)
                    stop = _resolve_before_tool_call_decision(decision, resolution)
                except Exception as exc:
                    yield await self._event_writer.emit(
                        _runtime_hook_event(
                            event_type=EventType.HOOK_FAILED,
                            hook_name=hook_name,
                            scope=scope,
                            phase=RuntimeHookPhase.BEFORE_TOOL_CALL,
                            session=session,
                            registered_agent=registered_agent,
                            registered_environment=registered_environment,
                            terminal_event=anchor_event,
                            payload={
                                "tool_name": tool_call.name,
                                "tool_call_id": tool_call.id,
                                "error": str(exc),
                                "error_type": type(exc).__name__,
                                "actions": context.actions,
                            },
                        )
                    )
                    continue
                yield await self._event_writer.emit(
                    _runtime_hook_event(
                        event_type=EventType.HOOK_COMPLETED,
                        hook_name=hook_name,
                        scope=scope,
                        phase=RuntimeHookPhase.BEFORE_TOOL_CALL,
                        session=session,
                        registered_agent=registered_agent,
                        registered_environment=registered_environment,
                        terminal_event=anchor_event,
                        payload={
                            "tool_name": tool_call.name,
                            "tool_call_id": tool_call.id,
                            "actions": context.actions,
                        },
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
        allow_modification: bool = False,
    ) -> AsyncIterator[tuple[Event, runtime_records.ToolCallOutcome | None]]:
        resolved_redactor = redactor if redactor is not None else self._secret_redactor
        event, result = _prepare_tool_result_event(
            event=event,
            result=result,
            redactor=resolved_redactor,
        )
        final_result = result
        async for hook_event, modified in self.run_tool_call_hooks(
            session=session,
            tool_event=event,
            registered_agent=registered_agent,
            registered_environment=registered_environment,
            tool_call=tool_call,
            result=final_result,
            task_id=task_id,
            redactor=resolved_redactor,
            allow_modification=allow_modification,
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
                redactor=resolved_redactor,
            )
        tool_event = await self._event_writer.emit(event)
        yield tool_event, runtime_records.ToolCallOutcome(call=tool_call, result=final_result)

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
        allow_modification: bool = False,
    ) -> AsyncIterator[tuple[Event, ToolResult | None]]:
        current_result = result
        for hooks, scope in (
            (self._runtime_hooks, "app"),
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
                hooks=hooks,
                scope=scope,
                redactor=redactor,
                allow_modification=allow_modification,
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
        hooks: tuple[RuntimeHook, ...],
        scope: str,
        redactor: SecretRedactor,
        allow_modification: bool = False,
    ) -> AsyncIterator[tuple[Event, ToolResult | None]]:
        current_result = result
        for hook in hooks:
            if not _runtime_hook_supports_phase(
                hook=hook,
                phase=RuntimeHookPhase.AFTER_TOOL_CALL,
            ):
                continue
            hook_name = require_clean_nonblank(hook.name, "runtime_hook.name")
            yield (
                await self._event_writer.emit(
                    _runtime_hook_event(
                        event_type=EventType.HOOK_STARTED,
                        hook_name=hook_name,
                        scope=scope,
                        phase=RuntimeHookPhase.AFTER_TOOL_CALL,
                        session=session,
                        registered_agent=registered_agent,
                        registered_environment=registered_environment,
                        terminal_event=tool_event,
                        payload={
                            "tool_name": tool_call.name,
                            "tool_call_id": tool_call.id,
                        },
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
                arguments=redactor.redact_json(tool_call.arguments),
                result=(
                    current_result
                    if _is_policy_denial_event(tool_event) and not allow_modification
                    else _redact_tool_result_for_event(
                        event=tool_event,
                        result=current_result,
                        redactor=redactor,
                    )
                ),
                task_id=task_id,
            )
            try:
                decision = await hook.after_tool_call(context)
                resolved = _resolve_after_tool_call_decision(decision)
                modified = resolved if allow_modification else None
            except Exception as exc:
                yield (
                    await self._event_writer.emit(
                        _runtime_hook_event(
                            event_type=EventType.HOOK_FAILED,
                            hook_name=hook_name,
                            scope=scope,
                            phase=RuntimeHookPhase.AFTER_TOOL_CALL,
                            session=session,
                            registered_agent=registered_agent,
                            registered_environment=registered_environment,
                            terminal_event=tool_event,
                            payload={
                                "tool_name": tool_call.name,
                                "tool_call_id": tool_call.id,
                                "error": str(exc),
                                "error_type": type(exc).__name__,
                                "actions": context.actions,
                            },
                        )
                    ),
                    None,
                )
                continue
            if modified is not None:
                current_result = modified
            yield (
                await self._event_writer.emit(
                    _runtime_hook_event(
                        event_type=EventType.HOOK_COMPLETED,
                        hook_name=hook_name,
                        scope=scope,
                        phase=RuntimeHookPhase.AFTER_TOOL_CALL,
                        session=session,
                        registered_agent=registered_agent,
                        registered_environment=registered_environment,
                        terminal_event=tool_event,
                        payload={
                            "tool_name": tool_call.name,
                            "tool_call_id": tool_call.id,
                            "actions": context.actions,
                        },
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
        self.stopped_for_limit = False

    async def run(
        self,
        *,
        messages: list[Message],
        tool_calls: list[runtime_records.ToolCallRequest],
        tool_round_id: str | None,
    ) -> AsyncIterator[Event]:
        self.stopped_for_limit = False
        executor = self._executor
        session = self._session
        tool_outcomes: list[runtime_records.ToolCallOutcome] = []
        try:
            await executor._session_control.raise_if_interrupted(session.id)
            policy_plan = await executor.policy_plan(
                session=session,
                registered_agent=self._registered_agent,
                registered_environment=self._registered_environment,
                tool_calls=tool_calls,
                request_metadata=self._request_metadata,
            )
            await executor._session_control.raise_if_interrupted(session.id)
        except (SessionInterruptedByRequest, asyncio.CancelledError) as exc:
            async for event in self.close_after_interrupt(
                exc,
                messages=messages,
                tool_calls=tool_calls,
                tool_outcomes=tool_outcomes,
                tool_round_id=tool_round_id,
            ):
                yield event
            raise

        limit_evaluation = await self._limit_gate.evaluate_limits(
            pending_tool_calls=len(tool_calls),
        )
        async for event in self._apply_limit_evaluation(
            limit_evaluation,
            messages=messages,
            tool_calls=tool_calls,
            tool_round_id=tool_round_id,
        ):
            yield event
        if limit_evaluation.decision is not None:
            self.stopped_for_limit = True
            return

        if policy_plan.pending_approval is not None:
            approval_plan = policy_plan.pending_approval
            try:
                approval, checkpoint_event = await executor.checkpoint_pending_tool_approval(
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
                )
                yield await executor._event_writer.emit(checkpoint_event)
                yield await executor._event_writer.emit(
                    Event(
                        type=EventType.TOOL_CALL_APPROVAL_REQUESTED,
                        session_id=session.id,
                        agent_name=self._registered_agent.spec.name,
                        environment_name=self._environment_name,
                        tool_name=approval.tool_name,
                        payload={"approval": approval.model_dump(mode="json")},
                    )
                )
            except (SessionInterruptedByRequest, asyncio.CancelledError) as exc:
                async for event in self.close_after_interrupt(
                    exc,
                    messages=messages,
                    tool_calls=tool_calls,
                    tool_outcomes=tool_outcomes,
                    tool_round_id=tool_round_id,
                    clear_pending_approval=True,
                ):
                    yield event
                raise
            raise ToolApprovalRequired(approval)

        policy_results_by_id = {outcome.call.id: outcome.result for outcome in policy_plan.outcomes}
        user_input_pause = _first_user_input_tool_call(
            self._registered_agent,
            tool_calls,
            policy_results_by_id,
        )
        if user_input_pause is not None:
            user_input_call, question, options = user_input_pause
            pending_input, checkpoint_event = await executor.checkpoint_pending_user_input(
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
            )
            yield await executor._event_writer.emit(checkpoint_event)
            yield await executor._event_writer.emit(
                Event(
                    type=EventType.SESSION_AWAITING_USER_INPUT,
                    session_id=session.id,
                    agent_name=self._registered_agent.spec.name,
                    environment_name=self._environment_name,
                    tool_name=user_input_call.name,
                    payload={
                        "input_id": pending_input.input_id,
                        "tool_call_id": pending_input.tool_call_id,
                        "question": pending_input.question,
                        "options": list(pending_input.options),
                    },
                )
            )
            raise UserInputRequired(pending_input)

        segments = self._tool_round_segments(tool_calls)
        any_parallel = any(run_parallel for run_parallel, _ in segments)
        try:
            for run_parallel, segment_calls in segments:
                if run_parallel:
                    call_stream = self._run_tool_calls_parallel(
                        tool_calls=segment_calls,
                        tool_outcomes=tool_outcomes,
                        policy_results_by_id=policy_results_by_id,
                        tool_round_id=tool_round_id,
                        taint_labels_by_id=policy_plan.active_taint_labels,
                    )
                else:
                    call_stream = self._run_tool_calls_sequential(
                        messages=messages,
                        tool_calls=segment_calls,
                        round_tool_calls=tool_calls,
                        tool_outcomes=tool_outcomes,
                        policy_results_by_id=policy_results_by_id,
                        tool_round_id=tool_round_id,
                        taint_labels_by_id=policy_plan.active_taint_labels,
                    )
                async for event, outcome in call_stream:
                    yield event
                    if outcome is not None:
                        tool_outcomes.append(outcome)
                if self.stopped_for_limit:
                    break
            if self.stopped_for_limit:
                return
        except (SessionInterruptedByRequest, asyncio.CancelledError) as exc:
            async for event in self.close_after_interrupt(
                exc,
                messages=messages,
                tool_calls=tool_calls,
                tool_outcomes=tool_outcomes,
                tool_round_id=tool_round_id,
            ):
                yield event
            raise

        tool_result_messages = ordered_tool_result_messages(
            tool_calls,
            tool_outcomes,
            parallel=any_parallel,
        )
        messages.extend(tool_result_messages)
        cleared_checkpoint = await executor.checkpoint_without_pending_tool_round(session.id)
        try:
            await executor._session_store.append_transcript_messages_and_transform_checkpoint(
                session.id,
                tool_result_messages,
                executor._checkpoint_transform(cleared_checkpoint),
            )
        except asyncio.CancelledError:
            if await executor._session_control.interrupt_requested(session.id):
                clear_current_task_cancellation()
                await executor._session_store.append_transcript_messages_and_transform_checkpoint(
                    session.id,
                    tool_result_messages,
                    executor._checkpoint_transform(cleared_checkpoint),
                )
            raise

    async def _apply_limit_evaluation(
        self,
        evaluation: LimitEvaluation,
        *,
        messages: list[Message],
        tool_calls: list[runtime_records.ToolCallRequest],
        completed_tool_outcomes: list[runtime_records.ToolCallOutcome] | None = None,
        tool_round_id: str | None,
    ) -> AsyncIterator[Event]:
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
            tool_round_id=tool_round_id,
            run_started_at=self._run_started_at,
            turn_usage_tracker=self._turn_usage_tracker,
            active_run=self._active_run,
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
        tool_round_id: str | None,
        clear_pending_approval: bool = False,
    ) -> AsyncIterator[Event]:
        cancellation_artifacts: list[dict[str, Any]] | None = None
        cancellation_artifacts_by_id: dict[str, list[dict[str, Any]]] | None = None
        if isinstance(exc, SessionInterruptedByRequest):
            pass
        elif isinstance(exc, asyncio.CancelledError):
            if not await self._executor._session_control.interrupt_requested(self._session.id):
                return
            clear_current_task_cancellation()
            cancellation_artifacts = _cancellation_artifacts(exc)
            producer_id = _cancellation_tool_call_id(exc)
            if producer_id is not None and cancellation_artifacts:
                cancellation_artifacts_by_id = {producer_id: cancellation_artifacts}
                cancellation_artifacts = None
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
            tool_round_id=tool_round_id,
            cancellation_artifacts=cancellation_artifacts,
            cancellation_artifacts_by_id=cancellation_artifacts_by_id,
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
        tool_round_id: str | None,
        round_tool_calls: list[runtime_records.ToolCallRequest] | None = None,
        taint_labels_by_id: Mapping[str, frozenset[str]] = MappingProxyType({}),
    ) -> AsyncIterator[tuple[Event, runtime_records.ToolCallOutcome | None]]:
        if round_tool_calls is None:
            round_tool_calls = tool_calls
        for tool_call in tool_calls:
            await self._executor._session_control.raise_if_interrupted(self._session.id)
            limit_evaluation = await self._limit_gate.evaluate_limits(pending_tool_calls=1)
            async for event in self._apply_limit_evaluation(
                limit_evaluation,
                messages=messages,
                tool_calls=round_tool_calls,
                completed_tool_outcomes=tool_outcomes,
                tool_round_id=tool_round_id,
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
                policy_result=policy_results_by_id.get(tool_call.id),
                tool_round_id=tool_round_id,
                taint_labels=taint_labels_by_id.get(tool_call.id, frozenset()),
            ):
                yield event, outcome
            await self._executor._session_control.raise_if_interrupted(self._session.id)

    async def _run_tool_calls_parallel(
        self,
        *,
        tool_calls: list[runtime_records.ToolCallRequest],
        tool_outcomes: list[runtime_records.ToolCallOutcome],
        policy_results_by_id: dict[str, ToolPolicyResult | None],
        tool_round_id: str | None,
        taint_labels_by_id: Mapping[str, frozenset[str]] = MappingProxyType({}),
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
                        policy_result=policy_results_by_id.get(tool_call.id),
                        tool_round_id=tool_round_id,
                        taint_labels=taint_labels_by_id.get(tool_call.id, frozenset()),
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

        try:
            async with asyncio.TaskGroup() as task_group:
                for index, tool_call in enumerate(tool_calls):
                    task_group.create_task(execute_call(index, tool_call))
        except BaseExceptionGroup as exc_group:
            flush_completed_outcomes()
            raise _parallel_tool_round_exception(exc_group) from exc_group
        except asyncio.CancelledError:
            flush_completed_outcomes()
            for index, child_exc in enumerate(child_cancellations):
                if child_exc is not None and _cancellation_artifacts(child_exc):
                    _set_cancellation_tool_call_id(child_exc, tool_calls[index].id)
                    raise child_exc from None
            raise
        for index, tool_call in enumerate(tool_calls):
            if all(outcome is None for _, outcome in buffers[index]):
                buffers[index].append(
                    self._abnormal_tool_termination_item(
                        tool_call=tool_call,
                        tool_round_id=tool_round_id,
                    )
                )
        for buffer in buffers:
            for item in buffer:
                yield item

    def _abnormal_tool_termination_item(
        self,
        *,
        tool_call: runtime_records.ToolCallRequest,
        tool_round_id: str | None,
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
                tool_round_id=tool_round_id,
            ),
            "abnormal_termination": True,
            "result": result.model_dump(),
        }
        if tool_round_id is not None:
            payload["tool_round_id"] = tool_round_id
        return (
            Event(
                type=EventType.TOOL_CALL_FAILED,
                session_id=self._session.id,
                agent_name=self._registered_agent.spec.name,
                environment_name=self._environment_name,
                tool_name=tool_call.name,
                payload=payload,
            ),
            runtime_records.ToolCallOutcome(call=tool_call, result=result),
        )


def ordered_tool_result_messages(
    tool_calls: list[runtime_records.ToolCallRequest],
    outcomes: list[runtime_records.ToolCallOutcome],
    *,
    parallel: bool,
) -> list[Message]:
    if parallel:
        order = {tool_call.id: index for index, tool_call in enumerate(tool_calls)}
        outcomes = sorted(outcomes, key=lambda outcome: order.get(outcome.call.id, len(order)))
    return transcript_helpers.tool_result_messages(outcomes)


def _first_user_input_tool_call(
    registered_agent: runtime_records.RegisteredAgentState,
    tool_calls: list[runtime_records.ToolCallRequest],
    policy_results_by_id: dict[str, ToolPolicyResult | None],
) -> tuple[runtime_records.ToolCallRequest, str, list[str]] | None:
    for tool_call in tool_calls:
        registered_tool = registered_agent.tools.get(tool_call.name)
        if registered_tool is None or not getattr(registered_tool.tool, "pauses_session", False):
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
    registered_tool = registered_agent.tools.get(tool_call.name)
    return True if registered_tool is None else registered_tool.parallel_safe


def _parallel_tool_round_exception(group: BaseExceptionGroup) -> BaseException:
    flattened: list[BaseException] = []

    def flatten(exc: BaseException) -> None:
        if isinstance(exc, BaseExceptionGroup):
            for sub_exc in exc.exceptions:
                flatten(sub_exc)
        else:
            flattened.append(exc)

    flatten(group)
    for exc in flattened:
        if isinstance(exc, SessionInterruptedByRequest | ToolApprovalRequired):
            return exc
    for exc in flattened:
        if not isinstance(exc, asyncio.CancelledError):
            return exc
    return flattened[0]


def _cancellation_artifacts(exc: asyncio.CancelledError) -> list[dict[str, Any]]:
    if isinstance(exc, RunnerCancelledError):
        return copy_json_value(exc.artifacts, "artifacts")
    artifacts = getattr(exc, "artifacts", None)
    if artifacts is not None:
        return copy_json_value(artifacts, "artifacts")
    return []


_CANCELLATION_TOOL_CALL_ID_ATTR = "_cayu_cancellation_tool_call_id"


def _set_cancellation_tool_call_id(exc: asyncio.CancelledError, tool_call_id: str) -> None:
    setattr(exc, _CANCELLATION_TOOL_CALL_ID_ATTR, tool_call_id)


def _cancellation_tool_call_id(exc: asyncio.CancelledError) -> str | None:
    value = getattr(exc, _CANCELLATION_TOOL_CALL_ID_ATTR, None)
    return value if isinstance(value, str) else None


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


def _runner(registered_environment: runtime_records.RegisteredEnvironment | None) -> Any:
    if registered_environment is None:
        return None
    return registered_environment.environment.runner


def _vault(registered_environment: runtime_records.RegisteredEnvironment | None) -> Any:
    if registered_environment is None:
        return None
    return registered_environment.environment.vault


def _knowledge_store(
    registered_environment: runtime_records.RegisteredEnvironment | None,
) -> Any:
    if registered_environment is None:
        return None
    return registered_environment.environment.knowledge_store


def _mcp_servers(
    registered_environment: runtime_records.RegisteredEnvironment | None,
) -> tuple[Any, ...]:
    if registered_environment is None:
        return ()
    return registered_environment.environment.mcp_servers


@dataclass(frozen=True)
class _ProxyAuthorizationRecord:
    destination: str
    credential: SecretRef | None
    action: str | None
    metadata: dict[str, Any]
    result: ProxyAuthorizationResult


class _RedactingCredentialProxy(CredentialProxy):
    def __init__(
        self,
        proxy: CredentialProxy,
        on_resolve: Callable[[ResolvedSecret], None],
        on_authorize: Callable[[_ProxyAuthorizationRecord], None],
    ) -> None:
        if not isinstance(proxy, CredentialProxy):
            raise TypeError("proxy must be a CredentialProxy.")
        if not callable(on_resolve):
            raise TypeError("on_resolve must be callable.")
        if not callable(on_authorize):
            raise TypeError("on_authorize must be callable.")
        self._proxy = proxy
        self._on_resolve = on_resolve
        self._on_authorize = on_authorize

    async def resolve(
        self,
        ref: SecretRef,
        *,
        scope: dict[str, Any] | None = None,
    ) -> ResolvedSecret:
        copied_ref = copy_secret_ref(ref)
        copied_scope = None if scope is None else copy_json_object(scope, "scope")
        secret = await self._proxy.resolve(
            copied_ref,
            scope=None if copied_scope is None else copy_json_object(copied_scope, "scope"),
        )
        if type(secret) is not ResolvedSecret:
            raise TypeError("Proxy secret resolution must return ResolvedSecret.")
        self._on_resolve(copy_resolved_secret(secret))
        return copy_resolved_secret(secret)

    async def authorize_request(
        self,
        *,
        destination: str,
        credential: SecretRef | None = None,
        action: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> ProxyAuthorizationResult:
        copied_destination = require_clean_nonblank(destination, "destination")
        copied_credential = None if credential is None else copy_secret_ref(credential)
        copied_action = None if action is None else require_clean_nonblank(action, "action")
        copied_metadata = {} if metadata is None else copy_json_object(metadata, "metadata")
        result = await self._proxy.authorize_request(
            destination=copied_destination,
            credential=copied_credential,
            action=copied_action,
            metadata=copy_json_object(copied_metadata, "metadata"),
        )
        if type(result) is not ProxyAuthorizationResult:
            raise TypeError("Proxy authorization must return ProxyAuthorizationResult.")
        copied_result = copy_proxy_authorization_result(result)
        self._on_authorize(
            _ProxyAuthorizationRecord(
                destination=copied_destination,
                credential=copied_credential,
                action=copied_action,
                metadata=copied_metadata,
                result=copied_result,
            )
        )
        return copied_result


def _proxy(
    registered_environment: runtime_records.RegisteredEnvironment | None,
    *,
    on_resolve: Callable[[ResolvedSecret], None],
    on_authorize: Callable[[_ProxyAuthorizationRecord], None],
) -> Any:
    if registered_environment is None:
        return None
    proxy = registered_environment.environment.proxy
    if proxy is None:
        return None
    return _RedactingCredentialProxy(proxy, on_resolve, on_authorize)


def _mcp_toolsets_for_agent(
    registered_agent: runtime_records.RegisteredAgentState,
) -> tuple[McpToolset, ...]:
    toolsets: list[McpToolset] = []
    for registered_tool in registered_agent.tools.values():
        tool = registered_tool.tool
        if isinstance(tool, McpToolAdapter):
            toolsets.append(tool.toolset)
    return tuple(toolsets)


def _mcp_current_server_counts(toolsets: tuple[McpToolset, ...]) -> dict[str, int]:
    counts: dict[str, int] = {}
    seen_toolsets: set[int] = set()
    for toolset in toolsets:
        toolset_key = id(toolset)
        if toolset_key in seen_toolsets:
            continue
        seen_toolsets.add(toolset_key)
        counts[toolset.server.name] = counts.get(toolset.server.name, 0) + 1
    return counts


def _mcp_prior_server_counts(
    records: list[EventRecord],
    *,
    environment_name: str | None,
) -> dict[str, int]:
    counts: dict[str, int] = {}
    seen_identities: set[str] = set()
    for record in records:
        if record.event.environment_name != environment_name:
            continue
        server_name = record.event.payload.get("server_name")
        manifest_identity = record.event.payload.get("manifest_identity")
        if not isinstance(server_name, str) or not isinstance(manifest_identity, str):
            continue
        if manifest_identity in seen_identities:
            continue
        seen_identities.add(manifest_identity)
        counts[server_name] = counts.get(server_name, 0) + 1
    return counts


def _latest_mcp_manifest_event(
    records: list[EventRecord],
    *,
    manifest_identity: str,
    environment_name: str | None,
) -> EventRecord | None:
    for record in reversed(records):
        if (
            record.event.environment_name == environment_name
            and record.event.payload.get("manifest_identity") == manifest_identity
        ):
            return record
    return None


def _latest_mcp_manifest_event_for_server(
    records: list[EventRecord],
    *,
    server_name: str,
    environment_name: str | None,
) -> EventRecord | None:
    for record in reversed(records):
        if (
            record.event.environment_name == environment_name
            and record.event.payload.get("server_name") == server_name
        ):
            return record
    return None


def _mcp_manifest_status(
    *,
    toolset: McpToolset,
    previous: EventRecord | None,
) -> tuple[str, dict[str, Any] | None, dict[str, Any]]:
    current_tools = _mcp_manifest_tool_hashes(toolset.manifest_tools)
    empty_diff = {
        "server_changed": False,
        "added_tools": [],
        "removed_tools": [],
        "changed_tools": [],
    }
    if previous is None:
        return "first_seen", None, empty_diff

    previous_payload = previous.event.payload
    previous_summary = {
        "event_id": previous.event.id,
        "session_id": previous.event.session_id,
        "sequence": previous.sequence,
        "manifest_identity": previous_payload.get("manifest_identity"),
        "manifest_hash": previous_payload.get("manifest_hash"),
        "server_hash": previous_payload.get("server_hash"),
        "status": previous_payload.get("status"),
    }
    if previous_payload.get("manifest_hash") == toolset.manifest_hash:
        return "unchanged", previous_summary, empty_diff

    previous_tools = _mcp_manifest_tool_hashes(previous_payload.get("tools"))
    added = sorted(name for name in current_tools if name not in previous_tools)
    removed = sorted(name for name in previous_tools if name not in current_tools)
    changed = sorted(
        name
        for name, tool_hash in current_tools.items()
        if name in previous_tools and previous_tools[name] != tool_hash
    )
    return (
        "changed",
        previous_summary,
        {
            "server_changed": previous_payload.get("server_hash") != toolset.manifest_server_hash,
            "added_tools": added,
            "removed_tools": removed,
            "changed_tools": changed,
        },
    )


def _mcp_manifest_tool_hashes(value: object) -> dict[str, str]:
    if not isinstance(value, list | tuple):
        return {}
    result: dict[str, str] = {}
    for item in value:
        if not isinstance(item, Mapping):
            continue
        entry = cast("Mapping[str, object]", item)
        cayu_name = entry.get("cayu_name")
        tool_hash = entry.get("hash")
        if isinstance(cayu_name, str) and isinstance(tool_hash, str):
            result[cayu_name] = tool_hash
    return result


def _tool_effect(
    registered_agent: runtime_records.RegisteredAgentState,
    tool_call: runtime_records.ToolCallRequest,
) -> ToolEffect:
    registered_tool = registered_agent.tools.get(tool_call.name)
    if registered_tool is None:
        return ToolEffect.EXTERNAL
    return registered_tool.effect


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


def _resolve_before_tool_call_decision(
    decision: BeforeToolCallDecision | None,
    resolution: _BeforeToolCallResolution,
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
        resolution.arguments = copy_json_value(modified_arguments, "modified_arguments")
        return False
    if decision.action == "short_circuit":
        synthetic = decision.synthetic_result
        if synthetic is None:
            raise TypeError("A short_circuit decision must carry a synthetic_result.")
        resolution.short_circuit_result = synthetic.model_copy(deep=True)
        return True
    reason = decision.block_reason
    if reason is None:
        raise TypeError("A block decision must carry a block_reason.")
    resolution.block_reason = reason
    return True


def _resolve_after_tool_call_decision(
    decision: AfterToolCallDecision | None,
) -> ToolResult | None:
    if decision is None:
        return None
    if type(decision) is not AfterToolCallDecision:
        raise TypeError("after_tool_call must return an AfterToolCallDecision or None.")
    if decision.action == "modify":
        modified = decision.modified_result
        if modified is None:
            raise TypeError("An after_tool_call modify decision must carry a modified_result.")
        return modified.model_copy(deep=True)
    return None


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
) -> Event:
    return _build_runtime_hook_event(
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


_POLICY_DENIAL_CONTROL_PAYLOAD_FIELDS = frozenset(
    {
        "approval_id",
        "blocked_by",
        "decision",
        "denied_by",
        "idempotency_key",
        "input_id",
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
) -> tuple[Event, ToolResult]:
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
        )
    return _bound_policy_denial_event(event=event, result=result)


def _redact_tool_result_for_event(
    *,
    event: Event,
    result: ToolResult,
    redactor: SecretRedactor,
) -> ToolResult:
    if _is_policy_denial_event(event):
        return _redact_policy_denial_result(result, redactor)
    return tool_results.redact_tool_result(result, redactor)


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
        if key in _POLICY_DENIAL_CONTROL_PAYLOAD_FIELDS:
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


def _redactor_with_resolved_secrets(
    redactor: SecretRedactor,
    secrets: list[ResolvedSecret],
) -> SecretRedactor:
    resolved_redactor = redactor
    for secret in secrets:
        if type(secret) is not ResolvedSecret:
            raise TypeError("Resolved proxy secrets must be ResolvedSecret instances.")
        resolved_redactor = resolved_redactor.with_secret(secret)
    return resolved_redactor
