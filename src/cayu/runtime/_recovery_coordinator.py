"""Approval and user-input continuation and recovery ownership.

The coordinator owns durable paused-round continuation behavior without
importing or accepting :class:`CayuApp`. Public request validation and registry
selection remain on the application façade. Session execution and terminal
hook orchestration are narrow callbacks until the session engine is extracted.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncGenerator, AsyncIterator, Awaitable, Callable
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from cayu._validation import copy_json_value
from cayu.core.events import Event, EventType
from cayu.core.messages import Message
from cayu.core.thinking import ThinkingConfig
from cayu.core.tools import _TOOL_POLICY_DENIAL_SOURCE, ToolResult
from cayu.environments import EnvironmentFactoryOperation
from cayu.runtime import _approval_support as approval_support
from cayu.runtime import _runtime_records as runtime_records
from cayu.runtime import _tool_execution as tool_execution
from cayu.runtime import _tool_results as tool_results
from cayu.runtime import _transcript as transcript_helpers
from cayu.runtime._environment_lifecycle import (
    EnvironmentLifecycle,
    exception_failure_payload,
)
from cayu.runtime._event_writer import RuntimeEventWriter
from cayu.runtime._run_limits import RunLimitController, SessionUsageTracker
from cayu.runtime._session_control import SessionControl
from cayu.runtime._tool_round_executor import (
    ToolRoundExecutor,
    policy_denial_payload_fields,
)
from cayu.runtime.approvals import (
    PendingToolApproval,
    PendingToolCallApproval,
    ToolApprovalDecision,
    ToolApprovalRecoveryOutcome,
    ToolApprovalRecoveryRequest,
    ToolApprovalRequest,
    expiry_resolution_actor,
    resolution_actor_payload,
)
from cayu.runtime.budgets import (
    BudgetLimit,
    copy_request_budget_limits,
    request_budget_limits_for_session,
)
from cayu.runtime.costs import SessionCostSummary
from cayu.runtime.hooks import RuntimeHookPhase
from cayu.runtime.loop_policies import LoopPolicy
from cayu.runtime.retry_policy import RetryPolicy
from cayu.runtime.sessions import CheckpointTransform, Session, SessionStatus, SessionStore
from cayu.runtime.stop_policy import RunLimits, StopDecision, copy_run_limits, has_run_limits
from cayu.runtime.structured_output import (
    StructuredOutputSpec,
    copy_structured_output_spec,
)
from cayu.runtime.tasks import Task, TaskStore
from cayu.runtime.tool_policy import ToolPolicyDecision
from cayu.runtime.usage import SessionUsageSummary, session_usage_summary
from cayu.runtime.user_input import (
    PENDING_USER_INPUT_CHECKPOINT_KEY,
    PendingUserInput,
    UserInputRecoveryRequest,
    UserInputResponse,
)
from cayu.vaults import SecretRedactor

_INTERRUPTION_TYPE_TOOL_APPROVAL_REQUIRED = "tool_approval_required"
_INTERRUPTION_TYPE_USER_INPUT_REQUIRED = "user_input_required"
_DEFAULT_APPROVAL_MAX_STEPS = 16

CheckpointTransformFactory = Callable[[dict[str, Any]], CheckpointTransform]
FinalizeAbandonedSession = Callable[[str], Awaitable[None]]
EffectiveRetryPolicy = Callable[[RetryPolicy | None], RetryPolicy]
RecoveryCleanup = Callable[[], Awaitable[None]]


async def _run_recovery_cleanup_steps(
    *,
    authoritative_failure: BaseException | None,
    steps: tuple[tuple[str, RecoveryCleanup], ...],
) -> None:
    """Run every handoff cleanup without obscuring its triggering failure.

    Once task cancellation starts a continuation handoff, a later ``cancel()``
    must not interrupt finalization or fence release. Run that cleanup in a
    shielded child task which inherits the current run-fence context, and wait
    through any repeated cancellation requests. ``GeneratorExit`` is different:
    an explicit ``aclose()`` consumes it, so a cleanup failure must remain visible
    to the caller instead of being reduced to an exception note.
    """

    async def run_steps() -> list[tuple[str, BaseException]]:
        cleanup_failures: list[tuple[str, BaseException]] = []
        for operation, cleanup in steps:
            try:
                await cleanup()
            except BaseException as cleanup_failure:
                cleanup_failures.append((operation, cleanup_failure))
        return cleanup_failures

    if isinstance(authoritative_failure, asyncio.CancelledError):
        cleanup_task = asyncio.create_task(run_steps())
        while not cleanup_task.done():
            try:
                await asyncio.shield(cleanup_task)
            except asyncio.CancelledError:
                # Preserve the first cancellation as the caller-visible outcome,
                # but do not let later cancellation requests strand durable state.
                continue
        cleanup_failures = cleanup_task.result()
    else:
        cleanup_failures = await run_steps()

    if not cleanup_failures:
        return
    if authoritative_failure is not None and not isinstance(authoritative_failure, GeneratorExit):
        for operation, cleanup_failure in cleanup_failures:
            authoritative_failure.add_note(
                "Continuation recovery cleanup failed during "
                f"{operation}: {type(cleanup_failure).__name__}. "
                "The original failure remains authoritative."
            )
        return

    operation, first_failure = cleanup_failures[0]
    for later_operation, later_failure in cleanup_failures[1:]:
        first_failure.add_note(
            "Additional continuation recovery cleanup failure during "
            f"{later_operation}: {type(later_failure).__name__}."
        )
    first_failure.add_note(f"Continuation recovery cleanup failed during {operation}.")
    raise first_failure


@dataclass(frozen=True)
class RecoverySessionRunRequest:
    session: Session
    registered_agent: runtime_records.RegisteredAgentState
    registered_provider: runtime_records.RegisteredProvider
    registered_environment: runtime_records.RegisteredEnvironment | None
    messages: list[Message]
    messages_to_append: list[Message]
    max_steps: int
    limits: RunLimits
    budget_limits: tuple[BudgetLimit, ...]
    retry_policy: RetryPolicy
    structured_output: StructuredOutputSpec | None
    thinking: ThinkingConfig | None
    request_loop_policies: tuple[LoopPolicy, ...]
    request_metadata: dict[str, Any]
    task_id: str | None
    task_worker_id: str | None
    start_event_type: EventType | None
    start_event_payload: dict[str, Any]
    start_task_on_enter: bool
    release_run_fence_on_exit: bool


@dataclass(frozen=True)
class RecoveryTerminalEventRequest:
    event: Event
    phase: RuntimeHookPhase
    session: Session
    registered_agent: runtime_records.RegisteredAgentState
    registered_environment: runtime_records.RegisteredEnvironment | None


@dataclass(frozen=True)
class RecoveryLimitStopRequest:
    session: Session
    registered_agent: runtime_records.RegisteredAgentState
    registered_environment: runtime_records.RegisteredEnvironment | None
    environment_name: str | None
    decision: StopDecision
    usage_summary: SessionUsageSummary
    cost_summary: SessionCostSummary | None
    messages: list[Message]
    tool_calls: list[runtime_records.ToolCallRequest]
    completed_tool_outcomes: list[runtime_records.ToolCallOutcome]
    pending_approval_to_clear: PendingToolApproval | None


@dataclass(frozen=True)
class RecoveryTaskEventRequest:
    event_type: EventType
    task: Task
    session: Session
    registered_agent: runtime_records.RegisteredAgentState
    registered_environment: runtime_records.RegisteredEnvironment | None


RunSession = Callable[[RecoverySessionRunRequest], AsyncGenerator[Event, None]]
TerminalEventStream = Callable[[RecoveryTerminalEventRequest], AsyncIterator[Event]]
LimitStopEventStream = Callable[[RecoveryLimitStopRequest], AsyncIterator[Event]]
TaskEventFactory = Callable[[RecoveryTaskEventRequest], Event]


class RecoveryCoordinator:
    """Continue approval and user-input pauses from durable state."""

    def __init__(
        self,
        *,
        session_store: SessionStore,
        task_store: TaskStore | None,
        event_writer: RuntimeEventWriter,
        session_control: SessionControl[SessionUsageTracker],
        environment_lifecycle: EnvironmentLifecycle,
        run_limit_controller: RunLimitController,
        tool_round_executor: ToolRoundExecutor,
        secret_redactor: SecretRedactor,
        clock: Callable[[], datetime],
        checkpoint_transform: CheckpointTransformFactory,
        effective_retry_policy: EffectiveRetryPolicy,
        run_session: RunSession,
        emit_terminal_event_with_hooks: TerminalEventStream,
        finalize_abandoned_session_by_id: FinalizeAbandonedSession,
        stop_session_for_limit_reached: LimitStopEventStream,
        task_event: TaskEventFactory,
    ) -> None:
        self._session_store = session_store
        self._task_store = task_store
        self._event_writer = event_writer
        self._session_control = session_control
        self._environment_lifecycle = environment_lifecycle
        self._run_limit_controller = run_limit_controller
        self._tool_round_executor = tool_round_executor
        self._secret_redactor = secret_redactor
        self._clock = clock
        self._checkpoint_transform = checkpoint_transform
        self._effective_retry_policy = effective_retry_policy
        self._run_session = run_session
        self._emit_terminal_event_with_hooks = emit_terminal_event_with_hooks
        self._finalize_abandoned_session_by_id = finalize_abandoned_session_by_id
        self._stop_session_for_limit_reached = stop_session_for_limit_reached
        self._task_event = task_event

    async def _cleanup_recovery_handoff(
        self,
        *,
        stream: AsyncGenerator[Event, None] | None,
        session_id: str,
        authoritative_failure: BaseException | None,
        finalize_abandoned: bool,
        release_run_fence: bool,
    ) -> None:
        cleanup_steps: list[tuple[str, RecoveryCleanup]] = []
        if stream is not None:
            cleanup_steps.append(("nested stream close", stream.aclose))
        if finalize_abandoned:
            cleanup_steps.append(
                (
                    "abandoned session finalization",
                    lambda: self._finalize_abandoned_session_by_id(session_id),
                )
            )
        if release_run_fence:
            cleanup_steps.append(
                ("run fence release", lambda: self._session_store.release_run_fence(session_id))
            )
        await _run_recovery_cleanup_steps(
            authoritative_failure=authoritative_failure,
            steps=tuple(cleanup_steps),
        )

    async def continue_user_input_resolution(
        self,
        *,
        response: UserInputResponse,
        session: Session,
        pending: PendingUserInput,
        registered_agent: runtime_records.RegisteredAgentState,
        registered_provider: runtime_records.RegisteredProvider,
        registered_environment: runtime_records.RegisteredEnvironment | None,
        emit_resume_event: bool = True,
    ) -> AsyncGenerator[Event, None]:
        environment_name = _environment_name(registered_environment)
        pending_cleared = False
        tool_outcomes: list[runtime_records.ToolCallOutcome] = []
        # Restore the original run's config persisted on the pending input; explicit overrides
        # on the resolution request win. Pending states written before this existed fall back to
        # the historical defaults.
        effective_max_steps = _effective_user_input_max_steps(
            max_steps=response.max_steps,
            pending=pending,
        )
        effective_limits = _effective_user_input_run_limits(
            limits=response.limits,
            pending=pending,
        )
        effective_budget_limits = _effective_user_input_budget_limits(
            budget_limits=response.budget_limits,
            pending=pending,
        )
        effective_retry_policy = self._effective_retry_policy(
            _effective_user_input_retry_policy(
                retry_policy=response.retry_policy,
                pending=pending,
            )
        )
        try:
            transcript = await self._session_store.load_transcript(session.id)
            resume_events = await self._session_store.load_events(session.id)
            factory_started_event = await self._environment_lifecycle.emit_factory_started(
                session=session,
                registered_agent=registered_agent,
                registered_environment=registered_environment,
            )
            if factory_started_event is not None:
                yield factory_started_event
            factory_resolution = await self._environment_lifecycle.resolve_factory(
                session=session,
                registered_agent=registered_agent,
                registered_environment=registered_environment,
                started_event=factory_started_event,
                operation=EnvironmentFactoryOperation.RECONNECT,
            )
            registered_environment = factory_resolution.registered_environment
            environment_name = _environment_name(registered_environment)
            for event in factory_resolution.events:
                yield event

            if factory_resolution.error is not None:
                raise factory_resolution.error
            if emit_resume_event:
                yield await self._event_writer.emit(
                    Event(
                        type=EventType.SESSION_RESUMED,
                        session_id=session.id,
                        agent_name=registered_agent.spec.name,
                        environment_name=environment_name,
                        payload={
                            "interruption_type": _INTERRUPTION_TYPE_USER_INPUT_REQUIRED,
                            "input_id": pending.input_id,
                            "tool_call_id": pending.tool_call_id,
                            "resolved_by": resolution_actor_payload(response.resolved_by),
                        },
                    )
                )
            binding_started_event = await self._environment_lifecycle.emit_binding_started(
                session=session,
                registered_agent=registered_agent,
                registered_environment=registered_environment,
            )
            if binding_started_event is not None:
                yield binding_started_event
            binding_result = await self._environment_lifecycle.bind(
                session=session,
                registered_agent=registered_agent,
                registered_environment=registered_environment,
                started_event=binding_started_event,
            )
            registered_environment = binding_result.registered_environment
            for event in binding_result.events:
                yield event
            if binding_result.error is not None:
                raise binding_result.error

            round_tool_calls = [
                runtime_records.ToolCallRequest(
                    id=pending_call.tool_call_id,
                    name=pending_call.tool_name,
                    arguments=copy_json_value(pending_call.arguments, "arguments"),
                )
                for pending_call in pending.tool_calls
            ]
            # Reuse any outcomes already recorded for this round — e.g. a prior resume attempt
            # that ran some tools before a mid-resume failure — so a retry never re-executes a
            # side-effecting tool. The round was already projected against limits at pause time;
            # its remaining tools run on resume without a fresh budget projection (so the user's
            # answer is never discarded by a limit check here).
            recorded_outcomes = approval_support.recorded_round_tool_outcomes(
                events=resume_events,
                pending_calls=pending.tool_calls,
                input_id=pending.input_id,
            )
            pending_by_id = {call.tool_call_id: call for call in pending.tool_calls}

            # Build the round's outcomes in model order: a call already recorded (retry) is
            # reused; the answered ask_user call gets the injected answer; every other allowed
            # call executes now (none ran before the pause); a denied call is blocked.
            for tool_call in round_tool_calls:
                recorded_outcome = recorded_outcomes.get(tool_call.id)
                if recorded_outcome is not None:
                    tool_outcomes.append(recorded_outcome)
                    continue

                if tool_call.id == pending.tool_call_id:
                    registered_tool = registered_agent.tools.get(tool_call.name)
                    idempotency_key = tool_execution.tool_idempotency_key(
                        session_id=session.id,
                        tool_call_id=tool_call.id,
                        pause_id=pending.input_id,
                    )
                    result = ToolResult(
                        content=response.answer,
                        structured=response.structured,
                        artifacts=response.artifacts,
                        is_error=False,
                    )
                    started_payload: dict[str, Any] = {
                        "tool_call_id": tool_call.id,
                        "idempotency_key": idempotency_key,
                        "arguments": deepcopy(tool_call.arguments),
                        "input_id": pending.input_id,
                    }
                    if registered_tool is not None:
                        started_payload["effect"] = registered_tool.effect.value
                    yield await self._event_writer.emit(
                        Event(
                            type=EventType.TOOL_CALL_STARTED,
                            session_id=session.id,
                            agent_name=registered_agent.spec.name,
                            environment_name=environment_name,
                            tool_name=tool_call.name,
                            payload=started_payload,
                        )
                    )
                    async for (
                        event,
                        outcome,
                    ) in self._tool_round_executor.emit_tool_call_result_with_hooks(
                        event=Event(
                            type=EventType.TOOL_CALL_COMPLETED,
                            session_id=session.id,
                            agent_name=registered_agent.spec.name,
                            environment_name=environment_name,
                            tool_name=tool_call.name,
                            payload={
                                "tool_call_id": tool_call.id,
                                "idempotency_key": idempotency_key,
                                "input_id": pending.input_id,
                                "resolved_by": resolution_actor_payload(response.resolved_by),
                                "result": result.model_dump(),
                            },
                        ),
                        session=session,
                        registered_agent=registered_agent,
                        registered_environment=registered_environment,
                        tool_call=tool_call,
                        result=result,
                        task_id=pending.task_id,
                    ):
                        yield event
                        if outcome is not None:
                            tool_outcomes.append(outcome)
                    continue

                pending_call = pending_by_id[tool_call.id]
                policy_result = approval_support.policy_result_from_pending_tool_call(pending_call)
                call_taint_labels = approval_support.taint_labels_from_pending_tool_call(
                    pending_call
                )
                # `ToolRoundExecutor.execute_tool_call(check_policy=False)` does not re-enforce
                # the decision, so a DENY must be blocked here explicitly (mirroring the approval
                # resume) — otherwise a policy-denied sibling would execute. REQUIRE_APPROVAL
                # cannot occur: it would have preempted the ask_user pause with an approval pause.
                if policy_result is not None and policy_result.decision == ToolPolicyDecision.DENY:
                    reason = tool_execution.policy_denial_reason(policy_result)
                    blocked_result = tool_execution.blocked_tool_result(
                        policy_result, reason=reason
                    )
                    idempotency_key = tool_execution.tool_idempotency_key(
                        session_id=session.id,
                        tool_call_id=tool_call.id,
                        pause_id=pending.input_id,
                    )
                    async for (
                        event,
                        outcome,
                    ) in self._tool_round_executor.emit_tool_call_result_with_hooks(
                        event=Event(
                            type=EventType.TOOL_CALL_BLOCKED,
                            session_id=session.id,
                            agent_name=registered_agent.spec.name,
                            environment_name=environment_name,
                            tool_name=tool_call.name,
                            payload={
                                "tool_call_id": tool_call.id,
                                "idempotency_key": idempotency_key,
                                **policy_denial_payload_fields(
                                    tool_name=tool_call.name,
                                    denied_by=_TOOL_POLICY_DENIAL_SOURCE,
                                    decision=policy_result.decision.value,
                                    reason=reason,
                                    metadata=policy_result.metadata,
                                ),
                                "result": blocked_result.model_dump(),
                            },
                        ),
                        session=session,
                        registered_agent=registered_agent,
                        registered_environment=registered_environment,
                        tool_call=tool_call,
                        result=blocked_result,
                        task_id=pending.task_id,
                    ):
                        yield event
                        if outcome is not None:
                            tool_outcomes.append(outcome)
                    continue

                async for event, outcome in self._tool_round_executor.execute_tool_call(
                    session=session,
                    registered_agent=registered_agent,
                    registered_environment=registered_environment,
                    tool_call=tool_call,
                    request_metadata=response.metadata,
                    task_id=pending.task_id,
                    check_policy=False,
                    policy_result=policy_result,
                    input_id=pending.input_id,
                    taint_labels=call_taint_labels,
                ):
                    yield event
                    if outcome is not None:
                        tool_outcomes.append(outcome)

            # The resume executes the round's tools sequentially in model order, so the outcome
            # list already lines up with the assistant tool-call parts.
            tool_result_messages = transcript_helpers.tool_result_messages(tool_outcomes)
            transcript.extend(tool_result_messages)
            cleared_checkpoint = await self._checkpoint_without_pending_user_input(session.id)
            await self._session_store.append_transcript_messages_and_transform_checkpoint(
                session.id,
                tool_result_messages,
                self._checkpoint_transform(cleared_checkpoint),
            )
            pending_cleared = True

            session_stream = self._run_session(
                RecoverySessionRunRequest(
                    session=session,
                    registered_agent=registered_agent,
                    registered_provider=registered_provider,
                    registered_environment=registered_environment,
                    messages=transcript,
                    messages_to_append=[],
                    max_steps=effective_max_steps,
                    limits=effective_limits,
                    budget_limits=effective_budget_limits,
                    retry_policy=effective_retry_policy,
                    structured_output=_effective_user_input_structured_output(
                        structured_output=response.structured_output,
                        pending=pending,
                    ),
                    thinking=response.thinking or pending.thinking,
                    request_loop_policies=response.loop_policies,
                    request_metadata=response.metadata,
                    task_id=pending.task_id,
                    task_worker_id=None,
                    start_event_type=None,
                    start_event_payload={},
                    start_task_on_enter=False,
                    release_run_fence_on_exit=False,
                )
            )
            try:
                async for event in self._session_control.stream_with_out_of_band_events(
                    session.id,
                    session_stream,
                ):
                    yield event
            except GeneratorExit:
                await session_stream.aclose()
                raise
        except Exception as exc:
            if not pending_cleared:
                # The pending_user_input checkpoint is still present, so restore the resumable
                # INTERRUPTED state and emit a terminal event for closure (a SESSION_RESUMED was
                # already emitted). The caller can retry resolve_user_input; recorded outcomes
                # prevent re-running a tool that already completed. A tool that started with no
                # terminal (a crash mid-tool) cannot be re-run safely — flag it as needing manual
                # recovery so the retry is not a silent double-execution.
                # Carry the failure so a caller can distinguish "your answer failed, retry" from a
                # fresh pause (whose interrupted event has no error fields).
                payload: dict[str, Any] = {
                    **exception_failure_payload(exc),
                    "interruption_type": _INTERRUPTION_TYPE_USER_INPUT_REQUIRED,
                    "user_input": pending.model_dump(mode="json"),
                }
                if isinstance(exc, approval_support.RoundToolManualRecoveryRequired):
                    payload["manual_recovery_required"] = True
                    payload["tool_call_id"] = exc.tool_call_id
                    payload["tool_name"] = exc.tool_name
                session = await self._session_store.update_status(
                    session.id, SessionStatus.INTERRUPTED
                )
                async for event in self._emit_terminal_event_with_hooks(
                    RecoveryTerminalEventRequest(
                        event=Event(
                            type=EventType.SESSION_INTERRUPTED,
                            session_id=session.id,
                            agent_name=registered_agent.spec.name,
                            environment_name=environment_name,
                            payload=payload,
                        ),
                        phase=RuntimeHookPhase.AFTER_SESSION_INTERRUPTED,
                        session=session,
                        registered_agent=registered_agent,
                        registered_environment=registered_environment,
                    )
                ):
                    yield event
                return
            raise

    async def _checkpoint_without_pending_user_input(
        self,
        session_id: str,
    ) -> dict[str, Any]:
        checkpoint = await self._session_store.load_checkpoint(session_id)
        checkpoint = {} if checkpoint is None else copy_json_value(checkpoint, "checkpoint")
        checkpoint.pop(PENDING_USER_INPUT_CHECKPOINT_KEY, None)
        return checkpoint

    async def continue_tool_approval_resolution(
        self,
        *,
        request: ToolApprovalRequest,
        session: Session,
        pending_approval: PendingToolApproval,
        registered_agent: runtime_records.RegisteredAgentState,
        registered_provider: runtime_records.RegisteredProvider,
        registered_environment: runtime_records.RegisteredEnvironment | None,
        emit_resume_event: bool = True,
        enforce_expiry: bool = True,
    ) -> AsyncGenerator[Event, None]:
        environment_name = _environment_name(registered_environment)
        pending_approval_cleared = False
        tool_outcomes: list[runtime_records.ToolCallOutcome] = []
        expired = False
        # Restore the original run's config persisted on the pending approval;
        # explicit overrides on the approval request win. Approvals persisted
        # before this state existed fall back to the historical defaults.
        effective_max_steps = _effective_approval_max_steps(
            max_steps=request.max_steps,
            pending_approval=pending_approval,
        )
        effective_limits = _effective_approval_run_limits(
            limits=request.limits,
            pending_approval=pending_approval,
        )
        effective_budget_limits = _effective_approval_budget_limits(
            budget_limits=request.budget_limits,
            pending_approval=pending_approval,
        )
        effective_retry_policy = self._effective_retry_policy(
            _effective_approval_retry_policy(
                retry_policy=request.retry_policy,
                pending_approval=pending_approval,
            )
        )
        try:
            transcript = await self._session_store.load_transcript(session.id)
            approval_events = await self._session_store.load_events(session.id)
            history = approval_support.approval_resolution_history(
                events=approval_events,
                approval=pending_approval,
            )
            # Expiry gates the FIRST grant only: a retry of an approval that
            # already has granted or executed activity was authorized
            # in-window before a crash, so coercing it to a denial would
            # contradict the recorded grant (and trip validate_retry_decision).
            if (
                enforce_expiry
                and approval_support.pending_approval_expired(pending_approval, self._clock())
                and not history.has_granted_activity
            ):
                expired = True
                # Captured before the coercion below replaces them on the request.
                requested_decision = request.decision
                triggered_by = request.resolved_by
                assert pending_approval.expires_at is not None
                expired_at_iso = pending_approval.expires_at.isoformat()
                request = ToolApprovalRequest(
                    session_id=request.session_id,
                    approval_id=request.approval_id,
                    decision=ToolApprovalDecision.DENY,
                    reason=f"Tool approval expired at {expired_at_iso}.",
                    metadata=copy_json_value(request.metadata, "metadata"),
                    resolved_by=expiry_resolution_actor(),
                    max_steps=request.max_steps,
                    limits=request.limits,
                    budget_limits=request.budget_limits,
                    retry_policy=request.retry_policy,
                    structured_output=request.structured_output,
                    thinking=request.thinking,
                    loop_policies=request.loop_policies,
                )
            approval_support.validate_retry_decision(
                history=history,
                approval=pending_approval,
                decision=request.decision,
            )
            resolved_by_payload = resolution_actor_payload(request.resolved_by)
            recorded_outcomes = approval_support.recorded_tool_outcomes(
                events=approval_events,
                approval=pending_approval,
            )
            factory_started_event = await self._environment_lifecycle.emit_factory_started(
                session=session,
                registered_agent=registered_agent,
                registered_environment=registered_environment,
            )
            if factory_started_event is not None:
                yield factory_started_event
            factory_resolution = await self._environment_lifecycle.resolve_factory(
                session=session,
                registered_agent=registered_agent,
                registered_environment=registered_environment,
                started_event=factory_started_event,
                operation=EnvironmentFactoryOperation.RECONNECT,
            )
            registered_environment = factory_resolution.registered_environment
            environment_name = _environment_name(registered_environment)
            for event in factory_resolution.events:
                yield event
            if factory_resolution.error is not None:
                raise factory_resolution.error
            if emit_resume_event:
                yield await self._event_writer.emit(
                    approval_support.resumed_event(
                        session=session,
                        agent_name=registered_agent.spec.name,
                        environment_name=environment_name,
                        approval=pending_approval,
                        decision=request.decision,
                        resolved_by=request.resolved_by,
                        expired=expired,
                    )
                )
            if expired:
                yield await self._event_writer.emit(
                    Event(
                        type=EventType.TOOL_CALL_APPROVAL_EXPIRED,
                        session_id=session.id,
                        agent_name=registered_agent.spec.name,
                        environment_name=environment_name,
                        tool_name=pending_approval.tool_name,
                        payload={
                            "approval_id": pending_approval.approval_id,
                            "tool_call_id": pending_approval.tool_call_id,
                            "expires_at": expired_at_iso,
                            "requested_decision": requested_decision.value,
                            "resolved_by": resolved_by_payload,
                            "triggered_by": resolution_actor_payload(triggered_by),
                        },
                    )
                )

            if request.decision not in {
                ToolApprovalDecision.APPROVE,
                ToolApprovalDecision.DENY,
            }:
                raise ValueError(f"Unsupported tool approval decision: {request.decision}")

            binding_started_event = await self._environment_lifecycle.emit_binding_started(
                session=session,
                registered_agent=registered_agent,
                registered_environment=registered_environment,
            )
            if binding_started_event is not None:
                yield binding_started_event
            binding_result = await self._environment_lifecycle.bind(
                session=session,
                registered_agent=registered_agent,
                registered_environment=registered_environment,
                started_event=binding_started_event,
            )
            registered_environment = binding_result.registered_environment
            for event in binding_result.events:
                yield event
            if binding_result.error is not None:
                raise binding_result.error

            if request.decision == ToolApprovalDecision.APPROVE:
                run_started_at = time.monotonic()
                limits = copy_run_limits(effective_limits)
                budget_limits = request_budget_limits_for_session(
                    limits=effective_budget_limits,
                    agent_name=registered_agent.spec.name,
                    causal_budget_id=session.causal_budget_id,
                )
                run_baseline = (
                    session_usage_summary(session.id, approval_events)
                    if limits.scope == "run" and has_run_limits(limits)
                    else None
                )
                budget_baseline_events = (
                    approval_events if _has_run_budget_limit(budget_limits) else []
                )
                request_budget_notify_events: list[Event] = []
                recorded_tool_outcomes = list(recorded_outcomes.values())
                pending_tool_calls: list[runtime_records.ToolCallRequest] = []
                executable_pending_tool_calls = 0
                for pending_tool_call in approval_support.pending_round_tool_calls(
                    pending_approval
                ):
                    if pending_tool_call.tool_call_id in recorded_outcomes:
                        continue
                    tool_call = runtime_records.ToolCallRequest(
                        id=pending_tool_call.tool_call_id,
                        name=pending_tool_call.tool_name,
                        arguments=copy_json_value(pending_tool_call.arguments, "arguments"),
                    )
                    pending_tool_calls.append(tool_call)
                    policy_result = approval_support.policy_result_from_pending_tool_call(
                        pending_tool_call
                    )
                    if (
                        policy_result is not None
                        and policy_result.decision == ToolPolicyDecision.DENY
                    ):
                        continue
                    executable_pending_tool_calls += 1
                limit_evaluation = await self._run_limit_controller.evaluate_request_limits(
                    session=session,
                    agent_name=registered_agent.spec.name,
                    environment_name=environment_name,
                    limits=limits,
                    budget_limits=budget_limits,
                    run_started_at=run_started_at,
                    run_baseline=run_baseline,
                    budget_baseline_events=budget_baseline_events,
                    pending_tool_calls=executable_pending_tool_calls,
                    budget_notify_events=request_budget_notify_events,
                    pricing_provider_name=(
                        registered_provider.provider.billing_provider_name
                        or registered_provider.name
                    ),
                )
                for event in limit_evaluation.events:
                    yield event
                if limit_evaluation.decision is not None:
                    async for event in self._stop_session_for_limit_reached(
                        RecoveryLimitStopRequest(
                            session=session,
                            registered_agent=registered_agent,
                            registered_environment=registered_environment,
                            environment_name=environment_name,
                            decision=limit_evaluation.decision,
                            usage_summary=limit_evaluation.usage_summary,
                            cost_summary=limit_evaluation.cost_summary,
                            messages=transcript,
                            tool_calls=pending_tool_calls,
                            completed_tool_outcomes=recorded_tool_outcomes,
                            pending_approval_to_clear=pending_approval,
                        )
                    ):
                        yield event
                    pending_approval_cleared = True
                    return

            for pending_tool_call in approval_support.pending_round_tool_calls(pending_approval):
                tool_call = runtime_records.ToolCallRequest(
                    id=pending_tool_call.tool_call_id,
                    name=pending_tool_call.tool_name,
                    arguments=copy_json_value(pending_tool_call.arguments, "arguments"),
                )
                policy_result = approval_support.policy_result_from_pending_tool_call(
                    pending_tool_call
                )
                call_taint_labels = approval_support.taint_labels_from_pending_tool_call(
                    pending_tool_call
                )
                recorded_outcome = recorded_outcomes.get(tool_call.id)
                if recorded_outcome is not None:
                    tool_outcomes.append(recorded_outcome)
                    continue

                if policy_result is not None and policy_result.decision == ToolPolicyDecision.DENY:
                    reason = tool_execution.policy_denial_reason(policy_result)
                    result = tool_execution.blocked_tool_result(policy_result, reason=reason)
                    idempotency_key = tool_execution.tool_idempotency_key(
                        session_id=session.id,
                        tool_call_id=tool_call.id,
                        approval_id=pending_approval.approval_id,
                    )
                    async for (
                        event,
                        outcome,
                    ) in self._tool_round_executor.emit_tool_call_result_with_hooks(
                        event=Event(
                            type=EventType.TOOL_CALL_BLOCKED,
                            session_id=session.id,
                            agent_name=registered_agent.spec.name,
                            environment_name=environment_name,
                            tool_name=tool_call.name,
                            payload={
                                "approval_id": pending_approval.approval_id,
                                "tool_call_id": tool_call.id,
                                "idempotency_key": idempotency_key,
                                **policy_denial_payload_fields(
                                    tool_name=tool_call.name,
                                    denied_by=_TOOL_POLICY_DENIAL_SOURCE,
                                    decision=policy_result.decision.value,
                                    reason=reason,
                                    metadata=policy_result.metadata,
                                ),
                                "result": result.model_dump(),
                            },
                        ),
                        session=session,
                        registered_agent=registered_agent,
                        registered_environment=registered_environment,
                        tool_call=tool_call,
                        result=result,
                        task_id=pending_approval.task_id,
                    ):
                        yield event
                        if outcome is not None:
                            tool_outcomes.append(outcome)
                    continue

                if (
                    policy_result is not None
                    and policy_result.decision == ToolPolicyDecision.REQUIRE_APPROVAL
                    and request.decision == ToolApprovalDecision.APPROVE
                ):
                    yield await self._event_writer.emit(
                        Event(
                            type=EventType.TOOL_CALL_APPROVED,
                            session_id=session.id,
                            agent_name=registered_agent.spec.name,
                            environment_name=environment_name,
                            tool_name=tool_call.name,
                            payload={
                                "approval_id": pending_approval.approval_id,
                                "tool_call_id": tool_call.id,
                                "reason": request.reason,
                                "metadata": request.metadata,
                                "resolved_by": resolved_by_payload,
                            },
                        )
                    )

                if request.decision == ToolApprovalDecision.DENY:
                    approval_required = (
                        policy_result is not None
                        and policy_result.decision == ToolPolicyDecision.REQUIRE_APPROVAL
                    )
                    result = approval_support.approval_denied_tool_result(
                        request,
                        approval=pending_approval,
                        tool_call=tool_call,
                        approval_required=approval_required,
                    )
                    idempotency_key = tool_execution.tool_idempotency_key(
                        session_id=session.id,
                        tool_call_id=tool_call.id,
                        approval_id=pending_approval.approval_id,
                    )
                    async for (
                        event,
                        outcome,
                    ) in self._tool_round_executor.emit_tool_call_result_with_hooks(
                        event=Event(
                            type=EventType.TOOL_CALL_APPROVAL_DENIED,
                            session_id=session.id,
                            agent_name=registered_agent.spec.name,
                            environment_name=environment_name,
                            tool_name=tool_call.name,
                            payload={
                                "approval_id": pending_approval.approval_id,
                                "tool_call_id": tool_call.id,
                                "idempotency_key": idempotency_key,
                                "approval_required": approval_required,
                                "reason": request.reason,
                                "metadata": request.metadata,
                                "resolved_by": resolved_by_payload,
                                "expired": expired,
                                "result": result.model_dump(),
                            },
                        ),
                        session=session,
                        registered_agent=registered_agent,
                        registered_environment=registered_environment,
                        tool_call=tool_call,
                        result=result,
                        task_id=pending_approval.task_id,
                    ):
                        yield event
                        if outcome is not None:
                            tool_outcomes.append(outcome)
                    continue

                async for event, outcome in self._tool_round_executor.execute_tool_call(
                    session=session,
                    registered_agent=registered_agent,
                    registered_environment=registered_environment,
                    tool_call=tool_call,
                    request_metadata=request.metadata,
                    task_id=pending_approval.task_id,
                    check_policy=False,
                    emit_started=True,
                    approval_id=pending_approval.approval_id,
                    taint_labels=call_taint_labels,
                ):
                    yield event
                    if outcome is not None:
                        tool_outcomes.append(outcome)

            tool_result_messages = transcript_helpers.tool_result_messages(tool_outcomes)
            transcript.extend(tool_result_messages)
            cleared_checkpoint = await approval_support.checkpoint_without_pending_approval(
                self._session_store,
                session.id,
            )
            await self._session_store.append_transcript_messages_and_transform_checkpoint(
                session.id,
                tool_result_messages,
                self._checkpoint_transform(cleared_checkpoint),
            )
            pending_approval_cleared = True
            yield await self._event_writer.emit(
                approval_support.cleared_event(
                    session=session,
                    agent_name=registered_agent.spec.name,
                    environment_name=environment_name,
                    approval_id=pending_approval.approval_id,
                )
            )

            session_stream = self._run_session(
                RecoverySessionRunRequest(
                    session=session,
                    registered_agent=registered_agent,
                    registered_provider=registered_provider,
                    registered_environment=registered_environment,
                    messages=transcript,
                    messages_to_append=[],
                    max_steps=effective_max_steps,
                    limits=effective_limits,
                    budget_limits=effective_budget_limits,
                    retry_policy=effective_retry_policy,
                    structured_output=_effective_approval_structured_output(
                        structured_output=request.structured_output,
                        pending_approval=pending_approval,
                    ),
                    # Restore the original run's thinking config across the approval pause
                    # (an override on the approval request itself wins).
                    thinking=_effective_approval_thinking(
                        thinking=request.thinking,
                        pending_approval=pending_approval,
                    ),
                    request_loop_policies=request.loop_policies,
                    request_metadata=request.metadata,
                    task_id=pending_approval.task_id,
                    task_worker_id=None,
                    start_event_type=None,
                    start_event_payload={},
                    start_task_on_enter=False,
                    release_run_fence_on_exit=False,
                )
            )
            try:
                async for event in self._session_control.stream_with_out_of_band_events(
                    session.id,
                    session_stream,
                ):
                    yield event
            except GeneratorExit:
                await session_stream.aclose()
                raise
        except GeneratorExit:
            await self._finalize_abandoned_session_by_id(session.id)
            raise
        except Exception as exc:
            if isinstance(exc, approval_support.ToolApprovalManualRecoveryRequired):
                session = await self._session_store.update_status(
                    session.id,
                    SessionStatus.INTERRUPTED,
                )
                async for event in self._emit_terminal_event_with_hooks(
                    RecoveryTerminalEventRequest(
                        event=Event(
                            type=EventType.SESSION_INTERRUPTED,
                            session_id=session.id,
                            agent_name=registered_agent.spec.name,
                            environment_name=environment_name,
                            payload={
                                **exception_failure_payload(exc),
                                "interruption_type": _INTERRUPTION_TYPE_TOOL_APPROVAL_REQUIRED,
                                "approval": pending_approval.model_dump(mode="json"),
                                "approval_id": pending_approval.approval_id,
                                "tool_call_id": exc.tool_call_id,
                                "tool_name": exc.tool_name,
                                "manual_recovery_required": True,
                            },
                        ),
                        phase=RuntimeHookPhase.AFTER_SESSION_INTERRUPTED,
                        session=session,
                        registered_agent=registered_agent,
                        registered_environment=registered_environment,
                    )
                ):
                    yield event
                return

            if not pending_approval_cleared:
                session = await self._session_store.update_status(
                    session.id,
                    SessionStatus.INTERRUPTED,
                )
                async for event in self._emit_terminal_event_with_hooks(
                    RecoveryTerminalEventRequest(
                        event=Event(
                            type=EventType.SESSION_INTERRUPTED,
                            session_id=session.id,
                            agent_name=registered_agent.spec.name,
                            environment_name=environment_name,
                            payload={
                                **exception_failure_payload(exc),
                                "interruption_type": _INTERRUPTION_TYPE_TOOL_APPROVAL_REQUIRED,
                                "approval": pending_approval.model_dump(mode="json"),
                            },
                        ),
                        phase=RuntimeHookPhase.AFTER_SESSION_INTERRUPTED,
                        session=session,
                        registered_agent=registered_agent,
                        registered_environment=registered_environment,
                    )
                ):
                    yield event
                return

            task_failure_error: Exception | None = None
            if pending_approval.task_id is not None and self._task_store is not None:
                try:
                    task = await self._task_store.fail_task(
                        pending_approval.task_id,
                        {
                            "message": str(exc),
                            "type": type(exc).__name__,
                            "session_id": session.id,
                            "approval_id": pending_approval.approval_id,
                        },
                    )
                    yield await self._event_writer.emit(
                        self._task_event(
                            RecoveryTaskEventRequest(
                                event_type=EventType.TASK_FAILED,
                                task=task,
                                session=session,
                                registered_agent=registered_agent,
                                registered_environment=registered_environment,
                            )
                        )
                    )
                except Exception as task_exc:
                    task_failure_error = task_exc
            session = await self._session_store.update_status(session.id, SessionStatus.FAILED)
            payload: dict[str, Any] = {
                **exception_failure_payload(exc),
                "approval_id": pending_approval.approval_id,
                "tool_call_id": pending_approval.tool_call_id,
            }
            if task_failure_error is not None:
                payload["task_update_error"] = str(task_failure_error)
                payload["task_update_error_type"] = type(task_failure_error).__name__
            async for event in self._emit_terminal_event_with_hooks(
                RecoveryTerminalEventRequest(
                    event=Event(
                        type=EventType.SESSION_FAILED,
                        session_id=session.id,
                        agent_name=registered_agent.spec.name,
                        environment_name=environment_name,
                        payload=payload,
                    ),
                    phase=RuntimeHookPhase.AFTER_SESSION_FAILED,
                    session=session,
                    registered_agent=registered_agent,
                    registered_environment=registered_environment,
                )
            ):
                yield event

    async def recover_user_input(
        self,
        *,
        request: UserInputRecoveryRequest,
        loaded_session: Session,
        session: Session,
        pending: PendingUserInput,
        pending_tool_call: PendingToolCallApproval,
        registered_agent: runtime_records.RegisteredAgentState,
        registered_provider: runtime_records.RegisteredProvider,
        registered_environment: runtime_records.RegisteredEnvironment | None,
    ) -> AsyncGenerator[Event, None]:
        recovery_prepared = False
        authoritative_failure: BaseException | None = None
        abandoned = False
        try:
            recovered_result = ToolResult(
                content=request.message,
                structured=request.structured,
                artifacts=request.artifacts,
                is_error=request.outcome == ToolApprovalRecoveryOutcome.FAILED,
            )
            event_type = (
                EventType.TOOL_CALL_FAILED
                if recovered_result.is_error
                else EventType.TOOL_CALL_COMPLETED
            )
            events = await self._session_store.load_events(session.id)
            approval_support.validate_round_recovery_target(
                events=events,
                pending_calls=pending.tool_calls,
                tool_call_id=request.tool_call_id,
                input_id=pending.input_id,
            )
            factory_started_event = await self._environment_lifecycle.emit_factory_started(
                session=session,
                registered_agent=registered_agent,
                registered_environment=registered_environment,
            )
            if factory_started_event is not None:
                yield factory_started_event
            factory_resolution = await self._environment_lifecycle.resolve_factory(
                session=session,
                registered_agent=registered_agent,
                registered_environment=registered_environment,
                started_event=factory_started_event,
                operation=EnvironmentFactoryOperation.RECONNECT,
            )
            registered_environment = factory_resolution.registered_environment
            environment_name = _environment_name(registered_environment)
            for event in factory_resolution.events:
                yield event
            if factory_resolution.error is not None:
                session = await self._session_store.update_status(
                    session.id,
                    SessionStatus.INTERRUPTED,
                )
                async for event in self._emit_terminal_event_with_hooks(
                    RecoveryTerminalEventRequest(
                        event=Event(
                            type=EventType.SESSION_INTERRUPTED,
                            session_id=session.id,
                            agent_name=registered_agent.spec.name,
                            environment_name=environment_name,
                            payload={
                                "interruption_type": _INTERRUPTION_TYPE_USER_INPUT_REQUIRED,
                                "user_input": pending.model_dump(mode="json"),
                                "error": str(factory_resolution.error),
                                "error_type": type(factory_resolution.error).__name__,
                            },
                        ),
                        phase=RuntimeHookPhase.AFTER_SESSION_INTERRUPTED,
                        session=session,
                        registered_agent=registered_agent,
                        registered_environment=registered_environment,
                    )
                ):
                    yield event
                return
            recovery_tool_event, recovered_result = tool_results.redact_tool_result_event(
                event=Event(
                    type=event_type,
                    session_id=session.id,
                    agent_name=registered_agent.spec.name,
                    environment_name=environment_name,
                    tool_name=pending_tool_call.tool_name,
                    payload={
                        "tool_call_id": pending_tool_call.tool_call_id,
                        "idempotency_key": tool_execution.tool_idempotency_key(
                            session_id=session.id,
                            tool_call_id=pending_tool_call.tool_call_id,
                            pause_id=pending.input_id,
                        ),
                        "input_id": pending.input_id,
                        "manual_recovery": True,
                        "reason": request.reason,
                        "metadata": request.metadata,
                        "resolved_by": resolution_actor_payload(request.resolved_by),
                        "result": recovered_result.model_dump(),
                    },
                ),
                result=recovered_result,
                redactor=self._secret_redactor,
            )
            recovery_events = [
                Event(
                    type=EventType.SESSION_RESUMED,
                    session_id=session.id,
                    agent_name=registered_agent.spec.name,
                    environment_name=environment_name,
                    payload={
                        "interruption_type": _INTERRUPTION_TYPE_USER_INPUT_REQUIRED,
                        "input_id": pending.input_id,
                        "tool_call_id": pending.tool_call_id,
                        "resolved_by": resolution_actor_payload(request.resolved_by),
                    },
                ),
                recovery_tool_event,
            ]
            emitted_recovery_events = await self._event_writer.emit_many(
                session.id, recovery_events
            )
            for event in emitted_recovery_events:
                yield event
            tool_call = runtime_records.ToolCallRequest(
                id=pending_tool_call.tool_call_id,
                name=pending_tool_call.tool_name,
                arguments=copy_json_value(pending_tool_call.arguments, "arguments"),
            )
            tool_event = emitted_recovery_events[-1]
            # Manual recovery persists the operator-supplied result before hooks run, so
            # after_tool_call is observe-only here (v1): the threaded modification is ignored.
            async for event, _modified in self._tool_round_executor.run_tool_call_hooks(
                session=session,
                tool_event=tool_event,
                registered_agent=registered_agent,
                registered_environment=registered_environment,
                tool_call=tool_call,
                result=recovered_result,
                task_id=pending.task_id,
                redactor=self._secret_redactor,
                allow_modification=False,
            ):
                yield event
            recovery_prepared = True
        except (GeneratorExit, asyncio.CancelledError) as exc:
            authoritative_failure = exc
            abandoned = True
            raise
        except Exception as exc:
            authoritative_failure = exc
            await self._session_store.update_status(session.id, loaded_session.status)
            raise
        finally:
            if not recovery_prepared:
                await self._cleanup_recovery_handoff(
                    stream=None,
                    session_id=session.id,
                    authoritative_failure=authoritative_failure,
                    finalize_abandoned=abandoned,
                    release_run_fence=True,
                )

        continuation_stream: AsyncGenerator[Event, None] | None = None
        authoritative_failure = None
        abandoned = False
        try:
            response = UserInputResponse(
                session_id=request.session_id,
                input_id=request.input_id,
                answer=request.answer,
                structured=request.structured,
                artifacts=request.artifacts,
                metadata=request.metadata,
                resolved_by=request.resolved_by,
                max_steps=request.max_steps,
                limits=request.limits,
                budget_limits=request.budget_limits,
                retry_policy=request.retry_policy,
                structured_output=request.structured_output,
                thinking=request.thinking,
                loop_policies=request.loop_policies,
            )
            continuation_stream = self.continue_user_input_resolution(
                response=response,
                session=session,
                pending=pending,
                registered_agent=registered_agent,
                registered_provider=registered_provider,
                registered_environment=registered_environment,
                emit_resume_event=False,
            )
            async for event in continuation_stream:
                yield event
        except BaseException as exc:
            authoritative_failure = exc
            abandoned = isinstance(exc, GeneratorExit | asyncio.CancelledError)
            raise
        finally:
            await self._cleanup_recovery_handoff(
                stream=continuation_stream,
                session_id=session.id,
                authoritative_failure=authoritative_failure,
                finalize_abandoned=abandoned,
                release_run_fence=True,
            )

    async def recover_tool_approval(
        self,
        *,
        request: ToolApprovalRecoveryRequest,
        loaded_session: Session,
        session: Session,
        pending_approval: PendingToolApproval,
        pending_tool_call: PendingToolCallApproval,
        registered_agent: runtime_records.RegisteredAgentState,
        registered_provider: runtime_records.RegisteredProvider,
        registered_environment: runtime_records.RegisteredEnvironment | None,
    ) -> AsyncGenerator[Event, None]:
        recovery_prepared = False
        authoritative_failure: BaseException | None = None
        abandoned = False
        try:
            recovered_result = approval_support.recovered_tool_result(
                request=request,
            )
            event_type = (
                EventType.TOOL_CALL_FAILED
                if recovered_result.is_error
                else EventType.TOOL_CALL_COMPLETED
            )
            # Recovery reconciles an externally executed side effect that was
            # authorized before the crash, so an expired window does not block it
            # (an expired-never-approved approval has no started tool to recover).
            # The out-of-window reconciliation is still stamped for the audit trail.
            recovered_after_expiry = approval_support.pending_approval_expired(
                pending_approval, self._clock()
            )
            events = await self._session_store.load_events(session.id)
            approval_support.validate_recovery_target(
                events=events,
                approval=pending_approval,
                tool_call_id=request.tool_call_id,
            )
            factory_started_event = await self._environment_lifecycle.emit_factory_started(
                session=session,
                registered_agent=registered_agent,
                registered_environment=registered_environment,
            )
            if factory_started_event is not None:
                yield factory_started_event
            factory_resolution = await self._environment_lifecycle.resolve_factory(
                session=session,
                registered_agent=registered_agent,
                registered_environment=registered_environment,
                started_event=factory_started_event,
                operation=EnvironmentFactoryOperation.RECONNECT,
            )
            registered_environment = factory_resolution.registered_environment
            environment_name = _environment_name(registered_environment)
            for event in factory_resolution.events:
                yield event
            if factory_resolution.error is not None:
                session = await self._session_store.update_status(
                    session.id,
                    SessionStatus.INTERRUPTED,
                )
                async for event in self._emit_terminal_event_with_hooks(
                    RecoveryTerminalEventRequest(
                        event=Event(
                            type=EventType.SESSION_INTERRUPTED,
                            session_id=session.id,
                            agent_name=registered_agent.spec.name,
                            environment_name=environment_name,
                            payload={
                                "interruption_type": _INTERRUPTION_TYPE_TOOL_APPROVAL_REQUIRED,
                                "approval": pending_approval.model_dump(mode="json"),
                                "error": str(factory_resolution.error),
                                "error_type": type(factory_resolution.error).__name__,
                                "approval_id": pending_approval.approval_id,
                            },
                        ),
                        phase=RuntimeHookPhase.AFTER_SESSION_INTERRUPTED,
                        session=session,
                        registered_agent=registered_agent,
                        registered_environment=registered_environment,
                    )
                ):
                    yield event
                return
            recovery_tool_event, recovered_result = tool_results.redact_tool_result_event(
                event=Event(
                    type=event_type,
                    session_id=session.id,
                    agent_name=registered_agent.spec.name,
                    environment_name=environment_name,
                    tool_name=pending_tool_call.tool_name,
                    payload={
                        "approval_id": pending_approval.approval_id,
                        "tool_call_id": pending_tool_call.tool_call_id,
                        "idempotency_key": tool_execution.tool_idempotency_key(
                            session_id=session.id,
                            tool_call_id=pending_tool_call.tool_call_id,
                            approval_id=pending_approval.approval_id,
                        ),
                        "manual_recovery": True,
                        "reason": request.reason,
                        "metadata": request.metadata,
                        "resolved_by": resolution_actor_payload(request.resolved_by),
                        "expired": recovered_after_expiry,
                        "result": recovered_result.model_dump(),
                    },
                ),
                result=recovered_result,
                redactor=self._secret_redactor,
            )
            recovery_events = [
                approval_support.resumed_event(
                    session=session,
                    agent_name=registered_agent.spec.name,
                    environment_name=environment_name,
                    approval=pending_approval,
                    decision=ToolApprovalDecision.APPROVE,
                    resolved_by=request.resolved_by,
                    expired=recovered_after_expiry,
                ),
                recovery_tool_event,
            ]
            emitted_recovery_events = await self._event_writer.emit_many(
                session.id, recovery_events
            )
            for event in emitted_recovery_events:
                yield event
            tool_call = runtime_records.ToolCallRequest(
                id=pending_tool_call.tool_call_id,
                name=pending_tool_call.tool_name,
                arguments=copy_json_value(pending_tool_call.arguments, "arguments"),
            )
            tool_event = emitted_recovery_events[-1]
            # Manual recovery persists the operator-supplied result before hooks run, so
            # after_tool_call is observe-only here (v1): the threaded modification is ignored.
            async for event, _modified in self._tool_round_executor.run_tool_call_hooks(
                session=session,
                tool_event=tool_event,
                registered_agent=registered_agent,
                registered_environment=registered_environment,
                tool_call=tool_call,
                result=recovered_result,
                task_id=pending_approval.task_id,
                redactor=self._secret_redactor,
                allow_modification=False,
            ):
                yield event
            recovery_prepared = True
        except (GeneratorExit, asyncio.CancelledError) as exc:
            authoritative_failure = exc
            abandoned = True
            raise
        except Exception as exc:
            authoritative_failure = exc
            await self._session_store.update_status(session.id, loaded_session.status)
            raise
        finally:
            if not recovery_prepared:
                await self._cleanup_recovery_handoff(
                    stream=None,
                    session_id=session.id,
                    authoritative_failure=authoritative_failure,
                    finalize_abandoned=abandoned,
                    release_run_fence=True,
                )

        continuation_stream: AsyncGenerator[Event, None] | None = None
        authoritative_failure = None
        abandoned = False
        try:
            approval_request = ToolApprovalRequest(
                session_id=request.session_id,
                approval_id=request.approval_id,
                decision=ToolApprovalDecision.APPROVE,
                reason=request.reason,
                metadata=request.metadata,
                resolved_by=request.resolved_by,
                max_steps=request.max_steps,
                limits=request.limits,
                budget_limits=request.budget_limits,
                retry_policy=request.retry_policy,
                structured_output=request.structured_output,
                thinking=request.thinking,
                loop_policies=request.loop_policies,
            )
            continuation_stream = self.continue_tool_approval_resolution(
                request=approval_request,
                session=session,
                pending_approval=pending_approval,
                registered_agent=registered_agent,
                registered_provider=registered_provider,
                registered_environment=registered_environment,
                emit_resume_event=False,
                enforce_expiry=False,
            )
            async for event in continuation_stream:
                yield event
        except BaseException as exc:
            authoritative_failure = exc
            abandoned = isinstance(exc, GeneratorExit | asyncio.CancelledError)
            raise
        finally:
            await self._cleanup_recovery_handoff(
                stream=continuation_stream,
                session_id=session.id,
                authoritative_failure=authoritative_failure,
                finalize_abandoned=abandoned,
                release_run_fence=True,
            )


def _effective_user_input_max_steps(
    *,
    max_steps: int | None,
    pending: PendingUserInput,
) -> int:
    # Restore the original run's max_steps on a user-input continuation; an explicit override
    # on the resolution request wins. Pending states written before run config was checkpointed
    # fall back to the historical request default.
    if type(pending) is not PendingUserInput:
        raise TypeError("Pending user input must be a PendingUserInput.")
    if max_steps is not None:
        return max_steps
    if pending.max_steps is not None:
        return pending.max_steps
    return _DEFAULT_APPROVAL_MAX_STEPS


def _effective_user_input_run_limits(
    *,
    limits: RunLimits | None,
    pending: PendingUserInput,
) -> RunLimits:
    if type(pending) is not PendingUserInput:
        raise TypeError("Pending user input must be a PendingUserInput.")
    if limits is not None:
        return copy_run_limits(limits)
    if pending.limits is not None:
        return copy_run_limits(pending.limits)
    return RunLimits()


def _effective_user_input_budget_limits(
    *,
    budget_limits: tuple[BudgetLimit, ...] | None,
    pending: PendingUserInput,
) -> tuple[BudgetLimit, ...]:
    if type(pending) is not PendingUserInput:
        raise TypeError("Pending user input must be a PendingUserInput.")
    if budget_limits is not None:
        return copy_request_budget_limits(budget_limits)
    if pending.budget_limits is not None:
        return copy_request_budget_limits(pending.budget_limits)
    return ()


def _effective_user_input_retry_policy(
    *,
    retry_policy: RetryPolicy | None,
    pending: PendingUserInput,
) -> RetryPolicy | None:
    # RetryPolicy is frozen, so the persisted reference is safe to reuse.
    if type(pending) is not PendingUserInput:
        raise TypeError("Pending user input must be a PendingUserInput.")
    if retry_policy is not None:
        return retry_policy
    return pending.retry_policy


def _effective_user_input_structured_output(
    *,
    structured_output: StructuredOutputSpec | None,
    pending: PendingUserInput,
) -> StructuredOutputSpec | None:
    # Mirror _effective_approval_structured_output: inherit the paused run's spec when the resolver
    # supplies none; adopt the resolver's spec when the run had none; a differing spec is a swap of
    # the contract fixed by the provider history and is rejected.
    if type(pending) is not PendingUserInput:
        raise TypeError("Pending user input must be a PendingUserInput.")
    if structured_output is None:
        return copy_structured_output_spec(pending.structured_output)
    if pending.structured_output is None:
        return copy_structured_output_spec(structured_output)
    if not _structured_output_specs_equal(structured_output, pending.structured_output):
        raise ValueError("structured_output does not match the paused run contract.")
    return copy_structured_output_spec(pending.structured_output)


def _effective_approval_thinking(
    *,
    thinking: ThinkingConfig | None,
    pending_approval: PendingToolApproval,
) -> ThinkingConfig | None:
    # Restore the original run's thinking config on an approval continuation; a thinking
    # override on the approval request itself takes precedence. (ThinkingConfig is frozen,
    # so the reference is safe to reuse.)
    if type(pending_approval) is not PendingToolApproval:
        raise TypeError("Pending approval must be a PendingToolApproval.")
    if thinking is not None:
        return thinking
    return pending_approval.thinking


def _effective_approval_max_steps(
    *,
    max_steps: int | None,
    pending_approval: PendingToolApproval,
) -> int:
    # Restore the original run's max_steps on an approval continuation; an explicit
    # override on the approval request wins. Approvals persisted before run config
    # was checkpointed fall back to the historical request default.
    if type(pending_approval) is not PendingToolApproval:
        raise TypeError("Pending approval must be a PendingToolApproval.")
    if max_steps is not None:
        return max_steps
    if pending_approval.max_steps is not None:
        return pending_approval.max_steps
    return _DEFAULT_APPROVAL_MAX_STEPS


def _effective_approval_run_limits(
    *,
    limits: RunLimits | None,
    pending_approval: PendingToolApproval,
) -> RunLimits:
    if type(pending_approval) is not PendingToolApproval:
        raise TypeError("Pending approval must be a PendingToolApproval.")
    if limits is not None:
        return copy_run_limits(limits)
    if pending_approval.limits is not None:
        return copy_run_limits(pending_approval.limits)
    return RunLimits()


def _effective_approval_budget_limits(
    *,
    budget_limits: tuple[BudgetLimit, ...] | None,
    pending_approval: PendingToolApproval,
) -> tuple[BudgetLimit, ...]:
    if type(pending_approval) is not PendingToolApproval:
        raise TypeError("Pending approval must be a PendingToolApproval.")
    if budget_limits is not None:
        return copy_request_budget_limits(budget_limits)
    if pending_approval.budget_limits is not None:
        return copy_request_budget_limits(pending_approval.budget_limits)
    return ()


def _effective_approval_retry_policy(
    *,
    retry_policy: RetryPolicy | None,
    pending_approval: PendingToolApproval,
) -> RetryPolicy | None:
    # RetryPolicy is frozen, so the persisted reference is safe to reuse.
    if type(pending_approval) is not PendingToolApproval:
        raise TypeError("Pending approval must be a PendingToolApproval.")
    if retry_policy is not None:
        return retry_policy
    return pending_approval.retry_policy


def _effective_approval_structured_output(
    *,
    structured_output: StructuredOutputSpec | None,
    pending_approval: PendingToolApproval,
) -> StructuredOutputSpec | None:
    if type(pending_approval) is not PendingToolApproval:
        raise TypeError("Pending approval must be a PendingToolApproval.")
    if structured_output is None:
        return copy_structured_output_spec(pending_approval.structured_output)
    if pending_approval.structured_output is None:
        return copy_structured_output_spec(structured_output)
    if not _structured_output_specs_equal(
        structured_output,
        pending_approval.structured_output,
    ):
        raise ValueError("Tool approval structured_output does not match the pending run contract.")
    return copy_structured_output_spec(pending_approval.structured_output)


def _structured_output_specs_equal(
    left: StructuredOutputSpec,
    right: StructuredOutputSpec,
) -> bool:
    if type(left) is not StructuredOutputSpec or type(right) is not StructuredOutputSpec:
        raise TypeError("Structured output comparison requires StructuredOutputSpec values.")
    return left.model_dump(mode="json") == right.model_dump(mode="json")


def _environment_name(
    registered_environment: runtime_records.RegisteredEnvironment | None,
) -> str | None:
    if registered_environment is None:
        return None
    return registered_environment.spec.name


def _has_run_budget_limit(limits: tuple[BudgetLimit, ...]) -> bool:
    return any(limit.scope == "run" for limit in limits)
