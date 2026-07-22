"""Durable runtime continuation and crash-recovery ownership.

The coordinator owns paused-round continuation, recorded tool-round repair,
incomplete-session recovery, subagent reattachment, and abandoned-stream
finalization without importing or accepting :class:`CayuApp`. Public request
validation and registry lookup remain on the application façade. Session
execution, interruption, turn accounting, and terminal hook orchestration are
narrow callbacks until the session engine is extracted.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import time
from collections.abc import AsyncGenerator, AsyncIterator, Awaitable, Callable
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any
from uuid import uuid4

from cayu._task_wait import await_shielded_task_outcome
from cayu._validation import copy_json_value, require_clean_nonblank
from cayu.core.events import Event, EventType
from cayu.core.messages import Message
from cayu.core.thinking import ThinkingConfig
from cayu.core.tools import _TOOL_POLICY_DENIAL_SOURCE, ToolResult
from cayu.environments import EnvironmentFactoryOperation
from cayu.runtime import _approval_support as approval_support
from cayu.runtime import _runtime_records as runtime_records
from cayu.runtime import _tool_execution as tool_execution
from cayu.runtime import _tool_results as tool_results
from cayu.runtime import _tool_round_recovery as tool_round_recovery
from cayu.runtime import _transcript as transcript_helpers
from cayu.runtime._environment_lifecycle import (
    EnvironmentLifecycle,
    exception_failure_payload,
)
from cayu.runtime._event_writer import RuntimeEventWriter
from cayu.runtime._interruption_coordinator import (
    _PENDING_INTERRUPTION_CASCADE_CHECKPOINT_KEY,
    _PENDING_SESSION_INTERRUPT_CHECKPOINT_KEY,
    _is_background_subagent_session,
)
from cayu.runtime._run_limits import RunLimitController, SessionUsageTracker
from cayu.runtime._session_control import ActiveSessionRun, SessionControl
from cayu.runtime._session_queries import query_all_sessions
from cayu.runtime._tool_round_executor import (
    InterruptedToolRoundRequest,
    ToolRoundExecutor,
    ordered_tool_result_messages,
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
from cayu.runtime.sessions import (
    _INCOMPLETE_RECOVERY_CLAIM_CHECKPOINT_KEY,
    CheckpointTransform,
    IncompleteSessionRecoveryAction,
    IncompleteSessionRecoveryRequest,
    IncompleteSessionRecoveryResult,
    IncompleteSessionsRecoveryRequest,
    Session,
    SessionOrder,
    SessionQuery,
    SessionRunFenced,
    SessionStatus,
    SessionStatusConflict,
    SessionStore,
    _activate_session_run_fence,
    _deactivate_session_run_fence,
    _incomplete_recovery_claim_from_checkpoint,
)
from cayu.runtime.stop_policy import RunLimits, StopDecision, copy_run_limits, has_run_limits
from cayu.runtime.structured_output import (
    StructuredOutputSpec,
    copy_structured_output_spec,
)
from cayu.runtime.tasks import Task, TaskStore
from cayu.runtime.tool_policy import ToolPolicyDecision
from cayu.runtime.tool_rounds import ToolRoundRecoveryRequest
from cayu.runtime.usage import SessionUsageSummary, session_usage_summary
from cayu.runtime.user_input import (
    PENDING_USER_INPUT_CHECKPOINT_KEY,
    PendingUserInput,
    UserInputRecoveryRequest,
    UserInputResponse,
    pending_user_input_from_checkpoint,
)
from cayu.vaults import SecretRedactor

_INTERRUPTION_TYPE_TOOL_APPROVAL_REQUIRED = "tool_approval_required"
_INTERRUPTION_TYPE_USER_INPUT_REQUIRED = "user_input_required"
_INTERRUPTION_TYPE_RUNTIME_INTERRUPTED = "runtime_interrupted"
_INTERRUPTION_TYPE_OPERATOR_REQUESTED = "operator_requested"
_DEFAULT_APPROVAL_MAX_STEPS = 16
_ABANDONED_RUN_REASON = "event_stream_closed"
_INCOMPLETE_RECOVERY_CLAIM_LEASE = timedelta(minutes=5)
_INCOMPLETE_RECOVERY_CLAIM_HEARTBEAT_INTERVAL_SECONDS = 30.0
_INCOMPLETE_RECOVERY_CLAIM_HEARTBEAT_RETRY_SECONDS = 5.0
_MANUAL_RECOVERY_INTERRUPT_POLL_INTERVAL_SECONDS = 0.25
_RECOVERY_RESUMABLE_SESSION_STATUSES = {
    SessionStatus.COMPLETED,
    SessionStatus.FAILED,
    SessionStatus.INTERRUPTED,
}
_TOOL_ROUND_RECOVERABLE_SESSION_STATUSES = {
    SessionStatus.RUNNING,
    SessionStatus.INTERRUPTING,
    SessionStatus.INTERRUPTED,
    SessionStatus.FAILED,
}

logger = logging.getLogger(__name__)

CheckpointTransformFactory = Callable[[dict[str, Any]], CheckpointTransform]
EffectiveRetryPolicy = Callable[[RetryPolicy | None], RetryPolicy]
RecoveryCleanup = Callable[[], Awaitable[None]]


def _recovery_abandonment_signal(
    error: BaseException | None,
    *,
    cancellation_baseline: int = 0,
) -> GeneratorExit | asyncio.CancelledError | None:
    """Find explicit abandonment, preferring cancellation for cleanup shielding."""
    if isinstance(error, GeneratorExit | asyncio.CancelledError):
        return error
    if isinstance(error, BaseExceptionGroup):
        try:
            task = asyncio.current_task()
        except RuntimeError:
            task = None
        cancellation_delivered = task is None or task.cancelling() > cancellation_baseline
        generator_exit: GeneratorExit | None = None
        for child in error.exceptions:
            abandonment = _recovery_abandonment_signal(
                child,
                cancellation_baseline=cancellation_baseline,
            )
            if isinstance(abandonment, asyncio.CancelledError) and cancellation_delivered:
                return abandonment
            if isinstance(abandonment, GeneratorExit) and generator_exit is None:
                generator_exit = abandonment
        return generator_exit
    return None


def _task_cancellation_count() -> int:
    """Return the current task's cancellation generation for boundary tracking."""
    task = asyncio.current_task()
    return 0 if task is None else task.cancelling()


def _prepend_exception_cause(error: BaseException, cause: BaseException) -> None:
    """Preserve a new structured cause without discarding an existing chain."""
    cause.__cause__ = error.__cause__
    error.__cause__ = cause


async def _run_recovery_cleanup_steps(
    *,
    authoritative_failure: BaseException | None,
    steps: tuple[tuple[str, RecoveryCleanup], ...],
    cancellation_baseline: int = 0,
) -> tuple[tuple[str, BaseException], ...]:
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

    abandonment = _recovery_abandonment_signal(
        authoritative_failure,
        cancellation_baseline=cancellation_baseline,
    )
    if isinstance(abandonment, asyncio.CancelledError):
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
        return ()
    if authoritative_failure is not None and not isinstance(authoritative_failure, GeneratorExit):
        failures = tuple(cleanup_failures)
        for operation, cleanup_failure in cleanup_failures:
            authoritative_failure.add_note(
                "Continuation recovery cleanup failed during "
                f"{operation}: {type(cleanup_failure).__name__}. "
                "The original failure remains authoritative."
            )
        _prepend_exception_cause(
            authoritative_failure,
            BaseExceptionGroup(
                "Continuation recovery cleanup failures",
                [failure for _operation, failure in failures],
            ),
        )
        return failures

    operation, first_failure = cleanup_failures[0]
    for later_operation, later_failure in cleanup_failures[1:]:
        first_failure.add_note(
            "Additional continuation recovery cleanup failure during "
            f"{later_operation}: {later_failure!r}."
        )
    first_failure.add_note(f"Continuation recovery cleanup failed during {operation}.")
    if len(cleanup_failures) > 1:
        _prepend_exception_cause(
            first_failure,
            BaseExceptionGroup(
                "Additional continuation recovery cleanup failures",
                [failure for _operation, failure in cleanup_failures[1:]],
            ),
        )
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


@dataclass(frozen=True)
class RecoveryInterruptionRequest:
    session: Session
    registered_agent: runtime_records.RegisteredAgentState
    registered_environment: runtime_records.RegisteredEnvironment | None
    environment_name: str | None


@dataclass(frozen=True)
class RecoveryAbandonedTurnRequest:
    session: Session
    registered_agent: runtime_records.RegisteredAgentState
    environment_name: str | None
    run_started_at: float
    usage_tracker: SessionUsageTracker
    active_run: ActiveSessionRun[SessionUsageTracker] | None


@dataclass(frozen=True)
class RecoveryAbandonedSessionRequest:
    session: Session
    registered_agent: runtime_records.RegisteredAgentState
    registered_environment: runtime_records.RegisteredEnvironment | None
    environment_name: str | None
    run_started_at: float | None = None
    turn_usage_tracker: SessionUsageTracker | None = None
    active_run: ActiveSessionRun[SessionUsageTracker] | None = None


@dataclass(frozen=True)
class _IncompleteRecoveryClaim:
    claim_id: str
    claim_expires_at: datetime
    session_before_fence: Session
    session: Session


class _IncompleteRecoveryClaimLost(RuntimeError):
    """The durable incomplete-session recovery lease is no longer owned."""


class _ManualRecoveryInterrupted(RuntimeError):
    """A durable interruption won before manual recovery could claim the session."""


class _ManualRecoveryCascadePending(RuntimeError):
    """Descendant interruption must finish before manual recovery can continue."""


@dataclass(frozen=True)
class _ManualRecoveryInterruptionFence:
    session: Session
    claim_id: str
    error: BaseException | None


@dataclass(frozen=True)
class _ManualRecoveryEventDelivery:
    event: Event
    consumed: asyncio.Event


@dataclass(frozen=True)
class _ManualRecoveryStreamOutcome:
    error: BaseException | None


@dataclass(frozen=True)
class _ManualRecoverySupervisorResult:
    error: BaseException | None
    cleanup_failure: BaseException | None


@dataclass(frozen=True)
class _ManualRecoveryPersistenceReconciliation:
    persisted: bool | None
    error: Exception | None = None
    cancellation: asyncio.CancelledError | None = None


RunSession = Callable[[RecoverySessionRunRequest], AsyncGenerator[Event, None]]
TerminalEventStream = Callable[[RecoveryTerminalEventRequest], AsyncIterator[Event]]
LimitStopEventStream = Callable[[RecoveryLimitStopRequest], AsyncIterator[Event]]
TaskEventFactory = Callable[[RecoveryTaskEventRequest], Event]
RegisteredAgentResolver = Callable[[str], runtime_records.RegisteredAgentState]
RegisteredEnvironmentResolver = Callable[[str | None], runtime_records.RegisteredEnvironment | None]
RecoveryInterruptionStream = Callable[[RecoveryInterruptionRequest], AsyncIterator[Event]]
PendingSessionInterruptCheckpoint = Callable[[dict[str, Any], datetime], CheckpointTransform]
AbandonedTurnCompleted = Callable[[RecoveryAbandonedTurnRequest], Awaitable[Event]]


class RecoveryCoordinator:
    """Continue paused work and repair incomplete sessions from durable state."""

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
        stop_session_for_limit_reached: LimitStopEventStream,
        task_event: TaskEventFactory,
        resolve_registered_agent: RegisteredAgentResolver,
        resolve_registered_environment: RegisteredEnvironmentResolver,
        interrupt_session_for_recovery: RecoveryInterruptionStream,
        pending_session_interrupt_checkpoint: PendingSessionInterruptCheckpoint,
        abandoned_turn_completed: AbandonedTurnCompleted,
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
        self._stop_session_for_limit_reached = stop_session_for_limit_reached
        self._task_event = task_event
        self._resolve_registered_agent = resolve_registered_agent
        self._resolve_registered_environment = resolve_registered_environment
        self._interrupt_session_for_recovery = interrupt_session_for_recovery
        self._pending_session_interrupt_checkpoint = pending_session_interrupt_checkpoint
        self._abandoned_turn_completed = abandoned_turn_completed

    async def _cleanup_recovery_handoff(
        self,
        *,
        stream: AsyncGenerator[Event, None] | None,
        session_id: str,
        authoritative_failure: BaseException | None,
        finalize_abandoned: bool,
        release_run_fence: bool,
        abort_environment_setup: bool = True,
    ) -> None:
        cleanup_steps: list[tuple[str, RecoveryCleanup]] = []
        if stream is not None:
            cleanup_steps.append(("nested stream close", stream.aclose))
        if finalize_abandoned:
            cleanup_steps.append(
                (
                    "abandoned session finalization",
                    lambda: self.finalize_abandoned_session_by_id(session_id),
                )
            )
        if abort_environment_setup and authoritative_failure is not None:
            cleanup_steps.append(
                (
                    "environment setup abort",
                    lambda: self._environment_lifecycle.abort_environment_setup(
                        session_id=session_id,
                        original_error=authoritative_failure,
                    ),
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
                                "input_id": pending.input_id,
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
            await self.finalize_abandoned_session_by_id(session.id)
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

    async def _interrupt_for_resumable_manual_recovery(
        self,
        *,
        session: Session,
        registered_agent: runtime_records.RegisteredAgentState,
        registered_environment: runtime_records.RegisteredEnvironment | None,
        payload: dict[str, Any],
    ) -> AsyncGenerator[Event, None]:
        """Close a durable or acknowledgement-ambiguous recovery to resumable state."""
        try:
            interrupted = await self._session_store.transition_status(
                session.id,
                from_statuses={SessionStatus.RUNNING},
                to_status=SessionStatus.INTERRUPTED,
            )
        except SessionStatusConflict:
            current = await self._require_session(session.id)
            if current.status not in {SessionStatus.INTERRUPTING, SessionStatus.INTERRUPTED}:
                raise
            # An operator interruption won the status transition. Finalize its
            # durable request so its identity, reason, and cascade are preserved.
            async for event in self._interrupt_session_for_recovery(
                RecoveryInterruptionRequest(
                    session=current,
                    registered_agent=registered_agent,
                    registered_environment=registered_environment,
                    environment_name=_environment_name(registered_environment),
                )
            ):
                yield event
            return
        async for event in self._emit_terminal_event_with_hooks(
            RecoveryTerminalEventRequest(
                event=Event(
                    type=EventType.SESSION_INTERRUPTED,
                    session_id=interrupted.id,
                    agent_name=registered_agent.spec.name,
                    environment_name=_environment_name(registered_environment),
                    payload=payload,
                ),
                phase=RuntimeHookPhase.AFTER_SESSION_INTERRUPTED,
                session=interrupted,
                registered_agent=registered_agent,
                registered_environment=registered_environment,
            )
        ):
            yield event

    async def _reconcile_manual_recovery_persistence(
        self,
        event: Event,
    ) -> _ManualRecoveryPersistenceReconciliation:
        """Classify an append failure using the preassigned durable event id."""
        outcome = await await_shielded_task_outcome(
            asyncio.create_task(self._event_writer.is_persisted(event))
        )
        if outcome.error is None:
            return _ManualRecoveryPersistenceReconciliation(
                persisted=bool(outcome.result),
                cancellation=outcome.cancellation,
            )
        if isinstance(outcome.error, asyncio.CancelledError):
            return _ManualRecoveryPersistenceReconciliation(
                persisted=None,
                cancellation=outcome.cancellation or outcome.error,
            )
        if not isinstance(outcome.error, Exception):
            raise outcome.error
        return _ManualRecoveryPersistenceReconciliation(
            persisted=None,
            error=outcome.error,
            cancellation=outcome.cancellation,
        )

    async def fence_expired_incomplete_recovery_claim(
        self,
        *,
        session: Session,
        claim_id: str,
    ) -> bool:
        """Fence and clear one observed expired recovery owner.

        The claim id makes the takeover conditional: a concurrent heartbeat or
        claimant that changes ownership causes this operation to leave the
        session untouched.
        """
        claim: _IncompleteRecoveryClaim | None = None
        authoritative_failure: BaseException | None = None
        try:
            claim = await self._claim_incomplete_recovery(
                session=session,
                inactive_before=None,
                required_expired_claim_id=claim_id,
            )
            return claim is not None
        except BaseException as exc:
            authoritative_failure = exc
            raise
        finally:
            if claim is not None:
                await self._cleanup_incomplete_recovery_claim(
                    session_id=session.id,
                    claim_id=claim.claim_id,
                    authoritative_failure=authoritative_failure,
                )

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
        recovery_persisted = False
        cancellation_baseline = _task_cancellation_count()
        recovery_event_to_reconcile: Event | None = None
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
            recovery_event_to_reconcile = recovery_tool_event
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
            emitted_recovery_events = await self._event_writer.persist_many(
                session.id, recovery_events
            )
            recovery_persisted = True
            await self._event_writer.fan_out_persisted(emitted_recovery_events)
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
            reconciliation_error: Exception | None = None
            if not recovery_persisted and recovery_event_to_reconcile is not None:
                try:
                    reconciliation = await self._reconcile_manual_recovery_persistence(
                        recovery_event_to_reconcile
                    )
                except BaseException as reconciliation_failure:
                    authoritative_failure = reconciliation_failure
                    abandoned = (
                        _recovery_abandonment_signal(
                            reconciliation_failure,
                            cancellation_baseline=cancellation_baseline,
                        )
                        is not None
                    )
                    raise
                if reconciliation.cancellation is not None:
                    reconciliation.cancellation.add_note(
                        "Manual user-input recovery append failed while persistence "
                        "reconciliation was running."
                    )
                    authoritative_failure = reconciliation.cancellation
                    abandoned = True
                    raise reconciliation.cancellation from exc
                recovery_persisted = reconciliation.persisted is True
                reconciliation_error = reconciliation.error
            if recovery_persisted or reconciliation_error is not None:
                persistence_payload = (
                    {"manual_recovery_persisted": True}
                    if recovery_persisted
                    else {
                        "manual_recovery_persistence_unknown": True,
                        "persistence_reconciliation_error_type": type(
                            reconciliation_error
                        ).__name__,
                    }
                )
                try:
                    async for event in self._interrupt_for_resumable_manual_recovery(
                        session=session,
                        registered_agent=registered_agent,
                        registered_environment=registered_environment,
                        payload={
                            "interruption_type": _INTERRUPTION_TYPE_USER_INPUT_REQUIRED,
                            "user_input": pending.model_dump(mode="json"),
                            "input_id": pending.input_id,
                            "tool_call_id": pending_tool_call.tool_call_id,
                            **persistence_payload,
                            "error": str(exc),
                            "error_type": type(exc).__name__,
                        },
                    ):
                        yield event
                except BaseException as interruption_failure:
                    authoritative_failure = interruption_failure
                    abandoned = (
                        _recovery_abandonment_signal(
                            interruption_failure,
                            cancellation_baseline=cancellation_baseline,
                        )
                        is not None
                    )
                    raise
                # The original failure is now represented by durable interrupted
                # state. It must not suppress a later fence-release failure.
                authoritative_failure = None
                return
            await self._session_store.update_status(session.id, loaded_session.status)
            raise
        except BaseExceptionGroup as exc:
            authoritative_failure = exc
            abandoned = (
                _recovery_abandonment_signal(
                    exc,
                    cancellation_baseline=cancellation_baseline,
                )
                is not None
            )
            if recovery_persisted and not abandoned:
                async for event in self._interrupt_session_for_recovery(
                    RecoveryInterruptionRequest(
                        session=session,
                        registered_agent=registered_agent,
                        registered_environment=registered_environment,
                        environment_name=_environment_name(registered_environment),
                    )
                ):
                    yield event
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
            abandoned = (
                _recovery_abandonment_signal(
                    exc,
                    cancellation_baseline=cancellation_baseline,
                )
                is not None
            )
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
        recovery_persisted = False
        cancellation_baseline = _task_cancellation_count()
        recovery_event_to_reconcile: Event | None = None
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
            recovery_event_to_reconcile = recovery_tool_event
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
            emitted_recovery_events = await self._event_writer.persist_many(
                session.id, recovery_events
            )
            recovery_persisted = True
            await self._event_writer.fan_out_persisted(emitted_recovery_events)
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
            reconciliation_error: Exception | None = None
            if not recovery_persisted and recovery_event_to_reconcile is not None:
                try:
                    reconciliation = await self._reconcile_manual_recovery_persistence(
                        recovery_event_to_reconcile
                    )
                except BaseException as reconciliation_failure:
                    authoritative_failure = reconciliation_failure
                    abandoned = (
                        _recovery_abandonment_signal(
                            reconciliation_failure,
                            cancellation_baseline=cancellation_baseline,
                        )
                        is not None
                    )
                    raise
                if reconciliation.cancellation is not None:
                    reconciliation.cancellation.add_note(
                        "Manual tool-approval recovery append failed while persistence "
                        "reconciliation was running."
                    )
                    authoritative_failure = reconciliation.cancellation
                    abandoned = True
                    raise reconciliation.cancellation from exc
                recovery_persisted = reconciliation.persisted is True
                reconciliation_error = reconciliation.error
            if recovery_persisted or reconciliation_error is not None:
                persistence_payload = (
                    {"manual_recovery_persisted": True}
                    if recovery_persisted
                    else {
                        "manual_recovery_persistence_unknown": True,
                        "persistence_reconciliation_error_type": type(
                            reconciliation_error
                        ).__name__,
                    }
                )
                try:
                    async for event in self._interrupt_for_resumable_manual_recovery(
                        session=session,
                        registered_agent=registered_agent,
                        registered_environment=registered_environment,
                        payload={
                            "interruption_type": _INTERRUPTION_TYPE_TOOL_APPROVAL_REQUIRED,
                            "approval": pending_approval.model_dump(mode="json"),
                            "approval_id": pending_approval.approval_id,
                            "tool_call_id": pending_tool_call.tool_call_id,
                            **persistence_payload,
                            "error": str(exc),
                            "error_type": type(exc).__name__,
                        },
                    ):
                        yield event
                except BaseException as interruption_failure:
                    authoritative_failure = interruption_failure
                    abandoned = (
                        _recovery_abandonment_signal(
                            interruption_failure,
                            cancellation_baseline=cancellation_baseline,
                        )
                        is not None
                    )
                    raise
                # The original failure is now represented by durable interrupted
                # state. It must not suppress a later fence-release failure.
                authoritative_failure = None
                return
            await self._session_store.update_status(session.id, loaded_session.status)
            raise
        except BaseExceptionGroup as exc:
            authoritative_failure = exc
            abandoned = (
                _recovery_abandonment_signal(
                    exc,
                    cancellation_baseline=cancellation_baseline,
                )
                is not None
            )
            if recovery_persisted and not abandoned:
                async for event in self._interrupt_session_for_recovery(
                    RecoveryInterruptionRequest(
                        session=session,
                        registered_agent=registered_agent,
                        registered_environment=registered_environment,
                        environment_name=_environment_name(registered_environment),
                    )
                ):
                    yield event
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
            abandoned = (
                _recovery_abandonment_signal(
                    exc,
                    cancellation_baseline=cancellation_baseline,
                )
                is not None
            )
            raise
        finally:
            await self._cleanup_recovery_handoff(
                stream=continuation_stream,
                session_id=session.id,
                authoritative_failure=authoritative_failure,
                finalize_abandoned=abandoned,
                release_run_fence=True,
            )

    async def _claim_manual_tool_round_recovery(
        self,
        *,
        session: Session,
        pending_round: tool_round_recovery.PendingToolRound,
        pending_tool_call: PendingToolCallApproval,
    ) -> _IncompleteRecoveryClaim | _ManualRecoveryInterruptionFence:
        """Claim recovery or fence an operator interruption that won the race."""
        claim_id = str(uuid4())
        claim_expires_at: datetime | None = None
        claim_run_epoch: int | None = None
        session_before_fence: Session | None = None

        def require_matching_pending_call(checkpoint: dict[str, Any] | None) -> None:
            current_round = tool_round_recovery.pending_tool_round_from_checkpoint(checkpoint)
            if current_round is None:
                raise RuntimeError("Session has no pending tool round.")
            if current_round.round_id != pending_round.round_id:
                raise RuntimeError("Pending tool round changed before recovery claimed it.")
            if current_round != pending_round:
                raise RuntimeError("Pending tool round changed before recovery claimed it.")
            current_tool_call = approval_support.round_tool_call_for_recovery(
                pending_calls=current_round.tool_calls,
                tool_call_id=pending_tool_call.tool_call_id,
            )
            if current_tool_call != pending_tool_call:
                raise RuntimeError("Pending tool call changed before recovery claimed it.")

        def claim_checkpoint(
            current_session: Session,
            checkpoint: dict[str, Any] | None,
        ) -> dict[str, Any]:
            nonlocal claim_expires_at, claim_run_epoch, session_before_fence
            claimed_at = self._clock()
            _require_aware_datetime(claimed_at, "manual recovery claim clock")
            pending_operator_interruption = (
                checkpoint is not None and _PENDING_SESSION_INTERRUPT_CHECKPOINT_KEY in checkpoint
            )
            interruption_advanced = (
                pending_operator_interruption
                or current_session.status == SessionStatus.INTERRUPTING
                or (
                    current_session.status == SessionStatus.INTERRUPTED
                    and (
                        session.status != SessionStatus.INTERRUPTED
                        or current_session.run_epoch != session.run_epoch
                    )
                )
            )
            if interruption_advanced:
                raise _ManualRecoveryInterrupted(
                    "Session interruption became durable before manual recovery claimed it."
                )
            if (
                checkpoint is not None
                and _PENDING_INTERRUPTION_CASCADE_CHECKPOINT_KEY in checkpoint
            ):
                raise _ManualRecoveryCascadePending(
                    "Session has an incomplete background interruption cascade."
                )
            checkpoint = _checkpoint_without_active_incomplete_recovery_claim(
                checkpoint,
                now=claimed_at,
            )
            require_matching_pending_call(checkpoint)
            claim_expires_at = claimed_at + _INCOMPLETE_RECOVERY_CLAIM_LEASE
            claim_run_epoch = current_session.run_epoch + 1
            session_before_fence = current_session.model_copy(deep=True)
            updated = {} if checkpoint is None else copy_json_value(checkpoint, "checkpoint")
            updated[_INCOMPLETE_RECOVERY_CLAIM_CHECKPOINT_KEY] = {
                "version": 1,
                "claim_id": claim_id,
                "claimed_at": claimed_at.isoformat(),
                "claim_expires_at": claim_expires_at.isoformat(),
                "operation": "manual_tool_round_recovery",
                "tool_round_id": pending_round.round_id,
                "tool_call_id": pending_tool_call.tool_call_id,
            }
            return updated

        transition_task = asyncio.create_task(
            self._session_store.transition_status_and_checkpoint(
                session.id,
                from_statuses=_TOOL_ROUND_RECOVERABLE_SESSION_STATUSES,
                to_status=SessionStatus.RUNNING,
                checkpoint_transform=claim_checkpoint,
            )
        )
        outcome = await await_shielded_task_outcome(transition_task)

        if outcome.error is not None:
            if isinstance(outcome.error, _ManualRecoveryCascadePending):
                if outcome.cancellation is not None:
                    outcome.cancellation.add_note(
                        "Manual tool-round recovery was blocked by an incomplete "
                        "background interruption cascade."
                    )
                    raise outcome.cancellation from outcome.error
                raise outcome.error
            if isinstance(outcome.error, _ManualRecoveryInterrupted):

                def fence_interruption(
                    _current_session: Session,
                    checkpoint: dict[str, Any] | None,
                ) -> dict[str, Any]:
                    nonlocal claim_expires_at, claim_run_epoch
                    require_matching_pending_call(checkpoint)
                    claimed_at = self._clock()
                    _require_aware_datetime(claimed_at, "manual recovery fence clock")
                    claim_expires_at = claimed_at + _INCOMPLETE_RECOVERY_CLAIM_LEASE
                    claim_run_epoch = _current_session.run_epoch + 1
                    updated = (
                        {} if checkpoint is None else copy_json_value(checkpoint, "checkpoint")
                    )
                    updated[_INCOMPLETE_RECOVERY_CLAIM_CHECKPOINT_KEY] = {
                        "version": 1,
                        "claim_id": claim_id,
                        "claimed_at": claimed_at.isoformat(),
                        "claim_expires_at": claim_expires_at.isoformat(),
                        "operation": "manual_tool_round_interruption_fence",
                        "tool_round_id": pending_round.round_id,
                        "tool_call_id": pending_tool_call.tool_call_id,
                    }
                    return updated

                fence_outcome = await await_shielded_task_outcome(
                    asyncio.create_task(
                        self._session_store.fence_run_and_transform_checkpoint(
                            session.id,
                            statuses={
                                SessionStatus.INTERRUPTING,
                                SessionStatus.INTERRUPTED,
                            },
                            checkpoint_transform=fence_interruption,
                        )
                    ),
                    cancellation=outcome.cancellation,
                )
                interruption_error: BaseException | None = fence_outcome.cancellation
                if fence_outcome.error is not None:
                    interruption_error = fence_outcome.cancellation or fence_outcome.error
                    reconciliation_outcome = await await_shielded_task_outcome(
                        asyncio.create_task(
                            self._load_owned_incomplete_recovery_claim(
                                session.id,
                                claim_id,
                                expected_run_epoch=claim_run_epoch,
                            )
                        ),
                        cancellation=fence_outcome.cancellation,
                    )
                    reconciliation_cancellation = reconciliation_outcome.cancellation
                    reconciliation_failure = reconciliation_outcome.error
                    if reconciliation_failure is not None:
                        if not isinstance(
                            reconciliation_failure,
                            Exception | asyncio.CancelledError,
                        ):
                            raise reconciliation_failure from fence_outcome.error
                        interruption_error.add_note(
                            "Could not reconcile whether the interrupted manual recovery "
                            "fence committed: "
                            f"{type(reconciliation_failure).__name__}."
                        )
                        if reconciliation_cancellation is not None:
                            reconciliation_cancellation.add_note(
                                "Interrupted manual recovery fence transition also failed: "
                                f"{type(fence_outcome.error).__name__}."
                            )
                            raise reconciliation_cancellation from fence_outcome.error
                        raise fence_outcome.error
                    elif reconciliation_outcome.result is not None:
                        fenced_session = reconciliation_outcome.result
                        if reconciliation_cancellation is not None:
                            interruption_error = reconciliation_cancellation
                            interruption_error.add_note(
                                "Interrupted manual recovery fence transition also failed: "
                                f"{type(fence_outcome.error).__name__}."
                            )
                    else:
                        if reconciliation_cancellation is not None:
                            reconciliation_cancellation.add_note(
                                "Interrupted manual recovery fence transition also failed: "
                                f"{type(fence_outcome.error).__name__}."
                            )
                            raise reconciliation_cancellation from fence_outcome.error
                        raise fence_outcome.error
                else:
                    fenced_session = fence_outcome.result
                if fenced_session is None:
                    raise RuntimeError("Interrupted manual recovery fence returned no session.")
                if claim_expires_at is None or claim_run_epoch is None:
                    raise RuntimeError("Interrupted manual recovery fence persisted no claim.")
                if fenced_session.run_epoch != claim_run_epoch:
                    raise RuntimeError(
                        "Interrupted manual recovery fence returned an unexpected run epoch."
                    )
                _activate_session_run_fence(fenced_session)
                return _ManualRecoveryInterruptionFence(
                    session=fenced_session,
                    claim_id=claim_id,
                    error=interruption_error,
                )

            reconciliation_outcome = await await_shielded_task_outcome(
                asyncio.create_task(
                    self._load_owned_incomplete_recovery_claim(
                        session.id,
                        claim_id,
                        expected_run_epoch=claim_run_epoch,
                    )
                ),
                cancellation=outcome.cancellation,
            )
            reconciliation_cancellation = reconciliation_outcome.cancellation
            reconciliation_failure = reconciliation_outcome.error
            if reconciliation_cancellation is None and isinstance(
                reconciliation_failure,
                asyncio.CancelledError,
            ):
                reconciliation_cancellation = reconciliation_failure
            authoritative_failure = reconciliation_cancellation or outcome.error
            if reconciliation_failure is not None:
                if not isinstance(reconciliation_failure, Exception | asyncio.CancelledError):
                    raise reconciliation_failure from outcome.error
                authoritative_failure.add_note(
                    "Could not reconcile whether the manual tool-round recovery claim "
                    f"committed: {type(reconciliation_failure).__name__}."
                )
            elif reconciliation_outcome.result is not None:
                reconciled_session = reconciliation_outcome.result
                _activate_session_run_fence(reconciled_session)
                await _run_recovery_cleanup_steps(
                    authoritative_failure=authoritative_failure,
                    steps=(
                        (
                            "ambiguous manual recovery claim finalization",
                            lambda: self.finalize_abandoned_session_by_id(reconciled_session.id),
                        ),
                        (
                            "ambiguous manual recovery claim cleanup",
                            lambda: self._cleanup_incomplete_recovery_claim(
                                session_id=reconciled_session.id,
                                claim_id=claim_id,
                                authoritative_failure=authoritative_failure,
                            ),
                        ),
                    ),
                )
            if reconciliation_cancellation is not None:
                reconciliation_cancellation.add_note(
                    "Manual tool-round recovery claim transition also failed: "
                    f"{type(outcome.error).__name__}."
                )
                raise reconciliation_cancellation from outcome.error
            raise outcome.error
        claimed_session = outcome.result
        if claimed_session is None:
            raise RuntimeError("Manual tool-round recovery claim returned no session.")

        # The durable transition ran in a shielded child task. Bind its epoch to
        # the caller that will perform recovery writes and eventual cleanup.
        _activate_session_run_fence(claimed_session)
        if (
            claim_expires_at is None
            or claim_run_epoch is None
            or session_before_fence is None
            or claimed_session.run_epoch != claim_run_epoch
        ):
            invariant_failure = RuntimeError(
                "Manual tool-round recovery transition did not persist its claim."
            )
            await _run_recovery_cleanup_steps(
                authoritative_failure=invariant_failure,
                steps=(
                    (
                        "abandoned manual recovery finalization",
                        lambda: self.finalize_abandoned_session_by_id(claimed_session.id),
                    ),
                    (
                        "manual recovery claim cleanup",
                        lambda: self._cleanup_incomplete_recovery_claim(
                            session_id=claimed_session.id,
                            claim_id=claim_id,
                            authoritative_failure=invariant_failure,
                        ),
                    ),
                ),
            )
            raise invariant_failure

        claim = _IncompleteRecoveryClaim(
            claim_id=claim_id,
            claim_expires_at=claim_expires_at,
            session_before_fence=session_before_fence,
            session=claimed_session,
        )
        if outcome.cancellation is None:
            return claim

        await _run_recovery_cleanup_steps(
            authoritative_failure=outcome.cancellation,
            steps=(
                (
                    "abandoned manual recovery finalization",
                    lambda: self.finalize_abandoned_session_by_id(claimed_session.id),
                ),
                (
                    "manual recovery claim cleanup",
                    lambda: self._cleanup_incomplete_recovery_claim(
                        session_id=claimed_session.id,
                        claim_id=claim.claim_id,
                        authoritative_failure=outcome.cancellation,
                    ),
                ),
            ),
        )
        raise outcome.cancellation

    async def recover_tool_round(
        self,
        *,
        request: ToolRoundRecoveryRequest,
        loaded_session: Session,
        pending_round: tool_round_recovery.PendingToolRound,
        pending_tool_call: PendingToolCallApproval,
        registered_agent: runtime_records.RegisteredAgentState,
        registered_provider: runtime_records.RegisteredProvider,
        registered_environment: runtime_records.RegisteredEnvironment | None,
        effective_structured_output: StructuredOutputSpec | None,
    ) -> AsyncGenerator[Event, None]:
        """Claim one manual recovery durably and stream its owned continuation."""
        caller_runtime_task = asyncio.current_task()
        interrupted_baseline = await self._session_control.latest_interrupted_event(
            loaded_session.id
        )
        interrupted_baseline_id = None if interrupted_baseline is None else interrupted_baseline.id
        claim = await self._claim_manual_tool_round_recovery(
            session=loaded_session,
            pending_round=pending_round,
            pending_tool_call=pending_tool_call,
        )
        if isinstance(claim, _ManualRecoveryInterruptionFence):
            authoritative_failure = claim.error
            interruption_request = RecoveryInterruptionRequest(
                session=claim.session,
                registered_agent=registered_agent,
                registered_environment=registered_environment,
                environment_name=_environment_name(registered_environment),
            )

            async def finalize_owned_interruption() -> None:
                async for _ in self._interrupt_session_for_recovery(interruption_request):
                    pass

            try:
                if claim.error is not None:
                    # Reconciliation proved this exact fence committed. Finish
                    # its operator interruption directly even when the session
                    # was already INTERRUPTED; the generic abandoned-session
                    # finalizer deliberately ignores terminal statuses.
                    await _run_recovery_cleanup_steps(
                        authoritative_failure=claim.error,
                        steps=(
                            (
                                "interrupted manual recovery finalization",
                                finalize_owned_interruption,
                            ),
                        ),
                    )
                    raise claim.error
                async for event in self._interrupt_session_for_recovery(interruption_request):
                    yield event
            except BaseException as exc:
                authoritative_failure = exc
                raise
            finally:
                await _run_recovery_cleanup_steps(
                    authoritative_failure=authoritative_failure,
                    steps=(
                        (
                            "interrupted manual recovery claim release",
                            lambda: self._cleanup_incomplete_recovery_claim(
                                session_id=claim.session.id,
                                claim_id=claim.claim_id,
                                authoritative_failure=authoritative_failure,
                            ),
                        ),
                    ),
                )
            return
        stop_heartbeat = asyncio.Event()
        heartbeat_task = asyncio.create_task(
            self._heartbeat_incomplete_recovery_claim(
                session_id=claim.session.id,
                claim_id=claim.claim_id,
                claim_expires_at=claim.claim_expires_at,
                stop=stop_heartbeat,
            )
        )
        stop_interruption_watch = asyncio.Event()
        interruption_watch_task = asyncio.create_task(
            self._watch_manual_recovery_interruption(
                session_id=claim.session.id,
                interrupted_baseline_id=interrupted_baseline_id,
                stop=stop_interruption_watch,
            )
        )
        recovery_stream = self._recover_tool_round_claimed(
            request=request,
            loaded_session=claim.session_before_fence,
            session=claim.session,
            pending_round=pending_round,
            pending_tool_call=pending_tool_call,
            registered_agent=registered_agent,
            registered_provider=registered_provider,
            registered_environment=registered_environment,
            effective_structured_output=effective_structured_output,
        )
        deliveries: asyncio.Queue[_ManualRecoveryEventDelivery | _ManualRecoveryStreamOutcome] = (
            asyncio.Queue(maxsize=2)
        )
        consumer_stopped = asyncio.Event()
        supervisor_started = asyncio.Event()
        consumer_stop_failure: BaseException | None = None

        def heartbeat_failure() -> BaseException | None:
            if not heartbeat_task.done():
                return None
            if heartbeat_task.cancelled():
                return _IncompleteRecoveryClaimLost(
                    "Manual tool-round recovery claim heartbeat was cancelled unexpectedly."
                )
            failure = heartbeat_task.exception()
            if failure is not None:
                return failure
            return _IncompleteRecoveryClaimLost(
                "Manual tool-round recovery claim heartbeat stopped unexpectedly."
            )

        def interruption_watch_failure() -> BaseException | None:
            if not interruption_watch_task.done():
                return None
            if interruption_watch_task.cancelled():
                return RuntimeError(
                    "Manual tool-round recovery interruption watcher was cancelled unexpectedly."
                )
            failure = interruption_watch_task.exception()
            if failure is not None:
                return failure
            if interruption_watch_task.result():
                return asyncio.CancelledError(
                    "Manual tool-round recovery was interrupted by a durable request."
                )
            return RuntimeError(
                "Manual tool-round recovery interruption watcher stopped unexpectedly."
            )

        async def stop_claim_heartbeat() -> None:
            stop_heartbeat.set()
            await heartbeat_task

        async def stop_interruption_watcher() -> None:
            stop_interruption_watch.set()
            await interruption_watch_task

        async def forward_recovery_events() -> None:
            async for event in recovery_stream:
                delivery = _ManualRecoveryEventDelivery(
                    event=event,
                    consumed=asyncio.Event(),
                )
                await deliveries.put(delivery)
                await delivery.consumed.wait()
                if consumer_stopped.is_set():
                    raise asyncio.CancelledError

        async def supervise_recovery() -> _ManualRecoverySupervisorResult:
            recovery_task = asyncio.create_task(forward_recovery_events())
            supervisor_runtime_task = asyncio.current_task()
            authoritative_failure: BaseException | None = None
            cleanup_failure: BaseException | None = None
            if supervisor_runtime_task is not None:
                self._session_control.register_active_control_task(
                    claim.session.id,
                    supervisor_runtime_task,
                )
            supervisor_started.set()

            async def stop_recovery_worker() -> None:
                if not recovery_task.done():
                    recovery_task.cancel()
                await asyncio.gather(recovery_task, return_exceptions=True)

            try:
                done, _pending = await asyncio.wait(
                    {recovery_task, heartbeat_task, interruption_watch_task},
                    return_when=asyncio.FIRST_COMPLETED,
                )
                if interruption_watch_task in done:
                    failure = interruption_watch_failure()
                    if failure is None:  # pragma: no cover - task completion is exhaustive.
                        raise AssertionError(
                            "Recovery interruption watcher completed without an outcome."
                        )
                    raise failure
                if heartbeat_task in done:
                    failure = heartbeat_failure()
                    if failure is None:  # pragma: no cover - task completion is exhaustive.
                        raise AssertionError("Recovery heartbeat completed without an outcome.")
                    raise failure
                recovery_task.result()
            except BaseException as exc:
                authoritative_failure = (
                    consumer_stop_failure
                    if (
                        consumer_stopped.is_set()
                        and consumer_stop_failure is not None
                        and isinstance(exc, asyncio.CancelledError)
                    )
                    else exc
                )

            try:
                try:
                    cleanup_failures = await _run_recovery_cleanup_steps(
                        authoritative_failure=authoritative_failure,
                        steps=(
                            ("manual tool-round recovery event worker stop", stop_recovery_worker),
                            (
                                "manual tool-round recovery interruption watcher stop",
                                stop_interruption_watcher,
                            ),
                            (
                                "manual tool-round recovery handoff cleanup",
                                lambda: self._cleanup_recovery_handoff(
                                    stream=recovery_stream,
                                    session_id=claim.session.id,
                                    authoritative_failure=authoritative_failure,
                                    finalize_abandoned=authoritative_failure is not None,
                                    release_run_fence=False,
                                ),
                            ),
                            ("manual tool-round recovery heartbeat stop", stop_claim_heartbeat),
                            (
                                "manual tool-round recovery claim release",
                                lambda: self._cleanup_incomplete_recovery_claim(
                                    session_id=claim.session.id,
                                    claim_id=claim.claim_id,
                                    authoritative_failure=authoritative_failure,
                                ),
                            ),
                        ),
                    )
                    if cleanup_failures:
                        cleanup_failure = BaseExceptionGroup(
                            "Manual recovery cleanup failed",
                            [failure for _operation, failure in cleanup_failures],
                        )
                except BaseException as cleanup_error:
                    cleanup_failure = cleanup_error
                    authoritative_failure = cleanup_error
                await deliveries.put(_ManualRecoveryStreamOutcome(error=authoritative_failure))
            finally:
                if supervisor_runtime_task is not None:
                    self._session_control.unregister_active_control_task(
                        claim.session.id,
                        supervisor_runtime_task,
                    )
            return _ManualRecoverySupervisorResult(
                error=authoritative_failure,
                cleanup_failure=cleanup_failure,
            )

        supervisor_task = asyncio.create_task(supervise_recovery())
        supervisor_start_outcome = await await_shielded_task_outcome(
            asyncio.create_task(supervisor_started.wait())
        )
        if caller_runtime_task is not None:
            # CayuApp reserves the caller task before the durable claim. Once
            # the supervisor is live, transfer process-local ownership so an
            # operator interrupt targets one recovery layer rather than both.
            self._session_control.unregister_active_task(
                claim.session.id,
                caller_runtime_task,
            )
        pending_delivery: _ManualRecoveryEventDelivery | None = None
        authoritative_failure: BaseException | None = None

        async def stop_supervisor() -> None:
            nonlocal consumer_stop_failure
            consumer_stop_failure = authoritative_failure
            consumer_stopped.set()
            if pending_delivery is not None:
                pending_delivery.consumed.set()
            if not supervisor_task.done():
                supervisor_task.cancel()
            await asyncio.gather(supervisor_task, return_exceptions=True)
            if isinstance(consumer_stop_failure, GeneratorExit):
                supervisor_result = supervisor_task.result()
                if supervisor_result.cleanup_failure is not None:
                    raise supervisor_result.cleanup_failure

        try:
            if supervisor_start_outcome.error is not None:
                raise supervisor_start_outcome.error
            if supervisor_start_outcome.cancellation is not None:
                raise supervisor_start_outcome.cancellation
            while True:
                item = await deliveries.get()
                if isinstance(item, _ManualRecoveryStreamOutcome):
                    if item.error is not None:
                        raise item.error
                    return
                pending_delivery = item
                yield item.event
                item.consumed.set()
                pending_delivery = None
        except BaseException as exc:
            authoritative_failure = exc
            raise
        finally:
            await _run_recovery_cleanup_steps(
                authoritative_failure=authoritative_failure,
                steps=(("manual tool-round recovery supervisor stop", stop_supervisor),),
            )

    async def _recover_tool_round_claimed(
        self,
        *,
        request: ToolRoundRecoveryRequest,
        loaded_session: Session,
        session: Session,
        pending_round: tool_round_recovery.PendingToolRound,
        pending_tool_call: PendingToolCallApproval,
        registered_agent: runtime_records.RegisteredAgentState,
        registered_provider: runtime_records.RegisteredProvider,
        registered_environment: runtime_records.RegisteredEnvironment | None,
        effective_structured_output: StructuredOutputSpec | None,
    ) -> AsyncGenerator[Event, None]:
        """Persist one operator-verified ordinary tool outcome and continue safely."""
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
        environment_name = _environment_name(registered_environment)
        recovery_persisted = False
        cancellation_baseline = _task_cancellation_count()
        recovery_event_to_reconcile: Event | None = None

        try:
            events = await self._session_store.load_events(session.id)
            tool_round_recovery.validate_tool_round_recovery_target(
                events=events,
                pending_round=pending_round,
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
                async for event in self._interrupt_for_resumable_manual_recovery(
                    session=session,
                    registered_agent=registered_agent,
                    registered_environment=registered_environment,
                    payload={
                        "interruption_type": _INTERRUPTION_TYPE_RUNTIME_INTERRUPTED,
                        "tool_round_id": pending_round.round_id,
                        "error": str(factory_resolution.error),
                        "error_type": type(factory_resolution.error).__name__,
                    },
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
                        "tool_round_id": pending_round.round_id,
                        "tool_call_id": pending_tool_call.tool_call_id,
                        "idempotency_key": tool_execution.tool_idempotency_key(
                            session_id=session.id,
                            tool_round_id=pending_round.round_id,
                            tool_call_id=pending_tool_call.tool_call_id,
                        ),
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
            recovery_event_to_reconcile = recovery_tool_event
            recovery_events = [
                Event(
                    type=EventType.SESSION_RESUMED,
                    session_id=session.id,
                    agent_name=registered_agent.spec.name,
                    environment_name=environment_name,
                    payload={
                        "interruption_type": _INTERRUPTION_TYPE_RUNTIME_INTERRUPTED,
                        "tool_round_id": pending_round.round_id,
                        "tool_call_id": pending_tool_call.tool_call_id,
                        "resolved_by": resolution_actor_payload(request.resolved_by),
                    },
                ),
                recovery_tool_event,
            ]
            emitted_recovery_events = await self._event_writer.persist_many(
                session.id, recovery_events
            )
            recovery_persisted = True
            await self._event_writer.fan_out_persisted(emitted_recovery_events)
            for event in emitted_recovery_events:
                yield event
            tool_call = runtime_records.ToolCallRequest(
                id=pending_tool_call.tool_call_id,
                name=pending_tool_call.tool_name,
                arguments=copy_json_value(pending_tool_call.arguments, "arguments"),
            )
            tool_event = emitted_recovery_events[-1]
            # The operator outcome is durable before hooks run. Recovery hooks are
            # observe-only so they cannot rewrite externally verified evidence.
            async for event, _modified in self._tool_round_executor.run_tool_call_hooks(
                session=session,
                tool_event=tool_event,
                registered_agent=registered_agent,
                registered_environment=registered_environment,
                tool_call=tool_call,
                result=recovered_result,
                task_id=pending_round.task_id,
                redactor=self._secret_redactor,
                allow_modification=False,
            ):
                yield event

            events = await self._session_store.load_events(session.id)
            recorded_outcomes, started_ids = tool_round_recovery.recorded_tool_outcomes(
                events=events,
                pending_round=pending_round,
            )
            remaining_ids = started_ids - set(recorded_outcomes)
            if remaining_ids:
                next_call = next(
                    call for call in pending_round.tool_calls if call.tool_call_id in remaining_ids
                )
                async for event in self._interrupt_for_resumable_manual_recovery(
                    session=session,
                    registered_agent=registered_agent,
                    registered_environment=registered_environment,
                    payload={
                        "interruption_type": _INTERRUPTION_TYPE_RUNTIME_INTERRUPTED,
                        "manual_recovery_required": True,
                        "tool_round_id": pending_round.round_id,
                        "tool_call_id": next_call.tool_call_id,
                        "tool_name": next_call.tool_name,
                    },
                ):
                    yield event
                return
        except (GeneratorExit, asyncio.CancelledError) as abandonment:
            await _run_recovery_cleanup_steps(
                authoritative_failure=abandonment,
                steps=(
                    (
                        "abandoned session finalization",
                        lambda: self.finalize_abandoned_session_by_id(session.id),
                    ),
                ),
            )
            raise
        except Exception as exc:
            reconciliation_error: Exception | None = None
            if not recovery_persisted and recovery_event_to_reconcile is not None:
                try:
                    reconciliation = await self._reconcile_manual_recovery_persistence(
                        recovery_event_to_reconcile
                    )
                except BaseException as reconciliation_failure:
                    if (
                        _recovery_abandonment_signal(
                            reconciliation_failure,
                            cancellation_baseline=cancellation_baseline,
                        )
                        is not None
                    ):
                        await _run_recovery_cleanup_steps(
                            authoritative_failure=reconciliation_failure,
                            steps=(
                                (
                                    "abandoned session finalization",
                                    lambda: self.finalize_abandoned_session_by_id(session.id),
                                ),
                            ),
                        )
                    raise
                if reconciliation.cancellation is not None:
                    reconciliation.cancellation.add_note(
                        "Manual tool-round recovery append failed while persistence "
                        "reconciliation was running."
                    )
                    await _run_recovery_cleanup_steps(
                        authoritative_failure=reconciliation.cancellation,
                        steps=(
                            (
                                "abandoned session finalization",
                                lambda: self.finalize_abandoned_session_by_id(session.id),
                            ),
                        ),
                    )
                    raise reconciliation.cancellation from exc
                recovery_persisted = reconciliation.persisted is True
                reconciliation_error = reconciliation.error
            if not recovery_persisted and reconciliation_error is None:
                if isinstance(exc, SessionRunFenced):
                    raise
                if loaded_session.status in {
                    SessionStatus.RUNNING,
                    SessionStatus.INTERRUPTING,
                }:
                    if loaded_session.status == SessionStatus.INTERRUPTING:
                        session = await self._session_store.transition_status(
                            session.id,
                            from_statuses={SessionStatus.RUNNING},
                            to_status=SessionStatus.INTERRUPTING,
                        )
                    async for event in self._interrupt_for_resumable_manual_recovery(
                        session=session,
                        registered_agent=registered_agent,
                        registered_environment=registered_environment,
                        payload={
                            "interruption_type": _INTERRUPTION_TYPE_RUNTIME_INTERRUPTED,
                            "tool_round_id": pending_round.round_id,
                            "tool_call_id": pending_tool_call.tool_call_id,
                            "manual_recovery_stale_live_failure": True,
                            "error": str(exc),
                            "error_type": type(exc).__name__,
                            "resolved_by": resolution_actor_payload(request.resolved_by),
                        },
                    ):
                        yield event
                    return
                try:
                    await self._session_store.transition_status(
                        session.id,
                        from_statuses={SessionStatus.RUNNING},
                        to_status=loaded_session.status,
                    )
                except SessionStatusConflict:
                    current = await self._require_session(session.id)
                    if current.status not in {
                        SessionStatus.INTERRUPTING,
                        SessionStatus.INTERRUPTED,
                    }:
                        raise
                    async for event in self._interrupt_session_for_recovery(
                        RecoveryInterruptionRequest(
                            session=current,
                            registered_agent=registered_agent,
                            registered_environment=registered_environment,
                            environment_name=_environment_name(registered_environment),
                        )
                    ):
                        yield event
                    return
                raise
            persistence_payload = (
                {"manual_recovery_persisted": True}
                if recovery_persisted
                else {
                    "manual_recovery_persistence_unknown": True,
                    "persistence_reconciliation_error_type": type(reconciliation_error).__name__,
                }
            )
            async for event in self._interrupt_for_resumable_manual_recovery(
                session=session,
                registered_agent=registered_agent,
                registered_environment=registered_environment,
                payload={
                    "interruption_type": _INTERRUPTION_TYPE_RUNTIME_INTERRUPTED,
                    "tool_round_id": pending_round.round_id,
                    "tool_call_id": pending_tool_call.tool_call_id,
                    **persistence_payload,
                    "error": str(exc),
                    "error_type": type(exc).__name__,
                    "resolved_by": resolution_actor_payload(request.resolved_by),
                },
            ):
                yield event
            return
        except BaseExceptionGroup as exc:
            abandonment = _recovery_abandonment_signal(
                exc,
                cancellation_baseline=cancellation_baseline,
            )
            if recovery_persisted and abandonment is None:
                async for event in self._interrupt_session_for_recovery(
                    RecoveryInterruptionRequest(
                        session=session,
                        registered_agent=registered_agent,
                        registered_environment=registered_environment,
                        environment_name=_environment_name(registered_environment),
                    )
                ):
                    yield event
            if abandonment is not None:
                await _run_recovery_cleanup_steps(
                    authoritative_failure=exc,
                    steps=(
                        (
                            "abandoned session finalization",
                            lambda: self.finalize_abandoned_session_by_id(session.id),
                        ),
                    ),
                )
            raise

        session_stream: AsyncGenerator[Event, None] | None = None
        authoritative_failure: BaseException | None = None
        try:
            transcript = await self._session_store.load_transcript(session.id)
            session_stream = self._run_session(
                RecoverySessionRunRequest(
                    session=session,
                    registered_agent=registered_agent,
                    registered_provider=registered_provider,
                    registered_environment=registered_environment,
                    messages=transcript,
                    messages_to_append=[],
                    max_steps=request.max_steps or _DEFAULT_APPROVAL_MAX_STEPS,
                    limits=request.limits or RunLimits(),
                    budget_limits=request.budget_limits or (),
                    retry_policy=self._effective_retry_policy(request.retry_policy),
                    structured_output=effective_structured_output,
                    thinking=request.thinking,
                    request_loop_policies=request.loop_policies,
                    request_metadata=request.metadata,
                    task_id=pending_round.task_id,
                    task_worker_id=None,
                    start_event_type=None,
                    start_event_payload={},
                    start_task_on_enter=False,
                    release_run_fence_on_exit=False,
                )
            )
            async for event in session_stream:
                yield event
        except BaseException as exc:
            authoritative_failure = exc
            raise
        finally:
            await self._cleanup_recovery_handoff(
                stream=session_stream,
                session_id=session.id,
                authoritative_failure=authoritative_failure,
                finalize_abandoned=(
                    _recovery_abandonment_signal(
                        authoritative_failure,
                        cancellation_baseline=cancellation_baseline,
                    )
                    is not None
                ),
                release_run_fence=False,
                abort_environment_setup=False,
            )

    async def close_interrupted_tool_round(
        self,
        request: InterruptedToolRoundRequest,
    ) -> AsyncGenerator[Event, None]:
        """Close an interrupted round without replaying unfinished tools."""
        if await transcript_helpers.tool_round_has_result_messages(
            self._session_store,
            request.session.id,
            request.tool_calls,
        ):
            return
        terminal_event_exists = (
            await self._session_control.latest_interrupted_event(request.session.id) is not None
        )
        interrupted_results = _interrupted_tool_round_results(
            tool_calls=request.tool_calls,
            completed_outcomes=request.tool_outcomes,
            tool_round_id=request.tool_round_id,
            cancellation_artifacts=request.cancellation_artifacts,
            cancellation_artifacts_by_id=request.cancellation_artifacts_by_id,
        )
        interrupted_results = await self.reattach_subagent_children_in_outcomes(
            session_id=request.session.id,
            tool_round_id=request.tool_round_id,
            outcomes=interrupted_results,
        )
        tool_outcomes = tool_results.redact_tool_call_outcomes(
            request.tool_outcomes,
            self._secret_redactor,
        )
        interrupted_results = tool_results.redact_tool_call_outcomes(
            interrupted_results,
            self._secret_redactor,
        )
        if not interrupted_results and not tool_outcomes:
            return
        if not terminal_event_exists:
            for interrupted_result in interrupted_results:
                yield await self._event_writer.emit(
                    _interrupted_tool_call_event(
                        session=request.session,
                        registered_agent=request.registered_agent,
                        registered_environment=request.registered_environment,
                        tool_call_outcome=interrupted_result,
                        tool_round_id=request.tool_round_id,
                    )
                )
        tool_outcomes.extend(interrupted_results)
        interrupted_messages = ordered_tool_result_messages(
            request.tool_calls,
            tool_outcomes,
            parallel=True,
        )
        request.messages.extend(interrupted_messages)
        cleared_checkpoint = await self._tool_round_executor.checkpoint_without_pending_tool_round(
            request.session.id
        )
        await self._session_store.append_transcript_messages_and_transform_checkpoint(
            request.session.id,
            interrupted_messages,
            self._checkpoint_transform(cleared_checkpoint),
        )

    async def reattach_subagent_children_in_outcomes(
        self,
        *,
        session_id: str,
        tool_round_id: str | None,
        outcomes: list[runtime_records.ToolCallOutcome],
    ) -> list[runtime_records.ToolCallOutcome]:
        """Replace unfinished spawn outcomes with matching durable child references."""
        if tool_round_id is None or not outcomes:
            return outcomes
        children = await self._subagent_children_by_idempotency_key(session_id)
        if not children:
            return outcomes
        reattached: list[runtime_records.ToolCallOutcome] = []
        for outcome in outcomes:
            result = self._reattached_subagent_result(
                children,
                tool_execution.tool_idempotency_key(
                    session_id=session_id,
                    tool_round_id=tool_round_id,
                    tool_call_id=outcome.call.id,
                ),
                tool_call_id=outcome.call.id,
                tool_name=outcome.call.name,
                tool_round_id=tool_round_id,
            )
            reattached.append(
                outcome
                if result is None
                else runtime_records.ToolCallOutcome(call=outcome.call, result=result)
            )
        return reattached

    async def recover_pending_tool_round(
        self,
        *,
        session: Session,
        registered_agent: runtime_records.RegisteredAgentState,
        registered_environment: runtime_records.RegisteredEnvironment | None,
        messages: list[Message],
        tail_message_count: int = 0,
    ) -> AsyncGenerator[Event, None]:
        """Repair one durable pending round strictly from recorded evidence."""
        checkpoint = await self._session_store.load_checkpoint(session.id)
        pending_round = tool_round_recovery.pending_tool_round_from_checkpoint(checkpoint)
        if pending_round is None:
            return
        environment_name = _environment_name(registered_environment)
        if pending_round.agent_name != registered_agent.spec.name:
            raise RuntimeError(
                f"Pending tool round belongs to a different agent: {pending_round.agent_name}."
            )
        if pending_round.environment_name != environment_name:
            raise RuntimeError(
                "Pending tool round belongs to a different environment: "
                f"{pending_round.environment_name}."
            )

        pending_tool_calls = tool_round_recovery.pending_round_tool_calls(pending_round)
        if await transcript_helpers.tool_round_has_result_messages(
            self._session_store,
            session.id,
            pending_tool_calls,
        ):
            await self._clear_pending_tool_round_if_matches(session.id, pending_round)
            yield await self._event_writer.emit(
                Event(
                    type=EventType.SESSION_CHECKPOINTED,
                    session_id=session.id,
                    agent_name=registered_agent.spec.name,
                    environment_name=environment_name,
                    payload={
                        "checkpoint": tool_round_recovery.PENDING_TOOL_ROUND_CHECKPOINT_KEY,
                        "tool_round_id": pending_round.round_id,
                        "cleared": True,
                    },
                )
            )
            return

        events = await self._session_store.load_events(session.id)
        recorded_outcomes, started_ids = tool_round_recovery.recorded_tool_outcomes(
            events=events,
            pending_round=pending_round,
        )
        subagent_children: dict[str, Session] = {}
        if any(
            recorded_outcomes.get(call.tool_call_id) is None for call in pending_round.tool_calls
        ):
            subagent_children = await self._subagent_children_by_idempotency_key(session.id)
        tool_outcomes: list[runtime_records.ToolCallOutcome] = []
        for pending_tool_call in pending_round.tool_calls:
            recorded_outcome = recorded_outcomes.get(pending_tool_call.tool_call_id)
            if recorded_outcome is not None:
                tool_outcomes.append(recorded_outcome)
                continue

            tool_call = runtime_records.ToolCallRequest(
                id=pending_tool_call.tool_call_id,
                name=pending_tool_call.tool_name,
                arguments=copy_json_value(pending_tool_call.arguments, "arguments"),
            )
            expected_idempotency_key = tool_execution.tool_idempotency_key(
                session_id=session.id,
                tool_round_id=pending_round.round_id,
                tool_call_id=pending_tool_call.tool_call_id,
            )
            result = self._reattached_subagent_result(
                subagent_children,
                expected_idempotency_key,
                tool_call_id=pending_tool_call.tool_call_id,
                tool_name=pending_tool_call.tool_name,
                tool_round_id=pending_round.round_id,
            )
            if result is None:
                result = tool_round_recovery.unknown_recovered_tool_result(
                    pending_tool_call=pending_tool_call,
                    pending_round=pending_round,
                    started=pending_tool_call.tool_call_id in started_ids,
                )
            event_type = (
                EventType.TOOL_CALL_FAILED if result.is_error else EventType.TOOL_CALL_COMPLETED
            )
            async for event, outcome in self._tool_round_executor.emit_tool_call_result_with_hooks(
                event=Event(
                    type=event_type,
                    session_id=session.id,
                    agent_name=registered_agent.spec.name,
                    environment_name=environment_name,
                    tool_name=tool_call.name,
                    payload={
                        "tool_round_id": pending_round.round_id,
                        "tool_call_id": tool_call.id,
                        "idempotency_key": expected_idempotency_key,
                        "recovered": True,
                        "result": result.model_dump(),
                    },
                ),
                session=session,
                registered_agent=registered_agent,
                registered_environment=registered_environment,
                tool_call=tool_call,
                result=result,
                task_id=pending_round.task_id,
            ):
                yield event
                if outcome is not None:
                    tool_outcomes.append(outcome)

        tool_result_messages = transcript_helpers.tool_result_messages(tool_outcomes)
        insert_at = len(messages) - tail_message_count
        if insert_at < 0:
            raise RuntimeError("Pending tool round recovery received an invalid tail size.")
        messages[insert_at:insert_at] = tool_result_messages
        cleared_checkpoint = await self._tool_round_executor.checkpoint_without_pending_tool_round(
            session.id
        )
        await self._session_store.append_transcript_messages_and_transform_checkpoint(
            session.id,
            tool_result_messages,
            self._checkpoint_transform(cleared_checkpoint),
        )
        yield await self._event_writer.emit(
            Event(
                type=EventType.SESSION_CHECKPOINTED,
                session_id=session.id,
                agent_name=registered_agent.spec.name,
                environment_name=environment_name,
                payload={
                    "checkpoint": tool_round_recovery.PENDING_TOOL_ROUND_CHECKPOINT_KEY,
                    "tool_round_id": pending_round.round_id,
                    "cleared": True,
                    "recovered_tool_calls": len(tool_outcomes),
                },
            )
        )

    async def finalize_abandoned_session_run(
        self,
        request: RecoveryAbandonedSessionRequest,
    ) -> None:
        """Best-effort finalization for a live session whose event stream closed."""
        try:
            finalized = await self._session_store.transition_status(
                request.session.id,
                from_statuses={
                    SessionStatus.PENDING,
                    SessionStatus.RUNNING,
                    SessionStatus.INTERRUPTING,
                },
                to_status=SessionStatus.INTERRUPTED,
            )
        except (KeyError, ValueError):
            return
        if request.run_started_at is not None and request.turn_usage_tracker is not None:
            with contextlib.suppress(Exception):
                await self._abandoned_turn_completed(
                    RecoveryAbandonedTurnRequest(
                        session=finalized,
                        registered_agent=request.registered_agent,
                        environment_name=request.environment_name,
                        run_started_at=request.run_started_at,
                        usage_tracker=request.turn_usage_tracker,
                        active_run=request.active_run,
                    )
                )
        with contextlib.suppress(BaseException):
            async for _ in self._emit_terminal_event_with_hooks(
                RecoveryTerminalEventRequest(
                    event=Event(
                        type=EventType.SESSION_INTERRUPTED,
                        session_id=finalized.id,
                        agent_name=request.registered_agent.spec.name,
                        environment_name=request.environment_name,
                        payload={
                            "interruption_type": _INTERRUPTION_TYPE_RUNTIME_INTERRUPTED,
                            "reason": _ABANDONED_RUN_REASON,
                            "abandoned": True,
                        },
                    ),
                    phase=RuntimeHookPhase.AFTER_SESSION_INTERRUPTED,
                    session=finalized,
                    registered_agent=request.registered_agent,
                    registered_environment=request.registered_environment,
                )
            ):
                pass

    async def finalize_abandoned_session_by_id(self, session_id: str) -> None:
        """Idempotently finalize a live session when setup-time streaming is abandoned."""
        try:
            session = await self._session_store.load(session_id)
        except Exception:
            return
        if session is None or session.status not in {
            SessionStatus.PENDING,
            SessionStatus.RUNNING,
            SessionStatus.INTERRUPTING,
        }:
            return
        try:
            registered_agent = self._resolve_registered_agent(session.agent_name)
        except Exception:
            await self._finalize_abandoned_without_registered_runtime(session.id)
            return
        try:
            registered_environment = self._resolve_registered_environment(session.environment_name)
        except Exception:
            await self._finalize_abandoned_without_registered_runtime(session.id)
            return
        if session.status == SessionStatus.INTERRUPTING:
            try:
                async for _ in self._interrupt_session_for_recovery(
                    RecoveryInterruptionRequest(
                        session=session,
                        registered_agent=registered_agent,
                        registered_environment=registered_environment,
                        environment_name=_environment_name(registered_environment),
                    )
                ):
                    pass
                return
            except BaseException:
                # Preserve the existing best-effort fallback if the durable
                # operator-interruption payload cannot be finalized.
                pass
        with contextlib.suppress(BaseException):
            await self.finalize_abandoned_session_run(
                RecoveryAbandonedSessionRequest(
                    session=session,
                    registered_agent=registered_agent,
                    registered_environment=registered_environment,
                    environment_name=_environment_name(registered_environment),
                )
            )

    async def _finalize_abandoned_without_registered_runtime(self, session_id: str) -> None:
        try:
            finalized = await self._session_store.transition_status(
                session_id,
                from_statuses={
                    SessionStatus.PENDING,
                    SessionStatus.RUNNING,
                    SessionStatus.INTERRUPTING,
                },
                to_status=SessionStatus.INTERRUPTED,
            )
        except Exception:
            return
        with contextlib.suppress(BaseException):
            await self._event_writer.emit(
                Event(
                    type=EventType.SESSION_INTERRUPTED,
                    session_id=finalized.id,
                    agent_name=finalized.agent_name,
                    environment_name=finalized.environment_name,
                    payload={
                        "interruption_type": _INTERRUPTION_TYPE_RUNTIME_INTERRUPTED,
                        "reason": _ABANDONED_RUN_REASON,
                        "abandoned": True,
                    },
                )
            )

    async def recover_incomplete_session(
        self,
        request: IncompleteSessionRecoveryRequest,
    ) -> IncompleteSessionRecoveryResult:
        """Repair one incomplete session without executing providers or tools."""
        session = await self._session_store.load(request.session_id)
        if session is None:
            raise KeyError(f"Session not found: {request.session_id}") from None
        return await self._recover_incomplete_session_scoped(
            session=session,
            inactive_before=request.inactive_before,
            reason=request.reason,
            metadata=request.metadata,
        )

    async def recover_incomplete_sessions(
        self,
        request: IncompleteSessionsRecoveryRequest,
    ) -> list[IncompleteSessionRecoveryResult]:
        """Fault-isolate recovery across the requested non-terminal sessions."""
        sessions: list[Session] = []
        seen_session_ids: set[str] = set()
        for status in (
            SessionStatus.INTERRUPTING,
            SessionStatus.RUNNING,
            SessionStatus.PENDING,
        ):
            if status not in request.statuses:
                continue
            if len(sessions) >= request.limit:
                break
            candidates = (
                await self._session_store.list_sessions(
                    SessionQuery(
                        status=status,
                        last_activity_before=request.inactive_before,
                        limit=min(1000, request.limit - len(sessions)),
                    )
                )
            ).sessions
            for candidate in candidates:
                if (
                    request.inactive_before is not None
                    and candidate.last_activity_at > request.inactive_before
                ):
                    continue
                if candidate.id in seen_session_ids:
                    continue
                seen_session_ids.add(candidate.id)
                sessions.append(candidate)
                if len(sessions) >= request.limit:
                    break

        results: list[IncompleteSessionRecoveryResult] = []
        for session in sessions:
            try:
                result = await self._recover_incomplete_session_scoped(
                    session=session,
                    inactive_before=request.inactive_before,
                    reason=request.reason,
                    metadata=request.metadata,
                )
            except Exception as exc:
                logger.warning(
                    "Recovery failed for session %s (agent %s): %s",
                    session.id,
                    session.agent_name,
                    exc,
                )
                try:
                    reloaded = await self._session_store.load(session.id)
                except Exception:
                    reloaded = None
                result = IncompleteSessionRecoveryResult(
                    session_id=session.id,
                    previous_status=session.status,
                    status=session.status if reloaded is None else reloaded.status,
                    actions=(IncompleteSessionRecoveryAction.FAILED,),
                    message=f"Recovery failed: {type(exc).__name__}: {exc}",
                )
            results.append(result)
        return results

    async def _recover_incomplete_session_scoped(
        self,
        *,
        session: Session,
        inactive_before: datetime | None,
        reason: str,
        metadata: dict[str, Any],
    ) -> IncompleteSessionRecoveryResult:
        reason = require_clean_nonblank(reason, "reason")
        metadata = copy_json_value(metadata, "metadata")
        previous_status = session.status

        if self._session_control.has_active_tasks(session.id):
            return IncompleteSessionRecoveryResult(
                session_id=session.id,
                previous_status=previous_status,
                status=session.status,
                actions=(IncompleteSessionRecoveryAction.SKIPPED_ACTIVE,),
                events=(),
                message="Session has active work in this CayuApp process; recovery skipped.",
            )

        return await self._recover_incomplete_session_owned(
            session=session,
            inactive_before=inactive_before,
            reason=reason,
            metadata=metadata,
            previous_status=previous_status,
        )

    async def _recover_incomplete_session_owned(
        self,
        *,
        session: Session,
        inactive_before: datetime | None,
        reason: str,
        metadata: dict[str, Any],
        previous_status: SessionStatus,
    ) -> IncompleteSessionRecoveryResult:

        checkpoint = await self._session_store.load_checkpoint(session.id)
        pending_approval = approval_support.pending_approval_from_checkpoint(checkpoint)
        pending_user_input = pending_user_input_from_checkpoint(checkpoint)
        pending_tool_round = tool_round_recovery.pending_tool_round_from_checkpoint(checkpoint)
        if (
            session.status in _RECOVERY_RESUMABLE_SESSION_STATUSES
            and pending_approval is None
            and pending_user_input is None
            and pending_tool_round is None
        ):
            return IncompleteSessionRecoveryResult(
                session_id=session.id,
                previous_status=previous_status,
                status=session.status,
                actions=(IncompleteSessionRecoveryAction.SKIPPED_TERMINAL,),
                events=(),
                message="Session is terminal; recovery skipped.",
            )

        try:
            registered_agent = self._resolve_registered_agent(session.agent_name)
        except KeyError:
            return IncompleteSessionRecoveryResult(
                session_id=session.id,
                previous_status=previous_status,
                status=session.status,
                actions=(IncompleteSessionRecoveryAction.SKIPPED_UNREGISTERED_AGENT,),
                events=(),
                message=(f"Agent not registered: {session.agent_name!r}; session left untouched."),
            )
        registered_environment = self._resolve_registered_environment(session.environment_name)

        claim: _IncompleteRecoveryClaim | None = None
        authoritative_failure: BaseException | None = None
        try:
            claim = await self._claim_incomplete_recovery(
                session=session,
                inactive_before=inactive_before,
            )
            if claim is None:
                current = await self._require_session(session.id)
                return IncompleteSessionRecoveryResult(
                    session_id=session.id,
                    previous_status=previous_status,
                    status=current.status,
                    actions=(IncompleteSessionRecoveryAction.SKIPPED_ACTIVE,),
                    events=(),
                    message="Session activity or recovery ownership changed; recovery skipped.",
                )
            return await self._recover_incomplete_session_with_heartbeat(
                claim=claim,
                recovery=lambda: self._recover_incomplete_session(
                    session=claim.session,
                    session_before_fence=claim.session_before_fence,
                    previous_status=previous_status,
                    inactive_before=inactive_before,
                    reason=reason,
                    metadata=metadata,
                    registered_agent=registered_agent,
                    registered_environment=registered_environment,
                ),
            )
        except BaseException as exc:
            authoritative_failure = exc
            raise
        finally:
            if claim is not None:
                await self._cleanup_incomplete_recovery_claim(
                    session_id=session.id,
                    claim_id=claim.claim_id,
                    authoritative_failure=authoritative_failure,
                )

    async def _cleanup_incomplete_recovery_claim(
        self,
        *,
        session_id: str,
        claim_id: str,
        authoritative_failure: BaseException | None,
    ) -> None:
        try:
            await _run_recovery_cleanup_steps(
                authoritative_failure=authoritative_failure,
                steps=(
                    (
                        "run fence release",
                        lambda: self._session_store.release_run_fence(session_id),
                    ),
                    (
                        "incomplete recovery claim release",
                        lambda: self._release_incomplete_recovery_claim(
                            session_id,
                            claim_id,
                        ),
                    ),
                ),
            )
        finally:
            _deactivate_session_run_fence(session_id)

    async def _load_owned_incomplete_recovery_claim(
        self,
        session_id: str,
        claim_id: str,
        *,
        expected_run_epoch: int | None,
    ) -> Session | None:
        """Return the session only while the exact claim and its epoch are owned."""
        if expected_run_epoch is None:
            return None
        checkpoint = await self._session_store.load_checkpoint(session_id)
        persisted_claim = _incomplete_recovery_claim_from_checkpoint(checkpoint)
        if persisted_claim is None or persisted_claim[0] != claim_id:
            return None
        session = await self._require_session(session_id)
        if session.run_epoch != expected_run_epoch:
            return None
        return session

    async def _claim_incomplete_recovery(
        self,
        *,
        session: Session,
        inactive_before: datetime | None,
        required_expired_claim_id: str | None = None,
    ) -> _IncompleteRecoveryClaim | None:
        if required_expired_claim_id is not None:
            required_expired_claim_id = require_clean_nonblank(
                required_expired_claim_id,
                "required_expired_claim_id",
            )
        claim_id = str(uuid4())
        claim_expires_at: datetime | None = None
        claim_run_epoch: int | None = None

        if required_expired_claim_id is not None:
            session_before_fence: Session | None = None

            def replace_expired_claim(
                current_session: Session,
                checkpoint: dict[str, Any] | None,
            ) -> dict[str, Any]:
                nonlocal claim_expires_at, claim_run_epoch, session_before_fence
                claimed_at = self._clock()
                _require_aware_datetime(claimed_at, "recovery claim clock")
                existing = _incomplete_recovery_claim_from_checkpoint(checkpoint)
                if (
                    existing is None
                    or existing[0] != required_expired_claim_id
                    or existing[1] > claimed_at
                ):
                    raise _IncompleteRecoveryClaimLost(
                        "Expired incomplete-session recovery ownership changed."
                    )
                claim_expires_at = claimed_at + _INCOMPLETE_RECOVERY_CLAIM_LEASE
                claim_run_epoch = current_session.run_epoch + 1
                session_before_fence = current_session.model_copy(deep=True)
                updated = {} if checkpoint is None else copy_json_value(checkpoint, "checkpoint")
                updated[_INCOMPLETE_RECOVERY_CLAIM_CHECKPOINT_KEY] = {
                    "version": 1,
                    "claim_id": claim_id,
                    "claimed_at": claimed_at.isoformat(),
                    "claim_expires_at": claim_expires_at.isoformat(),
                }
                return updated

            fence_task = asyncio.create_task(
                self._session_store.fence_run_and_transform_checkpoint(
                    session.id,
                    statuses={session.status},
                    checkpoint_transform=replace_expired_claim,
                )
            )
            outcome = await await_shielded_task_outcome(fence_task)
            if isinstance(
                outcome.error,
                _IncompleteRecoveryClaimLost | SessionStatusConflict,
            ):
                if outcome.cancellation is not None:
                    outcome.cancellation.add_note(
                        "Expired recovery takeover was rejected while cancellation was pending."
                    )
                    raise outcome.cancellation from outcome.error
                return None

            authoritative_failure = outcome.cancellation or outcome.error
            fenced = outcome.result
            if outcome.error is not None:
                reconciliation_outcome = await await_shielded_task_outcome(
                    asyncio.create_task(
                        self._load_owned_incomplete_recovery_claim(
                            session.id,
                            claim_id,
                            expected_run_epoch=claim_run_epoch,
                        )
                    ),
                    cancellation=outcome.cancellation,
                )
                reconciliation_cancellation = reconciliation_outcome.cancellation
                reconciliation_failure = reconciliation_outcome.error
                if reconciliation_cancellation is None and isinstance(
                    reconciliation_failure,
                    asyncio.CancelledError,
                ):
                    reconciliation_cancellation = reconciliation_failure
                authoritative_failure = (
                    reconciliation_cancellation or outcome.cancellation or outcome.error
                )
                if reconciliation_failure is not None:
                    if not isinstance(
                        reconciliation_failure,
                        Exception | asyncio.CancelledError,
                    ):
                        raise reconciliation_failure from outcome.error
                    authoritative_failure.add_note(
                        "Could not reconcile whether the expired recovery takeover "
                        f"committed: {type(reconciliation_failure).__name__}."
                    )
                    if reconciliation_cancellation is not None:
                        reconciliation_cancellation.add_note(
                            "Expired recovery takeover also failed: "
                            f"{type(outcome.error).__name__}."
                        )
                        raise reconciliation_cancellation from outcome.error
                    raise outcome.error
                fenced = reconciliation_outcome.result
                if fenced is None:
                    if reconciliation_cancellation is not None:
                        reconciliation_cancellation.add_note(
                            "Expired recovery takeover also failed: "
                            f"{type(outcome.error).__name__}."
                        )
                        raise reconciliation_cancellation from outcome.error
                    raise outcome.error

            if fenced is None:
                raise RuntimeError("Expired recovery takeover returned no session.")
            _activate_session_run_fence(fenced)
            if (
                claim_expires_at is None
                or claim_run_epoch is None
                or session_before_fence is None
                or fenced.run_epoch != claim_run_epoch
            ):
                invariant_failure = RuntimeError(
                    "Expired recovery takeover did not persist its claim."
                )
                await self._cleanup_incomplete_recovery_claim(
                    session_id=session.id,
                    claim_id=claim_id,
                    authoritative_failure=invariant_failure,
                )
                raise invariant_failure
            claim = _IncompleteRecoveryClaim(
                claim_id=claim_id,
                claim_expires_at=claim_expires_at,
                session_before_fence=session_before_fence,
                session=fenced,
            )
            if authoritative_failure is None:
                return claim
            await self._cleanup_incomplete_recovery_claim(
                session_id=session.id,
                claim_id=claim_id,
                authoritative_failure=authoritative_failure,
            )
            if outcome.cancellation is not None:
                if outcome.error is not None:
                    outcome.cancellation.add_note(
                        f"Expired recovery takeover also failed: {type(outcome.error).__name__}."
                    )
                raise outcome.cancellation from outcome.error
            raise outcome.error

        session_before_fence: Session | None = None

        def claim_checkpoint(
            current_session: Session,
            checkpoint: dict[str, Any] | None,
        ) -> dict[str, Any]:
            nonlocal claim_expires_at, claim_run_epoch, session_before_fence
            claimed_at = self._clock()
            _require_aware_datetime(claimed_at, "recovery claim clock")
            existing = _incomplete_recovery_claim_from_checkpoint(checkpoint)
            if (
                current_session.status != session.status
                or (
                    inactive_before is not None
                    and current_session.last_activity_at > inactive_before
                )
                or (existing is not None and existing[1] > claimed_at)
            ):
                raise _IncompleteRecoveryClaimLost(
                    "Incomplete-session recovery ownership changed before it was claimed."
                )
            claim_expires_at = claimed_at + _INCOMPLETE_RECOVERY_CLAIM_LEASE
            claim_run_epoch = current_session.run_epoch + 1
            session_before_fence = current_session.model_copy(deep=True)
            updated = {} if checkpoint is None else copy_json_value(checkpoint, "checkpoint")
            updated[_INCOMPLETE_RECOVERY_CLAIM_CHECKPOINT_KEY] = {
                "version": 1,
                "claim_id": claim_id,
                "claimed_at": claimed_at.isoformat(),
                "claim_expires_at": claim_expires_at.isoformat(),
            }
            return updated

        fence_claimed = False
        try:
            fence_task = asyncio.create_task(
                self._session_store.fence_run_and_transform_checkpoint(
                    session.id,
                    statuses={session.status},
                    checkpoint_transform=claim_checkpoint,
                )
            )
            outcome = await await_shielded_task_outcome(fence_task)
            if isinstance(
                outcome.error,
                _IncompleteRecoveryClaimLost | SessionStatusConflict,
            ):
                if outcome.cancellation is not None:
                    outcome.cancellation.add_note(
                        "Incomplete-session recovery claim was rejected while cancellation "
                        "was pending."
                    )
                    raise outcome.cancellation from outcome.error
                return None

            authoritative_failure = outcome.cancellation or outcome.error
            fenced = outcome.result
            if outcome.error is not None:
                reconciliation_outcome = await await_shielded_task_outcome(
                    asyncio.create_task(
                        self._load_owned_incomplete_recovery_claim(
                            session.id,
                            claim_id,
                            expected_run_epoch=claim_run_epoch,
                        )
                    ),
                    cancellation=outcome.cancellation,
                )
                reconciliation_cancellation = reconciliation_outcome.cancellation
                reconciliation_failure = reconciliation_outcome.error
                if reconciliation_cancellation is None and isinstance(
                    reconciliation_failure,
                    asyncio.CancelledError,
                ):
                    reconciliation_cancellation = reconciliation_failure
                authoritative_failure = (
                    reconciliation_cancellation or outcome.cancellation or outcome.error
                )
                if reconciliation_failure is not None:
                    if not isinstance(
                        reconciliation_failure,
                        Exception | asyncio.CancelledError,
                    ):
                        raise reconciliation_failure from outcome.error
                    authoritative_failure.add_note(
                        "Could not reconcile whether the incomplete-session recovery claim "
                        f"committed: {type(reconciliation_failure).__name__}."
                    )
                    if reconciliation_cancellation is not None:
                        reconciliation_cancellation.add_note(
                            "Incomplete-session recovery claim also failed: "
                            f"{type(outcome.error).__name__}."
                        )
                        raise reconciliation_cancellation from outcome.error
                    raise outcome.error
                fenced = reconciliation_outcome.result
                if fenced is None:
                    if reconciliation_cancellation is not None:
                        reconciliation_cancellation.add_note(
                            "Incomplete-session recovery claim also failed: "
                            f"{type(outcome.error).__name__}."
                        )
                        raise reconciliation_cancellation from outcome.error
                    raise outcome.error

            if fenced is None:
                raise RuntimeError("Incomplete-session recovery claim returned no session.")
            _activate_session_run_fence(fenced)
            fence_claimed = True
            if (
                claim_expires_at is None
                or claim_run_epoch is None
                or session_before_fence is None
                or fenced.run_epoch != claim_run_epoch
            ):
                raise RuntimeError(
                    "Incomplete-session recovery claim was not persisted atomically."
                )
            if authoritative_failure is not None:
                if outcome.cancellation is not None:
                    if outcome.error is not None:
                        outcome.cancellation.add_note(
                            "Incomplete-session recovery claim also failed: "
                            f"{type(outcome.error).__name__}."
                        )
                    raise outcome.cancellation from outcome.error
                raise outcome.error

            try:
                renewed_until = await self._renew_incomplete_recovery_claim(
                    session.id,
                    claim_id,
                )
            except SessionRunFenced:
                await self._cleanup_incomplete_recovery_claim(
                    session_id=session.id,
                    claim_id=claim_id,
                    authoritative_failure=None,
                )
                fence_claimed = False
                return None
            if renewed_until is None:
                await self._cleanup_incomplete_recovery_claim(
                    session_id=session.id,
                    claim_id=claim_id,
                    authoritative_failure=None,
                )
                fence_claimed = False
                return None
            return _IncompleteRecoveryClaim(
                claim_id=claim_id,
                claim_expires_at=renewed_until,
                session_before_fence=session_before_fence,
                session=fenced,
            )
        except BaseException as exc:
            cleanup_steps: list[tuple[str, RecoveryCleanup]] = []
            if fence_claimed:
                cleanup_steps.append(
                    (
                        "run fence release",
                        lambda: self._session_store.release_run_fence(session.id),
                    )
                )
            cleanup_steps.append(
                (
                    "incomplete recovery claim release",
                    lambda: self._release_incomplete_recovery_claim(session.id, claim_id),
                )
            )
            try:
                await _run_recovery_cleanup_steps(
                    authoritative_failure=exc,
                    steps=tuple(cleanup_steps),
                )
            finally:
                if fence_claimed:
                    _deactivate_session_run_fence(session.id)
            raise

    async def _recover_incomplete_session_with_heartbeat(
        self,
        *,
        claim: _IncompleteRecoveryClaim,
        recovery: Callable[[], Awaitable[IncompleteSessionRecoveryResult]],
    ) -> IncompleteSessionRecoveryResult:
        stop_heartbeat = asyncio.Event()

        async def run_recovery() -> IncompleteSessionRecoveryResult:
            return await recovery()

        recovery_task = asyncio.create_task(run_recovery())
        heartbeat_task = asyncio.create_task(
            self._heartbeat_incomplete_recovery_claim(
                session_id=claim.session.id,
                claim_id=claim.claim_id,
                claim_expires_at=claim.claim_expires_at,
                stop=stop_heartbeat,
            )
        )
        authoritative_failure: BaseException | None = None

        async def stop_workers() -> None:
            stop_heartbeat.set()
            for task in (recovery_task, heartbeat_task):
                if not task.done():
                    task.cancel()
            await asyncio.gather(recovery_task, heartbeat_task, return_exceptions=True)

        try:
            done, _pending = await asyncio.wait(
                {recovery_task, heartbeat_task},
                return_when=asyncio.FIRST_COMPLETED,
            )
            if heartbeat_task in done:
                heartbeat_failure = heartbeat_task.exception()
                if heartbeat_failure is None:
                    raise RuntimeError(
                        "Incomplete-session recovery claim heartbeat stopped unexpectedly."
                    )
                raise heartbeat_failure
            result = recovery_task.result()
            stop_heartbeat.set()
            await heartbeat_task
            return result
        except BaseException as exc:
            authoritative_failure = exc
            raise
        finally:
            await _run_recovery_cleanup_steps(
                authoritative_failure=authoritative_failure,
                steps=(("incomplete recovery worker shutdown", stop_workers),),
            )

    async def _heartbeat_incomplete_recovery_claim(
        self,
        *,
        session_id: str,
        claim_id: str,
        claim_expires_at: datetime,
        stop: asyncio.Event,
    ) -> None:
        sleep_seconds = _INCOMPLETE_RECOVERY_CLAIM_HEARTBEAT_INTERVAL_SECONDS
        while not stop.is_set():
            try:
                await asyncio.wait_for(
                    stop.wait(),
                    timeout=sleep_seconds,
                )
            except TimeoutError:
                try:
                    renewed_until = await self._renew_incomplete_recovery_claim(
                        session_id,
                        claim_id,
                    )
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    now = self._clock()
                    _require_aware_datetime(now, "recovery claim clock")
                    if now >= claim_expires_at:
                        raise _IncompleteRecoveryClaimLost(
                            "Incomplete-session recovery claim could not be renewed before "
                            f"expiry for session {session_id}."
                        ) from exc
                    sleep_seconds = min(
                        _INCOMPLETE_RECOVERY_CLAIM_HEARTBEAT_RETRY_SECONDS,
                        max(0.0, (claim_expires_at - now).total_seconds()),
                    )
                    continue
                if renewed_until is None:
                    raise _IncompleteRecoveryClaimLost(
                        f"Incomplete-session recovery claim lost for session {session_id}."
                    ) from None
                claim_expires_at = renewed_until
                sleep_seconds = _INCOMPLETE_RECOVERY_CLAIM_HEARTBEAT_INTERVAL_SECONDS

    async def _watch_manual_recovery_interruption(
        self,
        *,
        session_id: str,
        interrupted_baseline_id: str | None,
        stop: asyncio.Event,
    ) -> bool:
        """Observe another worker's durable stop request while delivery is paused."""
        while not stop.is_set():
            session = await self._require_session(session_id)
            if session.status == SessionStatus.INTERRUPTING:
                return True
            if session.status == SessionStatus.INTERRUPTED:
                latest_interrupted = await self._session_control.latest_interrupted_event(
                    session_id
                )
                if (
                    latest_interrupted is not None
                    and latest_interrupted.id != interrupted_baseline_id
                    and latest_interrupted.payload.get("interruption_type")
                    == _INTERRUPTION_TYPE_OPERATOR_REQUESTED
                ):
                    return True
            try:
                await asyncio.wait_for(
                    stop.wait(),
                    timeout=_MANUAL_RECOVERY_INTERRUPT_POLL_INTERVAL_SECONDS,
                )
            except TimeoutError:
                continue
        return False

    async def _renew_incomplete_recovery_claim(
        self,
        session_id: str,
        claim_id: str,
    ) -> datetime | None:
        renewed_until: datetime | None = None

        def renew_claim(
            _session: Session,
            checkpoint: dict[str, Any] | None,
        ) -> dict[str, Any] | None:
            nonlocal renewed_until
            existing = _incomplete_recovery_claim_from_checkpoint(checkpoint)
            now = self._clock()
            _require_aware_datetime(now, "recovery claim clock")
            if existing is None or existing[0] != claim_id or existing[1] <= now:
                return None
            updated = copy_json_value(checkpoint, "checkpoint")
            marker = copy_json_value(
                updated[_INCOMPLETE_RECOVERY_CLAIM_CHECKPOINT_KEY],
                "incomplete_session_recovery_claim",
            )
            renewed_until = now + _INCOMPLETE_RECOVERY_CLAIM_LEASE
            marker["claim_expires_at"] = renewed_until.isoformat()
            marker["renewed_at"] = now.isoformat()
            updated[_INCOMPLETE_RECOVERY_CLAIM_CHECKPOINT_KEY] = marker
            return updated

        await self._session_store.transform_checkpoint(session_id, renew_claim)
        return renewed_until

    async def _release_incomplete_recovery_claim(
        self,
        session_id: str,
        claim_id: str,
    ) -> None:
        def release_claim(
            _session: Session,
            checkpoint: dict[str, Any] | None,
        ) -> dict[str, Any] | None:
            if checkpoint is None:
                return None
            existing = _incomplete_recovery_claim_from_checkpoint(checkpoint)
            if existing is None or existing[0] != claim_id:
                return None
            updated = copy_json_value(checkpoint, "checkpoint")
            updated.pop(_INCOMPLETE_RECOVERY_CLAIM_CHECKPOINT_KEY, None)
            return updated

        await self._session_store.transform_checkpoint(session_id, release_claim)

    async def _recover_incomplete_session(
        self,
        *,
        session: Session,
        session_before_fence: Session,
        previous_status: SessionStatus,
        inactive_before: datetime | None,
        reason: str,
        metadata: dict[str, Any],
        registered_agent: runtime_records.RegisteredAgentState,
        registered_environment: runtime_records.RegisteredEnvironment | None,
    ) -> IncompleteSessionRecoveryResult:
        actions: list[IncompleteSessionRecoveryAction] = []
        events: list[Event] = []
        checkpoint = await self._session_store.load_checkpoint(session.id)
        pending_approval = approval_support.pending_approval_from_checkpoint(checkpoint)
        pending_user_input = pending_user_input_from_checkpoint(checkpoint)
        pending_tool_round = tool_round_recovery.pending_tool_round_from_checkpoint(checkpoint)
        environment_name = _environment_name(registered_environment)

        if inactive_before is not None:
            events.append(
                await self._event_writer.emit(
                    Event(
                        type=EventType.SESSION_RUN_FENCED,
                        session_id=session.id,
                        agent_name=session.agent_name,
                        environment_name=environment_name,
                        payload={
                            "previous_run_epoch": session_before_fence.run_epoch,
                            "run_epoch": session.run_epoch,
                            "inactive_before": inactive_before.isoformat(),
                            "reason": reason,
                            "metadata": metadata,
                        },
                    )
                )
            )

        if session.status in {SessionStatus.PENDING, SessionStatus.RUNNING}:
            if pending_approval is not None:
                interrupt_payload = {
                    "interruption_type": _INTERRUPTION_TYPE_TOOL_APPROVAL_REQUIRED,
                    "approval": pending_approval.model_dump(mode="json"),
                    "recovered": True,
                    "reason": reason,
                    "metadata": metadata,
                }
            elif pending_user_input is not None:
                interrupt_payload = {
                    "interruption_type": _INTERRUPTION_TYPE_USER_INPUT_REQUIRED,
                    "user_input": pending_user_input.model_dump(mode="json"),
                    "recovered": True,
                    "reason": reason,
                    "metadata": metadata,
                }
            else:
                interrupt_payload = {
                    "reason": reason,
                    "metadata": metadata,
                    "interruption_type": _INTERRUPTION_TYPE_RUNTIME_INTERRUPTED,
                    "recovered": True,
                }
            interrupt_payload["interruption_request_id"] = str(uuid4())
            try:
                session = await self._session_store.transition_status_and_checkpoint(
                    session.id,
                    from_statuses={SessionStatus.PENDING, SessionStatus.RUNNING},
                    to_status=SessionStatus.INTERRUPTING,
                    checkpoint_transform=self._pending_session_interrupt_checkpoint(
                        interrupt_payload,
                        self._clock(),
                    ),
                )
            except ValueError:
                session = await self._require_session(session.id)
                if session.status in _RECOVERY_RESUMABLE_SESSION_STATUSES:
                    return IncompleteSessionRecoveryResult(
                        session_id=session.id,
                        previous_status=previous_status,
                        status=session.status,
                        actions=(IncompleteSessionRecoveryAction.SKIPPED_TERMINAL,),
                        events=(),
                        message="Session changed during recovery; recovery skipped.",
                    )
                raise
            session = await self._require_session(session.id)
            checkpoint = await self._session_store.load_checkpoint(session.id)
            pending_tool_round = tool_round_recovery.pending_tool_round_from_checkpoint(checkpoint)

        if pending_tool_round is not None:
            transcript = await self._session_store.load_transcript(session.id)
            async for event in self.recover_pending_tool_round(
                session=session,
                registered_agent=registered_agent,
                registered_environment=registered_environment,
                messages=transcript,
            ):
                events.append(event)
            actions.append(IncompleteSessionRecoveryAction.REPAIRED_TOOL_ROUND)
            session = await self._require_session(session.id)
            checkpoint = await self._session_store.load_checkpoint(session.id)

        pending_approval = approval_support.pending_approval_from_checkpoint(checkpoint)
        if pending_approval is not None:
            session = await self._finalize_interrupting_for_recovery(
                session=session,
                registered_agent=registered_agent,
                registered_environment=registered_environment,
                environment_name=environment_name,
                events=events,
            )
            actions.append(IncompleteSessionRecoveryAction.PENDING_APPROVAL)
            return IncompleteSessionRecoveryResult(
                session_id=session.id,
                previous_status=previous_status,
                status=session.status,
                actions=tuple(actions),
                events=tuple(events),
                pending_approval_id=pending_approval.approval_id,
                message="Session has a pending tool approval; resolve it with ToolApprovalRequest.",
            )

        pending_user_input = pending_user_input_from_checkpoint(checkpoint)
        if pending_user_input is not None:
            session = await self._finalize_interrupting_for_recovery(
                session=session,
                registered_agent=registered_agent,
                registered_environment=registered_environment,
                environment_name=environment_name,
                events=events,
            )
            actions.append(IncompleteSessionRecoveryAction.PENDING_USER_INPUT)
            return IncompleteSessionRecoveryResult(
                session_id=session.id,
                previous_status=previous_status,
                status=session.status,
                actions=tuple(actions),
                events=tuple(events),
                pending_user_input_id=pending_user_input.input_id,
                message="Session is awaiting user input; answer it with UserInputResponse.",
            )

        if session.status == SessionStatus.INTERRUPTING:
            session = await self._finalize_interrupting_for_recovery(
                session=session,
                registered_agent=registered_agent,
                registered_environment=registered_environment,
                environment_name=environment_name,
                events=events,
            )
            actions.append(
                IncompleteSessionRecoveryAction.FINALIZED_INTERRUPT
                if previous_status == SessionStatus.INTERRUPTING
                else IncompleteSessionRecoveryAction.INTERRUPTED_ABANDONED
            )
        elif not actions:
            actions.append(IncompleteSessionRecoveryAction.SKIPPED_TERMINAL)

        message = "Recovered incomplete session."
        if actions == [IncompleteSessionRecoveryAction.SKIPPED_TERMINAL]:
            message = "Session is terminal; recovery skipped."
        return IncompleteSessionRecoveryResult(
            session_id=session.id,
            previous_status=previous_status,
            status=session.status,
            actions=tuple(actions),
            events=tuple(events),
            message=message,
        )

    async def _finalize_interrupting_for_recovery(
        self,
        *,
        session: Session,
        registered_agent: runtime_records.RegisteredAgentState,
        registered_environment: runtime_records.RegisteredEnvironment | None,
        environment_name: str | None,
        events: list[Event],
    ) -> Session:
        if session.status == SessionStatus.INTERRUPTING:
            async for event in self._interrupt_session_for_recovery(
                RecoveryInterruptionRequest(
                    session=session,
                    registered_agent=registered_agent,
                    registered_environment=registered_environment,
                    environment_name=environment_name,
                )
            ):
                events.append(event)
            session = await self._require_session(session.id)
        return session

    async def _require_session(self, session_id: str) -> Session:
        loaded = await self._session_store.load(session_id)
        if loaded is None:
            raise KeyError(f"Session not found: {session_id}") from None
        return loaded

    async def _clear_pending_tool_round_if_matches(
        self,
        session_id: str,
        pending_round: tool_round_recovery.PendingToolRound,
    ) -> None:
        checkpoint = await self._session_store.load_checkpoint(session_id)
        if checkpoint is None:
            return
        copied_checkpoint = copy_json_value(checkpoint, "checkpoint")
        current = tool_round_recovery.pending_tool_round_from_checkpoint(copied_checkpoint)
        if current is None or current.round_id != pending_round.round_id:
            return
        copied_checkpoint.pop(tool_round_recovery.PENDING_TOOL_ROUND_CHECKPOINT_KEY, None)
        await self._session_store.transform_checkpoint(
            session_id,
            self._checkpoint_transform(copied_checkpoint),
        )

    async def _subagent_children_by_idempotency_key(
        self,
        parent_session_id: str,
    ) -> dict[str, Session]:
        children: dict[str, Session] = {}
        sessions = await query_all_sessions(
            self._session_store,
            SessionQuery(
                parent_session_id=parent_session_id,
                order_by=SessionOrder.CREATED_AT_ASC,
            ),
        )
        for child in sessions:
            if not _is_background_subagent_session(child):
                continue
            idempotency_key = tool_round_recovery.subagent_child_idempotency_key(child)
            if idempotency_key is not None:
                children[idempotency_key] = child
        return children

    @staticmethod
    def _reattached_subagent_result(
        children: dict[str, Session],
        idempotency_key: str,
        *,
        tool_call_id: str,
        tool_name: str,
        tool_round_id: str,
    ) -> ToolResult | None:
        child = children.get(idempotency_key)
        if child is None:
            return None
        return tool_round_recovery.recovered_subagent_tool_result(
            tool_call_id=tool_call_id,
            tool_name=tool_name,
            tool_round_id=tool_round_id,
            child=child,
        )


def _checkpoint_without_active_incomplete_recovery_claim(
    checkpoint: dict[str, Any] | None,
    *,
    now: datetime,
) -> dict[str, Any] | None:
    """Reject live recovery ownership and remove an expired internal marker."""
    _require_aware_datetime(now, "now")
    if checkpoint is None:
        return None
    updated = copy_json_value(checkpoint, "checkpoint")
    existing = _incomplete_recovery_claim_from_checkpoint(updated)
    if existing is None:
        return updated
    if existing[1] > now:
        raise RuntimeError("Session has an active incomplete-session recovery operation.")
    updated.pop(_INCOMPLETE_RECOVERY_CLAIM_CHECKPOINT_KEY, None)
    return updated


def _require_aware_datetime(value: datetime, field_name: str) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{field_name} must be timezone-aware.")
    return value


def _interrupted_tool_round_results(
    *,
    tool_calls: list[runtime_records.ToolCallRequest],
    completed_outcomes: list[runtime_records.ToolCallOutcome],
    tool_round_id: str | None = None,
    cancellation_artifacts: list[dict[str, Any]] | None = None,
    cancellation_artifacts_by_id: dict[str, list[dict[str, Any]]] | None = None,
) -> list[runtime_records.ToolCallOutcome]:
    completed_ids = {outcome.call.id for outcome in completed_outcomes}
    artifacts_for_interrupted_tool = (
        [] if cancellation_artifacts is None else cancellation_artifacts
    )
    interrupted_outcomes: list[runtime_records.ToolCallOutcome] = []
    for tool_call in tool_calls:
        if tool_call.id in completed_ids:
            continue
        if cancellation_artifacts_by_id is not None:
            result_artifacts = cancellation_artifacts_by_id.get(tool_call.id, [])
        else:
            result_artifacts = artifacts_for_interrupted_tool
            artifacts_for_interrupted_tool = []
        structured = {
            "interrupted": True,
            "tool_call_id": tool_call.id,
            "tool_name": tool_call.name,
        }
        if tool_round_id is not None:
            structured["tool_round_id"] = tool_round_id
        interrupted_outcomes.append(
            runtime_records.ToolCallOutcome(
                call=tool_call,
                result=ToolResult(
                    content="Tool call interrupted before completion.",
                    structured=structured,
                    artifacts=result_artifacts,
                    is_error=True,
                ),
            )
        )
    return interrupted_outcomes


def _interrupted_tool_call_event(
    *,
    session: Session,
    registered_agent: runtime_records.RegisteredAgentState,
    registered_environment: runtime_records.RegisteredEnvironment | None,
    tool_call_outcome: runtime_records.ToolCallOutcome,
    tool_round_id: str | None = None,
) -> Event:
    payload = {
        "tool_call_id": tool_call_outcome.call.id,
        "idempotency_key": tool_execution.tool_idempotency_key(
            session_id=session.id,
            tool_round_id=tool_round_id,
            tool_call_id=tool_call_outcome.call.id,
        ),
        "result": tool_call_outcome.result.model_dump(),
    }
    if tool_round_id is not None:
        payload["tool_round_id"] = tool_round_id
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
        payload=payload,
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
