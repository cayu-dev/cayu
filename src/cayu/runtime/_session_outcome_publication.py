"""Interaction outcome publication composed below ordinary turn scheduling."""

from __future__ import annotations

import contextlib
from collections.abc import AsyncGenerator, AsyncIterator, Awaitable, Callable
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from cayu._exception_groups import (
    exception_cause,
    iter_exception_tree,
    set_exception_cause,
)
from cayu._validation import (
    copy_json_value,
)
from cayu.approvals.user_input import (
    event_with_pending_user_input_authority,
    pending_user_input_interruption_payload,
)
from cayu.context.base import (
    automatic_compaction_failure_payload,
)
from cayu.deadlines import (
    ExecutionDeadlineExceeded,
)
from cayu.events import (
    Event,
    EventType,
    copy_event,
    event_with_runtime_generated_id,
    event_with_runtime_payload_authority,
)
from cayu.exceptions import (
    _is_runtime_interaction_lifecycle_publication_rejection,
)
from cayu.execution_profiles import (
    ExecutionProfileIdentity,
    event_with_execution_profile_authority,
)
from cayu.failure_evidence import exception_evidence
from cayu.observability.hooks import (
    RuntimeHookPhase,
)
from cayu.providers.base import (
    ModelStreamDeadlineError,
)
from cayu.runtime import _approval_support as approval_support
from cayu.runtime import _resume_ledger as resume_ledger
from cayu.runtime import _runtime_records as runtime_records
from cayu.runtime._continuation_task_failure import (
    RuntimeTaskFailureIdentity,
    RuntimeTaskFailureTerminalizationPending,
    retain_runtime_task_failure_terminalization_pending,
    runtime_task_failure_event_id,
    runtime_task_failure_payload,
    runtime_task_failure_terminalization_request_sha256,
)
from cayu.runtime._diagnostics import (
    ExceptionDiagnostic,
    exception_diagnostic,
    task_failure_payload_from_diagnostic,
    task_update_error_payload,
)
from cayu.runtime._durable_subagents import (
    is_durable_subagent_submission_unsettled,
)
from cayu.runtime._environment_lifecycle import (
    EnvironmentBindingFinalizeResult,
    EnvironmentLifecycle,
    exception_failure_payload,
)
from cayu.runtime._event_writer import (
    RuntimeEventWriter,
)
from cayu.runtime._foreground_child_wait import (
    ForegroundChildActionRequired,
    event_with_foreground_child_wait_authority,
)
from cayu.runtime._foreground_gate_continuation import ForegroundGatePolicyOwner
from cayu.runtime._interaction_events import (
    build_interaction_started_event,
)
from cayu.runtime._interruption_coordinator import (
    BackgroundInterruptionCoordinator,
)
from cayu.runtime._invocation_lifecycle import (
    InvocationContext,
)
from cayu.runtime._provider_operation_start_owner import is_ambiguous_provider_operation_start_error
from cayu.runtime._run_limits import (
    ModelCompletionBudgetSettlementPending,
    SessionUsageTracker,
    budget_provider_cleanup_failure,
)
from cayu.runtime._session_control import (
    ActiveSessionRun,
    SessionControl,
)
from cayu.runtime._session_finalization import (
    _INTERRUPTION_TYPE_RUNTIME_INTERRUPTED,
    SessionFinalization,
    _interaction_transition_failure_diagnostics,
    _interaction_transition_replay_failures,
    _is_interaction_transition_run_fence,
    _raise_primary_with_secondary_failure,
    _task_event,
)
from cayu.runtime._session_task_lifecycle import SessionTaskLifecycle
from cayu.runtime._terminal_event_publication import (
    TerminalEventPublication,
)
from cayu.runtime._tool_invocation.admission import ToolApprovalRequired
from cayu.runtime._tool_round_executor import (
    UserInputRequired,
)
from cayu.runtime._work_attempt_session_mutation import (
    record_work_attempt_execution_stop,
)
from cayu.runtime.loop_policies import (
    BeforeStopDecision,
    LoopPolicy,
)
from cayu.runtime.provider_operations import (
    ProviderOperationInspectionStatus,
    inspect_provider_operation,
)
from cayu.runtime.tool_completion import (
    ToolCompletionResult,
)
from cayu.sessions import _tool_call_evidence as tool_call_evidence
from cayu.sessions._checkpoint_mutations import (
    SessionRuntimePublicationConflict,
)
from cayu.sessions._checkpoint_preservation import (
    _invocation_lifecycle_authority_mutation_scope,
)
from cayu.sessions._execution_profile_checkpoint import (
    active_invocation_execution_profile_from_checkpoint,
)
from cayu.sessions._invocation_scope import (
    _current_session_interaction_id,
)
from cayu.sessions._invocation_terminal_decision import (
    InvocationTerminalDecision,
    InvocationTerminalOutcome,
    build_invocation_terminal_decision,
    checkpoint_with_invocation_terminal_decision,
    invocation_terminal_decision_from_checkpoint,
    invocation_terminal_decision_matches_active_profile,
    settled_invocation_terminal_decision_from_checkpoint,
)
from cayu.sessions._terminal_evidence import (
    _INTERRUPTION_TYPE_TOOL_APPROVAL_REQUIRED,
    _INTERRUPTION_TYPE_USER_INPUT_REQUIRED,
    interruption_request_id_from_payload,
)
from cayu.sessions.base import SessionRunFenced
from cayu.sessions.event_queries import EventOrder, EventQuery
from cayu.sessions.interactions import (
    INTERACTION_LIFECYCLE_EVENT_TYPES,
    INTERACTION_TERMINAL_EVENT_TYPES,
    InteractionStatus,
)
from cayu.sessions.records import (
    Session,
    SessionStatus,
    SessionStatusConflict,
)
from cayu.sessions.store import SessionStore
from cayu.tasks.store import TaskStore
from cayu.vaults.redaction import SecretRedactor


@dataclass(slots=True)
class SessionOutcomeState:
    """Invocation-local session and task progress shared across outcome yields."""

    session: Session
    task_started: bool = False
    task_start_attempted: bool = False
    task_finished: bool = False


@dataclass(frozen=True, slots=True)
class SessionOutcomeContext:
    """Current runtime bindings captured when an interaction outcome is published."""

    state: SessionOutcomeState
    registered_agent: runtime_records.RegisteredAgentState
    registered_environment: runtime_records.RegisteredEnvironment | None
    execution_profile: ExecutionProfileIdentity
    invocation_context: InvocationContext
    environment_name: str | None
    run_started_at: float
    turn_usage_tracker: SessionUsageTracker
    active_run: ActiveSessionRun[SessionUsageTracker] | None
    task_id: str | None
    task_worker_id: str | None
    task_lease_expires_at: datetime | None
    task_handoff_id: str | None
    request_loop_policies: tuple[LoopPolicy, ...]


class SessionOutcomePublication:
    """Sequence interaction outcomes, task settlement and terminal hook publication."""

    def __init__(
        self,
        *,
        session_store: SessionStore,
        task_store: TaskStore | None,
        session_tasks: SessionTaskLifecycle,
        session_finalization: SessionFinalization,
        terminal_event_publication: TerminalEventPublication,
        environment_lifecycle: EnvironmentLifecycle,
        event_writer: RuntimeEventWriter,
        session_control: SessionControl[SessionUsageTracker],
        foreground_gate_policy_owner: ForegroundGatePolicyOwner,
        background_interruption_coordinator: BackgroundInterruptionCoordinator,
        secret_redactor: SecretRedactor,
        clock: Callable[[], datetime],
    ) -> None:
        self.session_store = session_store
        self.task_store = task_store
        self._session_tasks = session_tasks
        self._session_finalization = session_finalization
        self._terminal_event_publication = terminal_event_publication
        self._environment_lifecycle = environment_lifecycle
        self._event_writer = event_writer
        self._session_control = session_control
        self._foreground_gate_policy_owner = foreground_gate_policy_owner
        self._background_interruption_coordinator = background_interruption_coordinator
        self._secret_redactor = secret_redactor
        self._clock = clock

    async def complete_session(
        self,
        context: SessionOutcomeContext,
        completed_session: Session,
        completion_finalization_marker: dict[str, Any] | None,
        tool_completion_result: ToolCompletionResult | None,
    ) -> AsyncGenerator[Event, None]:
        state = context.state
        registered_agent = context.registered_agent
        registered_environment = context.registered_environment
        execution_profile = context.execution_profile
        invocation_context = context.invocation_context
        environment_name = context.environment_name
        run_started_at = context.run_started_at
        turn_usage_tracker = context.turn_usage_tracker
        active_run = context.active_run
        task_id = context.task_id
        task_worker_id = context.task_worker_id
        task_handoff_id = context.task_handoff_id

        def completion_payload() -> dict[str, Any]:
            if tool_completion_result is None:
                return {}
            return {
                "reason": tool_completion_result.reason,
                "tool_completion": tool_completion_result.model_dump(mode="json"),
            }

        state.session = completed_session
        prefinalized_completion: EnvironmentBindingFinalizeResult | None = None
        binding = (
            None if registered_environment is None else registered_environment.environment.binding
        )
        if (
            binding is not None
            and registered_environment is not None
            and registered_environment.bound_workspace is not None
            and binding._completion_requires_successful_finalization(
                registered_environment.bound_workspace
            )
        ):
            if execution_profile is None:
                raise RuntimeError(
                    "Completion-critical workspace finalization lost its execution profile."
                )
            completion_marker = completion_finalization_marker
            if completion_marker is None:
                raise RuntimeError(
                    "Completion-critical workspace finalization lost its durable marker."
                )
            completion_event = await self._terminal_event_publication.bind_to_run_operation(
                Event(
                    type=EventType.SESSION_COMPLETED,
                    payload=completion_payload(),
                    session_id=state.session.id,
                    agent_name=registered_agent.spec.name,
                    environment_name=environment_name,
                ),
                session=state.session,
            )
            prefinalized_completion = await self._environment_lifecycle.finalize_terminal_event(
                event=completion_event,
                session=state.session,
                registered_environment=registered_environment,
                execution_profile=execution_profile,
                invocation_context=invocation_context,
            )
            finalize_error = prefinalized_completion.event.payload.get("binding_finalize_error")
            if type(finalize_error) is dict:
                # Capture conditional-write progress made before a partial
                # failure so a restarted worker resumes at the exact
                # revision boundary. A stale pre-attempt marker remains
                # replay-safe if this refresh itself cannot be acknowledged.
                completion_marker = (
                    await self._environment_lifecycle.checkpoint_completion_finalization(
                        session=state.session,
                        registered_environment=registered_environment,
                        execution_profile=execution_profile,
                        task_id=task_id,
                    )
                )
                error_message = finalize_error.get("error")
                error_type = finalize_error.get("error_type")
                if type(error_message) is not str or type(error_type) is not str:
                    raise RuntimeError(
                        "Workspace finalization failure lost its durable diagnostic."
                    )
                if task_id is not None and invocation_context.work_attempt is None:
                    task = await self._session_finalization.fail_task(
                        task_id=task_id,
                        task_worker_id=task_worker_id,
                        task_handoff_id=task_handoff_id,
                        session=state.session,
                        error=task_failure_payload_from_diagnostic(
                            ExceptionDiagnostic(
                                message=error_message,
                                error_type=error_type,
                            ),
                            session_id=state.session.id,
                            additional_fields={
                                "phase": "workspace_finalize",
                                "workspace_output_committed": False,
                            },
                        ),
                    )
                    state.task_finished = True
                    if active_run is not None:
                        active_run.task_finished = True
                    yield await self._event_writer.emit(
                        _task_event(
                            event_type=EventType.TASK_FAILED,
                            task=task,
                            session=state.session,
                            registered_agent=registered_agent,
                            registered_environment=registered_environment,
                        )
                    )
                (
                    state.session,
                    interaction_failed_event,
                    _,
                ) = await self._session_finalization.publish_sibling_interaction_transition(
                    session=state.session,
                    invocation_context=invocation_context,
                    registered_agent=registered_agent,
                    registered_environment=registered_environment,
                    environment_name=environment_name,
                    to_status=SessionStatus.FAILED,
                    execution_profile=execution_profile,
                )
                if interaction_failed_event is not None:
                    yield interaction_failed_event
                for failed_turn_event in await self._session_finalization.emit_turn_completed_once(
                    session=state.session,
                    registered_agent=registered_agent,
                    environment_name=environment_name,
                    status=SessionStatus.FAILED,
                    run_started_at=run_started_at,
                    usage_tracker=turn_usage_tracker,
                    active_run=active_run,
                    invocation_context=invocation_context,
                ):
                    yield failed_turn_event
                failed_payload = copy_json_value(
                    prefinalized_completion.event.payload,
                    "payload",
                )
                failed_payload.update(
                    {
                        "error": error_message,
                        "error_type": error_type,
                        "failure_phase": "workspace_finalize",
                        "workspace_output_committed": False,
                    }
                )
                failed_terminal = copy_event(prefinalized_completion.event).model_copy(
                    update={
                        "type": EventType.SESSION_FAILED,
                        "payload": failed_payload,
                    },
                    deep=True,
                )
                failed_finalize_result = EnvironmentBindingFinalizeResult(
                    event=failed_terminal,
                    events=prefinalized_completion.events,
                    cancellation=prefinalized_completion.cancellation,
                    cancellation_requests_consumed=(
                        prefinalized_completion.cancellation_requests_consumed
                    ),
                )
                terminal_stream = self._terminal_event_publication.emit(
                    event=failed_terminal,
                    phase=RuntimeHookPhase.AFTER_SESSION_FAILED,
                    session=state.session,
                    registered_agent=registered_agent,
                    registered_environment=registered_environment,
                    execution_profile=execution_profile,
                    invocation_context=invocation_context,
                    environment_finalize_result=failed_finalize_result,
                )
                async with contextlib.aclosing(terminal_stream) as owned_terminal_stream:
                    async for failed_terminal_event in owned_terminal_stream:
                        yield failed_terminal_event
                return
            state.session = await self._environment_lifecycle.complete_completion_finalization(
                session=state.session,
                expected_marker=completion_marker,
            )
        if task_id is not None and invocation_context.work_attempt is None:
            await self._session_control.raise_if_interrupted(state.session.id)
            task = await self._session_tasks.complete_task(
                task_id=task_id,
                task_worker_id=task_worker_id,
                task_handoff_id=task_handoff_id,
                session=state.session,
                registered_agent=registered_agent,
                registered_environment=registered_environment,
            )
            state.task_finished = True
            if active_run is not None:
                active_run.task_finished = True
            yield await self._event_writer.emit(
                _task_event(
                    event_type=EventType.TASK_COMPLETED,
                    task=task,
                    session=state.session,
                    registered_agent=registered_agent,
                    registered_environment=registered_environment,
                )
            )
        for event in await self._session_finalization.emit_turn_completed_once(
            session=state.session,
            registered_agent=registered_agent,
            environment_name=environment_name,
            status=SessionStatus.COMPLETED,
            run_started_at=run_started_at,
            usage_tracker=turn_usage_tracker,
            active_run=active_run,
            invocation_context=invocation_context,
        ):
            yield event
        terminal_stream = self._terminal_event_publication.emit(
            event=(
                prefinalized_completion.event
                if prefinalized_completion is not None
                else Event(
                    type=EventType.SESSION_COMPLETED,
                    payload=completion_payload(),
                    session_id=state.session.id,
                    agent_name=registered_agent.spec.name,
                    environment_name=environment_name,
                )
            ),
            phase=RuntimeHookPhase.AFTER_SESSION_COMPLETED,
            session=state.session,
            registered_agent=registered_agent,
            registered_environment=registered_environment,
            execution_profile=execution_profile,
            invocation_context=invocation_context,
            environment_finalize_result=prefinalized_completion,
        )
        async with contextlib.aclosing(terminal_stream) as owned_terminal_stream:
            async for event in owned_terminal_stream:
                yield event

    async def publish_stop_interruption(
        self,
        context: SessionOutcomeContext,
        decision: BeforeStopDecision,
    ) -> AsyncIterator[Event]:
        state = context.state
        registered_agent = context.registered_agent
        registered_environment = context.registered_environment
        execution_profile = context.execution_profile
        invocation_context = context.invocation_context
        environment_name = context.environment_name
        run_started_at = context.run_started_at
        turn_usage_tracker = context.turn_usage_tracker
        active_run = context.active_run

        (
            state.session,
            interrupted,
            _,
        ) = await self._session_finalization.publish_interaction_transition(
            session=state.session,
            invocation_context=invocation_context,
            agent_name=registered_agent.spec.name,
            environment_name=environment_name,
            to_status=SessionStatus.INTERRUPTED,
            execution_profile=execution_profile,
        )
        if interrupted is not None:
            yield interrupted
        for event in await self._session_finalization.emit_turn_completed_once(
            session=state.session,
            registered_agent=registered_agent,
            environment_name=environment_name,
            status=SessionStatus.INTERRUPTED,
            run_started_at=run_started_at,
            usage_tracker=turn_usage_tracker,
            active_run=active_run,
            invocation_context=invocation_context,
        ):
            yield event
        async for event in self._terminal_event_publication.emit(
            event=Event(
                type=EventType.SESSION_INTERRUPTED,
                session_id=state.session.id,
                agent_name=registered_agent.spec.name,
                environment_name=environment_name,
                payload={
                    "interruption_type": _INTERRUPTION_TYPE_RUNTIME_INTERRUPTED,
                    "reason": decision.reason,
                    "policy_metadata": copy_json_value(decision.metadata, "policy_metadata"),
                },
            ),
            phase=RuntimeHookPhase.AFTER_SESSION_INTERRUPTED,
            session=state.session,
            registered_agent=registered_agent,
            registered_environment=registered_environment,
            execution_profile=execution_profile,
            invocation_context=invocation_context,
        ):
            yield event

    async def start_linked_task_if_needed(
        self,
        context: SessionOutcomeContext,
        *,
        only_if_exists: bool = False,
    ) -> Event | None:
        state = context.state
        registered_agent = context.registered_agent
        registered_environment = context.registered_environment
        invocation_context = context.invocation_context
        active_run = context.active_run
        task_id = context.task_id
        task_worker_id = context.task_worker_id
        task_lease_expires_at = context.task_lease_expires_at
        task_handoff_id = context.task_handoff_id

        if task_id is None or state.task_started or invocation_context.work_attempt is not None:
            return None
        if self.task_store is None:
            raise RuntimeError("task_store is required when RunRequest.task_id is set.")
        if only_if_exists and await self.task_store.load_task(task_id) is None:
            return None
        state.task_start_attempted = True
        task = await self._session_tasks.start_task(
            task_id=task_id,
            session=state.session,
            worker_id=task_worker_id,
            lease_expires_at=task_lease_expires_at,
            handoff_id=task_handoff_id,
        )
        state.task_started = True
        if active_run is not None:
            active_run.task_started = True
        from cayu.runtime._task_group_invocation import bind_invocation

        await bind_invocation(
            self.task_store,
            task_id=task_id,
            context=invocation_context,
            redactor=self._secret_redactor,
        )
        return await self._event_writer.emit(
            _task_event(
                event_type=EventType.TASK_STARTED,
                task=task,
                session=state.session,
                registered_agent=registered_agent,
                registered_environment=registered_environment,
            )
        )

    async def pause_for_tool_approval(
        self,
        context: SessionOutcomeContext,
        exc: ToolApprovalRequired,
        materialize_deferred_messages_after_failure: Callable[[], Awaitable[None]],
    ) -> AsyncGenerator[Event, None]:
        state = context.state
        registered_agent = context.registered_agent
        registered_environment = context.registered_environment
        execution_profile = context.execution_profile
        invocation_context = context.invocation_context
        environment_name = context.environment_name
        run_started_at = context.run_started_at
        turn_usage_tracker = context.turn_usage_tracker
        active_run = context.active_run

        await materialize_deferred_messages_after_failure()
        (
            state.session,
            interaction_paused_event,
            _,
        ) = await self._session_finalization.publish_sibling_interaction_transition(
            session=state.session,
            invocation_context=invocation_context,
            registered_agent=registered_agent,
            registered_environment=registered_environment,
            environment_name=environment_name,
            to_status=SessionStatus.INTERRUPTED,
            execution_profile=execution_profile,
        )
        if interaction_paused_event is not None:
            yield interaction_paused_event
        for event in await self._session_finalization.emit_turn_completed_once(
            session=state.session,
            registered_agent=registered_agent,
            environment_name=environment_name,
            status=SessionStatus.INTERRUPTED,
            run_started_at=run_started_at,
            usage_tracker=turn_usage_tracker,
            active_run=active_run,
            invocation_context=invocation_context,
        ):
            yield event
        async for event in self._terminal_event_publication.emit(
            event=approval_support.event_with_pending_approval_authority(
                Event(
                    type=EventType.SESSION_INTERRUPTED,
                    session_id=state.session.id,
                    agent_name=registered_agent.spec.name,
                    environment_name=environment_name,
                    payload={
                        "interruption_type": _INTERRUPTION_TYPE_TOOL_APPROVAL_REQUIRED,
                        "model_step_id": exc.approval.model_step_id,
                        "model_attempt_id": exc.approval.model_attempt_id,
                        "tool_round_id": exc.approval.tool_round_id,
                        **approval_support.bounded_pending_approval_event_payload(
                            exc.approval,
                            redactor=self._secret_redactor,
                        ),
                    },
                ),
                exc.approval,
            ),
            phase=RuntimeHookPhase.AFTER_SESSION_INTERRUPTED,
            session=state.session,
            registered_agent=registered_agent,
            registered_environment=registered_environment,
            execution_profile=execution_profile,
            invocation_context=invocation_context,
        ):
            yield event

    async def pause_for_user_input(
        self,
        context: SessionOutcomeContext,
        exc: UserInputRequired,
        materialize_deferred_messages_after_failure: Callable[[], Awaitable[None]],
    ) -> AsyncGenerator[Event, None]:
        state = context.state
        registered_agent = context.registered_agent
        registered_environment = context.registered_environment
        execution_profile = context.execution_profile
        invocation_context = context.invocation_context
        environment_name = context.environment_name
        run_started_at = context.run_started_at
        turn_usage_tracker = context.turn_usage_tracker
        active_run = context.active_run

        await materialize_deferred_messages_after_failure()
        (
            state.session,
            interaction_paused_event,
            _,
        ) = await self._session_finalization.publish_sibling_interaction_transition(
            session=state.session,
            invocation_context=invocation_context,
            registered_agent=registered_agent,
            registered_environment=registered_environment,
            environment_name=environment_name,
            to_status=SessionStatus.INTERRUPTED,
            execution_profile=execution_profile,
        )
        if interaction_paused_event is not None:
            yield interaction_paused_event
        for event in await self._session_finalization.emit_turn_completed_once(
            session=state.session,
            registered_agent=registered_agent,
            environment_name=environment_name,
            status=SessionStatus.INTERRUPTED,
            run_started_at=run_started_at,
            usage_tracker=turn_usage_tracker,
            active_run=active_run,
            invocation_context=invocation_context,
        ):
            yield event
        async for event in self._terminal_event_publication.emit(
            event=event_with_pending_user_input_authority(
                Event(
                    type=EventType.SESSION_INTERRUPTED,
                    session_id=state.session.id,
                    agent_name=registered_agent.spec.name,
                    environment_name=environment_name,
                    payload={
                        "interruption_type": _INTERRUPTION_TYPE_USER_INPUT_REQUIRED,
                        "model_step_id": exc.pending.model_step_id,
                        "model_attempt_id": exc.pending.model_attempt_id,
                        "tool_round_id": exc.pending.tool_round_id,
                        **pending_user_input_interruption_payload(exc.pending),
                    },
                ),
                exc.pending,
            ),
            phase=RuntimeHookPhase.AFTER_SESSION_INTERRUPTED,
            session=state.session,
            registered_agent=registered_agent,
            registered_environment=registered_environment,
            execution_profile=execution_profile,
            invocation_context=invocation_context,
        ):
            yield event

    async def pause_for_foreground_child(
        self,
        context: SessionOutcomeContext,
        exc: ForegroundChildActionRequired,
        materialize_deferred_messages_after_failure: Callable[[], Awaitable[None]],
    ) -> AsyncGenerator[Event, None]:
        state = context.state
        registered_agent = context.registered_agent
        registered_environment = context.registered_environment
        execution_profile = context.execution_profile
        invocation_context = context.invocation_context
        environment_name = context.environment_name
        run_started_at = context.run_started_at
        turn_usage_tracker = context.turn_usage_tracker
        active_run = context.active_run
        request_loop_policies = context.request_loop_policies

        self._foreground_gate_policy_owner.retain_wait(
            state.session, exc.wait, request_loop_policies
        )
        await materialize_deferred_messages_after_failure()
        (
            state.session,
            interaction_paused_event,
            _,
        ) = await self._session_finalization.publish_sibling_interaction_transition(
            session=state.session,
            invocation_context=invocation_context,
            registered_agent=registered_agent,
            registered_environment=registered_environment,
            environment_name=environment_name,
            to_status=SessionStatus.INTERRUPTED,
            execution_profile=execution_profile,
        )
        if interaction_paused_event is not None:
            yield interaction_paused_event
        for event in await self._session_finalization.emit_turn_completed_once(
            session=state.session,
            registered_agent=registered_agent,
            environment_name=environment_name,
            status=SessionStatus.INTERRUPTED,
            run_started_at=run_started_at,
            usage_tracker=turn_usage_tracker,
            active_run=active_run,
            invocation_context=invocation_context,
        ):
            yield event
        async for event in self._terminal_event_publication.emit(
            event=event_with_foreground_child_wait_authority(
                Event(
                    type=EventType.SESSION_INTERRUPTED,
                    session_id=state.session.id,
                    agent_name=registered_agent.spec.name,
                    environment_name=environment_name,
                    payload=exc.interruption_evidence(),
                ),
                exc.wait,
            ),
            phase=RuntimeHookPhase.AFTER_SESSION_INTERRUPTED,
            session=state.session,
            registered_agent=registered_agent,
            registered_environment=registered_environment,
            execution_profile=execution_profile,
            invocation_context=invocation_context,
        ):
            yield event

    async def fail(
        self,
        context: SessionOutcomeContext,
        exc: Exception,
        materialize_deferred_messages_after_failure: Callable[[], Awaitable[None]],
    ) -> AsyncGenerator[Event, None]:
        state = context.state
        registered_agent = context.registered_agent
        registered_environment = context.registered_environment
        execution_profile = context.execution_profile
        invocation_context = context.invocation_context
        environment_name = context.environment_name
        run_started_at = context.run_started_at
        turn_usage_tracker = context.turn_usage_tracker
        active_run = context.active_run
        task_id = context.task_id
        task_worker_id = context.task_worker_id
        task_handoff_id = context.task_handoff_id

        if (
            isinstance(exc, ExecutionDeadlineExceeded)
            and invocation_context.work_attempt is not None
        ):
            try:
                if self.task_store is None:
                    raise RuntimeError("Governed deadline stop requires its task store.")
                await record_work_attempt_execution_stop(
                    self.task_store,
                    admission=invocation_context.work_attempt.admission,
                    reason="elapsed_limit",
                    redactor=self._secret_redactor,
                )
            except BaseException as stop_failure:
                _raise_primary_with_secondary_failure(
                    exc,
                    stop_failure,
                    group_message="Execution deadline and durable stop publication failed.",
                )
        if any(
            isinstance(candidate, ModelStreamDeadlineError)
            for candidate in iter_exception_tree(exc)
        ):
            # A dispatched synchronous request has an unknown external outcome.
            # Keep the open interaction and exact model stage fenced so only
            # incomplete-session recovery followed by explicit stage settlement
            # can release or replace its authority.
            raise exc
        ambiguous_provider_start = is_ambiguous_provider_operation_start_error(exc)
        if not ambiguous_provider_start:
            provider_operation = await inspect_provider_operation(
                self.session_store,
                state.session.id,
            )
            ambiguous_provider_start = (
                provider_operation.status is ProviderOperationInspectionStatus.AMBIGUOUS_SUBMISSION
            )
        if ambiguous_provider_start:
            (
                interrupted_session,
                interaction_interrupted_event,
                _,
            ) = await self._session_finalization.publish_sibling_interaction_transition(
                session=state.session,
                invocation_context=invocation_context,
                registered_agent=registered_agent,
                registered_environment=registered_environment,
                environment_name=environment_name,
                to_status=SessionStatus.INTERRUPTED,
                execution_profile=execution_profile,
                from_statuses={
                    SessionStatus.RUNNING,
                    SessionStatus.INTERRUPTING,
                },
            )
            if interaction_interrupted_event is not None:
                yield interaction_interrupted_event
            async for event in self._terminal_event_publication.emit(
                event=Event(
                    type=EventType.SESSION_INTERRUPTED,
                    session_id=state.session.id,
                    agent_name=registered_agent.spec.name,
                    environment_name=environment_name,
                    payload={
                        "interruption_type": "provider_operation_unavailable",
                        "recovery_reason": "ambiguous_submission",
                        "duplicate_request_risk": True,
                    },
                ),
                phase=RuntimeHookPhase.AFTER_SESSION_INTERRUPTED,
                session=interrupted_session,
                registered_agent=registered_agent,
                registered_environment=registered_environment,
                execution_profile=execution_profile,
                invocation_context=invocation_context,
            ):
                yield event
            return
        if is_durable_subagent_submission_unsettled(
            exc,
            parent_session_id=state.session.id,
        ):
            raise exc
        if _is_interaction_transition_run_fence(exc):
            raise exc
        if _is_runtime_interaction_lifecycle_publication_rejection(
            exc,
            session_id=state.session.id,
            interaction_id=_current_session_interaction_id(state.session.id),
            runtime_authority=self._session_finalization.interaction_publication_authority,
        ):
            # A terminal interaction cannot be represented using the observed
            # wall clock. Preserve the live session, linked task, and durable
            # run-operation marker so normal incomplete-session recovery owns
            # the eventual terminal classification.
            raise exc
        try:
            failure_interaction_observed_at = (
                await self._session_finalization.failure_interaction_observed_at(state.session)
            )
        except Exception as preflight_failure:
            _raise_primary_with_secondary_failure(
                exc,
                preflight_failure,
                group_message=(
                    "Run failure evidence and interaction terminalization preflight failure."
                ),
            )
        failure_diagnostic = exception_diagnostic(
            exc,
            redactor=self._secret_redactor,
        )
        runtime_failure_identity: RuntimeTaskFailureIdentity | None = None
        failure_interaction_id = _current_session_interaction_id(state.session.id)
        if (
            task_id is not None
            and invocation_context.work_attempt is None
            and failure_interaction_id is not None
            and failure_interaction_observed_at is not None
        ):
            runtime_failure_identity = RuntimeTaskFailureIdentity(
                task_id=task_id,
                session_id=state.session.id,
                session_instance_id=state.session.instance_id,
                run_epoch=state.session.run_epoch,
                interaction_id=failure_interaction_id,
                execution_profile_fingerprint=execution_profile.fingerprint,
                observed_at=failure_interaction_observed_at,
            )

        def build_session_failure_payload(
            error: BaseException = exc,
        ) -> dict[str, Any]:
            failure_payload = exception_failure_payload(
                error,
                diagnostic=failure_diagnostic,
                redactor=self._secret_redactor,
            )
            failure_payload["failure_evidence"] = (
                exception_evidence(error)
                .model_copy(
                    update={
                        "session_id": state.session.id,
                        "run_epoch": state.session.run_epoch,
                        "terminal_event_id": None,
                    }
                )
                .model_dump(mode="json")
            )
            provider_cleanup_failure = budget_provider_cleanup_failure(error)
            if provider_cleanup_failure is not None:
                provider_cleanup_diagnostic = exception_diagnostic(
                    provider_cleanup_failure,
                    empty_message="provider iterator cleanup failed",
                    nonportable_message=(
                        "Provider iterator cleanup failed with a non-portable diagnostic."
                    ),
                    redactor=self._secret_redactor,
                )
                failure_payload["provider_cleanup_failure"] = {
                    **provider_cleanup_diagnostic.payload_fields(),
                    "phase": "provider_iterator_cleanup",
                }
            transition_replay_failures = _interaction_transition_replay_failures(error)
            if transition_replay_failures is not None:
                failure_payload["interaction_transition_failures"] = (
                    _interaction_transition_failure_diagnostics(
                        transition_replay_failures,
                        redactor=self._secret_redactor,
                    )
                )
            if isinstance(error, resume_ledger.ToolCallEvidenceConflict):
                failure_payload[tool_call_evidence.TOOL_EVIDENCE_CONFLICT_PAYLOAD_KEY] = True
            compaction_failure = automatic_compaction_failure_payload(
                error, redactor=self._secret_redactor
            )
            if compaction_failure is not None:
                failure_payload["compaction_failure"] = compaction_failure
            if runtime_failure_identity is not None:
                failure_payload["runtime_task_failure_id"] = runtime_failure_identity.failure_id
            return failure_payload

        payload = build_session_failure_payload() if runtime_failure_identity is not None else None
        prepared_failure_turn_completed = (
            None
            if runtime_failure_identity is None
            else await self._session_finalization.prepare_turn_completed_event(
                session=state.session,
                registered_agent=registered_agent,
                environment_name=environment_name,
                status=SessionStatus.FAILED,
                run_started_at=run_started_at,
                usage_tracker=turn_usage_tracker,
                event_id=runtime_task_failure_event_id(
                    runtime_failure_identity,
                    "turn_completed",
                ),
                timestamp=runtime_failure_identity.observed_at,
            )
        )
        prepared_runtime_task_error: dict[str, Any] | None = None
        failure_terminal_decision: InvocationTerminalDecision | None = None
        if (
            runtime_failure_identity is not None
            and state.task_started
            and not state.task_finished
            and task_id is not None
            and self.task_store is not None
        ):
            if payload is None or prepared_failure_turn_completed is None:
                raise AssertionError("Runtime failure lost its terminal evidence.") from exc
            compaction_failure = automatic_compaction_failure_payload(
                exc, redactor=self._secret_redactor
            )
            diagnostic_task_error = task_failure_payload_from_diagnostic(
                failure_diagnostic,
                session_id=state.session.id,
                additional_fields=(
                    {"compaction_failure": compaction_failure}
                    if compaction_failure is not None
                    else None
                ),
            )
            prepared_runtime_task_error = runtime_task_failure_payload(
                identity=runtime_failure_identity,
                diagnostic_payload=diagnostic_task_error,
                session_failure_payload=payload,
                turn_completed_payload=prepared_failure_turn_completed.payload,
            )
            task_terminalization_request_sha256 = (
                None
                if task_worker_id is None
                else runtime_task_failure_terminalization_request_sha256(
                    task_id=task_id,
                    task_worker_id=task_worker_id,
                    task_handoff_id=task_handoff_id,
                    session_id=state.session.id,
                    error=prepared_runtime_task_error,
                )
            )
            failure_terminal_decision = build_invocation_terminal_decision(
                outcome=InvocationTerminalOutcome.FAILED,
                session_id=state.session.id,
                session_instance_id=state.session.instance_id,
                run_epoch=state.session.run_epoch,
                profile_interaction_id=(invocation_context.active_profile.interaction_id),
                interaction_id=runtime_failure_identity.interaction_id,
                execution_profile_fingerprint=execution_profile.fingerprint,
                interaction_event_id=runtime_task_failure_event_id(
                    runtime_failure_identity,
                    "interaction_failed",
                ),
                terminal_event_id=runtime_task_failure_event_id(
                    runtime_failure_identity,
                    "session_failed",
                ),
                observed_at=runtime_failure_identity.observed_at,
                terminal_payload=payload,
                task_id=task_id,
                runtime_task_failure_id=runtime_failure_identity.failure_id,
                task_terminalization_request_sha256=(task_terminalization_request_sha256),
                task_error_payload=prepared_runtime_task_error,
                turn_completed_payload=prepared_failure_turn_completed.payload,
            )
            try:
                await self.claim_invocation_terminal_decision(
                    session=state.session,
                    decision=failure_terminal_decision,
                )
            except (SessionRunFenced, SessionStatusConflict):
                winner_session = await self.session_store.load(state.session.id)
                winner_checkpoint = await self.session_store.load_checkpoint(state.session.id)
                active_winner = invocation_terminal_decision_from_checkpoint(winner_checkpoint)
                settled_winner = settled_invocation_terminal_decision_from_checkpoint(
                    winner_checkpoint
                )
                winner = active_winner or settled_winner
                if (
                    winner_session is not None
                    and winner_session.instance_id == state.session.instance_id
                    and winner_session.run_epoch == state.session.run_epoch
                    and winner_session.status is SessionStatus.INTERRUPTING
                    and winner is None
                ):
                    pending_interrupt_payload = await self._background_interruption_coordinator.load_pending_session_interrupt_payload(
                        state.session.id,
                        default={},
                    )
                    pending_interruption_request_id = interruption_request_id_from_payload(
                        pending_interrupt_payload
                    )
                    if pending_interruption_request_id is not None:
                        active_winner = (
                            await self._session_finalization.ensure_interruption_terminal_decision(
                                session=winner_session,
                                terminal_payload=pending_interrupt_payload,
                                interruption_request_id=(pending_interruption_request_id),
                            )
                        )
                        winner = active_winner
                if (
                    winner_session is not None
                    and winner_session.instance_id == state.session.instance_id
                    and winner_session.status
                    in {SessionStatus.INTERRUPTING, SessionStatus.INTERRUPTED}
                    and winner is not None
                    and winner.outcome is InvocationTerminalOutcome.INTERRUPTED
                    and winner.session_id == state.session.id
                    and winner.session_instance_id == state.session.instance_id
                    and winner.run_epoch == state.session.run_epoch
                    and winner.interaction_id == runtime_failure_identity.interaction_id
                    and winner.execution_profile_fingerprint == execution_profile.fingerprint
                    and (winner_session.status is SessionStatus.INTERRUPTING)
                    == (active_winner is not None)
                ):
                    async for (
                        interruption_event
                    ) in self._session_finalization.handle_session_interrupted(
                        session=winner_session,
                        registered_agent=registered_agent,
                        registered_environment=registered_environment,
                        environment_name=environment_name,
                        execution_profile=execution_profile,
                        invocation_context=invocation_context,
                        run_started_at=run_started_at,
                        turn_usage_tracker=turn_usage_tracker,
                        active_run=active_run,
                    ):
                        yield interruption_event
                    return
                raise
        await materialize_deferred_messages_after_failure()
        task_failure_error: Exception | None = None
        if (
            not state.task_started
            and not state.task_start_attempted
            and task_id is not None
            and self.task_store is not None
        ):
            try:
                task_start_event = await self.start_linked_task_if_needed(
                    context, only_if_exists=True
                )
                if task_start_event is not None:
                    yield task_start_event
            except Exception as task_exc:
                task_failure_error = task_exc
        if (
            state.task_started
            and not state.task_finished
            and task_id is not None
            and self.task_store is not None
        ):
            try:
                task_error = prepared_runtime_task_error
                if task_error is None:
                    compaction_failure = automatic_compaction_failure_payload(
                        exc, redactor=self._secret_redactor
                    )
                    task_error = task_failure_payload_from_diagnostic(
                        failure_diagnostic,
                        session_id=state.session.id,
                        additional_fields=(
                            {"compaction_failure": compaction_failure}
                            if compaction_failure is not None
                            else None
                        ),
                    )
                task = await self._session_finalization.fail_task(
                    task_id=task_id,
                    task_worker_id=task_worker_id,
                    task_handoff_id=task_handoff_id,
                    session=state.session,
                    error=task_error,
                )
                state.task_finished = True
                if active_run is not None:
                    active_run.task_finished = True
                task_failed_event = _task_event(
                    event_type=EventType.TASK_FAILED,
                    task=task,
                    session=state.session,
                    registered_agent=registered_agent,
                    registered_environment=registered_environment,
                )
                if runtime_failure_identity is not None:
                    task_failed_event = event_with_runtime_generated_id(
                        event_with_execution_profile_authority(
                            event_with_runtime_payload_authority(
                                task_failed_event.model_copy(
                                    update={
                                        "id": runtime_task_failure_event_id(
                                            runtime_failure_identity,
                                            "task_failed",
                                        ),
                                        "interaction_id": (runtime_failure_identity.interaction_id),
                                        "timestamp": runtime_failure_identity.observed_at,
                                        "payload": {
                                            **task_failed_event.payload,
                                            "runtime_task_failure_id": (
                                                runtime_failure_identity.failure_id
                                            ),
                                        },
                                    }
                                ),
                                "runtime_task_failure_id",
                            ),
                            execution_profile,
                        )
                    )
                yield await self._event_writer.emit(task_failed_event)
            except Exception as task_exc:
                task_failure_error = task_exc
        if (
            failure_terminal_decision is not None
            and task_failure_error is not None
            and not state.task_finished
        ):
            # Keep the elected decision active until the task mutation has
            # positively committed. Recovery can reconcile a commit-then-
            # raise result from the exact task evidence, while settling the
            # session here would discard the only owner of a pre-commit
            # failure and leave the task runnable behind a terminal session.
            assert failure_terminal_decision.runtime_task_failure_id is not None
            assert failure_terminal_decision.task_id is not None
            pending_terminalization = RuntimeTaskFailureTerminalizationPending(
                decision_id=failure_terminal_decision.decision_id,
                runtime_task_failure_id=(failure_terminal_decision.runtime_task_failure_id),
                task_id=failure_terminal_decision.task_id,
                session_id=failure_terminal_decision.session_id,
                session_instance_id=failure_terminal_decision.session_instance_id,
                run_epoch=failure_terminal_decision.run_epoch,
            )
            if not retain_runtime_task_failure_terminalization_pending(
                exc,
                pending_terminalization,
            ):
                set_exception_cause(
                    pending_terminalization,
                    BaseExceptionGroup(
                        "Session failure and linked task terminalization failure.",
                        [exc, task_failure_error],
                    ),
                )
                raise pending_terminalization from exception_cause(pending_terminalization)
            _raise_primary_with_secondary_failure(
                exc,
                task_failure_error,
                group_message=("Session failure and linked task terminalization failure."),
            )
        if failure_terminal_decision is None:
            try:
                (
                    state.session,
                    interaction_failed_event,
                    _,
                ) = await self._session_finalization.publish_sibling_interaction_transition(
                    session=state.session,
                    invocation_context=invocation_context,
                    registered_agent=registered_agent,
                    registered_environment=registered_environment,
                    environment_name=environment_name,
                    to_status=SessionStatus.FAILED,
                    observed_at=failure_interaction_observed_at,
                    event_id=(
                        None
                        if runtime_failure_identity is None
                        else runtime_task_failure_event_id(
                            runtime_failure_identity,
                            "interaction_failed",
                        )
                    ),
                    execution_profile=execution_profile,
                    model_completion_failure=exc,
                )
            except ModelCompletionBudgetSettlementPending as accounting_pending:
                _raise_primary_with_secondary_failure(
                    exc,
                    accounting_pending,
                    group_message=("Session failure and model-completion budget recovery fence."),
                )
            if interaction_failed_event is not None:
                yield interaction_failed_event
        if payload is None:
            payload = build_session_failure_payload()
        if task_failure_error is not None:
            payload.update(
                task_update_error_payload(
                    task_failure_error,
                    redactor=self._secret_redactor,
                )
            )
        failure_turn_events = await self._session_finalization.emit_turn_completed_once(
            session=state.session,
            registered_agent=registered_agent,
            environment_name=environment_name,
            status=SessionStatus.FAILED,
            run_started_at=run_started_at,
            usage_tracker=turn_usage_tracker,
            active_run=active_run,
            invocation_context=invocation_context,
            prepared_turn_completed=prepared_failure_turn_completed,
            settle_interaction=failure_terminal_decision is None,
        )
        if failure_terminal_decision is None:
            for event in failure_turn_events:
                yield event
        if isinstance(exc, ExecutionDeadlineExceeded):
            payload["execution_deadline"] = exc.diagnostics
        session_failed_event = Event(
            type=EventType.SESSION_FAILED,
            session_id=state.session.id,
            agent_name=registered_agent.spec.name,
            environment_name=environment_name,
            payload=payload,
        )
        if runtime_failure_identity is not None:
            session_failed_event = event_with_runtime_generated_id(
                event_with_execution_profile_authority(
                    event_with_runtime_payload_authority(
                        session_failed_event.model_copy(
                            update={
                                "id": runtime_task_failure_event_id(
                                    runtime_failure_identity,
                                    "session_failed",
                                ),
                                "timestamp": runtime_failure_identity.observed_at,
                            }
                        ),
                        "runtime_task_failure_id",
                    ),
                    execution_profile,
                )
            )
        terminal_finalize_result: Any | None = None
        prepared_atomic_terminal_event: Event | None = None
        if failure_terminal_decision is not None:
            (
                terminal_finalize_result,
                prepared_atomic_terminal_event,
            ) = await self._session_finalization.prepare_terminal_event_for_atomic_transition(
                event=session_failed_event,
                session=state.session,
                registered_environment=registered_environment,
                execution_profile=execution_profile,
                invocation_context=invocation_context,
            )
            if (
                prepared_atomic_terminal_event.id != failure_terminal_decision.terminal_event_id
                or prepared_atomic_terminal_event.timestamp != failure_terminal_decision.observed_at
            ):
                raise SessionRuntimePublicationConflict(
                    "Finalized failure event changed its terminal decision identity."
                ) from exc
        if failure_terminal_decision is not None:
            try:
                (
                    state.session,
                    interaction_failed_event,
                    _,
                ) = await self._session_finalization.publish_sibling_interaction_transition(
                    session=state.session,
                    invocation_context=invocation_context,
                    registered_agent=registered_agent,
                    registered_environment=registered_environment,
                    environment_name=environment_name,
                    to_status=SessionStatus.FAILED,
                    from_statuses={SessionStatus.RUNNING},
                    observed_at=failure_interaction_observed_at,
                    event_id=failure_terminal_decision.interaction_event_id,
                    execution_profile=execution_profile,
                    model_completion_failure=exc,
                    terminal_event=prepared_atomic_terminal_event,
                    terminal_decision=failure_terminal_decision,
                )
            except ModelCompletionBudgetSettlementPending as accounting_pending:
                _raise_primary_with_secondary_failure(
                    exc,
                    accounting_pending,
                    group_message=("Session failure and model-completion budget recovery fence."),
                )
            if interaction_failed_event is not None:
                yield interaction_failed_event
            for event in failure_turn_events:
                yield event
        if failure_terminal_decision is None:
            async for event in self._terminal_event_publication.emit(
                event=session_failed_event,
                phase=RuntimeHookPhase.AFTER_SESSION_FAILED,
                session=state.session,
                registered_agent=registered_agent,
                registered_environment=registered_environment,
                execution_profile=execution_profile,
                invocation_context=invocation_context,
            ):
                yield event
        else:
            assert terminal_finalize_result is not None
            assert prepared_atomic_terminal_event is not None
            async for (
                event
            ) in self._session_finalization.emit_atomically_persisted_terminal_event_with_hooks(
                finalize_result=terminal_finalize_result,
                event=prepared_atomic_terminal_event,
                phase=RuntimeHookPhase.AFTER_SESSION_FAILED,
                session=state.session,
                registered_agent=registered_agent,
                registered_environment=registered_environment,
                execution_profile=execution_profile,
                invocation_context=invocation_context,
            ):
                yield event

    async def emit_interaction_started(
        self,
        *,
        session: Session,
        registered_agent: runtime_records.RegisteredAgentState,
        environment_name: str | None,
    ) -> Event:
        interaction_id = _current_session_interaction_id(session.id)
        if interaction_id is None:
            raise RuntimeError("Interaction identity is not active for the session.")
        return await self._event_writer.emit(
            build_interaction_started_event(
                clock=self._clock,
                session=session,
                registered_agent=registered_agent,
                environment_name=environment_name,
                interaction_id=interaction_id,
            )
        )

    async def emit_interaction_started_if_needed(
        self,
        *,
        session: Session,
        registered_agent: runtime_records.RegisteredAgentState,
        environment_name: str | None,
    ) -> Event | None:
        interaction_id = _current_session_interaction_id(session.id)
        if interaction_id is None:
            raise RuntimeError("Interaction identity is not active for the session.")
        existing = await self.session_store.query_events(
            EventQuery(
                session_id=session.id,
                interaction_id=interaction_id,
                event_types=INTERACTION_LIFECYCLE_EVENT_TYPES,
                order_by=EventOrder.SEQUENCE_DESC,
                limit=1,
            )
        )
        if not existing:
            return await self.emit_interaction_started(
                session=session,
                registered_agent=registered_agent,
                environment_name=environment_name,
            )
        latest = existing[0]
        if latest.event.type in INTERACTION_TERMINAL_EVENT_TYPES:
            raise RuntimeError(f"Interaction is already terminal: {interaction_id}")
        if latest.event.type in {EventType.INTERACTION_STARTED, EventType.INTERACTION_RESUMED}:
            return None
        return await self._session_finalization.emit_interaction_state(
            session=session,
            registered_agent=registered_agent,
            environment_name=environment_name,
            event_type=EventType.INTERACTION_RESUMED,
            status=InteractionStatus.ACTIVE,
        )

    async def claim_invocation_terminal_decision(
        self,
        *,
        session: Session,
        decision: InvocationTerminalDecision,
    ) -> None:
        """Install one exact terminal winner before outcome-specific effects."""

        def install_decision(
            current_session: Session,
            checkpoint: dict[str, Any] | None,
        ) -> dict[str, Any]:
            if (
                current_session.instance_id != session.instance_id
                or current_session.run_epoch != session.run_epoch
                or current_session.status is not SessionStatus.RUNNING
                or decision.session_id != current_session.id
                or decision.session_instance_id != current_session.instance_id
                or decision.run_epoch != current_session.run_epoch
            ):
                raise SessionRunFenced(
                    "Invocation terminal decision lost its exact session authority."
                )
            active_profile = active_invocation_execution_profile_from_checkpoint(checkpoint)
            if (
                active_profile is None
                or active_profile.session_id != current_session.id
                or active_profile.run_epoch != current_session.run_epoch
                or not invocation_terminal_decision_matches_active_profile(
                    decision,
                    session_id=current_session.id,
                    session_instance_id=current_session.instance_id,
                    run_epoch=current_session.run_epoch,
                    interaction_id=active_profile.interaction_id,
                    execution_profile_fingerprint=active_profile.profile.fingerprint,
                )
            ):
                raise SessionRunFenced(
                    "Invocation terminal decision lost its execution-profile authority."
                )
            return checkpoint_with_invocation_terminal_decision(checkpoint, decision)

        with _invocation_lifecycle_authority_mutation_scope():
            await self.session_store.publish_checkpoint_and_events(
                session.id,
                checkpoint_transform=install_decision,
                events=[],
                expected_statuses={SessionStatus.RUNNING},
                expected_run_epoch=session.run_epoch,
            )
