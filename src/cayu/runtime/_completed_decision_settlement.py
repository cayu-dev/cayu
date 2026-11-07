"""What a committed completion still owes, written once and finished by anyone.

A successful interaction commits its COMPLETED transition before its task is
completed, before TASK_COMPLETED, TURN_COMPLETED and SESSION_COMPLETED are
published and before its run fence is released. The completion transition
therefore installs a completed invocation terminal decision (a paired-write
intent, see ``cayu.sessions._paired_writes``) that freezes every payload and
derives every event ID those later writes use. The run publishes them from the
decision; when it cannot, recovery publishes the same events from the same
decision, and the fence release settles it once SESSION_COMPLETED is durable.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from cayu.events import (
    Event,
    EventType,
    event_with_runtime_generated_id,
    event_with_runtime_nested_payload_authority,
    event_with_runtime_payload_authority,
)
from cayu.sessions._invocation_terminal_decision import (
    InvocationTerminalDecision,
    InvocationTerminalOutcome,
    build_completed_invocation_terminal_decision,
    completed_decision_event_id,
    invocation_terminal_event_id,
)
from cayu.sessions._paired_writes import (
    completed_decision_install_mutation,
)
from cayu.sessions._terminal_evidence import (
    SESSION_RUN_OPERATION_ID_PAYLOAD_KEY,
    _session_run_operation_from_checkpoint,
)
from cayu.sessions.records import SessionStatus
from cayu.tasks.records import Task, TaskStatus

if TYPE_CHECKING:
    from cayu.runtime._session_finalization import SessionFinalization
    from cayu.runtime._session_outcome_publication import SessionOutcomeContext
    from cayu.runtime.tool_completion import ToolCompletionResult
    from cayu.sessions.records import Session


def completion_terminal_payload(
    tool_completion_result: ToolCompletionResult | None,
) -> dict[str, Any]:
    """Return the SESSION_COMPLETED payload a completion publishes before finalization."""

    if tool_completion_result is None:
        return {}
    return {
        "reason": tool_completion_result.reason,
        "tool_completion": tool_completion_result.model_dump(mode="json"),
    }


def completed_task_result(context: SessionOutcomeContext) -> dict[str, Any]:
    session = context.state.session
    return {
        "session_id": session.id,
        "agent_name": context.registered_agent.spec.name,
        "environment_name": context.environment_name,
    }


async def completed_decision_install(
    *,
    checkpoint: dict[str, Any] | None,
    session_finalization: SessionFinalization,
    context: SessionOutcomeContext,
    interaction_completed: Event,
    tool_completion_result: ToolCompletionResult | None,
) -> dict[str, Any]:
    """Build the decision for one completion event and the mutation that installs it."""

    session = context.state.session
    active_profile = context.invocation_context.active_profile
    terminal_payload = completion_terminal_payload(tool_completion_result)
    assert interaction_completed.interaction_id is not None
    terminal_event_id = invocation_terminal_event_id(
        outcome=InvocationTerminalOutcome.COMPLETED,
        session_id=session.id,
        session_instance_id=session.instance_id,
        run_epoch=session.run_epoch,
        interaction_id=interaction_completed.interaction_id,
        source_id=interaction_completed.id,
        event_kind="session",
    )
    run_operation = _session_run_operation_from_checkpoint(checkpoint)
    if run_operation is not None:
        # The terminal publication binds itself to the run operation the same way.
        terminal_payload[SESSION_RUN_OPERATION_ID_PAYLOAD_KEY] = run_operation.operation_id
        if run_operation.terminal_event_id is not None:
            terminal_event_id = run_operation.terminal_event_id
    turn_completed = await session_finalization.prepare_turn_completed_event(
        session=session,
        registered_agent=context.registered_agent,
        environment_name=context.environment_name,
        status=SessionStatus.COMPLETED,
        run_started_at=context.run_started_at,
        usage_tracker=context.turn_usage_tracker,
        timestamp=interaction_completed.timestamp,
    )
    task_id = context.task_id if context.invocation_context.work_attempt is None else None
    decision = build_completed_invocation_terminal_decision(
        session_id=session.id,
        session_instance_id=session.instance_id,
        run_epoch=session.run_epoch,
        profile_interaction_id=active_profile.interaction_id,
        interaction_id=interaction_completed.interaction_id,
        execution_profile_fingerprint=active_profile.profile.fingerprint,
        interaction_event_id=interaction_completed.id,
        terminal_event_id=terminal_event_id,
        observed_at=interaction_completed.timestamp,
        terminal_payload=terminal_payload,
        turn_completed_payload=turn_completed.payload,
        task_id=task_id,
        task_result_payload=None if task_id is None else completed_task_result(context),
    )
    context.state.completed_decision = decision
    return completed_decision_install_mutation(decision)


def completed_task_event(decision: InvocationTerminalDecision, task_event: Event) -> Event:
    """Give TASK_COMPLETED the identity and time its decision fixed."""

    return event_with_runtime_generated_id(
        task_event.model_copy(
            update={
                "id": completed_decision_event_id(decision, "task_completed"),
                "timestamp": decision.observed_at,
            },
            deep=True,
        )
    )


def completed_turn_event(decision: InvocationTerminalDecision, session: Session) -> Event:
    """Rebuild the TURN_COMPLETED event the decision prepared."""

    assert decision.turn_completed_payload is not None
    return event_with_runtime_generated_id(
        event_with_runtime_nested_payload_authority(
            Event(
                id=completed_decision_event_id(decision, "turn_completed"),
                type=EventType.TURN_COMPLETED,
                session_id=session.id,
                interaction_id=None,
                agent_name=session.agent_name,
                environment_name=session.environment_name,
                timestamp=decision.observed_at,
                payload=decision.turn_completed_payload,
            ),
            ("interaction_ids", "*"),
        )
    )


def completed_terminal_event(decision: InvocationTerminalDecision, session: Session) -> Event:
    """Rebuild the SESSION_COMPLETED event, before environment finalization adds to it."""

    event = event_with_runtime_generated_id(
        Event(
            id=decision.terminal_event_id,
            type=EventType.SESSION_COMPLETED,
            session_id=session.id,
            agent_name=session.agent_name,
            environment_name=session.environment_name,
            timestamp=decision.observed_at,
            payload=decision.terminal_payload,
        )
    )
    if SESSION_RUN_OPERATION_ID_PAYLOAD_KEY in decision.terminal_payload:
        event = event_with_runtime_payload_authority(event, SESSION_RUN_OPERATION_ID_PAYLOAD_KEY)
    return event


def task_lifecycle_event(
    *,
    event_type: EventType,
    task: Task,
    session_id: str,
    agent_name: str,
    environment_name: str | None,
) -> Event:
    event = Event(
        type=event_type,
        session_id=session_id,
        agent_name=agent_name,
        environment_name=environment_name,
        payload={
            "task_id": task.id,
            "task_type": task.type,
            "task_status": task.status.value,
            "task_session_id": task.session_id,
            "assigned_agent_name": task.assigned_agent_name,
            "parent_task_id": task.parent_task_id,
        },
    )
    return event_with_runtime_payload_authority(
        event,
        *(
            field_name
            for field_name in ("task_id", "task_session_id", "parent_task_id")
            if event.payload.get(field_name) is not None
        ),
    )


def owed_completion_events(
    decision: InvocationTerminalDecision,
    session: Session,
    task: Task | None,
) -> list[Event]:
    """The task and turn events a completion owes before its SESSION_COMPLETED.

    ``task`` is the decision's task once it is completed for this session; a task
    another owner decided gets no TASK_COMPLETED.
    """

    owed = []
    if task is not None and task.status is TaskStatus.COMPLETED:
        owed.append(
            completed_task_event(
                decision,
                task_lifecycle_event(
                    event_type=EventType.TASK_COMPLETED,
                    task=task,
                    session_id=session.id,
                    agent_name=session.agent_name,
                    environment_name=session.environment_name,
                ),
            )
        )
    owed.append(completed_turn_event(decision, session))
    return owed
