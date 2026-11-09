"""What a committed terminal outcome still owes, written once and finished by anyone.

A completed, failed or interrupted interaction commits its terminal transition
before its task is completed, before its TURN_COMPLETED and session event are
published and before its run fence is released. The transition therefore
installs a version-2 invocation terminal decision (a paired-write intent, see
``cayu.sessions._paired_writes``) that freezes every payload and derives every
event ID those later writes use. The run publishes them from the decision; when
it cannot, recovery publishes the same events from the same decision, and the
fence release settles it once the session event is durable.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from cayu.events import (
    Event,
    EventType,
    event_with_runtime_generated_id,
    event_with_runtime_nested_payload_authority,
    event_with_runtime_payload_authority,
    runtime_payload_authority_paths,
)
from cayu.sessions._invocation_terminal_decision import (
    InvocationTerminalDecision,
    InvocationTerminalOutcome,
    build_owed_invocation_terminal_decision,
    invocation_terminal_event_id,
    owed_decision_event_id,
)
from cayu.sessions._paired_writes import (
    OUTCOME_SESSION_EVENT,
    OUTCOME_SESSION_STATUS,
    owed_decision_install_mutation,
)
from cayu.sessions._terminal_evidence import (
    SESSION_RUN_OPERATION_ID_PAYLOAD_KEY,
    _session_run_operation_from_checkpoint,
)
from cayu.tasks.records import Task, TaskStatus

if TYPE_CHECKING:
    from cayu.runtime._session_finalization import SessionFinalization
    from cayu.runtime._session_outcome_publication import SessionOutcomeContext
    from cayu.runtime.tool_completion import ToolCompletionResult
    from cayu.sessions.records import Session


@dataclass(frozen=True, slots=True)
class OwedOutcome:
    """A terminal outcome its transition records, with the session event it owes.

    ``terminal_event`` is that event as the run would publish it before
    environment finalization adds to it: its type, payload and the payload
    authority its producer attested. ``turn_completed_payload`` is prepared
    before the transition starts, so no read of it lands between the
    transition's own steps; a completion prepares it with its decision.
    """

    context: SessionOutcomeContext
    terminal_event: Event
    turn_completed_payload: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        if self.terminal_event.type not in OUTCOME_SESSION_EVENT.values():
            raise ValueError("An owed outcome must owe a terminal session event.")

    @property
    def outcome(self) -> InvocationTerminalOutcome:
        return _outcome_of(self.terminal_event)


def _outcome_of(terminal_event: Event) -> InvocationTerminalOutcome:
    return next(
        outcome
        for outcome, event_type in OUTCOME_SESSION_EVENT.items()
        if event_type is terminal_event.type
    )


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


def completed_outcome(
    context: SessionOutcomeContext,
    tool_completion_result: ToolCompletionResult | None,
) -> OwedOutcome:
    """The outcome a completion transition records."""

    session = context.state.session
    return OwedOutcome(
        context=context,
        terminal_event=Event(
            type=EventType.SESSION_COMPLETED,
            session_id=session.id,
            agent_name=context.registered_agent.spec.name,
            environment_name=context.environment_name,
            payload=completion_terminal_payload(tool_completion_result),
        ),
    )


async def prepared_outcome(
    context: SessionOutcomeContext,
    terminal_event: Event,
    session_finalization: SessionFinalization,
) -> OwedOutcome:
    """A failure or interruption with the turn summary it will owe, read now."""

    turn_completed = await session_finalization.prepare_turn_completed_event(
        session=context.state.session,
        registered_agent=context.registered_agent,
        environment_name=context.environment_name,
        status=OUTCOME_SESSION_STATUS[_outcome_of(terminal_event)],
        run_started_at=context.run_started_at,
        usage_tracker=context.turn_usage_tracker,
    )
    return OwedOutcome(
        context=context,
        terminal_event=terminal_event,
        turn_completed_payload=session_finalization.publication_safe_event(turn_completed).payload,
    )


def completed_task_result(context: SessionOutcomeContext) -> dict[str, Any]:
    session = context.state.session
    return {
        "session_id": session.id,
        "agent_name": context.registered_agent.spec.name,
        "environment_name": context.environment_name,
    }


async def owed_decision_install(
    *,
    checkpoint: dict[str, Any] | None,
    session_finalization: SessionFinalization,
    owed: OwedOutcome,
    interaction_event: Event,
) -> dict[str, Any]:
    """Build the decision for one terminal interaction event and the mutation installing it."""

    context = owed.context
    outcome = owed.outcome
    session = context.state.session
    active_profile = context.invocation_context.active_profile
    # The decision is durable before its events are, so it holds only what their
    # publication would store: the payload after secret redaction.
    terminal_template = session_finalization.publication_safe_event(owed.terminal_event)
    terminal_payload = dict(terminal_template.payload)
    payload_authority, nested_payload_authority = runtime_payload_authority_paths(terminal_template)
    assert interaction_event.interaction_id is not None
    terminal_event_id = invocation_terminal_event_id(
        outcome=outcome,
        session_id=session.id,
        session_instance_id=session.instance_id,
        run_epoch=session.run_epoch,
        interaction_id=interaction_event.interaction_id,
        source_id=interaction_event.id,
        event_kind="session",
    )
    run_operation = _session_run_operation_from_checkpoint(checkpoint)
    if run_operation is not None:
        # The terminal publication binds itself to the run operation the same way.
        terminal_payload[SESSION_RUN_OPERATION_ID_PAYLOAD_KEY] = run_operation.operation_id
        if run_operation.terminal_event_id is not None:
            terminal_event_id = run_operation.terminal_event_id
    turn_completed_payload = owed.turn_completed_payload
    if turn_completed_payload is None:
        turn_completed_payload = session_finalization.publication_safe_event(
            await session_finalization.prepare_turn_completed_event(
                session=session,
                registered_agent=context.registered_agent,
                environment_name=context.environment_name,
                status=OUTCOME_SESSION_STATUS[outcome],
                run_started_at=context.run_started_at,
                usage_tracker=context.turn_usage_tracker,
                timestamp=interaction_event.timestamp,
            )
        ).payload
    task_id = (
        context.task_id
        if outcome is InvocationTerminalOutcome.COMPLETED
        and context.invocation_context.work_attempt is None
        else None
    )
    decision = build_owed_invocation_terminal_decision(
        outcome=outcome,
        session_id=session.id,
        session_instance_id=session.instance_id,
        run_epoch=session.run_epoch,
        profile_interaction_id=active_profile.interaction_id,
        interaction_id=interaction_event.interaction_id,
        execution_profile_fingerprint=active_profile.profile.fingerprint,
        interaction_event_id=interaction_event.id,
        terminal_event_id=terminal_event_id,
        observed_at=interaction_event.timestamp,
        terminal_payload=terminal_payload,
        turn_completed_payload=turn_completed_payload,
        task_id=task_id,
        task_result_payload=None if task_id is None else completed_task_result(context),
        terminal_payload_authority=payload_authority or None,
        terminal_nested_payload_authority=nested_payload_authority or None,
    )
    context.state.outcome_decision = decision
    return owed_decision_install_mutation(decision)


def completed_task_event(decision: InvocationTerminalDecision, task_event: Event) -> Event:
    """Give TASK_COMPLETED the identity and time its decision fixed."""

    return event_with_runtime_generated_id(
        task_event.model_copy(
            update={
                "id": owed_decision_event_id(decision, "task_completed"),
                "timestamp": decision.observed_at,
            },
            deep=True,
        )
    )


def owed_turn_event(decision: InvocationTerminalDecision, session: Session) -> Event:
    """Rebuild the TURN_COMPLETED event the decision prepared."""

    assert decision.turn_completed_payload is not None
    return event_with_runtime_generated_id(
        event_with_runtime_nested_payload_authority(
            Event(
                id=owed_decision_event_id(decision, "turn_completed"),
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


def owed_terminal_event(decision: InvocationTerminalDecision, session: Session) -> Event:
    """Rebuild the session event, before environment finalization adds to it."""

    event = event_with_runtime_generated_id(
        Event(
            id=decision.terminal_event_id,
            type=OUTCOME_SESSION_EVENT[decision.outcome],
            session_id=session.id,
            agent_name=session.agent_name,
            environment_name=session.environment_name,
            timestamp=decision.observed_at,
            payload=decision.terminal_payload,
        )
    )
    attested = set(decision.terminal_payload_authority or ())
    if SESSION_RUN_OPERATION_ID_PAYLOAD_KEY in decision.terminal_payload:
        attested.add(SESSION_RUN_OPERATION_ID_PAYLOAD_KEY)
    if attested:
        event = event_with_runtime_payload_authority(event, *sorted(attested))
    if decision.terminal_nested_payload_authority:
        event = event_with_runtime_nested_payload_authority(
            event, *decision.terminal_nested_payload_authority
        )
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


def owed_outcome_events(
    decision: InvocationTerminalDecision,
    session: Session,
    task: Task | None,
) -> list[Event]:
    """The task and turn events an outcome owes before its session event.

    ``task`` is a completion's task once it is completed for this session; a task
    another owner decided gets no TASK_COMPLETED.
    """

    owed = []
    if (
        decision.outcome is InvocationTerminalOutcome.COMPLETED
        and task is not None
        and task.status is TaskStatus.COMPLETED
    ):
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
    owed.append(owed_turn_event(decision, session))
    return owed
