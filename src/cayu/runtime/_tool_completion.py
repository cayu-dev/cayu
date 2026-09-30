"""Recover completion policy and result from exact durable model-step evidence."""

from __future__ import annotations

from enum import Enum, auto
from typing import Any

from cayu.events import EventType
from cayu.messages import MessageRole, ToolCallPart, ToolResultPart
from cayu.runtime import _runtime_records as runtime_records
from cayu.runtime._execution_profile_admission import model_finalization_material
from cayu.runtime._model_completion_publication import model_step_publication_from_checkpoint
from cayu.runtime._model_step_executor import (
    ModelCompletionRecoveryContext,
    model_completion_recovery_context_from_stage,
)
from cayu.runtime._tool_round_recovery import (
    PendingToolRound,
    pending_tool_round_from_checkpoint,
    pending_tool_round_identity,
)
from cayu.runtime.execution_profiles import (
    ExecutionProfileComponentClass,
    ExecutionProfileIdentity,
    ExecutionProfileIdentityStrength,
    _available_component,
    active_invocation_execution_profile_from_checkpoint,
)
from cayu.runtime.retry_policy import RetryPolicy
from cayu.runtime.stop_policy import RunLimits
from cayu.runtime.tool_completion import ToolCompletionPolicy, ToolCompletionResult
from cayu.sessions.base import EventOrder, EventQuery, Session, SessionStore
from cayu.sessions.interactions import INTERACTION_LIFECYCLE_EVENT_TYPES
from cayu.tools.base import ToolResult


class _CheckpointState(Enum):
    NOT_LOADED = auto()


async def load_recorded_tool_completion_policy(
    store: SessionStore,
    session: Session,
    checkpoint: dict[str, Any] | None | _CheckpointState = _CheckpointState.NOT_LOADED,
    *,
    execution_profile: ExecutionProfileIdentity,
    max_steps: int,
    limits: RunLimits,
    retry_policy: RetryPolicy,
    context: ModelCompletionRecoveryContext | None = None,
) -> ToolCompletionPolicy | None:
    """Read values only when default finalization differs from admitted authority.

    The ordinary unconfigured path performs no additional store reads. A hash
    never supplies configuration: the recorded context must authenticate its
    complete finalization material before its policy can be reused.
    An omitted checkpoint is loaded lazily; an explicit None remains an absent
    snapshot and cannot acquire authority from a later checkpoint.
    """
    expected = execution_profile.component(ExecutionProfileComponentClass.FINALIZATION)
    ordinary = _available_component(
        ExecutionProfileComponentClass.FINALIZATION,
        ExecutionProfileIdentityStrength.STRUCTURAL,
        model_finalization_material(max_steps=max_steps, limits=limits, retry_policy=retry_policy),
    )
    if expected == ordinary:
        return None
    if checkpoint is _CheckpointState.NOT_LOADED:
        checkpoint = await store.load_checkpoint(session.id)
    active = active_invocation_execution_profile_from_checkpoint(checkpoint)
    if active is None or active.profile != execution_profile or active.session_id != session.id:
        return None
    if context is None:
        active_stage = await store.load_active_model_completion_stage(session.id)
        if active_stage is not None:
            context = model_completion_recovery_context_from_stage(active_stage.stage)
        if context is None:
            pointer = model_step_publication_from_checkpoint(checkpoint)
            if pointer is None:
                return None
            stage = await store.load_model_completion_stage(session.id, pointer.stage_id)
            if stage is None or stage.logical_step_id != pointer.logical_step_id:
                raise RuntimeError("Tool completion policy lost its durable model-step evidence.")
            context = model_completion_recovery_context_from_stage(stage)
    if (
        context is None
        or context.interaction_id != active.interaction_id
        or context.execution_profile_fingerprint != execution_profile.fingerprint
    ):
        return None
    recorded = _available_component(
        ExecutionProfileComponentClass.FINALIZATION,
        ExecutionProfileIdentityStrength.STRUCTURAL,
        model_finalization_material(
            max_steps=context.max_steps,
            limits=context.limits,
            retry_policy=context.retry_policy,
            tool_completion=context.tool_completion,
        ),
    )
    if recorded != expected:
        raise RuntimeError("Tool completion policy conflicts with its admitted execution profile.")
    return context.tool_completion


async def tool_completion_requires_execution(
    store: SessionStore,
    session: Session,
    checkpoint: dict[str, Any] | None,
    registered_agent: runtime_records.RegisteredAgentState,
) -> bool:
    """Identify retained final-tool work before recovery can replace its epoch.

    Completion runs ordinary lifecycle hooks. Its recorded result does not
    grant the caller permission to execute those hooks after reconstruction.
    """
    pointer = model_step_publication_from_checkpoint(checkpoint)
    active = active_invocation_execution_profile_from_checkpoint(checkpoint)
    if pointer is None or pointer.tool_round_id is None or active is None:
        return False
    stage = await store.load_model_completion_stage(session.id, pointer.stage_id)
    if stage is None or stage.logical_step_id != pointer.logical_step_id:
        return False
    context = model_completion_recovery_context_from_stage(stage)
    if context is None or context.tool_completion is None:
        return False
    policy = await load_recorded_tool_completion_policy(
        store,
        session,
        checkpoint,
        execution_profile=active.profile,
        max_steps=context.max_steps,
        limits=context.limits,
        retry_policy=context.retry_policy,
        context=context,
    )
    if policy is None:
        return False
    pending = pending_tool_round_from_checkpoint(checkpoint)
    if pending is not None:
        return await pending_round_has_completion_success(
            store, session, pending, policy=policy, interaction_id=active.interaction_id
        )
    return (
        await recorded_tool_completion_result(
            store,
            session,
            policy=policy,
            execution_profile=active.profile,
            registered_agent=registered_agent,
        )
        is not None
    )


def require_registered_completion_tools(
    policy: ToolCompletionPolicy | None,
    registered_agent: runtime_records.RegisteredAgentState,
) -> None:
    if policy is None:
        return
    from cayu.tools.base import ToolEffect

    for name in policy.tool_names:
        tool = registered_agent.executable_tool(name)
        if tool is None:
            raise ValueError("tool_completion must name registered application tools.")
        if tool.effect not in (ToolEffect.NONE, ToolEffect.IDEMPOTENT):
            raise ValueError("tool_completion tools must declare none or idempotent effects.")


async def recorded_tool_completion_result(
    store: SessionStore,
    session: Session,
    *,
    policy: ToolCompletionPolicy | None,
    execution_profile: ExecutionProfileIdentity,
    registered_agent: runtime_records.RegisteredAgentState,
) -> ToolCompletionResult | None:
    """Select one closed application call; siblings and later input continue normally."""
    if policy is None:
        return None
    checkpoint = await store.load_checkpoint(session.id)
    pointer = model_step_publication_from_checkpoint(checkpoint)
    active = active_invocation_execution_profile_from_checkpoint(checkpoint)
    if (
        checkpoint is None
        or pointer is None
        or pointer.tool_round_id is None
        or active is None
        or active.profile != execution_profile
    ):
        return None
    # A pending marker still belongs to ordinary recovery, never early completion.
    if any(
        checkpoint.get(key) is not None
        for key in ("pending_tool_round", "pending_tool_approval", "pending_user_input")
    ):
        return None
    window = await store.load_transcript_window(
        session.id, start_index=pointer.source_transcript_cursor, limit=3
    )
    if (
        len(window.records) != 2
        or window.records[0].index != pointer.source_transcript_cursor
        or window.records[1].index != pointer.source_transcript_cursor + 1
        or window.cursor != pointer.source_transcript_cursor + 2
    ):
        return None
    assistant, results = (record.message for record in window.records)
    calls = [part for part in assistant.content if type(part) is ToolCallPart]
    if (
        assistant.role != MessageRole.ASSISTANT
        or results.role != MessageRole.TOOL
        or len(calls) != 1
        or len(results.content) != 1
        or type(results.content[0]) is not ToolResultPart
    ):
        return None
    call, result = calls[0], results.content[0]
    if call.tool_name not in policy.tool_names or result.is_error:
        return None
    if call.tool_round_id != pointer.tool_round_id:
        raise RuntimeError("Tool completion result conflicts with its source model step.")
    registered = registered_agent.executable_tool(call.tool_name)
    if registered is None:
        raise RuntimeError("Tool completion result has no registered application tool.")
    if not await successful_completion_terminal(
        store,
        session,
        interaction_id=active.interaction_id,
        call=call,
        expected_result=result,
    ):
        return None
    return ToolCompletionResult(call=call, effect=registered.effect, result=result)


async def recorded_terminal_tool_completion_payload(
    store: SessionStore,
    session: Session,
) -> dict[str, Any]:
    """Restore the basis committed atomically with the final interaction settlement."""
    records = await store.query_events(
        EventQuery(
            session_id=session.id,
            event_types=INTERACTION_LIFECYCLE_EVENT_TYPES,
            order_by=EventOrder.SEQUENCE_DESC,
            limit=1,
        )
    )
    if not records or records[0].event.type != EventType.INTERACTION_COMPLETED:
        return {}
    value = records[0].event.payload.get("tool_completion")
    if value is None:
        return {}
    result = ToolCompletionResult.model_validate(value)
    return {"reason": result.reason, "tool_completion": result.model_dump(mode="json")}


async def successful_completion_terminal(
    store: SessionStore,
    session: Session,
    *,
    interaction_id: str,
    call: ToolCallPart,
    expected_result: ToolResultPart | None = None,
) -> bool:
    """Require the sole durable terminal to be successful with the exact call identity."""
    terminals = await store.query_events(
        EventQuery(
            session_id=session.id,
            interaction_id=interaction_id,
            model_step_id=call.model_step_id,
            tool_name=call.tool_name,
            event_types=(
                EventType.TOOL_CALL_COMPLETED,
                EventType.TOOL_CALL_FAILED,
                EventType.TOOL_CALL_BLOCKED,
                EventType.TOOL_CALL_APPROVAL_DENIED,
            ),
            limit=2,
        )
    )
    if len(terminals) != 1 or terminals[0].event.type != EventType.TOOL_CALL_COMPLETED:
        return False
    terminal = terminals[0].event
    for field in ("tool_call_id", "model_step_id", "model_attempt_id", "tool_round_id"):
        if terminal.payload.get(field) != getattr(call, field):
            raise RuntimeError("Tool completion terminal identity conflicts with its transcript.")
    result = ToolResult.model_validate(terminal.payload.get("result"))
    if result.is_error:
        return False
    if expected_result is not None and any(
        getattr(result, field) != getattr(expected_result, field)
        for field in ("content", "structured", "artifacts", "is_error")
    ):
        raise RuntimeError("Tool completion terminal result conflicts with its transcript.")
    return True


async def pending_round_has_completion_success(
    store: SessionStore,
    session: Session,
    pending: PendingToolRound,
    *,
    policy: ToolCompletionPolicy,
    interaction_id: str,
) -> bool:
    if len(pending.tool_calls) != 1:
        return False
    call = pending.tool_calls[0]
    if call.tool_name not in policy.tool_names:
        return False
    identity = pending_tool_round_identity(pending)
    return await successful_completion_terminal(
        store,
        session,
        interaction_id=interaction_id,
        call=ToolCallPart(
            tool_name=call.tool_name,
            tool_call_id=call.tool_call_id,
            model_step_id=identity.model_step_id,
            model_attempt_id=identity.model_attempt_id,
            tool_round_id=identity.tool_round_id,
        ),
    )
