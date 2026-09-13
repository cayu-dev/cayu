"""Revisioned child-action discovery under the existing session-operation owner."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from hashlib import sha256
from typing import TYPE_CHECKING, Any

from cayu._task_wait import (
    await_shielded_task_outcome,
    capture_awaitable_outcome,
    restore_task_cancellation_requests,
)
from cayu._validation import canonical_durable_json_bytes
from cayu.core.events import Event, EventType
from cayu.runtime._foreground_child_wait import (
    FOREGROUND_CHILD_WAIT_KEY,
    ForegroundChildWait,
    event_with_foreground_child_wait_authority,
    foreground_child_state_from_checkpoint,
    observe_foreground_child_wait,
)
from cayu.runtime._invocation_terminal_decision import invocation_terminal_decision_from_checkpoint
from cayu.runtime._tool_effect_state import ToolEffectStateOwner
from cayu.runtime.execution_profiles import (
    active_invocation_execution_profile_from_checkpoint,
    active_invocation_execution_profile_is_released,
)
from cayu.runtime.sessions import (
    Session,
    SessionOperationPublication,
    SessionRunFenced,
    SessionStatus,
    SessionStore,
    _invocation_lifecycle_authority_read_scope,
)

if TYPE_CHECKING:
    from cayu.runtime._child_session_identity import ChildSessionRecoveryMatcher
    from cayu.runtime._event_writer import RuntimeEventWriter


async def refresh_foreground_child_action(
    wait: ForegroundChildWait,
    child_event: Event,
    *,
    store: SessionStore,
    writer: RuntimeEventWriter,
    matcher: ChildSessionRecoveryMatcher,
    before_mutation: Callable[[], Awaitable[None]],
) -> None:
    """Refresh discovery, never grant parent execution or resolve a child action."""
    await before_mutation()
    parent = await store.load(wait.parent_effect.session_id)
    if parent is None or parent.instance_id != wait.parent_effect.session_instance_id:
        raise SessionRunFenced("Foreground action refresh lost its parent incarnation.")
    source = await store.load_checkpoint(parent.id)
    current_wait, terminal = foreground_child_state_from_checkpoint(source)
    if current_wait != wait or terminal is not None:
        raise SessionRunFenced("Foreground action refresh lost its exact source wait.")
    profile = active_invocation_execution_profile_from_checkpoint(source)
    if (
        parent.status is not SessionStatus.INTERRUPTED
        or profile is None
        or profile.interaction_id != wait.parent_effect.interaction_id
        or profile.profile.fingerprint != wait.parent_effect.execution_profile_fingerprint
        or not active_invocation_execution_profile_is_released(
            profile, session_id=parent.id, run_epoch=parent.run_epoch
        )
        or invocation_terminal_decision_from_checkpoint(source) is not None
    ):
        raise SessionRunFenced("Foreground action refresh lacks a released waiting invocation.")
    effect = await ToolEffectStateOwner(store).resolve_call(
        parent,
        tool_round_id=wait.parent_effect.tool_round_id,
        tool_call_id=wait.parent_effect.tool_call_id,
    )
    if (
        effect is None
        or effect.intent != wait.parent_effect
        or effect.state != "outcome_unknown"
        or effect.child_recovery_arguments is None
    ):
        raise SessionRunFenced("Foreground action refresh lacks its original delegated effect.")
    observed = await observe_foreground_child_wait(
        store,
        parent=parent,
        intent=effect.intent,
        matcher=matcher,
        arguments=effect.child_recovery_arguments,
    )
    if observed is None:
        return  # The child's later terminal event owns the next notification.
    if (
        observed.child_session_id != wait.child_session_id
        or observed.child_session_instance_id != wait.child_session_instance_id
        or observed.child_interaction_id != wait.child_interaction_id
        or observed.child_spawn_fingerprint != wait.child_spawn_fingerprint
    ):
        raise SessionRunFenced("Foreground action refresh cannot replace its child identity.")
    if observed.child_action_run_epoch <= wait.child_action_run_epoch:
        if observed.model_copy(update={"revision": wait.revision}) != wait:
            raise SessionRunFenced("Foreground action refresh cannot rewind or replace an action.")
        return
    outcome = await store.summarize_outcome(observed.child_session_id)
    if outcome.terminal_event is None or outcome.terminal_event.event != child_event:
        return  # A stale notification cannot publish newer, unrelated evidence.
    target = observed.model_copy(update={"revision": wait.revision + 1})
    receipt = {
        "source": wait.model_dump(mode="json"),
        "target": target.model_dump(mode="json"),
        "child_event": child_event.model_dump(mode="json"),
    }
    key = (
        "foreground-action:"
        + sha256(canonical_durable_json_bytes(receipt, "foreground_action")).hexdigest()
    )
    notification = writer.prepare(
        event_with_foreground_child_wait_authority(
            Event(
                id=key,
                type=EventType.SESSION_DELEGATED_ACTION_UPDATED,
                session_id=parent.id,
                agent_name=parent.agent_name,
                environment_name=parent.environment_name,
                timestamp=child_event.timestamp,
                payload={
                    "interruption_type": "waiting_on_child_action",
                    **target.delegated_action_reference(),
                    "model_step_id": target.parent_effect.model_step_id,
                    "model_attempt_id": target.parent_effect.model_attempt_id,
                    "tool_call_id": target.parent_effect.tool_call_id,
                    "tool_round_id": target.parent_effect.tool_round_id,
                },
            ),
            target,
        )
    )

    def publish(
        current: Session,
        checkpoint: dict[str, Any] | None,
        existing: dict[str, Any] | None,
    ) -> SessionOperationPublication:
        retained, selected = foreground_child_state_from_checkpoint(checkpoint)
        if (
            current.instance_id != parent.instance_id
            or active_invocation_execution_profile_from_checkpoint(checkpoint) != profile
            or invocation_terminal_decision_from_checkpoint(checkpoint) is not None
            or selected is not None
        ):
            raise SessionRunFenced("Foreground action refresh lost its publication authority.")
        if existing is not None:
            if existing != receipt or retained != target:
                raise SessionRunFenced("Foreground action replay conflicts with its exact receipt.")
        elif retained != wait:
            raise SessionRunFenced("Foreground action source changed before publication.")
        return SessionOperationPublication(
            checkpoint={
                **(checkpoint or {}),
                FOREGROUND_CHILD_WAIT_KEY: target.model_dump(mode="json"),
            },
            operation_records={key: receipt},
        )

    await before_mutation()
    # Read-only lifecycle visibility permits an atomic check of the released
    # profile; it does not let this discovery operation replace lifecycle roots.
    with _invocation_lifecycle_authority_read_scope():
        publication = asyncio.create_task(
            capture_awaitable_outcome(
                lambda: store.publish_session_operation(
                    parent.id,
                    idempotency_key=key,
                    operation_transform=publish,
                    events=[notification],
                    expected_statuses={SessionStatus.INTERRUPTED},
                    expected_run_epoch=parent.run_epoch,
                )
            )
        )
        result = await await_shielded_task_outcome(publication)
    failures = [
        failure
        for failure in (
            result.error if result.result is None else result.result.error,
            result.cancellation,
            result.subsequent_cancellation,
        )
        if failure is not None
    ]
    if result.cancellation_requests_consumed:
        restore_task_cancellation_requests(
            result.cancellation_requests_consumed, cancellation=result.cancellation
        )
    if len(failures) == 1:
        raise failures[0]
    if failures:
        raise BaseExceptionGroup(
            "Foreground action publication failed during cancellation.", failures
        )
    await writer.fan_out_persisted([notification])
