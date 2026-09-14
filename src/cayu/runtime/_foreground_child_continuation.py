"""Deliver a child-terminal wakeup using persisted lineage, never a callback ID."""

from collections.abc import Awaitable, Callable
from hashlib import sha256

from cayu._validation import canonical_durable_json_bytes
from cayu.approvals.user_input import user_input_lifecycle_authority_from_checkpoint
from cayu.events import Event, EventType
from cayu.runtime import _approval_support as approval_support
from cayu.runtime._child_session_identity import ChildSessionKind, generate_child_session_id
from cayu.runtime._foreground_child_wait import (
    FOREGROUND_CHILD_WAIT_KEY,
    FOREGROUND_PARENT_CONTINUATION_KEY,
    ForegroundChildTerminal,
    ForegroundChildWait,
    ForegroundParentContinuation,
    foreground_child_state_from_checkpoint,
    owned_delegated_wait,
)
from cayu.runtime._tool_effect_state import ToolEffectStateOwner
from cayu.runtime.execution_profiles import (
    active_invocation_execution_profile_from_checkpoint,
    active_invocation_execution_profile_is_released,
)
from cayu.sessions.base import EventQuery, Session, SessionStatus, SessionStore
from cayu.sessions.pending_actions import pending_action_evidence_round_from_checkpoint


async def _has_unsettled_parent_spawn(
    child: Session, parent: Session, *, store: SessionStore
) -> bool:
    """Retain delivery before wait persistence; this never authorizes execution."""
    if parent.status in {SessionStatus.COMPLETED, SessionStatus.FAILED}:
        return False
    subagent = child.metadata.get("subagent")
    if type(subagent) is not dict or subagent.get("mode") != "foreground":
        return False
    call_id = subagent.get("tool_call_id")
    spawn_id = subagent.get("idempotency_key")
    if type(call_id) is not str or type(spawn_id) is not str:
        return False
    pending = pending_action_evidence_round_from_checkpoint(await store.load_checkpoint(parent.id))
    if pending is None or not any(call.tool_call_id == call_id for call in pending.tool_calls):
        return False
    effect = await ToolEffectStateOwner(store).resolve_call(
        parent, tool_round_id=pending.tool_round_id, tool_call_id=call_id
    )
    if (
        effect is None
        or effect.state not in {"executing", "outcome_unknown"}
        or effect.child_recovery_arguments is None
        or effect.intent.session_instance_id != parent.instance_id
        or effect.intent.idempotency_key != spawn_id
        or child.id
        != generate_child_session_id(
            kind=ChildSessionKind.SUBAGENT,
            parent_session_id=parent.id,
            logical_spawn_id=effect.intent.idempotency_key,
        )
    ):
        return False
    closed = await store.query_events(
        EventQuery(
            session_id=parent.id,
            interaction_id=effect.intent.interaction_id,
            event_types=(
                EventType.INTERACTION_COMPLETED,
                EventType.INTERACTION_FAILED,
                EventType.INTERACTION_INTERRUPTED,
            ),
            limit=1,
        )
    )
    return not closed


async def load_attached_foreground_continuation(
    parent: Session,
    *,
    store: SessionStore,
) -> ForegroundParentContinuation | None:
    """Authenticate attached continuation input against its insert-only receipt.

    This does not claim execution ownership. The invocation owner must still
    fence admission against the exact checkpoint and original interaction.
    """
    checkpoint = await store.load_checkpoint(parent.id)
    if checkpoint is None or FOREGROUND_PARENT_CONTINUATION_KEY not in checkpoint:
        return None
    continuation = ForegroundParentContinuation.model_validate(
        checkpoint[FOREGROUND_PARENT_CONTINUATION_KEY]
    )
    intent = continuation.terminal.wait.parent_effect
    if intent.session_id != parent.id or intent.session_instance_id != parent.instance_id:
        raise RuntimeError("Attached foreground continuation belongs to another parent.")
    receipt = await store.load_runtime_publication_receipt(parent.id, continuation.publication_id)
    if (
        receipt is None
        or receipt.session_id != parent.id
        or receipt.publication_id != continuation.publication_id
        or receipt.kind not in {"tool-round", "approval-close", "user-input-close"}
        or receipt.interaction_id != intent.interaction_id
        or receipt.source_run_epoch <= intent.source_run_epoch
        or canonical_durable_json_bytes(
            receipt.intent.get(FOREGROUND_PARENT_CONTINUATION_KEY),
            FOREGROUND_PARENT_CONTINUATION_KEY,
        )
        != canonical_durable_json_bytes(
            continuation.model_dump(mode="json"), FOREGROUND_PARENT_CONTINUATION_KEY
        )
    ):
        raise RuntimeError("Attached foreground continuation lacks its exact publication receipt.")
    return continuation


async def deliver_foreground_child_terminal(
    event: Event,
    *,
    store: SessionStore,
    has_active_tasks: Callable[[str], bool],
    resume: Callable[[ForegroundChildTerminal], Awaitable[None]],
    refresh: Callable[[ForegroundChildWait, Event], Awaitable[None]],
    settle: Callable[[ForegroundChildWait, Event], Awaitable[None]],
) -> bool:
    """Return false for owned work still settling; true only after durable handling.

    The caller owns a persisted event-delivery claim. A false result must release
    that claim for later delivery, not acknowledge or dead-letter the event.
    """

    child = await store.load(event.session_id)
    if child is None or child.parent_session_id is None:
        return True
    checkpoint = await store.load_checkpoint(child.parent_session_id)
    wait, prior_terminal = foreground_child_state_from_checkpoint(checkpoint)
    if wait is None:
        parent = await store.load(child.parent_session_id)
        if parent is None:
            return True
        attached = await load_attached_foreground_continuation(parent, store=store)
        if attached is None:
            # A child may resolve after observation but before the parent commits
            # its wait. Acknowledge only once its exact original effect settles;
            # otherwise keep the store-backed wakeup eligible across process loss.
            return not await _has_unsettled_parent_spawn(child, parent, store=store)
        if attached.terminal.event_id != event.id:
            return True
        selected = attached.terminal
        if (
            selected.wait.child_session_id != child.id
            or selected.wait.child_session_instance_id != child.instance_id
            or selected.event_digest
            != sha256(
                canonical_durable_json_bytes(
                    event.model_dump(mode="json"), "foreground_child_terminal"
                )
            ).hexdigest()
        ):
            raise RuntimeError(
                "Attached foreground wakeup conflicts with its consumed child outcome."
            )
        # Attaching a result does not finish the original interaction. In
        # particular, a crash immediately after publication must not silently
        # acknowledge the only remaining durable wakeup.
        terminal_interaction = await store.query_events(
            EventQuery(
                session_id=parent.id,
                interaction_id=selected.wait.parent_effect.interaction_id,
                event_types=(
                    EventType.INTERACTION_COMPLETED,
                    EventType.INTERACTION_FAILED,
                    EventType.INTERACTION_INTERRUPTED,
                ),
                limit=1,
            )
        )
        if terminal_interaction:
            return True
        if has_active_tasks(parent.id) or has_active_tasks(child.id):
            return False
        await resume(selected)
        completed = await store.query_events(
            EventQuery(
                session_id=parent.id,
                interaction_id=selected.wait.parent_effect.interaction_id,
                event_types=(
                    EventType.INTERACTION_COMPLETED,
                    EventType.INTERACTION_FAILED,
                    EventType.INTERACTION_INTERRUPTED,
                ),
                limit=1,
            )
        )
        return bool(completed)
    raw_wait = wait.model_dump(mode="json")
    if wait.child_session_id != child.id:
        return True
    if wait.child_session_instance_id != child.instance_id:
        raise RuntimeError("Child terminal wakeup conflicts with its retained wait identity.")
    child_checkpoint = await store.load_checkpoint(child.id)
    delegated = await owned_delegated_wait(store, child=child, checkpoint=child_checkpoint)
    # A retained tool round alone is recovery state, not positive proof of a
    # resumable human action. In particular, stopping an input pause can leave
    # such a round while closing the child interaction.
    pending_action_evidence_round_from_checkpoint(child_checkpoint)
    approval = approval_support.pending_approval_from_checkpoint(child_checkpoint)
    pending_input, _ = user_input_lifecycle_authority_from_checkpoint(
        child_checkpoint, current_run_epoch=child.run_epoch
    )
    parent = await store.load(child.parent_session_id)
    if parent is None:
        return True
    if parent.instance_id != wait.parent_effect.session_instance_id:
        raise RuntimeError("Child terminal wakeup belongs to another parent incarnation.")
    if parent.status in {SessionStatus.COMPLETED, SessionStatus.FAILED}:
        return True
    # An explicit parent stop closes the original interaction without claiming
    # that its delegated child/effect has settled. Keep that ownership evidence,
    # but acknowledge a later wakeup instead of retrying a revoked continuation.
    closed_interaction = await store.query_events(
        EventQuery(
            session_id=parent.id,
            interaction_id=wait.parent_effect.interaction_id,
            event_types=(
                EventType.INTERACTION_COMPLETED,
                EventType.INTERACTION_FAILED,
                EventType.INTERACTION_INTERRUPTED,
            ),
            limit=1,
        )
    )
    if closed_interaction:
        return True
    if has_active_tasks(parent.id) or has_active_tasks(child.id):
        return False
    child_profile = active_invocation_execution_profile_from_checkpoint(child_checkpoint)
    # Session terminal events are deliberately session-scoped. Their interaction
    # comes from the retained invocation profile, not an invented event field.
    if child_profile is not None and child_profile.interaction_id != wait.child_interaction_id:
        raise RuntimeError("Child terminal wakeup belongs to another interaction.")
    if child_profile is None or not active_invocation_execution_profile_is_released(
        child_profile, session_id=child.id, run_epoch=child.run_epoch
    ):
        if (
            child_profile is not None
            and approval is None
            and pending_input is None
            and delegated is None
        ):
            await settle(wait, event)
            settled_child = await store.load(child.id)
            settled_checkpoint = await store.load_checkpoint(child.id)
            settled_profile = active_invocation_execution_profile_from_checkpoint(
                settled_checkpoint
            )
            if (
                settled_child is None
                or settled_child.instance_id != child.instance_id
                or settled_profile is None
                or settled_profile.interaction_id != wait.child_interaction_id
                or not active_invocation_execution_profile_is_released(
                    settled_profile, session_id=child.id, run_epoch=settled_child.run_epoch
                )
            ):
                return False
            child = settled_child
        else:
            return False
    outcome = await store.summarize_outcome(child.id)
    if event.type == EventType.SESSION_INTERRUPTED and (
        approval is not None or pending_input is not None or delegated is not None
    ):
        if outcome.terminal_event is None or outcome.terminal_event.event != event:
            return True
        # Repeated pauses change discovery, not execution authority. The exact
        # child matcher validates the new action under the delivery owner.
        await refresh(wait, event)
        return True
    if outcome.terminal_event is None or outcome.terminal_event.event.id != event.id:
        raise RuntimeError("Child terminal wakeup is not the exact current outcome.")
    terminal = ForegroundChildTerminal.model_validate(
        {
            "wait": wait,
            "child_released_run_epoch": child.run_epoch,
            "event_id": event.id,
            "event_type": str(event.type),
            "event_digest": sha256(
                canonical_durable_json_bytes(
                    outcome.terminal_event.event.model_dump(mode="json"),
                    "foreground_child_terminal",
                )
            ).hexdigest(),
        }
    )
    if prior_terminal is not None and prior_terminal != terminal:
        raise RuntimeError("Child terminal delivery conflicts with its durable selection.")
    await resume(terminal)
    remaining = await store.load_checkpoint(parent.id)
    return remaining is None or remaining.get(FOREGROUND_CHILD_WAIT_KEY) != raw_wait
