"""Coordinate foreground calls suspended on a child's human action.

Saved wait and continuation evidence belongs to the session checkpoint component.
Runtime owners authenticate both ends before installing or consuming a wait.
"""

from __future__ import annotations

from typing import Any

from cayu._validation import (
    canonical_durable_json_bytes,
)
from cayu.approvals.user_input import user_input_lifecycle_authority_from_checkpoint
from cayu.events import Event, EventType, event_with_runtime_payload_authority
from cayu.runtime._child_session_identity import (
    ChildSessionKind,
    ChildSessionRecoveryMatcher,
    generate_child_session_id,
)
from cayu.runtime._foreground_subagent_recovery import (
    ForegroundSubagentRecoveryRequired,
    project_authenticated_child_result,
)
from cayu.runtime._tool_effect_state import ToolEffectRecord, ToolEffectStateOwner
from cayu.sessions import _foreground_child_checkpoint as foreground_checkpoint
from cayu.sessions import _pending_approval_reader as pending_approval_reader
from cayu.sessions._execution_profile_checkpoint import (
    active_invocation_execution_profile_from_checkpoint,
    active_invocation_execution_profile_is_released,
)
from cayu.sessions._pending_tool_round import PENDING_TOOL_ROUND_CHECKPOINT_KEY
from cayu.sessions._tool_effect_intent import ToolEffectIntent
from cayu.sessions.base import (
    SessionStore,
    runtime_publication_checkpoint_mutation,
    runtime_publication_checkpoint_value_digest,
)
from cayu.sessions.event_queries import EventOrder, EventQuery
from cayu.sessions.pending_actions import pending_action_evidence_round_from_checkpoint
from cayu.sessions.records import Session, SessionStatus
from cayu.tools.base import ToolResult


def event_with_foreground_child_wait_authority(
    event: Event, wait: foreground_checkpoint.ForegroundChildWait
) -> Event:
    """Attest discovery identities at the authenticated runtime wait publisher.

    Callers must first authenticate the wait against its child and parent effect;
    merely reconstructing a wait or copying its identifiers grants no authority.
    """
    if (
        event.type
        not in {EventType.SESSION_INTERRUPTED, EventType.SESSION_DELEGATED_ACTION_UPDATED}
        or event.session_id != wait.parent_effect.session_id
        or event.payload.get("interruption_type") != "waiting_on_child_action"
        or any(
            event.payload.get(key) != value
            for key, value in wait.delegated_action_reference().items()
        )
    ):
        raise ValueError("Foreground discovery event disagrees with its authenticated wait.")
    return event_with_runtime_payload_authority(event, "child_session_id", "action_id")


class ForegroundChildActionRequired(ForegroundSubagentRecoveryRequired):
    """Runtime-owned pause after exact child and effect evidence was joined."""

    def __init__(self, wait: foreground_checkpoint.ForegroundChildWait) -> None:
        super().__init__(
            child_session_id=wait.child_session_id,
            tool_round_id=wait.parent_effect.tool_round_id,
            tool_call_id=wait.parent_effect.tool_call_id,
        )
        self.wait = foreground_checkpoint.ForegroundChildWait.model_validate_json(
            wait.model_dump_json()
        )

    def interruption_evidence(self) -> dict[str, object]:
        return {
            **super().interruption_evidence(),
            "interruption_type": "waiting_on_child_action",
            **self.wait.delegated_action_reference(),
            "model_step_id": self.wait.parent_effect.model_step_id,
            "model_attempt_id": self.wait.parent_effect.model_attempt_id,
        }


async def retain_foreground_child_wait(
    store: SessionStore,
    *,
    parent: Session,
    effect: ToolEffectRecord,
    wait: foreground_checkpoint.ForegroundChildWait,
) -> None:
    """Retain the same wait under either live execution or owned recovery.

    The caller has authenticated the child through its registered matcher and
    owns the parent run fence. Recovery may already have classified the consumed
    effect as unknown; installing a wait must not invent a second observation or
    grant permission to replay that effect.
    """
    if wait.parent_effect != effect.intent or (
        parent.id != effect.intent.session_id
        or parent.instance_id != effect.intent.session_instance_id
    ):
        raise RuntimeError("Foreground wait conflicts with its owned parent effect.")
    source = await store.load_checkpoint(parent.id)
    existing, terminal = foreground_checkpoint.foreground_child_state_from_checkpoint(source)
    if existing is not None:
        if existing == wait and terminal is None and effect.state == "outcome_unknown":
            return
        prior = await ToolEffectStateOwner(store).resolve_call(
            parent,
            tool_round_id=existing.parent_effect.tool_round_id,
            tool_call_id=existing.parent_effect.tool_call_id,
        )
        # Sequential calls share a round, not a wait. Only a positively settled
        # prior effect authorizes transferring the slot; its immutable terminal
        # record remains the exactly-once evidence for delayed delivery/recovery.
        if (
            terminal is None
            or existing.parent_effect == wait.parent_effect
            or prior is None
            or prior.intent != existing.parent_effect
            or prior.state not in {"completed", "failed"}
            or prior.terminal is None
            or any(
                getattr(existing.parent_effect, field) != getattr(wait.parent_effect, field)
                for field in (
                    "session_id",
                    "session_instance_id",
                    "interaction_id",
                    "tool_round_id",
                    "execution_profile_fingerprint",
                    "approval_id",
                    "pause_id",
                )
            )
        ):
            raise RuntimeError("Foreground wait conflicts with the retained delegation.")
    if source is None:
        raise RuntimeError("Foreground wait has no source checkpoint.")
    profile = active_invocation_execution_profile_from_checkpoint(source)
    if (
        profile is None
        or profile.interaction_id != effect.intent.interaction_id
        or profile.profile.fingerprint != effect.intent.execution_profile_fingerprint
    ):
        raise RuntimeError("Foreground wait has no matching original invocation profile.")
    target = {
        **source,
        foreground_checkpoint.FOREGROUND_CHILD_WAIT_KEY: wait.model_dump(mode="json"),
    }
    target.pop(foreground_checkpoint.FOREGROUND_CHILD_TERMINAL_KEY, None)
    if effect.state == "executing":
        await ToolEffectStateOwner(store).transition(
            effect,
            state="outcome_unknown",
            run_epoch=parent.run_epoch,
            mutation=runtime_publication_checkpoint_mutation(source, target),
        )
        return
    if effect.state != "outcome_unknown":
        raise RuntimeError("Foreground wait cannot replace a settled or undispatched effect.")
    guarded_keys = (
        PENDING_TOOL_ROUND_CHECKPOINT_KEY,
        foreground_checkpoint.FOREGROUND_CHILD_WAIT_KEY,
        foreground_checkpoint.FOREGROUND_CHILD_TERMINAL_KEY,
        foreground_checkpoint.FOREGROUND_CHILD_POST_ACTION_CONTINUATION_KEY,
        foreground_checkpoint.FOREGROUND_PARENT_CONTINUATION_KEY,
    )

    def authority(checkpoint: dict[str, Any]) -> bytes:
        return canonical_durable_json_bytes(
            {
                key: {"present": key in checkpoint, "value": checkpoint.get(key)}
                for key in guarded_keys
            },
            "foreground_wait_source",
        )

    expected_checkpoint = authority(source)

    def retain(current_parent: Session, current: dict[str, Any] | None) -> dict[str, Any]:
        if (
            current_parent.instance_id != parent.instance_id
            or current is None
            or authority(current) != expected_checkpoint
        ):
            raise RuntimeError("Foreground wait source changed before recovery publication.")
        # Generic checkpoint callbacks deliberately cannot read or replace the
        # active invocation profile. The typed invocation owner preserves that
        # authority under the expected run epoch below. Compare only this
        # feature's visible fields, preserving independent recovery heartbeats.
        updated = {
            **current,
            foreground_checkpoint.FOREGROUND_CHILD_WAIT_KEY: wait.model_dump(mode="json"),
        }
        updated.pop(foreground_checkpoint.FOREGROUND_CHILD_TERMINAL_KEY, None)
        return updated

    await store.publish_checkpoint_and_events(
        parent.id,
        checkpoint_transform=retain,
        events=[],
        expected_statuses={parent.status},
        expected_run_epoch=parent.run_epoch,
    )


async def _load_matching_foreground_child(
    store: SessionStore,
    *,
    parent: Session,
    intent: ToolEffectIntent,
    matcher: ChildSessionRecoveryMatcher,
    arguments: dict[str, Any],
) -> Session | None:
    if intent.session_id != parent.id or intent.session_instance_id != parent.instance_id:
        raise RuntimeError("Foreground wait conflicts with its parent incarnation.")
    child = await store.load(
        generate_child_session_id(
            kind=ChildSessionKind.SUBAGENT,
            parent_session_id=parent.id,
            logical_spawn_id=intent.idempotency_key,
        )
    )
    if child is None:
        return None
    subagent = child.metadata.get("subagent")
    if type(subagent) is not dict or subagent.get("mode") != "foreground":
        return None
    if not matcher.matches_recoverable_child(
        child,
        parent_invocation=parent.invocation,
        parent_session_id=parent.id,
        causal_budget_id=parent.causal_budget_id,
        environment_name=parent.environment_name,
        tool_call_id=intent.tool_call_id,
        idempotency_key=intent.idempotency_key,
        arguments=arguments,
        require_fingerprint=True,
    ):
        raise RuntimeError("Foreground wait has conflicting child spawn evidence.")
    return child


async def project_current_foreground_child_result(
    store: SessionStore,
    *,
    parent: Session,
    intent: ToolEffectIntent,
    matcher: ChildSessionRecoveryMatcher,
    arguments: dict[str, Any],
) -> ToolResult | None:
    """Refresh a settled child projection before the live result is sealed."""
    child = await _load_matching_foreground_child(
        store, parent=parent, intent=intent, matcher=matcher, arguments=arguments
    )
    if child is None:
        return None
    if child.status in {
        SessionStatus.PENDING,
        SessionStatus.RUNNING,
        SessionStatus.INTERRUPTING,
    }:
        # The collector can return the prior pause after another caller has
        # already resumed this exact child. That old projection is not a
        # terminal outcome and must not settle the original parent effect.
        raise ForegroundSubagentRecoveryRequired(
            child_session_id=child.id,
            tool_round_id=intent.tool_round_id,
            tool_call_id=intent.tool_call_id,
        )
    if child.status not in {
        SessionStatus.COMPLETED,
        SessionStatus.FAILED,
        SessionStatus.INTERRUPTED,
    }:
        return None
    checkpoint = await store.load_checkpoint(child.id)
    if await owned_delegated_wait(store, child=child, checkpoint=checkpoint) is not None:
        raise ForegroundSubagentRecoveryRequired(
            child_session_id=child.id,
            tool_round_id=intent.tool_round_id,
            tool_call_id=intent.tool_call_id,
        )
    pending_action_evidence_round_from_checkpoint(checkpoint)
    if pending_approval_reader.pending_approval_from_checkpoint(checkpoint) is not None or (
        user_input_lifecycle_authority_from_checkpoint(
            checkpoint, current_run_epoch=child.run_epoch
        )[0]
        is not None
    ):
        return None
    profile = active_invocation_execution_profile_from_checkpoint(checkpoint)
    if profile is None or not active_invocation_execution_profile_is_released(
        profile, session_id=child.id, run_epoch=child.run_epoch
    ):
        raise ForegroundSubagentRecoveryRequired(
            child_session_id=child.id,
            tool_round_id=intent.tool_round_id,
            tool_call_id=intent.tool_call_id,
        )
    result = await project_authenticated_child_result(
        matcher,
        child,
        tool_call_id=intent.tool_call_id,
        tool_name=intent.tool_name,
        tool_round_id=intent.tool_round_id,
    )
    if await store.load(child.id) != child:
        raise RuntimeError("Foreground child changed during terminal projection.")
    return result


async def observe_foreground_child_wait(
    store: SessionStore,
    *,
    parent: Session,
    intent: ToolEffectIntent,
    matcher: ChildSessionRecoveryMatcher,
    arguments: dict[str, Any],
) -> foreground_checkpoint.ForegroundChildWait | None:
    """Observe an authenticated human pause; the parent owner fences its commit."""
    child = await _load_matching_foreground_child(
        store, parent=parent, intent=intent, matcher=matcher, arguments=arguments
    )
    if child is None:
        return None
    if child.status == SessionStatus.RUNNING:
        return await _observe_resumed_child_wait(store, child=child, intent=intent)
    if child.status in {SessionStatus.COMPLETED, SessionStatus.FAILED}:
        profile = active_invocation_execution_profile_from_checkpoint(
            await store.load_checkpoint(child.id)
        )
        if profile is not None and not active_invocation_execution_profile_is_released(
            profile, session_id=child.id, run_epoch=child.run_epoch
        ):
            # Terminal publication precedes cleanup/release. Preserve the prior
            # action wait until the exact terminal invocation actually settles.
            return await _observe_resumed_child_wait(store, child=child, intent=intent)
        return None
    if child.status != SessionStatus.INTERRUPTED:
        return None
    subagent = child.metadata.get("subagent")
    assert type(subagent) is dict
    checkpoint = await store.load_checkpoint(child.id)
    delegated = await owned_delegated_wait(store, child=child, checkpoint=checkpoint)
    approval = pending_approval_reader.pending_approval_from_checkpoint(checkpoint)
    pending_input, _ = user_input_lifecycle_authority_from_checkpoint(
        checkpoint, current_run_epoch=child.run_epoch
    )
    if approval is None and pending_input is None and delegated is None:
        return None
    # Reuse the existing paired-round validator, including its conflicting
    # approval/input topology checks; two individually valid actions are not one
    # unambiguous child pause.
    pending_action_evidence_round_from_checkpoint(checkpoint)
    interactions = await store.query_latest_interaction_events(child.id, limit=1)
    if (
        len(interactions) != 1
        or interactions[0].event.type != EventType.INTERACTION_PAUSED
        or interactions[0].event.interaction_id is None
    ):
        raise RuntimeError("Foreground child action lacks a paused interaction.")
    interaction_id = interactions[0].event.interaction_id
    if pending_input is not None and (
        pending_input.session_id != child.id
        or pending_input.session_instance_id != child.instance_id
        or pending_input.source_interaction_id != interaction_id
    ):
        raise RuntimeError("Foreground child input conflicts with its incarnation or interaction.")
    if await store.load(child.id) != child:
        raise RuntimeError("Foreground child changed during pause observation.")
    if delegated is not None:
        action_kind = "delegated_action"
        # Discovery points to the immediate child's logical delegation, not to
        # a copied approval/input owned by a deeper descendant.
        action_id = delegated.parent_effect.idempotency_key
    elif approval is not None:
        action_kind = "tool_approval"
        action_id = approval.approval_id
    else:
        assert pending_input is not None
        action_kind = "user_input"
        action_id = pending_input.input_id
    return foreground_checkpoint.ForegroundChildWait(
        parent_effect=intent,
        child_session_id=child.id,
        child_session_instance_id=child.instance_id,
        child_interaction_id=interaction_id,
        child_spawn_fingerprint=subagent.get("spawn_fingerprint"),
        child_action_kind=action_kind,
        child_action_id=action_id,
        child_action_run_epoch=child.run_epoch,
        revision=1,
    )


async def owned_delegated_wait(
    store: SessionStore, *, child: Session, checkpoint: dict[str, Any] | None
) -> foreground_checkpoint.ForegroundChildWait | None:
    """Authenticate one hop of a delegated pause, without recursively scanning.

    A stopped child can retain unresolved effect evidence. Its closed interaction
    revokes continuation; that retained wait must not hide its terminal outcome.
    """
    wait, selected = foreground_checkpoint.foreground_child_state_from_checkpoint(checkpoint)
    if wait is None:
        return None
    intent = wait.parent_effect
    if intent.session_id != child.id or intent.session_instance_id != child.instance_id:
        raise RuntimeError("Delegated child wait belongs to another invocation.")
    closed = await store.query_events(
        EventQuery(
            session_id=child.id,
            interaction_id=intent.interaction_id,
            event_types=(
                EventType.INTERACTION_COMPLETED,
                EventType.INTERACTION_FAILED,
                EventType.INTERACTION_INTERRUPTED,
            ),
            limit=1,
        )
    )
    if closed:
        return None
    profile = active_invocation_execution_profile_from_checkpoint(checkpoint)
    pending = pending_action_evidence_round_from_checkpoint(checkpoint)
    if (
        pending_approval_reader.pending_approval_from_checkpoint(checkpoint) is not None
        or user_input_lifecycle_authority_from_checkpoint(checkpoint)[0] is not None
    ):
        from cayu.runtime._foreground_gate_continuation import load_gate_request

        if await load_gate_request(store, parent=child, wait=wait) is None:
            raise RuntimeError("Delegated child gate lacks its accepted resolution.")
    effect = await ToolEffectStateOwner(store).resolve_call(
        child, tool_round_id=intent.tool_round_id, tool_call_id=intent.tool_call_id
    )
    descendant = await store.load(wait.child_session_id)
    if (
        profile is None
        or profile.interaction_id != intent.interaction_id
        or profile.profile.fingerprint != intent.execution_profile_fingerprint
        or pending is None
        or (pending.tool_round_id, pending.model_step_id, pending.model_attempt_id)
        != (intent.tool_round_id, intent.model_step_id, intent.model_attempt_id)
        or not any(call.tool_call_id == intent.tool_call_id for call in pending.tool_calls)
        or effect is None
        or effect.intent != intent
        or effect.state
        not in (
            {"outcome_unknown", "completed", "failed", "reconciled_completed", "reconciled_failed"}
            if selected is not None
            else {"outcome_unknown"}
        )
        or effect.child_recovery_arguments is None
        or descendant is None
        or descendant.instance_id != wait.child_session_instance_id
        or descendant.parent_session_id != child.id
        or descendant.id
        != generate_child_session_id(
            kind=ChildSessionKind.SUBAGENT,
            parent_session_id=child.id,
            logical_spawn_id=intent.idempotency_key,
        )
        or type(descendant.metadata.get("subagent")) is not dict
        or descendant.metadata["subagent"].get("spawn_fingerprint") != wait.child_spawn_fingerprint
    ):
        raise RuntimeError("Delegated child wait lacks its exact original effect and lineage.")
    if await store.load(child.id) != child:
        raise RuntimeError("Child changed while observing its delegated wait.")
    return wait


async def _observe_resumed_child_wait(
    store: SessionStore, *, child: Session, intent: ToolEffectIntent
) -> foreground_checkpoint.ForegroundChildWait | None:
    """Recover a retired human pause from its store-owned opening publication.

    Events select a candidate only. The insert-only receipt and its complete
    event digest authenticate the action, original epoch and interaction. This
    records a wait, never permission to resolve or replay the child action.
    """
    checkpoint = await store.load_checkpoint(child.id)
    profile = active_invocation_execution_profile_from_checkpoint(checkpoint)
    if profile is None:
        return None
    if profile.session_id != child.id or profile.run_epoch != child.run_epoch:
        raise RuntimeError("Resumed foreground child has conflicting invocation ownership.")
    pauses = await store.query_events(
        EventQuery(
            session_id=child.id,
            interaction_id=profile.interaction_id,
            event_type=EventType.INTERACTION_PAUSED,
            order_by=EventOrder.SEQUENCE_DESC,
            limit=1,
        )
    )
    if not pauses:
        return None
    kind = pauses[0].event.payload.get("pending_action_kind")
    if kind == "waiting_on_child_action":
        delegated = await owned_delegated_wait(store, child=child, checkpoint=checkpoint)
        if delegated is None:
            from cayu.runtime._foreground_child_continuation import (
                load_attached_foreground_continuation,
            )

            attached = await load_attached_foreground_continuation(child, store=store)
            if attached is None:
                raise RuntimeError("Resumed delegated child lacks its original wait or receipt.")
            delegated = attached.terminal.wait
        settlement = await store.load_historical_interaction_settlement(
            child.id,
            expected_session_instance_id=child.instance_id,
            expected_event=pauses[0].event,
            expected_profile=profile.profile,
        )
        if (
            settlement.transition.to_status != SessionStatus.INTERRUPTED
            or delegated.parent_effect.interaction_id != profile.interaction_id
            or await store.load(child.id) != child
        ):
            raise RuntimeError("Resumed delegated child has conflicting pause evidence.")
        return foreground_checkpoint.ForegroundChildWait(
            parent_effect=intent,
            child_session_id=child.id,
            child_session_instance_id=child.instance_id,
            child_interaction_id=profile.interaction_id,
            child_spawn_fingerprint=child.metadata["subagent"]["spawn_fingerprint"],
            child_action_kind="delegated_action",
            child_action_id=delegated.parent_effect.idempotency_key,
            child_action_run_epoch=settlement.session.run_epoch + 1,
            revision=1,
        )
    if kind not in {"tool_approval", "user_input"}:
        return None
    action_type = (
        EventType.TOOL_CALL_APPROVAL_REQUESTED
        if kind == "tool_approval"
        else EventType.SESSION_AWAITING_USER_INPUT
    )
    actions = await store.query_events(
        EventQuery(
            session_id=child.id,
            interaction_id=profile.interaction_id,
            event_type=action_type,
            before_sequence=pauses[0].sequence,
            order_by=EventOrder.SEQUENCE_DESC,
            limit=1,
        )
    )
    if not actions:
        raise RuntimeError("Resumed foreground child has no durable action opening.")
    action = actions[0].event
    identity_key = "approval_id" if kind == "tool_approval" else "input_id"
    action_id = action.payload.get(identity_key)
    if type(action_id) is not str or not action_id:
        raise RuntimeError("Resumed foreground child action identity is malformed.")
    publication_kind = "approval-open" if kind == "tool_approval" else "user-input-open"
    receipt = await store.load_runtime_publication_receipt(
        child.id, f"{publication_kind}:{action_id}"
    )
    if (
        receipt is None
        or receipt.kind != publication_kind
        or receipt.interaction_id != profile.interaction_id
        or receipt.source_run_epoch >= child.run_epoch
        or type(receipt.intent.get("schema_version")) is not int
        or receipt.intent.get("schema_version") != 1
        or receipt.intent.get(identity_key) != action_id
        or receipt.intent.get("event_ids") != list(receipt.appended_event_ids)
        or len(receipt.appended_event_ids) != 2
        or receipt.appended_event_ids[-1] != action.id
        or receipt.referenced_events
        or receipt.transcript_start_cursor != receipt.transcript_end_cursor
    ):
        raise RuntimeError("Resumed foreground child lacks an exact action opening receipt.")
    if kind == "user_input" and (
        receipt.intent.get("session_id") != child.id
        or receipt.intent.get("session_instance_id") != child.instance_id
        or receipt.intent.get("source_interaction_id") != profile.interaction_id
        or type(receipt.intent.get("source_run_epoch")) is not int
        or receipt.intent.get("source_run_epoch") != receipt.source_run_epoch
    ):
        raise RuntimeError("Resumed foreground child input opening has conflicting lineage.")
    opening_events = []
    for event_id in receipt.appended_event_ids:
        records = await store.query_events(
            EventQuery(session_id=child.id, event_id=event_id, limit=2)
        )
        if len(records) != 1:
            raise RuntimeError("Resumed foreground child opening evidence is missing.")
        opening_events.append(records[0].event)
    if [event.type for event in opening_events] != [
        EventType.SESSION_CHECKPOINTED,
        action_type,
    ] or runtime_publication_checkpoint_value_digest(
        [event.model_dump(mode="json") for event in opening_events]
    ) != receipt.events_digest:
        raise RuntimeError("Resumed foreground child opening evidence conflicts with its receipt.")
    settlement = await store.load_historical_interaction_settlement(
        child.id,
        expected_session_instance_id=child.instance_id,
        expected_event=pauses[0].event,
        expected_profile=profile.profile,
    )
    if (
        settlement.transition.to_status != SessionStatus.INTERRUPTED
        or settlement.session.run_epoch < receipt.source_run_epoch
    ):
        raise RuntimeError("Resumed foreground child lacks its exact original pause settlement.")
    if await store.load(child.id) != child:
        raise RuntimeError("Foreground child changed during resumed-pause observation.")
    subagent = child.metadata.get("subagent")
    assert type(subagent) is dict
    return foreground_checkpoint.ForegroundChildWait(
        parent_effect=intent,
        child_session_id=child.id,
        child_session_instance_id=child.instance_id,
        child_interaction_id=profile.interaction_id,
        child_spawn_fingerprint=subagent.get("spawn_fingerprint"),
        child_action_kind=kind,
        child_action_id=action_id,
        # Invocation release advances the settled source epoch by one. Match
        # the ordinary paused-child observer's released fence, not the epoch
        # at which the action's opening publication ran.
        child_action_run_epoch=settlement.session.run_epoch + 1,
        revision=1,
    )
