"""Close a ``sleep_until`` pause once, handing its round to the shared pause."""

from __future__ import annotations

from cayu._validation import copy_durable_record
from cayu.approvals.user_input import (
    PENDING_USER_INPUT_CHECKPOINT_KEY,
    PendingUserInput,
    pending_user_input_digest,
    pending_user_input_identity,
)
from cayu.approvals.wakes import (
    PENDING_WAKE_CHECKPOINT_KEY,
    WakeCloseTransition,
    pending_session_wake_digest,
    pending_session_wake_from_checkpoint,
)
from cayu.events import Event, EventType
from cayu.execution_units import ToolRoundIdentity
from cayu.runtime import _approval_publication as approval_publication
from cayu.runtime._event_writer import RuntimeEventWriter
from cayu.runtime._tool_round_staging import _event_with_tool_round_authority
from cayu.sessions.records import Session, SessionStatus
from cayu.sessions.store import SessionStore


async def close_pending_wake(
    *,
    session_store: SessionStore,
    event_writer: RuntimeEventWriter,
    session: Session,
    transition: WakeCloseTransition,
) -> PendingUserInput | None:
    """Publish ``wake-close`` for the session's open wake; ``None`` if it has none.

    Closing replaces ``pending_wake`` with its continuation under
    ``pending_user_input`` in one fenced publication on the interrupted session.
    The continuation is then resolved (``woke``) or superseded (``interrupted``)
    by the shared pause machinery, exactly once.
    """

    checkpoint = await session_store.load_checkpoint(session.id)
    wake = pending_session_wake_from_checkpoint(checkpoint)
    if wake is None:
        return None
    assert checkpoint is not None
    pending = wake.continuation
    identity = ToolRoundIdentity(
        tool_round_id=pending.tool_round_id,
        model_step_id=pending.model_step_id,
        model_attempt_id=pending.model_attempt_id,
    )
    target_checkpoint = copy_durable_record(checkpoint, "checkpoint")
    target_checkpoint.pop(PENDING_WAKE_CHECKPOINT_KEY)
    target_checkpoint[PENDING_USER_INPUT_CHECKPOINT_KEY] = pending.model_dump(mode="json")
    pause_digest = pending_user_input_digest(pending)
    event = _event_with_tool_round_authority(
        Event(
            type=EventType.SESSION_CHECKPOINTED,
            session_id=session.id,
            interaction_id=pending.source_interaction_id,
            agent_name=pending.agent_name,
            environment_name=pending.environment_name,
            tool_name=pending.tool_name,
            payload={
                "checkpoint": PENDING_WAKE_CHECKPOINT_KEY,
                "transition": transition,
                "wake_id": wake.wake_id,
                "input_id": pending.input_id,
                "tool_call_id": pending.tool_call_id,
                "source_run_epoch": pending.source_run_epoch,
                "pause_digest": pause_digest,
                **identity.payload(),
            },
        ),
        identity,
        "wake_id",
        "input_id",
        "tool_call_id",
    )
    prepared = approval_publication.prepare_pending_action_publication(
        session_id=session.id,
        publication_id=f"wake-close:{wake.wake_id}",
        kind="wake-close",
        intent={
            **pending_user_input_identity(pending),
            "wake_pause_digest": pending_session_wake_digest(wake),
            "transition": transition,
            "event_ids": [event.id],
        },
        source_checkpoint=checkpoint,
        target_checkpoint=target_checkpoint,
        events=[event],
        expected_statuses={SessionStatus.INTERRUPTED},
        expected_run_epoch=session.run_epoch,
        expected_transcript_cursor=await session_store.load_transcript_cursor(session.id),
    )
    cancellation = await approval_publication.publish_pending_action_with_exact_replay(
        prepared,
        session_store=session_store,
        event_writer=event_writer,
    )
    if cancellation is not None:
        raise cancellation
    return pending
