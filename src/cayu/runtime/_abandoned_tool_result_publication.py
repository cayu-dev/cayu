"""Publish the tool results a lost run staged before its session is abandoned.

A tool call that finished stages its terminal event in the pending tool round
before publishing it. A worker lost between the two leaves the call's result in
the round's ``staged_terminals``. Abandoning the session retires the round into
its abandonment record, so without this step that result is never published.

Abandoning continues nothing, so publishing runs nothing either: no tool, hook
or model. A stage whose after-tool hooks completed holds the exact terminal
event the lost run was about to append, and is published as it is. Any other
stage is published with its result withheld, as recovery publishes one whose
hook redactor it can't rebuild: its hooks could still rewrite or redact the
result, and running them needs the run's hooks and environment, which an
abandonment does not have.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from cayu.events import Event
from cayu.execution_profiles import ExecutionProfileIdentity
from cayu.execution_units import ToolRoundIdentity
from cayu.runtime import _approval_support as approval_support
from cayu.runtime import _resume_ledger as resume_ledger
from cayu.runtime import _runtime_records as runtime_records
from cayu.runtime import _tool_round_recovery as tool_round_recovery
from cayu.runtime._event_writer import RuntimeEventWriter
from cayu.runtime._tool_invocation.context import _restore_targeted_tool_invocation_event_authority
from cayu.runtime._tool_round_staging import (
    _committed_stage_capture,
    _ToolRoundPublicationCoordinator,
)
from cayu.sessions import _pending_tool_round as pending_rounds
from cayu.sessions import _pending_tool_round_reader as pending_round_reader
from cayu.sessions import _staged_tool_terminal_reader as staged_terminal_reader
from cayu.sessions._assistant_tool_round_publication import StagedToolCallTerminal
from cayu.sessions._execution_profile_checkpoint import (
    active_invocation_execution_profile_from_checkpoint,
)
from cayu.sessions.records import Session
from cayu.sessions.recovery import _incomplete_recovery_claim_from_checkpoint
from cayu.sessions.store import SessionStore
from cayu.vaults.redaction import SecretRedactor
from cayu.workspaces.observation_recovery import workspace_observations_from_checkpoint

if TYPE_CHECKING:
    from cayu.runtime._workspace_observation_recovery import WorkspaceObservationRecovery

CheckpointTransform = Callable[[Session, dict[str, Any] | None], dict[str, Any]]


@dataclass(frozen=True, slots=True)
class AbandonedToolResults:
    """What publishing an abandoned round's staged results did."""

    pending_round: pending_rounds.PendingToolRound
    # For the abandonment record: each staged result whose terminal event is durable,
    # published here or appended by the lost run just before it died.
    published_tool_calls: tuple[dict[str, str], ...] = ()
    # The terminal tool events, after any workspace observation events.
    events: tuple[Event, ...] = ()
    workspace_observation_events: tuple[Event, ...] = ()


async def publish_abandoned_tool_results(
    *,
    session: Session,
    claim_id: str,
    pending_round: pending_rounds.PendingToolRound,
    session_store: SessionStore,
    event_writer: RuntimeEventWriter,
    redactor: SecretRedactor,
    workspace_observation_recovery: WorkspaceObservationRecovery,
    registered_environment: runtime_records.RegisteredEnvironment | None,
) -> AbandonedToolResults:
    """Publish ``pending_round``'s staged, unpublished results under recovery claim ``claim_id``.

    ``session`` is the session as the claim fenced it. Each stage write checks the
    claim and the fenced run in the same transform, so a session that changed
    since publishes nothing more. A result whose terminal event is already durable
    (the lost run appended it and died before committing its round, or an earlier
    abandonment pass published it) isn't published again, but is listed. Workspace observations still open are closed first, as
    recovery closes them before publishing a round's results, since publishing
    a stage changes the event its observation is bound to. Returns the round as
    it now stands, with the published calls and the events.
    """

    identity = pending_rounds.pending_tool_round_identity(pending_round)
    # The round as it stands under the claim: the lost run may have staged more
    # before its run was fenced.
    pending_round = await _current_round(session_store, session, identity, redactor)
    unpublished, published_ids = await _unpublished_stages(session_store, session, pending_round)
    if not unpublished:
        return AbandonedToolResults(
            pending_round=pending_round,
            published_tool_calls=_published_tool_calls(pending_round, published_ids),
        )
    checkpoint = await session_store.load_checkpoint(session.id)
    snapshot = active_invocation_execution_profile_from_checkpoint(checkpoint)
    profile: ExecutionProfileIdentity | None = None if snapshot is None else snapshot.profile
    observation_events: tuple[Event, ...] = ()
    if workspace_observations_from_checkpoint(checkpoint):
        (
            _checkpoint,
            pending_round,
            observation_events,
        ) = await workspace_observation_recovery.settle_tool_round_workspace_observations(
            session=session,
            registered_environment=registered_environment,
            execution_profile=profile,
            invocation_context=None,
            checkpoint=checkpoint,
            pending_round=pending_round,
        )
        unpublished, published_ids = await _unpublished_stages(
            session_store, session, pending_round
        )

    async def write_stage(
        transform: CheckpointTransform, *, tool_call_id: str
    ) -> StagedToolCallTerminal | None:
        def fenced(current: Session, current_checkpoint: dict[str, Any] | None) -> dict[str, Any]:
            claim = _incomplete_recovery_claim_from_checkpoint(current_checkpoint)
            if (
                (current.instance_id, current.run_epoch) != (session.instance_id, session.run_epoch)
                or claim is None
                or claim[0] != claim_id
            ):
                raise RuntimeError("Session changed before its staged tool results were published.")
            return transform(current, current_checkpoint)

        capturing, committed = _committed_stage_capture(fenced, tool_call_id=tool_call_id)
        await session_store.transform_checkpoint(session.id, capturing)
        return committed()

    coordinator = _ToolRoundPublicationCoordinator(
        session_id=session.id,
        session_instance_id=session.instance_id,
        run_epoch=session.run_epoch,
        tool_round_identity=identity,
        session_store=session_store,
        redactor=redactor,
        execution_profile=profile,
        tool_exposure=pending_round.tool_exposure,
    )
    calls = {call.tool_call_id: call for call in pending_round.tool_calls}
    events: list[Event] = []
    for staged in unpublished:
        if staged.hooks_state != "completed":
            stored = await write_stage(
                tool_round_recovery.completed_staged_terminal_transform(
                    tool_round_identity=identity,
                    event=tool_round_recovery.abandoned_before_hooks_recovery_event(staged.event),
                ),
                tool_call_id=staged.tool_call_id,
            )
            if stored is None:
                raise RuntimeError("Abandoned tool result lost its stage.")
            staged = stored
        staged = await coordinator.start_publication(staged, write_stage=write_stage)
        # The checkpoint keeps a targeted invocation's attribution, not its authority.
        terminal = _restore_targeted_tool_invocation_event_authority(
            coordinator.restore_started_publication_authority(staged),
            approval_support.tool_call_request_from_pending(calls[staged.tool_call_id]),
            redactor=redactor,
        )
        event = await event_writer.emit(terminal)
        coordinator.terminal_published(event.id)
        events.append(event)
        published_ids[staged.tool_call_id] = event.id
    current = await _current_round(session_store, session, identity, redactor)
    return AbandonedToolResults(
        pending_round=current,
        published_tool_calls=_published_tool_calls(current, published_ids),
        events=(*observation_events, *events),
        workspace_observation_events=observation_events,
    )


async def _unpublished_stages(
    session_store: SessionStore,
    session: Session,
    pending_round: pending_rounds.PendingToolRound,
) -> tuple[list[StagedToolCallTerminal], dict[str, str]]:
    """The round's recovery-safe stages with no durable terminal event yet, and the
    event ids of those that have one, by call.

    A durable terminal must be the stage's own event. It is matched by id, not by
    content: the recovery-safe reading of a stage whose secret scope was
    incomplete differs from what an earlier, interrupted pass published for it.
    """

    stored = {item.tool_call_id: item for item in pending_round.staged_terminals}
    if not stored:
        return [], {}
    lifecycle = await tool_round_recovery.load_tool_round_lifecycle_events(
        session_store, session_id=session.id, pending_round=pending_round
    )
    durable: dict[str, str] = {}
    for event in lifecycle:
        if event.type not in tool_round_recovery._TOOL_ROUND_TERMINAL_EVENT_TYPES:
            continue
        tool_call_id = event.payload.get("tool_call_id")
        stage = stored.get(tool_call_id) if type(tool_call_id) is str else None
        if stage is None:
            continue
        if stage.event.id != event.id:
            raise resume_ledger.ToolCallEvidenceConflict(
                "Durable and staged terminal evidence conflict for one tool call."
            )
        durable[stage.tool_call_id] = event.id
    unpublished = [
        stage
        for stage in staged_terminal_reader.staged_terminal_records(pending_round)
        if stage.tool_call_id not in durable
    ]
    return unpublished, durable


def _published_tool_calls(
    pending_round: pending_rounds.PendingToolRound, event_ids: dict[str, str]
) -> tuple[dict[str, str], ...]:
    """The round's staged results whose terminal events are durable, in call order.

    Whether this abandonment published one or the lost run appended it just before
    it died, the stage holds the event as published, so it says which it was.
    """

    stages = {item.tool_call_id: item for item in pending_round.staged_terminals}
    return tuple(
        {
            "tool_round_id": pending_round.tool_round_id,
            "tool_call_id": call.tool_call_id,
            "tool_name": call.tool_name,
            "event_id": event_ids[call.tool_call_id],
            "result": "withheld" if _withheld(stages[call.tool_call_id].event) else "recorded",
        }
        for call in pending_round.tool_calls
        if call.tool_call_id in event_ids and call.tool_call_id in stages
    )


_WITHHELD_REASONS = frozenset(
    {"abandoned_before_tool_result_hooks", "recovery_hook_secret_scope_unavailable"}
)


def _withheld(event: Event) -> bool:
    """Whether a terminal replaces its call's result with an unavailable one.

    Only recovery writes those, and marks the event ``recovered``. A tool's own
    structured result can hold any JSON, including a ``reason`` of another type.
    """

    if event.payload.get("recovered") is not True:
        return False
    if event.payload.get("secret_scope_incomplete") is True:
        return True
    result = event.payload.get("result")
    structured = result.get("structured") if type(result) is dict else None
    reason = structured.get("reason") if type(structured) is dict else None
    return type(reason) is str and reason in _WITHHELD_REASONS


async def _current_round(
    session_store: SessionStore,
    session: Session,
    identity: ToolRoundIdentity,
    redactor: SecretRedactor,
) -> pending_rounds.PendingToolRound:
    _checkpoint, current = await pending_round_reader.load_pending_tool_round(
        session_store,
        session.id,
        redactor=redactor,
        consume_on_rejection=True,
        runtime_session=session,
    )
    if current is None or pending_rounds.pending_tool_round_identity(current) != identity:
        raise RuntimeError("Pending tool round changed before its staged results were published.")
    return current
