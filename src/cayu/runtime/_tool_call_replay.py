"""Replay of interrupted NONE/IDEMPOTENT tool calls after an abandoned execution.

Recovery cannot dispatch tools, so a takeover elects the replay and keeps the
round open; the continuation that follows dispatches each elected call once,
under its original identity and arguments, through ordinary tool admission.
One runtime-owned operation record per round moves ``elected`` -> ``dispatched``
before dispatch, so a crash during the replay never replays again.
"""

from __future__ import annotations

from collections.abc import Sequence
from hashlib import sha256
from typing import Any

from cayu.context.structured_output import STRUCTURED_OUTPUT_TOOL_NAME
from cayu.events import Event, EventType
from cayu.runtime import _runtime_records as runtime_records
from cayu.runtime._tool_identity import tool_idempotency_key
from cayu.sessions import _pending_approval_reader as pending_approval_reader
from cayu.sessions import _pending_tool_round as pending_rounds
from cayu.sessions._tool_call_replay_scope import STORAGE_KEY, owner_scope
from cayu.sessions.base import (
    Session,
    SessionOperationPublication,
    SessionRunFenced,
    SessionStore,
)
from cayu.tools._policy_evidence import ToolPolicyEvidence
from cayu.tools.base import ToolEffect
from cayu.tools.policy import ToolPolicyDecision

_REPLAYABLE_EFFECTS = frozenset({ToolEffect.NONE, ToolEffect.IDEMPOTENT})


class ToolCallReplayRequired(RuntimeError):
    """A takeover kept its round open for the continuation to replay."""


def _key(session_id: str, tool_round_id: str) -> str:
    digest = sha256(
        b"cayu-tool-call-replay-v1\0" + session_id.encode() + b"\0" + tool_round_id.encode()
    ).hexdigest()
    return STORAGE_KEY + digest


def replayable_calls(
    *,
    pending_round: pending_rounds.PendingToolRound,
    registered_agent: runtime_records.RegisteredAgentState,
    secret_resolution_scope: str,
    unfinished_call_ids: set[str],
    started_call_ids: set[str],
) -> tuple[str, ...]:
    """Return started unfinished calls when all unfinished calls qualify.

    Only plain static-scope application tools qualify: replay must reuse the
    round's sealed redactor, and tools that own child sessions, workspace
    mutation, effect journals or their own durable recovery keep those paths.
    """

    if (
        secret_resolution_scope != "static"
        or pending_round.environment_name is not None
        or pending_round.task_id is not None
        or pending_round.assistant_publication is None
        or pending_round.assistant_publication.state == "blocked"
        or not unfinished_call_ids & started_call_ids
    ):
        return ()
    selected: list[str] = []
    for call in pending_round.tool_calls:
        if call.tool_call_id not in unfinished_call_ids:
            continue
        registered = registered_agent.executable_tool(call.tool_name)
        if (
            registered is None
            or call.tool_name == STRUCTURED_OUTPUT_TOOL_NAME
            or registered.effect not in _REPLAYABLE_EFFECTS
            or registered.workspace_mutation
            or registered.execution_requirements
            or registered.auxiliary_inference is not None
            or registered.child_session_recovery is not None
            or registered.durable_tool_recovery is not None
            or registered.effect_reconciler is not None
            or call.targeted_tool_grant_id is not None
            or call.targeted_tool_invocation is not None
            or call.targeted_tool_rejection is not None
            or pending_approval_reader.effective_tool_policy_evidence(call)
            is not ToolPolicyEvidence.AUTHORITATIVE
            or call.policy_decision != ToolPolicyDecision.ALLOW.value
        ):
            return ()
        if call.tool_call_id in started_call_ids:
            selected.append(call.tool_call_id)
    return tuple(selected)


def started_dispatch_identities(
    events: Sequence[Event],
    *,
    session_id: str,
    tool_round_id: str,
    call_ids: tuple[str, ...],
) -> dict[str, tuple[str | None, str | None]] | None:
    """Return each call's original ``(approval_id, input_id)``, or None if unprovable.

    The replay must present the same idempotency key as the interrupted attempt, so
    it reuses the approval and pause identities recorded on that attempt's start
    event, and refuses when the recorded key can't be reproduced from them.
    """

    started: dict[str, Event] = {}
    for event in events:
        if event.type is EventType.TOOL_CALL_STARTED:
            call_id = event.payload.get("tool_call_id")
            if isinstance(call_id, str) and call_id in call_ids:
                started[call_id] = event
    identities: dict[str, tuple[str | None, str | None]] = {}
    for call_id in call_ids:
        event = started.get(call_id)
        if event is None:
            return None
        approval_id = event.payload.get("approval_id")
        input_id = event.payload.get("input_id")
        if not (approval_id is None or isinstance(approval_id, str)) or not (
            input_id is None or isinstance(input_id, str)
        ):
            return None
        expected = tool_idempotency_key(
            session_id=session_id,
            tool_call_id=call_id,
            tool_round_id=tool_round_id,
            approval_id=approval_id,
            pause_id=input_id,
        )
        if event.payload.get("idempotency_key") != expected:
            return None
        identities[call_id] = (approval_id, input_id)
    return identities


async def load_record(
    store: SessionStore, session: Session, tool_round_id: str
) -> tuple[str, tuple[str, ...]] | None:
    """Return the round's replay state and the call IDs it covers, if any."""

    with owner_scope():
        record = await store.load_session_operation(session.id, _key(session.id, tool_round_id))
    if record is None:
        return None
    state = record.get("state")
    call_ids = record.get("tool_call_ids")
    if (
        state not in {"elected", "dispatched"}
        or not isinstance(call_ids, list)
        or not all(isinstance(item, str) for item in call_ids)
    ):
        raise RuntimeError("Tool-call replay record is invalid.")
    return state, tuple(call_ids)


async def _transition(
    store: SessionStore,
    session: Session,
    *,
    tool_round_id: str,
    call_ids: tuple[str, ...],
    expected: str | None,
    desired: str,
) -> None:
    key = _key(session.id, tool_round_id)
    record: dict[str, Any] = {
        "schema_version": 1,
        "state": desired,
        "tool_round_id": tool_round_id,
        "tool_call_ids": list(call_ids),
    }

    def publish(
        current_session: Session,
        checkpoint: dict[str, Any] | None,
        current: dict[str, Any] | None,
    ) -> SessionOperationPublication:
        if (current_session.id, current_session.run_epoch) != (session.id, session.run_epoch):
            raise SessionRunFenced("Tool-call replay lost its run authority.")
        current_state = None if current is None else current.get("state")
        if current_state != expected or (
            current is not None and current.get("tool_call_ids") != list(call_ids)
        ):
            raise SessionRunFenced("Tool-call replay state changed before its transition.")
        return SessionOperationPublication(
            checkpoint={} if checkpoint is None else checkpoint,
            operation_records={key: record},
        )

    with owner_scope():
        await store.publish_session_operation(
            session.id,
            idempotency_key=key,
            operation_transform=publish,
            events=[],
            expected_statuses={session.status},
            expected_run_epoch=session.run_epoch,
        )


async def elect(
    store: SessionStore, session: Session, tool_round_id: str, call_ids: tuple[str, ...]
) -> None:
    await _transition(
        store,
        session,
        tool_round_id=tool_round_id,
        call_ids=call_ids,
        expected=None,
        desired="elected",
    )


async def begin_dispatch(
    store: SessionStore, session: Session, tool_round_id: str, call_ids: tuple[str, ...]
) -> None:
    await _transition(
        store,
        session,
        tool_round_id=tool_round_id,
        call_ids=call_ids,
        expected="elected",
        desired="dispatched",
    )
