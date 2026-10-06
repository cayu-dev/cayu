"""Detached saved tool-terminal evidence shared by inspection and recovery."""

from __future__ import annotations

from typing import Any

from cayu._validation import copy_durable_json_value
from cayu.approvals.tools import ToolPolicyEvidence
from cayu.approvals.user_input import PendingUserInput
from cayu.events import Event, EventType, copy_event
from cayu.runtime.execution_units import ToolRoundIdentity, copy_tool_round_identity
from cayu.sessions import _pending_tool_round as pending_rounds
from cayu.sessions import _pending_tool_round_reader as pending_round_reader
from cayu.sessions._assistant_tool_round_publication import (
    StagedToolCallTerminal,
    validate_staged_tool_exposure_terminal,
)
from cayu.tools import _shared_artifact_result_schema as shared_artifact_result_schema
from cayu.tools import _terminal_controls as tool_terminal_controls
from cayu.tools import _web_access_result_schema as web_access_result_schema
from cayu.tools.base import ToolResult

_NONEXECUTED_TERMINAL_EVENT_TYPES = frozenset(
    {EventType.TOOL_CALL_BLOCKED, EventType.TOOL_CALL_APPROVAL_DENIED}
)


def staged_terminal_events(pending_round: pending_rounds.PendingToolRound) -> list[Event]:
    """Return detached private terminal projections in pending-call order."""

    staged_by_id = {
        item.tool_call_id: item.event for item in staged_terminal_records(pending_round)
    }
    return [
        copy_event(staged_by_id[call.tool_call_id])
        for call in pending_round.tool_calls
        if call.tool_call_id in staged_by_id
    ]


def staged_terminal_records(
    pending_round: pending_rounds.PendingToolRound,
) -> list[StagedToolCallTerminal]:
    """Return recovery-safe staged records in pending-call order."""

    staged_by_id = {
        item.tool_call_id: item for item in _recovery_safe_staged_terminals(pending_round)
    }
    return [
        StagedToolCallTerminal.model_validate(
            staged_by_id[call.tool_call_id].model_dump(mode="json")
        )
        for call in pending_round.tool_calls
        if call.tool_call_id in staged_by_id
    ]


def checkpoint_staged_terminals(
    checkpoint: dict[str, Any] | None,
    *,
    tool_round_identity: ToolRoundIdentity,
) -> list[StagedToolCallTerminal]:
    """Read stages from an ordinary or user-input-owned tool round."""

    _owner_key, owner = _checkpoint_staged_terminal_owner(
        checkpoint,
        tool_round_identity=tool_round_identity,
    )
    return [
        StagedToolCallTerminal.model_validate(item.model_dump(mode="json"))
        for item in owner.staged_terminals
    ]


def _checkpoint_staged_terminal_owner(
    checkpoint: dict[str, Any] | None,
    *,
    tool_round_identity: ToolRoundIdentity,
) -> tuple[str, pending_rounds.PendingToolRound | PendingUserInput]:
    copied = {} if checkpoint is None else copy_durable_json_value(checkpoint, "checkpoint")
    if type(copied) is not dict:
        raise AssertionError("Checkpoint copied as a non-object.")
    return _staged_terminal_owner_from_owned_checkpoint(
        copied, tool_round_identity=tool_round_identity
    )


def _staged_terminal_owner_from_owned_checkpoint(
    copied: dict[str, Any],
    *,
    tool_round_identity: ToolRoundIdentity,
) -> tuple[str, pending_rounds.PendingToolRound | PendingUserInput]:
    """Use only an immediately validated, detached document; never cache it."""
    identity = copy_tool_round_identity(tool_round_identity)
    pending_round = pending_round_reader._pending_tool_round_from_owned_checkpoint(copied, copied)
    from cayu.approvals.user_input import (
        PENDING_USER_INPUT_CHECKPOINT_KEY,
        _user_input_lifecycle_authority_from_owned_checkpoint,
    )

    pending_input, _ = _user_input_lifecycle_authority_from_owned_checkpoint(copied, copied)
    if pending_round is not None and pending_input is not None:
        raise RuntimeError("Checkpoint has multiple staged-terminal owners.")
    if pending_round is not None:
        if pending_rounds.pending_tool_round_identity(pending_round) != identity:
            raise RuntimeError("Staged terminals target a different pending tool round.")
        return pending_rounds.PENDING_TOOL_ROUND_CHECKPOINT_KEY, pending_round
    if pending_input is not None:
        input_identity = ToolRoundIdentity(
            tool_round_id=pending_input.tool_round_id,
            model_step_id=pending_input.model_step_id,
            model_attempt_id=pending_input.model_attempt_id,
        )
        if input_identity != identity:
            raise RuntimeError("Staged terminals target a different pending user-input round.")
        return PENDING_USER_INPUT_CHECKPOINT_KEY, pending_input
    raise RuntimeError("Staged terminals have no pending tool-round owner.")


def _recovery_safe_staged_terminals(
    pending_round: pending_rounds.PendingToolRound,
) -> list[StagedToolCallTerminal]:
    publication = pending_round.assistant_publication
    expected_ids = {call.tool_call_id for call in pending_round.tool_calls}
    covered_ids = set() if publication is None else set(publication.covered_tool_call_ids)
    scope = "unknown" if publication is None else publication.secret_resolution_scope
    if scope == "static" or covered_ids == expected_ids:
        return [
            StagedToolCallTerminal.model_validate(item.model_dump(mode="json"))
            for item in pending_round.staged_terminals
        ]
    safe: list[StagedToolCallTerminal] = []
    calls_by_id = {call.tool_call_id: call for call in pending_round.tool_calls}
    for item in pending_round.staged_terminals:
        call = calls_by_id[item.tool_call_id]
        if call.policy_evidence is ToolPolicyEvidence.UNEXPOSED:
            validate_staged_tool_exposure_terminal(
                item,
                policy_evidence=call.policy_evidence,
                tool_exposure=pending_round.tool_exposure,
            )
            safe.append(StagedToolCallTerminal.model_validate(item.model_dump(mode="json")))
            continue
        terminal_controls = tool_terminal_controls.runtime_terminal_controls(item.event.payload)
        never_executed = item.event.type in _NONEXECUTED_TERMINAL_EVENT_TYPES
        fixed_result = ToolResult(
            content="Tool result unavailable because invocation secret scope was incomplete.",
            structured={
                **({} if never_executed else {"error": "invalid_tool_output"}),
                "outcome_unknown": True,
                **terminal_controls,
                **({"executed": False, "outcome_unknown": False} if never_executed else {}),
            },
            is_error=True,
        )
        payload = copy_durable_json_value(item.event.payload, "staged_terminal.payload")
        if type(payload) is not dict:
            raise AssertionError("Staged terminal payload copied as a non-object.")
        payload.pop(web_access_result_schema.WEB_ACCESS_RESULT_AUTHORITY_FIELD, None)
        payload.pop(shared_artifact_result_schema.SHARED_ARTIFACT_RESULT_AUTHORITY_FIELD, None)
        payload["result"] = fixed_result.model_dump(mode="json")
        payload["recovered"] = True
        payload["secret_scope_incomplete"] = True
        event = item.event.model_copy(
            update={
                "type": item.event.type if never_executed else EventType.TOOL_CALL_FAILED,
                "payload": payload,
            },
            deep=True,
        )
        safe.append(
            item.model_copy(
                update={
                    "event": event,
                    "hooks_state": "finalized",
                },
                deep=True,
            )
        )
    return safe
