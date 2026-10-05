from __future__ import annotations

import pytest
from tests.core.test_tool_round_publication import _lifecycle_events, _quarantined_pending_round

from cayu.sessions._assistant_tool_round_publication import StagedToolCallTerminal
from cayu.sessions._pending_tool_round import PendingToolRound
from cayu.sessions.base import EventRecord
from cayu.sessions.pending_actions import (
    _pending_tool_round_evidence,
    project_pending_action_event_record,
)


@pytest.mark.parametrize(
    "publication",
    [
        "absent",
        "complete",
        "other-round",
        "malformed",
        "duplicate-terminal",
        "conflicting-start",
        "conflicting-scope",
        "conflicting-tool",
        "unknown-call",
    ],
)
def test_staged_result_closes_only_the_unpublished_terminal_gap(publication):
    pending = _quarantined_pending_round()
    _, terminal, started = _lifecycle_events(pending)
    pending.staged_terminals = [StagedToolCallTerminal(tool_call_id="call-a", event=terminal)]
    pending = PendingToolRound.model_validate(pending.model_dump(mode="json"))
    events = [started]
    if publication == "complete":
        events.append(terminal)
    elif publication == "malformed":
        events.append(terminal.model_copy(update={"payload": {**terminal.payload, "result": None}}))
    elif publication == "conflicting-start":
        events.append(started.model_copy(update={"id": "duplicate-start"}))
    elif publication == "duplicate-terminal":
        events.extend([terminal, terminal.model_copy(update={"id": "duplicate-terminal"})])
    elif publication == "other-round":
        events.append(
            terminal.model_copy(
                update={
                    "payload": {
                        **terminal.payload,
                        "tool_round_id": "other-round",
                        "model_step_id": "other-step",
                        "model_attempt_id": "other-attempt",
                    }
                }
            )
        )
    elif publication == "conflicting-scope":
        events.append(
            terminal.model_copy(
                update={"payload": {**terminal.payload, "model_attempt_id": "other-attempt"}}
            )
        )
    elif publication == "conflicting-tool":
        events.append(terminal.model_copy(update={"tool_name": "other-tool"}))
    elif publication == "unknown-call":
        events.append(
            terminal.model_copy(update={"payload": {**terminal.payload, "tool_call_id": "unknown"}})
        )
    records = [
        project_pending_action_event_record(EventRecord(sequence=index, event=event))
        for index, event in enumerate(events, 1)
    ]

    evidence = _pending_tool_round_evidence(list(reversed(records)), pending)

    assert evidence.scope_conflicting is (publication == "unknown-call")
    if publication in {"absent", "complete", "other-round", "unknown-call"}:
        assert evidence.terminal_ids == {"call-a"}
        assert evidence.started_without_terminal_ids == set()
        assert evidence.conflicting_ids == set()
    else:
        assert evidence.terminal_ids == set()
        assert evidence.started_without_terminal_ids == {"call-a"}
        assert evidence.conflicting_ids == {"call-a"}
