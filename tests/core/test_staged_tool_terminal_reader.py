"""Saved terminal evidence is usable independently of runtime execution owners."""

import os
import pickle
import subprocess
import sys
from pathlib import Path
from typing import get_type_hints

import pytest
from tests.core.test_tool_round_publication import _lifecycle_events, _quarantined_pending_round

import cayu
from cayu.approvals.user_input import PendingUserInput
from cayu.runtime.execution_units import ToolRoundIdentity
from cayu.sessions import _staged_tool_terminal_reader as reader
from cayu.sessions._assistant_tool_round_publication import StagedToolCallTerminal
from cayu.sessions._pending_tool_round import PendingToolRound, pending_tool_round_identity
from cayu.tools import _terminal_controls as controls


def test_saved_terminal_reader_works_without_runtime_owners():
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            """
import sys

blocked = (
    "cayu.runtime._tool_round_recovery",
    "cayu.runtime._tool_results",
    "cayu.runtime._resume_ledger",
    "cayu.runtime._tool_round_staging",
    "cayu.runtime._tool_round_continuation",
    "cayu.runtime._durable_tool_round",
    "cayu.runtime._session_engine",
    "cayu.runtime._recovery_coordinator",
    "cayu.runtime._tool_round_executor",
)
class BlockOwners:
    def find_spec(self, fullname, path=None, target=None):
        if any(fullname == name or fullname.startswith(name + ".") for name in blocked):
            raise AssertionError(fullname)
sys.meta_path.insert(0, BlockOwners())

from cayu.approvals.tools import PendingToolCallApproval
from cayu.events import Event, EventType
from cayu.runtime.execution_units import ToolRoundIdentity
from cayu.sessions import _staged_tool_terminal_reader as reader
from cayu.sessions._assistant_tool_round_publication import StagedToolCallTerminal
from cayu.sessions._pending_tool_round import PendingToolRound
from cayu.tools._terminal_controls import runtime_terminal_controls
from cayu.tools.base import ToolResult

identity = ToolRoundIdentity(model_step_id="mstep_"+"1"*32,
    model_attempt_id="matt_"+"2"*32, tool_round_id="tround_"+"3"*32)
calls = [PendingToolCallApproval(tool_call_id=name, tool_name="probe", arguments={})
         for name in ("first", "second")]
stages = [StagedToolCallTerminal(tool_call_id=call.tool_call_id,
    event=Event(type=EventType.TOOL_CALL_COMPLETED, session_id="session",
        tool_name="probe", payload={**identity.payload(), "tool_call_id":call.tool_call_id,
            "result":ToolResult(content="private-result-canary").model_dump(mode="json")}))
    for call in reversed(calls)]
pending = PendingToolRound(**identity.payload(), agent_name="assistant",
    tool_calls=calls, staged_terminals=stages)
original = pending.model_dump(mode="json")
events = reader.staged_terminal_events(pending)
assert [event.payload["tool_call_id"] for event in events] == ["first", "second"]
assert all(event.type is EventType.TOOL_CALL_FAILED for event in events)
assert all(event.payload["secret_scope_incomplete"] for event in events)
assert all("private-result-canary" not in event.model_dump_json() for event in events)
events[0].payload["result"]["content"] = "changed"
assert pending.model_dump(mode="json") == original
checkpoint = {"pending_tool_round": original}
raw = reader.checkpoint_staged_terminals(checkpoint, tool_round_identity=identity)
assert [stage.tool_call_id for stage in raw] == ["second", "first"]
assert raw[0].event.payload["result"]["content"] == "private-result-canary"
raw[0].event.payload["result"]["content"] = "changed"
assert checkpoint["pending_tool_round"] == original
assert runtime_terminal_controls({"terminal_outcome":"tool_execution_error",
    "tool_effect":"external", "outcome_unknown":True,
    "manual_reconciliation_required":True})["manual_reconciliation_required"] is True
assert not any(name in sys.modules for name in blocked)
""",
        ],
        env={**os.environ, "PYTHONPATH": str(Path(cayu.__file__).resolve().parent.parent)},
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr


@pytest.mark.parametrize("owner_kind", ["pending_tool_round", "pending_user_input"])
def test_checkpoint_terminal_reads_are_detached_fresh_and_owner_scoped(owner_kind):
    pending = _quarantined_pending_round()
    failed, completed, _ = _lifecycle_events(pending)
    pending.staged_terminals = [
        StagedToolCallTerminal(tool_call_id="call-b", event=failed),
        StagedToolCallTerminal(tool_call_id="call-a", event=completed),
    ]
    pending = PendingToolRound.model_validate(pending.model_dump(mode="json"))
    identity = pending_tool_round_identity(pending)
    call = pending.tool_calls[0]
    pending_input = PendingUserInput(
        **identity.payload(),
        session_id="session-1",
        session_instance_id="instance-1",
        source_interaction_id="interaction-1",
        source_run_epoch=1,
        input_id="input-1",
        tool_call_id=call.tool_call_id,
        tool_name=call.tool_name,
        arguments=call.arguments,
        question="Continue?",
        agent_name=pending.agent_name,
        execution_profile_fingerprint="a" * 64,
        tool_calls=pending.tool_calls,
        assistant_message_state=pending.assistant_message_state,
        quarantined_assistant_message=pending.quarantined_assistant_message,
        assistant_publication=pending.assistant_publication,
        staged_terminals=pending.staged_terminals,
    )
    owners = {
        "pending_tool_round": pending.model_dump(mode="json"),
        "pending_user_input": pending_input.model_dump(mode="json"),
    }
    checkpoint = {owner_kind: owners[owner_kind]}
    first = reader.checkpoint_staged_terminals(checkpoint, tool_round_identity=identity)
    assert [stage.tool_call_id for stage in first] == ["call-b", "call-a"]
    first[0].event.payload["result"]["content"] = "changed reader result"
    assert (
        checkpoint[owner_kind]["staged_terminals"][0]["event"]["payload"]["result"]["content"]
        == "write failed"
    )
    checkpoint[owner_kind]["staged_terminals"][0]["event"]["payload"]["result"]["content"] = (
        "updated source"
    )
    second = reader.checkpoint_staged_terminals(checkpoint, tool_round_identity=identity)
    assert second[0].event.payload["result"]["content"] == "updated source"
    assert first[0].event.payload["result"]["content"] == "changed reader result"

    foreign_identity = ToolRoundIdentity(
        **{**identity.payload(), "tool_round_id": "tround_" + "9" * 32}
    )
    with pytest.raises(RuntimeError, match="different pending"):
        reader.checkpoint_staged_terminals(checkpoint, tool_round_identity=foreign_identity)
    with pytest.raises(RuntimeError, match="multiple staged-terminal owners"):
        reader.checkpoint_staged_terminals(owners, tool_round_identity=identity)
    checkpoint.clear()
    with pytest.raises(RuntimeError, match="no pending tool-round owner"):
        reader.checkpoint_staged_terminals(checkpoint, tool_round_identity=identity)


def test_terminal_reader_and_control_helpers_have_one_canonical_owner():
    from cayu.runtime import _tool_results as runtime_results
    from cayu.runtime import _tool_round_recovery as runtime_recovery
    from cayu.sessions import pending_actions

    assert runtime_recovery.staged_terminal_reader is reader
    assert pending_actions.staged_terminal_reader is reader
    assert runtime_results.tool_terminal_controls is controls
    assert reader.StagedToolCallTerminal is StagedToolCallTerminal
    assert reader.PendingUserInput is PendingUserInput
    for owner, former_owner, names in (
        (
            reader,
            runtime_recovery,
            (
                "staged_terminal_events",
                "staged_terminal_records",
                "checkpoint_staged_terminals",
                "_checkpoint_staged_terminal_owner",
                "_staged_terminal_owner_from_owned_checkpoint",
                "_recovery_safe_staged_terminals",
            ),
        ),
        (controls, runtime_results, ("runtime_terminal_controls", "_redacted_failure_evidence")),
    ):
        for name in names:
            function = getattr(owner, name)
            assert function.__module__ == owner.__name__
            assert pickle.loads(pickle.dumps(function)) is function
            get_type_hints(function)
            assert not hasattr(former_owner, name)
