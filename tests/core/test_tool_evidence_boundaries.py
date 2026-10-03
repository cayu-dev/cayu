"""Shared evidence readers remain usable without runtime execution or stores."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

import cayu


@pytest.mark.parametrize("component", ["arguments", "evidence"])
def test_tool_evidence_components_work_without_execution_or_stores(component):
    script = """
import importlib.abc
import pickle
import sys
from typing import get_type_hints

component = sys.argv[1]
blocked = {
    "cayu.storage", "cayu.sessions.base", "cayu.runtime._resume_ledger",
    "cayu.runtime._runtime_records", "cayu.runtime._tool_results",
    "cayu.runtime._session_engine", "cayu.runtime._recovery_coordinator",
    "cayu.runtime._tool_round_recovery", "cayu.runtime._approval_support",
    "cayu.runtime._event_projection",
}
if component == "arguments":
    blocked.add("cayu.runtime")

class RejectExecutionAndStores(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if any(fullname == name or fullname.startswith(name + ".") for name in blocked):
            raise AssertionError(f"Shared tool evidence imported {fullname}")

sys.meta_path.insert(0, RejectExecutionAndStores())
from cayu.tools import _argument_publication as arguments
from cayu.vaults import SecretRedactor

private = {"token": "private-tool-secret"}
projection = arguments.finalized_argument_projection(
    private, redactor=SecretRedactor().with_secret(private["token"]),
)
assert private == {"token": "private-tool-secret"}
assert private["token"] not in str(projection.payload_fields())
assert not arguments.argument_projection_is_exact(projection, private_arguments=private)
assert arguments.started_arguments_match_private_call(
    arguments.quarantined_argument_fields(), private_arguments=private,
)
assert get_type_hints(arguments.finalized_argument_projection)["return"] is type(projection)
values = [projection]

if component == "evidence":
    from cayu.approvals.tools import PendingToolCallApproval
    from cayu.events import Event, EventType
    from cayu.sessions import _tool_call_evidence as evidence

    call = PendingToolCallApproval(tool_call_id="call-1", tool_name="echo", arguments=private)
    started = Event(
        type=EventType.TOOL_CALL_STARTED, session_id="session-1", tool_name="echo",
        payload={"tool_call_id": call.tool_call_id, **arguments.quarantined_argument_fields()},
    )
    ledger = evidence.scan_projected_tool_call_evidence(
        events=iter([started]), pending_calls=iter([call]),
        in_scope=lambda event: event.session_id == "session-1",
        terminal_event_types=frozenset({EventType.TOOL_CALL_COMPLETED}),
        terminal_result_is_valid=lambda event: False,
    )
    assert ledger.started_without_terminal_ids == {call.tool_call_id}
    assert ledger.terminal_ids == ledger.conflicting_ids == set()
    assert not ledger.scope_conflicting
    assert get_type_hints(evidence.scan_projected_tool_call_evidence)["return"] is type(ledger)
    values.append(ledger)

for value in values:
    for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
        restored = pickle.loads(pickle.dumps(value, protocol=protocol))
        assert type(restored) is type(value)
        assert restored == value
"""
    result = subprocess.run(
        [sys.executable, "-c", script, component],
        env={**os.environ, "PYTHONPATH": str(Path(cayu.__file__).resolve().parent.parent)},
        text=True,
        capture_output=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
