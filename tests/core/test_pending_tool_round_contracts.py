"""Saved round contracts remain usable without runtime recovery owners."""

import os
import pickle
import subprocess
import sys
from pathlib import Path
from typing import get_args, get_type_hints

import cayu
from cayu.sessions import _pending_tool_round as pending_rounds


def _record():
    return pending_rounds.PendingToolRound(
        model_step_id="mstep_" + "1" * 32,
        model_attempt_id="matt_" + "2" * 32,
        tool_round_id="tround_" + "3" * 32,
        agent_name="worker",
        tool_calls=[
            {
                "tool_call_id": "call-1",
                "tool_name": "echo",
                "arguments": {"nested": ["before"]},
            }
        ],
    )


def test_pending_round_contract_composes_without_execution_or_store_owners():
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            """
import sys

blocked = (
    "cayu.runtime._tool_round_recovery", "cayu.runtime._approval_support",
    "cayu.runtime._runtime_records", "cayu.runtime._session_engine",
    "cayu.runtime._recovery_coordinator", "cayu.sessions.base", "cayu.storage",
)
class BlockOwners:
    def find_spec(self, fullname, path=None, target=None):
        if any(fullname == name or fullname.startswith(name + ".") for name in blocked):
            raise AssertionError(fullname)
sys.meta_path.insert(0, BlockOwners())

from tests.core.test_pending_tool_round_contracts import _record, pending_rounds
record = _record()
saved = record.model_dump_json()
loaded = pending_rounds.PendingToolRound.model_validate_json(saved)
assert loaded == record and loaded is not record
identity = pending_rounds.pending_tool_round_identity(loaded)
assert identity.tool_round_id == record.tool_round_id
loaded.tool_calls[0].arguments["nested"].append("changed")
assert record.tool_calls[0].arguments["nested"] == ["before"]
assert pending_rounds.PENDING_TOOL_ROUND_CHECKPOINT_KEY == "pending_tool_round"
assert not any(name in sys.modules for name in blocked)
""",
        ],
        env={**os.environ, "PYTHONPATH": str(Path(cayu.__file__).resolve().parent.parent)},
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr


def test_pending_round_contract_and_runtime_parser_share_one_identity():
    from cayu.runtime import _tool_round_recovery as recovery
    from cayu.sessions import _pending_tool_round_reader as pending_round_reader

    assert recovery.pending_rounds is pending_rounds
    for name in (
        "PendingToolRound",
        "pending_tool_round_identity",
        "PENDING_TOOL_ROUND_CHECKPOINT_KEY",
        "_OWNED_ROUND_JSON_CONTEXT",
    ):
        assert not hasattr(recovery, name)
    for value in (pending_rounds.PendingToolRound, pending_rounds.pending_tool_round_identity):
        assert value.__module__ == pending_rounds.__name__
        get_type_hints(value)
    assert pending_rounds.PendingToolRound in get_args(
        get_type_hints(pending_round_reader.pending_tool_round_from_checkpoint)["return"]
    )
    record = _record()
    restored = pickle.loads(pickle.dumps(record))
    assert type(restored) is pending_rounds.PendingToolRound
    assert restored == record
    parsed = pending_round_reader.pending_tool_round_from_checkpoint(
        {pending_rounds.PENDING_TOOL_ROUND_CHECKPOINT_KEY: record.model_dump(mode="json")}
    )
    assert type(parsed) is pending_rounds.PendingToolRound
    assert parsed == record
