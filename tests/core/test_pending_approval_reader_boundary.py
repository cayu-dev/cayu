"""Saved approval interpretation composes without runtime execution owners."""

import os
import subprocess
import sys
from pathlib import Path
from typing import get_type_hints

import cayu
from cayu.sessions import _pending_approval_reader as reader


def test_pending_approval_reader_composes_without_runtime_execution():
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            """
import sys

blocked = (
    "cayu.runtime._approval_support", "cayu.runtime._tool_round_recovery",
    "cayu.runtime._runtime_records", "cayu.runtime._session_engine",
    "cayu.runtime._recovery_coordinator", "cayu.runtime._tool_round_executor",
)
class BlockExecution:
    def find_spec(self, fullname, path=None, target=None):
        if any(fullname == name or fullname.startswith(name + ".") for name in blocked):
            raise AssertionError(fullname)
sys.meta_path.insert(0, BlockExecution())

from cayu.approvals.tools import PendingToolApproval, PendingToolCallApproval, ToolPolicyEvidence
from cayu.sessions import _pending_approval_reader as reader
from cayu.vaults.redaction import SecretRedactor

call = PendingToolCallApproval(
    tool_call_id="call-1", tool_name="echo", arguments={"text": "first"},
    policy_decision="require_approval",
)
approval = PendingToolApproval(
    approval_id="approval-1", model_step_id="mstep_" + "1" * 32,
    model_attempt_id="matt_" + "2" * 32, tool_round_id="tround_" + "3" * 32,
    tool_call_id=call.tool_call_id, tool_name=call.tool_name, arguments=call.arguments,
    tool_calls=[call], agent_name="worker", publish_arguments=True,
    reason="Review this call", secret_resolution_scope="unknown",
)
source = {reader.PENDING_TOOL_APPROVAL_CHECKPOINT_KEY: approval.model_dump(mode="json")}
first = reader.pending_approval_from_checkpoint(source)
assert first == approval
source["pending_tool_approval"]["arguments"]["text"] = "second"
source["pending_tool_approval"]["tool_calls"][0]["arguments"]["text"] = "second"
second = reader.pending_approval_from_checkpoint(source)
assert first.arguments["text"] == "first" and second.arguments["text"] == "second"

round = reader.planned_tool_round_from_pending_approval(first)
assert round.tool_round_id == first.tool_round_id and round.tool_calls == first.tool_calls
assert reader.tool_round_secret_resolution_scope(round) == "unknown"
assert reader.pending_approval_scope_matches_round(first, round)
assert reader.public_pending_approval_reason(first) is None
static = first.model_copy(update={"secret_resolution_scope": "static"})
assert not reader.pending_approval_scope_matches_round(static, round)
assert reader.public_pending_approval_reason(static) == "Review this call"
assert reader.public_pending_approval_reason(
    static.model_copy(update={"publish_arguments": False})
) is None
assert reader.effective_tool_policy_evidence(call) is ToolPolicyEvidence.AUTHORITATIVE
assert reader.effective_tool_policy_evidence(
    call.model_copy(update={"policy_evidence": None, "policy_decision": None})
) is ToolPolicyEvidence.UNREGISTERED
assert reader.effective_tool_policy_evidence(
    call.model_copy(update={"policy_evidence": ToolPolicyEvidence.AMBIGUOUS})
) is ToolPolicyEvidence.AMBIGUOUS

secret = "approval-reader-private-value"
source["pending_tool_approval"]["arguments"]["text"] = secret
source["pending_tool_approval"]["tool_calls"][0]["arguments"]["text"] = secret
for consume in (False, True):
    try:
        reader.pending_approval_from_checkpoint(
            source, redactor=SecretRedactor([secret]), consume_on_rejection=consume,
        )
    except ValueError as error:
        assert secret not in str(error)
    else:
        raise AssertionError("Secret-bearing approval was admitted")
    assert bool(source) is not consume
assert not any(name in sys.modules for name in blocked)
""",
        ],
        env={**os.environ, "PYTHONPATH": str(Path(cayu.__file__).resolve().parent.parent)},
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr


def test_pending_approval_reader_has_one_owner_and_resolvable_annotations():
    from cayu.approvals.tools import PendingToolApproval
    from cayu.runtime import _approval_support as runtime
    from cayu.sessions import _checkpoint_secret_validation as secrets
    from cayu.sessions import _pending_tool_round as rounds
    from cayu.sessions import pending_actions

    assert runtime.pending_approval_reader is reader
    assert pending_actions.pending_approval_reader is reader
    assert not hasattr(pending_actions, "approval_support")
    assert reader.PendingToolApproval is PendingToolApproval
    assert reader.pending_rounds is rounds
    assert reader.durable_value_contains_secret is secrets.durable_value_contains_secret
    assert not hasattr(runtime, "PENDING_TOOL_APPROVAL_CHECKPOINT_KEY")
    for name in (
        "pending_approval_from_checkpoint",
        "_pending_approval_from_owned_checkpoint",
        "tool_round_secret_resolution_scope",
        "pending_approval_scope_matches_round",
        "planned_tool_round_from_pending_approval",
        "public_pending_approval_reason",
        "effective_tool_policy_evidence",
    ):
        assert not hasattr(runtime, name)
        function = getattr(reader, name)
        assert function.__module__ == reader.__name__
        get_type_hints(function)
