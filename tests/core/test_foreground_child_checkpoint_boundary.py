"""Saved child state composes without its runtime wait and effect owners."""

import os
import subprocess
import sys
from pathlib import Path
from typing import get_type_hints

import cayu
from cayu.sessions import _foreground_child_checkpoint as checkpoint
from cayu.sessions import _pending_approval_reader as approvals
from cayu.sessions import _tool_effect_intent as effects


def test_foreground_checkpoint_reads_and_projects_without_runtime_owners():
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            """
import sys

blocked = (
    "cayu.runtime._foreground_child_wait",
    "cayu.runtime._foreground_gate_continuation",
    "cayu.runtime._tool_effect_state",
    "cayu.runtime._approval_support",
    "cayu.runtime._session_engine",
    "cayu.runtime._recovery_coordinator",
    "cayu.runtime._tool_round_executor",
)
class BlockOwners:
    def find_spec(self, fullname, path=None, target=None):
        if any(fullname == name or fullname.startswith(name + ".") for name in blocked):
            raise AssertionError(fullname)
sys.meta_path.insert(0, BlockOwners())

from cayu.approvals.tools import PendingToolCallApproval, ToolApprovalDecision
from cayu.budgets.run_limits import RunLimits
from cayu.sessions import _foreground_child_checkpoint as state
from cayu.sessions import _pending_approval_reader as approvals
from cayu.sessions._pending_tool_round import PendingToolRound
from cayu.sessions._tool_effect_intent import ToolEffectIntent

identity = dict(model_step_id="mstep_"+"1"*32, model_attempt_id="matt_"+"2"*32,
                tool_round_id="tround_"+"3"*32)
effect = ToolEffectIntent(
    **identity, session_id="parent", session_instance_id="parent-instance",
    source_run_epoch=3, interaction_id="parent-interaction", tool_call_id="call-1",
    agent_name="parent-agent", tool_name="child", idempotency_key="call-key",
    execution_profile_fingerprint="a"*64, schema_digest="b"*64, arguments_digest="c"*64,
)
wait = state.ForegroundChildWait(
    parent_effect=effect, child_session_id="child", child_session_instance_id="child-instance",
    child_interaction_id="child-interaction", child_spawn_fingerprint="sha256:"+"d"*64,
    child_action_kind="tool_approval", child_action_id="child-approval",
    child_action_run_epoch=4, revision=1,
)
terminal = state.ForegroundChildTerminal(
    wait=wait, child_released_run_epoch=5, event_id="child-completed",
    event_type="session.completed", event_digest="e"*64,
)
call = PendingToolCallApproval(tool_call_id="call-1", tool_name="child", arguments={})
round = PendingToolRound(**identity, agent_name="parent-agent", tool_calls=[call],
    policy_state="planned", policy_context_version=1, max_steps=4, limits=RunLimits(),
    budget_limits=(), model_step=1, source_model_step_id=identity["model_step_id"],
    source_transcript_cursor=0)
source = {
    state.FOREGROUND_CHILD_WAIT_KEY: wait.model_dump(mode="json"),
    state.FOREGROUND_CHILD_TERMINAL_KEY: terminal.model_dump(mode="json"),
    "pending_tool_round": round.model_dump(mode="json"),
}
assert state.foreground_child_state_from_checkpoint(source) == (wait, terminal)
assert state.post_action_continuation_round_from_checkpoint(source) == round
projection = state.gate_close_continuation(
    source, pending=round, publication_id="tool-round:"+round.tool_round_id,
    metadata={"nested": ["original"]},
)
parent = state.ForegroundParentContinuation.model_validate(projection)
assert parent.terminal == terminal and parent.request.session_id == "parent"
assert parent.request.messages == [] and parent.completed_model_step == 1
assert parent.request.max_steps == 4 and parent.request.limits == round.limits
assert type(parent.request) is state.ForegroundChildResumeRequest

first, _ = state.foreground_child_state_from_checkpoint(source)
source[state.FOREGROUND_CHILD_WAIT_KEY]["revision"] = 2
try:
    state.foreground_child_state_from_checkpoint(source)
except RuntimeError as error:
    assert "exact retained wait" in str(error)
else:
    raise AssertionError("Conflicting child evidence was accepted")
assert first.revision == 1

intent = approvals.ApprovalResolutionIntent(
    **identity, approval_id="approval-1", tool_call_id="call-1",
    decision=ToolApprovalDecision.APPROVE, resolution_request_digest="f"*64,
)
saved = {approvals.APPROVAL_RESOLUTION_INTENT_CHECKPOINT_KEY: intent.model_dump(mode="json")}
assert approvals.approval_resolution_intent_from_checkpoint(saved) == intent
saved[approvals.APPROVAL_RESOLUTION_INTENT_CHECKPOINT_KEY]["tool_call_id"] = "call-2"
assert approvals.approval_resolution_intent_from_checkpoint(saved).tool_call_id == "call-2"
assert intent.tool_call_id == "call-1"
assert not any(name in sys.modules for name in blocked)
""",
        ],
        env={**os.environ, "PYTHONPATH": str(Path(cayu.__file__).resolve().parent.parent)},
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr


def test_foreground_checkpoint_models_have_canonical_identity_and_annotations():
    import pickle

    from cayu.runtime import _approval_support as runtime_approvals
    from cayu.runtime import _foreground_child_wait as runtime_wait
    from cayu.runtime import _foreground_gate_continuation as runtime_gate
    from cayu.runtime import _tool_effect_state as runtime_effects

    assert runtime_wait.foreground_checkpoint is checkpoint
    assert runtime_gate.ForegroundChildWait is checkpoint.ForegroundChildWait
    assert (
        runtime_gate.foreground_child_state_from_checkpoint
        is checkpoint.foreground_child_state_from_checkpoint
    )
    assert runtime_approvals.pending_approval_reader is approvals
    assert runtime_effects.tool_effect_intent is effects
    assert (
        checkpoint.ForegroundChildWait.model_fields["parent_effect"].annotation
        is effects.ToolEffectIntent
    )
    assert (
        runtime_effects.ToolEffectRecord.model_fields["intent"].annotation
        is effects.ToolEffectIntent
    )
    for owner, names in (
        (
            checkpoint,
            (
                "ForegroundChildResumeRequest",
                "ForegroundChildWait",
                "ForegroundChildTerminal",
                "ForegroundParentContinuation",
                "ForegroundChildPostActionContinuation",
                "post_action_continuation_round_from_checkpoint",
                "post_action_continuation_for_close",
                "post_action_continuation_from_checkpoint",
                "foreground_child_state_from_checkpoint",
                "gate_close_continuation",
            ),
        ),
        (approvals, ("ApprovalResolutionIntent", "approval_resolution_intent_from_checkpoint")),
        (effects, ("ToolEffectIntent",)),
    ):
        for name in names:
            value = getattr(owner, name)
            assert value.__module__ == owner.__name__
            assert pickle.loads(pickle.dumps(value)) is value
            get_type_hints(value)
    for name in (
        "ForegroundChildResumeRequest",
        "ForegroundChildWait",
        "ForegroundChildTerminal",
        "ForegroundParentContinuation",
        "ForegroundChildPostActionContinuation",
        "FOREGROUND_CHILD_WAIT_KEY",
        "FOREGROUND_CHILD_TERMINAL_KEY",
        "FOREGROUND_PARENT_CONTINUATION_KEY",
        "FOREGROUND_CHILD_POST_ACTION_CONTINUATION_KEY",
        "post_action_continuation_round_from_checkpoint",
        "post_action_continuation_for_close",
        "post_action_continuation_from_checkpoint",
        "foreground_child_state_from_checkpoint",
    ):
        assert not hasattr(runtime_wait, name)
    assert not hasattr(runtime_gate, "gate_close_continuation")
    assert not hasattr(runtime_effects, "ToolEffectIntent")
    for name in (
        "ApprovalResolutionIntent",
        "approval_resolution_intent_from_checkpoint",
        "APPROVAL_RESOLUTION_INTENT_CHECKPOINT_KEY",
    ):
        assert not hasattr(runtime_approvals, name)
