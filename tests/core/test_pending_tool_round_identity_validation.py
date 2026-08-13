from __future__ import annotations

import pytest
from pydantic import ValidationError

from cayu.runtime import PendingToolApproval
from cayu.runtime import _approval_support as approval_support
from cayu.runtime._tool_round_recovery import PendingToolRound
from cayu.runtime.approvals import PendingToolCallApproval
from cayu.runtime.tool_policy import ToolPolicyDecision, ToolPolicyResult
from cayu.runtime.user_input import PendingUserInput


def _identity() -> dict[str, str]:
    return {
        "model_step_id": f"mstep_{'1' * 32}",
        "model_attempt_id": f"matt_{'2' * 32}",
        "tool_round_id": f"tround_{'3' * 32}",
    }


def _call(
    tool_call_id: str = "call_1",
    *,
    tool_name: str = "deploy",
) -> PendingToolCallApproval:
    return PendingToolCallApproval(
        tool_call_id=tool_call_id,
        tool_name=tool_name,
        arguments={"target": "production"},
    )


@pytest.mark.parametrize(
    "build",
    (
        lambda calls: PendingToolApproval(
            approval_id="approval_1",
            **_identity(),
            tool_call_id="call_1",
            tool_name="deploy",
            arguments={"target": "production"},
            agent_name="assistant",
            publish_arguments=True,
            tool_calls=calls,
        ),
        lambda calls: PendingUserInput(
            input_id="input_1",
            **_identity(),
            tool_call_id="call_1",
            tool_name="deploy",
            question="Continue?",
            arguments={"target": "production"},
            agent_name="assistant",
            tool_calls=calls,
        ),
        lambda calls: PendingToolRound(
            **_identity(),
            agent_name="assistant",
            tool_calls=calls,
        ),
    ),
)
def test_pending_tool_round_state_rejects_duplicate_call_identity(build) -> None:
    with pytest.raises(ValidationError, match="duplicate tool-call identities"):
        build([_call(), _call()])


@pytest.mark.parametrize(
    "build",
    (
        lambda: PendingToolApproval(
            approval_id="approval_1",
            **_identity(),
            tool_call_id="call_missing",
            tool_name="deploy",
            arguments={"target": "production"},
            agent_name="assistant",
            publish_arguments=True,
            tool_calls=[_call()],
        ),
        lambda: PendingUserInput(
            input_id="input_1",
            **_identity(),
            tool_call_id="call_missing",
            tool_name="deploy",
            question="Continue?",
            arguments={"target": "production"},
            agent_name="assistant",
            tool_calls=[_call()],
        ),
    ),
)
def test_pending_pause_state_requires_its_gating_call_in_the_round(build) -> None:
    with pytest.raises(ValidationError, match="exactly one call"):
        build()


@pytest.mark.parametrize(
    "build",
    (
        lambda: PendingToolApproval(
            approval_id="approval_1",
            **_identity(),
            tool_call_id="call_1",
            tool_name="rollback",
            arguments={"target": "production"},
            agent_name="assistant",
            publish_arguments=True,
            tool_calls=[_call()],
        ),
        lambda: PendingUserInput(
            input_id="input_1",
            **_identity(),
            tool_call_id="call_1",
            tool_name="rollback",
            question="Continue?",
            arguments={"target": "production"},
            agent_name="assistant",
            tool_calls=[_call()],
        ),
    ),
)
def test_pending_pause_state_rejects_conflicting_gating_call_details(build) -> None:
    with pytest.raises(ValidationError, match="details do not match"):
        build()


def test_pending_approval_scope_requires_matching_paired_round_evidence() -> None:
    pending_round = PendingToolRound(
        **_identity(),
        agent_name="assistant",
        tool_calls=[_call()],
    )
    approval = PendingToolApproval(
        approval_id="approval_1",
        **_identity(),
        tool_call_id="call_1",
        tool_name="deploy",
        arguments={"target": "production"},
        agent_name="assistant",
        publish_arguments=True,
        secret_resolution_scope="static",
        tool_calls=[_call()],
    )

    assert not approval_support.pending_approval_scope_matches_round(
        approval,
        pending_round,
    )
    assert approval_support.pending_approval_scope_matches_round(
        approval.model_copy(update={"secret_resolution_scope": "unknown"}),
        pending_round,
    )


def test_approval_closure_policy_output_requires_static_scope() -> None:
    approval = PendingToolApproval(
        approval_id="approval_1",
        **_identity(),
        tool_call_id="call_1",
        tool_name="deploy",
        arguments={"target": "production"},
        agent_name="assistant",
        publish_arguments=True,
        secret_resolution_scope="dynamic",
        tool_calls=[_call()],
    )
    denial = ToolPolicyResult(
        decision=ToolPolicyDecision.DENY,
        reason="private policy reason",
        metadata={"private": "policy metadata"},
    )

    assert approval_support.public_policy_denial_result(
        secret_resolution_scope=approval.secret_resolution_scope,
        policy_result=denial,
    ) == ToolPolicyResult(decision=ToolPolicyDecision.DENY)
    assert (
        approval_support.public_policy_denial_result(
            secret_resolution_scope="static",
            policy_result=denial,
        )
        == denial
    )
