"""Tests for ``AlwaysRequireApprovalToolPolicy``."""

from __future__ import annotations

import asyncio

from cayu import (
    AgentSpec,
    AlwaysRequireApprovalToolPolicy,
    ToolPolicyDecision,
    ToolPolicyRequest,
)
from cayu.runtime import Session, SessionStatus


def _request(tool_name: str) -> ToolPolicyRequest:
    return ToolPolicyRequest(
        session=Session(
            id="s",
            agent_name="assistant",
            provider_name="fake",
            model="fake-model",
            status=SessionStatus.RUNNING,
        ),
        agent=AgentSpec(name="assistant", model="fake-model"),
        tool_name=tool_name,
        tool_call_id="call_1",
    )


def test_requires_approval_for_named_tool_and_allows_others() -> None:
    policy = AlwaysRequireApprovalToolPolicy(tools=["post_pr_comment"])
    gated = asyncio.run(policy.authorize(_request("post_pr_comment")))
    other = asyncio.run(policy.authorize(_request("read_file")))
    assert gated.decision is ToolPolicyDecision.REQUIRE_APPROVAL
    assert other.decision is ToolPolicyDecision.ALLOW


def test_requires_approval_for_every_tool_when_unscoped() -> None:
    policy = AlwaysRequireApprovalToolPolicy()
    result = asyncio.run(policy.authorize(_request("anything")))
    assert result.decision is ToolPolicyDecision.REQUIRE_APPROVAL


def test_expires_in_seconds_flows_into_policy_result() -> None:
    policy = AlwaysRequireApprovalToolPolicy(expires_in_seconds=90)
    result = asyncio.run(policy.authorize(_request("anything")))
    assert result.decision is ToolPolicyDecision.REQUIRE_APPROVAL
    assert result.approval_expires_in_seconds == 90.0

    unbounded = asyncio.run(AlwaysRequireApprovalToolPolicy().authorize(_request("anything")))
    assert unbounded.approval_expires_in_seconds is None


def test_expires_in_seconds_rejects_non_positive_and_non_finite_values() -> None:
    import pytest
    from pydantic import ValidationError

    for invalid in (0, -5, float("inf"), float("nan"), True):
        with pytest.raises(ValidationError):
            AlwaysRequireApprovalToolPolicy(expires_in_seconds=invalid)
