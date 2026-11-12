"""An approval floor under another tool policy: what it decides, and what ``cayu check`` reads
from it, the connector toolkits' approval policies included."""

from __future__ import annotations

import asyncio
from typing import Any

import pytest
from tests.connector_broker_support import policy_request

from cayu import AgentSpec, CayuApp, Tool, ToolEffect, ToolSpec
from cayu.connections import LocalConnections
from cayu.connectors import slack, snowflake
from cayu.runtime import check_manifest
from cayu.runtime._tool_invocation.admission import _taint_policy
from cayu.server.tool_policy_details import policy_details
from cayu.tools import (
    AllowAllToolPolicy,
    AlwaysRequireApprovalToolPolicy,
    ApprovalFloorToolPolicy,
    EnvironmentScopedToolPolicy,
    ParameterConstrainedToolPolicy,
    RequiredFieldRule,
    StaticToolPolicy,
    TaintAwareToolPolicy,
    ToolPolicy,
    ToolPolicyDecision,
    ToolPolicyRequest,
    ToolPolicyResult,
)


class _Send(Tool):
    spec = ToolSpec(name="send", effect=ToolEffect.EXTERNAL)

    async def run(self, ctx, args):
        raise AssertionError("Not run while checking.")


class _OpaqueAllow(ToolPolicy):
    async def authorize(self, request: ToolPolicyRequest) -> ToolPolicyResult:
        return ToolPolicyResult(decision=ToolPolicyDecision.ALLOW)


def _decide(policy: ToolPolicy, tool_name: str, **arguments: Any) -> ToolPolicyResult:
    return asyncio.run(policy.authorize(policy_request(tool_name, **arguments)))


def _described(tools: list[Tool], policy: ToolPolicy) -> tuple[dict[str, Any], dict[str, str]]:
    """Each tool's manifest entry, and each external tool ``cayu check --deploy`` flags."""

    app = CayuApp(enable_logging=False)
    app.register_agent(AgentSpec(name="assistant", model="fake"), tools=tools, tool_policy=policy)
    manifest = app.describe()
    flagged = {
        str(diagnostic.parameters["tool"]): diagnostic.code
        for diagnostic in check_manifest(manifest, deploy_only=True).diagnostics
        if diagnostic.code.startswith("EXTERNAL_TOOL_")
    }
    return {tool.name: tool for tool in manifest.agents[0].tools}, flagged


def test_named_tools_need_approval_where_inner_allows_them() -> None:
    policy = ApprovalFloorToolPolicy(
        {"send": "Send it.", "post": lambda arguments: f"Post to {arguments['to']}."},
        StaticToolPolicy(deny=["post"]),
    )
    sent = _decide(policy, "send")
    posted = _decide(policy, "post", to="ops")
    read = _decide(policy, "read")

    assert (sent.decision, sent.reason) == (ToolPolicyDecision.REQUIRE_APPROVAL, "Send it.")
    assert posted.decision == ToolPolicyDecision.DENY
    assert read.decision == ToolPolicyDecision.ALLOW
    assert policy.tools == {"send", "post"}
    gated = ApprovalFloorToolPolicy(
        {"send": "Send it."}, AlwaysRequireApprovalToolPolicy(approval_gate="people")
    )
    kept = _decide(gated, "send")
    assert kept.metadata == {"cayu:approval_gate": "people"}
    with pytest.raises(TypeError):
        ApprovalFloorToolPolicy({"send": "x"}, object())  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        ApprovalFloorToolPolicy({})


def test_cayu_check_reads_the_floor_through_to_its_inner_policy() -> None:
    entries, flagged = _described([_Send()], ApprovalFloorToolPolicy({"send": "Send it."}))
    assert (entries["send"].policy_coverage, flagged) == ("approval_required", {})

    denied = ApprovalFloorToolPolicy({"send": "x"}, StaticToolPolicy(deny=["send"]))
    assert _described([_Send()], denied)[0]["send"].policy_coverage == "denied"

    unnamed, flagged = _described([_Send()], ApprovalFloorToolPolicy({"other": "x"}))
    assert (unnamed["send"].policy_coverage, flagged) == (
        "allowed",
        {"send": "EXTERNAL_TOOL_UNGUARDED"},
    )


def test_an_opaque_inner_policy_stays_unknown() -> None:
    entries, flagged = _described([_Send()], ApprovalFloorToolPolicy({"send": "x"}, _OpaqueAllow()))
    assert entries["send"].policy_coverage == "unknown"
    assert flagged == {"send": "EXTERNAL_TOOL_COVERAGE_UNKNOWN"}


def test_environment_and_parameter_facts_are_read_through_the_floor() -> None:
    scoped = ApprovalFloorToolPolicy(
        {"send": "x"}, EnvironmentScopedToolPolicy(allow={"send": ["staging", "dev"]})
    )
    constrained = ApprovalFloorToolPolicy(
        {"send": "x"},
        ParameterConstrainedToolPolicy(
            rules={"send": [RequiredFieldRule("to")]},
            decision=ToolPolicyDecision.REQUIRE_APPROVAL,
        ),
    )

    assert _described([_Send()], scoped)[0]["send"].policy_environment_names == ("dev", "staging")
    assert (
        _described([_Send()], constrained)[0]["send"].parameter_policy_decision
        == "require_approval"
    )


def test_cayu_check_accepts_the_slack_and_snowflake_write_tools_under_their_policies() -> None:
    local = LocalConnections({})
    tools = [
        *slack.tools(local, include_writes=True),
        *snowflake.tools(local, include_writes=True),
    ]
    writes = {"slack_post_message", "snowflake_execute"}

    entries, flagged = _described(tools, slack.approval_policy(snowflake.approval_policy()))
    assert flagged == {}
    assert {name for name, entry in entries.items() if entry.policy_coverage != "allowed"} == {
        *writes,
        "slack_update_message",
    }
    assert {entries[name].policy_coverage for name in writes} == {"approval_required"}
    assert isinstance(slack.approval_policy(), ApprovalFloorToolPolicy)

    _, unguarded = _described(tools, AllowAllToolPolicy())
    assert unguarded == dict.fromkeys(writes, "EXTERNAL_TOOL_UNGUARDED")


class _Delete(Tool):
    spec = ToolSpec(name="delete", effect=ToolEffect.EXTERNAL)

    async def run(self, ctx, args):
        raise AssertionError("Not run while checking.")


def test_a_tool_the_inner_allowlist_names_stays_deliberately_allowed() -> None:
    floor = ApprovalFloorToolPolicy(
        {"delete": "Review it."}, StaticToolPolicy(allow=["send", "delete"])
    )

    entries, flagged = _described([_Send(), _Delete()], floor)

    assert entries["send"].policy_coverage == "allowed"
    assert entries["delete"].policy_coverage == "approval_required"
    assert flagged == {}


def test_the_runtime_finds_a_taint_policy_under_the_floor() -> None:
    taint = TaintAwareToolPolicy(
        taint_sources={"read_inbox": ["untrusted"]},
        protected_tools={"send": ["untrusted"]},
        decision=ToolPolicyDecision.DENY,
    )

    assert _taint_policy(ApprovalFloorToolPolicy({"delete": "x"}, taint)) is taint


def test_the_published_policy_details_name_the_floor() -> None:
    floor = ApprovalFloorToolPolicy({"delete": "x"}, StaticToolPolicy(allow=["send", "delete"]))

    named, named_complete = policy_details(floor, "delete")
    other = policy_details(floor, "send")

    assert named[0].startswith("A person's approval is required")
    assert named[1:] == ["This policy allows this tool without approval."]
    assert named_complete
    assert other == policy_details(floor.inner, "send")
