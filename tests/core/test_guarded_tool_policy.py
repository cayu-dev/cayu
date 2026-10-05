from __future__ import annotations

import asyncio

import pytest
from tests.core.test_tool_policies import _request

from cayu import (
    AgentSpec,
    AllowAllToolPolicy,
    AlwaysRequireApprovalToolPolicy,
    CayuApp,
    EnvironmentScopedToolPolicy,
    ExecutionProfileBehaviorIdentity,
    GuardedToolPolicy,
    RequiredArguments,
    StaticToolPolicy,
    Tool,
    ToolEffect,
    ToolPolicyDecision,
    ToolPolicyResult,
    ToolSpec,
    UserInputTool,
    check_manifest,
)
from cayu.runtime._execution_profile_admission import _cayu_policy_material
from cayu.runtime.manifest import _tool_policy_coverage


class Guard:
    def __init__(self, version="1", result=None):
        self.execution_profile_identity = ExecutionProfileBehaviorIdentity(
            name="test.guard",
            behavior_version=version,
            implementation_version=version,
        )
        self.result = result

    async def check(self, request):
        return self.result


@pytest.mark.parametrize("value", [None, "", "   ", [], {}])
def test_required_arguments_deny_without_calling_approval_policy(value):
    policy = GuardedToolPolicy(
        guards=[RequiredArguments({"ask_user": ["question"]})],
        then=AlwaysRequireApprovalToolPolicy(),
    )
    result = asyncio.run(
        policy.authorize(
            _request(
                tool_name="ask_user",
                arguments={"question": value},
            )
        )
    )
    assert result.decision == ToolPolicyDecision.DENY
    assert "question" in result.reason
    assert result.metadata["reason"] == "invalid_arguments"
    assert result.metadata["parameter"] == "question"


def test_valid_call_reaches_downstream_approval_with_expiry():
    policy = GuardedToolPolicy(
        guards=[RequiredArguments({"ask_user": ["question"]})],
        then=AlwaysRequireApprovalToolPolicy(expires_in_seconds=60),
    )
    result = asyncio.run(
        policy.authorize(_request(tool_name="ask_user", arguments={"question": "q"}))
    )
    assert result.decision == ToolPolicyDecision.REQUIRE_APPROVAL
    assert result.approval_expires_in_seconds == 60


@pytest.mark.parametrize(
    "result",
    [
        "allow",
        {},
        ToolPolicyResult(decision=ToolPolicyDecision.ALLOW),
        ToolPolicyResult(decision=ToolPolicyDecision.REQUIRE_APPROVAL),
    ],
)
def test_invalid_guard_return_fails_closed(result):
    policy = GuardedToolPolicy(guards=[Guard(result=result)], then=StaticToolPolicy())
    outcome = asyncio.run(policy.authorize(_request()))
    assert outcome.decision == ToolPolicyDecision.DENY
    assert "failed closed" in outcome.reason


def test_guard_exception_is_content_free_and_cancellation_propagates():
    class Raise:
        def __init__(self, error):
            self.error = error

        async def check(self, request):
            raise self.error

    policy = GuardedToolPolicy(
        guards=[Raise(ValueError("private-argument"))], then=StaticToolPolicy()
    )
    outcome = asyncio.run(policy.authorize(_request()))
    assert "private-argument" not in outcome.model_dump_json()
    policy = GuardedToolPolicy(guards=[Raise(asyncio.CancelledError())], then=StaticToolPolicy())
    with pytest.raises(asyncio.CancelledError):
        asyncio.run(policy.authorize(_request()))


def test_guard_cannot_mutate_downstream_authority():
    class Mutate:
        async def check(self, request):
            request.tool_name = "allowed"

    policy = GuardedToolPolicy(guards=[Mutate()], then=StaticToolPolicy(allow=["allowed"]))
    assert asyncio.run(policy.authorize(_request())).decision == ToolPolicyDecision.DENY


def test_checker_does_not_infer_coverage_from_custom_guards():
    class Uncalled:
        async def check(self, request):
            raise AssertionError("Do not execute during inspect")

    class ExternalTool(Tool):
        spec = ToolSpec(name="send_message", effect=ToolEffect.EXTERNAL)

        async def run(self, ctx, args):
            raise AssertionError("Do not execute during inspect")

    app = CayuApp(enable_logging=False)
    app.register_agent(
        AgentSpec(name="assistant", model="fake"),
        tools=[ExternalTool()],
        tool_policy=GuardedToolPolicy(
            guards=[Uncalled()], then=AlwaysRequireApprovalToolPolicy(tools=["remember_knowledge"])
        ),
    )
    manifest = app.describe()
    assert manifest.agents[0].tools[0].policy_coverage == "allowed"
    assert any(
        d.code == "EXTERNAL_TOOL_UNGUARDED"
        for d in check_manifest(manifest, deploy_only=True).diagnostics
    )


def test_recursive_coverage_keeps_overriding_subclasses_unknown():
    class Derived(GuardedToolPolicy):
        pass

    class Unknown(StaticToolPolicy):
        pass

    guard = RequiredArguments({"send_email": ["to"]})
    builtin = GuardedToolPolicy(guards=[guard], then=AlwaysRequireApprovalToolPolicy())
    outer = GuardedToolPolicy(guards=[guard], then=builtin)
    assert _tool_policy_coverage(outer, "send_email", {}) == "approval_required"
    assert (
        _tool_policy_coverage(Derived(guards=[guard], then=builtin), "send_email", {}) == "unknown"
    )
    assert (
        _tool_policy_coverage(GuardedToolPolicy(guards=[guard], then=Unknown()), "send_email", {})
        == "unknown"
    )


def test_composite_profile_material_is_stable_and_tracks_every_component():
    def policy(version="1", path="to", expiry=60):
        return GuardedToolPolicy(
            guards=[RequiredArguments({"send_email": [path]}), Guard(version)],
            then=AlwaysRequireApprovalToolPolicy(expires_in_seconds=expiry),
        )

    assert _cayu_policy_material(policy()) == _cayu_policy_material(policy())
    for changed in (policy(version="2"), policy(path="recipient"), policy(expiry=120)):
        assert _cayu_policy_material(changed) != _cayu_policy_material(policy())

    class Undeclared:
        async def check(self, request):
            return None

    assert (
        _cayu_policy_material(GuardedToolPolicy(guards=[Undeclared()], then=StaticToolPolicy()))
        is None
    )


def test_required_arguments_only_cover_tools_with_nonempty_exact_rules():
    policy = GuardedToolPolicy(
        guards=[RequiredArguments({"ask_user": ["question"], "send_payment": []})],
        then=AllowAllToolPolicy(),
    )
    assert _tool_policy_coverage(policy, "ask_user", {}) == "conditional"
    assert _tool_policy_coverage(policy, "send_payment", {}) == "allowed"
    assert _tool_policy_coverage(policy, "other_tool", {}) == "allowed"


def test_nested_guards_preserve_environment_scope_in_manifest():
    app = CayuApp(enable_logging=False)
    app.register_agent(
        AgentSpec(name="assistant", model="fake"),
        tools=[UserInputTool()],
        tool_policy=GuardedToolPolicy(
            guards=(),
            then=GuardedToolPolicy(
                guards=(), then=EnvironmentScopedToolPolicy(allow={"ask_user": ["isolated"]})
            ),
        ),
    )
    tool = app.describe().agents[0].tools[0]
    assert tool.policy_coverage == "conditional"
    assert tool.policy_environment_names == ("isolated",)
