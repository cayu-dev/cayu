"""Re-authorizing changed arguments sees the pause's own authority and never escalates it."""

from __future__ import annotations

import asyncio
from contextlib import aclosing

import pytest

from cayu import AgentSpec, CayuApp, Message, RunRequest
from cayu.approvals.business import BusinessApprovalRouting, TieredApprovalPolicy
from cayu.approvals.tools import (
    ToolApprovalArgumentsRefused,
    ToolApprovalDecision,
    ToolApprovalRequest,
)
from cayu.environments import Environment, EnvironmentSpec
from cayu.environments.factory import (
    EnvironmentFactory,
    EnvironmentFactoryResult,
)
from cayu.evals import ScriptedModelProvider
from cayu.events import EventType
from cayu.providers import ModelStreamEvent
from cayu.runtime.execution_identity import ExecutionProfileBehaviorIdentity
from cayu.sessions import _pending_approval_reader as pending_approval_reader
from cayu.sessions.memory import InMemorySessionStore
from cayu.tools.base import Tool, ToolContext, ToolResult, ToolSpec
from cayu.tools.policy import ToolPolicy, ToolPolicyDecision, ToolPolicyResult
from cayu.workspaces import LocalWorkspace


def _identity(name: str) -> ExecutionProfileBehaviorIdentity:
    return ExecutionProfileBehaviorIdentity(
        name=name, behavior_version="1", implementation_version="1"
    )


class _PayTool(Tool):
    spec = ToolSpec(
        name="pay",
        description="Pay an amount.",
        input_schema={"type": "object", "properties": {"amount": {"type": "integer"}}},
        execution_profile_identity=_identity("tests:edit-authority:pay"),
    )

    def __init__(self) -> None:
        super().__init__()
        self.calls: list[dict] = []

    async def run(self, ctx: ToolContext, args: dict) -> ToolResult:
        self.calls.append(dict(args))
        return ToolResult(content="paid")


class _FinanceFactory(EnvironmentFactory):
    """Allocates the application-named ``finance`` workspace when a run starts."""

    def __init__(self, root) -> None:
        self.root = root

    @property
    def secret_resolution_scope(self):
        return "static"

    @property
    def execution_profile_identity(self):
        return _identity("tests:edit-authority:factory")

    async def create(self, request) -> EnvironmentFactoryResult:
        return EnvironmentFactoryResult(
            environment=Environment(
                EnvironmentSpec(name=request.environment_name),
                workspace=LocalWorkspace(self.root, workspace_id="finance"),
            ),
        )


class _FinanceWorkspacePolicy(ToolPolicy):
    """Payments need approval inside the named finance workspace and nowhere else."""

    @property
    def execution_profile_identity(self):
        return _identity("tests:edit-authority:workspace-policy")

    async def authorize(self, request):
        if request.workspace_id != "finance" or not request.workspace_id_explicit:
            return ToolPolicyResult(decision=ToolPolicyDecision.DENY, reason="Wrong workspace.")
        return ToolPolicyResult(decision=ToolPolicyDecision.REQUIRE_APPROVAL)


class _ExpiringPolicy(ToolPolicy):
    """Large payments have a much shorter approval window."""

    async def authorize(self, request):
        large = request.arguments.get("amount", 0) > 1000
        return ToolPolicyResult(
            decision=ToolPolicyDecision.REQUIRE_APPROVAL,
            approval_expires_in_seconds=0.001 if large else 3600,
        )


def _tiered() -> TieredApprovalPolicy:
    def route(request):
        tier = "cfo" if request.arguments.get("amount", 0) > 1000 else "team_lead"
        return BusinessApprovalRouting(required_tier=tier, chain=("team_lead", "cfo"))

    return TieredApprovalPolicy(compute_routing=route)


def _app(policy: ToolPolicy, *, factory_root=None) -> tuple[CayuApp, _PayTool]:
    tool = _PayTool()
    app = CayuApp(session_store=InMemorySessionStore(), enable_logging=False)
    app.register_provider(
        ScriptedModelProvider(
            [
                [
                    ModelStreamEvent.tool_call(id="pay", name="pay", arguments={"amount": 100}),
                    ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
                ],
                [
                    ModelStreamEvent.text_delta("done"),
                    ModelStreamEvent.completed({"finish_reason": "stop"}),
                ],
            ]
        ),
        default=True,
    )
    if factory_root is not None:
        app.register_environment_factory(
            EnvironmentSpec(name="finance", execution_profile_identity=_identity("tests:env")),
            _FinanceFactory(factory_root),
            default=True,
        )
    app.register_agent(
        AgentSpec(name="assistant", model="scripted-model"), tools=[tool], tool_policy=policy
    )
    return app, tool


async def _drain(stream):
    async with aclosing(stream):
        return [event async for event in stream]


async def _pause(app: CayuApp, session_id: str) -> ToolApprovalRequest:
    events = await _drain(
        app.run(
            RunRequest(
                agent_name="assistant",
                session_id=session_id,
                messages=[Message.text("user", "pay it")],
            )
        )
    )
    paused = next(e for e in events if e.type is EventType.TOOL_CALL_APPROVAL_REQUESTED)
    return ToolApprovalRequest(
        session_id=session_id,
        approval_id=paused.payload["approval_id"],
        tool_round_id=paused.payload["tool_round_id"],
        tool_call_id=paused.payload["tool_call_id"],
        decision=ToolApprovalDecision.APPROVE,
    )


async def _assert_still_pending(app: CayuApp, session_id: str) -> None:
    checkpoint = await app.session_store.load_checkpoint(session_id)
    assert pending_approval_reader.pending_approval_from_checkpoint(checkpoint) is not None
    assert pending_approval_reader.approval_resolution_intent_from_checkpoint(checkpoint) is None


def test_reauthorization_sees_the_factory_workspace_the_pause_saw(tmp_path):
    app, tool = _app(_FinanceWorkspacePolicy(), factory_root=tmp_path)

    async def scenario():
        request = await _pause(app, "edit-factory")
        events = await _drain(
            app.resolve_tool_approval(request.model_copy(update={"arguments": {"amount": 90}}))
        )
        assert events[-1].type is EventType.SESSION_COMPLETED

    asyncio.run(scenario())
    assert tool.calls == [{"amount": 90}]


def test_an_edit_that_escalates_a_business_tier_is_refused():
    app, tool = _app(_tiered())

    async def scenario():
        request = await _pause(app, "edit-tier-escalation")
        with pytest.raises(ToolApprovalArgumentsRefused, match="authority"):
            await _drain(
                app.resolve_tool_approval(
                    request.model_copy(update={"arguments": {"amount": 10_000_000}})
                )
            )
        await _assert_still_pending(app, request.session_id)
        # An edit within the same tier is covered by the approval.
        events = await _drain(
            app.resolve_tool_approval(request.model_copy(update={"arguments": {"amount": 500}}))
        )
        assert events[-1].type is EventType.SESSION_COMPLETED

    asyncio.run(scenario())
    assert tool.calls == [{"amount": 500}]


def test_an_edit_whose_own_approval_window_has_passed_is_refused():
    app, tool = _app(_ExpiringPolicy())

    async def scenario():
        request = await _pause(app, "edit-expiry")
        await asyncio.sleep(0.01)
        with pytest.raises(ToolApprovalArgumentsRefused, match="approval window"):
            await _drain(
                app.resolve_tool_approval(
                    request.model_copy(update={"arguments": {"amount": 5000}})
                )
            )
        await _assert_still_pending(app, request.session_id)

    asyncio.run(scenario())
    assert tool.calls == []
