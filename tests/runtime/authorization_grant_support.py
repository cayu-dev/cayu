"""Shared fixtures for authorization-grant runtime and server tests."""

from __future__ import annotations

from datetime import UTC, datetime

from tests.runtime.runtime_provider_support import FakeProvider

from cayu.agents import AgentSpec
from cayu.applications import CayuApp
from cayu.approvals import (
    TOOL_APPROVAL_GATE_METADATA_KEY,
    TOOL_MANDATORY_APPROVAL_METADATA_KEY,
    AllowedActorsGrantAuthority,
    InMemoryToolAuthorizationGrantStore,
    PendingToolApproval,
    ResolutionActor,
    ResolutionActorSource,
    ToolApprovalDecision,
    ToolApprovalRequest,
    ToolAuthorizationGrantControl,
    ToolAuthorizationGrantPolicy,
    ToolAuthorizationGrantRequest,
    ToolAuthorizationGrantService,
)
from cayu.environments.base import Environment, EnvironmentSpec
from cayu.messages import Message
from cayu.providers.base import ModelStreamEvent
from cayu.runtime.execution_identity import ExecutionProfileBehaviorIdentity
from cayu.sessions.memory import InMemorySessionStore
from cayu.sessions.requests import ResumeRequest, RunRequest
from cayu.tools.base import Tool, ToolContext, ToolResult, ToolSpec
from cayu.tools.policy import (
    ToolPolicy,
    ToolPolicyDecision,
    ToolPolicyRequest,
    ToolPolicyResult,
)
from cayu.workspaces.local import LocalWorkspace

OWNER = ResolutionActor(subject="owner", tenant="acme", source=ResolutionActorSource.HTTP_AUTH)
INTRUDER = ResolutionActor(
    subject="intruder", tenant="acme", source=ResolutionActorSource.HTTP_AUTH
)


class Clock:
    def __init__(self) -> None:
        self.now = datetime(2026, 10, 1, tzinfo=UTC)

    def __call__(self) -> datetime:
        return self.now


class CommandTool(Tool):
    """Records process commands instead of running them."""

    spec = ToolSpec(
        name="exec_command",
        description="Record a command.",
        input_schema={
            "type": "object",
            "properties": {
                "argv": {"type": "array", "items": {"type": "string"}},
                "shell": {"type": "string"},
                "cwd": {"type": "string"},
            },
        },
        execution_profile_identity=ExecutionProfileBehaviorIdentity(
            name="tests:grants:command", behavior_version="1", implementation_version="1"
        ),
    )

    def __init__(self) -> None:
        super().__init__()
        self.calls: list[dict] = []

    async def run(self, ctx: ToolContext, args: dict) -> ToolResult:
        self.calls.append(args)
        return ToolResult(content="ran")


class GatedPolicy(ToolPolicy):
    """Requires approval under one named gate, optionally marked mandatory."""

    def __init__(self, gate: str = "commands", *, mandatory: bool = False) -> None:
        self.gate = gate
        self.mandatory = mandatory

    @property
    def execution_profile_identity(self) -> ExecutionProfileBehaviorIdentity:
        return ExecutionProfileBehaviorIdentity(
            name=f"tests:grants:gated:{self.gate}:{self.mandatory}",
            behavior_version="1",
            implementation_version="1",
        )

    async def authorize(self, request: ToolPolicyRequest) -> ToolPolicyResult:
        metadata = {TOOL_APPROVAL_GATE_METADATA_KEY: self.gate}
        if self.mandatory:
            metadata[TOOL_MANDATORY_APPROVAL_METADATA_KEY] = True
        return ToolPolicyResult(
            decision=ToolPolicyDecision.REQUIRE_APPROVAL,
            reason="commands need approval",
            metadata=metadata,
        )


def _command_turn(call_id: str, argv: list[str]) -> list[list[ModelStreamEvent]]:
    return [
        [
            ModelStreamEvent.tool_call(id=call_id, name="exec_command", arguments={"argv": argv}),
            ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
        ],
        [
            ModelStreamEvent.text_delta("done"),
            ModelStreamEvent.completed({"finish_reason": "stop"}),
        ],
    ]


class Harness:
    def __init__(
        self,
        tmp_path,
        *,
        turns: list[list[ModelStreamEvent]],
        then: ToolPolicy | None = None,
        workspace_id: str | None = "ws-1",
        explicit_workspace: bool = True,
        tool: Tool | None = None,
        boundary_labels: tuple[str, ...] = (),
        mandatory_controls: tuple[ToolAuthorizationGrantControl, ...] = (),
        default_sampled_review_rate_bps: int = 0,
        grants: InMemoryToolAuthorizationGrantStore | None = None,
        sessions: InMemorySessionStore | None = None,
    ) -> None:
        self.clock = Clock()
        self.sessions = sessions or InMemorySessionStore()
        self.grants = grants or InMemoryToolAuthorizationGrantStore()
        self.tool = tool or CommandTool()
        self.app = CayuApp(session_store=self.sessions, enable_logging=False)
        self.app.register_provider(FakeProvider(turns), default=True)
        if workspace_id is not None:
            root = tmp_path / workspace_id
            root.mkdir(exist_ok=True)
            self.app.register_environment(
                Environment(
                    EnvironmentSpec(name="local"),
                    workspace=(
                        LocalWorkspace(root, workspace_id=workspace_id)
                        if explicit_workspace
                        else LocalWorkspace(root)
                    ),
                ),
                default=True,
            )
        self.policy = ToolAuthorizationGrantPolicy(
            policy_id="commands",
            store=self.grants,
            grantable={"exec_command": "process_command"},
            then=then or GatedPolicy(),
            mandatory_controls=mandatory_controls,
            boundary_labels=boundary_labels,
            clock=self.clock,
        )
        self.app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            tools=[self.tool],
            tool_policy=self.policy,
        )
        self.service = ToolAuthorizationGrantService(
            self.app,
            store=self.grants,
            authority=AllowedActorsGrantAuthority(creators=["owner"]),
            default_sampled_review_rate_bps=default_sampled_review_rate_bps,
            clock=self.clock,
        )

    async def run(self, session_id: str, *, labels: dict[str, str] | None = None):
        if await self.sessions.load(session_id) is not None:
            return [
                event
                async for event in self.app.resume(
                    ResumeRequest(session_id=session_id, messages=[Message.text("user", "again")])
                )
            ]
        return [
            event
            async for event in self.app.run(
                RunRequest(
                    agent_name="assistant",
                    session_id=session_id,
                    messages=[Message.text("user", "run it")],
                    labels=labels or {},
                )
            )
        ]

    async def pending(self, session_id: str) -> PendingToolApproval:
        checkpoint = await self.sessions.load_checkpoint(session_id)
        pending = self.app._pending_tool_approval_from_checkpoint(checkpoint)
        assert pending is not None
        return pending

    async def approval(self, session_id: str, *, actor=OWNER, **kwargs) -> ToolApprovalRequest:
        pending = await self.pending(session_id)
        return ToolApprovalRequest(
            session_id=session_id,
            approval_id=pending.approval_id,
            tool_round_id=pending.tool_round_id,
            tool_call_id=pending.tool_call_id,
            decision=kwargs.pop("decision", ToolApprovalDecision.APPROVE),
            resolved_by=actor,
            **kwargs,
        )

    async def remember(self, session_id: str, grant: ToolAuthorizationGrantRequest, **kwargs):
        approval = await self.approval(session_id, **kwargs)
        return [event async for event in self.service.approve_and_grant(approval, grant)]
