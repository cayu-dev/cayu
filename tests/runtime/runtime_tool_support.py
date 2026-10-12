"""Tool support."""

from __future__ import annotations

from cayu.runtime.execution_identity import ExecutionProfileBehaviorIdentity
from cayu.tools.base import Tool, ToolContext, ToolEffect, ToolResult, ToolSpec
from cayu.tools.policy import ToolPolicy, ToolPolicyDecision, ToolPolicyRequest, ToolPolicyResult


class EchoTool(Tool):
    spec = ToolSpec(
        name="echo",
        description="Echo text.",
        input_schema={
            "type": "object",
            "properties": {"text": {"type": "string"}},
            "required": ["text"],
        },
    )

    async def run(self, ctx: ToolContext, args: dict) -> ToolResult:
        return ToolResult(
            content=args["text"],
            structured={"agent": ctx.agent_name, "echoed": args["text"]},
        )


class SideEffectTool(Tool):
    spec = ToolSpec(
        name="side_effect",
        description="Record execution.",
        input_schema={"type": "object", "properties": {}},
        execution_profile_identity=ExecutionProfileBehaviorIdentity(
            name="tests:runtime:side-effect-tool",
            behavior_version="1",
            implementation_version="1",
        ),
    )

    def __init__(self) -> None:
        super().__init__()
        self.calls: list[dict] = []
        self.contexts: list[ToolContext] = []

    async def run(self, ctx: ToolContext, args: dict) -> ToolResult:
        self.calls.append(args)
        self.contexts.append(ctx)
        return ToolResult(content="recorded")


class PureRecordingTool(SideEffectTool):
    """Compute a fixed result; calls/contexts are test instrumentation only.

    Manual-recovery lifecycle tests use this tool because operator disposition
    remains supported for NONE effects, not for selected external evidence.
    """

    spec = SideEffectTool.spec.model_copy(update={"effect": ToolEffect.NONE})


class RequireApprovalPolicy(ToolPolicy):
    async def authorize(self, request: ToolPolicyRequest) -> ToolPolicyResult:
        return ToolPolicyResult(
            decision=ToolPolicyDecision.REQUIRE_APPROVAL,
            reason=f"Approval required for {request.tool_name}.",
            metadata={"scope": "human"},
        )


class SideEffectApprovalPolicy(ToolPolicy):
    async def authorize(self, request: ToolPolicyRequest) -> ToolPolicyResult:
        if request.tool_name == "side_effect":
            return ToolPolicyResult(
                decision=ToolPolicyDecision.REQUIRE_APPROVAL,
                reason=f"Approval required for {request.tool_name}.",
                metadata={"scope": "human"},
            )
        return ToolPolicyResult(decision=ToolPolicyDecision.ALLOW)


class UpperTool(Tool):
    spec = ToolSpec(
        name="upper",
        description="Uppercase text.",
        input_schema={
            "type": "object",
            "properties": {"text": {"type": "string"}},
            "required": ["text"],
        },
    )

    async def run(self, ctx: ToolContext, args: dict) -> ToolResult:
        return ToolResult(content=args["text"].upper())


class _TaintSourceTool(Tool):
    spec = ToolSpec(
        name="read_web",
        description="Fetch untrusted web content.",
        input_schema={"type": "object", "properties": {}},
    )

    async def run(self, ctx: ToolContext, args: dict) -> ToolResult:
        return ToolResult(content="untrusted page content")


class _ProtectedEmailTool(Tool):
    spec = ToolSpec(
        name="send_email",
        description="Send an email to an external address.",
        input_schema={
            "type": "object",
            "properties": {"body": {"type": "string"}},
        },
    )

    def __init__(self) -> None:
        super().__init__()
        self.calls: list[dict] = []

    async def run(self, ctx: ToolContext, args: dict) -> ToolResult:
        self.calls.append(args)
        return ToolResult(content="email sent")


class ExpiringApprovalPolicy(ToolPolicy):
    def __init__(self, expires_in_seconds: float = 60.0) -> None:
        self.expires_in_seconds = expires_in_seconds

    async def authorize(self, request: ToolPolicyRequest) -> ToolPolicyResult:
        return ToolPolicyResult(
            decision=ToolPolicyDecision.REQUIRE_APPROVAL,
            reason=f"Approval required for {request.tool_name}.",
            approval_expires_in_seconds=self.expires_in_seconds,
        )
