"""Offer native Cayu tools to vendor harnesses without bypassing tool authority."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Iterable
from contextlib import asynccontextmanager
from typing import Any
from uuid import uuid4

from cayu.agents import AgentSpec
from cayu.applications import CayuApp
from cayu.approvals.authorization_grants import APPROVED_ARGUMENTS_REAUTHORIZATION_METADATA_KEY
from cayu.budgets.billing import BillingIdentity
from cayu.budgets.pricing import PriceBook, _external_tool_dispatch_price
from cayu.deadlines import ExecutionDeadline, execution_deadline_scope
from cayu.events import EventType
from cayu.external_agents._authority import run_authority
from cayu.external_agents.environment import join_cleanup
from cayu.messages import Message, TextPart, ToolResultPart
from cayu.observability.hooks import BeforeToolCallDecision, RuntimeHook
from cayu.providers.base import ModelProvider, ModelRequest, ModelStreamEvent, UsageDialect
from cayu.sessions.records import SessionStatus
from cayu.sessions.requests import RunRequest
from cayu.tools._terminal_bounds import MIN_TOOL_TERMINAL_PAYLOAD_BYTES
from cayu.tools.base import Tool, ToolContext, ToolEffect, ToolResult, ToolSpec
from cayu.tools.policy import ToolPolicy, ToolPolicyDecision, ToolPolicyRequest, ToolPolicyResult

_DISPATCH_PROVIDER = "cayu-tool-dispatch"
_DISPATCH_MODEL = "tool-dispatch"


def external_tool_price_book(pricing: PriceBook) -> PriceBook:
    """Price native bridge control steps at zero: they never call a model."""
    owned = _external_tool_dispatch_price()
    present = tuple(price for price in pricing.prices if price.provider_name == _DISPATCH_PROVIDER)
    if present == (owned,):
        return pricing.model_copy(deep=True)
    if present:
        raise ValueError("The tool-dispatch price is runtime-owned.")
    return pricing.model_copy(update={"prices": (*pricing.prices, owned)})


class _ToolDispatchProvider(ModelProvider):
    """Deterministic control input to the native loop; no external harness/model."""

    name = _DISPATCH_PROVIDER
    usage_dialect = UsageDialect.GENERIC

    def __init__(self, calls: dict[str, dict[str, Any]]) -> None:
        self.calls = calls

    async def billing_identity_for_request(self, request: ModelRequest) -> BillingIdentity:
        return BillingIdentity(provider_name=_DISPATCH_PROVIDER, resource_id=_DISPATCH_MODEL)

    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
        for message in reversed(request.messages):
            for part in message.content:
                if isinstance(part, ToolResultPart):
                    yield ModelStreamEvent.text_delta("Native tool dispatch completed.")
                    yield ModelStreamEvent.completed(
                        {"usage": {"input_tokens": 0, "output_tokens": 0}}
                    )
                    return
        first = next(
            part.text
            for message in request.messages
            if message.role == "user"
            for part in message.content
            if isinstance(part, TextPart)
        )
        call = self.calls[first]
        yield ModelStreamEvent.tool_call(
            name=call["name"], arguments=call["arguments"], id=call["id"]
        )
        yield ModelStreamEvent.completed({"usage": {"input_tokens": 0, "output_tokens": 0}})


class _LiveToolHook(RuntimeHook):
    def __init__(self, app, active, invocations) -> None:
        self.app, self.active, self.invocations = app, active, invocations

    async def before_tool_call(self, context):
        if context.session.id not in self.active:
            return BeforeToolCallDecision(
                action="block", block_reason="External caller is no longer active."
            )
        try:
            authority = run_authority(self.app, context.session.parent_session_id)
            if authority.context.pending_model_usage:
                return BeforeToolCallDecision(
                    action="block", block_reason="External accounting is unsettled."
                )
            if authority.context.check_budget is not None:
                await authority.context.check_budget()
        except Exception:
            # Runtime hook exceptions are observations, not authorization denials.
            return BeforeToolCallDecision(
                action="block", block_reason="External run cannot admit more work."
            )
        if context.session.id not in self.active:
            return BeforeToolCallDecision(
                action="block", block_reason="External caller is no longer active."
            )
        task = asyncio.current_task()
        assert task is not None
        self.invocations.setdefault(context.session.id, set()).add(task)
        return None

    async def after_tool_call(self, context):
        self.invocations.get(context.session.id, set()).discard(asyncio.current_task())
        return None


class ExternalPermissionTool(Tool):
    """A permission receipt, not execution of the vendor's built-in tool.

    A native tool policy sees the vendor name and exact arguments. An approval
    pauses this child session in Cayu's ordinary durable approval checkpoint.
    Only the live SDK call can consume the successful receipt.
    Approval edits are refused because the SDK consumes a boolean permission
    for its original arguments, not a replacement invocation.
    """

    def __init__(self, name: str, *, requested_effect: ToolEffect = ToolEffect.EXTERNAL) -> None:
        self.requested_effect = requested_effect
        super().__init__(
            ToolSpec(
                name=name,
                description="Authorize one external harness built-in tool call.",
                input_schema={"type": "object"},
                effect=ToolEffect.NONE,
                # The receipt is a fixed few bytes; this is the smallest bound Cayu accepts.
                max_terminal_payload_bytes=MIN_TOOL_TERMINAL_PAYLOAD_BYTES,
            )
        )

    async def run(self, ctx: ToolContext, args: dict[str, Any]) -> ToolResult:
        return ToolResult(content="authorized", structured={"authorized": True})


class _ExternalToolPolicy(ToolPolicy):
    def __init__(self, inner: ToolPolicy, tools: dict[str, Tool]) -> None:
        self.inner = inner
        self.tools = tools

    async def authorize(self, request: ToolPolicyRequest) -> ToolPolicyResult:
        tool = self.tools.get(request.tool_name)
        if isinstance(tool, ExternalPermissionTool):
            if request.metadata.get(APPROVED_ARGUMENTS_REAUTHORIZATION_METADATA_KEY) is True:
                return ToolPolicyResult(
                    decision=ToolPolicyDecision.DENY,
                    reason="External permission calls cannot approve changed arguments.",
                )
            request = request.model_copy(
                update={
                    "tool_effect": tool.requested_effect,
                    "metadata": {**request.metadata, "cayu:external_builtin_permission": True},
                }
            )
        return await self.inner.authorize(request)


class CayuToolBridge:
    """Execute offered tools through native sessions, policy, approvals and effects.

    Register the bridge before starting the application. It invokes no language
    model: a deterministic dispatcher submits one exact tool request to Cayu's
    existing loop. Its runtime-owned control identity is always priced at zero;
    no commercial model pricing is required for these native dispatch steps.
    """

    def __init__(
        self,
        app: CayuApp,
        *,
        name: str,
        tools: Iterable[Tool],
        policy: ToolPolicy,
        environment_name: str | None = None,
    ) -> None:
        self.app = app
        self.name = name
        self.environment_name = environment_name
        self.tools = {tool.name: tool for tool in tools}
        self.active: set[str] = set()
        self.invocations: dict[str, set[asyncio.Task]] = {}
        self.calls: dict[str, dict[str, Any]] = {}
        # Each bridge owns its provider registration; sharing its commercial
        # identity does not share mutable call state.
        provider = _ToolDispatchProvider(self.calls)
        provider.name = f"{_DISPATCH_PROVIDER}-{name}"
        provider.billing_provider_name = _DISPATCH_PROVIDER
        app.register_provider(provider)
        app.register_agent(
            AgentSpec(name=name, provider_name=provider.name, model=_DISPATCH_MODEL),
            tools=list(self.tools.values()),
            runtime_hooks=[_LiveToolHook(app, self.active, self.invocations)],
            tool_policy=_ExternalToolPolicy(policy, self.tools),
        )

    async def call(
        self,
        name: str,
        arguments: dict[str, Any],
        *,
        parent_session_id: str,
        execution_deadline: ExecutionDeadline,
        call_id: str,
    ) -> ToolResult:
        if name not in self.tools:
            return ToolResult(content="Tool is not offered to this external agent.", is_error=True)
        session_id = f"external-tool-{uuid4().hex}"
        parent = await self.app.session_store.load(parent_session_id)
        if (
            parent is None
            or parent.runtime_name != "external_agent"
            or parent.status is not SessionStatus.RUNNING
        ):
            return ToolResult(content="External session is not active.", is_error=True)
        authority = run_authority(self.app, parent_session_id)
        if authority.context.pending_model_usage:
            return ToolResult(content="External accounting is unsettled.", is_error=True)
        if authority.context.check_budget is not None:
            await authority.context.check_budget()
        self.active.add(session_id)
        # The transcript only needs a dispatch identity. Native tool publication
        # decides whether the exact arguments may enter events or model history.
        self.calls[session_id] = {"name": name, "arguments": arguments, "id": call_id}
        try:
            request = RunRequest(
                agent_name=self.name,
                session_id=session_id,
                parent_session_id=parent_session_id,
                causal_budget_id=parent.causal_budget_id,
                budget_limits=authority.budget_limits,
                execution_deadline=execution_deadline,
                environment_name=self.environment_name,
                messages=[Message.text("user", session_id)],
                max_steps=2,
            )
            async for _event in self.app.run(request):
                pass
            while True:
                state = await self.app.session_store.load_state(session_id)
                if state is None:
                    return ToolResult(content="Native tool session is unavailable.", is_error=True)
                events = await self.app.session_store.load_events(session_id)
                terminal = next(
                    (
                        event
                        for event in reversed(events)
                        if event.type
                        in {
                            EventType.TOOL_CALL_COMPLETED,
                            EventType.TOOL_CALL_FAILED,
                            EventType.TOOL_CALL_BLOCKED,
                            EventType.TOOL_CALL_APPROVAL_DENIED,
                            EventType.TOOL_CALL_APPROVAL_EXPIRED,
                        }
                    ),
                    None,
                )
                if terminal is not None:
                    if terminal.type in {
                        EventType.TOOL_CALL_COMPLETED,
                        EventType.TOOL_CALL_FAILED,
                    } and isinstance(terminal.payload.get("result"), dict):
                        return ToolResult.model_validate(terminal.payload["result"])
                    return ToolResult(
                        content="Native tool authorization or execution failed.", is_error=True
                    )
                if state.status not in {SessionStatus.RUNNING, SessionStatus.INTERRUPTED}:
                    return ToolResult(
                        content="Native tool run ended without a result.", is_error=True
                    )
                execution_deadline.require_admission("external tool approval wait")
                await asyncio.sleep(min(0.1, execution_deadline.remaining_seconds() or 0.1))
        finally:
            # A late approval cannot execute a tool after its external caller
            # has left, nor after process restart where this set is empty.
            self.active.discard(session_id)
            self.calls.pop(session_id, None)
            calls = self.invocations.pop(session_id, set()) - {asyncio.current_task()}

            async def stop_calls() -> None:
                for task in calls:
                    task.cancel()
                await asyncio.gather(*calls, return_exceptions=True)

            await join_cleanup(stop_calls())

    async def authorize(
        self,
        name: str,
        arguments: dict[str, Any],
        call_id: str,
        *,
        parent_session_id: str,
        execution_deadline: ExecutionDeadline,
    ) -> bool:
        result = await self.call(
            name,
            arguments,
            parent_session_id=parent_session_id,
            execution_deadline=execution_deadline,
            call_id=call_id,
        )
        return (
            not result.is_error
            and result.structured is not None
            and result.structured.get("authorized") is True
        )

    async def mcp_tools(self, parent_session_id: str) -> list[dict[str, Any]]:
        authority = run_authority(self.app, parent_session_id)
        if authority.context.check_budget is not None:
            await authority.context.check_budget()
        parent = await self.app.session_store.load(parent_session_id)
        if parent is None:
            raise PermissionError("External session is unavailable.")
        registered = self.app._get_registered_agent(self.name)
        async for _ in self.app._tool_round_executor.emit_mcp_manifest_checks(
            session=parent,
            registered_agent=registered,
            environment_name=self.environment_name,
        ):
            pass
        return [
            {"name": tool.name, "description": tool.description, "inputSchema": tool.schema}
            for tool in registered.tools.values()
        ]


@asynccontextmanager
async def external_tool_scope(
    bridge: CayuToolBridge,
    *,
    session_id: str,
    execution_deadline: ExecutionDeadline,
) -> AsyncIterator[CallableToolEndpoint]:
    """Bind one revocable MCP endpoint to one live external run."""
    endpoint = CallableToolEndpoint(bridge, session_id, execution_deadline)
    try:
        yield endpoint
    finally:
        await join_cleanup(endpoint.close())


class CallableToolEndpoint:
    def __init__(
        self, bridge: CayuToolBridge, session_id: str, deadline: ExecutionDeadline
    ) -> None:
        self.bridge = bridge
        self.session_id = session_id
        self.deadline = deadline
        self.active = True
        self.tasks: set[asyncio.Task] = set()

    async def close(self) -> None:
        self.active = False
        tasks = tuple(self.tasks)
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)

    async def dispatch(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        """MCP method handler; transport authentication stays in the service."""
        if not self.active:
            raise PermissionError("External tool capability has expired.")
        self.deadline.require_admission("external MCP request")
        if method == "tools/list":
            return {"tools": await self.bridge.mcp_tools(self.session_id)}
        if method != "tools/call":
            raise ValueError("Unsupported external tool method.")
        if type(params.get("name")) is not str or type(params.get("arguments", {})) is not dict:
            raise ValueError("MCP tool call requires a name and object arguments.")
        task = asyncio.current_task()
        assert task is not None
        self.tasks.add(task)
        try:
            async with execution_deadline_scope(self.deadline):
                result = await self.bridge.call(
                    params["name"],
                    params.get("arguments", {}),
                    parent_session_id=self.session_id,
                    execution_deadline=self.deadline,
                    call_id=f"mcp-{uuid4().hex}",
                )
        finally:
            self.tasks.discard(task)
        return {"content": [{"type": "text", "text": result.content}], "isError": result.is_error}
