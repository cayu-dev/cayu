"""The changed-arguments notice survives every last step of terminal publication."""

from __future__ import annotations

import asyncio
from contextlib import aclosing

from cayu import AgentSpec, CayuApp, Message, RunRequest
from cayu.approvals.tools import ToolApprovalDecision, ToolApprovalRequest
from cayu.artifacts import LocalArtifactStore
from cayu.environments import Environment, EnvironmentSpec
from cayu.evals import ScriptedModelProvider
from cayu.events import EventType
from cayu.messages import ToolResultPart
from cayu.observability.hooks import AfterToolCallDecision, RuntimeHook
from cayu.providers import ModelStreamEvent
from cayu.runtime._tool_execution import APPROVED_ARGUMENTS_NOTICE
from cayu.runtime._tool_invocation.terminal import _durable_payload_utf8_size
from cayu.sessions.memory import InMemorySessionStore
from cayu.tools.base import Tool, ToolContext, ToolResult, ToolSpec
from cayu.tools.policy import ToolPolicy, ToolPolicyDecision, ToolPolicyResult
from cayu.tools.result_projection import ArtifactExternalizingToolResultPolicy

_LIMIT = 65_536


def _content_at_limit() -> str:
    """Content whose result payload is exactly the tool's terminal limit."""

    content = "x" * _LIMIT
    while _durable_payload_utf8_size(ToolResult(content=content).model_dump(mode="json")) > _LIMIT:
        content = content[:-1]
    assert _durable_payload_utf8_size(ToolResult(content=content).model_dump(mode="json")) == _LIMIT
    return content


class _ResultTool(Tool):
    spec = ToolSpec(
        name="send",
        description="Send a message and report the result.",
        input_schema={"type": "object", "properties": {"to": {"type": "string"}}},
        max_terminal_payload_bytes=_LIMIT,
    )

    def __init__(self, content: str) -> None:
        super().__init__()
        self.content = content
        self.calls: list[dict] = []

    async def run(self, ctx: ToolContext, args: dict) -> ToolResult:
        self.calls.append(dict(args))
        return ToolResult(content=self.content)


class _ApprovalPolicy(ToolPolicy):
    async def authorize(self, request):
        return ToolPolicyResult(decision=ToolPolicyDecision.REQUIRE_APPROVAL)


class _ReplacingHook(RuntimeHook):
    async def after_tool_call(self, context):
        return AfterToolCallDecision(action="modify", modified_result=ToolResult(content="Sent"))


def _app(tool: Tool, tmp_path=None, *, hooks=(), externalize=False) -> CayuApp:
    app = CayuApp(
        session_store=InMemorySessionStore(),
        enable_logging=False,
        runtime_hooks=list(hooks),
        tool_result_projection_policy=(
            ArtifactExternalizingToolResultPolicy(
                max_inline_bytes=2048, max_inline_token_estimate=None, preview_bytes=0
            )
            if externalize
            else None
        ),
    )
    app.register_provider(
        ScriptedModelProvider(
            [
                [
                    ModelStreamEvent.tool_call(id="pause", name="send", arguments={"to": "ops"}),
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
    if externalize:
        app.register_environment(
            Environment(
                EnvironmentSpec(name="local"),
                artifact_store=LocalArtifactStore(tmp_path / "artifacts"),
            ),
            default=True,
        )
    app.register_agent(
        AgentSpec(name="assistant", model="scripted-model"),
        tools=[tool],
        tool_policy=_ApprovalPolicy(),
    )
    return app


async def _drain(stream):
    async with aclosing(stream):
        return [event async for event in stream]


async def _edit_and_resolve(app: CayuApp, session_id: str, *, edit: bool = True):
    events = await _drain(
        app.run(
            RunRequest(
                agent_name="assistant",
                session_id=session_id,
                messages=[Message.text("user", "send it")],
            )
        )
    )
    paused = next(e for e in events if e.type is EventType.TOOL_CALL_APPROVAL_REQUESTED)
    await _drain(
        app.resolve_tool_approval(
            ToolApprovalRequest(
                session_id=session_id,
                approval_id=paused.payload["approval_id"],
                tool_round_id=paused.payload["tool_round_id"],
                tool_call_id=paused.payload["tool_call_id"],
                decision=ToolApprovalDecision.APPROVE,
                arguments={"to": "finance"} if edit else None,
            )
        )
    )
    store = app.session_store
    [terminal] = [
        e
        for e in await store.load_events(session_id)
        if e.type in {EventType.TOOL_CALL_COMPLETED, EventType.TOOL_CALL_FAILED}
    ]
    [result] = [
        part
        for message in await store.load_transcript(session_id)
        for part in message.content
        if type(part) is ToolResultPart
    ]
    return terminal, result


def test_a_result_at_its_limit_stays_valid_when_the_call_was_edited():
    content = _content_at_limit()

    async def scenario():
        unedited = await _edit_and_resolve(_app(_ResultTool(content)), "at-limit", edit=False)
        edited = await _edit_and_resolve(_app(_ResultTool(content)), "at-limit-edited")
        return unedited, edited

    (plain, _plain_result), (terminal, result) = asyncio.run(scenario())

    assert plain.type is EventType.TOOL_CALL_COMPLETED
    # The runtime's notice has its own allowance beyond the tool's limit.
    assert terminal.type is EventType.TOOL_CALL_COMPLETED
    assert terminal.payload.get("terminal_outcome") != "invalid_tool_output"
    assert result.content == f"{APPROVED_ARGUMENTS_NOTICE}\n\n{content}"


def test_a_hook_replacement_keeps_the_notice():
    tool = _ResultTool("delivered")

    async def scenario():
        return await _edit_and_resolve(_app(tool, hooks=[_ReplacingHook()]), "hook-replaced")

    terminal, result = asyncio.run(scenario())

    assert tool.calls == [{"to": "finance"}]
    assert result.content == f"{APPROVED_ARGUMENTS_NOTICE}\n\nSent"
    assert terminal.payload["result"]["content"] == result.content


def test_externalizing_the_result_without_a_preview_keeps_the_notice(tmp_path):
    tool = _ResultTool("y" * 10_000)

    async def scenario():
        return await _edit_and_resolve(_app(tool, tmp_path, externalize=True), "externalized")

    terminal, result = asyncio.run(scenario())

    assert "tool_result_projection" in terminal.payload
    assert result.content.startswith(APPROVED_ARGUMENTS_NOTICE)
    assert terminal.payload["result"]["content"] == result.content
