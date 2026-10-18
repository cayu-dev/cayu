"""An edited external call's staged and durable terminals stay byte-identical.

A registered secret that overlaps the runtime notice redacts it inside the
staged terminal; publication must never rewrite that staged evidence, or a
retry after a lost terminal acknowledgement finds conflicting evidence.
"""

from __future__ import annotations

import asyncio
import base64
from contextlib import aclosing

import pytest
from pydantic import SecretStr

from cayu import AgentSpec, CayuApp, Message, RunRequest
from cayu.approvals.tools import ToolApprovalDecision, ToolApprovalRequest
from cayu.evals import ScriptedModelProvider
from cayu.events import EventType
from cayu.messages import ToolResultPart
from cayu.observability.hooks import RuntimeHook
from cayu.providers import ModelStreamEvent
from cayu.runtime.public_authority import PublicAuthorityAliasCodec, PublicAuthorityAliasKeyring
from cayu.sessions.memory import InMemorySessionStore
from cayu.tools.base import Tool, ToolContext, ToolEffect, ToolResult, ToolSpec
from cayu.tools.policy import ToolPolicy, ToolPolicyDecision, ToolPolicyResult
from cayu.vaults.redaction import SecretRedactor

# "person" is one word of the runtime notice.
_SECRET = "person"


class _LostTerminalAckStore(InMemorySessionStore):
    """The first terminal append commits, then its acknowledgement is lost."""

    invocation_lifecycle_command_version = 1

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.lost = False

    async def append_events(self, session_id, events):
        await super().append_events(session_id, events)
        if not self.lost and any(
            event.type in {EventType.TOOL_CALL_COMPLETED, EventType.TOOL_CALL_FAILED}
            for event in events
        ):
            self.lost = True
            raise ConnectionError("terminal acknowledgement lost")


class _ExternalSendTool(Tool):
    spec = ToolSpec(
        name="send",
        description="Send a message.",
        input_schema={"type": "object", "properties": {"to": {"type": "string"}}},
        effect=ToolEffect.EXTERNAL,
    )

    def __init__(self) -> None:
        super().__init__()
        self.calls: list[dict] = []

    async def run(self, ctx: ToolContext, args: dict) -> ToolResult:
        self.calls.append(dict(args))
        return ToolResult(content="sent")


class _ApprovalPolicy(ToolPolicy):
    async def authorize(self, request):
        return ToolPolicyResult(decision=ToolPolicyDecision.REQUIRE_APPROVAL)


class _ObservingHook(RuntimeHook):
    """An after-tool hook routes staged publication through the hook-finalizing path."""

    async def after_tool_call(self, context):
        return None


def _app(store, *, hooks=()) -> tuple[CayuApp, _ExternalSendTool]:
    tool = _ExternalSendTool()
    app = CayuApp(
        session_store=store,
        enable_logging=False,
        secret_redactor=SecretRedactor(_SECRET),
        runtime_hooks=list(hooks),
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
    app.register_agent(
        AgentSpec(name="assistant", model="scripted-model"),
        tools=[tool],
        tool_policy=_ApprovalPolicy(),
    )
    return app, tool


async def _drain(stream):
    async with aclosing(stream):
        return [event async for event in stream]


@pytest.mark.parametrize("hooks", ["emit", "emit_staged"])
def test_a_lost_terminal_acknowledgement_retries_without_staged_conflict(hooks):
    key = base64.urlsafe_b64encode(bytes([61]) * 32).decode().rstrip("=")
    store = _LostTerminalAckStore(
        public_authority_alias_codec=PublicAuthorityAliasCodec(
            PublicAuthorityAliasKeyring(active_key_id="test", keys={"test": SecretStr(key)})
        )
    )
    app, tool = _app(store, hooks=[_ObservingHook()] if hooks == "emit" else ())

    async def scenario():
        events = await _drain(
            app.run(
                RunRequest(
                    agent_name="assistant",
                    session_id=f"edit-staged-{hooks}",
                    messages=[Message.text("user", "send it")],
                )
            )
        )
        paused = next(e for e in events if e.type is EventType.TOOL_CALL_APPROVAL_REQUESTED)
        request = ToolApprovalRequest(
            session_id=f"edit-staged-{hooks}",
            approval_id=paused.payload["approval_id"],
            tool_round_id=paused.payload["tool_round_id"],
            tool_call_id=paused.payload["tool_call_id"],
            decision=ToolApprovalDecision.APPROVE,
            arguments={"to": "finance"},
        )
        first = await _drain(app.resolve_tool_approval(request))
        assert store.lost and first[-1].type is EventType.SESSION_INTERRUPTED
        retried = await _drain(app.resolve_tool_approval(request))
        assert retried[-1].type is EventType.SESSION_COMPLETED, retried[-1].payload
        terminals = [
            e
            for e in await store.load_events(request.session_id)
            if e.type in {EventType.TOOL_CALL_COMPLETED, EventType.TOOL_CALL_FAILED}
        ]
        results = [
            part
            for message in await store.load_transcript(request.session_id)
            for part in message.content
            if type(part) is ToolResultPart
        ]
        return terminals, results

    terminals, results = asyncio.run(scenario())

    assert tool.calls == [{"to": "finance"}]
    [terminal] = terminals
    assert terminal.payload["approved_with_changes"] is True
    notice = SecretRedactor(_SECRET).redact_text(
        "A person changed this call's arguments before approving it"
    )
    [result] = results
    assert result.content.count(notice) == 1
    assert result.content.startswith(notice)
    assert terminal.payload["result"]["content"] == result.content
