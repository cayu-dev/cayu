"""Only the runtime marks a tool-policy request as a re-check of a call."""

from __future__ import annotations

import asyncio
from contextlib import aclosing

import pytest

from cayu import AgentSpec, CayuApp, Message
from cayu.approvals.authorization_grants import APPROVED_ARGUMENTS_REAUTHORIZATION_METADATA_KEY
from cayu.approvals.tools import ToolApprovalDecision, ToolApprovalRequest
from cayu.evals.testing import ScriptedModelProvider
from cayu.events import EventType
from cayu.observability.hooks import BeforeToolCallDecision, RuntimeHook
from cayu.providers import ModelStreamEvent
from cayu.sessions.memory import InMemorySessionStore
from cayu.sessions.requests import ForkSessionRequest, ResumeRequest, RunRequest
from cayu.tools.base import Tool, ToolContext, ToolResult, ToolSpec
from cayu.tools.policy import (
    TOOL_POLICY_DRY_RUN_METADATA_KEY,
    TOOL_POLICY_REAUTHORIZATION_METADATA_KEY,
    ToolPolicy,
    ToolPolicyDecision,
    ToolPolicyRequest,
    ToolPolicyResult,
)

_MARKERS = (
    TOOL_POLICY_REAUTHORIZATION_METADATA_KEY,
    APPROVED_ARGUMENTS_REAUTHORIZATION_METADATA_KEY,
)
_FORGED = {
    TOOL_POLICY_REAUTHORIZATION_METADATA_KEY: True,
    APPROVED_ARGUMENTS_REAUTHORIZATION_METADATA_KEY: True,
    "ticket": "T-1",
}


class _Echo(Tool):
    spec = ToolSpec(
        name="echo",
        description="Echo text.",
        input_schema={
            "type": "object",
            "properties": {"text": {"type": "string"}},
            "required": ["text"],
        },
    )

    def __init__(self) -> None:
        self.calls: list[dict] = []

    async def run(self, ctx: ToolContext, args: dict) -> ToolResult:
        self.calls.append(dict(args))
        return ToolResult(content=args["text"])


class _RecordingPolicy(ToolPolicy):
    """Record each request's metadata; send ``gated`` text to a person."""

    def __init__(self) -> None:
        self.seen: list[tuple[object, dict]] = []

    async def authorize(self, request: ToolPolicyRequest) -> ToolPolicyResult:
        self.seen.append((request.arguments.get("text"), dict(request.metadata)))
        if request.arguments.get("text") == "gated":
            return ToolPolicyResult(decision=ToolPolicyDecision.REQUIRE_APPROVAL)
        return ToolPolicyResult(decision=ToolPolicyDecision.ALLOW)


class _Rewrite(RuntimeHook):
    """Change ``rewrite`` arguments, so the runtime re-checks the call."""

    async def before_tool_call(self, context) -> BeforeToolCallDecision | None:
        if context.arguments.get("text") != "rewrite":
            return None
        return BeforeToolCallDecision(
            action="proceed_modified", modified_arguments={"text": "rewritten"}
        )


def _call(text: str) -> list[ModelStreamEvent]:
    return [
        ModelStreamEvent.tool_call(id=f"call-{text}", name="echo", arguments={"text": text}),
        ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
    ]


_DONE = [ModelStreamEvent.text_delta("Done."), ModelStreamEvent.completed({})]


def _app(turns) -> tuple[CayuApp, _Echo, _RecordingPolicy]:
    tool, policy = _Echo(), _RecordingPolicy()
    app = CayuApp(session_store=InMemorySessionStore(), enable_logging=False)
    app.register_provider(ScriptedModelProvider(turns), default=True)
    app.register_agent(
        AgentSpec(name="assistant", model="scripted-model"),
        tools=[tool],
        tool_policy=policy,
        runtime_hooks=[_Rewrite()],
    )
    return app, tool, policy


async def _drain(stream) -> list:
    async with aclosing(stream):
        return [event async for event in stream]


@pytest.mark.parametrize(
    ("key", "authority"),
    [
        (TOOL_POLICY_REAUTHORIZATION_METADATA_KEY, "tool-policy reauthorization authority"),
        (APPROVED_ARGUMENTS_REAUTHORIZATION_METADATA_KEY, "approved-arguments reauthorization"),
    ],
)
@pytest.mark.parametrize("value", [True, False])
def test_run_and_fork_requests_refuse_a_reauthorization_marker(key, authority, value) -> None:
    for build in (
        lambda: RunRequest(
            agent_name="assistant",
            messages=[Message.text("user", "Hi")],
            metadata={key: value},
        ),
        lambda: ForkSessionRequest(source_session_id="chat", metadata={key: value}),
    ):
        with pytest.raises(ValueError, match=f"{key}.*runtime-owned {authority}"):
            build()


def test_forged_markers_never_reach_the_policy_and_the_runtime_still_sets_its_own() -> None:
    async def scenario():
        app, tool, policy = _app(
            [
                _call("rewrite"),
                _DONE,
                _call("gated"),
                _DONE,
            ]
        )
        # The runtime marks its own re-check of hook-changed arguments.
        await _drain(
            app.run(
                RunRequest(
                    agent_name="assistant",
                    session_id="chat",
                    messages=[Message.text("user", "Rewrite it.")],
                )
            )
        )
        assert [(text, [m.get(k) for k in _MARKERS]) for text, m in policy.seen] == [
            ("rewrite", [None, None]),
            ("rewritten", [True, None]),
        ]
        # A resume's metadata reaches the policy, but admission drops forged markers, so a
        # fresh call goes to a person as a first check.
        policy.seen.clear()
        events = await _drain(
            app.resume(
                ResumeRequest(
                    session_id="chat",
                    messages=[Message.text("user", "Gate it.")],
                    metadata=_FORGED,
                )
            )
        )
        [(text, fresh)] = policy.seen
        assert text == "gated" and fresh.get("ticket") == "T-1"
        assert not any(key in fresh for key in _MARKERS)
        pause = next(e for e in events if e.type is EventType.TOOL_CALL_APPROVAL_REQUESTED)
        # Approving with changed arguments re-checks them with both runtime markers.
        policy.seen.clear()
        resolved = await _drain(
            app.resolve_tool_approval(
                ToolApprovalRequest(
                    session_id="chat",
                    approval_id=pause.payload["approval_id"],
                    tool_round_id=pause.payload["tool_round_id"],
                    tool_call_id=pause.payload["tool_call_id"],
                    decision=ToolApprovalDecision.APPROVE,
                    arguments={"text": "approved"},
                )
            )
        )
        assert resolved[-1].type is EventType.SESSION_COMPLETED
        recheck = policy.seen[0][1]
        assert policy.seen[0][0] == "approved"
        assert [recheck.get(key) for key in _MARKERS] == [True, True]
        assert recheck.get("ticket") == "T-1"
        assert tool.calls == [{"text": "rewritten"}, {"text": "approved"}]

        # The runtime never stores its markers on the session, so a fork is not refused.
        source = await app.session_store.load("chat")
        assert source is not None
        assert not any(key in source.metadata for key in _MARKERS)
        await _drain(
            app.fork_session(ForkSessionRequest(source_session_id="chat", session_id="forked"))
        )
        forked = await app.session_store.load("forked")
        assert forked is not None
        assert not any(key in forked.metadata for key in _MARKERS)

        # A preview's metadata is dropped the same way; only its dry-run marker is set.
        policy.seen.clear()
        preview = await app.preview_tool_call(
            "assistant", "echo", {"text": "plain"}, metadata=_FORGED
        )
        await app.aclose()
        return preview, policy.seen

    preview, seen = asyncio.run(scenario())
    assert preview.decision == "allow"
    [(_text, previewed)] = seen
    assert previewed[TOOL_POLICY_DRY_RUN_METADATA_KEY] is True
    assert not any(key in previewed for key in _MARKERS)
