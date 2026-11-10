"""A foreground child pause does not discard the parent's after-tool phase."""

import asyncio

import pytest
from tests.core.test_foreground_subagent_recovery import _identity, _Provider

from cayu import AgentSpec, CayuApp, EventType, Message, RunRequest
from cayu.approvals.tools import ToolApprovalRequest
from cayu.approvals.user_input import UserInputResponse
from cayu.observability.hooks import RuntimeHook
from cayu.providers.base import ModelStreamEvent
from cayu.sessions.memory import InMemorySessionStore
from cayu.sessions.queries import SessionQuery
from cayu.sessions.records import SessionStatus
from cayu.tools.policy import AlwaysRequireApprovalToolPolicy
from cayu.tools.subagents import SubagentSpec, SubagentTool
from cayu.tools.user_input import UserInputTool


class _ObserveParentResult(RuntimeHook):
    def __init__(self):
        self.results = []

    @property
    def execution_profile_identity(self):
        return _identity("observe-parent-result")

    async def after_tool_call(self, context):
        if context.tool_name == "subagent":
            self.results.append(context.result)


@pytest.mark.parametrize("gate", ["approval", "input"])
@pytest.mark.parametrize("child_pauses", [False, True])
def test_foreground_gate_recovery_runs_parent_after_tool_hook(gate, child_pauses):
    asyncio.run(_scenario(gate, child_pauses))


async def _scenario(gate, child_pauses):
    store = InMemorySessionStore()
    calls = [
        ModelStreamEvent.tool_call(
            id="delegate", name="subagent", arguments={"agent": "child", "task": "work"}
        )
    ]
    if gate == "input":
        calls.append(
            ModelStreamEvent.tool_call(
                id="parent-question", name="ask_user", arguments={"question": "Start?"}
            )
        )
    child_question = [
        ModelStreamEvent.tool_call(
            id="child-question", name="ask_user", arguments={"question": "Continue?"}
        ),
        ModelStreamEvent.completed(),
    ]
    provider = _Provider(
        [[*calls, ModelStreamEvent.completed()]]
        + ([child_question] if child_pauses else [])
        + [
            [ModelStreamEvent.text_delta("child done"), ModelStreamEvent.completed()],
            [ModelStreamEvent.text_delta("parent done"), ModelStreamEvent.completed()],
        ]
    )
    observer = _ObserveParentResult()
    app = CayuApp(session_store=store, enable_logging=False)
    app.register_provider(provider, default=True)
    app.register_agent(
        AgentSpec(name="parent", model="test"),
        tools=[
            SubagentTool(
                app,
                agents={"child": SubagentSpec(agent_name="child")},
                execution_profile_identity=_identity("hook-test-subagent"),
            ),
            UserInputTool(),
        ],
        runtime_hooks=[observer],
        tool_policy=AlwaysRequireApprovalToolPolicy(tools=["subagent"])
        if gate == "approval"
        else None,
    )
    app.register_agent(AgentSpec(name="child", model="test"), tools=[UserInputTool()])
    try:
        initial = [
            event
            async for event in app.run(
                RunRequest(
                    session_id="parent",
                    agent_name="parent",
                    messages=[Message.text("user", "go")],
                )
            )
        ]
        if gate == "approval":
            approval = next(e for e in initial if e.type is EventType.TOOL_CALL_APPROVAL_REQUESTED)
            stream = app.resolve_tool_approval(
                ToolApprovalRequest(
                    session_id="parent",
                    approval_id=approval.payload["approval"]["approval_id"],
                    tool_round_id=approval.payload["tool_round_id"],
                    tool_call_id=approval.payload["tool_call_id"],
                    decision="approve",
                )
            )
        else:
            pause = next(e for e in initial if e.type is EventType.SESSION_AWAITING_USER_INPUT)
            stream = app.resolve_user_input(
                UserInputResponse(
                    session_id="parent", input_id=pause.payload["input_id"], answer="yes"
                )
            )
        resumed = [event async for event in stream]
        if child_pauses:
            assert any(
                e.payload.get("interruption_type") == "waiting_on_child_action" for e in resumed
            )
            [child] = (await store.list_sessions(SessionQuery(parent_session_id="parent"))).sessions
            checkpoint = await store.load_checkpoint(child.id)
            [
                event
                async for event in app.resolve_user_input(
                    UserInputResponse(
                        session_id=child.id,
                        input_id=checkpoint["pending_user_input"]["input_id"],
                        answer="yes",
                    )
                )
            ]
        assert (await store.load("parent")).status is SessionStatus.COMPLETED
        terminals = [
            e
            for e in await store.load_events("parent")
            if e.type is EventType.TOOL_CALL_COMPLETED
            and e.payload.get("tool_call_id") == "delegate"
        ]
        assert len(terminals) == 1
        assert len(observer.results) == 1, "The completed parent call never ran its after-tool hook"
        assert "child done" in observer.results[0].content
    finally:
        await app.aclose()
