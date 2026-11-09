"""Limit closure must seal already recovered calls in a quarantined approval round."""

from datetime import UTC, datetime, timedelta

import pytest
from tests.core.test_foreground_subagent_recovery import _identity, _Provider

from cayu import AgentSpec, CayuApp, EventType, Message, RunRequest, SessionStatus
from cayu.approvals.tools import ToolApprovalDecision, ToolApprovalRequest
from cayu.approvals.user_input import UserInputResponse
from cayu.budgets.run_limits import RunLimits
from cayu.providers.base import ModelStreamEvent
from cayu.sessions.memory import InMemorySessionStore
from cayu.sessions.queries import SessionQuery
from cayu.storage.sqlite import SQLiteSessionStore
from cayu.tools.policy import AlwaysRequireApprovalToolPolicy
from cayu.tools.subagents import SubagentSpec, SubagentTool
from cayu.tools.user_input import UserInputTool


@pytest.mark.anyio
@pytest.mark.parametrize("backend", ["memory", "sqlite"])
async def test_parent_limit_closes_approval_after_recovering_child(tmp_path, backend):
    now = [datetime.now(UTC)]
    store = (
        InMemorySessionStore()
        if backend == "memory"
        else SQLiteSessionStore(tmp_path / "limited-parent.sqlite")
    )
    provider = _Provider(
        [
            [
                ModelStreamEvent.tool_call(
                    id="spawn", name="subagent", arguments={"agent": "child", "task": "work"}
                ),
                ModelStreamEvent.completed(),
            ],
            [
                ModelStreamEvent.tool_call(
                    id="question", name="ask_user", arguments={"question": "Continue?"}
                ),
                ModelStreamEvent.completed(),
            ],
            [ModelStreamEvent.text_delta("child done"), ModelStreamEvent.completed()],
        ]
    )
    app = CayuApp(session_store=store, enable_logging=False, clock=lambda: now[0])
    app.register_provider(provider, default=True)
    app.register_agent(
        AgentSpec(name="parent", model="test"),
        tools=[
            SubagentTool(
                app,
                agents={"child": SubagentSpec(agent_name="child")},
                execution_profile_identity=_identity("limited-parent-gate"),
            )
        ],
        tool_policy=AlwaysRequireApprovalToolPolicy(tools=["subagent"]),
    )
    app.register_agent(AgentSpec(name="child", model="test"), tools=[UserInputTool()])
    try:
        _ = [
            event
            async for event in app.run(
                RunRequest(
                    session_id="parent",
                    agent_name="parent",
                    messages=[Message.text("user", "go")],
                    limits=RunLimits(max_elapsed_seconds=600, scope="session"),
                )
            )
        ]
        pending = (await store.load_checkpoint("parent"))["pending_tool_approval"]
        _ = [
            event
            async for event in app.resolve_tool_approval(
                ToolApprovalRequest(
                    session_id="parent",
                    approval_id=pending["approval_id"],
                    tool_round_id=pending["tool_round_id"],
                    tool_call_id=pending["tool_call_id"],
                    decision=ToolApprovalDecision.APPROVE,
                )
            )
        ]
        assert "foreground_child_wait" in await store.load_checkpoint("parent")
        child = (await store.list_sessions(SessionQuery(parent_session_id="parent"))).sessions[0]
        question = (await store.load_checkpoint(child.id))["pending_user_input"]
        # Session-scope elapsed limits include the human wait, unlike run-scope limits.
        now[0] = (await store.load("parent")).created_at + timedelta(seconds=601)
        _ = [
            event
            async for event in app.resolve_user_input(
                UserInputResponse(session_id=child.id, input_id=question["input_id"], answer="yes")
            )
        ]
        assert await app.drain_background_interruptions(timeout_s=10)
        await app.recover_persisted_event_side_effects()
        assert (await store.load(child.id)).status is SessionStatus.COMPLETED
        events = await store.load_events("parent")
        assert any(event.type is EventType.TOOL_CALL_COMPLETED for event in events)
        assert any(event.type is EventType.SESSION_LIMIT_REACHED for event in events)
        stops = [
            event
            for event in events
            if event.type is EventType.SESSION_INTERRUPTED
            and event.payload.get("reason") == "limit_reached"
        ]
        errors = [event.payload["error"] for event in events if "error" in event.payload]
        assert stops, f"Limit closure failed instead of settling the approval: {errors}"
        assert (await store.load("parent")).status is SessionStatus.INTERRUPTED
        assert "pending_tool_approval" not in await store.load_checkpoint("parent")
        assert len(provider.requests) == 3  # No redispatch of the child or parent model.
    finally:
        await app.drain_background_interruptions(timeout_s=10)
        if isinstance(store, SQLiteSessionStore):
            await store.close()
