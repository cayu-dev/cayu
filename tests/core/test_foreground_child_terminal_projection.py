"""A reconstructed parent preserves the child's ordinary terminal projection."""

import asyncio

import pytest
from tests.core.test_foreground_subagent_recovery import _identity, _Provider
from tests.core.test_tool_round_execution_identities import _RecordingTool

from cayu.agents import AgentSpec
from cayu.applications import CayuApp
from cayu.approvals.tools import ToolApprovalDecision, ToolApprovalRequest
from cayu.approvals.user_input import UserInputResponse
from cayu.messages import Message
from cayu.providers.base import ModelStreamEvent
from cayu.sessions.base import InMemorySessionStore, RunRequest, SessionQuery, SessionStatus
from cayu.storage.sqlite import SQLiteSessionStore
from cayu.tools.policy import AlwaysRequireApprovalToolPolicy
from cayu.tools.subagents import SubagentSpec, SubagentTool
from cayu.tools.user_input import UserInputTool


class _VersionedRecordingTool(_RecordingTool):
    spec = _RecordingTool.spec.model_copy(
        update={"execution_profile_identity": _identity("terminal-projection-record")}
    )


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
@pytest.mark.parametrize("action", ["approval", "input"])
@pytest.mark.parametrize("outcome", ["success", "failure", "step_limit", "parent_step_limit"])
def test_reconstructed_child_terminal_outcome_preserves_projection(
    tmp_path, backend, action, outcome
):
    async def scenario():
        path = tmp_path / "terminal-projection.sqlite"
        store = InMemorySessionStore() if backend == "memory" else SQLiteSessionStore(path)
        protected = _VersionedRecordingTool()
        answer = "Unicode 😀é日本語 " * 4
        final_child_batch = (
            [ModelStreamEvent.error("expected child failure")]
            if outcome == "failure"
            else [ModelStreamEvent.text_delta(answer), ModelStreamEvent.completed()]
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
                        id="action",
                        name="ask_user" if action == "input" else "record",
                        arguments={"question": "Continue?"} if action == "input" else {"value": 7},
                    ),
                    ModelStreamEvent.completed(),
                ],
                final_child_batch,
                [ModelStreamEvent.text_delta("parent finished"), ModelStreamEvent.completed()],
            ]
        )

        def make_app(store):
            app = CayuApp(session_store=store, enable_logging=False)
            app.register_provider(provider, default=True)
            subagent = SubagentTool(
                app,
                agents={
                    "child": SubagentSpec(
                        agent_name="child",
                        max_steps=1 if outcome == "step_limit" else 5,
                        result_max_chars=12,
                    )
                },
                execution_profile_identity=_identity("nested-terminal-projection"),
            )
            app.register_agent(AgentSpec(name="parent", model="test"), tools=[subagent])
            app.register_agent(
                AgentSpec(name="child", model="test"),
                tools=[UserInputTool()] if action == "input" else [protected],
                tool_policy=None
                if action == "input"
                else AlwaysRequireApprovalToolPolicy(tools=["record"]),
            )
            return app, subagent

        app, _ = make_app(store)
        try:
            paused = [
                event
                async for event in app.run(
                    RunRequest(
                        session_id="parent",
                        agent_name="parent",
                        max_steps=1 if outcome == "parent_step_limit" else 16,
                        messages=[Message.text("user", "go")],
                    )
                )
            ]
            assert paused[-1].type == "session.interrupted"
            original = await store.load_checkpoint("parent")
            assert original is not None
            wait = original["foreground_child_wait"]
            assert len(provider.requests) == 2 and protected.values == []
            assert await app.drain_background_interruptions(timeout_s=10)
            if isinstance(store, SQLiteSessionStore):
                await store.close()
                store = SQLiteSessionStore(path)
            app, subagent = make_app(store)
            children = await store.list_sessions(SessionQuery(parent_session_id="parent"))
            assert len(children.sessions) == 1
            child = children.sessions[0]
            if action == "input":
                stream = app.resolve_user_input(
                    UserInputResponse(
                        session_id=child.id, input_id=wait["child_action_id"], answer="accepted"
                    )
                )
            else:
                child_checkpoint = await store.load_checkpoint(child.id)
                assert child_checkpoint is not None
                approval = child_checkpoint["pending_tool_approval"]
                stream = app.resolve_tool_approval(
                    ToolApprovalRequest(
                        session_id=child.id,
                        approval_id=approval["approval_id"],
                        tool_round_id=approval["tool_round_id"],
                        tool_call_id=approval["tool_call_id"],
                        decision=ToolApprovalDecision.APPROVE,
                    )
                )
            _ = [event async for event in stream]
            assert await app.drain_background_interruptions(timeout_s=10)
            parent = await store.load("parent")
            finished_child = await store.load(child.id)
            expected_parent_status = (
                SessionStatus.INTERRUPTED
                if outcome == "parent_step_limit"
                else SessionStatus.COMPLETED
            )
            assert parent is not None and parent.status is expected_parent_status, "\n".join(
                [
                    str(event.payload.get("error") or event.payload.get("reason") or event.type)
                    for owner in ("parent", child.id)
                    for event in (await store.load_events(owner))[-4:]
                ]
            )
            assert finished_child is not None
            assert (
                finished_child.status
                is {
                    "success": SessionStatus.COMPLETED,
                    "failure": SessionStatus.FAILED,
                    "step_limit": SessionStatus.INTERRUPTED,
                    "parent_step_limit": SessionStatus.COMPLETED,
                }[outcome]
            )
            assert len(provider.requests) == (
                3 if outcome in {"step_limit", "parent_step_limit"} else 4
            )
            assert protected.values == ([7] if action == "approval" else [])
            events = await store.load_events("parent")
            tool_events = [
                event
                for event in events
                if event.type in {"tool.call.completed", "tool.call.failed"}
            ]
            assert len(tool_events) == 1
            projection = await subagent.project_recoverable_child(finished_child)
            assert projection is not None
            assert tool_events[0].payload["result"] == projection.model_dump(mode="json")
            assert projection.is_error is (outcome in {"failure", "step_limit"})
            if outcome in {"success", "parent_step_limit"}:
                assert projection.content == answer[:12]
                assert projection.structured["result_truncated"] is True
            assert sum(event.type == "interaction.started" for event in events) == 1
            closing_type = (
                "interaction.interrupted"
                if outcome == "parent_step_limit"
                else "interaction.completed"
            )
            assert sum(event.type == closing_type for event in events) == 1
            if outcome in {"step_limit", "parent_step_limit"}:
                limited_events = (
                    events if outcome == "parent_step_limit" else await store.load_events(child.id)
                )
                limits = [
                    event for event in limited_events if event.type == "session.limit_reached"
                ]
                assert len(limits) == 1
                assert limits[0].payload["limit"] == "model_steps"
                assert limits[0].payload["actual"] == limits[0].payload["maximum"] == 1
            assert (
                len(
                    [
                        message
                        for message in await store.load_transcript("parent")
                        if message.role == "tool"
                    ]
                )
                == 1
            )
            await app.recover_persisted_event_side_effects()
            assert await store.load_events("parent") == events
        finally:
            assert await app.drain_background_interruptions(timeout_s=10)
            if isinstance(store, SQLiteSessionStore):
                await store.close()

    asyncio.run(scenario())
