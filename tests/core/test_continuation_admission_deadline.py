"""An admitted human continuation owns cleanup before exposing its first event."""

from __future__ import annotations

import asyncio
from datetime import timedelta

import pytest
from tests.core.test_execution_deadlines import clock as clock
from tests.core.test_run_limit_finish import spec
from tests.runtime.runtime_tool_support import SideEffectTool

from cayu import AgentSpec, CayuApp, EventType, Message, RunRequest
from cayu.approvals import ToolApprovalDecision, ToolApprovalRequest, UserInputResponse
from cayu.deadlines import ExecutionDeadline
from cayu.evals.testing import ScriptedModelProvider
from cayu.providers import ModelStreamEvent
from cayu.sessions import InMemorySessionStore, SessionStatus
from cayu.storage import SQLiteSessionStore, SQLiteTaskStore
from cayu.tasks import InMemoryTaskStore, TaskCreate, TaskStatus
from cayu.tools import AlwaysRequireApprovalToolPolicy, UserInputTool


@pytest.mark.store_semantics
@pytest.mark.parametrize("backend", ["memory", "sqlite"])
@pytest.mark.parametrize("gate", ["approval", "user_input"])
@pytest.mark.parametrize("delivery", ["expiry", "close"])
@pytest.mark.parametrize("structured", [False, True])
def test_human_continuation_cleanup_owns_first_admission_event(
    tmp_path, clock, backend, gate, delivery, structured
):
    async def scenario():
        store = (
            InMemorySessionStore()
            if backend == "memory"
            else SQLiteSessionStore(tmp_path / "sessions.db")
        )
        tasks = (
            InMemoryTaskStore() if backend == "memory" else SQLiteTaskStore(tmp_path / "tasks.db")
        )
        provider = ScriptedModelProvider(
            [
                [
                    ModelStreamEvent.tool_call(
                        id="call",
                        name="side_effect" if gate == "approval" else "ask_user",
                        arguments={} if gate == "approval" else {"question": "Continue?"},
                    ),
                    ModelStreamEvent.completed(),
                ]
            ]
        )
        tool = SideEffectTool()
        app = CayuApp(session_store=store, task_store=tasks, enable_logging=False)
        app.register_provider(provider, default=True)
        app.register_agent(
            AgentSpec(name="agent", model="fake"),
            tools=[tool, UserInputTool()],
            tool_policy=AlwaysRequireApprovalToolPolicy() if gate == "approval" else None,
        )
        try:
            await tasks.create_task(TaskCreate(task_id="task", type="test"))
            initial = [
                event
                async for event in app.run(
                    RunRequest(
                        agent_name="agent",
                        session_id="continuation",
                        task_id="task",
                        messages=[Message.text("user", "Wait")],
                        execution_deadline=ExecutionDeadline.after(3600),
                        structured_output=spec("tool") if structured else None,
                    )
                )
            ]
            assert initial[-1].type is EventType.SESSION_INTERRUPTED
            checkpoint = await store.load_checkpoint("continuation")
            if gate == "approval":
                pending = checkpoint["pending_tool_approval"]
                stream = app.resolve_tool_approval(
                    ToolApprovalRequest(
                        session_id="continuation",
                        approval_id=pending["approval_id"],
                        tool_round_id=pending["tool_round_id"],
                        tool_call_id=pending["tool_call_id"],
                        decision=ToolApprovalDecision.APPROVE,
                    )
                )
            else:
                stream = app.resolve_user_input(
                    UserInputResponse(
                        session_id="continuation",
                        input_id=checkpoint["pending_user_input"]["input_id"],
                        answer="Continue",
                    )
                )
            admitted = await anext(stream)
            assert admitted.type is EventType.INTERACTION_RESUMED
            epoch = (await store.load("continuation")).run_epoch
            admitted_event_count = len(await store.load_events("continuation"))
            if delivery == "expiry":
                clock.wall += timedelta(seconds=3601)
                clock.mono += 3601
                events = [event async for event in stream]
                assert events[-1].type is EventType.SESSION_INTERRUPTED
                assert events[-1].payload["limit"] == "execution_deadline"
                assert events[-1].payload["structured_output_pending"] is structured
                assert (await tasks.load_task("task")).status is TaskStatus.FAILED
            else:
                await stream.aclose()
            session = await store.load("continuation")
            assert session.status is SessionStatus.INTERRUPTED
            stored_events = await store.load_events("continuation")
            terminals = [
                event
                for event in stored_events[admitted_event_count:]
                if event.type is EventType.SESSION_INTERRUPTED
            ]
            assert len(terminals) == 1
            assert terminals[0].payload["failure_evidence"]["run_epoch"] == epoch
            assert len(provider.requests) == 1 and not tool.calls
        finally:
            await app.aclose()

    asyncio.run(scenario())
