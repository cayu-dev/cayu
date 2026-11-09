"""Approval continuations own the publication of a subsequent approval pause."""

import asyncio

import pytest
from tests.core.test_continuation_deadlines import skip_ahead as skip_ahead
from tests.core.test_tool_completion import FinalTool

from cayu import AgentSpec, CayuApp, EventType, Message, RunRequest, SessionStatus
from cayu.approvals import ToolApprovalDecision, ToolApprovalRequest
from cayu.deadlines import ExecutionDeadline, default_run_deadline
from cayu.evals import ScriptedModelProvider
from cayu.providers import ModelStreamEvent
from cayu.sessions import InMemorySessionStore
from cayu.storage.sqlite import SQLiteSessionStore
from cayu.tools.policy import AlwaysRequireApprovalToolPolicy


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
@pytest.mark.parametrize(
    "close_after", [None, EventType.INTERACTION_PAUSED, EventType.TURN_COMPLETED]
)
@pytest.mark.parametrize("wait_seconds", [0, 3600], ids=["immediate", "after-human-wait"])
def test_repeated_approval_pause_close_finishes_publication(
    tmp_path, skip_ahead, backend, close_after, wait_seconds
):
    async def scenario():
        store = (
            InMemorySessionStore()
            if backend == "memory"
            else SQLiteSessionStore(tmp_path / "repeated-pause-close.sqlite")
        )
        provider = ScriptedModelProvider(
            [
                [
                    ModelStreamEvent.tool_call(name="ask_customer", arguments={}, id=call_id),
                    ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
                ]
                for call_id in ("first-question", "second-question")
            ]
            + [
                [
                    ModelStreamEvent.text_delta("done"),
                    ModelStreamEvent.completed({"finish_reason": "stop"}),
                ]
            ]
        )
        tool = FinalTool()
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_provider(provider, default=True)
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            tools=[tool],
            tool_policy=AlwaysRequireApprovalToolPolicy(),
        )

        async def approval_request():
            pending = (await store.load_checkpoint("repeated-pause-close"))["pending_tool_approval"]
            return ToolApprovalRequest(
                session_id="repeated-pause-close",
                approval_id=pending["approval_id"],
                tool_round_id=pending["tool_round_id"],
                tool_call_id=pending["tool_call_id"],
                decision=ToolApprovalDecision.APPROVE,
            )

        try:
            with default_run_deadline(ExecutionDeadline.after(60, source="task_worker")):
                first_pause = [
                    event
                    async for event in app.run(
                        RunRequest(
                            agent_name="assistant",
                            session_id="repeated-pause-close",
                            messages=[Message.text("user", "Ask twice")],
                        )
                    )
                ]
            assert first_pause[-1].type is EventType.SESSION_INTERRUPTED
            # Unlike app.run, approval resolution enters through the recovered
            # session path, which has no RunLedger to finish publication for it.
            stream = app.resolve_tool_approval(await approval_request())
            seen = []
            try:
                async for event in stream:
                    seen.append(event.type)
                    if event.type is close_after:
                        break
            finally:
                await stream.aclose()
            if close_after is not None:
                assert seen[-1] is close_after
            assert await app.drain_recovery_cleanups(timeout_s=5)
            assert await app.drain_environment_cleanups(timeout_s=5)
            paused = await store.load("repeated-pause-close")
            assert paused.status is SessionStatus.INTERRUPTED
            assert paused.execution_deadline.origin == "run_default"
            assert tool.calls == 1
            durable = await store.load_events("repeated-pause-close")
            interrupted = [
                event for event in durable if event.type is EventType.SESSION_INTERRUPTED
            ]
            assert len(interrupted) == 2, [event.type for event in durable]
            assert all(
                event.payload["interruption_type"] == "tool_approval_required"
                for event in interrupted
            )
            assert [event.type for event in durable].count(EventType.TURN_COMPLETED) == 2
            skip_ahead(wait_seconds)
            resolved = [
                event async for event in app.resolve_tool_approval(await approval_request())
            ]
            assert resolved[-1].type is EventType.SESSION_COMPLETED
            assert tool.calls == 2
        finally:
            await app.aclose()
            if backend == "sqlite":
                await store.close()

    asyncio.run(scenario())
