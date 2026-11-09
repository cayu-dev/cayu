"""Closing at a durable human pause must finish its terminal publication."""

import asyncio
from contextlib import suppress

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
def test_approval_pause_close_finishes_publication(
    tmp_path, skip_ahead, backend, close_after, wait_seconds
):
    async def scenario():
        store = (
            InMemorySessionStore()
            if backend == "memory"
            else SQLiteSessionStore(tmp_path / "pause-close.sqlite")
        )
        provider = ScriptedModelProvider(
            [
                [
                    ModelStreamEvent.tool_call(name="ask_customer", arguments={}, id="question"),
                    ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
                ],
                [
                    ModelStreamEvent.text_delta("done"),
                    ModelStreamEvent.completed({"finish_reason": "stop"}),
                ],
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
        try:
            # The same default deadline source that a bounded task handler supplies.
            with default_run_deadline(ExecutionDeadline.after(60, source="task_worker")):
                stream = app.run(
                    RunRequest(
                        agent_name="assistant",
                        session_id="pause-close",
                        messages=[Message.text("user", "Ask the customer")],
                    )
                )
                try:
                    async for event in stream:
                        if event.type is close_after:
                            break
                finally:
                    await stream.aclose()
            assert await app.drain_recovery_cleanups(timeout_s=5)
            assert await app.drain_environment_cleanups(timeout_s=5)
            paused = await store.load("pause-close")
            assert paused.status is SessionStatus.INTERRUPTED
            assert paused.execution_deadline.origin == "run_default"
            durable = await store.load_events("pause-close")
            pending = (await store.load_checkpoint("pause-close"))["pending_tool_approval"]
            skip_ahead(wait_seconds)
            # Waiting for a person must not spend an inherited handler deadline.
            resolved = [
                event
                async for event in app.resolve_tool_approval(
                    ToolApprovalRequest(
                        session_id="pause-close",
                        approval_id=pending["approval_id"],
                        tool_round_id=pending["tool_round_id"],
                        tool_call_id=pending["tool_call_id"],
                        decision=ToolApprovalDecision.APPROVE,
                    )
                )
            ]
            assert resolved[-1].type is EventType.SESSION_COMPLETED
            assert tool.calls == 1
            interrupted = [
                event for event in durable if event.type is EventType.SESSION_INTERRUPTED
            ]
            assert len(interrupted) == 1, [event.type for event in durable]
            assert interrupted[0].payload["interruption_type"] == "tool_approval_required"
        finally:
            await app.aclose()
            if backend == "sqlite":
                await store.close()

    asyncio.run(scenario())


@pytest.mark.parametrize("continuation", [False, True])
def test_pause_close_retains_slow_terminal_publication(continuation):
    from cayu import CayuConfig
    from cayu.configuration import OperationsConfig
    from cayu.observability.hooks import RuntimeHook
    from cayu.runtime.execution_identity import ExecutionProfileBehaviorIdentity
    from cayu.sessions._execution_profile_checkpoint import (
        active_invocation_execution_profile_from_checkpoint,
    )
    from cayu.sessions.cleanup import RecoveryCleanupDeadlineExceeded, RecoveryCleanupPolicy

    async def scenario():
        started = asyncio.Event()
        release = asyncio.Event()

        class SlowHook(RuntimeHook):
            execution_profile_identity = ExecutionProfileBehaviorIdentity(
                name="tests:slow-pause", behavior_version="1", implementation_version="1"
            )

            calls = 0

            async def after_session_interrupted(self, context):
                self.calls += 1
                if continuation and self.calls == 1:
                    return
                started.set()
                while not release.is_set():
                    with suppress(asyncio.CancelledError):
                        await release.wait()

        store = InMemorySessionStore()
        app = CayuApp(
            session_store=store,
            runtime_hooks=[SlowHook()],
            config=CayuConfig(
                operations=OperationsConfig(
                    recovery_cleanup_policy=RecoveryCleanupPolicy(
                        step_timeout_seconds=0.05,
                        overall_timeout_seconds=1,
                    )
                )
            ),
            enable_logging=False,
        )
        app.register_provider(
            ScriptedModelProvider(
                [
                    [
                        ModelStreamEvent.tool_call(
                            name="ask_customer", arguments={}, id=f"question-{index}"
                        ),
                        ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
                    ]
                    for index in range(2 if continuation else 1)
                ]
            ),
            default=True,
        )
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            tools=[FinalTool()],
            tool_policy=AlwaysRequireApprovalToolPolicy(),
        )
        stream = app.run(
            RunRequest(
                agent_name="assistant",
                session_id="slow-pause",
                messages=[Message.text("user", "ask")],
            )
        )
        if continuation:
            async for _event in stream:
                pass
            pending = (await store.load_checkpoint("slow-pause"))["pending_tool_approval"]
            stream = app.resolve_tool_approval(
                ToolApprovalRequest(
                    session_id="slow-pause",
                    approval_id=pending["approval_id"],
                    tool_round_id=pending["tool_round_id"],
                    tool_call_id=pending["tool_call_id"],
                    decision=ToolApprovalDecision.APPROVE,
                )
            )
        try:
            async for event in stream:
                if event.type is EventType.INTERACTION_PAUSED:
                    break

            async def release_after_retention():
                await asyncio.wait_for(started.wait(), 5)
                await asyncio.sleep(0.2)
                checkpoint = await store.load_checkpoint("slow-pause")
                assert active_invocation_execution_profile_from_checkpoint(checkpoint) is not None
                release.set()

            releasing = asyncio.create_task(release_after_retention())
            with suppress(RecoveryCleanupDeadlineExceeded):
                await stream.aclose()
            assert not app._session_control.has_active_tasks("slow-pause")
            await asyncio.wait_for(releasing, 5)
            assert await app.drain_recovery_cleanups(timeout_s=5)
            events = await store.load_events("slow-pause")
            assert sum(
                event.type is EventType.HOOK_COMPLETED
                and event.payload.get("phase") == "after_session_interrupted"
                for event in events
            ) == (2 if continuation else 1)
        finally:
            release.set()
            await stream.aclose()
            await app.aclose()

    asyncio.run(scenario())
