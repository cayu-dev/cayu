"""Completed tool results survive repeated cancellation during settlement rechecks."""

import asyncio

import pytest
from tests.core.test_workspace_mutation_receipts import _portable_environment_spec

from cayu import AgentSpec, CayuApp, EventType, Message, RunRequest, Tool, ToolResult, ToolSpec
from cayu.environments import Environment
from cayu.environments.bindings import DeterministicWorkspaceBinding
from cayu.evals import ScriptedModelProvider
from cayu.providers import ModelStreamEvent
from cayu.sessions import InMemorySessionStore
from cayu.sessions.requests import InterruptSessionRequest
from cayu.tools._resources import InvocationWorkspaceMutationOwner
from cayu.workspaces import LocalWorkspace


@pytest.mark.parametrize("interrupt_session", [False, True])
def test_recheck_cancellation_preserves_completed_result(tmp_path, monkeypatch, interrupt_session):
    async def scenario():
        sealing = asyncio.Event()
        rechecking = asyncio.Event()
        release = asyncio.Event()
        invocation_task = None
        original_seal = InvocationWorkspaceMutationOwner.seal_and_wait

        async def seal(owner):
            nonlocal invocation_task
            if invocation_task is None:
                invocation_task = asyncio.current_task()
                sealing.set()
                await asyncio.Event().wait()
            rechecking.set()
            await release.wait()
            return await original_seal(owner)

        monkeypatch.setattr(InvocationWorkspaceMutationOwner, "seal_and_wait", seal)

        class CompletedTool(Tool):
            spec = ToolSpec(name="completed", parallel_safe=False, workspace_mutation=True)

            async def run(self, ctx, args):
                await ctx.workspace.create_bytes("completed.txt", b"done")
                return ToolResult(content="completed before cancellation")

        store = InMemorySessionStore()
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_provider(
            ScriptedModelProvider(
                [
                    [
                        ModelStreamEvent.tool_call(id="write", name="completed", arguments={}),
                        ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
                    ],
                ]
            ),
            default=True,
        )
        app.register_environment(
            Environment(
                _portable_environment_spec("local"),
                workspace=LocalWorkspace(tmp_path, workspace_id="repeated-settlement"),
                binding=DeterministicWorkspaceBinding(),
            ),
            default=True,
        )
        app.register_agent(
            AgentSpec(name="worker", model="scripted-model"), tools=[CompletedTool()]
        )

        async def collect(stream):
            return [event async for event in stream]

        consumer = asyncio.create_task(
            collect(
                app.run(
                    RunRequest(
                        session_id="repeated-settlement",
                        agent_name="worker",
                        messages=[Message.text("user", "write")],
                    )
                )
            )
        )
        interruption = None
        try:
            await asyncio.wait_for(sealing.wait(), 10)
            if interrupt_session:
                interruption = asyncio.create_task(
                    collect(
                        app.interrupt_session(
                            InterruptSessionRequest(session_id="repeated-settlement")
                        )
                    )
                )
            else:
                invocation_task.cancel("first settlement cancellation")
            await asyncio.wait_for(rechecking.wait(), 10)
            assert invocation_task.cancelling() == 0
            invocation_task.cancel("second settlement cancellation")
            release.set()
            if interrupt_session:
                public, interrupted = await asyncio.wait_for(
                    asyncio.gather(consumer, interruption), 10
                )
                assert any(event.type is EventType.TOOL_CALL_COMPLETED for event in public)
                assert [event.type for event in interrupted] == [EventType.SESSION_INTERRUPTED]
            else:
                with pytest.raises(asyncio.CancelledError, match="first settlement cancellation"):
                    await asyncio.wait_for(consumer, 10)
                assert invocation_task.cancelling() == 2
            durable = await store.load_events("repeated-settlement")
            terminals = [event for event in durable if event.type is EventType.TOOL_CALL_COMPLETED]
            assert len(terminals) == 1
            assert terminals[0].payload["result"]["content"] == "completed before cancellation"
            assert any(event.type is EventType.WORKSPACE_OBSERVATION_FINALIZED for event in durable)
        finally:
            release.set()
            consumer.cancel()
            await asyncio.gather(consumer, return_exceptions=True)
            if interruption is not None:
                await asyncio.gather(interruption, return_exceptions=True)
            await app.aclose()

    asyncio.run(scenario())
