"""Grouped mutation-settlement failures must stop the rest of the tool round."""

import asyncio

from tests.core.test_workspace_mutation_receipts import (
    DeterministicWorkspaceBinding,
    _DetachedRunnerMutationTool,
    _FailingDeferredBackgroundMutationRunner,
    _portable_environment_spec,
)

from cayu import AgentSpec, CayuApp, EventType, Message, RunRequest, Tool, ToolResult, ToolSpec
from cayu._exception_groups import exception_tree_contains
from cayu._workspace_mutation import WorkspaceMutationSettlementError
from cayu.environments import Environment
from cayu.evals import ScriptedModelProvider
from cayu.providers import ModelStreamEvent
from cayu.sessions import InMemorySessionStore
from cayu.tools._resources import InvocationWorkspaceMutationOwner
from cayu.tools.base import ToolEffect
from cayu.workspaces import LocalWorkspace


def test_grouped_unproven_mutation_stops_following_tool(tmp_path, monkeypatch):
    async def scenario():
        sealing = asyncio.Event()
        release_failure = asyncio.Event()
        original_seal = InvocationWorkspaceMutationOwner.seal_and_wait
        settlement_errors = []

        async def observe_seal(owner):
            sealing.set()
            try:
                return await original_seal(owner)
            except Exception as error:
                settlement_errors.append(error)
                raise

        monkeypatch.setattr(InvocationWorkspaceMutationOwner, "seal_and_wait", observe_seal)

        class DisconnectedRunner(_FailingDeferredBackgroundMutationRunner):
            async def exec(self, command, **kwargs):
                # The remote command keeps running after its transport fails.
                await super().exec(command, **kwargs)
                await release_failure.wait()
                raise OSError("connection lost while command is still running")

        runner = DisconnectedRunner(tmp_path)
        following_started = asyncio.Event()
        followed_while_mutating = []

        class Following(Tool):
            spec = ToolSpec(name="following", parallel_safe=False, effect=ToolEffect.NONE)

            async def run(self, ctx, args):
                followed_while_mutating.append(not runner.mutation_finished.is_set())
                following_started.set()
                return ToolResult(content="following tool ran")

        store = InMemorySessionStore()
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_provider(
            ScriptedModelProvider(
                [
                    [
                        ModelStreamEvent.tool_call(
                            id="mutate", name="detached_runner_mutation", arguments={}
                        ),
                        ModelStreamEvent.tool_call(id="next", name="following", arguments={}),
                        ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
                    ],
                    [ModelStreamEvent.text_delta("done"), ModelStreamEvent.completed({})],
                ]
            ),
            default=True,
        )
        app.register_environment(
            Environment(
                _portable_environment_spec("local"),
                runner=runner,
                workspace=LocalWorkspace(tmp_path, workspace_id="grouped-settlement"),
                binding=DeterministicWorkspaceBinding(),
            ),
            default=True,
        )
        app.register_agent(
            AgentSpec(name="worker", model="scripted-model"),
            tools=[_DetachedRunnerMutationTool(), Following()],
        )

        async def consume():
            return [
                event
                async for event in app.run(
                    RunRequest(
                        session_id="grouped-settlement",
                        agent_name="worker",
                        messages=[Message.text("user", "mutate then inspect")],
                    )
                )
            ]

        consumer = asyncio.create_task(consume())
        following_wait = asyncio.create_task(following_started.wait())
        try:
            await asyncio.wait_for(sealing.wait(), 10)
            await asyncio.wait_for(runner.started.wait(), 10)
            release_failure.set()
            done, _ = await asyncio.wait(
                {consumer, following_wait}, timeout=10, return_when=asyncio.FIRST_COMPLETED
            )
            assert done, "Neither failure nor following tool was delivered"
        finally:
            release_failure.set()
            runner.release_mutation.set()
            following_wait.cancel()
            await asyncio.gather(following_wait, return_exceptions=True)
            await asyncio.wait_for(consumer, 10)
            if runner._mutation is not None:
                await runner._mutation
            await app.drain_environment_cleanups(timeout_s=10)

        assert len(settlement_errors) == 1
        assert isinstance(settlement_errors[0], ExceptionGroup)
        assert exception_tree_contains(settlement_errors[0], WorkspaceMutationSettlementError)
        assert not followed_while_mutating, (
            "The next tool ran despite the preceding mutation having no settlement proof: "
            f"ran while remote mutation was active = {followed_while_mutating}"
        )
        durable = await store.load_events("grouped-settlement")
        terminal = next(event for event in durable if event.type is EventType.TOOL_CALL_COMPLETED)
        assert (
            terminal.payload["workspace_mutation_capture_detail_code"]
            == "mutation_settlement_unproven"
        )

    asyncio.run(scenario())
