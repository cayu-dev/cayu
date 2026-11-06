"""Parallel cancellation must not turn a sibling's process-control leaf into an error."""

from __future__ import annotations

import asyncio

from cayu import AgentSpec, CayuApp, Message, RunRequest
from cayu._exception_groups import iter_exception_tree
from cayu.evals.testing import ScriptedModelProvider
from cayu.providers.base import ModelStreamEvent
from cayu.tools.base import Tool, ToolEffect, ToolSpec


def test_parallel_grouped_cancellation_preserves_sibling_system_exit():
    async def scenario():
        started = asyncio.Event()

        class BlockingTool(Tool):
            spec = ToolSpec(name="block", effect=ToolEffect.NONE)

            async def run(self, ctx, args):
                started.set()
                try:
                    await asyncio.Event().wait()
                except asyncio.CancelledError as cancellation:
                    raise BaseExceptionGroup("cancelled child cleanup", [cancellation]) from None

        class ExitingTool(Tool):
            spec = ToolSpec(name="exit", effect=ToolEffect.NONE)

            async def run(self, ctx, args):
                await started.wait()
                # A nested process-control exception must remain supervisory control,
                # as it does at the single-invocation boundary.
                raise BaseExceptionGroup("process stopping", [SystemExit(7)])

        app = CayuApp(enable_logging=False)
        app.register_provider(
            ScriptedModelProvider(
                [
                    [
                        ModelStreamEvent.tool_call(id="blocked", name="block", arguments={}),
                        ModelStreamEvent.tool_call(id="exiting", name="exit", arguments={}),
                        ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
                    ],
                ]
            ),
            default=True,
        )
        app.register_agent(
            AgentSpec(name="assistant", model="scripted-model"),
            tools=[BlockingTool(), ExitingTool()],
        )
        failure = None
        try:
            async with asyncio.timeout(10):
                async for _ in app.run(
                    RunRequest(
                        agent_name="assistant",
                        session_id="parallel-process-control",
                        messages=[Message.text("user", "run both")],
                    )
                ):
                    pass
        except BaseException as exc:
            failure = exc
        assert failure is not None
        process_exits = [
            leaf for leaf in iter_exception_tree(failure) if isinstance(leaf, SystemExit)
        ]
        assert [leaf.code for leaf in process_exits] == [7], repr(failure)

    asyncio.run(scenario())
