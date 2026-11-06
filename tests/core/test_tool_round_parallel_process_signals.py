"""Parallel rounds surface process-control leaves, not TaskGroup wrappers."""

from __future__ import annotations

import asyncio

import pytest

from cayu import AgentSpec, CayuApp, Message, RunRequest
from cayu.evals.testing import ScriptedModelProvider
from cayu.providers.base import ModelStreamEvent
from cayu.tools.base import Tool, ToolEffect, ToolResult, ToolSpec


@pytest.mark.parametrize("signal_type", [SystemExit, KeyboardInterrupt, GeneratorExit])
@pytest.mark.parametrize("sibling_mode", ["return", "error", "cancel", "grouped_cancel"])
def test_parallel_process_signal_is_raised_directly(signal_type, sibling_mode):
    async def scenario():
        started = asyncio.Event()

        class SiblingTool(Tool):
            spec = ToolSpec(name="sibling", effect=ToolEffect.NONE)

            async def run(self, ctx, args):
                started.set()
                if sibling_mode == "return":
                    return ToolResult(content="done")
                if sibling_mode == "error":
                    raise ValueError("sibling failed")
                try:
                    await asyncio.Event().wait()
                except asyncio.CancelledError as cancellation:
                    if sibling_mode == "grouped_cancel":
                        raise BaseExceptionGroup(
                            "cancelled child cleanup", [cancellation]
                        ) from None
                    raise

        class ExitingTool(Tool):
            spec = ToolSpec(name="exit", effect=ToolEffect.NONE)

            async def run(self, ctx, args):
                await started.wait()
                raise BaseExceptionGroup("process stopping", [signal_type(7)])

        app = CayuApp(enable_logging=False)
        app.register_provider(
            ScriptedModelProvider(
                [
                    [
                        ModelStreamEvent.tool_call(id="sibling", name="sibling", arguments={}),
                        ModelStreamEvent.tool_call(id="exiting", name="exit", arguments={}),
                        ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
                    ],
                ]
            ),
            default=True,
        )
        app.register_agent(
            AgentSpec(name="assistant", model="scripted-model"),
            tools=[SiblingTool(), ExitingTool()],
        )
        with pytest.raises(signal_type) as captured:
            async with asyncio.timeout(10):
                async for _ in app.run(
                    RunRequest(
                        agent_name="assistant",
                        session_id="parallel-direct-process-signal",
                        messages=[Message.text("user", "run both")],
                    )
                ):
                    pass
        assert type(captured.value) is signal_type
        if signal_type is SystemExit:
            assert captured.value.code == 7

    asyncio.run(scenario())
