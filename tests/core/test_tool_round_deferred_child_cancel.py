"""Deferred rounds must still close a child-only cancellation of a non-external tool."""

from __future__ import annotations

import asyncio

import pytest

from cayu import AgentSpec, CayuApp, EventType, Message, RunRequest
from cayu.environments.base import Environment, EnvironmentSpec
from cayu.evals.testing import ScriptedModelProvider
from cayu.messages import ToolResultPart
from cayu.providers.base import ModelStreamEvent
from cayu.tools.base import Tool, ToolEffect, ToolResult, ToolSpec
from cayu.vaults.static import StaticVault


@pytest.mark.parametrize("effect", [ToolEffect.NONE, ToolEffect.IDEMPOTENT])
@pytest.mark.parametrize("deferral", ["none", "dynamic_environment", "external_sibling"])
def test_nonexternal_child_cancel_closes_even_when_round_defers_terminals(effect, deferral):
    async def scenario():
        calls = []

        class CancelTool(Tool):
            spec = ToolSpec(name="cancel", effect=effect)

            async def run(self, ctx, args):
                calls.append("cancel")
                # Simulate an adapter leaking its own cancelled scope. The caller
                # has not cancelled the run and no operator interrupt is requested.
                raise asyncio.CancelledError()

        class SiblingTool(Tool):
            spec = ToolSpec(
                name="sibling",
                effect=ToolEffect.EXTERNAL if deferral == "external_sibling" else ToolEffect.NONE,
            )

            async def run(self, ctx, args):
                calls.append("sibling")
                return ToolResult(content="completed sibling result")

        provider = ScriptedModelProvider(
            [
                [
                    ModelStreamEvent.tool_call(id="cancelled", name="cancel", arguments={}),
                    ModelStreamEvent.tool_call(id="succeeded", name="sibling", arguments={}),
                    ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
                ],
                [
                    ModelStreamEvent.text_delta("done"),
                    ModelStreamEvent.completed({"finish_reason": "stop"}),
                ],
            ]
        )
        app = CayuApp(enable_logging=False)
        app.register_provider(provider, default=True)
        app.register_agent(
            AgentSpec(name="assistant", model="scripted-model"),
            tools=[CancelTool(), SiblingTool()],
        )
        if deferral == "dynamic_environment":
            app.register_environment(
                Environment(EnvironmentSpec(name="local"), vault=StaticVault({})), default=True
            )
        session_id = f"nonexternal-child-cancel-{deferral}-{effect.value}"
        try:
            events = [
                event
                async for event in app.run(
                    RunRequest(
                        agent_name="assistant",
                        session_id=session_id,
                        messages=[Message.text("user", "run both")],
                    )
                )
            ]
            assert sorted(calls) == ["cancel", "sibling"]
            assert events[-1].type is EventType.SESSION_COMPLETED, events[-1].payload
            results = [
                part
                for message in await app.session_store.load_transcript(session_id)
                for part in message.content
                if isinstance(part, ToolResultPart)
            ]
            assert [result.tool_call_id for result in results] == ["cancelled", "succeeded"]
            assert results[0].is_error is True
            assert results[1].content == "completed sibling result"
            assert results[1].is_error is False
        finally:
            await app.aclose()

    asyncio.run(scenario())
