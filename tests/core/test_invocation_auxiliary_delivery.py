"""Completed tool outcomes survive abandonment while delivering nested-model events."""

import asyncio

import pytest

from cayu import AgentSpec, CayuApp, EventType, Message, RunRequest, Tool, ToolResult, ToolSpec
from cayu.evals import ScriptedModelProvider
from cayu.providers import ModelRequest, ModelStreamEvent
from cayu.sessions import InMemorySessionStore
from cayu.sessions.requests import ResumeRequest
from cayu.tools.base import ToolEffect
from cayu.tools.inference import AuxiliaryInferencePolicy, InferenceLimits


@pytest.mark.parametrize("abandon", [False, True])
def test_completed_effect_survives_auxiliary_event_delivery(abandon):
    async def scenario():
        completed = []
        limits = InferenceLimits(max_input_tokens=10, max_output_tokens=10, timeout_seconds=10)

        class Send(Tool):
            spec = ToolSpec(
                name="send",
                parallel_safe=False,
                effect=ToolEffect.EXTERNAL,
                auxiliary_inference=AuxiliaryInferencePolicy(
                    limits=limits, purposes=("tool.summary",)
                ),
            )

            async def run(self, ctx, args):
                await ctx.inference.invoke(
                    ModelRequest(model="scripted-model", messages=[Message.text("user", "nested")]),
                    purpose="tool.summary",
                    limits=limits,
                )
                completed.append("receipt-123")
                return ToolResult(content="sent", structured={"receipt_id": "receipt-123"})

        store = InMemorySessionStore()
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_provider(
            ScriptedModelProvider(
                [
                    [
                        ModelStreamEvent.tool_call(id="send-1", name="send", arguments={}),
                        ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
                    ],
                    [
                        ModelStreamEvent.text_delta("summary"),
                        ModelStreamEvent.completed(
                            {"usage": {"input_tokens": 1, "output_tokens": 1}}
                        ),
                    ],
                    [ModelStreamEvent.text_delta("done"), ModelStreamEvent.completed({})],
                ]
            ),
            default=True,
        )
        app.register_agent(AgentSpec(name="worker", model="scripted-model"), tools=[Send()])
        stream = app.run(
            RunRequest(
                session_id="auxiliary-delivery",
                agent_name="worker",
                messages=[Message.text("user", "send")],
            )
        )
        observed = []
        try:
            async for event in stream:
                observed.append(event)
                if abandon and "auxiliary_inference" in event.payload:
                    # Nested-model events are delivered only after dispatch.run returns.
                    assert completed == ["receipt-123"]
                    break
        finally:
            await stream.aclose()
        try:
            assert completed == ["receipt-123"]
            if abandon:
                # Normal continuation must recover the exact completed receipt,
                # rather than ask for reconciliation of an effect that already returned.
                _ = [
                    event
                    async for event in app.resume(
                        ResumeRequest(
                            session_id="auxiliary-delivery",
                            messages=[Message.text("user", "continue")],
                        )
                    )
                ]
            events = await store.load_events("auxiliary-delivery")
            terminals = [event for event in events if event.type is EventType.TOOL_CALL_COMPLETED]
            checkpoint = await store.load_checkpoint("auxiliary-delivery")
            assert "receipt-123" in repr(([event.payload for event in terminals], checkpoint)), (
                "The completed external receipt was lost while delivering auxiliary telemetry",
                checkpoint,
            )
        finally:
            await app.aclose()

    asyncio.run(scenario())
