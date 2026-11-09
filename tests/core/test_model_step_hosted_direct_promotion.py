from __future__ import annotations

import asyncio

import pytest
from tests.core.test_openai_background_operations import (
    BackgroundTransport,
    _completed,
    _hosted_tool_search_completion,
    _RememberKnowledgeTool,
)

from cayu import AgentSpec, CayuApp, EventType, Message, RunRequest
from cayu.providers import OpenAIProvider
from cayu.tools.exposure import (
    ToolExposureDecision,
    ToolExposurePolicy,
    ToolExposurePolicyRequest,
)


@pytest.mark.parametrize("promote_to_direct", [False, True])
@pytest.mark.parametrize("reasoning_state", ["inline", "server"])
def test_hosted_discovery_history_survives_direct_exposure_promotion(
    promote_to_direct, reasoning_state
):
    class PhasePolicy(ToolExposurePolicy):
        def select(self, request: ToolExposurePolicyRequest) -> ToolExposureDecision:
            direct = promote_to_direct and request.step > 1
            return ToolExposureDecision(
                profile_id="direct" if direct else "discovery-only",
                tool_names=("remember_knowledge",) if direct else (),
            )

    class RecordingTool(_RememberKnowledgeTool):
        def __init__(self):
            super().__init__()
            self.calls = []

        async def run(self, ctx, args):
            self.calls.append(dict(args))
            return await super().run(ctx, args)

    async def run():
        tool = RecordingTool()
        transport = BackgroundTransport()
        transport.start_batches = [
            [{"type": "response.completed", "response": _hosted_tool_search_completion()}],
            [_completed(response_id="resp_final")],
        ]
        provider = OpenAIProvider(
            api_key="unit-test-key",
            transport=transport,
            hosted_tool_search_models=("gpt-test",),
            reasoning_state=reasoning_state,
        )
        app = CayuApp(enable_logging=False)
        app.register_provider(provider, default=True)
        app.register_agent(
            AgentSpec(name="assistant", model="gpt-test"),
            tools=[tool],
            tool_discovery_mode="openai_tool_search_hosted",
            tool_exposure_policy=PhasePolicy(),
        )
        try:
            events = [
                event
                async for event in app.run(
                    RunRequest(
                        agent_name="assistant",
                        session_id="hosted-direct-promotion",
                        messages=[Message.text("user", "Remember the fact, then confirm it.")],
                    )
                )
            ]
            assert tool.calls == [{"fact": "recover hosted authority"}]
            assert events[-1].type is EventType.SESSION_COMPLETED, events[-1].payload
            assert len(transport.start_calls) == 2
        finally:
            await app.aclose()

    asyncio.run(run())
