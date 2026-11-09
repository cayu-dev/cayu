from __future__ import annotations

import asyncio

import pytest
from tests.core.test_openai_provider import RecordingTransport

from cayu import AgentSpec, CayuApp, EventType, Message, RunRequest
from cayu.context import (
    DefaultContextPolicy,
    MessageWindowContextPolicy,
    UsageTriggeredContextPolicy,
)
from cayu.context.counting import ContextCountingConfig
from cayu.providers import OpenAIProvider


@pytest.mark.anyio
@pytest.mark.parametrize("counter_boundary", ["observe", "policy"])
@pytest.mark.parametrize("counter_failure", ["ordinary", "child_cancel", "caller_cancel"])
async def test_optional_counter_child_cancellation_does_not_cancel_the_run(
    counter_boundary, counter_failure
):
    entered = asyncio.Event()

    class Transport(RecordingTransport):
        async def create_response(self, **kwargs):
            self.count_calls.append(kwargs)
            entered.set()
            if counter_failure == "caller_cancel":
                await asyncio.Event().wait()

            # This dependency task cancels itself. It never requests cancellation
            # on the caller or on the model-step owner.
            async def failed_dependency():
                if counter_failure == "child_cancel":
                    raise asyncio.CancelledError("counter dependency shut down")
                raise OSError("counter unavailable")

            await asyncio.create_task(failed_dependency())

    transport = Transport(
        stream_events=[
            [
                {"type": "response.output_text.delta", "delta": "done"},
                {
                    "type": "response.completed",
                    "response": {
                        "id": "resp_counter_test",
                        "model": "test-model",
                        "status": "completed",
                        "output": [
                            {
                                "type": "message",
                                "role": "assistant",
                                "content": [
                                    {"type": "output_text", "text": "done", "annotations": []}
                                ],
                            }
                        ],
                        "usage": {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
                    },
                },
            ]
        ]
    )
    provider = OpenAIProvider(api_key="test-key", transport=transport)
    app = CayuApp(
        enable_logging=False,
        context_counting=ContextCountingConfig(
            mode="observe" if counter_boundary == "observe" else "off"
        ),
    )
    app.register_provider(provider, default=True)
    policy = (
        UsageTriggeredContextPolicy(
            base_policy=DefaultContextPolicy(),
            triggered_policy=MessageWindowContextPolicy(max_messages=1),
            trigger_estimated_context_tokens=1,
            verify_estimate_with_provider_count=True,
        )
        if counter_boundary == "policy"
        else DefaultContextPolicy()
    )
    app.register_agent(AgentSpec(name="assistant", model="test-model"), context_policy=policy)
    events = []

    async def consume():
        async for event in app.run(
            RunRequest(agent_name="assistant", messages=[Message.text("user", "hello")])
        ):
            events.append(event)

    caller = asyncio.create_task(consume())
    try:
        if counter_failure == "caller_cancel":
            await asyncio.wait_for(entered.wait(), 10)
            caller.cancel()
        result = (await asyncio.gather(caller, return_exceptions=True))[0]
        assert transport.count_calls, "The optional counter must have been attempted"
        if counter_failure == "caller_cancel":
            assert caller.cancelled() and caller.cancelling() == 1
            assert not transport.calls
            return
        assert caller.cancelling() == 0, "This test never requests caller cancellation"
        assert not caller.cancelled(), (
            "A cancelled counter dependency is a counter failure, not caller cancellation",
            type(result).__name__,
            f"generation_requests={len(transport.calls)}",
        )
        assert result is None
        assert len(transport.calls) == 1
        assert events[-1].type is EventType.SESSION_COMPLETED, events[-1].payload
        if counter_boundary == "observe":
            assert any(event.type is EventType.CONTEXT_COUNT_FAILED for event in events)
    finally:
        if not caller.done():
            caller.cancel()
            await asyncio.gather(caller, return_exceptions=True)
        await app.aclose()
