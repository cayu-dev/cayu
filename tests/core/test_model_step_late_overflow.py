from __future__ import annotations

import asyncio

import pytest

from cayu import AgentSpec, CayuApp, EventType, Message, RunRequest
from cayu.context import CheckpointCompactionContextPolicy, ModelCompactor
from cayu.evals import ScriptedModelProvider
from cayu.providers.base import (
    ModelContextOverflowError,
    ModelProvider,
    ModelProviderError,
    ModelStreamEvent,
)


class _CompletedThenLateFailureProvider(ModelProvider):
    name = "completed-then-late-failure"

    def __init__(self, *, overflow):
        self.requests = []
        self.overflow = overflow

    async def stream(self, request):
        self.requests.append(request)
        yield ModelStreamEvent.text_delta("already completed")
        yield ModelStreamEvent.completed(
            {
                "finish_reason": "stop",
                "usage": {"input_tokens": 2, "output_tokens": 2, "total_tokens": 4},
            }
        )
        if len(self.requests) == 1:
            if self.overflow:
                raise ModelContextOverflowError(
                    "late overflow", provider=self.name, status_code=400
                )
            raise ModelProviderError(
                "late transport failure",
                provider=self.name,
                status_code=503,
                retryable=True,
            )


@pytest.mark.parametrize("overflow", [True, False], ids=["overflow", "transport-control"])
def test_late_failure_does_not_start_paid_context_recovery(overflow):
    async def run():
        provider = _CompletedThenLateFailureProvider(overflow=overflow)
        summary = ScriptedModelProvider(
            [
                [
                    ModelStreamEvent.text_delta("old conversation summary"),
                    ModelStreamEvent.completed(),
                ]
            ],
            name="summary",
        )
        app = CayuApp(enable_logging=False)
        app.register_provider(provider, default=True)
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            context_overflow_policy=CheckpointCompactionContextPolicy(
                compactor=ModelCompactor(provider=summary, model="summary-model"),
                max_user_turns=1,
                compact_after_messages=1,
            ),
        )
        try:
            events = [
                event
                async for event in app.run(
                    RunRequest(
                        agent_name="assistant",
                        session_id="late-overflow",
                        messages=[
                            Message.text("user", "Old question"),
                            Message.text("user", "Answer once"),
                        ],
                    )
                )
            ]
            main_completions = [
                event
                for event in events
                if event.type is EventType.MODEL_COMPLETED
                and event.payload.get("requested_model") == "fake-model"
            ]
            assert len(main_completions) == 1
            assert main_completions[0].payload["step_classification"]["type"] == "failed"
            assert len(provider.requests) == 1
            assert events[-1].type is EventType.SESSION_FAILED
            assert not summary.requests, (
                "A terminally failed completion must not start paid context recovery"
            )
            assert not any(event.type is EventType.CONTEXT_OVERFLOW_RECOVERING for event in events)
        finally:
            await app.aclose()

    asyncio.run(run())
