"""Overflow recovery remains in the original model step's retry time budget."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from cayu import AgentSpec, CayuApp, EventType, Message, RunRequest
from cayu.context import RecentTurnsContextPolicy
from cayu.providers.base import (
    ModelContextOverflowError,
    ModelProvider,
    ModelProviderError,
    ModelStreamEvent,
)
from cayu.providers.retry_policy import RetryPolicy
from cayu.runtime import _model_step_executor


@pytest.mark.parametrize("as_frame", [False, True], ids=["raised", "error-frame"])
def test_overflow_rebuild_does_not_restart_exhausted_retry_time_budget(monkeypatch, as_frame):
    elapsed_s = 0.0
    waits = []

    async def advance_retry_clock(delay):
        nonlocal elapsed_s
        waits.append(delay)
        elapsed_s += delay
        await asyncio.sleep(0)

    # Match test_retry_time_budget.py: change only the executor's retry clock,
    # not the actual loop or provider stream deadlines.
    monkeypatch.setattr(
        _model_step_executor,
        "asyncio",
        SimpleNamespace(
            **(
                vars(asyncio)
                | {
                    "get_running_loop": lambda: SimpleNamespace(time=lambda: elapsed_s),
                    "sleep": advance_retry_clock,
                }
            )
        ),
    )

    class OverflowThenOverloadedProvider(ModelProvider):
        name = "overflow-overload"

        def __init__(self):
            self.requests = []

        async def stream(self, request):
            nonlocal elapsed_s
            self.requests.append(request)
            if len(self.requests) == 1:
                # Each response is within the ordinary 300-second idle deadline,
                # but together the initial attempts spend the step's retry budget.
                elapsed_s += 200.0
                failure = ModelProviderError(
                    "overloaded", provider=self.name, status_code=503, retryable=True
                )
            elif len(self.requests) == 2:
                elapsed_s += 101.0
                failure = ModelContextOverflowError("context too large", provider=self.name)
            elif len(self.requests) == 3:
                failure = ModelProviderError(
                    "overloaded again", provider=self.name, status_code=503, retryable=True
                )
            else:
                yield ModelStreamEvent.text_delta("a retry after the budget expired")
                yield ModelStreamEvent.completed()
                return
            if as_frame:
                yield ModelStreamEvent.error(str(failure), cause=failure)
            else:
                raise failure

    async def scenario():
        provider = OverflowThenOverloadedProvider()
        app = CayuApp(enable_logging=False)
        app.register_provider(provider, default=True)
        app.register_agent(
            AgentSpec(name="assistant", model="test-model"),
            context_overflow_policy=RecentTurnsContextPolicy(max_user_turns=1),
        )
        policy = RetryPolicy(jitter_s=0.0)
        assert policy.max_retry_elapsed_s == 300.0
        try:
            events = [
                event
                async for event in app.run(
                    RunRequest(
                        agent_name="assistant",
                        messages=[
                            Message.text("user", "old question"),
                            Message.text("assistant", "old answer"),
                            Message.text("user", "current question"),
                        ],
                        retry_policy=policy,
                    )
                )
            ]
            assert any(event.type is EventType.CONTEXT_OVERFLOW_RECOVERING for event in events)
            assert len(provider.requests[2].messages) < len(provider.requests[0].messages)
            assert len(provider.requests) == 3, (
                "Overflow rebuilding is still the same model step: a subsequent 503 "
                "must not retry after its 300-second budget was spent; "
                f"dispatched {len(provider.requests)} requests at elapsed={elapsed_s}"
            )
            assert len([event for event in events if event.type is EventType.MODEL_RETRY]) == 1
            assert events[-1].type is EventType.SESSION_FAILED
            error = [event for event in events if event.type is EventType.MODEL_ERROR][-1]
            assert error.payload["retry_disposition"] == "configured_attempt_exhaustion"
        finally:
            await app.aclose()

    asyncio.run(scenario())
