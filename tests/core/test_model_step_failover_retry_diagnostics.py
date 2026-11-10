"""Retry evidence must reflect the shared ordinary-model attempt ceiling."""

from __future__ import annotations

import asyncio

import pytest
from tests.core.test_model_failover_recovery import _RecoveryProvider

from cayu import AgentSpec, CayuApp, EventType, Message, RunRequest
from cayu.providers.base import ModelProviderError, ModelStreamEvent
from cayu.providers.retry_policy import RetryPolicy
from cayu.sessions import ModelFailoverPolicy, ModelTarget
from cayu.sessions.memory import InMemorySessionStore


@pytest.mark.parametrize("stream_error", [False, True], ids=["raised", "error-frame"])
@pytest.mark.parametrize("total_attempts", [1, 3])
def test_attempt_cap_does_not_advertise_another_retry(stream_error, total_attempts):
    class UnavailableProvider(_RecoveryProvider):
        async def stream(self, request):
            self.requests.append(request)
            failure = ModelProviderError(
                "service unavailable", provider=self.name, status_code=503, retryable=True
            )
            if stream_error:
                yield ModelStreamEvent.error(str(failure), cause=failure)
            else:
                raise failure

    async def scenario():
        store = InMemorySessionStore()
        primary = UnavailableProvider("primary")
        backup = _RecoveryProvider("backup")
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_provider(primary, default=True)
        app.register_provider(backup)
        app.register_agent(AgentSpec(name="agent", model="small"))
        try:
            events = [
                event
                async for event in app.run(
                    RunRequest(
                        agent_name="agent",
                        session_id="attempt-cap-diagnostics",
                        messages=[Message.text("user", "answer")],
                        retry_policy=RetryPolicy(max_attempts=5, initial_delay_s=0),
                        failover=ModelFailoverPolicy(
                            fallbacks=(ModelTarget(provider_name="backup", model="large"),),
                            max_total_attempts=total_attempts,
                        ),
                    )
                )
            ]
            assert events[-1].type is EventType.SESSION_FAILED
            assert len(primary.requests) == total_attempts
            assert not backup.requests
            assert sum(event.type is EventType.MODEL_RETRY for event in events) == (
                total_attempts - 1
            )
            exhausted = next(
                event for event in events if event.type is EventType.MODEL_FAILOVER_EXHAUSTED
            )
            assert exhausted.payload["reason"] == "attempt_limit"
            stored = await store.load_events("attempt-cap-diagnostics")
            final_error = [event for event in stored if event.type is EventType.MODEL_ERROR][-1]
            assert final_error.payload["provider_retryable"] is True
            assert final_error.payload["retry"] is False, (
                "No retry is authorized or scheduled at max_total_attempts; "
                f"terminal error advertises {final_error.payload['retry_disposition']!r}"
            )
            assert final_error.payload["retry_disposition"] == "configured_attempt_exhaustion"
        finally:
            await app.aclose()

    asyncio.run(scenario())
