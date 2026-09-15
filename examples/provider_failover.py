"""Run an explicit fallback chain locally, with no credentials or network calls."""

from __future__ import annotations

import asyncio

from cayu import (
    AgentSpec,
    CayuApp,
    EventType,
    Message,
    ModelFailoverPolicy,
    ModelTarget,
    RunRequest,
    ScriptedModelProvider,
)
from cayu.providers.base import ModelProviderError, ModelRequest, ModelStreamEvent
from cayu.runtime.retry_policy import RetryPolicy


async def main() -> None:
    def unavailable(_request: ModelRequest) -> list[ModelStreamEvent]:
        raise ModelProviderError(
            "Service unavailable", provider="primary", status_code=503, retryable=True
        )

    primary = ScriptedModelProvider(name="primary", response_factory=unavailable)
    backup = ScriptedModelProvider(
        [[ModelStreamEvent.text_delta("Answered by the backup."), ModelStreamEvent.completed()]],
        name="backup",
    )
    app = CayuApp(enable_logging=False)
    app.register_provider(primary, default=True)
    app.register_provider(backup)
    app.register_agent(AgentSpec(name="assistant", model="primary-model"))
    events = [
        event
        async for event in app.run(
            RunRequest(
                agent_name="assistant",
                session_id="local-failover",
                messages=[Message.text("user", "Answer locally.")],
                retry_policy=RetryPolicy(max_attempts=1),
                failover=ModelFailoverPolicy(
                    fallbacks=(ModelTarget(provider_name="backup", model="backup-model"),),
                    max_total_attempts=2,
                ),
            )
        )
    ]
    assert events[-1].type is EventType.SESSION_COMPLETED
    assert len(primary.requests) == len(backup.requests) == 1
    session = await app.session_store.load("local-failover")
    assert session is not None and session.provider_name == "primary"
    for event in events:
        if event.type is EventType.MODEL_FAILOVER_SELECTED:
            print(f"Selected {event.payload['provider']}/{event.payload['model']}")
    print("Completed with one request per provider; configured session target unchanged.")


if __name__ == "__main__":
    asyncio.run(main())
