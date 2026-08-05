from __future__ import annotations

import asyncio
from typing import Any

import httpx
import pytest

from cayu import AgentSpec, CayuApp, Event, EventType, Message, RetryPolicy, RunRequest
from cayu.providers import (
    AnthropicProvider,
    ChatCompletionsProvider,
    ModelProvider,
    OpenAIProvider,
)


async def _collect_events(app: CayuApp, request: RunRequest) -> list[Event]:
    return [event async for event in app.run(request)]


@pytest.mark.parametrize("provider_kind", ["openai", "anthropic", "chat_completions"])
def test_cayu_app_does_not_retry_conflicting_buffered_provider_identity(
    monkeypatch: pytest.MonkeyPatch,
    provider_kind: str,
) -> None:
    error: dict[str, Any] = {
        "type": "authentication_error",
        "message": "Conflicting permanent provider failure.",
    }
    if provider_kind != "anthropic":
        error["code"] = "server_error"
    response_body = (
        {"type": "error", "error": error} if provider_kind == "anthropic" else {"error": error}
    )

    class StreamContext:
        def __init__(self, response: httpx.Response) -> None:
            self.response = response

        async def __aenter__(self) -> httpx.Response:
            return self.response

        async def __aexit__(self, *args: Any) -> None:
            await self.response.aclose()

    class ConflictingHttpClient:
        calls = 0

        def __init__(self, **_kwargs: Any) -> None:
            self.is_closed = False

        def stream(self, method: str, url: str, **_kwargs: Any) -> StreamContext:
            type(self).calls += 1
            request = httpx.Request(method, url)
            return StreamContext(
                httpx.Response(
                    500,
                    request=request,
                    headers={"content-type": "application/json", "retry-after": "9"},
                    json=response_body,
                )
            )

        async def aclose(self) -> None:
            self.is_closed = True

    monkeypatch.setattr("cayu.providers._http.httpx.AsyncClient", ConflictingHttpClient)
    if provider_kind == "openai":
        provider: ModelProvider = OpenAIProvider(api_key="test-key")
    elif provider_kind == "anthropic":
        provider = AnthropicProvider(api_key="test-key")
    else:
        provider = ChatCompletionsProvider(api_key="test-key")
    app = CayuApp(retry_policy=RetryPolicy(max_attempts=2, initial_delay_s=0.0))
    app.register_provider(provider, default=True)
    app.register_agent(AgentSpec(name="assistant", model="test-model"))

    events = asyncio.run(
        _collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id=f"sess_conflicting_buffered_{provider_kind}",
                messages=[Message.text("user", "hi")],
            ),
        )
    )

    assert ConflictingHttpClient.calls == 1
    assert EventType.MODEL_RETRY not in {event.type for event in events}
    model_errors = [event for event in events if event.type == EventType.MODEL_ERROR]
    assert len(model_errors) == 1
    assert model_errors[0].payload["status_code"] == 500
    assert model_errors[0].payload["provider_error_type"] == "authentication_error"
    assert model_errors[0].payload["retryable"] is False
    assert model_errors[0].payload["retry_after_s"] == 9.0
    assert events[-1].type == EventType.SESSION_FAILED
