from __future__ import annotations

import asyncio
from collections.abc import Mapping
from typing import Any

from cayu import Message
from cayu.providers import ModelRequest, ModelStreamEventType
from cayu.providers.openai_subscription import (
    OpenAISubscriptionCredentials,
    OpenAISubscriptionProvider,
)


class StaticSubscriptionAuth:
    async def credentials(self) -> OpenAISubscriptionCredentials:
        return OpenAISubscriptionCredentials(
            access_token="subscription-access",
            refresh_token="subscription-refresh",
            expires_at=2_000_000_000,
            account_id="acct-cayu",
        )


class RecordingTransport:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []
        self.closed = False

    async def stream_response_events(
        self,
        *,
        url: str,
        headers: Mapping[str, str],
        payload: Mapping[str, Any],
        timeout_s: float,
        stream_idle_timeout_s: float,
    ):
        self.calls.append(
            {
                "url": url,
                "headers": dict(headers),
                "payload": dict(payload),
                "timeout_s": timeout_s,
                "stream_idle_timeout_s": stream_idle_timeout_s,
            }
        )
        yield {"type": "response.output_text.delta", "delta": "hello"}
        yield {
            "type": "response.completed",
            "response": {
                "id": "resp-subscription",
                "model": "gpt-5.4",
                "status": "completed",
                "output": [],
                "usage": {"input_tokens": 9, "output_tokens": 2, "total_tokens": 11},
            },
        }

    async def create_response(
        self,
        *,
        url: str,
        headers: Mapping[str, str],
        payload: Mapping[str, Any],
        timeout_s: float,
    ) -> Mapping[str, Any]:
        raise AssertionError("The subscription provider has no token-counting endpoint.")

    async def aclose(self) -> None:
        self.closed = True


def test_subscription_provider_uses_codex_endpoint_with_honest_cayu_identity() -> None:
    transport = RecordingTransport()
    provider = OpenAISubscriptionProvider(auth=StaticSubscriptionAuth(), transport=transport)
    request = ModelRequest(
        model="gpt-5.4",
        messages=[Message.text("user", "Say hello")],
    )

    async def collect():
        return [event async for event in provider.stream(request)]

    events = asyncio.run(collect())

    assert [event.type for event in events] == [
        ModelStreamEventType.TEXT_DELTA,
        ModelStreamEventType.COMPLETED,
    ]
    call = transport.calls[0]
    assert call["url"] == "https://chatgpt.com/backend-api/codex/responses"
    assert call["headers"]["authorization"] == "Bearer subscription-access"
    assert call["headers"]["ChatGPT-Account-ID"] == "acct-cayu"
    assert call["headers"]["originator"] == "cayu"
    assert call["headers"]["user-agent"].startswith("cayu/")
    assert call["payload"]["store"] is False
    assert call["payload"]["stream"] is True


def test_subscription_provider_declares_remote_token_counting_unavailable() -> None:
    provider = OpenAISubscriptionProvider(
        auth=StaticSubscriptionAuth(),
        transport=RecordingTransport(),
    )
    request = ModelRequest(
        model="gpt-5.4",
        messages=[Message.text("user", "Count this")],
    )

    assert asyncio.run(provider.count_input_tokens(request)) is None
