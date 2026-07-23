from __future__ import annotations

import asyncio
import traceback
from collections.abc import Mapping
from typing import Any

import pytest
from tests.provider_traceback_assertions import is_cayu_source_filename

from cayu import Message
from cayu.providers import ModelContextOverflowError, ModelRequest, ModelStreamEventType
from cayu.providers.openai_subscription import (
    OpenAISubscriptionAuthError,
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


def test_subscription_transport_error_projection_redacts_all_provider_authority() -> None:
    canaries = {
        "access": "provider-access-canary-0123456789",
        "refresh": "provider-refresh-canary-0123456789",
        "account": "provider-account-canary-0123456789",
        "header": "Bearer provider-access-canary-0123456789",
    }

    class CanaryAuth:
        async def credentials(self) -> OpenAISubscriptionCredentials:
            return OpenAISubscriptionCredentials(
                access_token=canaries["access"],
                refresh_token=canaries["refresh"],
                expires_at=2_000_000_000,
                account_id=canaries["account"],
            )

    class LeakingTransport(RecordingTransport):
        async def stream_response_events(self, **_kwargs: Any):
            raise RuntimeError(" | ".join(canaries.values()))
            yield  # pragma: no cover

    provider = OpenAISubscriptionProvider(auth=CanaryAuth(), transport=LeakingTransport())
    request = ModelRequest(
        model="gpt-5.4",
        messages=[Message.text("user", "Say hello")],
    )

    async def collect():
        return [event async for event in provider.stream(request)]

    events = asyncio.run(collect())

    assert len(events) == 1
    rendered = repr(events[0]) + str(events[0].model_dump(mode="json"))
    assert all(value not in rendered for value in canaries.values())
    assert events[0].payload["error"] == "OpenAI subscription provider failed."


def test_subscription_authentication_error_projection_is_allowlisted() -> None:
    canary = "provider-refresh-canary-0123456789"

    class LeakingAuth:
        async def credentials(self) -> OpenAISubscriptionCredentials:
            raise OpenAISubscriptionAuthError(f"refresh failed for {canary}")

    provider = OpenAISubscriptionProvider(auth=LeakingAuth(), transport=RecordingTransport())
    request = ModelRequest(
        model="gpt-5.4",
        messages=[Message.text("user", "Say hello")],
    )

    async def collect():
        return [event async for event in provider.stream(request)]

    events = asyncio.run(collect())

    rendered = repr(events[0]) + str(events[0].model_dump(mode="json"))
    assert canary not in rendered
    assert events[0].payload["error"] == (
        "OpenAI subscription authentication failed. Run `cayu auth openai login` again."
    )


def test_subscription_provider_preserves_sanitized_cancellation() -> None:
    canary = "provider-access-canary-0123456789"

    class CanaryAuth:
        async def credentials(self) -> OpenAISubscriptionCredentials:
            return OpenAISubscriptionCredentials(
                access_token=canary,
                refresh_token="provider-refresh-canary-0123456789",
                expires_at=2_000_000_000,
            )

    class CancellingTransport(RecordingTransport):
        async def stream_response_events(self, **_kwargs: Any):
            error = asyncio.CancelledError(f"provider cancelled with {canary}")
            error.artifacts = [{"transport": canary}]
            error.headers = {"Authorization": f"Bearer {canary}"}
            error.add_note(f"cancelled near {canary}")
            raise error
            yield  # pragma: no cover

    provider = OpenAISubscriptionProvider(auth=CanaryAuth(), transport=CancellingTransport())
    request = ModelRequest(
        model="gpt-5.4",
        messages=[Message.text("user", "Say hello")],
    )

    async def collect():
        return [event async for event in provider.stream(request)]

    with pytest.raises(asyncio.CancelledError) as exc_info:
        asyncio.run(collect())

    rendered = repr(exc_info.value) + repr(vars(exc_info.value))
    assert canary not in rendered
    assert exc_info.value.artifacts == []
    assert not hasattr(exc_info.value, "headers")
    assert not hasattr(exc_info.value, "__notes__")
    assert exc_info.value.__cause__ is None
    assert exc_info.value.__context__ is None
    captured = traceback.TracebackException.from_exception(
        exc_info.value,
        capture_locals=True,
    )
    cayu_frames = [frame for frame in captured.stack if is_cayu_source_filename(frame.filename)]
    assert canary not in repr([(frame.name, frame.locals) for frame in cayu_frames])
    assert all(frame.name != "stream_response_events" for frame in captured.stack)


def test_subscription_context_overflow_projection_omits_provider_authority() -> None:
    canary = "provider-account-canary-0123456789"

    class CanaryAuth:
        async def credentials(self) -> OpenAISubscriptionCredentials:
            return OpenAISubscriptionCredentials(
                access_token="provider-access-canary-0123456789",
                refresh_token="provider-refresh-canary-0123456789",
                expires_at=2_000_000_000,
                account_id=canary,
            )

    class OverflowTransport(RecordingTransport):
        async def stream_response_events(self, **_kwargs: Any):
            raise ModelContextOverflowError(
                f"context overflow near {canary}",
                provider="openai_subscription",
                status_code=400,
                request_id=canary,
                response_body=canary,
            )
            yield  # pragma: no cover

    provider = OpenAISubscriptionProvider(auth=CanaryAuth(), transport=OverflowTransport())
    request = ModelRequest(
        model="gpt-5.4",
        messages=[Message.text("user", "Say hello")],
    )

    async def collect():
        return [event async for event in provider.stream(request)]

    with pytest.raises(ModelContextOverflowError) as exc_info:
        asyncio.run(collect())

    rendered = str(exc_info.value) + repr(exc_info.value) + repr(vars(exc_info.value))
    assert canary not in rendered
    assert str(exc_info.value) == "OpenAI subscription context window exceeded."
    assert exc_info.value.status_code == 400
    assert exc_info.value.request_id is None
    assert exc_info.value.response_body is None
    assert exc_info.value.__cause__ is None
    assert exc_info.value.__context__ is None
    captured = traceback.TracebackException.from_exception(
        exc_info.value,
        capture_locals=True,
    )
    cayu_frames = [frame for frame in captured.stack if is_cayu_source_filename(frame.filename)]
    assert canary not in repr([(frame.name, frame.locals) for frame in cayu_frames])
    assert all(frame.name != "stream_response_events" for frame in captured.stack)
