from __future__ import annotations

import asyncio
import traceback
from collections.abc import Mapping
from typing import Any

import pytest
from tests.provider_traceback_assertions import is_cayu_source_filename

from cayu import Message, __version__
from cayu.providers import (
    HostedToolCapabilityError,
    ModelContextOverflowError,
    ModelRequest,
    ModelStreamEventType,
    OpenAIWebSearch,
)
from cayu.providers.openai_subscription import (
    OpenAISubscriptionAuthError,
    OpenAISubscriptionCredentials,
    OpenAISubscriptionProvider,
)

_MISSING = object()


class StaticSubscriptionAuth:
    async def credentials(self) -> OpenAISubscriptionCredentials:
        return OpenAISubscriptionCredentials(
            access_token="subscription-access",
            refresh_token="subscription-refresh",
            expires_at=2_000_000_000,
            account_id="acct-cayu",
        )


class RecordingTransport:
    def __init__(self, *, end_turn: object = _MISSING) -> None:
        self.calls: list[dict[str, Any]] = []
        self.closed = False
        self.end_turn = end_turn

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
        response = {
            "id": "resp-subscription",
            "model": "gpt-5.4",
            "status": "completed",
            "output": [],
            "usage": {"input_tokens": 9, "output_tokens": 2, "total_tokens": 11},
        }
        if self.end_turn is not _MISSING:
            response["end_turn"] = self.end_turn
        yield {
            "type": "response.completed",
            "response": response,
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


def test_subscription_provider_projects_privacy_safe_openai_options() -> None:
    provider = OpenAISubscriptionProvider(
        auth=StaticSubscriptionAuth(),
        transport=RecordingTransport(),
    )
    request = ModelRequest(
        model="gpt-5.4",
        messages=[Message.text("user", "Say hello")],
        options={
            "openai": {
                "reasoning_effort": "medium",
                "metadata": {"private": "provider-option-secret"},
            }
        },
    )

    assert provider.request_footprint_options(request) == {"openai": {"reasoning_effort": "medium"}}
    assert provider.request_fingerprint_options(request) == {
        "openai": {
            "reasoning_effort": "medium",
            "metadata": {"private": "provider-option-secret"},
        }
    }


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
    assert call["headers"]["user-agent"] == f"cayu/{__version__}"
    assert call["payload"]["store"] is False
    assert call["payload"]["stream"] is True


def test_subscription_provider_projects_native_hosted_web_search() -> None:
    transport = RecordingTransport()
    provider = OpenAISubscriptionProvider(auth=StaticSubscriptionAuth(), transport=transport)
    request = ModelRequest(
        model="gpt-5.6-luna",
        messages=[Message.text("user", "Search official OpenAI documentation")],
        hosted_tools=(
            OpenAIWebSearch(
                search_context_size="low",
                external_web_access=False,
                allowed_domains=("openai.com",),
            ),
        ),
    )

    async def collect():
        return [event async for event in provider.stream(request)]

    asyncio.run(collect())

    assert transport.calls[0]["payload"]["tools"] == [
        {
            "type": "web_search",
            "search_context_size": "low",
            "external_web_access": False,
            "filters": {"allowed_domains": ["openai.com"]},
        }
    ]
    assert transport.calls[0]["payload"]["include"] == [
        "reasoning.encrypted_content",
        "web_search_call.action.sources",
    ]


def test_subscription_provider_maps_backend_hosted_tool_rejection_to_capability_error() -> None:
    class RejectingTransport(RecordingTransport):
        async def stream_response_events(self, **kwargs):
            self.calls.append(dict(kwargs))
            yield {
                "type": "response.failed",
                "response": {
                    "error": {
                        "type": "invalid_request_error",
                        "code": "unsupported_tool",
                        "param": "tools[0].type",
                        "message": "raw backend detail must not escape",
                    }
                },
            }

    provider = OpenAISubscriptionProvider(
        auth=StaticSubscriptionAuth(),
        transport=RejectingTransport(),
    )
    request = ModelRequest(
        model="gpt-5.6-luna",
        messages=[Message.text("user", "Search")],
        hosted_tools=(OpenAIWebSearch(),),
    )

    async def collect() -> None:
        with pytest.raises(
            HostedToolCapabilityError,
            match="experimental OpenAI subscription backend rejected hosted web search",
        ) as raised:
            _ = [event async for event in provider.stream(request)]
        assert "raw backend detail" not in str(raised.value)

    asyncio.run(collect())


@pytest.mark.parametrize(
    ("param", "expect_capability_error"),
    [
        ("tools[0].type", True),
        ("reasoning.effort", False),
    ],
)
def test_subscription_provider_requires_tool_provenance_for_generic_value_rejection(
    param: str,
    expect_capability_error: bool,
) -> None:
    class RejectingTransport(RecordingTransport):
        async def stream_response_events(self, **kwargs):
            self.calls.append(dict(kwargs))
            yield {
                "type": "response.failed",
                "response": {
                    "error": {
                        "type": "invalid_request_error",
                        "code": "invalid_value",
                        "param": param,
                        "message": "raw backend detail must not escape",
                    }
                },
            }

    provider = OpenAISubscriptionProvider(
        auth=StaticSubscriptionAuth(),
        transport=RejectingTransport(),
    )
    request = ModelRequest(
        model="gpt-5.6-luna",
        messages=[Message.text("user", "Search")],
        hosted_tools=(OpenAIWebSearch(),),
    )

    async def collect():
        return [event async for event in provider.stream(request)]

    if expect_capability_error:
        with pytest.raises(
            HostedToolCapabilityError,
            match="experimental OpenAI subscription backend rejected hosted web search",
        ):
            asyncio.run(collect())
        return

    events = asyncio.run(collect())
    assert [event.type for event in events] == [ModelStreamEventType.ERROR]
    assert events[0].payload["error"] == "OpenAI subscription provider failed."
    assert "raw backend detail" not in str(events[0].payload)


def test_subscription_provider_preserves_codex_end_turn_false() -> None:
    provider = OpenAISubscriptionProvider(
        auth=StaticSubscriptionAuth(),
        transport=RecordingTransport(end_turn=False),
    )
    request = ModelRequest(
        model="gpt-5.4",
        messages=[Message.text("user", "Continue until done")],
    )

    async def collect():
        return [event async for event in provider.stream(request)]

    events = asyncio.run(collect())

    assert events[-1].completion is not None
    assert events[-1].completion.end_turn is False


@pytest.mark.parametrize("end_turn", [True, None])
def test_subscription_provider_preserves_other_valid_codex_end_turn_values(
    end_turn: bool | None,
) -> None:
    provider = OpenAISubscriptionProvider(
        auth=StaticSubscriptionAuth(),
        transport=RecordingTransport(end_turn=end_turn),
    )
    request = ModelRequest(
        model="gpt-5.4",
        messages=[Message.text("user", "Continue until done")],
    )

    async def collect():
        return [event async for event in provider.stream(request)]

    events = asyncio.run(collect())

    assert events[-1].completion is not None
    assert events[-1].completion.end_turn is end_turn


@pytest.mark.parametrize("end_turn", ["false", 0, 1, [], {}])
def test_subscription_provider_rejects_malformed_codex_end_turn(end_turn: object) -> None:
    provider = OpenAISubscriptionProvider(
        auth=StaticSubscriptionAuth(),
        transport=RecordingTransport(end_turn=end_turn),
    )
    request = ModelRequest(
        model="gpt-5.4",
        messages=[Message.text("user", "Continue until done")],
    )

    async def collect():
        return [event async for event in provider.stream(request)]

    events = asyncio.run(collect())

    assert events[-1].type == ModelStreamEventType.ERROR
    assert events[-1].payload["error"] == "OpenAI subscription provider failed."
    assert events[-1].payload["error_type"] == "OpenAIProtocolError"


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
