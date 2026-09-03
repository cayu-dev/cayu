from __future__ import annotations

import asyncio
import traceback
from collections.abc import Mapping
from typing import Any

import pytest
from tests.provider_traceback_assertions import is_cayu_source_filename

from cayu import (
    AgentSpec,
    CayuApp,
    EventType,
    Message,
    RetryPolicy,
    RunRequest,
    __version__,
    default_price_book,
)
from cayu.providers import (
    HostedToolCapabilityError,
    ModelContextOverflowError,
    ModelRequest,
    ModelStreamDeadlineError,
    ModelStreamEventType,
    OpenAIWebSearch,
)
from cayu.providers.deadlines import (
    ProviderDeadlineKind,
    ProviderStreamDeadlineEvidence,
)
from cayu.providers.openai import OpenAIAPIError
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
        transport_idle_timeout_s: float,
        protocol_idle_timeout_s: float,
        semantic_progress_timeout_s: float,
        absolute_stream_timeout_s: float,
    ):
        self.calls.append(
            {
                "url": url,
                "headers": dict(headers),
                "payload": dict(payload),
                "timeout_s": timeout_s,
                "transport_idle_timeout_s": transport_idle_timeout_s,
                "protocol_idle_timeout_s": protocol_idle_timeout_s,
                "semantic_progress_timeout_s": semantic_progress_timeout_s,
                "absolute_stream_timeout_s": absolute_stream_timeout_s,
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


@pytest.mark.parametrize("timeout_s", [float("nan"), float("inf"), float("-inf"), 10**1000])
def test_subscription_provider_rejects_nonfinite_timeout(timeout_s: int | float) -> None:
    with pytest.raises(ValueError, match="timeout_s"):
        OpenAISubscriptionProvider(auth=StaticSubscriptionAuth(), timeout_s=timeout_s)


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


@pytest.mark.anyio
async def test_subscription_provider_uses_openai_pricing_identity() -> None:
    provider = OpenAISubscriptionProvider(auth=StaticSubscriptionAuth())
    request = ModelRequest(
        model="gpt-5.4",
        messages=[Message.text("user", "Say hello")],
    )

    assert provider.name == "openai_subscription"
    assert provider.billing_provider_name == "openai"
    identity = await provider.billing_identity_for_request(request)
    assert identity.provider_name == "openai"
    assert identity.resource_id == "gpt-5.4"
    assert identity.request_evidence == {
        "access_mode": "chatgpt_subscription",
        "pricing_basis": "openai_api_equivalent_estimate",
    }


@pytest.mark.anyio
async def test_subscription_runtime_retains_openai_pricing_identity() -> None:
    provider = OpenAISubscriptionProvider(
        auth=StaticSubscriptionAuth(),
        transport=RecordingTransport(),
    )
    app = CayuApp()
    app.register_provider(provider, default=True)
    app.register_agent(AgentSpec(name="assistant", model="gpt-5.6-sol"))

    events = [
        event
        async for event in app.run(
            RunRequest(
                agent_name="assistant",
                session_id="subscription-pricing-identity",
                messages=[Message.text("user", "Say hello")],
                max_steps=1,
            )
        )
    ]

    completed = next(event for event in events if event.type == EventType.MODEL_COMPLETED)
    assert completed.payload["provider_name"] == "openai_subscription"
    assert completed.payload["billing_identity"] == {
        "provider_name": "openai",
        "resource_id": "gpt-5.6-sol",
        "request_evidence": {
            "access_mode": "chatgpt_subscription",
            "pricing_basis": "openai_api_equivalent_estimate",
        },
        "completion_evidence": {},
        "pricing_contexts": [],
    }
    cost = await app.get_session_cost(
        "subscription-pricing-identity",
        default_price_book(),
    )
    assert cost.line_items[0].priced is True
    assert cost.line_items[0].provider_name == "openai_subscription"
    assert cost.line_items[0].pricing_provider_name == "openai"


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


@pytest.mark.anyio
async def test_subscription_runtime_retries_generic_stream_error_once_then_completes() -> None:
    canary = "raw-provider-error-canary-0123456789"

    class SequencedTransport(RecordingTransport):
        async def stream_response_events(self, **kwargs: Any):
            self.calls.append(dict(kwargs))
            if len(self.calls) == 1:
                yield {
                    "type": "error",
                    "message": canary,
                }
                return
            yield {"type": "response.output_text.delta", "delta": "recovered"}
            yield {
                "type": "response.completed",
                "response": {
                    "id": "resp-subscription-retry",
                    "model": "gpt-5.4",
                    "status": "completed",
                    "output": [],
                    "usage": {},
                },
            }

    transport = SequencedTransport()
    provider = OpenAISubscriptionProvider(auth=StaticSubscriptionAuth(), transport=transport)
    app = CayuApp()
    app.register_provider(provider, default=True)
    app.register_agent(AgentSpec(name="assistant", model="gpt-5.4"))

    events = [
        event
        async for event in app.run(
            RunRequest(
                agent_name="assistant",
                messages=[Message.text("user", "hello")],
                max_steps=1,
                retry_policy=RetryPolicy(
                    max_attempts=5,
                    max_unknown_attempts=2,
                    initial_delay_s=0.0,
                ),
            )
        )
    ]

    assert len(transport.calls) == 2
    model_error = next(event for event in events if event.type == EventType.MODEL_ERROR)
    retry = next(event for event in events if event.type == EventType.MODEL_RETRY)
    assert model_error.payload["attempt"] == 1
    assert model_error.payload["max_attempts"] == 5
    assert model_error.payload["effective_max_attempts"] == 2
    assert model_error.payload["reason"] == "unknown_provider"
    assert model_error.payload["provider_error_type"] == "error"
    assert retry.payload["reason"] == "unknown_provider"
    assert retry.payload["max_attempts"] == 5
    assert retry.payload["effective_max_attempts"] == 2
    assert retry.payload["provider_error_type"] == "error"
    assert retry.payload["delay_seconds"] == 0.0
    assert canary not in repr([event.model_dump(mode="json") for event in events])
    assert events[-1].type == EventType.SESSION_COMPLETED


@pytest.mark.anyio
async def test_subscription_runtime_unknown_stream_error_uses_nested_attempt_cap() -> None:
    class FailingTransport(RecordingTransport):
        async def stream_response_events(self, **kwargs: Any):
            self.calls.append(dict(kwargs))
            yield {"type": "error", "message": "provider detail must stay private"}

    transport = FailingTransport()
    provider = OpenAISubscriptionProvider(auth=StaticSubscriptionAuth(), transport=transport)
    app = CayuApp()
    app.register_provider(provider, default=True)
    app.register_agent(AgentSpec(name="assistant", model="gpt-5.4"))

    events = [
        event
        async for event in app.run(
            RunRequest(
                agent_name="assistant",
                messages=[Message.text("user", "hello")],
                max_steps=1,
                retry_policy=RetryPolicy(
                    max_attempts=5,
                    max_unknown_attempts=2,
                    initial_delay_s=0.0,
                ),
            )
        )
    ]

    model_errors = [event for event in events if event.type == EventType.MODEL_ERROR]
    retries = [event for event in events if event.type == EventType.MODEL_RETRY]
    assert len(transport.calls) == 2
    assert [event.payload["attempt"] for event in model_errors] == [1, 2]
    assert [event.payload["max_attempts"] for event in model_errors] == [5, 5]
    assert [event.payload["effective_max_attempts"] for event in model_errors] == [2, 2]
    assert [event.payload["reason"] for event in model_errors] == [
        "unknown_provider",
        "unknown_provider",
    ]
    assert len(retries) == 1
    assert retries[0].payload["reason"] == "unknown_provider"
    assert retries[0].payload["max_attempts"] == 5
    assert retries[0].payload["effective_max_attempts"] == 2
    assert events[-1].type == EventType.SESSION_FAILED


@pytest.mark.anyio
@pytest.mark.parametrize("status_code", [429, 500, 502, 503, 504])
async def test_subscription_runtime_retries_typed_sse_status(status_code: int) -> None:
    class SequencedTransport(RecordingTransport):
        async def stream_response_events(self, **kwargs: Any):
            self.calls.append(dict(kwargs))
            if len(self.calls) == 1:
                yield {
                    "type": "error",
                    "status_code": status_code,
                    "message": "raw provider detail",
                }
                return
            yield {
                "type": "response.completed",
                "response": {
                    "id": f"resp-subscription-{status_code}",
                    "model": "gpt-5.4",
                    "status": "completed",
                    "output": [],
                    "usage": {},
                },
            }

    transport = SequencedTransport()
    provider = OpenAISubscriptionProvider(auth=StaticSubscriptionAuth(), transport=transport)
    app = CayuApp()
    app.register_provider(provider, default=True)
    app.register_agent(AgentSpec(name="assistant", model="gpt-5.4"))

    events = [
        event
        async for event in app.run(
            RunRequest(
                agent_name="assistant",
                messages=[Message.text("user", "hello")],
                max_steps=1,
                retry_policy=RetryPolicy(max_attempts=2, initial_delay_s=0.0),
            )
        )
    ]

    retry = next(event for event in events if event.type == EventType.MODEL_RETRY)
    assert len(transport.calls) == 2
    assert retry.payload["status_code"] == status_code
    assert retry.payload["reason"] == "http_status"
    assert retry.payload["provider_error_type"] == "error"
    assert events[-1].type == EventType.SESSION_COMPLETED


@pytest.mark.anyio
async def test_subscription_runtime_honors_bounded_typed_retry_after() -> None:
    class SequencedTransport(RecordingTransport):
        async def stream_response_events(self, **kwargs: Any):
            self.calls.append(dict(kwargs))
            if len(self.calls) == 1:
                raise OpenAIAPIError(
                    "raw rate-limit detail",
                    status_code=429,
                    error_type="rate_limit_error",
                    retryable=True,
                    retry_after_s=120.0,
                )
            yield {
                "type": "response.completed",
                "response": {
                    "id": "resp-subscription-retry-after",
                    "model": "gpt-5.4",
                    "status": "completed",
                    "output": [],
                    "usage": {},
                },
            }

    transport = SequencedTransport()
    provider = OpenAISubscriptionProvider(auth=StaticSubscriptionAuth(), transport=transport)
    app = CayuApp()
    app.register_provider(provider, default=True)
    app.register_agent(AgentSpec(name="assistant", model="gpt-5.4"))

    events = [
        event
        async for event in app.run(
            RunRequest(
                agent_name="assistant",
                messages=[Message.text("user", "hello")],
                max_steps=1,
                retry_policy=RetryPolicy(
                    max_attempts=2,
                    initial_delay_s=0.0,
                    max_delay_s=0.01,
                ),
            )
        )
    ]

    model_error = next(event for event in events if event.type == EventType.MODEL_ERROR)
    retry = next(event for event in events if event.type == EventType.MODEL_RETRY)
    assert len(transport.calls) == 2
    assert model_error.payload["retry_after_s"] == 120.0
    assert retry.payload["delay_seconds"] == 0.01
    assert retry.payload["provider_error_type"] == "rate_limit_error"
    assert "raw rate-limit detail" not in repr([event.model_dump(mode="json") for event in events])
    assert events[-1].type == EventType.SESSION_COMPLETED


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("error_type", "error_code", "status_code"),
    [
        ("authentication_error", None, 401),
        ("permission_error", None, 403),
        ("insufficient_quota", "insufficient_quota", 429),
        ("invalid_request_error", "bad_request", 400),
        ("invalid_request_error", "unsupported_tool", 400),
        ("not_found_error", None, 404),
    ],
)
async def test_subscription_runtime_keeps_known_permanent_sse_errors_terminal(
    error_type: str,
    error_code: str | None,
    status_code: int,
) -> None:
    class FailingTransport(RecordingTransport):
        async def stream_response_events(self, **kwargs: Any):
            self.calls.append(dict(kwargs))
            yield {
                "type": "response.failed",
                "response": {
                    "status_code": status_code,
                    "error": {
                        "type": error_type,
                        "code": error_code,
                        "message": "raw permanent provider detail",
                    },
                },
            }

    transport = FailingTransport()
    provider = OpenAISubscriptionProvider(auth=StaticSubscriptionAuth(), transport=transport)
    app = CayuApp()
    app.register_provider(provider, default=True)
    app.register_agent(AgentSpec(name="assistant", model="gpt-5.4"))

    events = [
        event
        async for event in app.run(
            RunRequest(
                agent_name="assistant",
                messages=[Message.text("user", "hello")],
                max_steps=1,
                retry_policy=RetryPolicy(
                    max_attempts=5,
                    max_unknown_attempts=2,
                    initial_delay_s=0.0,
                ),
            )
        )
    ]

    model_error = next(event for event in events if event.type == EventType.MODEL_ERROR)
    assert len(transport.calls) == 1
    assert EventType.MODEL_RETRY not in [event.type for event in events]
    assert model_error.payload["status_code"] == status_code
    assert model_error.payload["retryable"] is False
    assert model_error.payload["provider_error_type"] == error_type
    assert "raw permanent provider detail" not in repr(
        [event.model_dump(mode="json") for event in events]
    )
    assert events[-1].type == EventType.SESSION_FAILED


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
    assert exc_info.value.provider == "openai_subscription"
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


@pytest.mark.anyio
async def test_subscription_deadline_uses_execution_provider_identity() -> None:
    evidence = ProviderStreamDeadlineEvidence(
        deadline_kind=ProviderDeadlineKind.TRANSPORT_IDLE,
        configured_timeout_s=0.02,
        elapsed_s=0.02,
        last_progress_kind=None,
        last_progress_elapsed_s=None,
        last_progress_at=None,
    )

    class DeadlineTransport(RecordingTransport):
        async def stream_response_events(self, **_kwargs: Any):
            raise ModelStreamDeadlineError(
                provider="openai",
                evidence=evidence,
            )
            yield  # pragma: no cover

    provider = OpenAISubscriptionProvider(
        auth=StaticSubscriptionAuth(),
        transport=DeadlineTransport(),
    )
    request = ModelRequest(
        model="gpt-5.4",
        messages=[Message.text("user", "Say hello")],
    )

    direct_events = [event async for event in provider.runtime_stream(request)]

    assert [event.type for event in direct_events] == [ModelStreamEventType.ERROR]
    assert direct_events[0].payload["provider"] == "openai_subscription"
    assert direct_events[0].payload["provider_deadline_kind"] == "transport_idle"

    app = CayuApp(enable_logging=False)
    app.register_provider(provider, default=True)
    app.register_agent(AgentSpec(name="assistant", model="gpt-5.4"))
    session_id = "subscription-deadline-execution-provider"
    with pytest.raises(ModelStreamDeadlineError):
        [
            event
            async for event in app.run(
                RunRequest(
                    agent_name="assistant",
                    session_id=session_id,
                    messages=[Message.text("user", "Say hello")],
                )
            )
        ]

    durable_events = await app.session_store.load_events(session_id)
    model_error = next(event for event in durable_events if event.type is EventType.MODEL_ERROR)
    assert model_error.payload["provider"] == "openai_subscription"
    assert model_error.payload["model_attempt_id"]
