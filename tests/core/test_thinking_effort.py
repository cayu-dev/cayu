from __future__ import annotations

import asyncio
import json
from typing import get_args

import httpx
import pytest
from pydantic import ValidationError

from cayu import (
    AgentSpec,
    AnthropicProvider,
    CayuConfig,
    ChatCompletionsProvider,
    Message,
    OpenAIProvider,
    ResumeRequest,
    RunDefaults,
    RunRequest,
    ThinkingConfig,
)
from cayu.core.thinking import ThinkingEffort
from cayu.providers import ModelRequest, ModelStreamEventType
from cayu.providers.anthropic import build_anthropic_payload
from cayu.providers.bedrock import BedrockProvider, build_bedrock_converse_payload
from cayu.providers.chat_completions import HttpxChatCompletionsTransport
from cayu.providers.openai import HttpxOpenAITransport, build_openai_payload
from cayu.providers.openai_subscription import OpenAISubscriptionProvider
from cayu.workflows import StepRunOptions

EFFORTS = get_args(ThinkingEffort)


def request(model: str, effort: str, **options) -> ModelRequest:
    return ModelRequest(
        model=model,
        messages=[Message.text("user", "reply ok")],
        options={"thinking": {"effort": effort}, **options},
    )


@pytest.mark.parametrize("effort", EFFORTS)
def test_effort_owned_public_and_durable_round_trips(effort) -> None:
    from cayu.runtime.approvals import PendingToolApproval, PendingToolCallApproval
    from cayu.runtime.dispatch import DispatchRequest, copy_dispatch_request

    config = ThinkingConfig(effort=effort, include_in_transcript=False)
    values = [
        config,
        AgentSpec(name="a", model="gpt-5.6-luna", thinking=config),
        CayuConfig(run=RunDefaults(thinking=config)),
        RunRequest(agent_name="a", messages=[Message.text("user", "hi")], thinking=config),
        ResumeRequest(session_id="s", messages=[Message.text("user", "hi")], thinking=config),
        StepRunOptions(thinking=config),
        PendingToolApproval(
            approval_id="a",
            model_step_id=f"mstep_{'1' * 32}",
            model_attempt_id=f"matt_{'2' * 32}",
            tool_round_id=f"tround_{'3' * 32}",
            tool_call_id="t",
            tool_name="x",
            agent_name="a",
            publish_arguments=True,
            tool_calls=[PendingToolCallApproval(tool_call_id="t", tool_name="x")],
            thinking=config,
        ),
    ]
    for value in values:
        restored = type(value).model_validate_json(value.model_dump_json())
        assert restored == value
    dispatch = DispatchRequest(
        session_id="s", dispatch_id="d", messages=[Message.text("user", "hi")], thinking=config
    )
    assert copy_dispatch_request(dispatch).thinking == config
    with pytest.raises(ValidationError):
        config.effort = "low"
    with pytest.raises(ValidationError, match="enabled=False"):
        ThinkingConfig(effort=effort, enabled=False)
    with pytest.raises(ValidationError, match="mutually exclusive"):
        ThinkingConfig(effort=effort, max_tokens=1024)


@pytest.mark.parametrize("effort", ["xlow", "ultra", "MAX", "", 3, True])
def test_unverified_and_invalid_vocabulary_is_rejected(effort) -> None:
    with pytest.raises(ValidationError):
        ThinkingConfig(effort=effort)


@pytest.mark.parametrize("effort", EFFORTS)
@pytest.mark.parametrize("protocol", ["responses", "chat"])
def test_real_http_transport_emits_exact_effort(protocol, effort) -> None:
    # Unknown endpoint/model compatibility is intentional: local wire fidelity is
    # provable without claiming upstream acceptance. No socket or credential access.
    async def exercise():
        calls = []

        def handler(req):
            calls.append((str(req.url), json.loads(req.content)))
            if protocol == "responses":
                body = 'data: {"type":"response.completed","response":{"id":"r","status":"completed","output":[]}}\n\n'
            else:
                body = 'data: {"choices":[{"delta":{"content":"ok"},"finish_reason":"stop"}]}\n\ndata: [DONE]\n\n'

            class ResponseStream(httpx.AsyncByteStream):
                async def __aiter__(self):
                    yield body.encode()

            return httpx.Response(
                200, stream=ResponseStream(), headers={"content-type": "text/event-stream"}
            )

        transport = (
            HttpxOpenAITransport() if protocol == "responses" else HttpxChatCompletionsTransport()
        )
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            transport._client._client = client
            provider = (OpenAIProvider if protocol == "responses" else ChatCompletionsProvider)(
                api_key="test-only", base_url="https://compatible.invalid", transport=transport
            )
            events = [
                event async for event in provider.stream(request("arbitrary-deployment", effort))
            ]
        assert events[-1].type == ModelStreamEventType.COMPLETED, events[-1].payload
        assert len(calls) == 1
        url, payload = calls[0]
        assert payload["model"] == "arbitrary-deployment"
        if protocol == "responses":
            assert url.endswith("/v1/responses")
            assert payload["reasoning"]["effort"] == effort
        else:
            assert url.endswith("/chat/completions")
            assert payload["reasoning_effort"] == effort

    asyncio.run(exercise())


def test_gaia_luna_max_compatible_responses_wire() -> None:
    from tests.core.test_openai_provider import RecordingTransport

    async def exercise():
        transport = RecordingTransport(
            stream_events=[
                [
                    {
                        "type": "response.completed",
                        "response": {
                            "id": "r",
                            "status": "completed",
                            "output": [],
                            "model": "backend-reported-alias",
                        },
                    }
                ]
            ]
        )
        provider = OpenAIProvider(
            api_key="unit-test", base_url="https://codex-lb.cayu.ai", transport=transport
        )
        req = request(
            "gpt-5.6-luna", "max", openai={"reasoning": {"effort": "low", "summary": "detailed"}}
        )
        events = [event async for event in provider.stream(req)]
        assert events[-1].type == ModelStreamEventType.COMPLETED, events[-1].payload
        assert len(transport.calls) == 1
        assert transport.calls[0]["url"] == "https://codex-lb.cayu.ai/v1/responses"
        payload = transport.calls[0]["payload"]
        assert payload["model"] == "gpt-5.6-luna"
        assert payload["reasoning"] == {"effort": "max", "summary": "detailed"}
        assert req.options["thinking"]["effort"] == "max"
        assert provider.request_fingerprint_options(req)["openai"]["reasoning"]["effort"] == "max"

    asyncio.run(exercise())


@pytest.mark.parametrize("effort", ["low", "medium", "high", "xhigh", "max"])
@pytest.mark.parametrize("vertex", [False, True])
def test_anthropic_and_vertex_transport_effort(vertex, effort) -> None:
    from tests.core.test_anthropic_provider import RecordingTransport
    from tests.core.test_vertex_provider import _provider

    async def exercise():
        transport = RecordingTransport(
            [{"content": [{"type": "text", "text": "ok"}], "stop_reason": "end_turn"}]
        )
        provider = (
            _provider(transport)
            if vertex
            else AnthropicProvider(api_key="unit-test", transport=transport)
        )
        req = request(
            "claude-opus-4-7",
            effort,
            anthropic={"output_config": {"effort": "low", "format": {"type": "json_schema"}}},
        )
        events = [event async for event in provider.stream(req)]
        assert events[-1].type == ModelStreamEventType.COMPLETED, events[-1].payload
        payload = transport.calls[0]["payload"]
        assert payload["thinking"]["type"] == "adaptive"
        assert payload["output_config"] == {"effort": effort, "format": {"type": "json_schema"}}
        assert (
            provider.request_fingerprint_options(req)["anthropic"]["output_config"]["effort"]
            == effort
        )

    asyncio.run(exercise())


@pytest.mark.parametrize("effort", ["xhigh", "max"])
def test_subscription_transport_preserves_effort(effort) -> None:
    from tests.core.test_openai_subscription_provider import (
        RecordingTransport,
        StaticSubscriptionAuth,
    )

    async def exercise():
        transport = RecordingTransport()
        provider = OpenAISubscriptionProvider(auth=StaticSubscriptionAuth(), transport=transport)
        events = [event async for event in provider.stream(request("gpt-5.6-luna", effort))]
        assert events[-1].type == ModelStreamEventType.COMPLETED, events[-1].payload
        assert transport.calls[0]["payload"]["reasoning"]["effort"] == effort

    asyncio.run(exercise())


@pytest.mark.parametrize(
    ("adapter", "model", "effort"),
    [
        ("openai", "gpt-5.1", "xhigh"),
        ("openai", "gpt-5.5", "max"),
        ("openai", "gpt-5.6-luna", "minimal"),
        ("openai", "gpt-5-pro", "low"),
        ("openai", "gpt-4o", "high"),
        ("chat", "gpt-5-2025-08-07", "none"),
        ("chat", "gemini-2.5-pro", "none"),
        ("chat", "gemini-3-flash-preview", "max"),
        ("anthropic", "claude-opus-4-6", "xhigh"),
        ("anthropic", "claude-opus-4-5-20251101", "high"),
        ("anthropic", "unknown-model-secret", "none"),
        ("anthropic", "claude-opus-4-7", "minimal"),
        ("vertex", "claude-opus-4-6@20260205", "xhigh"),
        ("bedrock", "anthropic.claude-opus-4-6-v1", "max"),
        ("subscription", "gpt-5.1", "max"),
    ],
)
def test_known_incompatibility_fails_before_auth_and_network(adapter, model, effort) -> None:
    from tests.core.test_vertex_provider import _provider

    class NoAccess:
        def __getattr__(self, name):
            raise AssertionError("No auth or transport access allowed")

    providers = {
        "openai": lambda: OpenAIProvider(api_key="secret-key", transport=NoAccess()),
        "chat": lambda: ChatCompletionsProvider(api_key="secret-key", transport=NoAccess()),
        "anthropic": lambda: AnthropicProvider(api_key="secret-key", transport=NoAccess()),
        "vertex": lambda: _provider(NoAccess()),
        "bedrock": lambda: BedrockProvider(region_name="us-east-1", client=NoAccess()),
        "subscription": lambda: OpenAISubscriptionProvider(auth=NoAccess(), transport=NoAccess()),
    }
    provider = providers[adapter]()
    req = request(model, effort)

    async def exercise():
        with pytest.raises(ValueError, match="Local thinking incompatibility") as caught:
            _ = [event async for event in provider.stream(req)]
        assert "secret" not in str(caught.value)
        assert len(str(caught.value)) < 400

    asyncio.run(exercise())
    with pytest.raises(ValueError, match="Local thinking incompatibility"):
        provider.request_fingerprint_options(req)


@pytest.mark.parametrize(
    "builder", [build_openai_payload, build_anthropic_payload, build_bedrock_converse_payload]
)
def test_raw_neutral_options_cannot_bypass_contradictory_validation(builder) -> None:
    req = ModelRequest(
        model="arbitrary",
        messages=[Message.text("user", "hi")],
        options={"thinking": {"enabled": False, "effort": "max"}},
    )
    with pytest.raises(ValueError, match="Invalid thinking configuration"):
        builder(req)


@pytest.mark.parametrize("effort", ["none", "minimal", "xhigh", "max"])
def test_run_override_reaches_model_request(effort) -> None:
    from tests.core.test_thinking import _run

    from cayu.core.thinking import thinking_config_payload

    provider, _events, _transcript = asyncio.run(
        _run(
            agent_thinking=ThinkingConfig(effort="high"),
            run_thinking=ThinkingConfig(effort=effort),
        )
    )
    assert provider.options["thinking"] == thinking_config_payload(ThinkingConfig(effort=effort))


def test_upstream_rejection_is_redacted_without_effort_fallback() -> None:
    async def exercise():
        calls = []

        def handler(req):
            calls.append(json.loads(req.content))
            return httpx.Response(
                400,
                json={
                    "error": {
                        "message": "secret-key full private request body",
                        "type": "invalid_request_error",
                        "code": "unsupported_value",
                        "param": "reasoning.effort",
                    }
                },
            )

        transport = HttpxOpenAITransport()
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            transport._client._client = client
            provider = OpenAIProvider(
                api_key="secret-key", transport=transport, base_url="https://compatible.invalid"
            )
            events = [event async for event in provider.stream(request("custom-model", "max"))]
        assert len(calls) == 1
        assert calls[0]["reasoning"]["effort"] == "max"
        assert events[-1].type == ModelStreamEventType.ERROR
        assert events[-1].payload["status_code"] == 400
        rendered = json.dumps(events[-1].payload)
        assert "secret-key" not in rendered
        assert "private request body" not in rendered
        assert "Local thinking incompatibility" not in rendered

    asyncio.run(exercise())


@pytest.mark.parametrize(
    ("model", "effort"),
    [
        ("gpt-5", "minimal"),
        ("gpt-5.1", "none"),
        ("gpt-5.5", "xhigh"),
        ("gpt-5.6-luna", "max"),
        ("gemini-2.5-flash", "none"),
        ("gemini-3-flash-preview", "minimal"),
    ],
)
def test_known_models_preserve_native_chat_effort(model, effort) -> None:
    from cayu.providers.chat_completions import build_chat_completions_payload

    payload = build_chat_completions_payload(request(model, effort))
    assert payload["reasoning_effort"] == effort
    assert payload["model"] == model


def test_responses_only_model_rejects_chat_effort() -> None:
    from cayu.providers.chat_completions import build_chat_completions_payload

    with pytest.raises(ValueError, match="Local thinking incompatibility"):
        build_chat_completions_payload(request("gpt-5.1-codex-max", "xhigh"))
