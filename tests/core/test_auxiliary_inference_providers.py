from __future__ import annotations

import asyncio
import warnings
from importlib import import_module
from pathlib import Path

import pytest

from cayu import (
    AgentSpec,
    AuxiliaryInferencePolicy,
    CayuApp,
    EventType,
    InferenceLimits,
    RunRequest,
    SQLiteSessionStore,
    Tool,
    ToolResult,
    ToolSpec,
)
from cayu.evals.testing import ScriptedModelProvider
from cayu.messages import Message
from cayu.providers import (
    AnthropicProvider,
    BedrockProvider,
    ChatCompletionsProvider,
    ModelProvider,
    ModelRequest,
    ModelStreamEvent,
    OpenAIProvider,
    OpenAISubscriptionProvider,
    OpenAIWebSearch,
    VertexProvider,
    build_anthropic_payload,
    build_bedrock_converse_payload,
    build_chat_completions_payload,
    build_openai_payload,
)
from cayu.providers.base import (
    AuxiliaryInferenceUnsupportedError,
    TargetedToolProjectionRequest,
    ToolDiscoveryProjectionRequest,
)


class NoIO:
    def __getattr__(self, name):
        raise AssertionError("Request preparation must not access credentials or transport")


def provider_for(kind):
    if kind == "openai":
        return OpenAIProvider(api_key="test-only", transport=NoIO())
    if kind == "subscription":
        return OpenAISubscriptionProvider(auth=NoIO(), transport=NoIO())
    if kind == "chat":
        return ChatCompletionsProvider(name="custom_gateway", api_key="test-only", transport=NoIO())
    if kind == "anthropic":
        return AnthropicProvider(api_key="test-only", transport=NoIO())
    if kind == "vertex":
        return VertexProvider(project_id="test-project", credentials=NoIO(), transport=NoIO())
    if kind == "bedrock":
        return BedrockProvider(client=NoIO())
    return ScriptedModelProvider([])


KINDS = ("openai", "subscription", "chat", "anthropic", "vertex", "bedrock", "scripted")


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("projection", ["targeted", "discovery"])
def test_direct_preparation_rejects_projection_before_serialization(
    kind, projection, caplog, capsys
):
    canary = "rejected-projection-secret-canary"

    class Rejected:
        def __repr__(self):
            return canary

        def __str__(self):
            return canary

    request = ModelRequest(model="model", messages=[])
    if projection == "targeted":
        request.targeted_tool_projection = TargetedToolProjectionRequest.model_construct(
            marker_id=Rejected(), tools=()
        )
    else:
        request.tool_discovery_projection = ToolDiscoveryProjectionRequest.model_construct(
            generation_id=Rejected()
        )
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        with pytest.raises(ValueError, match="cannot supply") as caught:
            provider_for(kind).prepare_auxiliary_request(request, max_output_tokens=10)
    output = capsys.readouterr()
    assert canary not in str(caught.value) + repr(caught.value)
    assert canary not in caplog.text + output.out + output.err
    assert all(canary not in str(item.message) for item in captured)


@pytest.mark.parametrize("kind", KINDS)
def test_preparation_preserves_request_and_bounds_effective_payload_without_io(kind):
    provider = provider_for(kind)
    request = ModelRequest(model="test-model", messages=[Message.text("user", "Summarize this")])
    prepared = provider.prepare_auxiliary_request(request, max_output_tokens=257)
    assert prepared.model == request.model
    assert prepared.messages == request.messages
    assert prepared.messages is not request.messages
    assert prepared.messages[0] is not request.messages[0]
    assert not request.options
    if kind in {"openai", "subscription"}:
        assert build_openai_payload(prepared)["max_output_tokens"] == 257
    elif kind == "chat":
        assert (
            build_chat_completions_payload(prepared, options_key=provider.name)[
                "max_completion_tokens"
            ]
            == 257
        )
    elif kind in {"anthropic", "vertex"}:
        assert build_anthropic_payload(prepared, default_max_tokens=4096)["max_tokens"] == 257
    elif kind == "bedrock":
        assert (
            build_bedrock_converse_payload(prepared, default_max_tokens=4096)["inferenceConfig"][
                "maxTokens"
            ]
            == 257
        )
    else:
        assert prepared.options == {"scripted": {"max_output_tokens": 257}}
        assert not provider.requests


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize(
    "fields",
    [
        {"options": {"thinking": {"budget_tokens": 999999}}},
        {"options": {"openai": {"max_output_tokens": 999999}}},
        {"tools": [{"name": "hidden_dispatch", "input_schema": {}}]},
        {"hosted_tools": (OpenAIWebSearch(),)},
    ],
)
def test_preparation_rejects_raw_control_options_and_tools(kind, fields):
    request = ModelRequest(model="test-model", messages=[], **fields)
    with pytest.raises(ValueError, match="cannot supply"):
        provider_for(kind).prepare_auxiliary_request(request, max_output_tokens=257)


@pytest.mark.parametrize("value", [True, False, 0, -1, "257", 2**64, 1.5])
def test_preparation_rejects_invalid_output_limit(value):
    with pytest.raises(ValueError, match="positive durable integer"):
        provider_for("scripted").prepare_auxiliary_request(
            ModelRequest(model="test-model", messages=[]), max_output_tokens=value
        )


@pytest.mark.parametrize("kind", ["openai", "scripted"])
def test_background_mode_cannot_opt_in(kind):
    provider = (
        OpenAIProvider(api_key="test-only", background=True)
        if kind == "openai"
        else ScriptedModelProvider([], background=True)
    )
    with pytest.raises(AuxiliaryInferenceUnsupportedError) as failure:
        provider.prepare_auxiliary_request(
            ModelRequest(model="test-model", messages=[]), max_output_tokens=1
        )
    assert failure.value.retryable is False
    assert failure.value.provider == provider.name


def test_custom_provider_default_is_explicitly_unsupported():
    class Custom(ModelProvider):
        name = "custom"

        async def stream(self, request):
            raise AssertionError("Must not dispatch")
            yield

    with pytest.raises(AuxiliaryInferenceUnsupportedError):
        Custom().prepare_auxiliary_request(
            ModelRequest(model="test-model", messages=[]), max_output_tokens=1
        )


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
def test_public_custom_provider_remains_ordinary_only(sqlite_resources, backend):
    bounds = InferenceLimits(max_input_tokens=10, max_output_tokens=10, timeout_seconds=10)
    refusals = []

    class Custom(ModelProvider):
        name = "custom"

        def __init__(self):
            self.requests = []

        async def stream(self, request):
            self.requests.append(request.model_copy(deep=True))
            assert request.messages != [Message.text("user", "nested")]
            if len(self.requests) == 1:
                yield ModelStreamEvent.tool_call(name="summarize", id="parent", arguments={})
            else:
                yield ModelStreamEvent.text_delta("done")
            yield ModelStreamEvent.completed({"usage": {"input_tokens": 1, "output_tokens": 1}})

    class Summarize(Tool):
        spec = ToolSpec(
            name="summarize",
            description="Request managed inference from an ordinary-only provider",
            input_schema={"type": "object"},
            auxiliary_inference=AuxiliaryInferencePolicy(limits=bounds, purposes=("tool.summary",)),
        )

        async def run(self, ctx, args):
            assert ctx.inference is not None
            with pytest.raises(AuxiliaryInferenceUnsupportedError) as failure:
                await ctx.inference.invoke(
                    ModelRequest(model="model", messages=[Message.text("user", "nested")]),
                    purpose="tool.summary",
                    limits=bounds,
                )
            assert failure.value.provider == "custom" and failure.value.retryable is False
            refusals.append(failure.value)
            return ToolResult(content="managed inference unsupported")

    async def scenario():
        path = sqlite_resources.path("ordinary-only.sqlite")
        store = sqlite_resources.own(SQLiteSessionStore(path)) if backend == "sqlite" else None
        try:
            app = CayuApp(session_store=store, enable_logging=False)
            provider = Custom()
            app.register_provider(provider, default=True)
            app.register_agent(AgentSpec(name="assistant", model="model"), tools=[Summarize()])
            events = [
                event
                async for event in app.run(
                    RunRequest(
                        session_id="ordinary-only",
                        agent_name="assistant",
                        messages=[Message.text("user", "go")],
                    )
                )
            ]
            assert events[-1].type is EventType.SESSION_COMPLETED
            assert len(refusals) == 1 and len(provider.requests) == 2
            if store is not None:
                await store.close()
                store = sqlite_resources.own(SQLiteSessionStore(path))
                app = CayuApp(session_store=store, enable_logging=False)
            stored = await app.session_store.load_events("ordinary-only")
            assert not any(
                event.type
                in {
                    EventType.MODEL_AUXILIARY_ATTEMPT_STARTED,
                    EventType.MODEL_AUXILIARY_ATTEMPT_SETTLED,
                }
                for event in events + stored
            )
            assert (
                await app.session_store.load_active_model_completion_stage("ordinary-only") is None
            )
            usage = await app.get_session_usage("ordinary-only")
            assert usage.model_steps == 2 and usage.usage.total_tokens == 4
        finally:
            if store is not None:
                await store.close()

    async def run():
        async with sqlite_resources:
            await scenario()

    asyncio.run(run())


@pytest.mark.parametrize(
    "module",
    ["examples.tool_result_projection_live", "examples.prompt_cache_compaction.scenario"],
)
def test_worked_example_wrappers_preserve_auxiliary_preparation(module, monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2] / "examples"))
    wrapper = import_module(module).RecordingProvider(provider_for("chat"))
    request = ModelRequest(model="test-model", messages=[])
    prepared = wrapper.prepare_auxiliary_request(request, max_output_tokens=257)
    assert prepared.options == {"custom_gateway": {"max_completion_tokens": 257}}
    assert not wrapper.requests
    assert not request.options


def test_scripted_actual_usage_is_not_clipped_to_declared_limit():
    provider = ScriptedModelProvider([ModelStreamEvent.completed({"usage": {"output_tokens": 50}})])
    prepared = provider.prepare_auxiliary_request(
        ModelRequest(model="test-model", messages=[]), max_output_tokens=1
    )

    async def collect():
        return [event async for event in provider.runtime_stream(prepared)]

    events = asyncio.run(collect())
    assert events[-1].payload["usage"]["output_tokens"] == 50
    assert len(provider.requests) == 1
