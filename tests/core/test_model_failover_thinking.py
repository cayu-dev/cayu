"""Thinking compatibility through routed admission and real selected requests."""

from __future__ import annotations

import asyncio
import warnings
from typing import Any

import pytest
from tests.core.test_model_failover_stages import _StageMemoryStore, _StageSQLiteStore
from tests.core.test_vertex_provider import _provider as vertex_provider

from cayu import (
    AgentSpec,
    AnthropicProvider,
    CayuApp,
    ChatCompletionsProvider,
    EventQuery,
    EventType,
    Message,
    ModelFailoverPolicy,
    ModelTarget,
    OpenAIProvider,
    RunRequest,
    ScriptedModelProvider,
    ThinkingConfig,
)
from cayu.evals.runtime_replay import _RecordedProvider
from cayu.providers._thinking import copy_preflight_thinking
from cayu.providers.base import ModelProvider, ModelProviderError, ModelStreamEvent
from cayu.providers.bedrock import BedrockProvider
from cayu.providers.openai_subscription import OpenAISubscriptionProvider
from cayu.runtime.retry_policy import RetryPolicy


class _NoAccess:
    def __getattr__(self, name):
        raise AssertionError("Thinking preflight must not contact auth or transport")


def test_recorded_provider_delegates_thinking_without_dispatch():
    class Source(ScriptedModelProvider):
        def preflight_thinking(self, *, model, thinking):
            super().preflight_thinking(model=model, thinking=thinking)
            assert model == "source-model"
            if thinking is not None and thinking.effort == "max":
                raise ValueError("source rejects max")

    source = Source([[ModelStreamEvent.completed()]])
    provider = _RecordedProvider(source, (), (), ())
    provider.preflight_thinking(model="source-model", thinking=ThinkingConfig(effort="high"))
    with pytest.raises(ValueError, match="source rejects max"):
        provider.preflight_thinking(model="source-model", thinking=ThinkingConfig(effort="max"))
    assert source.requests == []


def _provider(adapter):
    # Intentionally poison every SDK member; none may be resolved during preflight.
    no_access: Any = _NoAccess()
    return {
        "openai": lambda: OpenAIProvider(api_key="unit-test", transport=no_access),
        "chat": lambda: ChatCompletionsProvider(api_key="unit-test", transport=no_access),
        "anthropic": lambda: AnthropicProvider(api_key="unit-test", transport=no_access),
        "vertex": lambda: vertex_provider(no_access),
        "bedrock": lambda: BedrockProvider(region_name="us-east-1", client=no_access),
        "subscription": lambda: OpenAISubscriptionProvider(auth=no_access, transport=no_access),
    }[adapter]()


@pytest.mark.parametrize(
    "adapter,model,effort",
    [
        ("openai", "gpt-4o", "high"),
        ("chat", "gpt-5-pro", "low"),
        ("anthropic", "claude-opus-4-6", "xhigh"),
        ("vertex", "claude-opus-4-6@20260205", "xhigh"),
        ("bedrock", "anthropic.claude-opus-4-6-v1", "high"),
        ("subscription", "gpt-5.1", "max"),
    ],
)
def test_bundled_thinking_preflight_is_local_and_preserves_existing_semantics(
    adapter, model, effort
):
    provider = _provider(adapter)
    with pytest.raises(ValueError, match="Local thinking incompatibility"):
        provider.preflight_thinking(model=model, thinking=ThinkingConfig(effort=effort))
    # These are documented best-effort controls, not a new cross-provider scale.
    for thinking in (
        None,
        ThinkingConfig(),
        ThinkingConfig(enabled=False),
        ThinkingConfig(max_tokens=1024),
    ):
        provider.preflight_thinking(model=model, thinking=thinking)
    if adapter != "bedrock":
        # Unknown aliases retain backend-dependent acceptance, not a local allowlist.
        provider.preflight_thinking(model="future-model", thinking=ThinkingConfig(effort="high"))


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
@pytest.mark.parametrize("source", ["request", "agent", "raw-options"])
@pytest.mark.parametrize("backup_kind", ["undeclared", "known-incompatible"])
def test_public_run_rejects_unsupported_fallback_thinking_before_mutation(
    monkeypatch, tmp_path, backend, source, backup_kind
):

    class UndeclaredThinking(ScriptedModelProvider):
        preflight_thinking = ModelProvider.preflight_thinking

    async def scenario():
        store = (
            _StageMemoryStore()
            if backend == "memory"
            else _StageSQLiteStore(tmp_path / "thinking.sqlite")
        )
        primary = ScriptedModelProvider([[ModelStreamEvent.completed()]], name="primary")
        backup = (
            UndeclaredThinking([[ModelStreamEvent.completed()]], name="backup")
            if backup_kind == "undeclared"
            else _provider("openai")
        )
        config = ThinkingConfig(effort="high")
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_provider(primary, default=True)
        app.register_provider(backup)
        app.register_agent(
            AgentSpec(
                name="agent",
                model="primary-model",
                thinking=config if source == "agent" else None,
                provider_options={"thinking": config.model_dump()}
                if source == "raw-options"
                else {},
            )
        )
        try:
            with pytest.raises(ValueError, match="thinking"):
                _ = [
                    event
                    async for event in app.run(
                        RunRequest(
                            agent_name="agent",
                            session_id="unsupported-thinking",
                            messages=[Message.text("user", "hello")],
                            thinking=config if source == "request" else None,
                            failover=ModelFailoverPolicy(
                                fallbacks=(ModelTarget(provider_name=backup.name, model="gpt-4o"),)
                            ),
                        )
                    )
                ]
            assert primary.requests == []
            if isinstance(backup, ScriptedModelProvider):
                assert backup.requests == []
            assert await store.load("unsupported-thinking") is None
            assert await store.load_checkpoint("unsupported-thinking") is None
            assert await store.query_events(EventQuery(session_id="unsupported-thinking")) == []
        finally:
            if isinstance(store, _StageSQLiteStore):
                await store.close()

    asyncio.run(scenario())


@pytest.mark.parametrize("source", ["request", "agent", "raw-options"])
def test_public_failover_thinking_copy_preserves_precedence_and_dispatch(monkeypatch, source):

    class MutatingPreflight(ScriptedModelProvider):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.controls = []

        def preflight_thinking(self, *, model, thinking):
            super().preflight_thinking(model=model, thinking=thinking)
            assert thinking is not None
            self.controls.append(thinking.model_dump())
            object.__setattr__(thinking, "effort", "low")

    def unavailable(_request):
        raise ModelProviderError("unavailable", provider="primary", status_code=503, retryable=True)

    async def scenario():
        primary = MutatingPreflight(response_factory=unavailable, name="primary")
        backup = MutatingPreflight(
            [[ModelStreamEvent.text_delta("done"), ModelStreamEvent.completed()]], name="backup"
        )
        store = _StageMemoryStore()
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_provider(primary, default=True)
        app.register_provider(backup)
        config = ThinkingConfig(effort="high", include_in_transcript=False)
        app.register_agent(
            AgentSpec(
                name="agent",
                model="small",
                thinking=config
                if source == "agent"
                else ThinkingConfig(effort="medium")
                if source == "request"
                else None,
                provider_options={
                    "thinking": config.model_dump()
                    if source == "raw-options"
                    else {"effort": "minimal"}
                },
            )
        )
        events = [
            event
            async for event in app.run(
                RunRequest(
                    agent_name="agent",
                    session_id="thinking-precedence",
                    messages=[Message.text("user", "hello")],
                    thinking=config if source == "request" else None,
                    retry_policy=RetryPolicy(max_attempts=1, initial_delay_s=0),
                    failover=ModelFailoverPolicy(
                        fallbacks=(ModelTarget(provider_name="backup", model="large"),)
                    ),
                )
            )
        ]
        assert events[-1].type is EventType.SESSION_COMPLETED
        assert config.effort == "high"
        for provider in (primary, backup):
            assert len(provider.requests) == 1
            assert provider.requests[0].options["thinking"] == config.model_dump()
            # Initial admission and selected-request validation both ran.
            assert len(provider.controls) >= 2
            assert all(control == config.model_dump() for control in provider.controls)
        checkpoint = await store.load_checkpoint("thinking-precedence")
        assert checkpoint is not None and checkpoint["model_failover"]["candidate_index"] == 1
        assert await store.load_active_model_completion_stage("thinking-precedence") is None

    asyncio.run(scenario())


@pytest.mark.parametrize("field", ["enabled", "effort", "max_tokens", "include_in_transcript"])
def test_thinking_preflight_revalidates_without_serializer_diagnostics(field, caplog, capsys):
    canary = "thinking-secret-canary"

    class Hostile:
        def __repr__(self):
            return canary

        def __str__(self):
            return canary

    config = ThinkingConfig()
    object.__setattr__(config, field, Hostile())
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        with pytest.raises(ValueError, match="Invalid thinking configuration") as caught:
            copy_preflight_thinking(config)
    assert not captured
    assert canary not in str(caught.value) + repr(caught.value) + caplog.text
    output = capsys.readouterr()
    assert canary not in output.out + output.err


@pytest.mark.parametrize("field", ["enabled", "effort", "max_tokens", "include_in_transcript"])
def test_public_routed_thinking_rejects_mutated_input_without_diagnostic_leak(
    monkeypatch, field, caplog, capsys
):
    canary = "public-thinking-secret-canary"

    class Hostile:
        def __repr__(self):
            return canary

        def __str__(self):
            return canary

    async def scenario():
        store = _StageMemoryStore()
        provider = ScriptedModelProvider([[ModelStreamEvent.completed()]])
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_provider(provider, default=True)
        app.register_agent(AgentSpec(name="agent", model="small"))
        request = RunRequest(
            agent_name="agent",
            session_id="mutated-thinking",
            messages=[Message.text("user", "hello")],
            thinking=ThinkingConfig(),
            failover=ModelFailoverPolicy(
                fallbacks=(ModelTarget(provider_name=provider.name, model="large"),)
            ),
        )
        assert request.thinking is not None
        object.__setattr__(request.thinking, field, Hostile())
        with pytest.raises((TypeError, ValueError)) as caught:
            _ = [event async for event in app.run(request)]
        assert canary not in str(caught.value) + repr(caught.value)
        assert provider.requests == []
        assert await store.load("mutated-thinking") is None
        assert await store.load_checkpoint("mutated-thinking") is None
        assert await store.query_events(EventQuery(session_id="mutated-thinking")) == []

    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        asyncio.run(scenario())
    assert not captured
    assert canary not in caplog.text
    output = capsys.readouterr()
    assert canary not in output.out + output.err


def test_public_failover_rechecks_thinking_before_successor_preparation(monkeypatch):

    class RevokedThinking(ScriptedModelProvider):
        def __init__(self):
            super().__init__([[ModelStreamEvent.completed()]], name="backup")
            self.preflights = 0

        def preflight_thinking(self, *, model, thinking):
            super().preflight_thinking(model=model, thinking=thinking)
            self.preflights += 1
            if self.preflights > 1:
                raise ValueError("Thinking support no longer available")

    def unavailable(_request):
        raise ModelProviderError("unavailable", provider="primary", status_code=503, retryable=True)

    async def scenario():
        store = _StageMemoryStore()
        primary = ScriptedModelProvider(response_factory=unavailable, name="primary")
        backup = RevokedThinking()
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_provider(primary, default=True)
        app.register_provider(backup)
        app.register_agent(
            AgentSpec(name="agent", model="small", thinking=ThinkingConfig(effort="high"))
        )
        events = [
            event
            async for event in app.run(
                RunRequest(
                    agent_name="agent",
                    session_id="revoked-thinking",
                    messages=[Message.text("user", "hello")],
                    retry_policy=RetryPolicy(max_attempts=1, initial_delay_s=0),
                    failover=ModelFailoverPolicy(
                        fallbacks=(ModelTarget(provider_name="backup", model="large"),)
                    ),
                )
            )
        ]
        assert events[-1].type is EventType.SESSION_FAILED
        assert "Thinking support no longer available" in repr(events[-1].payload)
        assert len(primary.requests) == 1 and backup.requests == []
        assert backup.preflights == 2
        checkpoint = await store.load_checkpoint("revoked-thinking")
        assert checkpoint is not None
        assert checkpoint["model_failover"]["candidate_index"] == 0
        assert checkpoint["model_failover"]["attempts_used"] == 1

    asyncio.run(scenario())
