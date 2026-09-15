"""Native tool projections retain their existing authority during failover."""

from __future__ import annotations

import asyncio
import json
from typing import cast

import pytest
from tests.core.test_model_failover_stages import _StageMemoryStore, _StageSQLiteStore
from tests.core.test_targeted_tool_grants import (
    _codec,
    _GatewayRememberTool,
    _NativeOpenAITransport,
    _NativeTestOpenAIProvider,
)
from tests.core.test_tool_discovery import _RememberKnowledgeTool

from cayu import (
    AgentSpec,
    CayuApp,
    EventType,
    Message,
    MessageRole,
    ModelFailoverPolicy,
    ModelTarget,
    RequestFootprintConfig,
    RunRequest,
    ScriptedModelProvider,
    TargetedToolGrant,
)
from cayu.providers.base import (
    OPENAI_CLIENT_TOOL_SEARCH_PROTOCOL,
    ModelProviderError,
    ModelStreamEvent,
)
from cayu.providers.openai import OpenAITransport
from cayu.runtime.retry_policy import RetryPolicy
from cayu.tools.exposure import StaticToolExposurePolicy
from cayu.tools.targeted_projection import targeted_tool_projection_marker_id


def test_routed_error_attribution_comes_from_execution_not_stream_payload(monkeypatch):

    async def scenario():
        error = ModelStreamEvent.error(
            "unavailable",
            cause=ModelProviderError(
                "unavailable", provider="upstream-protocol", status_code=503, retryable=True
            ),
        )
        error.payload.update(provider_name="forged-registration", requested_model="forged-model")

        class ErrorProvider(ScriptedModelProvider):
            async def stream(self, request):
                self.requests.append(request)
                yield error

        primary = ErrorProvider([[ModelStreamEvent.completed()]], name="primary")
        backup = ScriptedModelProvider([[ModelStreamEvent.completed()]], name="backup")
        store = _StageMemoryStore()
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_provider(primary, default=True)
        app.register_provider(backup)
        app.register_agent(AgentSpec(name="agent", model="small"))
        events = [
            event
            async for event in app.run(
                RunRequest(
                    agent_name="agent",
                    session_id="error-attribution",
                    messages=[Message.text("user", "hello")],
                    retry_policy=RetryPolicy(max_attempts=1, initial_delay_s=0),
                    failover=ModelFailoverPolicy(
                        fallbacks=(ModelTarget(provider_name="backup", model="large"),)
                    ),
                )
            )
        ]
        assert events[-1].type is EventType.SESSION_COMPLETED
        assert len(primary.requests) == len(backup.requests) == 1
        [failure] = [event for event in events if event.type is EventType.MODEL_ERROR]
        assert failure.payload["provider"] == "upstream-protocol"
        assert failure.payload["provider_name"] == "primary"
        assert failure.payload["requested_model"] == "small"
        durable = await store.load_events("error-attribution")
        [stored_failure] = [event for event in durable if event.type is EventType.MODEL_ERROR]
        # Public event/attempt IDs are projected aliases, not raw store IDs.
        # Execution attribution and typed failure facts must survive unchanged.
        for key in (
            "provider",
            "provider_name",
            "requested_model",
            "status_code",
            "provider_retryable",
        ):
            assert stored_failure.payload[key] == failure.payload[key]
        assert "forged-registration" not in repr([event.payload for event in durable])
        assert "forged-model" not in repr([event.payload for event in durable])

    asyncio.run(scenario())


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
@pytest.mark.parametrize("native_first", [True, False])
def test_failover_reprojects_targeted_grants_without_replacing_authority(
    monkeypatch, tmp_path, backend, native_first
):

    async def scenario():
        store = (
            _StageMemoryStore(public_authority_alias_codec=_codec())
            if backend == "memory"
            else _StageSQLiteStore(tmp_path / "mixed.sqlite", public_authority_alias_codec=_codec())
        )
        native_transport = _NativeOpenAITransport(retry_first=native_first)
        native = _NativeTestOpenAIProvider(
            api_key="unit-test",
            name="primary" if native_first else "backup",
            transport=cast("OpenAITransport", native_transport),
            additional_tools_models=("fake-model",),
        )
        gateway_calls = 0

        def gateway_response(request):
            nonlocal gateway_calls
            gateway_calls += 1
            if not native_first:
                raise ModelProviderError(
                    "unavailable", provider="gateway", status_code=503, retryable=True
                )
            if gateway_calls > 1:
                return [ModelStreamEvent.completed()]
            assert [tool["name"] for tool in request.tools] == ["call_tool"]
            gateway_message = next(
                message
                for message in request.messages
                if message.role == "user"
                and message.content[0].text.startswith("Cayu runtime targeted-tool context")
            )
            [descriptor] = json.loads(gateway_message.content[0].text.rsplit("\n", 1)[1])["tools"]
            return [
                ModelStreamEvent.tool_call(
                    id="gateway-call",
                    name="call_tool",
                    arguments={
                        "tool_ref": descriptor["tool_ref"],
                        "arguments": {"fact": "Keep native identity stable."},
                    },
                ),
                ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
            ]

        gateway = ScriptedModelProvider(
            name="backup" if native_first else "primary", response_factory=gateway_response
        )
        tool = _GatewayRememberTool()
        app = CayuApp(
            session_store=store,
            enable_logging=False,
            request_footprint=RequestFootprintConfig(enabled=True),
        )
        app.register_provider(native, default=native_first)
        app.register_provider(gateway, default=not native_first)
        app.register_agent(
            AgentSpec(name="agent", model="fake-model"),
            tools=(tool,),
            targeted_tool_mode="openai_additional_tools_or_call_tool",
            tool_exposure_policy=StaticToolExposurePolicy(profile_id="targeted-only", tools=()),
        )
        try:
            events = [
                event
                async for event in app.run(
                    RunRequest(
                        agent_name="agent",
                        session_id="mixed",
                        messages=[Message.text("user", "Remember one fact")],
                        tool_grants=(
                            TargetedToolGrant(
                                request_id="fact",
                                tool_id="cayu:remember",
                                max_calls=1,
                                lifetime_seconds=60,
                            ),
                        ),
                        retry_policy=RetryPolicy(max_attempts=1, initial_delay_s=0),
                        failover=ModelFailoverPolicy(
                            fallbacks=(ModelTarget(provider_name="backup", model="fake-model"),)
                        ),
                    )
                )
            ]
            assert events[-1].type is EventType.SESSION_COMPLETED, events[-1].payload
            assert tool.calls == [{"fact": "Keep native identity stable."}]
            assert len(native_transport.calls) == (1 if native_first else 2)
            assert len(gateway.requests) == (2 if native_first else 1)
            [started] = [event for event in events if event.type is EventType.TOOL_CALL_STARTED]
            assert started.tool_name == "remember"
            assert started.payload["dispatch_kind"] == ("gateway" if native_first else "native")
            [record] = await store.list_targeted_tool_grants("mixed")
            assert record.used_calls == 1 and record.remaining_calls == 0
            assert (
                sum(event.type is EventType.TARGETED_TOOL_REFERENCE_CONSUMED for event in events)
                == 1
            )
            footprints = [
                event.payload
                for event in events
                if event.type is EventType.REQUEST_FOOTPRINT_RECORDED
            ]
            assert len(footprints) == 3
            native_kind, gateway_kind = "openai_additional_tools", "call_tool"
            assert [item["targeted_tool_grants"]["projection"] for item in footprints] == (
                [native_kind, gateway_kind, gateway_kind]
                if native_first
                else [gateway_kind, native_kind, native_kind]
            )
            assert all(
                item["targeted_tool_grants"]["grant_ids"] == [record.grant_id]
                for item in footprints
            )
            assert [item["targeted_tool_grants"]["used_calls"] for item in footprints] == [0, 0, 1]
            checkpoint = await store.load_checkpoint("mixed")
            assert checkpoint is not None
            assert checkpoint["model_failover"]["candidate_index"] == 1
            assert await store.load_active_model_completion_stage("mixed") is None
        finally:
            if isinstance(store, _StageSQLiteStore):
                await store.close()

    asyncio.run(scenario())


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
def test_native_openai_failover_preserves_grant_authority_and_preflights_additional_tools(
    monkeypatch, tmp_path, backend
):

    class CapturingOpenAI(_NativeTestOpenAIProvider):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.portable_tools = []

        def preflight_portable_messages(self, *, model, messages, tools):
            super().preflight_portable_messages(model=model, messages=messages, tools=tools)
            self.portable_tools.append([tool["name"] for tool in tools])

    async def scenario():
        store = (
            _StageMemoryStore(public_authority_alias_codec=_codec())
            if backend == "memory"
            else _StageSQLiteStore(
                tmp_path / "native-tools.sqlite", public_authority_alias_codec=_codec()
            )
        )
        primary_transport = _NativeOpenAITransport(retry_first=True)
        backup_transport = _NativeOpenAITransport()
        primary = CapturingOpenAI(
            api_key="unit-test",
            name="primary",
            transport=primary_transport,
            additional_tools_models=("fake-model",),
        )
        backup = CapturingOpenAI(
            api_key="unit-test",
            name="backup",
            transport=backup_transport,
            additional_tools_models=("fake-model",),
        )
        tool = _GatewayRememberTool()
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_provider(primary, default=True)
        app.register_provider(backup)
        app.register_agent(
            AgentSpec(name="agent", model="fake-model"),
            tools=(tool,),
            targeted_tool_mode="openai_additional_tools",
            tool_exposure_policy=StaticToolExposurePolicy(profile_id="targeted-only", tools=()),
        )
        try:
            events = [
                event
                async for event in app.run(
                    RunRequest(
                        agent_name="agent",
                        session_id="native-failover",
                        messages=[Message.text("user", "Remember the reviewed fact")],
                        tool_grants=(
                            TargetedToolGrant(
                                request_id="remember-fact",
                                tool_id="cayu:remember",
                                max_calls=1,
                                lifetime_seconds=60,
                            ),
                        ),
                        retry_policy=RetryPolicy(max_attempts=1, initial_delay_s=0),
                        failover=ModelFailoverPolicy(
                            fallbacks=(ModelTarget(provider_name="backup", model="fake-model"),)
                        ),
                    )
                )
            ]
            assert events[-1].type is EventType.SESSION_COMPLETED, events[-1].payload
            assert len(primary_transport.calls) == 1 and len(backup_transport.calls) == 2
            assert tool.calls == [{"fact": "Keep native identity stable."}]
            errors = [event for event in events if event.type is EventType.MODEL_ERROR]
            assert len(errors) == 1
            assert errors[0].payload["provider"] == "openai"
            assert errors[0].payload["provider_name"] == "primary"
            assert errors[0].payload["requested_model"] == "fake-model"
            assert errors[0].payload["status_code"] == 503
            assert errors[0].payload["provider_retryable"] is True
            for provider, transport in ((primary, primary_transport), (backup, backup_transport)):
                assert len(provider.portable_tools) >= 2
                assert "remember" in provider.portable_tools[1]
                first = transport.calls[0]
                assert all(tool.get("name") != "remember" for tool in first["tools"])
                additional = [
                    item for item in first["input"] if item.get("type") == "additional_tools"
                ]
                assert len(additional) == 1
                assert any(tool["name"] == "remember" for tool in additional[0]["tools"])
            started = [event for event in events if event.type is EventType.TOOL_CALL_STARTED]
            assert len(started) == 1 and started[0].tool_name == "remember"
            assert started[0].payload["dispatch_kind"] == "native"
            assert (
                sum(event.type is EventType.TARGETED_TOOL_REFERENCE_CONSUMED for event in events)
                == 1
            )
            # Reprojection is ephemeral, not another grant acquisition or row.
            transcript = await store.load_transcript("native-failover")
            assert (
                sum(
                    targeted_tool_projection_marker_id(message) is not None
                    for message in transcript
                )
                == 1
            )
            checkpoint = await store.load_checkpoint("native-failover")
            assert checkpoint is not None and checkpoint["model_failover"]["candidate_index"] == 1
            assert await store.load_active_model_completion_stage("native-failover") is None
        finally:
            if isinstance(store, _StageSQLiteStore):
                await store.close()

    asyncio.run(scenario())


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
@pytest.mark.parametrize("reject_loaded", [False, True])
def test_public_failover_preflights_loaded_native_discovery_tools(
    monkeypatch, tmp_path, backend, reject_loaded
):

    class DiscoveryProvider(ScriptedModelProvider):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.loaded_preflights = []

        def supports_tool_discovery_projection(self, *, model, protocol):
            return model == "fake-model" and protocol == OPENAI_CLIENT_TOOL_SEARCH_PROTOCOL

        def preflight_portable_messages(self, *, model, messages, tools):
            super().preflight_portable_messages(model=model, messages=messages, tools=tools)
            if any(message.role is MessageRole.TOOL for message in messages):
                names = [tool["name"] for tool in tools]
                self.loaded_preflights.append(names)
                if self.name == "backup" and reject_loaded and "remember_knowledge" in names:
                    raise ValueError("Discovered tool is incompatible with backup renderer")

    async def scenario():
        store = (
            _StageMemoryStore()
            if backend == "memory"
            else _StageSQLiteStore(tmp_path / "discovered-tools.sqlite")
        )
        primary_calls = 0

        def primary_response(_request):
            nonlocal primary_calls
            primary_calls += 1
            if primary_calls == 1:
                return [
                    ModelStreamEvent.tool_call(
                        id="search",
                        name="search_tools",
                        arguments={"query": "remember durable knowledge", "limit": 3},
                    ),
                    ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
                ]
            raise ModelProviderError(
                "unavailable", provider="primary", status_code=503, retryable=True
            )

        primary = DiscoveryProvider(response_factory=primary_response, name="primary")
        backup = DiscoveryProvider(
            [
                [
                    ModelStreamEvent.tool_call(
                        id="remember",
                        name="remember_knowledge",
                        arguments={"fact": "Preserve discovery authority across failover."},
                    ),
                    ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
                ],
                [ModelStreamEvent.text_delta("done"), ModelStreamEvent.completed()],
            ],
            name="backup",
        )
        remembered = _RememberKnowledgeTool()
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_provider(primary, default=True)
        app.register_provider(backup)
        app.register_agent(
            AgentSpec(name="agent", model="fake-model"),
            tools=(remembered,),
            tool_discovery_mode="openai_tool_search_client",
        )
        try:
            events = [
                event
                async for event in app.run(
                    RunRequest(
                        agent_name="agent",
                        session_id="discovery-failover",
                        messages=[Message.text("user", "Find and save the lesson")],
                        retry_policy=RetryPolicy(max_attempts=1, initial_delay_s=0),
                        failover=ModelFailoverPolicy(
                            fallbacks=(ModelTarget(provider_name="backup", model="fake-model"),)
                        ),
                    )
                )
            ]
            assert len(primary.requests) == 2
            assert backup.loaded_preflights and "remember_knowledge" in backup.loaded_preflights[0]
            assert all(tool["name"] != "remember_knowledge" for tool in primary.requests[1].tools)
            projection = primary.requests[1].tool_discovery_projection
            assert projection is not None and projection.loaded_tool_names == (
                "remember_knowledge",
            )
            checkpoint = await store.load_checkpoint("discovery-failover")
            assert checkpoint is not None
            if reject_loaded:
                assert events[-1].type is EventType.SESSION_FAILED
                assert "Discovered tool is incompatible" in repr(events[-1].payload)
                assert backup.requests == [] and remembered.calls == []
                assert checkpoint["model_failover"]["candidate_index"] == 0
            else:
                assert events[-1].type is EventType.SESSION_COMPLETED
                assert len(backup.requests) == 2
                assert remembered.calls == [
                    {"fact": "Preserve discovery authority across failover."}
                ]
                assert checkpoint["model_failover"]["candidate_index"] == 1
                assert await store.load_active_model_completion_stage("discovery-failover") is None
        finally:
            if isinstance(store, _StageSQLiteStore):
                await store.close()

    asyncio.run(scenario())


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
@pytest.mark.parametrize("native_first", [True, False])
def test_failover_preserves_discovery_grants_across_projection_modes(
    monkeypatch, tmp_path, backend, native_first
):

    class DiscoveryProvider(ScriptedModelProvider):
        def supports_tool_discovery_projection(self, *, model, protocol):
            native = native_first if self.name == "primary" else not native_first
            return (
                native and model == "fake-model" and protocol == OPENAI_CLIENT_TOOL_SEARCH_PROTOCOL
            )

    async def scenario():
        store = (
            _StageMemoryStore()
            if backend == "memory"
            else _StageSQLiteStore(tmp_path / "mixed-discovery.sqlite")
        )
        calls = {"primary": 0, "backup": 0}

        def primary_response(_request):
            calls["primary"] += 1
            if calls["primary"] == 1:
                return [
                    ModelStreamEvent.tool_call(
                        id="search",
                        name="search_tools",
                        arguments={"query": "remember durable knowledge", "limit": 3},
                    ),
                    ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
                ]
            raise ModelProviderError(
                "unavailable", provider="primary", status_code=503, retryable=True
            )

        def backup_response(request):
            calls["backup"] += 1
            if calls["backup"] > 1:
                return [ModelStreamEvent.completed()]
            arguments = {"fact": "Preserve the discovered tool across representations."}
            if native_first:
                assert request.tool_discovery_projection is None
                [search_result] = [
                    part
                    for message in request.messages
                    for part in message.content
                    if part.type == "tool_result" and part.tool_name == "search_tools"
                ]
                assert search_result.structured is not None
                [match] = search_result.structured["matches"]
                tool_name = "call_tool"
                arguments = {"tool_ref": match["tool_ref"], "arguments": arguments}
            else:
                assert request.tool_discovery_projection is not None
                assert request.tool_discovery_projection.loaded_tool_names == (
                    "remember_knowledge",
                )
                tool_name = "remember_knowledge"
            return [
                ModelStreamEvent.tool_call(id="remember", name=tool_name, arguments=arguments),
                ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
            ]

        primary = DiscoveryProvider(name="primary", response_factory=primary_response)
        backup = DiscoveryProvider(name="backup", response_factory=backup_response)
        remembered = _RememberKnowledgeTool()
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_provider(primary, default=True)
        app.register_provider(backup)
        app.register_agent(
            AgentSpec(name="agent", model="fake-model"),
            tools=(remembered,),
            tool_discovery_mode="openai_tool_search_client_or_search_tools",
        )
        try:
            events = [
                event
                async for event in app.run(
                    RunRequest(
                        agent_name="agent",
                        session_id="mixed-discovery",
                        messages=[Message.text("user", "Discover and save a fact")],
                        retry_policy=RetryPolicy(max_attempts=1, initial_delay_s=0),
                        failover=ModelFailoverPolicy(
                            fallbacks=(ModelTarget(provider_name="backup", model="fake-model"),)
                        ),
                    )
                )
            ]
            assert events[-1].type is EventType.SESSION_COMPLETED, events[-1].payload
            assert calls == {"primary": 2, "backup": 2}
            assert remembered.calls == [
                {"fact": "Preserve the discovered tool across representations."}
            ]
            [started] = [
                event
                for event in events
                if event.type is EventType.TOOL_CALL_STARTED
                and event.tool_name == "remember_knowledge"
            ]
            assert started.payload["dispatch_kind"] == ("gateway" if native_first else "native")
            assert all(
                tool["name"] != "remember_knowledge"
                for request in backup.requests
                for tool in request.tools
            )
            checkpoint = await store.load_checkpoint("mixed-discovery")
            assert checkpoint is not None and checkpoint["model_failover"]["candidate_index"] == 1
            assert await store.load_active_model_completion_stage("mixed-discovery") is None
        finally:
            if isinstance(store, _StageSQLiteStore):
                await store.close()

    asyncio.run(scenario())
