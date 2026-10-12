"""Request footprints contracts."""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from typing import Any

import pytest
from pydantic import SecretStr
from tests.runtime.runtime_attachment_support import _app_with_artifact_store, _valid_png_bytes
from tests.runtime.runtime_provider_support import (
    ContextOverflowProvider,
    CountingProvider,
    FakeProvider,
)
from tests.runtime.runtime_session_support import _without_interaction_lifecycle
from tests.runtime.runtime_session_support import (
    request_footprint_session_store as request_footprint_session_store,
)
from tests.runtime.runtime_tool_support import EchoTool

import cayu.runtime._live_model_attempt as live_model_attempt_module
import cayu.runtime._model_step_executor as model_step_executor_module
import cayu.runtime.execution_profiles as execution_profiles_module
from cayu.agents import AgentSpec
from cayu.applications import CayuApp
from cayu.artifacts.attachments import RESOLVED_FILE_ATTACHMENTS_OPTION, FileAttachmentKind
from cayu.configuration import CayuConfig, RunDefaults
from cayu.context.base import (
    CheckpointCompactionContextPolicy,
    ModelCompactor,
    RecentTurnsContextPolicy,
)
from cayu.context.counting import ContextCountingConfig, ContextCountingMode
from cayu.context.footprints import (
    PromptContributionManifest,
    RequestFootprint,
    RequestFootprintConfig,
    RequestVariant,
)
from cayu.context.structured_output import STRUCTURED_OUTPUT_TOOL_NAME, StructuredOutputSpec
from cayu.events import Event, EventType
from cayu.messages import Message, TextPart
from cayu.providers.base import (
    InputTokenCountConfidence,
    InputTokenCountMethod,
    InputTokenCountResult,
    ModelProvider,
    ModelProviderError,
    ModelRequest,
    ModelStreamEvent,
)
from cayu.providers.cache import CacheBreakpoint, CachePolicy, RequestCacheProjection
from cayu.providers.retry_policy import RetryPolicy
from cayu.sessions.event_queries import EventQuery
from cayu.sessions.requests import RunRequest
from cayu.storage.sqlite import SQLiteSessionStore
from cayu.vaults.redaction import SecretRedactor


async def collect_request_footprint_events(
    app: CayuApp,
    request: RunRequest,
) -> list[Event]:
    events = [event async for event in app.run(request)]
    return _without_interaction_lifecycle(events)


def test_request_footprint_is_default_on_before_model_start_without_provider_counting(
    request_footprint_session_store,
) -> None:
    user_text = "request footprint raw prompt sentinel"
    provider = CountingProvider(
        [
            ModelStreamEvent.text_delta("done"),
            ModelStreamEvent.completed({"finish_reason": "stop"}),
        ]
    )
    app = CayuApp(
        session_store=request_footprint_session_store,
        enable_logging=False,
    )
    app.register_provider(provider, default=True)
    app.register_agent(AgentSpec(name="assistant", model="fake-model"))

    events = asyncio.run(
        collect_request_footprint_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_request_footprint_default_on",
                messages=[Message.text("user", user_text)],
            ),
        )
    )

    event_types = [event.type for event in events]
    footprint_index = event_types.index(EventType.REQUEST_FOOTPRINT_RECORDED)
    started_index = event_types.index(EventType.MODEL_STARTED)
    assert footprint_index < started_index
    assert event_types.count(EventType.REQUEST_FOOTPRINT_RECORDED) == 1
    assert provider.count_requests == []

    footprint = events[footprint_index].payload
    assert footprint["schema_version"] == 3
    assert footprint["tool_exposure"]["registered_count"] == 0
    assert footprint["tool_exposure"]["exposed_count"] == 0
    assert footprint["execution_profile_fingerprint"] == next(
        event.payload["execution_profile_fingerprint"]
        for event in events
        if event.type is EventType.MODEL_STARTED
    )
    assert footprint["provider_name"] == "fake"
    assert footprint["model"] == "fake-model"
    assert footprint["step"] == 1
    assert footprint["attempt"] == 1
    assert footprint["request_variant"] == RequestVariant.INITIAL
    assert footprint["messages"]["count"] == 1
    assert footprint["context_pressure"]["method"] == "local_full_request_estimate"
    assert user_text not in json.dumps(footprint, sort_keys=True)


def test_request_footprint_reuses_canonical_context_analysis_for_completion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = CountingProvider(
        [ModelStreamEvent.completed({"finish_reason": "stop"})],
    )
    app = CayuApp(enable_logging=False)
    app.register_provider(provider, default=True)
    app.register_agent(AgentSpec(name="assistant", model="fake-model"))

    def fail_parallel_estimator(*_args, **_kwargs):
        raise AssertionError("completion path recomputed request context pressure")

    monkeypatch.setattr(
        model_step_executor_module,
        "analyze_request_context_pressure",
        fail_parallel_estimator,
    )
    monkeypatch.setattr(
        live_model_attempt_module,
        "analyze_request_context_pressure",
        fail_parallel_estimator,
    )

    events = asyncio.run(
        collect_request_footprint_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_request_footprint_reuses_analysis",
                messages=[Message.text("user", "hello")],
            ),
        )
    )

    assert EventType.REQUEST_FOOTPRINT_RECORDED in {event.type for event in events}
    assert EventType.MODEL_COMPLETED in {event.type for event in events}


def test_request_footprint_can_be_disabled_independently_of_context_counting() -> None:
    class EffectiveOptionsCountingProvider(CountingProvider):
        supports_native_structured_output = True

        def request_cache_projection(
            self,
            _request: ModelRequest,
        ) -> RequestCacheProjection:
            return RequestCacheProjection(
                policy=CachePolicy(breakpoints=(CacheBreakpoint.SYSTEM_PROMPT,))
            )

        def request_footprint_options(self, _request: ModelRequest) -> dict[str, Any]:
            return {"custom": {"max_tokens": 4096}}

    def run(*, enabled: bool, session_id: str) -> tuple[list[Event], CountingProvider]:
        provider = EffectiveOptionsCountingProvider(
            [
                ModelStreamEvent.text_delta('{"answer":"ok"}'),
                ModelStreamEvent.completed({"finish_reason": "stop"}),
            ],
            count_result=InputTokenCountResult(
                input_tokens=2,
                method=InputTokenCountMethod.OFFICIAL,
                confidence=InputTokenCountConfidence.HIGH,
            ),
        )
        app = CayuApp(
            request_footprint=RequestFootprintConfig(enabled=enabled),
            context_counting=ContextCountingConfig(mode=ContextCountingMode.OBSERVE),
            enable_logging=False,
        )
        app.register_provider(provider, default=True)
        app.register_agent(
            AgentSpec(
                name="assistant",
                model="fake-model",
                system_prompt="private instructions that must not be manifested",
            )
        )
        events = asyncio.run(
            collect_request_footprint_events(
                app,
                RunRequest(
                    agent_name="assistant",
                    session_id=session_id,
                    messages=[Message.text("user", "hello")],
                    structured_output=StructuredOutputSpec(
                        name="answer",
                        json_schema={
                            "type": "object",
                            "properties": {"answer": {"type": "string"}},
                            "required": ["answer"],
                            "additionalProperties": False,
                        },
                        strategy="native",
                    ),
                ),
            )
        )
        return events, provider

    enabled_events, enabled_provider = run(
        enabled=True,
        session_id="sess_request_footprint_enabled_pressure",
    )
    disabled_events, disabled_provider = run(
        enabled=False,
        session_id="sess_request_footprint_disabled",
    )

    assert EventType.REQUEST_FOOTPRINT_RECORDED in {event.type for event in enabled_events}
    assert EventType.REQUEST_FOOTPRINT_RECORDED not in {event.type for event in disabled_events}
    assert len(enabled_provider.count_requests) == 1
    assert len(disabled_provider.count_requests) == 1
    assert EventType.CONTEXT_COUNTED in {event.type for event in disabled_events}
    enabled_footprint_pressure = next(
        event.payload["context_pressure"]
        for event in enabled_events
        if event.type == EventType.REQUEST_FOOTPRINT_RECORDED
    )
    enabled_completion_pressure = next(
        event.payload["context_pressure"]
        for event in enabled_events
        if event.type == EventType.MODEL_COMPLETED
    )
    disabled_completion_pressure = next(
        event.payload["context_pressure"]
        for event in disabled_events
        if event.type == EventType.MODEL_COMPLETED
    )
    assert enabled_completion_pressure == {
        key: enabled_footprint_pressure[key] for key in enabled_completion_pressure
    }
    assert disabled_completion_pressure == enabled_completion_pressure
    assert disabled_completion_pressure["estimated_request_options_input_tokens"] > 0
    assert disabled_completion_pressure["estimated_structured_output_input_tokens"] > 0
    started = next(event for event in disabled_events if event.type == EventType.SESSION_STARTED)
    assert "prompt_contribution_manifest" not in started.payload


def test_request_footprint_records_each_tool_round(request_footprint_session_store) -> None:
    provider = FakeProvider(
        [
            [
                ModelStreamEvent.tool_call(
                    id="call_footprint",
                    name="echo",
                    arguments={"text": "hello"},
                ),
                ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
            ],
            [ModelStreamEvent.completed({"finish_reason": "stop"})],
        ]
    )
    app = CayuApp(
        session_store=request_footprint_session_store,
        enable_logging=False,
    )
    app.register_provider(provider, default=True)
    app.register_agent(
        AgentSpec(name="assistant", model="fake-model"),
        tools=[EchoTool()],
    )

    events = asyncio.run(
        collect_request_footprint_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_request_footprint_tool_rounds",
                messages=[Message.text("user", "use the tool")],
            ),
        )
    )

    footprints = [
        RequestFootprint.model_validate(event.payload)
        for event in events
        if event.type == EventType.REQUEST_FOOTPRINT_RECORDED
    ]
    assert [footprint.step for footprint in footprints] == [1, 2]
    assert all(footprint.request_variant == RequestVariant.INITIAL for footprint in footprints)
    assert footprints[1].messages.count > footprints[0].messages.count


def test_request_footprint_records_resolved_attachment_shape_without_identity(
    tmp_path,
    request_footprint_session_store,
) -> None:
    provider = FakeProvider([ModelStreamEvent.completed({"finish_reason": "stop"})])
    app, _ = _app_with_artifact_store(
        tmp_path,
        session_store=request_footprint_session_store,
    )
    app.register_provider(provider, default=True)
    app.register_agent(AgentSpec(name="assistant", model="fake-model"))
    image = _valid_png_bytes()

    async def run() -> list[Event]:
        part = await app.attach_file(
            image,
            filename="private-file-name.png",
            kind="image",
            session_id="sess_request_footprint_attachment",
        )
        return await collect_request_footprint_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_request_footprint_attachment",
                messages=[Message(role="user", content=[TextPart(text="describe"), part])],
            ),
        )

    events = asyncio.run(run())
    footprint_event = next(
        event for event in events if event.type == EventType.REQUEST_FOOTPRINT_RECORDED
    )
    footprint = RequestFootprint.model_validate(footprint_event.payload)

    assert footprint.attachments.count == 1
    assert footprint.attachments.source_bytes == len(image)
    assert [(group.kind, group.count) for group in footprint.attachments.groups] == [
        (FileAttachmentKind.IMAGE, 1)
    ]
    serialized = json.dumps(footprint.model_dump(mode="json"), sort_keys=True)
    assert "private-file-name.png" not in serialized
    assert provider.requests[0].options[RESOLVED_FILE_ATTACHMENTS_OPTION]


def test_request_footprint_attributes_matching_creation_time_prompt_manifest() -> None:
    system_prompt = "private agent instruction sentinel"
    provider = FakeProvider([ModelStreamEvent.completed({"finish_reason": "stop"})])
    app = CayuApp(
        request_footprint=RequestFootprintConfig(
            fingerprint_key_id="prompt-key",
            fingerprint_key=SecretStr("p" * 32),
        ),
        enable_logging=False,
    )
    app.register_provider(provider, default=True)
    app.register_agent(
        AgentSpec(
            name="assistant",
            model="fake-model",
            system_prompt=system_prompt,
        )
    )

    events = asyncio.run(
        collect_request_footprint_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_request_footprint_prompt_manifest",
                messages=[Message.text("user", "hello")],
            ),
        )
    )

    started = next(event for event in events if event.type == EventType.SESSION_STARTED)
    manifest = started.payload["prompt_contribution_manifest"]
    footprint = next(
        event.payload for event in events if event.type == EventType.REQUEST_FOOTPRINT_RECORDED
    )
    assert manifest["schema_version"] == 1
    assert [item["kind"] for item in manifest["contributions"]] == ["agent_instructions"]
    assert footprint["prompt_contributions"]["availability"] == "available"
    assert [item["kind"] for item in footprint["prompt_contributions"]["contributions"]] == [
        "agent_instructions"
    ]
    assert system_prompt not in json.dumps(
        {"started": started.payload, "footprint": footprint},
        sort_keys=True,
    )


def test_request_footprint_redaction_collision_preserves_typed_unavailable_evidence() -> None:
    class CacheAwareProvider(FakeProvider):
        def request_cache_policy(self, request: ModelRequest) -> CachePolicy | None:
            return CachePolicy(
                breakpoints=(
                    CacheBreakpoint.SYSTEM_PROMPT,
                    CacheBreakpoint.CONVERSATION_PREFIX,
                ),
                conversation_prefix_strategy="all_but_last",
                ttl="standard",
            )

        def request_footprint_options(self, request: ModelRequest) -> dict[str, Any]:
            return {"custom": {"category": 1}}

    provider = CacheAwareProvider([ModelStreamEvent.completed({"finish_reason": "stop"})])
    app = CayuApp(
        request_footprint=RequestFootprintConfig(
            fingerprint_key_id="redaction-key-category",
            fingerprint_key=SecretStr("r" * 32),
        ),
        secret_redactor=SecretRedactor("category"),
        enable_logging=False,
    )
    app.register_provider(provider, default=True)
    app.register_agent(
        AgentSpec(
            name="assistant",
            model="fake-model",
            system_prompt="private system prompt",
        )
    )

    events = asyncio.run(
        collect_request_footprint_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_request_footprint_redaction_collision",
                messages=[
                    Message.text("user", "hello"),
                    Message.text("assistant", "previous answer"),
                    Message.text("user", "continue"),
                ],
            ),
        )
    )

    assert EventType.SESSION_FAILED not in {event.type for event in events}
    started = next(event for event in events if event.type == EventType.SESSION_STARTED)
    manifest = PromptContributionManifest.model_validate(
        started.payload["prompt_contribution_manifest"]
    )
    footprint = RequestFootprint.model_validate(
        next(
            event.payload for event in events if event.type == EventType.REQUEST_FOOTPRINT_RECORDED
        )
    )
    assert manifest.system_fingerprint.availability == "unavailable"
    assert manifest.system_fingerprint.unavailable_reason == "fingerprint_evidence_redacted"
    assert footprint.fingerprints.provider_neutral_request.availability == "unavailable"
    assert footprint.prompt_contributions.availability == "unavailable"
    assert footprint.prompt_contributions.unavailable_reason == "system_identity_unavailable"
    assert footprint.request_variant == "initial"
    assert {group.role for group in footprint.messages.groups} >= {"assistant", "user"}
    assert {item.kind for item in footprint.prompt_contributions.contributions} == set()
    assert footprint.context_pressure.method == "local_full_request_estimate"
    assert footprint.context_pressure.confidence == "estimated"
    assert footprint.component_tokens.method == "local_full_request_estimate"
    assert footprint.component_tokens.confidence == "estimated"
    assert "cache_policy" in footprint.options.known_categories
    assert "custom.category" not in footprint.options.known_categories
    assert len(footprint.options.known_categories) == 2
    assert {item.kind for item in footprint.cache_breakpoints} == {
        "system_prompt",
        "conversation_prefix",
    }
    assert {item.ttl for item in footprint.cache_breakpoints} == {"standard"}


def test_request_footprint_redaction_collision_preserves_public_tool_exposure() -> None:
    def run(redactor: SecretRedactor) -> list[Event]:
        provider = FakeProvider([ModelStreamEvent.completed({"finish_reason": "stop"})])
        app = CayuApp(
            secret_redactor=redactor,
            enable_logging=False,
        )
        app.register_provider(provider, default=True)
        app.register_agent(AgentSpec(name="assistant", model="fake-model"))
        return asyncio.run(
            collect_request_footprint_events(
                app,
                RunRequest(
                    agent_name="assistant",
                    session_id="sess_request_footprint_exposure_collision",
                    messages=[Message.text("user", "hello")],
                ),
            )
        )

    baseline_events = run(SecretRedactor())
    baseline_exposure = next(
        event.payload for event in baseline_events if event.type is EventType.TOOL_EXPOSURE_RECORDED
    )
    fingerprint_secret = baseline_exposure["exposure_fingerprint"][10:18]

    events = run(SecretRedactor(["cayu", fingerprint_secret]))
    exposure = next(
        event.payload for event in events if event.type is EventType.TOOL_EXPOSURE_RECORDED
    )
    footprint = RequestFootprint.model_validate(
        next(
            event.payload for event in events if event.type is EventType.REQUEST_FOOTPRINT_RECORDED
        )
    )

    assert footprint.tool_exposure is not None
    assert footprint.tool_exposure.profile_id == exposure["profile_id"]
    assert footprint.tool_exposure.exposure_fingerprint == exposure["exposure_fingerprint"]


def test_request_footprint_records_effective_provider_cache_breakpoints() -> None:
    class CacheAwareProvider(FakeProvider):
        def request_cache_policy(self, request: ModelRequest) -> CachePolicy | None:
            return CachePolicy(
                breakpoints=(
                    CacheBreakpoint.SYSTEM_PROMPT,
                    CacheBreakpoint.TOOL_DEFINITIONS,
                    CacheBreakpoint.CONVERSATION_PREFIX,
                ),
                conversation_prefix_strategy="all_but_last",
                ttl="extended",
            )

    provider = CacheAwareProvider([ModelStreamEvent.completed({"finish_reason": "stop"})])
    app = CayuApp(
        request_footprint=RequestFootprintConfig(
            fingerprint_key_id="cache-key",
            fingerprint_key=SecretStr("c" * 32),
        ),
        enable_logging=False,
    )
    app.register_provider(provider, default=True)
    app.register_agent(
        AgentSpec(
            name="assistant",
            model="fake-model",
            system_prompt="stable system prompt",
        )
    )

    events = asyncio.run(
        collect_request_footprint_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_request_footprint_cache_policy",
                messages=[
                    Message.text("user", "first"),
                    Message.text("user", "last"),
                ],
            ),
        )
    )

    assert events[-1].type != EventType.SESSION_FAILED, events[-1].payload
    footprint = next(
        event.payload for event in events if event.type == EventType.REQUEST_FOOTPRINT_RECORDED
    )
    assert [item["kind"] for item in footprint["cache_breakpoints"]] == [
        "conversation_prefix",
        "system_prompt",
        "tool_definitions",
    ]
    assert all(item["ttl"] == "extended" for item in footprint["cache_breakpoints"])
    assert footprint["fingerprints"]["conversation_prefix"]["availability"] == "available"


def test_request_footprint_retry_keeps_shape_identity_and_changes_attempt_identity(
    request_footprint_session_store,
) -> None:
    class RetryThenCompleteProvider(ModelProvider):
        name = "retry-footprint"

        def __init__(self) -> None:
            self.requests: list[ModelRequest] = []

        async def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
            self.requests.append(request)
            if len(self.requests) == 1:
                raise ModelProviderError(
                    "provider overloaded",
                    provider=self.name,
                    status_code=503,
                    retryable=True,
                )
            yield ModelStreamEvent.completed({"finish_reason": "stop"})

    provider = RetryThenCompleteProvider()
    app = CayuApp(
        session_store=request_footprint_session_store,
        request_footprint=RequestFootprintConfig(
            fingerprint_key_id="test-key",
            fingerprint_key=SecretStr("k" * 32),
        ),
        config=CayuConfig(
            run=RunDefaults(retry_policy=RetryPolicy(max_attempts=2, initial_delay_s=0.0))
        ),
        enable_logging=False,
    )
    app.register_provider(provider, default=True)
    app.register_agent(AgentSpec(name="assistant", model="fake-model"))

    events = asyncio.run(
        collect_request_footprint_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_request_footprint_retry",
                messages=[Message.text("user", "retry me")],
            ),
        )
    )

    footprints = [
        event.payload for event in events if event.type == EventType.REQUEST_FOOTPRINT_RECORDED
    ]
    assert len(footprints) == 2
    assert [footprint["attempt"] for footprint in footprints] == [1, 2]
    assert (
        footprints[0]["fingerprints"]["provider_neutral_request"]["value"]
        == footprints[1]["fingerprints"]["provider_neutral_request"]["value"]
    )
    durable_footprints = [
        event.payload
        for event in asyncio.run(app.session_store.load_events("sess_request_footprint_retry"))
        if event.type == EventType.REQUEST_FOOTPRINT_RECORDED
    ]
    assert durable_footprints[0]["observation_id"] != durable_footprints[1]["observation_id"]
    assert durable_footprints[0]["model_attempt_id"] != durable_footprints[1]["model_attempt_id"]


def test_request_footprint_marks_structured_output_repair_and_contribution(
    request_footprint_session_store,
) -> None:
    provider = FakeProvider(
        [
            [
                ModelStreamEvent.tool_call(
                    id="call-invalid",
                    name=STRUCTURED_OUTPUT_TOOL_NAME,
                    arguments={"output": {"wrong": "value"}},
                ),
                ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
            ],
            [
                ModelStreamEvent.tool_call(
                    id="call-valid",
                    name=STRUCTURED_OUTPUT_TOOL_NAME,
                    arguments={"output": {"answer": "fixed"}},
                ),
                ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
            ],
        ]
    )
    app = CayuApp(
        session_store=request_footprint_session_store,
        request_footprint=RequestFootprintConfig(
            fingerprint_key_id="structured-key",
            fingerprint_key=SecretStr("s" * 32),
        ),
        enable_logging=False,
    )
    app.register_provider(provider, default=True)
    app.register_agent(AgentSpec(name="assistant", model="fake-model"))

    events = asyncio.run(
        collect_request_footprint_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_request_footprint_structured_repair",
                messages=[Message.text("user", "answer with structured output")],
                structured_output=StructuredOutputSpec(
                    json_schema={
                        "type": "object",
                        "properties": {"answer": {"type": "string"}},
                        "required": ["answer"],
                        "additionalProperties": False,
                    },
                    max_retries=1,
                ),
            ),
        )
    )

    footprints = [
        event.payload for event in events if event.type == EventType.REQUEST_FOOTPRINT_RECORDED
    ]
    assert [item["request_variant"] for item in footprints] == [
        "initial",
        "structured_output_repair",
    ]
    assert [item["structured_output"]["count"] for item in footprints] == [1, 1]
    assert all(item["structured_output"]["size"]["utf8_bytes"] > 0 for item in footprints)
    assert all(
        item["component_tokens"]["structured_output_input_tokens"] > 0 for item in footprints
    )
    assert (
        footprints[0]["fingerprints"]["provider_neutral_request"]["value"]
        != footprints[1]["fingerprints"]["provider_neutral_request"]["value"]
    )


def test_request_footprint_marks_context_overflow_recovery_shape(
    request_footprint_session_store,
) -> None:
    provider = ContextOverflowProvider()
    app = CayuApp(
        session_store=request_footprint_session_store,
        request_footprint=RequestFootprintConfig(
            fingerprint_key_id="overflow-key",
            fingerprint_key=SecretStr("o" * 32),
        ),
        enable_logging=False,
    )
    app.register_provider(provider, default=True)
    app.register_agent(
        AgentSpec(name="assistant", model="fake-model"),
        context_overflow_policy=RecentTurnsContextPolicy(max_user_turns=1),
    )

    events = asyncio.run(
        collect_request_footprint_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_request_footprint_overflow_recovery",
                messages=[
                    Message.text("user", "old request"),
                    Message.text("user", "new request"),
                ],
            ),
        )
    )

    footprints = [
        event.payload for event in events if event.type == EventType.REQUEST_FOOTPRINT_RECORDED
    ]
    assert [item["request_variant"] for item in footprints] == [
        "initial",
        "context_overflow_recovery",
    ]
    assert [item["messages"]["count"] for item in footprints] == [2, 1]
    assert (
        footprints[0]["fingerprints"]["provider_neutral_request"]["value"]
        != footprints[1]["fingerprints"]["provider_neutral_request"]["value"]
    )


def test_request_footprint_records_model_backed_compaction_dispatch(
    request_footprint_session_store,
) -> None:
    class CompactionProvider(FakeProvider):
        name = "compaction-provider"

    class RuntimeProvider(FakeProvider):
        name = "runtime-provider"

    compaction_provider = CompactionProvider(
        [
            ModelStreamEvent.text_delta("summary"),
            ModelStreamEvent.completed({"finish_reason": "stop"}),
        ]
    )
    runtime_provider = RuntimeProvider([ModelStreamEvent.completed({"finish_reason": "stop"})])
    app = CayuApp(
        session_store=request_footprint_session_store,
        enable_logging=False,
    )
    app.register_provider(runtime_provider, default=True)
    app.register_agent(
        AgentSpec(name="assistant", model="runtime-model"),
        context_policy=CheckpointCompactionContextPolicy(
            compactor=ModelCompactor(
                provider=compaction_provider,
                model="summary-model",
            ),
            max_user_turns=1,
            compact_after_messages=2,
        ),
    )

    events = asyncio.run(
        collect_request_footprint_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_request_footprint_compaction",
                messages=[
                    Message.text("user", "old request"),
                    Message.text("assistant", "old answer"),
                    Message.text("user", "current request"),
                ],
            ),
        )
    )

    footprints = [
        event.payload for event in events if event.type == EventType.REQUEST_FOOTPRINT_RECORDED
    ]
    assert [item["request_variant"] for item in footprints] == [
        "context_compaction",
        "initial",
    ]
    assert [item["provider_name"] for item in footprints] == [
        "compaction-provider",
        "runtime-provider",
    ]
    assert [item["model"] for item in footprints] == [
        "summary-model",
        "runtime-model",
    ]
    compaction_footprint = next(
        event
        for event in events
        if event.type == EventType.REQUEST_FOOTPRINT_RECORDED
        and event.payload["request_variant"] == "context_compaction"
    )
    compaction_started = next(
        event
        for event in events
        if event.type == EventType.MODEL_STARTED
        and event.payload["provider"] == "compaction-provider"
    )
    session = asyncio.run(request_footprint_session_store.load("sess_request_footprint_compaction"))
    assert session is not None
    profile_fingerprint = execution_profiles_module.execution_profile_from_session_metadata(
        session.metadata
    ).fingerprint
    governed_compaction_events = [
        event
        for event in events
        if event.type
        in {
            EventType.CONTEXT_COMPACTION_STARTED,
            EventType.CONTEXT_COMPACTION_COMPLETED,
            EventType.SESSION_CHECKPOINTED,
        }
        or (
            event.type
            in {
                EventType.REQUEST_FOOTPRINT_RECORDED,
                EventType.MODEL_STARTED,
                EventType.MODEL_COMPLETED,
            }
            and (
                event.payload.get("request_variant") == "context_compaction"
                or event.payload.get("provider") == "compaction-provider"
            )
        )
    ]
    assert governed_compaction_events
    assert {
        event.payload.get("execution_profile_fingerprint") for event in governed_compaction_events
    } == {profile_fingerprint}
    assert events.index(compaction_footprint) < events.index(compaction_started)
    assert (
        compaction_started.payload["model_step_id"]
        == (compaction_footprint.payload["model_step_id"])
    )
    assert (
        compaction_started.payload["model_attempt_id"]
        == (compaction_footprint.payload["model_attempt_id"])
    )
    assert len(compaction_provider.requests) == 1
    assert len(runtime_provider.requests) == 1


def test_request_footprint_records_each_model_compaction_retry(
    request_footprint_session_store,
) -> None:
    class RetryCompactionProvider(ModelProvider):
        name = "retry-compactor"

        def __init__(self) -> None:
            self.requests: list[ModelRequest] = []

        async def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
            self.requests.append(request)
            if len(self.requests) == 1:
                raise ModelProviderError(
                    "compactor overloaded",
                    provider=self.name,
                    status_code=503,
                    retryable=True,
                )
            yield ModelStreamEvent.text_delta("summary")
            yield ModelStreamEvent.completed({"finish_reason": "stop"})

    compaction_provider = RetryCompactionProvider()
    runtime_provider = FakeProvider([ModelStreamEvent.completed({"finish_reason": "stop"})])
    app = CayuApp(
        session_store=request_footprint_session_store,
        request_footprint=RequestFootprintConfig(
            fingerprint_key_id="compaction-key",
            fingerprint_key=SecretStr("m" * 32),
        ),
        enable_logging=False,
    )
    app.register_provider(runtime_provider, default=True)
    app.register_agent(
        AgentSpec(name="assistant", model="runtime-model"),
        context_policy=CheckpointCompactionContextPolicy(
            compactor=ModelCompactor(
                provider=compaction_provider,
                model="summary-model",
                retry_policy=RetryPolicy(max_attempts=2, initial_delay_s=0.0),
            ),
            max_user_turns=1,
            compact_after_messages=2,
        ),
    )

    events = asyncio.run(
        collect_request_footprint_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_request_footprint_compaction_retry",
                messages=[
                    Message.text("user", "old request"),
                    Message.text("assistant", "old answer"),
                    Message.text("user", "current request"),
                ],
            ),
        )
    )

    compaction_footprints = [
        event.payload
        for event in events
        if event.type == EventType.REQUEST_FOOTPRINT_RECORDED
        and event.payload["request_variant"] == "context_compaction"
    ]
    assert [item["attempt"] for item in compaction_footprints] == [1, 2]
    assert (
        compaction_footprints[0]["fingerprints"]["provider_neutral_request"]["value"]
        == compaction_footprints[1]["fingerprints"]["provider_neutral_request"]["value"]
    )
    assert len(compaction_provider.requests) == 2


def test_request_footprint_key_rotation_is_durable_across_store_backends(
    request_footprint_session_store,
) -> None:
    async def run_with_key(*, key_id: str, key: str, session_id: str) -> None:
        provider = FakeProvider([ModelStreamEvent.completed({"finish_reason": "stop"})])
        app = CayuApp(
            session_store=request_footprint_session_store,
            request_footprint=RequestFootprintConfig(
                fingerprint_key_id=key_id,
                fingerprint_key=SecretStr(key),
            ),
            enable_logging=False,
        )
        app.register_provider(provider, default=True)
        app.register_agent(AgentSpec(name="assistant", model="fake-model"))
        await collect_request_footprint_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id=session_id,
                messages=[Message.text("user", "stable request")],
            ),
        )

    async def run() -> tuple[RequestFootprint, RequestFootprint]:
        first_session_id = "sess_request_footprint_rotation_one"
        second_session_id = "sess_request_footprint_rotation_two"
        await run_with_key(
            key_id="rotation-key-one",
            key="1" * 32,
            session_id=first_session_id,
        )
        await run_with_key(
            key_id="rotation-key-two",
            key="2" * 32,
            session_id=second_session_id,
        )
        evidence_store = request_footprint_session_store
        reopened = None
        if isinstance(request_footprint_session_store, SQLiteSessionStore):
            reopened = SQLiteSessionStore(
                request_footprint_session_store.path,
                public_authority_alias_codec=(
                    request_footprint_session_store.public_authority_alias_codec
                ),
            )
            evidence_store = reopened
        try:
            footprints: list[RequestFootprint] = []
            for session_id in (first_session_id, second_session_id):
                durable = [
                    event
                    for event in await evidence_store.load_events(session_id)
                    if event.type == EventType.REQUEST_FOOTPRINT_RECORDED
                ]
                assert len(durable) == 1
                footprints.append(RequestFootprint.model_validate(durable[0].payload))
            return footprints[0], footprints[1]
        finally:
            if reopened is not None:
                await reopened.close()

    first, second = asyncio.run(run())

    first_identity = first.fingerprints.provider_neutral_request
    second_identity = second.fingerprints.provider_neutral_request
    assert first_identity.key_id == "rotation-key-one"
    assert second_identity.key_id == "rotation-key-two"
    assert first_identity.value != second_identity.value


@pytest.mark.parametrize(
    ("redactor_secret", "expected_availability"),
    [
        (None, "available"),
        ("sqlitefingerprintcollisiontoken", "unavailable"),
        ("keycollisionmarker", "unavailable"),
    ],
)
def test_request_footprint_survives_sqlite_reconstruction(
    tmp_path,
    redactor_secret: str | None,
    expected_availability: str,
) -> None:
    async def run() -> tuple[str | None, RequestFootprint]:
        import base64

        from cayu.runtime.public_authority import (
            PublicAuthorityAliasCodec,
            PublicAuthorityAliasKeyring,
        )

        path = tmp_path / "request-footprints.sqlite"
        alias_codec = PublicAuthorityAliasCodec(
            PublicAuthorityAliasKeyring(
                active_key_id="test",
                keys={
                    "test": SecretStr(
                        base64.urlsafe_b64encode(bytes(range(32))).decode("ascii").rstrip("=")
                    )
                },
            )
        )
        config = RequestFootprintConfig(
            fingerprint_key_id=("sqlitefingerprintcollisiontoken-keycollisionmarker-key"),
            fingerprint_key=SecretStr("q" * 32),
        )
        store = SQLiteSessionStore(path, public_authority_alias_codec=alias_codec)
        provider = FakeProvider([ModelStreamEvent.completed({"finish_reason": "stop"})])
        app = CayuApp(
            session_store=store,
            request_footprint=config,
            secret_redactor=SecretRedactor(redactor_secret),
            enable_logging=False,
        )
        app.register_provider(provider, default=True)
        app.register_agent(
            AgentSpec(
                name="assistant",
                model="fake-model",
                system_prompt="private reconstructed system prompt",
            )
        )
        returned = await collect_request_footprint_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_request_footprint_sqlite",
                messages=[Message.text("user", "private reconstructed user prompt")],
            ),
        )
        original = next(
            event.payload
            for event in returned
            if event.type == EventType.REQUEST_FOOTPRINT_RECORDED
        )
        expected_identity = original["fingerprints"]["provider_neutral_request"].get("value")
        await store.close()

        reopened = SQLiteSessionStore(path, public_authority_alias_codec=alias_codec)
        records = await reopened.query_events(
            EventQuery(
                session_id="sess_request_footprint_sqlite",
                event_types=(EventType.REQUEST_FOOTPRINT_RECORDED,),
            )
        )
        await reopened.close()
        assert len(records) == 1
        return expected_identity, RequestFootprint.model_validate(records[0].event.payload)

    expected_identity, reconstructed = asyncio.run(run())

    assert reconstructed.schema_version == 3
    assert reconstructed.tool_exposure is not None
    assert reconstructed.fingerprints.provider_neutral_request.canonicalization_version == 1
    assert reconstructed.fingerprints.provider_neutral_request.value == expected_identity
    assert reconstructed.fingerprints.provider_neutral_request.availability == (
        expected_availability
    )
    assert reconstructed.prompt_contributions.availability == expected_availability
    serialized = json.dumps(reconstructed.model_dump(mode="json"), sort_keys=True)
    assert "private reconstructed system prompt" not in serialized
    assert "private reconstructed user prompt" not in serialized
