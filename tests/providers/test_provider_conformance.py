from __future__ import annotations

import asyncio
import copy
from collections.abc import AsyncIterator
from typing import Any

import pytest

import cayu.providers as providers_module
import cayu.providers.openai as openai_module
import tests.providers.registrations as registrations_module
from cayu import (
    RESOLVED_FILE_ATTACHMENTS_OPTION,
    AgentSpec,
    CayuApp,
    EventType,
    FileAttachmentKind,
    FilePart,
    Message,
    RunRequest,
    TextPart,
    file_attachment,
)
from cayu.core.tools import Tool, ToolContext, ToolResult, ToolSpec
from cayu.providers import (
    InputTokenCountConfidence,
    InputTokenCountMethod,
    ModelContextOverflowError,
    ModelFinishReason,
    ModelProvider,
    ModelRequest,
    ModelStreamEvent,
    ModelStreamEventType,
)
from cayu.runtime.usage import normalize_usage_metrics
from tests.providers.conformance import (
    CapabilityClaim,
    ProviderCapabilities,
    ProviderConformanceFailure,
    ProviderConformanceRegistration,
    ProviderHarness,
    ProviderScenario,
    require_conformance,
)
from tests.providers.registrations import REGISTRATIONS


class _NoopProvider(ModelProvider):
    name = "noop"

    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
        del request
        if False:
            yield ModelStreamEvent.completed()


class _DifferentlyNamedAdapter(_NoopProvider):
    name = "differently-named"


async def _noop_factory(_scenario: str) -> ProviderHarness:
    return ProviderHarness(provider=_NoopProvider(), model="noop-model")


class _BrokenOrderedTextProvider(ModelProvider):
    name = "broken-order"

    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
        del request
        yield ModelStreamEvent.completed(
            {"usage": {"input_tokens": 9, "output_tokens": 2, "total_tokens": 11}}
        )
        yield ModelStreamEvent.text_delta("hello")


class _BrokenUnfinishedProvider(ModelProvider):
    name = "broken-unfinished"

    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
        del request
        yield ModelStreamEvent.text_delta("partial")


class _BrokenCapabilityProvider(_NoopProvider):
    name = "broken-capability"


class _BrokenDuplicateToolProvider(ModelProvider):
    name = "broken-duplicate-tool"

    def __init__(self) -> None:
        self.calls = 0

    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
        del request
        self.calls += 1
        if self.calls == 1:
            for _ in range(2):
                yield ModelStreamEvent.tool_call(
                    id="call-conformance",
                    name="echo",
                    arguments={"text": "conformance-tool"},
                )
            yield ModelStreamEvent.completed({"finish_reason": "tool_calls"})
            return
        yield ModelStreamEvent.text_delta("tool-observed")
        yield ModelStreamEvent.completed({"finish_reason": "stop"})


class _BrokenBlankErrorProvider(ModelProvider):
    name = "broken-blank-error"

    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
        del request
        yield ModelStreamEvent.model_construct(
            type=ModelStreamEventType.ERROR,
            delta="",
            payload={"error": ""},
            completion=None,
        )


class _BrokenDishonestUsageProvider(ModelProvider):
    name = "broken-dishonest-usage"

    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
        del request
        yield ModelStreamEvent.text_delta("hel")
        yield ModelStreamEvent.text_delta("lo")
        yield ModelStreamEvent.completed(
            {
                "model": "broken-model",
                "finish_reason": "stop",
                "usage": {"input_tokens": 900, "output_tokens": 2, "total_tokens": 902},
            }
        )


class _BrokenModelIdentityProvider(ModelProvider):
    name = "broken-model-identity"

    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
        del request
        yield ModelStreamEvent.text_delta("hel")
        yield ModelStreamEvent.text_delta("lo")
        yield ModelStreamEvent.completed(
            {
                "model": "wrong-model",
                "finish_reason": "stop",
                "usage": {"input_tokens": 9, "output_tokens": 2, "total_tokens": 11},
            }
        )


class _BrokenUnboundedCloseProvider(_NoopProvider):
    name = "broken-unbounded-close"

    async def aclose(self) -> None:
        await asyncio.Event().wait()


class _BrokenCollectionProvider(ModelProvider):
    name = "broken-collection"

    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
        del request
        raise RuntimeError("seeded collection failure")
        yield ModelStreamEvent.completed()


class _EchoTool(Tool):
    spec = ToolSpec(
        name="echo",
        description="Return the supplied text unchanged.",
        input_schema={
            "type": "object",
            "properties": {"text": {"type": "string"}},
            "required": ["text"],
            "additionalProperties": False,
        },
    )

    async def run(self, ctx: ToolContext, args: dict[str, Any]) -> ToolResult:
        del ctx
        return ToolResult(content=args["text"], structured={"echoed": args["text"]})


def _capabilities() -> ProviderCapabilities:
    return ProviderCapabilities(
        token_counting=CapabilityClaim.not_applicable("No counting endpoint."),
        native_structured_output=CapabilityClaim.not_applicable("No native schema mode."),
        attachments=CapabilityClaim.not_applicable("No attachment transport."),
        reasoning=CapabilityClaim.not_applicable("No reasoning blocks."),
        provider_cache_observation=CapabilityClaim.not_applicable("No cache counters."),
    )


def _registration(
    *,
    name: str,
    provider_type: type[ModelProvider],
    provider: ModelProvider,
    capabilities: ProviderCapabilities | None = None,
) -> ProviderConformanceRegistration:
    async def factory(_scenario: str) -> ProviderHarness:
        return ProviderHarness(provider=provider, model="broken-model")

    return ProviderConformanceRegistration(
        name=name,
        provider_type=provider_type,
        factory=factory,
        capabilities=capabilities or _capabilities(),
    )


def _assert_text_contract(
    registration: ProviderConformanceRegistration,
    harness: ProviderHarness,
    events: list[ModelStreamEvent],
) -> None:
    event_types = [event.type for event in events]
    expected_types = [
        ModelStreamEventType.TEXT_DELTA,
        ModelStreamEventType.TEXT_DELTA,
        ModelStreamEventType.COMPLETED,
    ]
    require_conformance(
        event_types == expected_types,
        registration=registration,
        scenario="text",
        observed=event_types,
    )
    require_conformance(
        "".join(event.delta for event in events[:-1]) == "hello",
        registration=registration,
        scenario="text",
        observed=[event.delta for event in events[:-1]],
    )
    completed = events[-1]
    require_conformance(
        completed.completion is not None
        and completed.completion.finish_reason == ModelFinishReason.STOP,
        registration=registration,
        scenario="text",
        observed=completed.completion,
    )
    reported_model = completed.payload.get("model")
    if registration.reports_model_identity:
        require_conformance(
            reported_model == harness.model,
            registration=registration,
            scenario="text",
            observed={"reported_model": reported_model, "requested_model": harness.model},
        )
        normalized_model = reported_model
    else:
        require_conformance(
            reported_model is None,
            registration=registration,
            scenario="text",
            observed={"unexpected_reported_model": reported_model},
        )
        normalized_model = harness.model
    usage = normalize_usage_metrics(
        provider_name=registration.name,
        model=normalized_model,
        requested_model=harness.model,
        raw_usage=completed.payload.get("usage"),
        usage_dialect=harness.provider.usage_dialect,
    )
    require_conformance(
        usage is not None
        and usage.provider_name == registration.name
        and usage.requested_model == harness.model
        and usage.model == harness.model
        and usage.input_tokens == 9
        and usage.output_tokens == 2
        and usage.total_tokens == 11,
        registration=registration,
        scenario="text",
        observed=usage,
    )


def _assert_protocol_failure(
    registration: ProviderConformanceRegistration,
    scenario: ProviderScenario,
    events: list[ModelStreamEvent],
) -> None:
    event_types = [event.type for event in events]
    error_type = events[-1].payload.get("error_type") if events else None
    valid = (
        bool(events)
        and ModelStreamEventType.COMPLETED not in event_types
        and events[-1].type == ModelStreamEventType.ERROR
        and bool(events[-1].payload.get("error"))
        and isinstance(error_type, str)
        and error_type.endswith("ProtocolError")
    )
    require_conformance(
        valid,
        registration=registration,
        scenario=scenario,
        observed={"event_types": event_types, "error_type": error_type},
    )


def _assert_token_count_contract(
    registration: ProviderConformanceRegistration,
    result: object,
) -> None:
    claim = registration.capabilities.token_counting
    if claim.state != "supported":
        require_conformance(
            bool(claim.reason) and result is None,
            registration=registration,
            scenario="token_counting",
            capability="token_counting",
            observed={"claim": claim, "result": result},
        )
        return
    valid = (
        result is not None
        and getattr(result, "input_tokens", None) == 13
        and getattr(result, "method", None) == InputTokenCountMethod.OFFICIAL
        and getattr(result, "confidence", None) == InputTokenCountConfidence.HIGH
    )
    require_conformance(
        valid,
        registration=registration,
        scenario="token_counting",
        capability="token_counting",
        observed=result,
    )


def _assert_tool_round_trip_contract(
    registration: ProviderConformanceRegistration,
    events: list[Any],
) -> None:
    tool_events = [
        event
        for event in events
        if event.type in {EventType.TOOL_CALL_STARTED, EventType.TOOL_CALL_COMPLETED}
    ]
    observed = {
        "terminal": None if not events else events[-1].type,
        "tool_events": [(event.type, event.payload.get("tool_call_id")) for event in tool_events],
    }
    valid = (
        bool(events)
        and events[-1].type == EventType.SESSION_COMPLETED
        and observed["tool_events"]
        == [
            (EventType.TOOL_CALL_STARTED, "call-conformance"),
            (EventType.TOOL_CALL_COMPLETED, "call-conformance"),
        ]
        and any(
            event.type == EventType.MODEL_TEXT_DELTA
            and event.payload.get("delta") == "tool-observed"
            for event in events
        )
    )
    require_conformance(
        valid,
        registration=registration,
        scenario="tool_round_trip",
        observed=observed,
    )


def _assert_typed_error_contract(
    registration: ProviderConformanceRegistration,
    events: list[ModelStreamEvent],
) -> None:
    payload = events[0].payload if len(events) == 1 else {}
    valid = (
        len(events) == 1
        and events[0].type == ModelStreamEventType.ERROR
        and bool(payload.get("error"))
        and payload.get("provider") == registration.expected_error_provider
        and payload.get("status_code") == 429
        and payload.get("provider_error_type") == "rate_limit_error"
        and payload.get("provider_error_code") == "rate_limit_exceeded"
        and payload.get("request_id") == "req-conformance"
        and payload.get("retryable") is True
        and payload.get("retry_after_s") == 0.25
    )
    require_conformance(
        valid,
        registration=registration,
        scenario="typed_error",
        observed={"event_count": len(events), "payload": payload},
    )


async def _assert_close_contract(
    registration: ProviderConformanceRegistration,
    harness: ProviderHarness,
    *,
    timeout_s: float = 0.5,
) -> None:
    try:
        await asyncio.wait_for(harness.aclose(), timeout=timeout_s)
    except TimeoutError:
        require_conformance(
            False,
            registration=registration,
            scenario="close",
            observed=f"close exceeded {timeout_s:g} seconds",
        )
    is_closed = harness.is_closed
    require_conformance(
        is_closed is not None,
        registration=registration,
        scenario="close",
        observed="close-state probe is not configured",
    )
    if is_closed is None:
        return
    require_conformance(
        is_closed(),
        registration=registration,
        scenario="close",
        observed="owned resource remained open",
    )


async def _collect_and_assert_close_contract(
    registration: ProviderConformanceRegistration,
    harness: ProviderHarness,
) -> None:
    try:
        await harness.collect()
    finally:
        await _assert_close_contract(registration, harness)


async def _collect_tool_round_trip_events(harness: ProviderHarness) -> list[Any]:
    app = CayuApp(enable_logging=False)
    app.register_provider(harness.provider, default=True)
    app.register_agent(
        AgentSpec(name="assistant", model=harness.model, system_prompt="Use the echo tool."),
        tools=[_EchoTool()],
    )
    return [
        event
        async for event in app.run(
            RunRequest(
                agent_name="assistant",
                messages=[Message.text("user", "Echo the text conformance-tool.")],
                max_steps=3,
            )
        )
    ]


def test_provider_conformance_registration_requires_bounded_capability_reasons() -> None:
    with pytest.raises(ValueError, match="require a reason"):
        CapabilityClaim.not_applicable("")
    with pytest.raises(ValueError, match="at most"):
        CapabilityClaim.unsupported("x" * 241)

    registration = ProviderConformanceRegistration(
        name="noop",
        provider_type=_NoopProvider,
        factory=_noop_factory,
        capabilities=_capabilities(),
    )

    assert registration.provider_type is _NoopProvider


def _exported_builtin_provider_types() -> set[type[ModelProvider]]:
    return {
        value
        for name in providers_module.__all__
        if isinstance((value := getattr(providers_module, name)), type)
        and value is not ModelProvider
        and issubclass(value, ModelProvider)
    }


def test_provider_conformance_registry_covers_every_exported_builtin_provider() -> None:
    registered = {registration.provider_type for registration in REGISTRATIONS}
    exported = _exported_builtin_provider_types()

    assert registered == exported
    assert len({registration.name for registration in REGISTRATIONS}) == len(REGISTRATIONS)


def test_provider_registry_discovers_exported_adapter_without_provider_suffix(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    export_name = "DifferentlyNamedAdapter"
    monkeypatch.setattr(providers_module, export_name, _DifferentlyNamedAdapter, raising=False)
    monkeypatch.setattr(providers_module, "__all__", [*providers_module.__all__, export_name])

    assert _DifferentlyNamedAdapter in _exported_builtin_provider_types()


@pytest.mark.parametrize(
    ("validator", "assistant_echo_only"),
    [
        (
            registrations_module._require_openai_tool_result,
            {
                "input": [
                    {
                        "type": "function_call",
                        "call_id": "call-conformance",
                        "name": "echo",
                        "arguments": '{"text":"conformance-tool"}',
                    }
                ]
            },
        ),
        (
            registrations_module._require_anthropic_tool_result,
            {
                "messages": [
                    {
                        "role": "assistant",
                        "content": [
                            {
                                "type": "tool_use",
                                "id": "call-conformance",
                                "name": "echo",
                                "input": {"text": "conformance-tool"},
                            }
                        ],
                    }
                ]
            },
        ),
        (
            registrations_module._require_chat_completions_tool_result,
            {
                "messages": [
                    {
                        "role": "assistant",
                        "tool_calls": [
                            {
                                "id": "call-conformance",
                                "type": "function",
                                "function": {
                                    "name": "echo",
                                    "arguments": '{"text":"conformance-tool"}',
                                },
                            }
                        ],
                    }
                ]
            },
        ),
        (
            registrations_module._require_bedrock_tool_result,
            {
                "messages": [
                    {
                        "role": "assistant",
                        "content": [
                            {
                                "toolUse": {
                                    "toolUseId": "call-conformance",
                                    "name": "echo",
                                    "input": {"text": "conformance-tool"},
                                }
                            }
                        ],
                    }
                ]
            },
        ),
    ],
    ids=("openai", "anthropic", "chat-completions", "bedrock"),
)
def test_tool_result_validators_reject_assistant_echo_without_result(
    validator: Any,
    assistant_echo_only: dict[str, Any],
) -> None:
    with pytest.raises(AssertionError, match="omitted the completed tool result"):
        validator(assistant_echo_only, tool_call_id="call-conformance")


@pytest.mark.anyio
async def test_tool_round_trip_contract_rejects_request_builder_that_drops_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_builder = openai_module.build_openai_payload

    def build_without_tool_result(
        request: ModelRequest,
        *,
        stream: bool = False,
        reasoning_state: str = "inline",
        chain: bool = True,
    ) -> dict[str, Any]:
        payload = original_builder(
            request,
            stream=stream,
            reasoning_state=reasoning_state,
            chain=chain,
        )
        payload["input"] = [
            item
            for item in payload["input"]
            if not (isinstance(item, dict) and item.get("type") == "function_call_output")
        ]
        return payload

    monkeypatch.setattr(openai_module, "build_openai_payload", build_without_tool_result)
    registration = registrations_module.OPENAI
    harness = await registration.factory("tool_round_trip")
    try:
        events = await _collect_tool_round_trip_events(harness)
    finally:
        await harness.aclose()

    with pytest.raises(
        ProviderConformanceFailure,
        match=r"adapter=openai scenario=tool_round_trip .*observed=",
    ):
        _assert_tool_round_trip_contract(registration, events)


@pytest.mark.anyio
async def test_provider_conformance_seeded_bad_order_reports_actionable_failure() -> None:
    provider = _BrokenOrderedTextProvider()
    registration = _registration(
        name="broken-order",
        provider_type=_BrokenOrderedTextProvider,
        provider=provider,
    )
    harness = await registration.factory("text")
    events = await harness.collect()

    with pytest.raises(
        ProviderConformanceFailure,
        match=r"adapter=broken-order scenario=text .*observed=",
    ):
        _assert_text_contract(registration, harness, events)


@pytest.mark.anyio
async def test_provider_conformance_seeded_unfinished_stream_reports_failure() -> None:
    provider = _BrokenUnfinishedProvider()
    registration = _registration(
        name="broken-unfinished",
        provider_type=_BrokenUnfinishedProvider,
        provider=provider,
    )
    harness = await registration.factory("unfinished")
    events = await harness.collect()

    with pytest.raises(
        ProviderConformanceFailure,
        match=r"adapter=broken-unfinished scenario=unfinished .*observed=",
    ):
        _assert_protocol_failure(registration, "unfinished", events)


@pytest.mark.anyio
async def test_provider_conformance_seeded_false_capability_reports_failure() -> None:
    provider = _BrokenCapabilityProvider()
    capabilities = _capabilities()
    registration = _registration(
        name="broken-capability",
        provider_type=_BrokenCapabilityProvider,
        provider=provider,
        capabilities=ProviderCapabilities(
            token_counting=CapabilityClaim.supported(),
            native_structured_output=capabilities.native_structured_output,
            attachments=capabilities.attachments,
            reasoning=capabilities.reasoning,
            provider_cache_observation=capabilities.provider_cache_observation,
        ),
    )
    harness = await registration.factory("token_counting")
    result = await harness.provider.count_input_tokens(
        ModelRequest(
            model=harness.model,
            messages=[Message.text("user", "Count this.")],
        )
    )

    with pytest.raises(
        ProviderConformanceFailure,
        match=(
            r"adapter=broken-capability scenario=token_counting "
            r"capability=token_counting observed="
        ),
    ):
        _assert_token_count_contract(registration, result)


@pytest.mark.anyio
async def test_provider_conformance_seeded_duplicate_tool_reports_failure() -> None:
    provider = _BrokenDuplicateToolProvider()
    registration = _registration(
        name="broken-duplicate-tool",
        provider_type=_BrokenDuplicateToolProvider,
        provider=provider,
    )
    app = CayuApp(enable_logging=False)
    app.register_provider(provider, default=True)
    app.register_agent(
        AgentSpec(name="assistant", model="broken-model", system_prompt="Use echo."),
        tools=[_EchoTool()],
    )
    events = [
        event
        async for event in app.run(
            RunRequest(
                agent_name="assistant",
                messages=[Message.text("user", "Echo conformance-tool.")],
                max_steps=3,
            )
        )
    ]
    tool_events = [
        event
        for event in events
        if event.type in {EventType.TOOL_CALL_STARTED, EventType.TOOL_CALL_COMPLETED}
    ]

    assert tool_events
    assert {event.payload.get("tool_call_id") for event in tool_events} == {"call-conformance"}

    with pytest.raises(
        ProviderConformanceFailure,
        match=r"adapter=broken-duplicate-tool scenario=tool_round_trip .*observed=",
    ):
        _assert_tool_round_trip_contract(registration, events)


@pytest.mark.anyio
async def test_provider_conformance_seeded_blank_error_reports_failure() -> None:
    provider = _BrokenBlankErrorProvider()
    registration = _registration(
        name="broken-blank-error",
        provider_type=_BrokenBlankErrorProvider,
        provider=provider,
    )
    harness = await registration.factory("typed_error")
    events = await harness.collect()

    with pytest.raises(
        ProviderConformanceFailure,
        match=r"adapter=broken-blank-error scenario=typed_error .*observed=",
    ):
        _assert_typed_error_contract(registration, events)


@pytest.mark.anyio
async def test_provider_conformance_seeded_dishonest_usage_reports_failure() -> None:
    provider = _BrokenDishonestUsageProvider()
    registration = _registration(
        name="broken-dishonest-usage",
        provider_type=_BrokenDishonestUsageProvider,
        provider=provider,
    )
    harness = await registration.factory("text")
    events = await harness.collect()

    with pytest.raises(
        ProviderConformanceFailure,
        match=r"adapter=broken-dishonest-usage scenario=text .*observed=",
    ):
        _assert_text_contract(registration, harness, events)


@pytest.mark.anyio
async def test_provider_conformance_seeded_wrong_model_identity_reports_failure() -> None:
    provider = _BrokenModelIdentityProvider()
    registration = _registration(
        name="broken-model-identity",
        provider_type=_BrokenModelIdentityProvider,
        provider=provider,
    )
    harness = await registration.factory("text")
    events = await harness.collect()

    with pytest.raises(
        ProviderConformanceFailure,
        match=r"adapter=broken-model-identity scenario=text .*observed=",
    ):
        _assert_text_contract(registration, harness, events)


@pytest.mark.anyio
async def test_provider_conformance_seeded_unbounded_close_reports_failure() -> None:
    provider = _BrokenUnboundedCloseProvider()

    async def factory(_scenario: str) -> ProviderHarness:
        return ProviderHarness(
            provider=provider,
            model="broken-model",
            is_closed=lambda: False,
        )

    registration = ProviderConformanceRegistration(
        name="broken-unbounded-close",
        provider_type=_BrokenUnboundedCloseProvider,
        factory=factory,
        capabilities=_capabilities(),
    )
    harness = await registration.factory("close")

    with pytest.raises(
        ProviderConformanceFailure,
        match=r"adapter=broken-unbounded-close scenario=close .*observed=",
    ):
        await _assert_close_contract(registration, harness, timeout_s=0.01)


@pytest.mark.anyio
async def test_provider_conformance_closes_harness_after_collection_failure() -> None:
    provider = _BrokenCollectionProvider()
    closed = False

    async def close() -> None:
        nonlocal closed
        closed = True

    registration = _registration(
        name="broken-collection",
        provider_type=_BrokenCollectionProvider,
        provider=provider,
    )
    harness = ProviderHarness(
        provider=provider,
        model="broken-model",
        close=close,
        is_closed=lambda: closed,
    )

    with pytest.raises(RuntimeError, match="seeded collection failure"):
        await _collect_and_assert_close_contract(registration, harness)

    assert closed is True


@pytest.mark.anyio
@pytest.mark.parametrize("registration", REGISTRATIONS, ids=lambda item: item.name)
async def test_provider_conformance_streams_ordered_text_and_normalized_usage(
    registration: ProviderConformanceRegistration,
) -> None:
    harness = await registration.factory("text")
    try:
        events = await harness.collect()
    finally:
        await harness.aclose()

    _assert_text_contract(registration, harness, events)


@pytest.mark.anyio
@pytest.mark.parametrize("registration", REGISTRATIONS, ids=lambda item: item.name)
async def test_provider_conformance_round_trips_one_logical_tool_call(
    registration: ProviderConformanceRegistration,
) -> None:
    harness = await registration.factory("tool_round_trip")
    try:
        events = await _collect_tool_round_trip_events(harness)
    finally:
        await harness.aclose()

    _assert_tool_round_trip_contract(registration, events)


@pytest.mark.anyio
@pytest.mark.parametrize("registration", REGISTRATIONS, ids=lambda item: item.name)
async def test_provider_conformance_preserves_typed_retryable_errors(
    registration: ProviderConformanceRegistration,
) -> None:
    harness = await registration.factory("typed_error")
    try:
        events = await harness.collect()
    finally:
        await harness.aclose()

    _assert_typed_error_contract(registration, events)


@pytest.mark.anyio
@pytest.mark.parametrize("registration", REGISTRATIONS, ids=lambda item: item.name)
async def test_provider_conformance_propagates_context_overflow_as_typed_exception(
    registration: ProviderConformanceRegistration,
) -> None:
    harness = await registration.factory("context_overflow")
    try:
        with pytest.raises(ModelContextOverflowError) as exc_info:
            await harness.collect()
    finally:
        await harness.aclose()

    error = exc_info.value
    assert error.provider == registration.expected_error_provider
    assert error.status_code == 400
    assert error.error_code == "context_length_exceeded"
    assert error.request_id == "req-overflow"
    assert error.retryable is False


@pytest.mark.anyio
@pytest.mark.parametrize("registration", REGISTRATIONS, ids=lambda item: item.name)
async def test_provider_conformance_rejects_unfinished_streams(
    registration: ProviderConformanceRegistration,
) -> None:
    harness = await registration.factory("unfinished")
    try:
        events = await harness.collect()
    finally:
        await harness.aclose()

    _assert_protocol_failure(registration, "unfinished", events)


@pytest.mark.anyio
@pytest.mark.parametrize("registration", REGISTRATIONS, ids=lambda item: item.name)
async def test_provider_conformance_rejects_malformed_stream_events(
    registration: ProviderConformanceRegistration,
) -> None:
    harness = await registration.factory("malformed")
    try:
        events = await harness.collect()
    finally:
        await harness.aclose()

    _assert_protocol_failure(registration, "malformed", events)


@pytest.mark.anyio
@pytest.mark.parametrize("scenario", ["malformed_terminal", "malformed_usage"])
@pytest.mark.parametrize("registration", REGISTRATIONS, ids=lambda item: item.name)
async def test_provider_conformance_rejects_malformed_terminal_and_usage_payloads(
    registration: ProviderConformanceRegistration,
    scenario: ProviderScenario,
) -> None:
    harness = await registration.factory(scenario)
    try:
        events = await harness.collect()
    finally:
        await harness.aclose()

    _assert_protocol_failure(registration, scenario, events)


@pytest.mark.anyio
@pytest.mark.parametrize("registration", REGISTRATIONS, ids=lambda item: item.name)
async def test_provider_conformance_rejects_unfinished_reasoning_blocks_when_supported(
    registration: ProviderConformanceRegistration,
) -> None:
    claim = registration.capabilities.reasoning
    if claim.state != "supported":
        assert claim.reason
        return
    harness = await registration.factory("unfinished_reasoning")
    try:
        events = await harness.collect()
    finally:
        await harness.aclose()

    _assert_protocol_failure(registration, "unfinished_reasoning", events)


@pytest.mark.anyio
@pytest.mark.parametrize("registration", REGISTRATIONS, ids=lambda item: item.name)
async def test_provider_conformance_cancellation_is_bounded_and_releases_stream(
    registration: ProviderConformanceRegistration,
) -> None:
    harness = await registration.factory("cancellation")
    assert harness.wait_started is not None
    assert harness.wait_stopped is not None
    task = asyncio.create_task(harness.collect())
    try:
        await asyncio.wait_for(harness.wait_started(), timeout=0.5)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, timeout=0.5)
        await asyncio.wait_for(harness.wait_stopped(), timeout=0.5)
    finally:
        if not task.done():
            task.cancel()
        await harness.aclose()


@pytest.mark.anyio
@pytest.mark.parametrize("registration", REGISTRATIONS, ids=lambda item: item.name)
async def test_provider_conformance_idle_failure_is_bounded(
    registration: ProviderConformanceRegistration,
) -> None:
    harness = await registration.factory("idle_timeout")
    try:
        events = await asyncio.wait_for(harness.collect(), timeout=0.5)
    finally:
        await harness.aclose()

    assert [event.type for event in events] == [ModelStreamEventType.ERROR]
    expected_error_type = (
        "BedrockProtocolError" if registration.name == "bedrock" else "TimeoutError"
    )
    assert events[0].payload["error_type"] == expected_error_type
    assert events[0].payload["error"].strip()


@pytest.mark.anyio
@pytest.mark.parametrize("registration", REGISTRATIONS, ids=lambda item: item.name)
async def test_provider_conformance_close_is_bounded_and_releases_owned_resources(
    registration: ProviderConformanceRegistration,
) -> None:
    harness = await registration.factory("close")
    await _collect_and_assert_close_contract(registration, harness)


@pytest.mark.anyio
@pytest.mark.parametrize("registration", REGISTRATIONS, ids=lambda item: item.name)
async def test_provider_conformance_token_counting_matches_capability_claim(
    registration: ProviderConformanceRegistration,
) -> None:
    harness = await registration.factory("token_counting")
    request = ModelRequest(
        model=harness.model,
        messages=[Message.text("user", "Count this request.")],
    )
    try:
        result = await harness.provider.count_input_tokens(request)
    finally:
        await harness.aclose()

    _assert_token_count_contract(registration, result)


@pytest.mark.anyio
@pytest.mark.parametrize("registration", REGISTRATIONS, ids=lambda item: item.name)
async def test_provider_conformance_native_structured_output_matches_capability_claim(
    registration: ProviderConformanceRegistration,
) -> None:
    claim = registration.capabilities.native_structured_output
    provider_supports_native = registration.provider_type.supports_native_structured_output

    assert provider_supports_native is (claim.state == "supported")
    if claim.state != "supported":
        assert claim.reason
        return
    harness = await registration.factory("native_structured_output")
    schema = {
        "type": "object",
        "properties": {"answer": {"type": "string"}},
        "required": ["answer"],
        "additionalProperties": False,
    }
    original = copy.deepcopy(schema)
    try:
        harness.provider.preflight_native_structured_output_schema(schema)
    finally:
        await harness.aclose()

    assert schema == original


@pytest.mark.anyio
@pytest.mark.parametrize("registration", REGISTRATIONS, ids=lambda item: item.name)
async def test_provider_conformance_projects_resolved_attachments_when_supported(
    registration: ProviderConformanceRegistration,
) -> None:
    claim = registration.capabilities.attachments
    if claim.state != "supported":
        assert claim.reason
        return
    harness = await registration.factory("attachments")
    attachment = file_attachment(
        artifact_id="image-conformance",
        kind=FileAttachmentKind.IMAGE,
        filename="conformance.png",
        content_type="image/png",
        size_bytes=5,
    )
    request = ModelRequest(
        model=harness.model,
        messages=[
            Message(
                role="user",
                content=[
                    TextPart(text="Inspect this image."),
                    FilePart(attachment=attachment),
                ],
            )
        ],
        options={
            RESOLVED_FILE_ATTACHMENTS_OPTION: {
                "image-conformance": {
                    "artifact_id": "image-conformance",
                    "kind": "image",
                    "filename": "conformance.png",
                    "content_type": "image/png",
                    "data_base64": "aGVsbG8=",
                    "metadata": {},
                }
            }
        },
    )
    try:
        events = await harness.collect(request)
    finally:
        await harness.aclose()

    assert events[-1].type == ModelStreamEventType.COMPLETED, events


@pytest.mark.anyio
@pytest.mark.parametrize("registration", REGISTRATIONS, ids=lambda item: item.name)
async def test_provider_conformance_reasoning_matches_capability_claim(
    registration: ProviderConformanceRegistration,
) -> None:
    claim = registration.capabilities.reasoning
    if claim.state != "supported":
        assert claim.reason
        return
    harness = await registration.factory("reasoning")
    try:
        events = await harness.collect()
    finally:
        await harness.aclose()

    thinking = [event for event in events if event.type == ModelStreamEventType.THINKING]
    assert thinking
    assert "".join(event.delta for event in thinking) == "think"
    terminal = events[-1]
    require_conformance(
        terminal.type == ModelStreamEventType.COMPLETED
        and terminal.completion is not None
        and terminal.completion.finish_reason == ModelFinishReason.STOP,
        registration=registration,
        scenario="reasoning",
        capability="reasoning",
        observed=terminal,
    )


@pytest.mark.anyio
@pytest.mark.parametrize("registration", REGISTRATIONS, ids=lambda item: item.name)
async def test_provider_conformance_cache_observation_matches_capability_claim(
    registration: ProviderConformanceRegistration,
) -> None:
    claim = registration.capabilities.provider_cache_observation
    if claim.state != "supported":
        assert claim.reason
        return
    harness = await registration.factory("provider_cache_observation")
    try:
        events = await harness.collect()
    finally:
        await harness.aclose()

    completed = events[-1]
    assert completed.type == ModelStreamEventType.COMPLETED
    usage = normalize_usage_metrics(
        provider_name=registration.name,
        model=harness.model,
        requested_model=harness.model,
        raw_usage=completed.payload.get("usage"),
        usage_dialect=harness.provider.usage_dialect,
    )
    assert usage is not None
    assert usage.cache.read_tokens == 3
    assert usage.cache.cached_input_tokens == 3
