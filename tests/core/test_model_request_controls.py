"""Sessions record the generation controls each model call sent and what came back."""

from __future__ import annotations

import io
import json
from collections.abc import Mapping
from contextlib import redirect_stdout
from typing import Any

import pytest
from tests.core.test_openai_provider import RecordingTransport
from tests.core.test_vertex_provider import FakeCredentials

from cayu import AgentSpec, CayuApp, EventType, Message, RunRequest
from cayu.cli.session import _render_usage
from cayu.context import ThinkingConfig
from cayu.context.thinking import thinking_config_payload
from cayu.evals.testing import ScriptedModelProvider
from cayu.providers import (
    AnthropicProvider,
    BedrockProvider,
    ChatCompletionsProvider,
    GatewayProvider,
    ModelProvider,
    ModelRequest,
    ModelRequestControls,
    ModelStreamEvent,
    OpenAIProvider,
    VertexProvider,
)
from cayu.providers.request_controls import (
    copy_echoed_request_controls,
    openai_responses_echoed_request_controls,
    request_controls_from_values,
    request_controls_mismatch,
)
from cayu.server.contracts import ApiSessionModelCall
from cayu.sessions._call_inspection import InspectionWindow, inspect_model_calls
from cayu.sessions.event_queries import EventQuery


def _request(model: str, **options: Any) -> ModelRequest:
    return ModelRequest(model=model, messages=[Message.text("user", "Hi.")], options=options)


def _thinking(**values: Any) -> dict[str, Any]:
    return thinking_config_payload(ThinkingConfig(**values))


# --- the provider-neutral record -------------------------------------------------------


def test_controls_payload_omits_unsent_controls_and_keeps_field_order() -> None:
    controls = ModelRequestControls(
        max_output_tokens=128_000, temperature=1, reasoning_effort="high"
    )

    assert controls.payload() == {
        "reasoning_effort": "high",
        "temperature": 1.0,
        "max_output_tokens": 128_000,
    }
    assert list(controls.payload()) == ["reasoning_effort", "temperature", "max_output_tokens"]
    assert ModelRequestControls().payload() == {}


@pytest.mark.parametrize(
    "values",
    [
        {"temperature": float("nan")},
        {"temperature": True},
        {"temperature": "0.7"},
        {"max_output_tokens": -1},
        {"max_output_tokens": 1.5},
        {"top_k": False},
        {"reasoning_effort": " "},
        {"reasoning_effort": "x" * 65},
    ],
)
def test_controls_reject_values_that_are_not_controls(values: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        ModelRequestControls(**values)


def test_controls_from_wire_values_drop_unusable_values_instead_of_failing() -> None:
    controls = request_controls_from_values(
        temperature="0.7",
        top_p=0.9,
        max_output_tokens=None,
        reasoning_effort="medium",
    )

    assert controls.payload() == {"reasoning_effort": "medium", "top_p": 0.9}
    with pytest.raises(TypeError):
        request_controls_from_values(seed=1)


def test_echoed_controls_keep_explicit_nulls_and_drop_unknown_or_malformed_fields() -> None:
    assert copy_echoed_request_controls(
        {"max_output_tokens": None, "temperature": "hot", "seed": 3, "reasoning_effort": "xhigh"}
    ) == {"reasoning_effort": "xhigh", "max_output_tokens": None}
    assert copy_echoed_request_controls({"seed": 3}) is None
    assert copy_echoed_request_controls(["max_output_tokens"]) is None


def test_mismatch_names_sent_controls_echoed_differently() -> None:
    sent = {"reasoning_effort": "high", "temperature": 0.7, "max_output_tokens": 128_000}

    assert request_controls_mismatch(
        sent,
        {"reasoning_effort": "high", "temperature": 0.7, "top_p": 0.98, "max_output_tokens": None},
    ) == ["max_output_tokens"]
    assert request_controls_mismatch(sent, {"reasoning_effort": "xhigh", "temperature": 0.7}) == [
        "reasoning_effort"
    ]
    # An echoed value for a control Cayu didn't send is the provider default.
    assert request_controls_mismatch({"temperature": 1}, {"temperature": 1.0, "top_p": 1.0}) == []
    assert request_controls_mismatch(None, {"max_output_tokens": None}) == []
    assert request_controls_mismatch(sent, None) == []


def test_openai_responses_echo_reads_only_fields_the_response_carries() -> None:
    assert openai_responses_echoed_request_controls(
        {
            "id": "resp_1",
            "reasoning": {"effort": "xhigh", "summary": "auto"},
            "max_output_tokens": None,
            "temperature": 1.0,
            "top_p": 0.98,
            "instructions": "secret system prompt",
        }
    ) == {
        "reasoning_effort": "xhigh",
        "temperature": 1.0,
        "top_p": 0.98,
        "max_output_tokens": None,
    }
    assert openai_responses_echoed_request_controls({"reasoning": None}) == {
        "reasoning_effort": None
    }
    assert openai_responses_echoed_request_controls({"id": "resp_1", "status": "completed"}) is None


# --- each built-in adapter reports its effective controls ------------------------------


def test_custom_providers_report_no_controls_by_default() -> None:
    class Custom(ModelProvider):
        name = "custom"

        async def stream(self, request):
            yield ModelStreamEvent.completed({})

    assert Custom().request_controls(_request("m", temperature=0.2)) is None
    assert ScriptedModelProvider([]).request_controls(_request("m")) is None


def test_openai_reports_typed_effort_over_raw_options_and_the_raw_cap() -> None:
    provider = OpenAIProvider(api_key="test-key")
    request = _request(
        "gpt-5.6",
        thinking=_thinking(effort="high"),
        openai={
            "reasoning": {"effort": "low"},
            "temperature": 0.4,
            "top_p": 0.9,
            "max_output_tokens": 128_000,
            "metadata": {"tenant": "acme"},
        },
    )

    assert provider.request_controls(request) == ModelRequestControls(
        reasoning_effort="high", temperature=0.4, top_p=0.9, max_output_tokens=128_000
    )
    assert provider.request_controls(_request("gpt-5.6")) == ModelRequestControls()


def test_anthropic_and_vertex_report_their_default_cap_and_thinking_mode() -> None:
    anthropic = AnthropicProvider(api_key="test-key", max_tokens=4096)
    vertex = VertexProvider(
        project_id="project", region="us-east5", credentials=FakeCredentials(), max_tokens=2048
    )

    assert anthropic.request_controls(_request("claude-opus-4-8")) == ModelRequestControls(
        max_output_tokens=4096
    )
    assert anthropic.request_controls(
        _request("claude-opus-4-8", thinking=_thinking(effort="xhigh"), anthropic={"top_k": 40})
    ) == ModelRequestControls(reasoning_effort="xhigh", top_k=40, max_output_tokens=4096)
    budget = anthropic.request_controls(
        _request("claude-sonnet-4-5", thinking=_thinking(max_tokens=8000))
    )
    assert budget is not None
    assert budget.thinking_budget_tokens == 8000
    # The adapter raises the cap above the budget, and the record says so.
    assert budget.max_output_tokens is not None and budget.max_output_tokens > 8000
    assert vertex.request_controls(
        _request("claude-opus-4-8", anthropic={"temperature": 0.2})
    ) == ModelRequestControls(temperature=0.2, max_output_tokens=2048)


def test_chat_completions_and_gateway_report_effort_and_either_cap_field() -> None:
    chat = ChatCompletionsProvider(api_key="test-key", base_url="https://example.test/v1")
    gateway = GatewayProvider(api_key="test-key", base_url="https://gateway.example.test/v1")

    assert chat.request_controls(
        _request(
            "gpt-5.6",
            thinking=_thinking(effort="medium"),
            openai_chat={"max_completion_tokens": 900, "max_tokens": 100, "temperature": 0},
        )
    ) == ModelRequestControls(reasoning_effort="medium", temperature=0.0, max_output_tokens=900)
    assert chat.request_controls(
        _request("compatible-model", openai_chat={"max_tokens": 100, "top_k": 20})
    ) == ModelRequestControls(top_k=20, max_output_tokens=100)
    assert gateway.request_controls(
        _request("openai/gpt-5.6", cayu_gateway={"top_p": 0.5})
    ) == ModelRequestControls(top_p=0.5)


def test_bedrock_reports_inference_config_and_anthropic_style_fields() -> None:
    provider = BedrockProvider(client=object(), max_tokens=1000)

    assert provider.request_controls(
        _request(
            "anthropic.claude-sonnet-4-5",
            bedrock={
                "inferenceConfig": {"temperature": 0.3, "topP": 0.8},
                "additionalModelRequestFields": {
                    "top_k": 50,
                    "thinking": {"type": "enabled", "budget_tokens": 2048},
                },
            },
        )
    ) == ModelRequestControls(
        thinking_budget_tokens=2048,
        temperature=0.3,
        top_p=0.8,
        top_k=50,
        max_output_tokens=1000,
    )


# --- the runtime records sent and echoed controls --------------------------------------


def _openai_completed(response_fields: Mapping[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "type": "response.completed",
            "response": {
                "id": "resp_1",
                "model": "gpt-5.6",
                "status": "completed",
                "output": [
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": "done"}],
                    }
                ],
                "usage": {"input_tokens": 3, "output_tokens": 2, "total_tokens": 5},
                **response_fields,
            },
        }
    ]


async def _run_openai(
    transport: RecordingTransport,
    *,
    session_id: str,
    provider: OpenAIProvider | None = None,
) -> tuple[CayuApp, list[Any]]:
    app = CayuApp(enable_logging=False)
    app.register_provider(
        provider or OpenAIProvider(api_key="test-key", transport=transport), default=True
    )
    app.register_agent(
        AgentSpec(
            name="assistant",
            model="gpt-5.6",
            thinking=ThinkingConfig(effort="high"),
            provider_options={"openai": {"temperature": 0.5, "max_output_tokens": 128_000}},
        )
    )
    events = [
        event
        async for event in app.run(
            RunRequest(
                agent_name="assistant",
                session_id=session_id,
                messages=[Message.text("user", "Solve it.")],
                max_steps=1,
            )
        )
    ]
    return app, events


@pytest.mark.anyio
async def test_session_records_sent_controls_echo_and_a_dropped_output_cap() -> None:
    transport = RecordingTransport(
        stream_events=[
            _openai_completed(
                {
                    "reasoning": {"effort": "high", "summary": "auto"},
                    "max_output_tokens": None,
                    "temperature": 0.5,
                    "top_p": 0.98,
                }
            )
        ]
    )
    app, events = await _run_openai(transport, session_id="controls-dropped-cap")
    try:
        assert events[-1].type is EventType.SESSION_COMPLETED
        sent = transport.calls[0]["payload"]
        assert sent["reasoning"]["effort"] == "high"
        assert sent["max_output_tokens"] == 128_000
        expected_sent = {
            "reasoning_effort": "high",
            "temperature": 0.5,
            "max_output_tokens": 128_000,
        }
        expected_echo = {
            "reasoning_effort": "high",
            "temperature": 0.5,
            "top_p": 0.98,
            "max_output_tokens": None,
        }

        records = await app.session_store.query_events(
            EventQuery(
                session_id="controls-dropped-cap",
                event_types=(EventType.MODEL_STARTED, EventType.MODEL_COMPLETED),
            )
        )
        started, completed = (record.event for record in records)
        assert started.payload["request_controls"] == expected_sent
        assert completed.payload["echoed_request_controls"] == expected_echo
        assert completed.payload["request_controls_mismatch"] == ["max_output_tokens"]
        # Prompts never enter either record.
        assert "Solve it." not in json.dumps(started.payload["request_controls"])

        usage = await inspect_model_calls(
            app.session_store,
            "controls-dropped-cap",
            InspectionWindow(),
            offset=0,
            limit=10,
        )
        (call,) = usage["calls"]
        assert call["request_controls"] == expected_sent
        assert call["echoed_request_controls"] == expected_echo
        assert call["request_controls_mismatch"] == ["max_output_tokens"]
        ApiSessionModelCall.model_validate(call)

        output = io.StringIO()
        with redirect_stdout(output):
            _render_usage(
                "table",
                {"schema_version": "9", "aggregate": {}, "price_book": None},
                usage["calls"],
                [],
            )
        section = output.getvalue().split("Request controls\n", 1)[1].splitlines()
        assert section[0].split() == ["sequence", "sent", "echoed", "mismatch"]
        row = section[1]
        assert "reasoning_effort=high temperature=0.5 max_output_tokens=128000" in row
        assert "top_p=0.98 max_output_tokens=null" in row
        assert row.rstrip().endswith("max_output_tokens")
    finally:
        await app.aclose()


@pytest.mark.anyio
async def test_matching_echo_records_no_mismatch() -> None:
    transport = RecordingTransport(
        stream_events=[
            _openai_completed(
                {
                    "reasoning": {"effort": "high"},
                    "max_output_tokens": 128_000,
                    "temperature": 0.5,
                }
            )
        ]
    )
    app, _ = await _run_openai(transport, session_id="controls-match")
    try:
        records = await app.session_store.query_events(
            EventQuery(session_id="controls-match", event_types=(EventType.MODEL_COMPLETED,))
        )
        payload = records[0].event.payload
        assert payload["echoed_request_controls"] == {
            "reasoning_effort": "high",
            "temperature": 0.5,
            "max_output_tokens": 128_000,
        }
        assert "request_controls_mismatch" not in payload
        usage = await inspect_model_calls(
            app.session_store, "controls-match", InspectionWindow(), offset=0, limit=10
        )
        assert usage["calls"][0]["request_controls_mismatch"] == []
    finally:
        await app.aclose()


@pytest.mark.anyio
async def test_an_unstorable_echoed_number_does_not_fail_the_call() -> None:
    transport = RecordingTransport(
        stream_events=[_openai_completed({"temperature": 1e20, "top_p": 0.98})]
    )
    app, events = await _run_openai(transport, session_id="controls-unstorable-echo")
    try:
        assert events[-1].type is EventType.SESSION_COMPLETED
        completed = next(event for event in events if event.type is EventType.MODEL_COMPLETED)
        assert completed.payload["echoed_request_controls"] == {"top_p": 0.98}
    finally:
        await app.aclose()


@pytest.mark.anyio
async def test_a_failing_controls_hook_leaves_the_call_unrecorded_not_failed() -> None:
    class BrokenHook(OpenAIProvider):
        def request_controls(self, request):
            raise RuntimeError("cannot report")

    transport = RecordingTransport(stream_events=[_openai_completed({})])
    app, events = await _run_openai(
        transport,
        session_id="controls-broken-hook",
        provider=BrokenHook(api_key="test-key", transport=transport),
    )
    try:
        assert events[-1].type is EventType.SESSION_COMPLETED
        started = next(event for event in events if event.type is EventType.MODEL_STARTED)
        assert "request_controls" not in started.payload
        usage = await inspect_model_calls(
            app.session_store, "controls-broken-hook", InspectionWindow(), offset=0, limit=10
        )
        call = usage["calls"][0]
        assert call["request_controls"] is None
        assert call["echoed_request_controls"] is None
        assert call["request_controls_mismatch"] is None
    finally:
        await app.aclose()


@pytest.mark.anyio
async def test_provider_payload_cannot_claim_the_runtime_mismatch_flag() -> None:
    class Provider(ScriptedModelProvider):
        def request_controls(self, request):
            return ModelRequestControls(max_output_tokens=50)

        async def stream(self, request):
            yield ModelStreamEvent.text_delta("ok")
            yield ModelStreamEvent.completed(
                {
                    "finish_reason": "stop",
                    "echoed_request_controls": {"max_output_tokens": 50, "seed": 9},
                    "request_controls_mismatch": ["temperature"],
                }
            )

    app = CayuApp(enable_logging=False)
    app.register_provider(Provider([]), default=True)
    app.register_agent(AgentSpec(name="agent", model="fake"))
    try:
        events = [
            event
            async for event in app.run(
                RunRequest(
                    agent_name="agent",
                    session_id="controls-spoofed-flag",
                    messages=[Message.text("user", "Hi.")],
                    max_steps=1,
                )
            )
        ]
        completed = next(event for event in events if event.type is EventType.MODEL_COMPLETED)
        assert completed.payload["echoed_request_controls"] == {"max_output_tokens": 50}
        assert "request_controls_mismatch" not in completed.payload
    finally:
        await app.aclose()


def test_calls_recorded_before_controls_existed_still_read() -> None:
    call = {
        "sequence": 4,
        "timestamp": "2026-10-31T00:00:00+00:00",
        "provider": "openai",
        "requested_model": "gpt-5.6",
        "resolved_model": "gpt-5.6",
        "input_tokens": 1,
        "output_tokens": 1,
        "total_tokens": 2,
        "reasoning_tokens": None,
        "cache_read_tokens": None,
        "cache_write_tokens": None,
        "cached_input_tokens": None,
        "uncached_input_tokens": None,
        "transcript_cursor": 1,
        "pricing_state": "unknown",
        "ledger": [],
    }
    validated = ApiSessionModelCall.model_validate(call)
    assert validated.request_controls is None
    output = io.StringIO()
    with redirect_stdout(output):
        _render_usage(
            "table", {"schema_version": "9", "aggregate": {}, "price_book": None}, [call], []
        )
    assert "Request controls" not in output.getvalue()


# --- review follow-ups: compaction calls, unknown sent controls, hostile echoes ---------


def test_hostile_or_oversized_echoes_are_dropped_not_raised() -> None:
    assert openai_responses_echoed_request_controls(
        {"temperature": 10**400, "top_p": -(10**400), "max_output_tokens": 10}
    ) == {"max_output_tokens": 10}
    # Finite numbers that can't be stored durably are dropped, not failed later.
    assert openai_responses_echoed_request_controls(
        {"temperature": 1e20, "top_p": 2**63 - 1, "max_output_tokens": 10}
    ) == {"max_output_tokens": 10}
    with pytest.raises(ValueError):
        ModelRequestControls(temperature=1e20)
    assert (
        openai_responses_echoed_request_controls(
            {"reasoning": {"effort": "\x1b]52;c;YXR0YWNrZXI=\x07"}}
        )
        is None
    )
    with pytest.raises(ValueError):
        ModelRequestControls(reasoning_effort="high\x1b[2J")


def test_usage_table_escapes_control_characters() -> None:
    from cayu.cli.session import _request_controls_text

    assert _request_controls_text({"reasoning_effort": "a\x1bb"}, unset=None) == (
        "reasoning_effort=a\\x1bb"
    )


def test_bedrock_reports_effort_sent_through_additional_fields() -> None:
    provider = BedrockProvider(client=object(), max_tokens=1000)

    assert provider.request_controls(
        _request(
            "anthropic.claude-opus-4-8",
            bedrock={
                "additionalModelRequestFields": {
                    "thinking": {"type": "adaptive"},
                    "output_config": {"effort": "high"},
                }
            },
        )
    ) == ModelRequestControls(reasoning_effort="high", max_output_tokens=1000)


def test_reader_does_not_report_a_match_without_sent_controls() -> None:
    from cayu.events import Event
    from cayu.sessions._call_inspection import model_call_usage
    from cayu.sessions.records import EventRecord

    completed = Event(
        type=EventType.MODEL_COMPLETED,
        session_id="s",
        payload={
            "model_attempt_id": "matt_x",
            "echoed_request_controls": {"max_output_tokens": None},
        },
    )
    (call,), _ = model_call_usage([EventRecord(sequence=2, event=completed)])
    assert call["request_controls"] is None
    assert call["echoed_request_controls"] == {"max_output_tokens": None}
    assert call["request_controls_mismatch"] is None


def test_compaction_completion_drops_a_provider_supplied_mismatch_flag() -> None:
    import cayu.context.base as context_base

    payload = context_base._compaction_model_completed_payload(
        completed_payload={
            "model": "summary-model",
            "echoed_request_controls": {"max_output_tokens": None, "seed": 3},
            "request_controls_mismatch": ["temperature"],
        },
        provider_name="openai",
        fallback_model="summary-model",
        compactor="ModelCompactor",
    )
    assert payload["echoed_request_controls"] == {"max_output_tokens": None}
    assert "request_controls_mismatch" not in payload


@pytest.mark.anyio
async def test_automatic_compaction_records_its_controls_and_a_dropped_cap() -> None:
    from cayu.context.base import CheckpointCompactionContextPolicy, ModelCompactor

    class CompactorProvider(ScriptedModelProvider):
        name = "compactor"

        def request_controls(self, request):
            return ModelRequestControls(max_output_tokens=4096)

        async def stream(self, request):
            yield ModelStreamEvent.text_delta("summary")
            yield ModelStreamEvent.completed(
                {
                    "model": "summary-model",
                    "finish_reason": "stop",
                    "echoed_request_controls": {"max_output_tokens": None},
                    "request_controls_mismatch": ["temperature"],
                }
            )

    class AgentProvider(ScriptedModelProvider):
        async def stream(self, request):
            yield ModelStreamEvent.text_delta("done")
            yield ModelStreamEvent.completed({"finish_reason": "stop"})

    app = CayuApp(enable_logging=False)
    app.register_provider(AgentProvider([]), default=True)
    app.register_agent(
        AgentSpec(name="assistant", model="fake-model"),
        context_policy=CheckpointCompactionContextPolicy(
            compactor=ModelCompactor(provider=CompactorProvider([]), model="summary-model"),
            max_user_turns=1,
            compact_after_messages=2,
        ),
    )
    try:
        events = [
            event
            async for event in app.run(
                RunRequest(
                    agent_name="assistant",
                    session_id="controls-compaction",
                    messages=[
                        Message.text("user", "old"),
                        Message.text("assistant", "old answer"),
                        Message.text("user", "current"),
                    ],
                    max_steps=1,
                )
            )
        ]
        assert events[-1].type is EventType.SESSION_COMPLETED
        records = await app.session_store.query_events(
            EventQuery(
                session_id="controls-compaction",
                event_types=(EventType.MODEL_STARTED, EventType.MODEL_COMPLETED),
            )
        )
        compaction = [
            record.event
            for record in records
            if record.event.payload.get("purpose") == "context_compaction"
        ]
        started = next(event for event in compaction if event.type is EventType.MODEL_STARTED)
        completed = next(event for event in compaction if event.type is EventType.MODEL_COMPLETED)
        assert started.payload["request_controls"] == {"max_output_tokens": 4096}
        assert completed.payload["echoed_request_controls"] == {"max_output_tokens": None}
        assert "request_controls_mismatch" not in completed.payload

        usage = await inspect_model_calls(
            app.session_store, "controls-compaction", InspectionWindow(), offset=0, limit=10
        )
        rows = [call for call in usage["calls"] if call["echoed_request_controls"] is not None]
        assert len(rows) == 1
        assert rows[0]["request_controls"] == {"max_output_tokens": 4096}
        assert rows[0]["request_controls_mismatch"] == ["max_output_tokens"]
    finally:
        await app.aclose()
