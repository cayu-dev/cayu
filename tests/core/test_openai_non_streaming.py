"""Native final-response transport controls; all responses are synthetic."""

import asyncio
import json
from copy import deepcopy

import httpx
import pytest

from cayu import (
    AgentSpec,
    CayuApp,
    EventType,
    Message,
    OpenAIProvider,
    OpenAIWebSearch,
    ProviderStatePart,
    RetryPolicy,
    RunRequest,
    SQLiteSessionStore,
    StructuredOutputSpec,
    Tool,
    ToolResult,
    ToolSpec,
)
from cayu.providers import HttpxOpenAITransport, ModelRequest
from cayu.providers.base import ModelStreamDeadlineError
from cayu.providers.deadlines import ProviderStreamDeadlines


def reasoning(identity):
    return {
        "type": "reasoning",
        "id": identity,
        "summary": [],
        "encrypted_content": f"synthetic-{identity}",
    }


def response(*items, identity="resp_synthetic"):
    return {
        "id": identity,
        "model": "gpt-5.6",
        "status": "completed",
        "output": list(items),
        "usage": {"input_tokens": 11, "output_tokens": 7, "total_tokens": 18},
    }


def message(text="Done."):
    return {
        "type": "message",
        "id": "msg_synthetic",
        "role": "assistant",
        "status": "completed",
        "content": [{"type": "output_text", "text": text, "annotations": []}],
    }


def function():
    return {
        "type": "function_call",
        "id": "fc_synthetic",
        "call_id": "call_synthetic",
        "name": "echo",
        "arguments": '{"text":"safe"}',
        "status": "completed",
    }


async def run_responses(tmp_path, responses, *, structured_output=None):
    requests, executed = [], []

    class Echo(Tool):
        spec = ToolSpec(
            name="echo",
            description="Synthetic echo",
            input_schema={"type": "object", "properties": {"text": {"type": "string"}}},
        )

        async def run(self, ctx, args):
            executed.append(args)
            return ToolResult(content="ok")

    async def handler(request):
        assert len(requests) < len(responses), "unexpected provider dispatch"
        body = json.loads(request.content)
        assert not body.get("stream", False)
        requests.append(body)
        result = responses[len(requests) - 1]
        if isinstance(result, httpx.Response):
            return result
        return httpx.Response(200, json=result)

    transport = HttpxOpenAITransport()
    store = SQLiteSessionStore(tmp_path / "non-streaming.sqlite3")
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        transport._client._client = client
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_provider(
            OpenAIProvider(api_key="synthetic-key", transport=transport, streaming=False),
            default=True,
        )
        app.register_agent(
            AgentSpec(name="assistant", model="gpt-5.6"),
            tools=[Echo()],
            hosted_tools=[OpenAIWebSearch()],
        )
        events = [
            event
            async for event in app.run(
                RunRequest(
                    agent_name="assistant",
                    session_id="non-streaming",
                    messages=[Message.text("user", "go")],
                    structured_output=structured_output,
                    retry_policy=RetryPolicy(
                        max_attempts=5, max_unknown_attempts=2, initial_delay_s=0
                    ),
                )
            )
        ]
    durable = await store.load_events("non-streaming")
    assert len(requests) == len(responses)
    return events, durable, requests, executed


@pytest.mark.anyio
async def test_final_response_preserves_reasoning_search_usage_and_exactly_one_tool(tmp_path):
    search = {"type": "web_search_call", "id": "ws_synthetic", "status": "completed"}
    events, durable, requests, executed = await run_responses(
        tmp_path,
        [
            response(reasoning("rs_a"), search, reasoning("rs_b"), function()),
            response(message(), identity="resp_final"),
        ],
    )
    assert events[-1].type == EventType.SESSION_COMPLETED
    assert executed == [{"text": "safe"}]
    assert len([e for e in durable if e.type == EventType.TOOL_CALL_STARTED]) == 1
    assert not [e for e in durable if e.type == EventType.MODEL_ERROR]
    completed = [e.payload for e in durable if e.type == EventType.MODEL_COMPLETED]
    assert len(completed) == 2
    assert all(e["usage_metrics"]["total_tokens"] == 18 for e in completed)
    hosted = [e.payload for e in durable if e.type == EventType.MODEL_HOSTED_TOOL_CALL]
    assert len(hosted) == 1 and hosted[0]["status"] == "completed"
    replay = requests[1]["input"]
    assert [i["id"] for i in replay if i.get("type") == "reasoning"] == ["rs_a", "rs_b"]
    assert len([i for i in replay if i.get("type") == "function_call_output"]) == 1


@pytest.mark.anyio
@pytest.mark.parametrize(
    "malformation",
    [
        "null_output",
        "invalid_tail",
        "bad_arguments",
        "incomplete_function",
        "duplicate_call",
        "duplicate_item",
        "nonterminal_response",
    ],
)
async def test_invalid_final_response_cannot_partially_dispatch_tools(tmp_path, malformation):
    raw = response(function())
    if malformation == "null_output":
        raw["output"] = None
    elif malformation == "invalid_tail":
        raw["output"].append("not-an-item")
    elif malformation == "bad_arguments":
        raw["output"][0]["arguments"] = "not-json"
    elif malformation == "incomplete_function":
        raw["output"][0]["status"] = "in_progress"
    elif malformation == "duplicate_call":
        raw["output"].append({**function(), "id": "fc_other"})
    elif malformation == "duplicate_item":
        raw["output"].append({**function(), "call_id": "call_other"})
    else:
        raw["status"] = "in_progress"
    events, durable, _requests, executed = await run_responses(tmp_path, [raw, deepcopy(raw)])
    assert events[-1].type == EventType.SESSION_FAILED
    assert executed == []
    assert not [e for e in durable if e.type == EventType.MODEL_COMPLETED]
    errors = [e.payload for e in durable if e.type == EventType.MODEL_ERROR]
    assert len(errors) == 2
    assert errors[-1]["effective_max_attempts"] == 2
    assert errors[-1]["retry_disposition"] == "unknown_provider_attempt_cap"


@pytest.mark.anyio
@pytest.mark.parametrize("status", ["incomplete", "failed"])
@pytest.mark.parametrize("call_status", ["in_progress", "incomplete"])
async def test_interrupted_final_response_cannot_dispatch_unfinished_function(
    tmp_path, status, call_status
):
    raw = response({**function(), "status": call_status})
    raw["status"] = status
    if status == "incomplete":
        raw["incomplete_details"] = {"reason": "max_output_tokens"}
    events, durable, _requests, executed = await run_responses(tmp_path, [raw, deepcopy(raw)])
    assert events[-1].type == EventType.SESSION_FAILED
    assert executed == []
    assert not [e for e in durable if e.type == EventType.MODEL_COMPLETED]


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("code", "param", "recover"),
    [
        (None, "previous_response_id", True),
        ("previous_response_not_found", None, True),
        ("previous_response_not_found", "model", False),
        (None, "model", False),
    ],
)
async def test_final_response_stale_chain_preserves_typed_recovery_identity(code, param, recover):
    requests = []

    async def handler(request):
        requests.append(json.loads(request.content))
        if len(requests) == 1:
            return httpx.Response(
                404,
                json={
                    "error": {
                        "type": "invalid_request_error",
                        "code": code,
                        "param": param,
                        "message": "synthetic previous_response_id error",
                    }
                },
            )
        return httpx.Response(200, json=response(message()))

    transport = HttpxOpenAITransport()
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        transport._client._client = client
        provider = OpenAIProvider(
            api_key="synthetic-key", transport=transport, streaming=False, reasoning_state="server"
        )
        request = ModelRequest(
            model="gpt-5.6",
            messages=[
                Message.text("user", "first"),
                Message(
                    role="assistant",
                    content=[
                        ProviderStatePart(
                            provider="openai",
                            state={
                                "type": "response_ref",
                                "id": "resp_previous",
                                "targeted_tool_marker_id": None,
                            },
                        )
                    ],
                ),
                Message.text("user", "second"),
            ],
        )
        events = [event async for event in provider.stream(request)]
    assert requests[0]["previous_response_id"] == "resp_previous"
    assert len(requests) == (2 if recover else 1)
    assert events[-1].type.value == ("completed" if recover else "error")
    if recover:
        assert "previous_response_id" not in requests[1]
        assert requests[1]["input"] == [
            {"role": "user", "content": [{"type": "input_text", "text": "first"}]},
            {"role": "user", "content": [{"type": "input_text", "text": "second"}]},
        ]
        assert requests[1]["store"] is True


@pytest.mark.anyio
@pytest.mark.parametrize("body", ["not-json", "[]"])
async def test_final_response_transport_protocol_errors_keep_unknown_retry_cap(tmp_path, body):
    events, durable, _requests, executed = await run_responses(
        tmp_path, [httpx.Response(200, text=body), httpx.Response(200, text=body)]
    )
    assert events[-1].type == EventType.SESSION_FAILED
    assert executed == []
    errors = [e.payload for e in durable if e.type == EventType.MODEL_ERROR]
    assert len(errors) == 2
    assert errors[-1]["provider_error_type"] == "protocol_error"
    assert errors[-1]["retry_disposition"] == "unknown_provider_attempt_cap"


@pytest.mark.anyio
async def test_native_http_retry_keeps_attempt_identity_and_final_usage(tmp_path):
    events, durable, requests, _executed = await run_responses(
        tmp_path,
        [
            httpx.Response(500, json={"error": {"type": "server_error", "message": "synthetic"}}),
            response(message()),
        ],
    )
    assert events[-1].type == EventType.SESSION_COMPLETED
    assert len(requests) == 2
    errors = [e.payload for e in durable if e.type == EventType.MODEL_ERROR]
    assert len(errors) == 1 and errors[0]["status_code"] == 500
    attempts = [e.payload for e in durable if e.type == EventType.MODEL_STARTED]
    assert len({e["model_attempt_id"] for e in attempts}) == 2
    assert len({e["model_step_id"] for e in attempts}) == 1


@pytest.mark.anyio
async def test_final_response_uses_native_structured_output_validation(tmp_path):
    spec = StructuredOutputSpec(
        name="answer",
        json_schema={
            "type": "object",
            "properties": {"answer": {"type": "string"}},
            "required": ["answer"],
            "additionalProperties": False,
        },
        strategy="native",
        max_retries=0,
    )
    events, durable, requests, executed = await run_responses(
        tmp_path, [response(message('{"answer":"ready"}'))], structured_output=spec
    )
    assert events[-1].type == EventType.SESSION_COMPLETED
    assert executed == []
    assert requests[0]["text"]["format"]["type"] == "json_schema"
    assert any(e.type == EventType.STRUCTURED_OUTPUT_VALIDATED for e in durable)


@pytest.mark.anyio
@pytest.mark.parametrize("deadline", ["semantic_progress_timeout_s", "absolute_stream_timeout_s"])
async def test_final_response_wait_keeps_runtime_deadlines_and_joins_cancellation(deadline):
    started, stopped = asyncio.Event(), asyncio.Event()

    class WaitingTransport:
        async def create_response(self, **kwargs):
            started.set()
            try:
                await asyncio.Event().wait()
            finally:
                stopped.set()

    provider = OpenAIProvider(
        api_key="synthetic-key",
        transport=WaitingTransport(),
        streaming=False,
        stream_deadlines=ProviderStreamDeadlines(**{deadline: 0.05}),
    )
    request = ModelRequest(model="gpt-5.6", messages=[Message.text("user", "go")])
    with pytest.raises(ModelStreamDeadlineError) as caught:
        _events = [event async for event in provider.runtime_stream(request)]
    assert started.is_set() and stopped.is_set()
    assert caught.value.deadline_evidence.deadline_kind.value == (
        "semantic_idle" if deadline.startswith("semantic") else "absolute"
    )


def test_response_transport_mode_has_distinct_request_identity():
    from cayu.providers.openai import _execution_profile_material

    request = ModelRequest(model="gpt-5.6", messages=[Message.text("user", "go")])
    streaming = OpenAIProvider(api_key="synthetic-key")
    final = OpenAIProvider(api_key="synthetic-key", streaming=False)
    assert streaming.request_fingerprint_options(request) == {}
    assert final.request_fingerprint_options(request) == {"openai": {"stream": False}}
    assert final.request_footprint_options(request) == {"openai": {"stream": False}}
    assert request.options == {}
    current_material = _execution_profile_material(streaming)
    final_material = _execution_profile_material(final)
    assert current_material is not None and final_material is not None
    assert "streaming" not in current_material
    assert final_material == {**current_material, "streaming": False}


@pytest.mark.parametrize("invalid", [None, 0, 1, "false"])
def test_response_transport_mode_requires_boolean(invalid):
    with pytest.raises(TypeError, match="streaming must be a bool"):
        OpenAIProvider(api_key="synthetic-key", streaming=invalid)


def test_background_operations_cannot_silently_ignore_final_response_mode():
    with pytest.raises(ValueError, match="background operations require streaming=True"):
        OpenAIProvider(api_key="synthetic-key", streaming=False, background=True)
