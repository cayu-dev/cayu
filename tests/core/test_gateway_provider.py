from __future__ import annotations

import asyncio
import json

import httpx
import pytest

from cayu import (
    AgentSpec,
    CayuApp,
    EventType,
    GatewayProvider,
    HttpxGatewayTransport,
    Message,
    RunRequest,
)
from cayu.context.structured_output import StructuredOutputSpec
from cayu.providers.base import ModelProviderError, ModelRequest, ModelStreamEventType
from cayu.providers.retry_policy import RetryPolicy
from cayu.storage.sqlite import SQLiteSessionStore
from cayu.tools.base import Tool, ToolEffect, ToolResult, ToolSpec


class EchoTool(Tool):
    spec = ToolSpec(
        name="echo",
        description="Return a deterministic answer.",
        effect=ToolEffect.NONE,
        input_schema={"type": "object", "properties": {}, "additionalProperties": False},
    )

    async def run(self, ctx, args):
        return ToolResult(content="tool answer")


def chunk(delta=None, finish=None, **extra):
    return {
        "id": "req_example",
        "object": "chat.completion.chunk",
        "model": "example/model",
        "choices": [{"index": 0, "delta": delta or {}, "finish_reason": finish}],
    } | extra


def wire(*chunks):
    return (
        b"".join(b"data: " + json.dumps(c).encode() + b"\n\n" for c in chunks) + b"data: [DONE]\n\n"
    )


def provider_for(handler):
    transport = HttpxGatewayTransport()
    transport._client._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return GatewayProvider(
        base_url="https://gateway.example/v1", api_key="private-key-canary", transport=transport
    )


def response(body, *, status=200, content_type="application/json", **headers):
    return httpx.Response(
        status, headers={"content-type": content_type, **headers}, stream=httpx.ByteStream(body)
    )


@pytest.mark.anyio
@pytest.mark.parametrize(
    "cost,status", [(None, "pending"), ("0.000000000", "reported"), ("0.001234567", "reported")]
)
async def test_gateway_completion_observations_survive_sqlite_reopen(tmp_path, cost, status):
    calls = []
    usage = {
        "prompt_tokens": 10,
        "completion_tokens": 3,
        "total_tokens": 13,
        "cost": cost,
        "cost_currency": "USD",
        "cost_status": status,
    }

    def handle(request):
        calls.append(request)
        return response(
            wire(chunk({"content": "Hello"}), chunk(finish="stop"), chunk(choices=[], usage=usage)),
            content_type="text/event-stream",
        )

    provider = provider_for(handle)
    path = tmp_path / "session.sqlite3"
    store = SQLiteSessionStore(path)
    app = CayuApp(session_store=store)
    app.register_provider(provider, default=True)
    app.register_agent(AgentSpec(name="assistant", model="example/model"))
    try:
        events = [
            e
            async for e in app.run(
                RunRequest(
                    agent_name="assistant",
                    session_id="gateway-usage",
                    messages=[Message.text("user", "Hello")],
                )
            )
        ]
        assert events[-1].type is EventType.SESSION_COMPLETED
    finally:
        await store.close()
        await provider.aclose()
    reopened = SQLiteSessionStore(path)
    try:
        events = await reopened.load_events("gateway-usage")
        completed = next(e for e in events if e.type is EventType.MODEL_COMPLETED)
        assert completed.payload["id"] == "req_example"
        assert completed.payload["usage"] == usage
        from datetime import timedelta

        from cayu.sessions.usage import UsageRollupQuery

        rollup = await reopened.aggregate_usage(
            UsageRollupQuery(
                start_at=completed.timestamp - timedelta(seconds=1),
                end_at=completed.timestamp + timedelta(seconds=1),
            )
        )
        observed = rollup.reported_costs.records[0]
        assert observed.request_id == "req_example"
        assert observed.cost == cost and observed.status == status
    finally:
        await reopened.close()
    assert len(calls) == 1
    payload = json.loads(calls[0].content)
    assert payload["stream_options"] == {"include_usage": True}
    assert calls[0].headers["authorization"] == "Bearer private-key-canary"
    assert calls[0].url.path == "/v1/chat/completions"


@pytest.mark.anyio
async def test_tools_reasoning_and_native_schema_use_normal_chat_protocol():
    calls = []

    def handle(request):
        calls.append(json.loads(request.content))
        return response(
            wire(
                chunk(
                    {
                        "reasoning_content": "Think",
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "call_a",
                                "type": "function",
                                "function": {"name": "echo", "arguments": '{"x":1}'},
                            }
                        ],
                    }
                ),
                chunk(finish="tool_calls"),
                chunk(choices=[], usage={"cost": None, "cost_status": "pending"}),
            ),
            content_type="text/event-stream",
        )

    provider = provider_for(handle)
    schema = {
        "type": "object",
        "properties": {"x": {"type": "integer"}},
        "required": ["x"],
        "additionalProperties": False,
    }
    request = ModelRequest(
        model="example/model",
        messages=[Message.text("user", "Go")],
        options={
            "structured_output": {"strategy": "native", "schema": schema},
            "cayu_gateway": {"max_completion_tokens": 128},
        },
    )
    try:
        provider.preflight_native_structured_output_schema(schema)
        events = [e async for e in provider.stream(request)]
        assert any(e.type is ModelStreamEventType.THINKING for e in events)
        assert any(e.type is ModelStreamEventType.TOOL_CALL for e in events)
        assert events[-1].type is ModelStreamEventType.COMPLETED
        assert calls[0]["response_format"]["json_schema"]["schema"] == schema
        assert calls[0]["max_completion_tokens"] == 128
        assert (
            provider.request_fingerprint_options(request)["cayu_gateway"]["response_format"]
            == calls[0]["response_format"]
        )
        assert "response_format" not in request.options["cayu_gateway"]
    finally:
        await provider.aclose()


@pytest.mark.anyio
async def test_gateway_failure_does_not_retry_a_new_billable_request(tmp_path):
    calls = []

    def handle(request):
        calls.append(request)
        return response(b'{"error":{"code":"gateway_unavailable"}}', status=503)

    provider = provider_for(handle)
    store = SQLiteSessionStore(tmp_path / "failure.sqlite3")
    app = CayuApp(session_store=store)
    app.register_provider(provider, default=True)
    app.register_agent(AgentSpec(name="assistant", model="example/model"))
    try:
        events = [
            e
            async for e in app.run(
                RunRequest(
                    agent_name="assistant",
                    messages=[Message.text("user", "Hello")],
                    retry_policy=RetryPolicy(max_attempts=3, initial_delay_s=0.0),
                )
            )
        ]
        assert len(calls) == 1
        assert not any(e.type is EventType.MODEL_RETRY for e in events)
        assert events[-1].type is EventType.SESSION_FAILED
    finally:
        await store.close()
        await provider.aclose()


@pytest.mark.anyio
@pytest.mark.parametrize("mode", ["tool", "native"])
async def test_gateway_public_runtime_tool_and_native_output(mode):
    calls = []

    def handle(request):
        packet = json.loads(request.content)
        calls.append(packet)
        if mode == "tool" and len(calls) == 1:
            events = [
                chunk(
                    {
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "call_echo",
                                "type": "function",
                                "function": {"name": "echo", "arguments": "{}"},
                            }
                        ]
                    }
                ),
                chunk(finish="tool_calls"),
            ]
        else:
            events = [chunk({"content": '{"answer":"OK"}'}), chunk(finish="stop")]
        return response(wire(*events), content_type="text/event-stream")

    provider = provider_for(handle)
    app = CayuApp(enable_logging=False)
    app.register_provider(provider, default=True)
    app.register_agent(
        AgentSpec(name="assistant", model="example/model"),
        tools=[EchoTool()] if mode == "tool" else [],
    )
    schema = {
        "type": "object",
        "properties": {"answer": {"type": "string"}},
        "required": ["answer"],
        "additionalProperties": False,
    }
    try:
        events = [
            e
            async for e in app.run(
                RunRequest(
                    agent_name="assistant",
                    messages=[Message.text("user", "Answer")],
                    structured_output=StructuredOutputSpec(strategy="native", json_schema=schema)
                    if mode == "native"
                    else None,
                )
            )
        ]
        assert events[-1].type is EventType.SESSION_COMPLETED
        if mode == "tool":
            assert len(calls) == 2
            assert any(e.type is EventType.TOOL_CALL_COMPLETED for e in events)
            assert any(
                m["role"] == "tool" and "tool answer" in m["content"] for m in calls[1]["messages"]
            )
        else:
            assert len(calls) == 1
            assert calls[0]["response_format"]["json_schema"]["schema"] == schema
            assert any(e.type is EventType.STRUCTURED_OUTPUT_VALIDATED for e in events)
    finally:
        await provider.aclose()


@pytest.mark.anyio
async def test_lookup_is_authenticated_get_and_unknown_cost_remains_unknown():
    calls = []

    def handle(request):
        calls.append(request)
        data = {
            "id": "req_example",
            "status": "unknown",
            "usage": {"cost": None, "cost_status": "pending"},
        }
        if request.url.path.endswith("models"):
            data = [{"id": "example/model"}]
        return response(json.dumps({"data": data}).encode())

    provider = provider_for(handle)
    try:
        assert (await provider.get_generation("req_example"))["usage"]["cost"] is None
        assert await provider.get_models() == [{"id": "example/model"}]
    finally:
        await provider.aclose()
    assert all(
        c.method == "GET" and c.headers["authorization"] == "Bearer private-key-canary"
        for c in calls
    )
    assert calls[0].url.params["id"] == "req_example"


@pytest.mark.anyio
@pytest.mark.parametrize(
    "kind", ["redirect", "oversized", "invalid", "wrong_id", "encoded", "exception"]
)
async def test_lookup_bounds_and_diagnostics(kind):
    calls = []

    def handle(request):
        calls.append(request)
        if kind == "exception":
            raise RuntimeError("private-key-canary")
        if kind == "redirect":
            return response(b"private-key-canary", status=302, location="https://foreign.example")
        if kind == "oversized":
            return response(b" " * 1_048_577)
        if kind == "encoded":
            return response(b"private-key-canary", **{"content-encoding": "gzip"})
        return response(b"{}" if kind == "invalid" else b'{"data":{"id":"other"}}')

    provider = provider_for(handle)
    try:
        with pytest.raises(ModelProviderError) as failure:
            await provider.get_generation("req_example")
        assert "private-key-canary" not in str(failure.value)
        assert len(calls) == 1
    finally:
        await provider.aclose()


def test_gateway_requires_https_and_explicit_endpoint():
    with pytest.raises(ValueError):
        GatewayProvider(api_key="test", base_url="http://gateway.example/v1")
    with pytest.raises(ValueError):
        GatewayProvider(api_key="test", base_url="https://gateway.example/v1?key=bad")


@pytest.mark.anyio
async def test_stream_error_preserves_uncertainty_without_completion_or_retry():
    def handle(request):
        return response(
            wire(
                chunk({"content": "Partial"}),
                {
                    "error": {
                        "code": "outcome_unknown",
                        "message": "Request outcome is unknown.",
                        "request_id": "req_example",
                        "retryable": False,
                    },
                },
            ),
            content_type="text/event-stream",
        )

    provider = provider_for(handle)
    try:
        events = [
            e
            async for e in provider.stream(
                ModelRequest(
                    model="example/model",
                    messages=[Message.text("user", "Go")],
                )
            )
        ]
        assert events[-1].type is ModelStreamEventType.ERROR
        # Reuse the normal provider privacy boundary, which omits untrusted
        # error request IDs. The service still owns the uncertain request.
        assert events[-1].payload["request_id"] == "req_example"
        assert events[-1].payload["retryable"] is False
        assert not any(e.type is ModelStreamEventType.COMPLETED for e in events)
    finally:
        await provider.aclose()


@pytest.mark.anyio
@pytest.mark.parametrize("caller_cancels", [False, True])
async def test_lookup_deadline_and_cancellation_close_response(caller_cancels):
    entered = asyncio.Event()
    closed = asyncio.Event()

    class SlowBody(httpx.AsyncByteStream):
        async def __aiter__(self):
            entered.set()
            await asyncio.Event().wait()
            yield b"{}"

        async def aclose(self):
            closed.set()

    provider = provider_for(lambda _: httpx.Response(200, stream=SlowBody()))
    provider.timeout_s = 1.0 if caller_cancels else 0.01
    task = asyncio.create_task(provider.get_generation("req_example"))
    try:
        await asyncio.wait_for(entered.wait(), 1)
        if caller_cancels:
            task.cancel("private-key-canary")
            with pytest.raises(asyncio.CancelledError) as failure:
                await task
            assert "private-key-canary" not in str(failure.value)
            assert task.cancelled()
            assert task.cancelling() == 1
        else:
            with pytest.raises(ModelProviderError):
                await task
            assert not task.cancelled()
        assert closed.is_set()
    finally:
        await provider.aclose()
