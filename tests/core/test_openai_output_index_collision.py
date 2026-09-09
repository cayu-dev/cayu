"""Synthetic index collisions; these fixtures are not historical wire evidence."""

import json

import pytest
from tests.core.test_openai_function_ordering import added, created, parse, run_sse

from cayu import EventType
from cayu.providers._openai_protocol import protocol_exception_fields
from cayu.providers.openai import OpenAIProtocolError


def collision():
    return [
        created(),
        {
            "type": "response.output_item.done",
            "output_index": 7,
            "item": {
                "type": "message",
                "id": "private-message-id",
                "role": "assistant",
                "status": "completed",
                "content": [{"type": "output_text", "text": "private-body", "annotations": []}],
            },
        },
        added(7),
    ]


@pytest.mark.anyio
async def test_completed_message_index_cannot_be_reused_by_function():
    with pytest.raises(OpenAIProtocolError) as caught:
        await parse(collision(), [])
    assert caught.value.reason_code == "function_call_output_index_type_mismatch"
    fields = protocol_exception_fields(caught.value)
    rows = json.loads(fields["provider_protocol_stream_trace"])
    assert rows[-1][2:5] == [7, "completed", "differs"]
    types = json.loads(fields["provider_protocol_stream_item_types"])
    assert types[-1] == [3, "function_call", "message"]
    assert len(rows) <= 16
    for secret in ("private-message-id", "private-body", "fc_7", "resp_safe"):
        assert secret not in json.dumps(fields)


@pytest.mark.anyio
async def test_collision_retries_preserve_bounded_durable_diagnostics(tmp_path):
    events, durable, executed = await run_sse(tmp_path, [collision(), collision()])
    assert not executed
    errors = [e.payload for e in durable if e.type == EventType.MODEL_ERROR]
    assert len(errors) == 2
    for error in errors:
        assert error["provider_protocol_reason"] == "function_call_output_index_type_mismatch"
        assert json.loads(error["provider_protocol_stream_item_types"])[-1] == [
            3,
            "function_call",
            "message",
        ]
    assert events[-1].type == EventType.SESSION_FAILED


@pytest.mark.anyio
@pytest.mark.parametrize("boundary", ["upstream", "intermediary"])
async def test_loopback_capture_locates_synthetic_first_divergence(boundary):
    import asyncio

    import httpx

    raw = collision()
    if boundary == "intermediary":
        raw[-1]["output_index"] = 8
    body = b"".join(b"data: " + json.dumps(e).encode() + b"\n\n" for e in raw)
    finished = asyncio.Event()

    async def serve(reader, writer):
        try:
            await reader.readuntil(b"\r\n\r\n")
            writer.write(
                b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nContent-Length: "
                + str(len(body)).encode()
                + b"\r\n\r\n"
                + body
            )
            await writer.drain()
        finally:
            writer.close()
            await writer.wait_closed()
            finished.set()

    # Correlate identities with local aliases, never persist raw provider IDs.
    aliases = {}

    def alias(value):
        if value is None:
            return "missing"
        return aliases.setdefault(value, f"identity-{len(aliases) + 1}")

    def structural(event):
        item = event.get("item", {})
        return [
            event["type"],
            event.get("output_index", -1),
            item.get("type", "missing"),
            alias(item.get("id")),
            alias(event.get("response", {}).get("id")),
        ]

    server = await asyncio.start_server(serve, "127.0.0.1", 0)
    upstream, intermediary = [], []
    try:

        async def source():
            async with (
                httpx.AsyncClient(trust_env=False) as client,
                client.stream(
                    "GET", f"http://127.0.0.1:{server.sockets[0].getsockname()[1]}/responses"
                ) as response,
            ):
                async for line in response.aiter_lines():
                    if not line.startswith("data: "):
                        continue
                    event = json.loads(line[6:])
                    upstream.append(structural(event))
                    if (
                        boundary == "intermediary"
                        and event.get("item", {}).get("type") == "function_call"
                    ):
                        event["output_index"] = 7
                    intermediary.append(structural(event))
                    yield event

        from cayu.providers.openai import openai_stream_events

        with pytest.raises(OpenAIProtocolError, match="changed item type"):
            async for _ in openai_stream_events(source()):
                pass
        assert len(upstream) == len(intermediary) == 3
        assert upstream[:2] == intermediary[:2]
        assert upstream[-1][1] == (8 if boundary == "intermediary" else 7)
        assert intermediary[-1][1] == 7
        assert upstream[-1][2:] == intermediary[-1][2:]
        assert "private" not in json.dumps([upstream, intermediary])
        await asyncio.wait_for(finished.wait(), 2)
    finally:
        server.close()
        await server.wait_closed()


@pytest.mark.parametrize(
    "types",
    [
        ((1, "private-type", "message"),),
        ((1, "function_call", "private-type"),),
        ((2, "function_call", "message"),),
        ((1, "function_call", "message"),) * 17,
    ],
)
def test_collision_diagnostic_revalidates_item_types(types):
    from cayu.providers._openai_search_trace import (
        SearchStreamDiagnostic,
        search_stream_diagnostic_fields,
    )

    trace = SearchStreamDiagnostic(
        ((1, "response.output_item.added", 7, "completed", "differs", "missing"),),
        False,
        types,
    )
    assert search_stream_diagnostic_fields(trace) == {}
