"""A validated Responses terminal bounds tail reads, not cleanup ownership."""

from __future__ import annotations

import asyncio
import json
from typing import Any

import httpx
import pytest

from cayu import Message
from cayu.agents import AgentSpec
from cayu.applications import CayuApp
from cayu.events import EventType
from cayu.providers._credential_boundary import (
    ProviderStreamCleanupError,
    aclosing_provider_stream,
    provider_cancellation_failures,
)
from cayu.providers.base import (
    ModelFinishReason,
    ModelProviderError,
    ModelRequest,
    ModelStreamDeadlineError,
    ModelStreamEventType,
)
from cayu.providers.deadlines import ProviderStreamDeadlines
from cayu.providers.openai import (
    HttpxOpenAITransport,
    OpenAIProvider,
    _openai_background_stream_events,
    openai_stream_events,
)
from cayu.providers.openai_subscription import (
    OpenAISubscriptionCredentials,
    OpenAISubscriptionProvider,
)
from cayu.providers.operations import ProviderOperationState
from cayu.sessions.base import RunRequest
from cayu.storage.sqlite import SQLiteSessionStore
from cayu.tools.base import Tool, ToolResult, ToolSpec


def terminal(status: str = "completed") -> dict[str, Any]:
    response: dict[str, Any] = {
        "id": "resp_synthetic",
        "model": "synthetic",
        "status": status,
        "output": [],
        "usage": {"input_tokens": 2, "output_tokens": 1},
    }
    if status == "incomplete":
        response["incomplete_details"] = {"reason": "max_output_tokens"}
    return {"type": f"response.{status}", "response": response}


def frame(event):
    return b"data: " + json.dumps(event).encode() + b"\n\n"


class Body(httpx.AsyncByteStream):
    def __init__(self, chunks, *, stall=True, close_failure=False):
        self.chunks = chunks
        self.stall = stall
        self.close_failure = close_failure
        self.tail_read = False
        self.closed = False

    async def __aiter__(self):
        for chunk in self.chunks:
            yield chunk
        self.tail_read = True
        if self.stall:
            await asyncio.Event().wait()

    async def aclose(self):
        self.closed = True
        if self.close_failure:
            raise RuntimeError("synthetic close failure")


class StaticAuth:
    async def credentials(self):
        return OpenAISubscriptionCredentials(
            access_token="synthetic-access",
            refresh_token="synthetic-refresh",
            expires_at=2_000_000_000,
            account_id="synthetic-account",
        )


DEADLINES = ProviderStreamDeadlines(
    semantic_progress_timeout_s=0.15,
    transport_idle_timeout_s=1,
    protocol_idle_timeout_s=1,
    absolute_stream_timeout_s=2,
)
REQUEST = ModelRequest(model="synthetic", messages=[Message.text("user", "Synthetic probe")])


async def consume(body, *, subscription=False, direct=False, background=False):
    transport = HttpxOpenAITransport()
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda _: httpx.Response(200, stream=body))
    ) as client:
        transport._client._client = client
        provider = (
            OpenAISubscriptionProvider(
                auth=StaticAuth(), transport=transport, stream_deadlines=DEADLINES
            )
            if subscription
            else OpenAIProvider(
                api_key="synthetic", transport=transport, stream_deadlines=DEADLINES
            )
        )
        events = []
        failure = None
        try:
            async with asyncio.timeout(3):
                stream = provider.runtime_stream(REQUEST)
                if direct:
                    stream = openai_stream_events(
                        transport.stream_response_events(
                            url="https://synthetic.invalid/responses",
                            headers={},
                            payload={},
                            timeout_s=2,
                            transport_idle_timeout_s=1,
                            protocol_idle_timeout_s=1,
                            semantic_progress_timeout_s=0.15,
                            absolute_stream_timeout_s=2,
                        )
                    )
                if background:
                    stream = _openai_background_stream_events(
                        transport.reconnect_response_events(
                            url="https://synthetic.invalid/responses/resp_synthetic",
                            headers={},
                            starting_after=0,
                            timeout_s=2,
                            transport_idle_timeout_s=1,
                            protocol_idle_timeout_s=1,
                            semantic_progress_timeout_s=0.15,
                            absolute_stream_timeout_s=2,
                        ),
                        state=ProviderOperationState.model_validate(
                            {
                                "operation_id": "resp_synthetic",
                                "stream_protocol": "openai-responses-background-v1",
                                "recovery_metadata": {
                                    "cursor": 0,
                                    "opaque": {"sequence_number": 0},
                                },
                            }
                        ),
                        first=None,
                        reasoning_state="inline",
                    )
                async with aclosing_provider_stream(stream):
                    async for event in stream:
                        events.append(event)
        except Exception as error:
            failure = error
        finally:
            await provider.aclose()
        return events, failure


@pytest.mark.parametrize("subscription", [False, True])
@pytest.mark.parametrize("status", ["completed", "incomplete"])
@pytest.mark.parametrize("split", [False, True])
def test_validated_terminal_does_not_require_http_eof(subscription, status, split):
    data = frame(terminal(status))
    body = Body([data[:13], data[13:]] if split else [data])
    events, error = asyncio.run(consume(body, subscription=subscription))
    assert error is None
    completions = [e for e in events if e.type is ModelStreamEventType.COMPLETED]
    assert len(completions) == 1
    assert completions[0].payload["usage"]["input_tokens"] == 2
    assert completions[0].completion.finish_reason is (
        ModelFinishReason.STOP if status == "completed" else ModelFinishReason.LENGTH
    )
    assert body.closed
    assert body.tail_read


def test_direct_parser_acknowledges_its_own_transport_without_runtime_scope():
    body = Body([frame(terminal())])
    events, error = asyncio.run(consume(body, direct=True))
    assert error is None
    assert sum(e.type is ModelStreamEventType.COMPLETED for e in events) == 1
    assert body.closed and body.tail_read


def test_without_acknowledgment_original_idle_failure_reproduces(monkeypatch):
    monkeypatch.setattr("cayu.providers.openai._accept_sse_terminal", lambda _: None)
    body = Body([frame(terminal())])
    events, error = asyncio.run(consume(body))
    assert sum(e.type is ModelStreamEventType.COMPLETED for e in events) == 1
    assert isinstance(error, ModelStreamDeadlineError)
    assert error.error_payload_fields()["provider_last_progress_kind"] == "terminal"
    assert body.closed and body.tail_read


@pytest.mark.parametrize(
    "tail",
    [
        {"type": "response.output_text.delta", "delta": "forbidden"},
        terminal(),
        {"type": "response.failed", "response": {"error": {"code": "server_error"}}},
    ],
)
@pytest.mark.parametrize("separate_chunks", [False, True])
def test_already_buffered_trailing_events_still_fail(tail, separate_chunks):
    chunks = [frame(terminal()), frame(tail)]
    body = Body(chunks if separate_chunks else [b"".join(chunks)])
    events, error = asyncio.run(consume(body))
    assert sum(e.type is ModelStreamEventType.COMPLETED for e in events) == 1
    assert isinstance(error, ModelProviderError)
    assert "OpenAIProtocolError" in str(error)
    assert error.retryable is False
    assert body.closed and not body.tail_read


def test_heartbeats_do_not_extend_terminal_drain():
    class Heartbeats(Body):
        count = 0

        async def __aiter__(self):
            yield frame(terminal())
            while True:
                await asyncio.sleep(0.001)
                self.count += 1
                yield b": heartbeat\n\n"

    body = Heartbeats([])
    events, error = asyncio.run(consume(body))
    assert error is None
    assert sum(e.type is ModelStreamEventType.COMPLETED for e in events) == 1
    assert body.closed and body.count > 0


def test_real_tail_timeout_error_is_not_treated_as_drain_expiry():
    class Failure(Body):
        async def __aiter__(self):
            yield frame(terminal())
            raise TimeoutError("Synthetic unrelated tail failure")

    body = Failure([])
    events, error = asyncio.run(consume(body))
    assert isinstance(error, ModelProviderError)
    assert error.retryable is False
    assert sum(e.type is ModelStreamEventType.COMPLETED for e in events) == 1
    assert body.closed


@pytest.mark.parametrize("mode", ["api", "subscription", "direct", "background"])
@pytest.mark.parametrize("chain_cancellation", [False, True])
def test_tail_timeout_during_drain_cancellation_is_not_success(mode, chain_cancellation):
    class Failure(Body):
        async def __aiter__(self):
            yield frame({**terminal(), "sequence_number": 1})
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError as cancellation:
                raise TimeoutError("synthetic private tail failure") from (
                    cancellation if chain_cancellation else None
                )

    body = Failure([])
    events, error = asyncio.run(
        consume(
            body,
            subscription=mode == "subscription",
            direct=mode == "direct",
            background=mode == "background",
        )
    )
    assert error is not None
    if mode in {"api", "subscription"}:
        assert isinstance(error, ModelProviderError)
        assert error.retryable is False
        assert "synthetic private tail failure" not in str(error)
    else:
        assert isinstance(error, TimeoutError)
    assert sum(e.type is ModelStreamEventType.COMPLETED for e in events) == 1
    assert body.closed


@pytest.mark.parametrize("mode", ["api", "subscription", "direct", "background"])
@pytest.mark.parametrize("close_failure", [False, True])
def test_nested_tail_cleanup_outcome_survives_drain_expiry(mode, close_failure):
    class Source:
        started = False
        close_attempted = False

        def __aiter__(self):
            return self

        async def __anext__(self):
            if not self.started:
                self.started = True
                return frame({**terminal(), "sequence_number": 1})
            await asyncio.Event().wait()

        async def aclose(self):
            self.close_attempted = True
            if close_failure:
                raise RuntimeError("synthetic private inner close failure")

    source = Source()

    class Wrapped(Body):
        async def __aiter__(self):
            async with aclosing_provider_stream(source):
                async for chunk in source:
                    yield chunk

    body = Wrapped([])
    events, error = asyncio.run(
        consume(
            body,
            subscription=mode == "subscription",
            direct=mode == "direct",
            background=mode == "background",
        )
    )
    if close_failure:
        assert isinstance(error, ProviderStreamCleanupError)
        assert error.retryable is False
        assert error.error_type == "ProviderStreamCleanupError"
        assert "synthetic private inner close failure" not in str(error)
    else:
        assert error is None
    assert sum(e.type is ModelStreamEventType.COMPLETED for e in events) == 1
    assert source.close_attempted and body.closed


@pytest.mark.parametrize("subscription", [False, True])
@pytest.mark.parametrize("close_failure", [False, True])
def test_caller_cancellation_during_tail_preserves_cleanup_evidence(subscription, close_failure):
    async def run():
        reading = asyncio.Event()

        class Source:
            started = False
            close_attempted = False

            def __aiter__(self):
                return self

            async def __anext__(self):
                if not self.started:
                    self.started = True
                    return frame(terminal())
                reading.set()
                await asyncio.Event().wait()

            async def aclose(self):
                self.close_attempted = True
                if close_failure:
                    raise RuntimeError("synthetic private close failure")

        source = Source()

        class Wrapped(Body):
            async def __aiter__(self):
                async with aclosing_provider_stream(source):
                    async for chunk in source:
                        yield chunk

        body = Wrapped([])
        task = asyncio.create_task(consume(body, subscription=subscription))
        try:
            await asyncio.wait_for(reading.wait(), 1)
            task.cancel()
            with pytest.raises(asyncio.CancelledError) as raised:
                await asyncio.wait_for(task, 1)
            failures = provider_cancellation_failures(raised.value)
            assert bool(failures) is close_failure
            if close_failure:
                assert any(f["phase"] == "provider_stream_cleanup" for f in failures)
            assert source.close_attempted and body.closed
        finally:
            if not task.done():
                task.cancel()
            await asyncio.gather(task, return_exceptions=True)

    asyncio.run(run())


def test_noncooperative_tail_remains_owned_and_cannot_manufacture_success():
    async def run():
        release = asyncio.Event()
        settled = asyncio.Event()

        class Noncooperative(Body):
            async def __aiter__(self):
                try:
                    yield frame(terminal())
                    while not release.is_set():
                        try:
                            await release.wait()
                        except asyncio.CancelledError:
                            continue
                finally:
                    settled.set()

        body = Noncooperative([])
        task = asyncio.create_task(consume(body))
        try:
            events, error = await asyncio.wait_for(asyncio.shield(task), 1)
            assert sum(e.type is ModelStreamEventType.COMPLETED for e in events) == 1
            assert isinstance(error, ModelStreamDeadlineError)
            assert error.stream_cleanup_failed
            assert not settled.is_set()
        finally:
            release.set()
            await asyncio.wait_for(settled.wait(), 1)
            await asyncio.wait_for(task, 1)

    asyncio.run(run())


def test_incomplete_terminal_frame_does_not_authorize_stopping():
    body = Body([frame(terminal())[:-1]])
    events, error = asyncio.run(consume(body))
    assert not any(e.type is ModelStreamEventType.COMPLETED for e in events)
    # Preterminal errors remain error events, not successful empty completions.
    assert error is not None or any(e.type is ModelStreamEventType.ERROR for e in events)
    assert body.closed and body.tail_read


def test_invalid_completed_response_is_not_acknowledged():
    event = terminal()
    event["response"]["output"] = "invalid output"
    body = Body([frame(event)])
    events, error = asyncio.run(consume(body))
    assert not any(e.type is ModelStreamEventType.COMPLETED for e in events)
    assert error is not None or any(e.type is ModelStreamEventType.ERROR for e in events)
    assert body.closed


def test_wire_fields_cannot_forge_terminal_acknowledgment():
    body = Body(
        [
            frame(
                {
                    "type": "response.created",
                    "_terminal_accepted": True,
                    "response": {"id": "resp_synthetic", "status": "in_progress"},
                }
            )
        ]
    )
    events, error = asyncio.run(consume(body))
    assert not any(e.type is ModelStreamEventType.COMPLETED for e in events)
    assert error is not None or any(e.type is ModelStreamEventType.ERROR for e in events)
    assert body.closed and body.tail_read


@pytest.mark.parametrize("acknowledge", [False, True], ids=["prior-eof", "terminal"])
def test_cancellation_during_terminal_close_remains_authoritative(monkeypatch, acknowledge):
    if not acknowledge:
        monkeypatch.setattr("cayu.providers.openai._accept_sse_terminal", lambda _: None)

    async def run():
        closing = asyncio.Event()
        release = asyncio.Event()
        close_settled = asyncio.Event()
        close_cancelled = asyncio.Event()

        class HeldClose(Body):
            async def aclose(self):
                closing.set()
                try:
                    await release.wait()
                    self.closed = True
                except asyncio.CancelledError:
                    close_cancelled.set()
                    raise
                finally:
                    close_settled.set()

        body = HeldClose([frame(terminal())], stall=acknowledge)
        task = asyncio.create_task(consume(body))
        try:
            await asyncio.wait_for(closing.wait(), 1)
            task.cancel()
            release.set()
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(task, 1)
            await asyncio.wait_for(close_settled.wait(), 1)
            # An interrupted close is not evidence of successful closure. The
            # prior EOF path and terminal path both preserve caller cancellation.
            assert body.closed or close_cancelled.is_set()
            assert body.tail_read
        finally:
            release.set()
            if not task.done():
                task.cancel()
            await asyncio.gather(task, return_exceptions=True)

    asyncio.run(run())


@pytest.mark.parametrize("subscription", [False, True])
def test_close_failure_remains_visible_after_completion(subscription):
    body = Body([frame(terminal())], close_failure=True)
    events, error = asyncio.run(consume(body, subscription=subscription))
    assert sum(e.type is ModelStreamEventType.COMPLETED for e in events) == 1
    assert error is not None
    assert body.closed and body.tail_read


@pytest.mark.parametrize("status", ["completed", "incomplete", "failed", "cancelled", "expired"])
def test_background_reconnect_terminal_does_not_require_eof(status):
    event: dict[str, Any] = {**terminal(status), "sequence_number": 1}
    if status == "failed":
        event["response"]["error"] = {"code": "server_error", "message": "synthetic"}
    body = Body([frame(event)])
    events, error = asyncio.run(consume(body, background=True))
    assert error is None
    assert events[-1].type is (
        ModelStreamEventType.COMPLETED
        if status in {"completed", "incomplete"}
        else ModelStreamEventType.ERROR
    )
    assert body.closed and body.tail_read


@pytest.mark.parametrize("socket", [False, True], ids=["mock", "loopback"])
@pytest.mark.parametrize("subscription", [False, True])
@pytest.mark.parametrize("drain_failure", [None, "timeout", "cleanup"])
def test_native_runtime_settles_terminal_drain_before_dispatching_tools(
    sqlite_resources, socket, subscription, drain_failure
):
    async def run():
        requests = 0
        closed = 0
        handlers = set()
        failures = []
        bodies = []
        effects = []
        expected_requests = 2 if drain_failure is None else 1

        class FailureSource:
            def __init__(self, source):
                self.source = source
                self.iterator = source.__aiter__()

            def __aiter__(self):
                return self

            async def __anext__(self):
                return await anext(self.iterator)

            async def aclose(self):
                await self.source.aclose()
                raise RuntimeError("synthetic private nested close failure")

        class FailingBody(httpx.AsyncByteStream):
            def __init__(self, source):
                self.source = source

            async def __aiter__(self):
                if drain_failure == "cleanup":
                    async with aclosing_provider_stream(FailureSource(self.source)) as source:
                        async for chunk in source:
                            yield chunk
                else:
                    try:
                        async for chunk in self.source:
                            yield chunk
                    except asyncio.CancelledError:
                        raise TimeoutError("synthetic private tail failure") from None

            async def aclose(self):
                await self.source.aclose()

        def response_payload():
            nonlocal requests
            requests += 1
            assert requests <= expected_requests, "Completed dispatch must not be retried"
            event = terminal()
            event["response"]["id"] = f"resp_{requests}"
            event["response"]["output"] = (
                [
                    {
                        "type": "function_call",
                        "id": "fc_synthetic",
                        "call_id": "call_synthetic",
                        "name": "record_effect",
                        "arguments": "{}",
                        "status": "completed",
                    }
                ]
                if requests == 1
                else [
                    {
                        "type": "message",
                        "role": "assistant",
                        "status": "completed",
                        "content": [{"type": "output_text", "text": "Done", "annotations": []}],
                    }
                ]
            )
            return frame(event)

        async def serve(reader, writer):
            nonlocal closed
            handlers.add(asyncio.current_task())
            try:
                headers = await reader.readuntil(b"\r\n\r\n")
                length = next(
                    int(line.split(b":", 1)[1])
                    for line in headers.split(b"\r\n")
                    if line.lower().startswith(b"content-length:")
                )
                await reader.readexactly(length)
                writer.write(
                    b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\n"
                    b"Connection: close\r\n\r\n" + response_payload()
                )
                await writer.drain()
                assert await reader.read() == b""
                closed += 1
            except Exception as error:
                failures.append(error)
            finally:
                writer.close()
                await writer.wait_closed()
                handlers.discard(asyncio.current_task())

        server = await asyncio.start_server(serve, "127.0.0.1", 0) if socket else None

        class LocalOnlyTransport(httpx.AsyncBaseTransport):
            def __init__(self):
                self.inner = httpx.AsyncHTTPTransport()

            async def handle_async_request(self, request):
                assert server is not None
                request.url = httpx.URL(
                    f"http://127.0.0.1:{server.sockets[0].getsockname()[1]}/responses"
                )
                response = await self.inner.handle_async_request(request)
                if drain_failure is not None:
                    response.stream = FailingBody(response.stream)
                return response

            async def aclose(self):
                await self.inner.aclose()

        def mock_handle(_):
            body = Body([response_payload()])
            bodies.append(body)
            return httpx.Response(200, stream=body if drain_failure is None else FailingBody(body))

        class RecordEffect(Tool):
            spec = ToolSpec(
                name="record_effect",
                description="Record a synthetic effect.",
                input_schema={"type": "object", "properties": {}, "additionalProperties": False},
            )

            async def run(self, ctx, args):
                effects.append("effect")
                return ToolResult(content="Recorded")

        try:
            async with sqlite_resources:
                path = sqlite_resources.path()
                store = sqlite_resources.own(SQLiteSessionStore(path))
                transport = HttpxOpenAITransport()
                async with httpx.AsyncClient(
                    transport=LocalOnlyTransport() if socket else httpx.MockTransport(mock_handle)
                ) as client:
                    transport._client._client = client
                    deadlines = ProviderStreamDeadlines(
                        semantic_progress_timeout_s=1,
                        transport_idle_timeout_s=5,
                        protocol_idle_timeout_s=5,
                        absolute_stream_timeout_s=10,
                    )
                    provider = (
                        OpenAISubscriptionProvider(
                            auth=StaticAuth(), transport=transport, stream_deadlines=deadlines
                        )
                        if subscription
                        else OpenAIProvider(
                            api_key="synthetic", transport=transport, stream_deadlines=deadlines
                        )
                    )
                    app = CayuApp(enable_logging=False, session_store=store)
                    app.register_provider(provider, default=True)
                    app.register_agent(
                        AgentSpec(name="probe", model="synthetic"), tools=[RecordEffect()]
                    )
                    async with asyncio.timeout(10):
                        events = [
                            event
                            async for event in app.run(
                                RunRequest(
                                    agent_name="probe",
                                    session_id="terminal_probe",
                                    messages=REQUEST.messages,
                                )
                            )
                        ]
                        # Verify peer closure before provider/client-wide shutdown.
                        while socket and closed < expected_requests and not failures:
                            await asyncio.sleep(0.001)
                    assert not failures
                    assert requests == expected_requests
                    assert effects == (["effect"] if drain_failure is None else [])
                    if drain_failure is None:
                        assert events[-1].type is EventType.SESSION_COMPLETED
                        assert not any(e.type is EventType.MODEL_ERROR for e in events)
                    else:
                        assert events[-1].type is EventType.SESSION_FAILED
                        assert not any(e.type is EventType.TOOL_CALL_STARTED for e in events)
                        assert "synthetic private" not in str([e.payload for e in events])
                    assert not any(e.type is EventType.MODEL_RETRY for e in events)
                    if not socket:
                        assert all(body.closed and body.tail_read for body in bodies)
                    durable = await store.load_events("terminal_probe")
                    completed = [e for e in durable if e.type is EventType.MODEL_COMPLETED]
                    assert len(completed) == expected_requests
                    assert completed[0].payload["usage_metrics"]["input_tokens"] == 2
                    assert completed[0].payload["usage_metrics"]["output_tokens"] == 1
                    await store.close()
                    reopened = sqlite_resources.own(SQLiteSessionStore(path))
                    assert await reopened.load_events("terminal_probe") == durable
                    assert requests == expected_requests
                    assert effects == (["effect"] if drain_failure is None else [])
                    await provider.aclose()
        finally:
            if server is not None:
                server.close()
                await server.wait_closed()
            for task in tuple(handlers):
                task.cancel()
            await asyncio.gather(*handlers, return_exceptions=True)

    asyncio.run(run())
