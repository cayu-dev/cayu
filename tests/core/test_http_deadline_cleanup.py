"""Loopback socket regression for interrupted SSE read/close ownership."""

from __future__ import annotations

import asyncio
import json

import httpx
import pytest

from cayu import (
    AgentSpec,
    CayuApp,
    ExecutionDeadline,
    SQLiteSessionStore,
    StepError,
    WorkflowBase,
    WorkflowSpec,
    execution_deadline_scope,
    parallel,
    step,
)
from cayu.providers import deadlines as deadline_state
from cayu.providers.openai import HttpxOpenAITransport, OpenAIProvider
from cayu.workflows import StepRunOptions


class ProbeWorkflow(WorkflowBase):
    spec = WorkflowSpec(name="http-deadline-probe")

    async def run(self, sid):
        yield await self.context(sid).start()


def frame(event):
    return b"data: " + json.dumps(event).encode() + b"\n\n"


@pytest.mark.parametrize("mock", [False, True], ids=["socket", "mock"])
@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize("children,bounded_parent", [(1, False), (2, True)])
@pytest.mark.parametrize("cancel_parent", [False, True], ids=["deadline", "parent-cancel"])
def test_http_child_shutdown(tmp_path, mock, streaming, children, bounded_parent, cancel_parent):
    async def scenario():
        owners_before = set(deadline_state._PROVIDER_DEADLINE_AWAIT_OWNERS)
        received = 0
        closed = 0
        all_received = asyncio.Event()
        all_closed = asyncio.Event()
        handlers = set()
        payload = frame(
            {
                "type": "response.created",
                "response": {
                    "id": "resp_synthetic",
                    "status": "in_progress",
                    "output": [],
                },
            }
        )
        if streaming:
            payload += frame({"type": "response.output_text.delta", "delta": "hello"})

        def mark_received():
            nonlocal received
            received += 1
            if received == children:
                all_received.set()

        def mark_closed():
            nonlocal closed
            closed += 1
            if closed == children:
                all_closed.set()

        async def serve(reader, writer):
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
                    b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nConnection: close\r\n\r\n"
                    + payload
                )
                await writer.drain()
                mark_received()
                assert await reader.read() == b""
            finally:
                writer.close()
                await writer.wait_closed()
                mark_closed()
                handlers.discard(asyncio.current_task())

        server = await asyncio.start_server(serve, "127.0.0.1", 0)
        port = server.sockets[0].getsockname()[1]

        class LocalOnlyTransport(httpx.AsyncBaseTransport):
            def __init__(self):
                self.inner = httpx.AsyncHTTPTransport()

            async def handle_async_request(self, request):
                request.url = httpx.URL(f"http://127.0.0.1:{port}/responses")
                return await self.inner.handle_async_request(request)

            async def aclose(self):
                await self.inner.aclose()

        class MockStream(httpx.AsyncByteStream):
            async def __aiter__(self):
                yield payload
                mark_received()
                await asyncio.Event().wait()

            async def aclose(self):
                mark_closed()

        def mock_handle(request):
            return httpx.Response(
                200, headers={"content-type": "text/event-stream"}, stream=MockStream()
            )

        transport = HttpxOpenAITransport()
        path = tmp_path / "sessions.db"
        store = SQLiteSessionStore(path)
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(mock_handle) if mock else LocalOnlyTransport()
        ) as client:
            transport._client._client = client
            provider = OpenAIProvider(
                api_key="synthetic", base_url="https://synthetic.invalid", transport=transport
            )
            app = CayuApp(enable_logging=False, session_store=store)
            app.register_provider(provider, default=True)
            app.register_agent(AgentSpec(name="probe", model="synthetic"))
            ctx = ProbeWorkflow(app).context("probe")
            await ctx.start()

            async def run():
                async with execution_deadline_scope(
                    ExecutionDeadline.after(30 if bounded_parent else None, scope="parent")
                ):
                    return await parallel(
                        [
                            step(
                                ctx,
                                agent="probe",
                                step_id=str(i),
                                prompt="synthetic",
                                run_options=StepRunOptions(
                                    execution_deadline=ExecutionDeadline.after(
                                        30 if cancel_parent else 3, scope="child"
                                    )
                                ),
                            )
                            for i in range(children)
                        ]
                    )

            task = asyncio.create_task(run())
            try:
                await asyncio.wait_for(all_received.wait(), 10)
                if cancel_parent:
                    task.cancel()
                    with pytest.raises(asyncio.CancelledError):
                        await asyncio.wait_for(task, 10)
                else:
                    result = await asyncio.wait_for(task, 10)
                    assert len(result.failures) == children
                    for failure in result.failures:
                        evidence = failure.evidence
                        assert evidence.classification == "deadline"
                        assert evidence.deadline.scope == "child"
                        assert evidence.deadline_phase == "in_flight"
                        assert evidence.secondary_failures is False
                        assert evidence.settlement == "unknown"
                        assert evidence.run_epoch == 1
                        assert evidence.terminal_event_id
                await asyncio.wait_for(all_closed.wait(), 2)
                assert received == closed == children
                await store.close()
                store = SQLiteSessionStore(path)
                if not cancel_parent:
                    replay_app = CayuApp(enable_logging=False, session_store=store)
                    replay_app.register_provider(provider, default=True)
                    replay_app.register_agent(AgentSpec(name="probe", model="synthetic"))
                    replay_ctx = ProbeWorkflow(replay_app).context("probe")
                    await replay_ctx.start()
                    for failure in result.failures:
                        events = await store.load_events(failure.session_id)
                        terminal = next(
                            e for e in events if e.id == failure.evidence.terminal_event_id
                        )
                        assert not terminal.payload.get("provider_cancellation_failures")
                        assert terminal.payload["failure_evidence"]["secondary_failures"] is False
                        with pytest.raises(StepError) as raised:
                            await step(
                                replay_ctx,
                                agent="probe",
                                step_id=failure.step_id,
                                prompt="synthetic",
                            )
                        assert raised.value.evidence.model_dump(
                            exclude={"exception_types"}
                        ) == failure.evidence.model_dump(exclude={"exception_types"})
                    assert received == children
                async with asyncio.timeout(2):
                    while deadline_state._PROVIDER_DEADLINE_AWAIT_OWNERS - owners_before:
                        await asyncio.sleep(0)
                assert not handlers
            finally:
                if not task.done():
                    task.cancel()
                    await asyncio.gather(task, return_exceptions=True)
                await provider.aclose()
                await store.close()
                server.close()
                await server.wait_closed()
                for handler in list(handlers):
                    handler.cancel()
                await asyncio.gather(*handlers, return_exceptions=True)

    asyncio.run(scenario())


@pytest.mark.parametrize("close_kind", ["delayed", "failure", "timeout", "cancelled"])
def test_http_close_failure_or_retention_is_not_clean(close_kind):
    from cayu import Message, ModelRequest
    from cayu.providers._credential_boundary import (
        aclosing_provider_stream,
        provider_cancellation_failures,
    )

    async def scenario():
        owners_before = set(deadline_state._PROVIDER_DEADLINE_AWAIT_OWNERS)
        reading = asyncio.Event()
        closing = asyncio.Event()
        release = asyncio.Event()
        settled = asyncio.Event()
        closes = 0

        class Stream(httpx.AsyncByteStream):
            async def __aiter__(self):
                yield frame(
                    {
                        "type": "response.created",
                        "response": {
                            "id": "resp_synthetic",
                            "status": "in_progress",
                            "output": [],
                        },
                    }
                )
                reading.set()
                await asyncio.Event().wait()

            async def aclose(self):
                nonlocal closes
                closes += 1
                closing.set()
                try:
                    if close_kind == "delayed":
                        await release.wait()
                    elif close_kind == "failure":
                        raise RuntimeError("synthetic close failure")
                    elif close_kind == "timeout":
                        raise TimeoutError("synthetic close timeout")
                    else:
                        raise asyncio.CancelledError("synthetic close cancellation")
                finally:
                    settled.set()

        transport = HttpxOpenAITransport()
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(lambda request: httpx.Response(200, stream=Stream()))
        ) as client:
            transport._client._client = client
            provider = OpenAIProvider(api_key="synthetic", transport=transport)

            async def consume():
                stream = provider.runtime_stream(
                    ModelRequest(model="synthetic", messages=[Message.text("user", "synthetic")])
                )
                async with aclosing_provider_stream(stream):
                    async for _ in stream:
                        pass

            task = asyncio.create_task(consume())
            try:
                await asyncio.wait_for(reading.wait(), 2)
                task.cancel()
                with pytest.raises(asyncio.CancelledError) as caught:
                    await asyncio.wait_for(task, 2)
                failures = provider_cancellation_failures(caught.value)
                assert failures
                expected_reason = {
                    "delayed": "cleanup_pending",
                    "failure": "close_exception",
                    "timeout": "cleanup_timeout",
                    "cancelled": "cleanup_cancelled",
                }[close_kind]
                assert any(item.get("cleanup_reason") == expected_reason for item in failures)
                assert all(
                    item.get("remote_settlement_state", "unknown") == "unknown" for item in failures
                )
                assert closing.is_set()
                assert closes == 1
                if close_kind == "delayed":
                    assert not settled.is_set()
                    assert deadline_state._PROVIDER_DEADLINE_AWAIT_OWNERS - owners_before
                    assert any(item.get("cleanup_reason") == "cleanup_pending" for item in failures)
                release.set()
                await asyncio.wait_for(settled.wait(), 2)
                async with asyncio.timeout(2):
                    while deadline_state._PROVIDER_DEADLINE_AWAIT_OWNERS - owners_before:
                        await asyncio.sleep(0)
            finally:
                release.set()
                if not task.done():
                    task.cancel()
                await asyncio.gather(task, return_exceptions=True)
                await provider.aclose()

    asyncio.run(scenario())
