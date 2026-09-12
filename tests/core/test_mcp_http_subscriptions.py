from __future__ import annotations

import asyncio
import json
import sys
from collections.abc import AsyncIterator, Callable
from pathlib import Path
from typing import Any

import httpx
import pytest
from tests.core.test_mcp_http import CancellationResistantMcpServerMessageStream
from tests.core.test_mcp_http_modern import ModernMcpHttpServer, _client, _server_spec

from cayu import (
    AgentSpec,
    CayuApp,
    HttpMcpClient,
    McpProtocolEra,
    McpProtocolError,
    McpServerSpec,
    McpToolsetRefreshState,
    McpToolsetUnavailable,
    McpTransportLimits,
    ToolContext,
    connect_mcp_toolset,
)
from cayu.mcp._subscriptions import ModernToolSubscription, SubscriptionEvent

_META_KEY = "io.modelcontextprotocol/subscriptionId"
_ACK = "notifications/subscriptions/acknowledged"
_CHANGED = "notifications/tools/list_changed"


def _notification(method: str, request_id: object, **params: Any) -> dict[str, Any]:
    return {
        "jsonrpc": "2.0",
        "method": method,
        "params": {"_meta": {_META_KEY: request_id}, **params},
    }


def _ack(request_id: object, **notifications: Any) -> dict[str, Any]:
    return _notification(_ACK, request_id, notifications=notifications)


def _complete(request_id: int) -> dict[str, Any]:
    return {
        "jsonrpc": "2.0",
        "id": request_id,
        "result": {"resultType": "complete", "_meta": {_META_KEY: request_id}},
    }


class SubscriptionStream(httpx.AsyncByteStream):
    def __init__(self) -> None:
        self.frames: asyncio.Queue[bytes | None] = asyncio.Queue()
        self.closed = False

    def send(self, message: dict[str, Any]) -> None:
        self.frames.put_nowait(f"data: {json.dumps(message)}\n\n".encode())

    async def __aiter__(self) -> AsyncIterator[bytes]:
        while (frame := await self.frames.get()) is not None:
            yield frame

    async def aclose(self) -> None:
        self.closed = True


class SubscriptionServer(ModernMcpHttpServer):
    def __init__(self, *, automatic_ack: bool = True) -> None:
        super().__init__(tools_list_changed=True)
        self.automatic_ack = automatic_ack
        self.streams: list[SubscriptionStream] = []
        self.subscription_ids: list[int] = []

    def _handle(self, request: httpx.Request) -> httpx.Response:
        if request.method == "POST":
            body = json.loads(request.content)
            if body["method"] == "subscriptions/listen":
                assert not self.streams or self.streams[-1].closed
                self.calls.append((body, dict(request.headers)))
                stream = SubscriptionStream()
                self.streams.append(stream)
                self.subscription_ids.append(body["id"])
                if self.automatic_ack:
                    stream.send(_ack(body["id"], toolsListChanged=True))
                return httpx.Response(
                    200, headers={"content-type": "text/event-stream"}, stream=stream
                )
        return super()._handle(request)


async def _until(predicate: Callable[[], bool]) -> None:
    async with asyncio.timeout(5):
        while not predicate():
            await asyncio.sleep(0.001)


@pytest.mark.parametrize("subscription_id", [True, False, "1", None, 2, [], {}])
def test_subscription_rejects_mismatched_or_coerced_id(subscription_id: object) -> None:
    state = ModernToolSubscription(1)
    with pytest.raises(McpProtocolError, match="subscription ID"):
        state.consume(_ack(subscription_id, toolsListChanged=True))
    assert not state.acknowledged


@pytest.mark.parametrize(
    "filter_value",
    [None, [], {"toolsListChanged": 1}, {"toolsListChanged": "true"}, {"promptsListChanged": True}],
)
def test_subscription_rejects_malformed_or_wider_ack_filter(filter_value: object) -> None:
    state = ModernToolSubscription(1)
    with pytest.raises(McpProtocolError):
        state.consume(_notification(_ACK, 1, notifications=filter_value))
    assert not state.acknowledged


def test_subscription_requires_ack_and_rejects_duplicate_ack() -> None:
    state = ModernToolSubscription(1)
    with pytest.raises(McpProtocolError, match="preceded acknowledgement"):
        state.consume(_notification(_CHANGED, 1))
    assert state.consume(_ack(1, toolsListChanged=True)) is SubscriptionEvent.ACKNOWLEDGED
    with pytest.raises(McpProtocolError, match="unrequested"):
        state.consume(_ack(1, toolsListChanged=True))
    assert state.consume(_notification(_CHANGED, 1)) is SubscriptionEvent.TOOLS_CHANGED
    assert state.consume(_complete(1)) is SubscriptionEvent.COMPLETE


def test_subscription_false_and_empty_filters_do_not_widen_the_honored_subset() -> None:
    state = ModernToolSubscription(1)
    assert (
        state.consume(
            _ack(
                1,
                toolsListChanged=True,
                promptsListChanged=False,
                resourcesListChanged=False,
                resourceSubscriptions=[],
            )
        )
        is SubscriptionEvent.ACKNOWLEDGED
    )


def test_modern_http_subscription_ack_fences_then_reconciles_and_delivers() -> None:
    server = SubscriptionServer(automatic_ack=False)

    async def run() -> None:
        session = await _client(server).connect(_server_spec())
        continuity: list[bool] = []
        changes: list[bool] = []
        try:
            assert session._set_tools_list_changed_continuity_handler(continuity.append)
            assert session._set_tools_list_changed_handler(lambda: changes.append(True))
            assert continuity == [False]
            await _until(lambda: bool(server.streams))
            assert continuity == [False]
            stream = server.streams[0]
            request_id = server.subscription_ids[0]
            stream.send(_ack(request_id, toolsListChanged=True))
            await _until(lambda: continuity == [False, True])
            stream.send(_notification(_CHANGED, request_id))
            await _until(lambda: len(changes) == 1)
        finally:
            await session.close()
        assert all(stream.closed for stream in server.streams)

    asyncio.run(run())
    body, headers = server.requests_for("subscriptions/listen")[0]
    assert body["params"]["notifications"] == {"toolsListChanged": True}
    assert (
        body["params"]["_meta"] == server.requests_for("server/discover")[0][0]["params"]["_meta"]
    )
    assert headers["mcp-method"] == "subscriptions/listen"
    assert headers["mcp-protocol-version"] == "2026-07-28"
    assert "last-event-id" not in headers
    assert "mcp-session-id" not in headers
    assert server.get_calls == server.delete_calls == 0


@pytest.mark.parametrize("graceful", [False, True])
def test_modern_http_subscription_reconnect_settles_old_stream_and_uses_new_id(
    graceful: bool,
) -> None:
    server = SubscriptionServer()

    async def run() -> None:
        session = await _client(server).connect(_server_spec())
        continuity: list[bool] = []
        try:
            session._set_tools_list_changed_continuity_handler(continuity.append)
            session._set_tools_list_changed_handler(lambda: None)
            await _until(lambda: continuity == [False, True])
            if graceful:
                server.streams[0].send(_complete(server.subscription_ids[0]))
            else:
                server.streams[0].frames.put_nowait(None)
            await _until(lambda: continuity == [False, True, False, True])
            assert len(server.streams) == 2
            assert server.subscription_ids[0] != server.subscription_ids[1]
        finally:
            await session.close()
        assert all(stream.closed for stream in server.streams)

    asyncio.run(run())


@pytest.mark.parametrize("honored", [{}, {"toolsListChanged": False}])
def test_modern_http_subscription_unsupported_filter_returns_to_manual_refresh(honored) -> None:
    server = SubscriptionServer(automatic_ack=False)

    async def run() -> None:
        session = await _client(server).connect(_server_spec())
        continuity: list[bool] = []
        try:
            session._set_tools_list_changed_continuity_handler(continuity.append)
            session._set_tools_list_changed_handler(lambda: pytest.fail("unrequested refresh"))
            await _until(lambda: bool(server.streams))
            server.streams[0].send(_ack(server.subscription_ids[0], **honored))
            await _until(lambda: continuity == [False, True])
            assert server.streams[0].closed
            assert len(server.streams) == 1
            assert len(await session.list_tools()) == 1
        finally:
            await session.close()

    asyncio.run(run())


def test_modern_http_subscription_uses_atomic_refresh_owner() -> None:
    server = SubscriptionServer(automatic_ack=False)

    async def run() -> None:
        toolset = await connect_mcp_toolset(
            _server_spec().model_copy(update={"connection_id": "modern-subscription"}),
            client=_client(server),
        )
        app = CayuApp(enable_logging=False)
        stale_adapter = toolset.tools[0]
        app.register_agent(AgentSpec(name="assistant", model="fake-model"), mcp_toolsets=(toolset,))
        try:
            assert toolset.refresh_state is McpToolsetRefreshState.DIRTY
            assert not stale_adapter._dispatch_authority_is_current()
            await _until(lambda: bool(server.streams))
            server.tools.append({"name": "new_tool", "description": "New tool.", "inputSchema": {}})
            server.streams[0].send(_ack(server.subscription_ids[0], toolsListChanged=True))
            await _until(lambda: toolset.refresh_state is McpToolsetRefreshState.READY)
            assert "mcp__modern__new_tool" in app.get_agent("assistant").tools
            assert not stale_adapter._dispatch_authority_is_current()
            with pytest.raises(McpToolsetUnavailable):
                await stale_adapter.run(ToolContext(session_id="test", agent_name="assistant"), {})
            assert server.requests_for("tools/call") == []
            server.tools = server.tools[:1]
            for _ in range(10):
                server.streams[0].send(_notification(_CHANGED, server.subscription_ids[0]))
            await _until(
                lambda: (
                    server.streams[0].frames.empty()
                    and toolset.refresh_state is McpToolsetRefreshState.READY
                    and "mcp__modern__new_tool" not in app.get_agent("assistant").tools
                )
            )
            assert toolset.refresh_state is McpToolsetRefreshState.READY
        finally:
            await toolset.close()

    asyncio.run(run())


def test_modern_http_subscription_official_sdk_interoperability(tmp_path: Path) -> None:
    ready = tmp_path / "ready"
    log = tmp_path / "requests.jsonl"
    fixture = Path(__file__).parents[1] / "fixtures" / "official_mcp_http_subscription_server.py"

    async def run() -> None:
        with (tmp_path / "stderr").open("wb") as stderr:
            process = await asyncio.create_subprocess_exec(
                sys.executable,
                str(fixture),
                "--ready-path",
                str(ready),
                "--request-log",
                str(log),
                stdout=asyncio.subprocess.DEVNULL,
                stderr=stderr,
            )
            try:
                await _until(lambda: ready.exists() or process.returncode is not None)
                assert process.returncode is None, (tmp_path / "stderr").read_text()
                toolset = await connect_mcp_toolset(
                    McpServerSpec(
                        name="sdk", connection_id="sdk-subscription", url=ready.read_text()
                    ),
                    client=HttpMcpClient(protocol_era=McpProtocolEra.MODERN_2026_07_28),
                )
                try:
                    app = CayuApp(enable_logging=False)
                    app.register_agent(
                        AgentSpec(name="assistant", model="unused"), mcp_toolsets=(toolset,)
                    )
                    await _until(lambda: toolset.refresh_state is McpToolsetRefreshState.READY)
                    old = toolset.tools[0]
                    await toolset.session.read_resource("control://advance")
                    await _until(lambda: "mcp__sdk__added" in app.get_agent("assistant").tools)
                    assert not old._dispatch_authority_is_current()
                finally:
                    await toolset.close()
                await _until(lambda: '"subscriptionClosed": true' in log.read_text())
            finally:
                if process.returncode is None:
                    process.terminate()
                try:
                    await asyncio.wait_for(process.wait(), timeout=10)
                except TimeoutError:
                    process.kill()
                    await process.wait()
                    raise
            assert process.returncode is not None

    asyncio.run(run())
    requests = [json.loads(line) for line in log.read_text().splitlines()]
    listeners = [request for request in requests if request.get("method") == "subscriptions/listen"]
    assert len(listeners) == 1
    assert listeners[0]["params"]["notifications"] == {"toolsListChanged": True}
    assert all(
        request.get("method") not in {"initialize", "notifications/initialized"}
        for request in requests
    )


@pytest.mark.parametrize(
    "fault",
    [
        "wrong-id",
        "boolean-id",
        "duplicate-ack",
        "unrequested",
        "bad-result",
        "wrong-response-id",
        "oversize",
        "malformed-json",
    ],
)
def test_modern_http_subscription_protocol_failure_fences_and_closes(fault: str) -> None:
    server = SubscriptionServer()

    async def run() -> None:
        session = await _client(server).connect(_server_spec())
        continuity: list[bool] = []
        changes: list[bool] = []
        try:
            session.transport_limits = McpTransportLimits(
                max_message_bytes=2048, max_response_bytes=4096
            )
            session._set_tools_list_changed_continuity_handler(continuity.append)
            session._set_tools_list_changed_handler(lambda: changes.append(True))
            await _until(lambda: continuity == [False, True])
            stream = server.streams[0]
            identity = server.subscription_ids[0]
            if fault == "oversize":
                stream.frames.put_nowait(b"data: " + b"x" * 8192)
            elif fault == "malformed-json":
                stream.frames.put_nowait(b"data: {invalid}\n\n")
            else:
                messages = {
                    "wrong-id": _notification(_CHANGED, identity + 1),
                    "boolean-id": _notification(_CHANGED, True),
                    "duplicate-ack": _ack(identity, toolsListChanged=True),
                    "unrequested": _notification("notifications/prompts/list_changed", identity),
                    "bad-result": {
                        **_complete(identity),
                        "result": {"resultType": "input_required", "_meta": {_META_KEY: identity}},
                    },
                    "wrong-response-id": {**_complete(identity), "id": identity + 1},
                }
                stream.send(messages[fault])
            await _until(lambda: session._tools_list_changed_listener_failure_message() is not None)
            assert continuity == [False, True, False]
            assert changes == []
            with pytest.raises(McpProtocolError):
                await session.list_tools()
            await _until(lambda: stream.closed)
            assert len(server.streams) == 1
        finally:
            await session.close()

    asyncio.run(run())


def test_modern_http_subscription_heartbeat_cannot_extend_ack_deadline() -> None:
    server = SubscriptionServer(automatic_ack=False)

    async def run() -> None:
        session = await _client(server).connect(_server_spec())
        continuity: list[bool] = []
        session.transport_limits = McpTransportLimits(idle_timeout_s=2, total_call_timeout_s=0.1)
        heartbeat: asyncio.Task[None] | None = None
        try:
            session._set_tools_list_changed_continuity_handler(continuity.append)
            session._set_tools_list_changed_handler(lambda: None)
            await _until(lambda: bool(server.streams))

            async def send_heartbeats() -> None:
                while not server.streams[0].closed:
                    server.streams[0].frames.put_nowait(b": heartbeat\n\n")
                    await asyncio.sleep(0.01)

            heartbeat = asyncio.create_task(send_heartbeats())
            await _until(lambda: len(server.streams) >= 2)
            assert server.streams[0].closed
            assert continuity == [False]
        finally:
            await session.close()
            if heartbeat is not None:
                heartbeat.cancel()
                await asyncio.gather(heartbeat, return_exceptions=True)

    asyncio.run(run())


def test_modern_http_acknowledged_subscription_outlives_call_and_aggregate_limits() -> None:
    server = SubscriptionServer()

    async def run() -> None:
        session = await _client(server).connect(_server_spec())
        continuity: list[bool] = []
        changes: list[bool] = []
        session.transport_limits = McpTransportLimits(
            max_message_bytes=1024,
            max_response_bytes=1024,
            idle_timeout_s=2,
            total_call_timeout_s=0.1,
        )
        try:
            session._set_tools_list_changed_continuity_handler(continuity.append)
            session._set_tools_list_changed_handler(lambda: changes.append(True))
            await _until(lambda: continuity == [False, True])
            await asyncio.sleep(0.15)
            for _ in range(30):
                server.streams[0].send(_notification(_CHANGED, server.subscription_ids[0]))
            await _until(lambda: len(changes) == 30)
            assert len(server.streams) == 1
            assert continuity == [False, True]
        finally:
            await session.close()

    asyncio.run(run())


def test_modern_http_idle_subscription_reconnects_under_a_new_acknowledgement() -> None:
    server = SubscriptionServer()

    async def run() -> None:
        session = await _client(server).connect(_server_spec())
        continuity: list[bool] = []
        session.transport_limits = McpTransportLimits(idle_timeout_s=0.1, total_call_timeout_s=2)
        try:
            session._set_tools_list_changed_continuity_handler(continuity.append)
            session._set_tools_list_changed_handler(lambda: None)
            await _until(lambda: continuity == [False, True, False, True])
            assert server.streams[0].closed
        finally:
            await session.close()

    asyncio.run(run())


def test_modern_http_ordinary_post_notifications_cannot_trigger_subscription_handler() -> None:
    server = SubscriptionServer()
    original_handle = server._handle

    def handle(request: httpx.Request) -> httpx.Response:
        response = original_handle(request)
        body = json.loads(request.content)
        if body["method"] == "tools/list":
            result = response.json()
            # Even a correlated signal belongs only on the subscription stream.
            notification = _notification(_CHANGED, server.subscription_ids[0])
            return httpx.Response(
                200,
                headers={"content-type": "text/event-stream"},
                content=(
                    f"data: {json.dumps(notification)}\n\ndata: {json.dumps(result)}\n\n".encode()
                ),
            )
        return response

    server._handle = handle

    async def run() -> None:
        session = await _client(server).connect(_server_spec())
        continuity: list[bool] = []
        changes: list[bool] = []
        try:
            session._set_tools_list_changed_continuity_handler(continuity.append)
            session._set_tools_list_changed_handler(lambda: changes.append(True))
            await _until(lambda: continuity == [False, True])
            await session.list_tools()
            assert changes == []
        finally:
            await session.close()

    asyncio.run(run())


def test_modern_http_subscription_close_settles_cancellation_resistant_read() -> None:
    server = SubscriptionServer()
    reading = asyncio.Event()

    class ObservedRead(CancellationResistantMcpServerMessageStream):
        async def __aiter__(self):
            reading.set()
            async for chunk in super().__aiter__():
                yield chunk

    stream = ObservedRead()
    original_handle = server._handle
    entered = asyncio.Event()

    def handle(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        if body["method"] == "subscriptions/listen":
            entered.set()
            return httpx.Response(200, headers={"content-type": "text/event-stream"}, stream=stream)
        return original_handle(request)

    server._handle = handle

    async def run() -> None:
        session = await _client(server).connect(_server_spec())
        existing = set(asyncio.all_tasks())
        try:
            session._set_tools_list_changed_continuity_handler(lambda _ready: None)
            session._set_tools_list_changed_handler(lambda: None)
            await asyncio.wait_for(entered.wait(), timeout=5)
            await asyncio.wait_for(reading.wait(), timeout=5)
        finally:
            await asyncio.wait_for(session.close(), timeout=5)
        assert stream.closed.is_set()
        assert stream.cancelled.is_set()
        await _until(
            lambda: not [task for task in asyncio.all_tasks() - existing if not task.done()]
        )

    asyncio.run(run())


@pytest.mark.parametrize("before_ack", [True, False])
def test_modern_http_subscription_rejects_raw_controls_without_secret_diagnostics(
    before_ack: bool,
) -> None:
    server = SubscriptionServer(automatic_ack=not before_ack)
    canary = "subscription-private-control-canary"

    async def run() -> None:
        session = await _client(server).connect(_server_spec())
        continuity: list[bool] = []
        try:
            session._set_tools_list_changed_continuity_handler(continuity.append)
            session._set_tools_list_changed_handler(
                lambda: pytest.fail("untrusted event dispatched")
            )
            await _until(lambda: bool(server.streams))
            if not before_ack:
                await _until(lambda: continuity == [False, True])
            server.streams[0].send(_notification(_ACK if before_ack else _CHANGED, canary))
            await _until(lambda: session._tools_list_changed_listener_failure_message() is not None)
            assert canary not in session._tools_list_changed_listener_failure_message()
            assert not (before_ack and True in continuity)
        finally:
            await session.close()

    asyncio.run(run())
