from __future__ import annotations

import asyncio
import json
import logging
import time
import traceback
import warnings
from collections.abc import AsyncIterator, Callable
from typing import Any

import httpx
import pytest
from tests.provider_traceback_assertions import is_cayu_source_filename

from cayu import (
    MCP_PROTOCOL_VERSION,
    HttpMcpClient,
    HttpMcpSession,
    McpCallDeadlineExceededError,
    McpIdleTimeoutError,
    McpInitializeResult,
    McpMessageTooLargeError,
    McpPeerClosedError,
    McpProtocolError,
    McpResponseTooLargeError,
    McpServerSpec,
    McpToolDefinition,
    McpToolset,
    McpTransportLimits,
    SecretRedactor,
    SecretRef,
    StaticVault,
    ToolContext,
    connect_mcp_toolset,
)
from cayu.mcp import http as mcp_http_module
from cayu.mcp import tools as mcp_tools_module
from cayu.mcp.http import (
    MCP_PROTOCOL_VERSION_HEADER,
    MCP_SESSION_ID_HEADER,
    _http_settlement_task,
    _HttpCallBudget,
)
from cayu.vaults import REDACTED_SECRET


def _limits(
    *,
    max_message_bytes: int = 1_024,
    max_response_bytes: int = 4_096,
    idle_timeout_s: float = 1.0,
    total_call_timeout_s: float = 2.0,
) -> McpTransportLimits:
    return McpTransportLimits(
        max_message_bytes=max_message_bytes,
        max_response_bytes=max_response_bytes,
        idle_timeout_s=idle_timeout_s,
        total_call_timeout_s=total_call_timeout_s,
    )


def _jsonrpc_body(request_id: int, *, padding: int = 0) -> bytes:
    return json.dumps(
        {
            "jsonrpc": "2.0",
            "id": request_id,
            "result": {
                "content": [{"type": "text", "text": "x" * padding}],
                "structuredContent": {},
            },
        },
        separators=(",", ":"),
    ).encode()


def _initialize_body(request_id: int) -> bytes:
    return json.dumps(
        {
            "jsonrpc": "2.0",
            "id": request_id,
            "result": {
                "protocolVersion": MCP_PROTOCOL_VERSION,
                "capabilities": {},
                "serverInfo": {"name": "bounded-test", "version": "1"},
            },
        },
        separators=(",", ":"),
    ).encode()


def _deep_json_arguments(depth: int, canary: str) -> dict[str, Any]:
    value: Any = canary
    for _ in range(depth):
        value = [value]
    return {"nested": value}


def _deep_jsonrpc_response_body(request_id: int, *, depth: int, canary: str) -> bytes:
    prefix = ('{"jsonrpc":"2.0","id":' + str(request_id) + ',"result":{"nested":').encode()
    leaf = json.dumps(canary, separators=(",", ":")).encode()
    return prefix + b"[" * depth + leaf + b"]" * depth + b"}}"


def _deep_initialize_response_body(request_id: int, *, depth: int, canary: str) -> bytes:
    prefix = (
        '{"jsonrpc":"2.0","id":'
        + str(request_id)
        + ',"result":{"protocolVersion":'
        + json.dumps(MCP_PROTOCOL_VERSION)
        + ',"nested":'
    ).encode()
    leaf = json.dumps(canary, separators=(",", ":")).encode()
    return prefix + b"[" * depth + leaf + b"]" * depth + b"}}"


def _session(
    handler: Callable[[httpx.Request], httpx.Response],
    *,
    limits: McpTransportLimits,
    secret_redactor: SecretRedactor | None = None,
) -> HttpMcpSession:
    return HttpMcpSession(
        server=McpServerSpec(name="bounded-http", url="https://mcp.example/rpc"),
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
        url="https://mcp.example/rpc",
        client_name="cayu-test",
        client_version="1",
        transport_limits=limits,
        secret_redactor=secret_redactor,
    )


def _assert_cayu_traceback_does_not_retain(
    error: BaseException,
    *canaries: str,
) -> None:
    traceback_cursor = error.__traceback__
    while traceback_cursor is not None:
        if is_cayu_source_filename(traceback_cursor.tb_frame.f_code.co_filename):
            for value in traceback_cursor.tb_frame.f_locals.values():
                rendered = repr(value)
                assert all(canary not in rendered for canary in canaries)
        traceback_cursor = traceback_cursor.tb_next


class _ChunkStream(httpx.AsyncByteStream):
    def __init__(self, chunks: list[bytes], *, delay_s: float = 0) -> None:
        self.chunks = chunks
        self.delay_s = delay_s
        self.closed = False
        self.close_calls = 0

    async def __aiter__(self) -> AsyncIterator[bytes]:
        for chunk in self.chunks:
            if self.delay_s:
                await asyncio.sleep(self.delay_s)
            yield chunk

    async def aclose(self) -> None:
        self.close_calls += 1
        self.closed = True


class _BlockingStream(httpx.AsyncByteStream):
    def __init__(self) -> None:
        self.started = asyncio.Event()
        self.closed = False

    async def __aiter__(self) -> AsyncIterator[bytes]:
        self.started.set()
        await asyncio.Event().wait()
        yield b""  # pragma: no cover - the cancellation path cannot reach this

    async def aclose(self) -> None:
        self.closed = True


class _FirstCloseBlocksStream(_ChunkStream):
    def __init__(self, chunks: list[bytes]) -> None:
        super().__init__(chunks)
        self.close_calls = 0
        self.close_started = asyncio.Event()
        self.release_close = asyncio.Event()
        self.close_active = False
        self.overlapping_close = False

    async def aclose(self) -> None:
        if self.close_active:
            self.overlapping_close = True
        self.close_active = True
        self.close_calls += 1
        self.close_started.set()
        try:
            if self.close_calls == 1:
                await self.release_close.wait()
            self.closed = True
        finally:
            self.close_active = False


class _FirstCloseBlocksThenFailsStream(_FirstCloseBlocksStream):
    def __init__(self, chunks: list[bytes], failure: RuntimeError) -> None:
        super().__init__(chunks)
        self.failure = failure

    async def aclose(self) -> None:
        if self.close_active:
            self.overlapping_close = True
        self.close_active = True
        self.close_calls += 1
        self.close_started.set()
        try:
            if self.close_calls == 1:
                await self.release_close.wait()
            raise self.failure
        finally:
            self.close_active = False


class _CancellationResistantCloseStream(_FirstCloseBlocksStream):
    def __init__(self, chunks: list[bytes]) -> None:
        super().__init__(chunks)
        self.cancellation_seen = asyncio.Event()

    async def aclose(self) -> None:
        if self.close_active:
            self.overlapping_close = True
        self.close_active = True
        self.close_calls += 1
        self.close_started.set()
        try:
            if self.close_calls == 1:
                try:
                    await self.release_close.wait()
                except asyncio.CancelledError:
                    self.cancellation_seen.set()
                    await self.release_close.wait()
            self.closed = True
        finally:
            self.close_active = False


class _ReleaseClosesStream(_BlockingStream):
    def __init__(self) -> None:
        super().__init__()
        self.release_close = asyncio.Event()

    async def aclose(self) -> None:
        await self.release_close.wait()
        self.closed = True


class _CleanupFailsStream(_ChunkStream):
    def __init__(self, chunks: list[bytes], failure: RuntimeError) -> None:
        super().__init__(chunks)
        self.failure = failure
        self.close_calls = 0

    async def aclose(self) -> None:
        self.close_calls += 1
        raise self.failure


class _ScalarFatalCleanupStream(_ChunkStream):
    def __init__(self, chunks: list[bytes], failure: BaseException) -> None:
        super().__init__(chunks)
        self.failure = failure

    async def aclose(self) -> None:
        self.close_calls += 1
        raise self.failure


class _BlockingCleanupFailsStream(_BlockingStream):
    def __init__(self, failure: RuntimeError) -> None:
        super().__init__()
        self.failure = failure

    async def aclose(self) -> None:
        raise self.failure


class _ChunkThenBlocksStream(httpx.AsyncByteStream):
    def __init__(self, chunk: bytes) -> None:
        self.chunk = chunk
        self.blocked = asyncio.Event()
        self.closed = False

    async def __aiter__(self) -> AsyncIterator[bytes]:
        yield self.chunk
        self.blocked.set()
        await asyncio.Event().wait()

    async def aclose(self) -> None:
        self.closed = True


class _TransportFailureStream(httpx.AsyncByteStream):
    def __init__(self, failure: BaseException) -> None:
        self.failure = failure

    async def __aiter__(self) -> AsyncIterator[bytes]:
        raise self.failure
        yield b""  # pragma: no cover - makes this an async generator

    async def aclose(self) -> None:
        return None


class _HostileRequestError(httpx.RequestError):
    def __init__(self, secret: str, *, request: httpx.Request) -> None:
        super().__init__("request adapter failed", request=request)
        self.secret = secret
        self.render_calls = 0

    def __str__(self) -> str:
        self.render_calls += 1
        print(self.secret)
        logging.warning(self.secret)
        warnings.warn(self.secret, stacklevel=1)
        return self.secret


class _HostileCancellationDetail:
    def __init__(self, secret: str) -> None:
        self.secret = secret
        self.render_calls = 0

    def __str__(self) -> str:
        self.render_calls += 1
        return self.secret


class _FailingCloseTransport(httpx.AsyncBaseTransport):
    def __init__(self, failure: RuntimeError) -> None:
        self.failure = failure
        self.close_calls = 0

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        raise AssertionError("failing-close transport should not receive a request")

    async def aclose(self) -> None:
        self.close_calls += 1
        raise self.failure


class _BlockingFailingCloseTransport(httpx.AsyncBaseTransport):
    def __init__(self, failure: RuntimeError) -> None:
        self.failure = failure
        self.close_calls = 0
        self.close_started = asyncio.Event()
        self.release_close = asyncio.Event()

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        raise AssertionError("blocking failing-close transport should not receive a request")

    async def aclose(self) -> None:
        self.close_calls += 1
        self.close_started.set()
        await self.release_close.wait()
        raise self.failure


class _InitializationCleanupFailureTransport(httpx.AsyncBaseTransport):
    def __init__(self, failure: RuntimeError) -> None:
        self.failure = failure
        self.events: list[str] = []
        self.delete_headers: httpx.Headers | None = None
        self.close_started = asyncio.Event()
        self.release_close = asyncio.Event()

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        if request.method == "DELETE":
            self.events.append("delete")
            self.delete_headers = request.headers
            return httpx.Response(200)
        payload = json.loads(request.content)
        assert payload["method"] == "initialize"
        self.events.append("initialize")
        return httpx.Response(
            200,
            headers={
                "content-type": "application/json",
                MCP_SESSION_ID_HEADER: "processing-deadline-session",
            },
            content=_initialize_body(payload["id"]),
        )

    async def aclose(self) -> None:
        self.events.append("close")
        self.close_started.set()
        await self.release_close.wait()
        raise self.failure


class _DeepInitializationTransport(httpx.AsyncBaseTransport):
    def __init__(self, *, depth: int, canary: str) -> None:
        self.depth = depth
        self.canary = canary
        self.events: list[str] = []
        self.delete_headers: httpx.Headers | None = None

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        if request.method == "DELETE":
            self.events.append("delete")
            self.delete_headers = request.headers
            return httpx.Response(200)
        payload = json.loads(request.content)
        assert payload["method"] == "initialize"
        self.events.append("initialize")
        return httpx.Response(
            200,
            headers={
                "content-type": "application/json",
                MCP_SESSION_ID_HEADER: "deep-initialize-session",
            },
            content=_deep_initialize_response_body(
                payload["id"],
                depth=self.depth,
                canary=self.canary,
            ),
        )

    async def aclose(self) -> None:
        self.events.append("close")


class _RejectedInitializationTransport(httpx.AsyncBaseTransport):
    def __init__(self, *, media_type: str, rejection: str) -> None:
        self.media_type = media_type
        self.rejection = rejection
        self.events: list[str] = []
        self.delete_headers: httpx.Headers | None = None

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        if request.method == "DELETE":
            self.events.append("delete")
            self.delete_headers = request.headers
            return httpx.Response(200)
        payload = json.loads(request.content)
        assert payload["method"] == "initialize"
        self.events.append("initialize")
        if self.rejection == "server_request":
            assert self.media_type == "text/event-stream"
            response = {
                "jsonrpc": "2.0",
                "id": payload["id"] + 1,
                "method": "roots/list",
            }
            body = b"data: " + json.dumps(response, separators=(",", ":")).encode() + b"\n\n"
            return httpx.Response(
                200,
                headers={
                    "content-type": self.media_type,
                    MCP_SESSION_ID_HEADER: "rejected-initialize-session",
                },
                content=body,
            )
        response_id = payload["id"] + int(self.rejection == "mismatched_id")
        response = {
            "jsonrpc": "1.0" if self.rejection == "invalid_envelope" else "2.0",
            "id": response_id,
            "result": {
                "protocolVersion": MCP_PROTOCOL_VERSION,
                "capabilities": {},
                "serverInfo": {"name": "rejected-initialize", "version": "1"},
            },
        }
        body = json.dumps(response, separators=(",", ":")).encode()
        if self.media_type == "text/event-stream":
            body = b"data: " + body + b"\n\n"
        return httpx.Response(
            200,
            headers={
                "content-type": self.media_type,
                MCP_SESSION_ID_HEADER: "rejected-initialize-session",
            },
            content=body,
        )

    async def aclose(self) -> None:
        self.events.append("close")


class _RejectedInitializationHeaderTransport(httpx.AsyncBaseTransport):
    def __init__(self, *, delete_failure: BaseException | None = None) -> None:
        self.delete_failure = delete_failure
        self.events: list[str] = []
        self.delete_headers: httpx.Headers | None = None

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        if request.method == "DELETE":
            self.events.append("delete")
            self.delete_headers = request.headers
            if self.delete_failure is not None:
                raise self.delete_failure
            return httpx.Response(200)
        payload = json.loads(request.content)
        assert payload["method"] == "initialize"
        self.events.append("initialize")
        return httpx.Response(
            200,
            headers={
                "content-type": "application/json",
                "content-encoding": "gzip",
                MCP_SESSION_ID_HEADER: "rejected-header-initialize-session",
            },
            stream=_ChunkStream([_initialize_body(payload["id"])]),
        )

    async def aclose(self) -> None:
        self.events.append("close")


class _RetainedInitializationSettlementTransport(httpx.AsyncBaseTransport):
    def __init__(
        self,
        stream: httpx.AsyncByteStream,
        *,
        delete_failure: BaseException | None = None,
        delete_stream: httpx.AsyncByteStream | None = None,
        client_close_failure: BaseException | None = None,
    ) -> None:
        self.stream = stream
        self.delete_failure = delete_failure
        self.delete_stream = delete_stream
        self.client_close_failure = client_close_failure
        self.events: list[str] = []
        self.delete_headers: httpx.Headers | None = None

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        if request.method == "DELETE":
            self.events.append("delete")
            self.delete_headers = request.headers
            if self.delete_failure is not None:
                raise self.delete_failure
            if self.delete_stream is not None:
                return httpx.Response(200, stream=self.delete_stream)
            return httpx.Response(200)
        payload = json.loads(request.content)
        assert payload["method"] == "initialize"
        self.events.append("initialize")
        return httpx.Response(
            200,
            headers={
                "content-type": "application/json",
                MCP_SESSION_ID_HEADER: "retained-initialize-session",
            },
            stream=self.stream,
        )

    async def aclose(self) -> None:
        self.events.append("close")
        if self.client_close_failure is not None:
            raise self.client_close_failure


class _LateInitializationResponseTransport(httpx.AsyncBaseTransport):
    def __init__(self, *, delete_stream: httpx.AsyncByteStream | None = None) -> None:
        self.release_response = asyncio.Event()
        self.cancellation_seen = asyncio.Event()
        self.delete_stream = delete_stream
        self.events: list[str] = []
        self.delete_headers: httpx.Headers | None = None

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        if request.method == "DELETE":
            self.events.append("delete")
            self.delete_headers = request.headers
            if self.delete_stream is not None:
                return httpx.Response(200, stream=self.delete_stream)
            return httpx.Response(200)
        payload = json.loads(request.content)
        assert payload["method"] == "initialize"
        self.events.append("initialize")
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            self.cancellation_seen.set()
            await self.release_response.wait()
        return httpx.Response(
            200,
            headers={
                "content-type": "application/json",
                MCP_SESSION_ID_HEADER: "late-initialize-session",
            },
            stream=_ChunkStream([_initialize_body(payload["id"])]),
        )

    async def aclose(self) -> None:
        self.events.append("close")


class _InitializedNotificationSettlementTransport(httpx.AsyncBaseTransport):
    def __init__(self) -> None:
        self.stream = _BlockingStream()
        self.events: list[str] = []
        self.delete_headers: httpx.Headers | None = None

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        if request.method == "DELETE":
            self.events.append("delete")
            self.delete_headers = request.headers
            return httpx.Response(200)
        payload = json.loads(request.content)
        if payload["method"] == "initialize":
            self.events.append("initialize")
            return httpx.Response(
                200,
                headers={
                    "content-type": "application/json",
                    MCP_SESSION_ID_HEADER: "initialized-notification-session",
                },
                content=_initialize_body(payload["id"]),
            )
        assert payload["method"] == "notifications/initialized"
        self.events.append("initialized")
        return httpx.Response(202, stream=self.stream)

    async def aclose(self) -> None:
        self.events.append("close")


class _GroupedDeleteFailureTransport(httpx.AsyncBaseTransport):
    def __init__(self) -> None:
        self.close_calls = 0

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        assert request.method == "DELETE"
        raise BaseExceptionGroup(
            "grouped DELETE failure",
            [asyncio.CancelledError("delete cancelled"), RuntimeError("delete failed")],
        )

    async def aclose(self) -> None:
        self.close_calls += 1


class _EstablishedDeadlineCleanupTransport(httpx.AsyncBaseTransport):
    def __init__(
        self,
        *,
        delete_failure: BaseException | None = None,
        delete_stream: httpx.AsyncByteStream | None = None,
        client_close_failure: BaseException | None = None,
    ) -> None:
        if (delete_failure is None) == (delete_stream is None):
            raise ValueError("exactly one DELETE outcome is required")
        self.delete_failure = delete_failure
        self.delete_stream = delete_stream
        self.client_close_failure = client_close_failure
        self.events: list[str] = []

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        if request.method == "DELETE":
            self.events.append("delete")
            if self.delete_failure is not None:
                raise self.delete_failure
            assert self.delete_stream is not None
            return httpx.Response(200, stream=self.delete_stream)
        self.events.append("request")
        request_id = json.loads(request.content)["id"]
        return httpx.Response(
            200,
            headers={"content-type": "application/json"},
            content=_jsonrpc_body(request_id),
        )

    async def aclose(self) -> None:
        self.events.append("close")
        if self.client_close_failure is not None:
            raise self.client_close_failure


def test_http_rejects_conflicting_legacy_and_limits_configuration() -> None:
    with pytest.raises(ValueError, match="cannot be combined"):
        HttpMcpClient(timeout_s=1, transport_limits=_limits())

    async def connect_with_metadata_override() -> None:
        await HttpMcpClient(
            transport=httpx.MockTransport(lambda request: httpx.Response(500)),
            transport_limits=_limits(),
        ).connect(
            McpServerSpec(
                name="conflicting-http",
                url="https://mcp.example/rpc",
                metadata={"timeout": 1},
            )
        )

    with pytest.raises(ValueError, match="metadata.timeout"):
        asyncio.run(connect_with_metadata_override())

    async def connect_with_nonfinite_metadata_timeout(timeout: float) -> None:
        await HttpMcpClient(
            transport=httpx.MockTransport(lambda request: httpx.Response(500)),
        ).connect(
            McpServerSpec(
                name="invalid-http-timeout",
                url="https://mcp.example/rpc",
                metadata={"timeout": timeout},
            )
        )

    for timeout in (float("nan"), float("inf")):
        with pytest.raises(ValueError, match="finite"):
            asyncio.run(connect_with_nonfinite_metadata_timeout(timeout))


def test_toolset_discovery_preserves_typed_timeout_after_secret_redaction() -> None:
    secret = "typed-discovery-secret-canary"
    blocked = _BlockingStream()

    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "DELETE":
            return httpx.Response(200)
        payload = json.loads(request.content)
        method = payload["method"]
        if method == "initialize":
            return httpx.Response(
                200,
                headers={"content-type": "application/json"},
                content=_initialize_body(payload["id"]),
            )
        if method == "notifications/initialized":
            return httpx.Response(202)
        assert method == "tools/list"
        return httpx.Response(
            200,
            headers={"content-type": "application/json"},
            stream=blocked,
        )

    async def run() -> BaseException:
        client = HttpMcpClient(
            transport=httpx.MockTransport(handler),
            transport_limits=_limits(idle_timeout_s=0.02, total_call_timeout_s=0.2),
            secret_resolver=StaticVault({"token": secret}),
        )
        with pytest.raises(McpIdleTimeoutError) as exc_info:
            await connect_mcp_toolset(
                McpServerSpec(
                    name="typed-discovery",
                    url="https://mcp.example/rpc",
                    secret_headers={"authorization": SecretRef(name="token")},
                ),
                client=client,
            )
        settlement = _http_settlement_task(exc_info.value)
        assert settlement is not None
        await settlement
        return exc_info.value

    error = asyncio.run(run())

    assert type(error) is McpIdleTimeoutError
    assert secret not in str(error)


@pytest.mark.parametrize("response_kind", ["json", "sse"])
@pytest.mark.parametrize("over_limit", [False, True])
def test_http_initialized_notification_response_has_exact_message_limit(
    response_kind: str,
    over_limit: bool,
) -> None:
    max_message_bytes = 256
    calls: list[str] = []
    if response_kind == "sse":
        event_bytes = max_message_bytes + int(over_limit)
        line = b"data:" + b"x" * (event_bytes - len(b"data:") - 1)
        notification_body = line + b"\n\n"
        content_type = "text/event-stream"
    else:
        notification_body = b"x" * (max_message_bytes + int(over_limit))
        content_type = "application/json"

    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        method = payload["method"]
        calls.append(method)
        if method == "initialize":
            return httpx.Response(
                200,
                headers={"content-type": "application/json"},
                content=_initialize_body(payload["id"]),
            )
        assert method == "notifications/initialized"
        assert request.headers[MCP_PROTOCOL_VERSION_HEADER] == MCP_PROTOCOL_VERSION
        return httpx.Response(
            200,
            headers={"content-type": content_type},
            stream=_ChunkStream([notification_body]),
        )

    async def run() -> bool:
        client = HttpMcpClient(
            transport=httpx.MockTransport(handler),
            transport_limits=_limits(
                max_message_bytes=max_message_bytes,
                max_response_bytes=1_024,
            ),
        )
        if over_limit:
            with pytest.raises(McpMessageTooLargeError) as exc_info:
                await client.connect(
                    McpServerSpec(name="bounded-notification", url="https://mcp.example/rpc")
                )
            settlement = _http_settlement_task(exc_info.value)
            assert settlement is not None
            await settlement
            return False
        session = await client.connect(
            McpServerSpec(name="bounded-notification", url="https://mcp.example/rpc")
        )
        await session.close()
        return True

    assert asyncio.run(run()) is (not over_limit)
    assert calls == ["initialize", "notifications/initialized"]


def test_http_notification_overflow_never_commits_direct_session_initialization() -> None:
    max_message_bytes = 256
    delete_headers: httpx.Headers | None = None

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal delete_headers
        if request.method == "DELETE":
            delete_headers = request.headers
            return httpx.Response(200)
        payload = json.loads(request.content)
        if payload["method"] == "initialize":
            return httpx.Response(
                200,
                headers={
                    "content-type": "application/json",
                    MCP_SESSION_ID_HEADER: "bounded-notification-session",
                },
                content=_initialize_body(payload["id"]),
            )
        return httpx.Response(
            200,
            headers={"content-type": "application/json"},
            content=b"x" * (max_message_bytes + 1),
        )

    async def run() -> tuple[bool, bool]:
        session = _session(
            handler,
            limits=_limits(
                max_message_bytes=max_message_bytes,
                max_response_bytes=1_024,
            ),
        )
        with pytest.raises(McpMessageTooLargeError) as exc_info:
            await session.initialize()
        settlement = _http_settlement_task(exc_info.value)
        assert settlement is not None
        await settlement
        with pytest.raises(McpProtocolError, match="not been initialized"):
            _ = session.initialize_result
        with pytest.raises(McpProtocolError, match="closed"):
            await session.list_tools()
        return session._closed, session._http.is_closed

    session_closed, client_closed = asyncio.run(run())

    assert session_closed is True
    assert client_closed is True
    assert delete_headers is not None
    assert delete_headers[MCP_SESSION_ID_HEADER] == "bounded-notification-session"
    assert delete_headers[MCP_PROTOCOL_VERSION_HEADER] == MCP_PROTOCOL_VERSION


@pytest.mark.parametrize("over_limit", [False, True])
def test_http_json_message_limit_has_an_exact_boundary(over_limit: bool) -> None:
    body = _jsonrpc_body(1, padding=300)
    max_message_bytes = len(body) - int(over_limit)

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["accept-encoding"] == "identity"
        return httpx.Response(
            200,
            headers={"content-type": "application/json"},
            stream=_ChunkStream([body[: len(body) // 2], body[len(body) // 2 :]]),
        )

    async def run() -> bool:
        session = _session(
            handler,
            limits=_limits(
                max_message_bytes=max_message_bytes,
                max_response_bytes=len(body),
            ),
        )
        try:
            if over_limit:
                with pytest.raises(McpMessageTooLargeError):
                    await session.call_tool("echo", {})
                return False
            result = await session.call_tool("echo", {})
            return len(result.content[0]["text"]) == 300
        finally:
            await session.close()

    assert asyncio.run(run()) is (not over_limit)


@pytest.mark.parametrize("status_code", [400, 404])
@pytest.mark.parametrize("over_limit", [False, True])
def test_http_json_error_message_limit_has_an_exact_boundary(
    status_code: int,
    over_limit: bool,
) -> None:
    error_body = json.dumps(
        {
            "jsonrpc": "2.0",
            "error": {"code": -32000, "message": "x" * 300},
        },
        separators=(",", ":"),
    ).encode()
    max_message_bytes = len(error_body) - int(over_limit)
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        if request.method == "DELETE":
            return httpx.Response(200)
        calls += 1
        if calls == 1:
            midpoint = len(error_body) // 2
            return httpx.Response(
                status_code,
                headers={"content-type": "application/json; charset=utf-8"},
                stream=_ChunkStream([error_body[:midpoint], error_body[midpoint:]]),
            )
        return httpx.Response(
            200,
            headers={"content-type": "application/json"},
            content=_jsonrpc_body(2),
        )

    async def run() -> tuple[type[BaseException], bool]:
        session = _session(
            handler,
            limits=_limits(
                max_message_bytes=max_message_bytes,
                max_response_bytes=len(error_body),
            ),
        )
        try:
            with pytest.raises(McpProtocolError) as exc_info:
                await session.call_tool("first", {})
            if status_code == 404:
                assert session._closed is True
                with pytest.raises(McpProtocolError, match="closed"):
                    await session.call_tool("second", {})
            else:
                result = await session.call_tool("second", {})
                assert result.content[0]["text"] == ""
            return type(exc_info.value), session._closed
        finally:
            await session.close()

    error_type, session_closed = asyncio.run(run())

    assert error_type is (McpMessageTooLargeError if over_limit else McpProtocolError)
    assert session_closed is (status_code == 404)
    assert calls == (1 if status_code == 404 else 2)


@pytest.mark.parametrize("status_code", [400, 404])
@pytest.mark.parametrize("over_limit", [False, True])
def test_http_sse_error_event_limit_has_an_exact_boundary(
    status_code: int,
    over_limit: bool,
) -> None:
    event = b"data: " + _jsonrpc_body(1, padding=300) + b"\n"
    body = event + b"\n"
    max_message_bytes = len(event) - int(over_limit)
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls > 1:
            return httpx.Response(
                200,
                headers={"content-type": "application/json"},
                content=_jsonrpc_body(2),
            )
        midpoint = len(body) // 2
        return httpx.Response(
            status_code,
            headers={"content-type": "text/event-stream; charset=utf-8"},
            stream=_ChunkStream([body[:midpoint], body[midpoint:]]),
        )

    async def run() -> tuple[type[BaseException], bool]:
        session = _session(
            handler,
            limits=_limits(
                max_message_bytes=max_message_bytes,
                max_response_bytes=len(body),
            ),
        )
        try:
            with pytest.raises(McpProtocolError) as exc_info:
                await session.call_tool("echo", {})
            if status_code == 404:
                with pytest.raises(McpProtocolError, match="closed"):
                    await session.call_tool("second", {})
            else:
                result = await session.call_tool("second", {})
                assert result.content[0]["text"] == ""
            return type(exc_info.value), session._closed
        finally:
            await session.close()

    error_type, session_closed = asyncio.run(run())

    assert error_type is (McpMessageTooLargeError if over_limit else McpProtocolError)
    assert session_closed is (status_code == 404)
    assert calls == (1 if status_code == 404 else 2)


def test_http_truncated_sse_error_omits_partial_secret_diagnostic(
    caplog: pytest.LogCaptureFixture,
    capsys: pytest.CaptureFixture[str],
    recwarn: pytest.WarningsRecorder,
) -> None:
    secret = "mcp-http-sse-cross-boundary-secret-canary"
    secret_bytes = secret.encode()
    max_message_bytes = 128
    first_event = b"data: x\n\n"
    second_prefix = b"data: "
    exposed_prefix_bytes = len(secret_bytes) - 4
    padding_size = max_message_bytes - len(first_event) - len(second_prefix) - exposed_prefix_bytes
    assert padding_size >= 0
    second_event = second_prefix + b"y" * padding_size + secret_bytes + b"\n\n"
    assert len(second_event.rstrip(b"\n")) <= max_message_bytes
    body = first_event + second_event
    assert body[:max_message_bytes].endswith(secret_bytes[:exposed_prefix_bytes])
    stream = _ChunkStream([first_event, second_event])
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls == 1:
            return httpx.Response(
                400,
                headers={"content-type": "text/event-stream"},
                stream=stream,
            )
        return httpx.Response(
            200,
            headers={"content-type": "application/json"},
            content=_jsonrpc_body(2),
        )

    async def run() -> tuple[BaseException, bool]:
        session = _session(
            handler,
            limits=_limits(
                max_message_bytes=max_message_bytes,
                max_response_bytes=len(body),
            ),
            secret_redactor=SecretRedactor(secret),
        )
        try:
            with pytest.raises(McpProtocolError) as exc_info:
                await session.call_tool("first", {})
            result = await session.call_tool("second", {})
            assert result.content[0]["text"] == ""
            return exc_info.value, session._closed
        finally:
            await session.close()

    with caplog.at_level(logging.DEBUG):
        error, session_closed = asyncio.run(run())

    exposed_prefix = secret[:-4]
    assert type(error) is McpProtocolError
    assert "safe retention was incomplete" in str(error)
    assert secret not in repr(error)
    assert exposed_prefix not in repr(error)
    assert secret not in "".join(traceback.format_exception(error))
    assert exposed_prefix not in "".join(traceback.format_exception(error))
    _assert_cayu_traceback_does_not_retain(error, secret, exposed_prefix)
    captured = capsys.readouterr()
    assert secret not in captured.out
    assert secret not in captured.err
    assert exposed_prefix not in captured.out
    assert exposed_prefix not in captured.err
    assert all(secret not in record.getMessage() for record in caplog.records)
    assert all(exposed_prefix not in record.getMessage() for record in caplog.records)
    assert all(secret not in str(warning.message) for warning in recwarn)
    assert all(exposed_prefix not in str(warning.message) for warning in recwarn)
    assert stream.close_calls == 1
    assert session_closed is False
    assert calls == 2


def test_http_oversized_outbound_is_rejected_before_serialization(monkeypatch) -> None:
    class SerializationMustNotRun:
        @staticmethod
        def dumps(*args, **kwargs):
            raise AssertionError("oversized request reached json.dumps")

    async def run() -> None:
        session = _session(
            lambda request: pytest.fail("oversized request was dispatched"),
            limits=_limits(max_message_bytes=256),
        )
        try:
            monkeypatch.setattr(mcp_http_module, "json", SerializationMustNotRun)
            with pytest.raises(McpMessageTooLargeError):
                await session.call_tool("echo", {"text": "x" * 1_000})
        finally:
            await session.close()

    asyncio.run(run())


@pytest.mark.parametrize("line_ending", [b"\n", b"\r\n"])
@pytest.mark.parametrize("over_limit", [False, True])
def test_http_sse_event_limit_counts_exact_wire_framing(
    over_limit: bool,
    line_ending: bytes,
) -> None:
    message = _jsonrpc_body(1, padding=300)
    event = b"event: message" + line_ending + b"data: " + message + line_ending
    body = event + line_ending
    max_message_bytes = len(event) - int(over_limit)

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            stream=_ChunkStream([body[: len(body) // 2], body[len(body) // 2 :]]),
        )

    async def run() -> bool:
        session = _session(
            handler,
            limits=_limits(
                max_message_bytes=max_message_bytes,
                max_response_bytes=len(body),
            ),
        )
        try:
            if over_limit:
                with pytest.raises(McpMessageTooLargeError):
                    await session.call_tool("echo", {})
                return False
            result = await session.call_tool("echo", {})
            return len(result.content[0]["text"]) == 300
        finally:
            await session.close()

    assert asyncio.run(run()) is (not over_limit)


def test_http_sse_standalone_cr_dispatches_without_waiting_for_another_byte() -> None:
    body = b"data: " + _jsonrpc_body(1) + b"\r\r"
    stream = _ChunkThenBlocksStream(body)

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            stream=stream,
        )

    async def run() -> tuple[bool, bool]:
        session = _session(handler, limits=_limits(idle_timeout_s=0.02))
        try:
            result = await session.call_tool("echo", {})
            return result.is_error, stream.blocked.is_set()
        finally:
            await session.close()

    is_error, waited_for_another_byte = asyncio.run(run())

    assert is_error is False
    assert waited_for_another_byte is False


@pytest.mark.parametrize("line_ending", [b"", b"\n", b"\r", b"\r\n"])
def test_http_sse_peer_eof_does_not_dispatch_unfinished_event(line_ending: bytes) -> None:
    body = b"data: " + _jsonrpc_body(1) + line_ending

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={
                "content-type": "text/event-stream",
                "mcp-session-id": "uncommitted-session",
            },
            stream=_ChunkStream([body]),
        )

    async def run() -> tuple[bool, bool, str | None]:
        session = _session(handler, limits=_limits())
        try:
            with pytest.raises(McpPeerClosedError) as exc_info:
                await session.call_tool("echo", {})
            settlement = _http_settlement_task(exc_info.value)
            assert settlement is not None
            await settlement
            return session._closed, session._http.is_closed, session._session_id
        finally:
            await session.close()

    session_closed, client_closed, session_id = asyncio.run(run())

    assert session_closed is True
    assert client_closed is True
    assert session_id is None


def test_http_later_discovery_page_timeout_preserves_exact_settlement() -> None:
    cursor = "private-pagination-cursor-canary"
    blocked = _BlockingCleanupFailsStream(RuntimeError(f"pagination cleanup exposed {cursor}"))
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        payload = json.loads(request.content)
        if calls == 1:
            body = json.dumps(
                {
                    "jsonrpc": "2.0",
                    "id": payload["id"],
                    "result": {"tools": [], "nextCursor": cursor},
                },
                separators=(",", ":"),
            ).encode()
            return httpx.Response(
                200,
                headers={"content-type": "application/json"},
                content=body,
            )
        assert payload["params"] == {"cursor": cursor}
        return httpx.Response(
            200,
            headers={"content-type": "application/json"},
            stream=blocked,
        )

    async def run() -> tuple[BaseException, BaseException]:
        session = _session(
            handler,
            limits=_limits(idle_timeout_s=0.02, total_call_timeout_s=0.2),
        )
        try:
            with pytest.raises(McpIdleTimeoutError) as exc_info:
                await session.list_tools()
            settlement = _http_settlement_task(exc_info.value)
            assert settlement is not None
            with pytest.raises(McpProtocolError) as settlement_exc:
                await settlement
            return exc_info.value, settlement_exc.value
        finally:
            await session.close()

    error, settlement_error = asyncio.run(run())

    assert cursor not in str(error)
    assert cursor not in "".join(traceback.format_exception(settlement_error))
    assert REDACTED_SECRET in str(settlement_error)
    assert calls == 2


@pytest.mark.parametrize(
    ("method", "result_key", "identity_key", "list_method", "mapping_attribute"),
    [
        ("tools/list", "tools", "name", "list_tools", "_tool_transport_names"),
        (
            "resources/list",
            "resources",
            "uri",
            "list_resources",
            "_resource_transport_uris",
        ),
    ],
)
def test_http_later_discovery_failure_clears_partial_page_state(
    method: str,
    result_key: str,
    identity_key: str,
    list_method: str,
    mapping_attribute: str,
) -> None:
    secret = f"mcp-http-partial-{result_key}-authority-secret"
    page_canary = f"mcp-http-partial-{result_key}-page-canary"
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        payload = json.loads(request.content)
        assert payload["method"] == method
        if calls == 1:
            item = {
                identity_key: secret,
                "description": page_canary,
            }
            if result_key == "tools":
                item["inputSchema"] = {"type": "object"}
            body = json.dumps(
                {
                    "jsonrpc": "2.0",
                    "id": payload["id"],
                    "result": {
                        result_key: [item],
                        "nextCursor": "failing-page",
                    },
                },
                separators=(",", ":"),
            ).encode()
            return httpx.Response(
                200,
                headers={"content-type": "application/json"},
                content=body,
            )
        return httpx.Response(
            200,
            headers={"content-type": "application/json"},
            content=b"{invalid-json",
        )

    async def run() -> tuple[BaseException, dict[str, str]]:
        session = _session(
            handler,
            limits=_limits(),
            secret_redactor=SecretRedactor(secret),
        )
        try:
            with pytest.raises(McpProtocolError) as exc_info:
                await getattr(session, list_method)()
            mapping = dict(getattr(session, mapping_attribute))
            return exc_info.value, mapping
        finally:
            await session.close()

    error, mapping = asyncio.run(run())

    assert calls == 2
    assert mapping == {}
    _assert_cayu_traceback_does_not_retain(error, secret, page_canary)


def test_http_paginated_cancellation_preserves_exact_settlement() -> None:
    blocked = _BlockingStream()

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"content-type": "application/json"},
            stream=blocked,
        )

    async def run() -> tuple[int, bool]:
        session = _session(handler, limits=_limits())
        try:
            task = asyncio.create_task(session.list_resources())
            await blocked.started.wait()
            task.cancel("cancel paginated resources")
            cancelling = task.cancelling()
            with pytest.raises(asyncio.CancelledError, match="cancel paginated resources") as exc:
                await task
            settlement = _http_settlement_task(exc.value)
            assert settlement is not None
            await settlement
            return cancelling, task.cancelled()
        finally:
            await session.close()

    cancelling, cancelled = asyncio.run(run())

    assert cancelling == 1
    assert cancelled is True


def test_http_sse_multiline_event_rejects_overflow_before_waiting_for_line_end() -> None:
    first_line = b":" + (b"x" * 300) + b"\n"
    unfinished_data_line = b"data: " + (b"y" * 20)
    body = first_line + unfinished_data_line
    max_message_bytes = len(body) - 1
    stream = _ChunkThenBlocksStream(body)

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            stream=stream,
        )

    async def run() -> tuple[bool, bool]:
        session = _session(
            handler,
            limits=_limits(
                max_message_bytes=max_message_bytes,
                max_response_bytes=len(body),
                idle_timeout_s=0.02,
            ),
        )
        try:
            with pytest.raises(McpMessageTooLargeError):
                await session.call_tool("echo", {})
            return stream.blocked.is_set(), stream.closed
        finally:
            await session.close()

    waited_for_more_bytes, response_closed = asyncio.run(run())

    assert waited_for_more_bytes is False
    assert response_closed is True


@pytest.mark.parametrize("over_limit", [False, True])
def test_http_sse_aggregate_response_limit_has_an_exact_boundary(over_limit: bool) -> None:
    notification = json.dumps(
        {
            "jsonrpc": "2.0",
            "method": "notifications/progress",
            "params": {"padding": "n" * 300},
        },
        separators=(",", ":"),
    ).encode()
    response = _jsonrpc_body(1, padding=300)
    first_event = b"data: " + notification + b"\n"
    second_event = b"data: " + response + b"\n"
    body = first_event + b"\n" + second_event + b"\n"
    max_event_bytes = max(len(first_event), len(second_event))

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            stream=_ChunkStream([first_event, b"\n", second_event, b"\n"]),
        )

    async def run() -> bool:
        session = _session(
            handler,
            limits=_limits(
                max_message_bytes=max_event_bytes,
                max_response_bytes=len(body) - int(over_limit),
            ),
        )
        try:
            if over_limit:
                with pytest.raises(McpResponseTooLargeError):
                    await session.call_tool("echo", {})
                return False
            result = await session.call_tool("echo", {})
            return len(result.content[0]["text"]) == 300
        finally:
            await session.close()

    assert asyncio.run(run()) is (not over_limit)


def test_http_message_overflow_is_isolated_and_does_not_commit_session_id() -> None:
    calls = 0
    oversized = _jsonrpc_body(1, padding=500)
    valid = _jsonrpc_body(2)

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        if request.method == "DELETE":
            return httpx.Response(200)
        calls += 1
        return httpx.Response(
            200,
            headers={
                "content-type": "application/json",
                "mcp-session-id": "unvalidated-session",
            },
            content=oversized if calls == 1 else valid,
        )

    async def run() -> tuple[str | None, bool, bool]:
        session = _session(
            handler,
            limits=_limits(max_message_bytes=256, max_response_bytes=1_024),
        )
        try:
            with pytest.raises(McpMessageTooLargeError):
                await session.call_tool("echo", {})
            session_id_after_rejection = session._session_id
            result = await session.call_tool("echo", {})
            return session_id_after_rejection, session._closed, result.is_error
        finally:
            await session.close()

    session_id, closed, is_error = asyncio.run(run())

    assert session_id is None
    assert closed is False
    assert is_error is False
    assert calls == 2


def test_http_rejected_tool_result_does_not_replace_session_id() -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        if request.method == "DELETE":
            return httpx.Response(200)
        calls += 1
        assert request.headers[MCP_SESSION_ID_HEADER] == "established-session"
        request_id = json.loads(request.content)["id"]
        if calls == 1:
            return httpx.Response(
                200,
                headers={
                    "content-type": "application/json",
                    MCP_SESSION_ID_HEADER: "rejected-replacement",
                },
                content=json.dumps(
                    {
                        "jsonrpc": "2.0",
                        "id": request_id,
                        "result": {"content": "not-a-list"},
                    }
                ).encode(),
            )
        return httpx.Response(
            200,
            headers={"content-type": "application/json"},
            content=_jsonrpc_body(request_id),
        )

    async def run() -> tuple[str | None, bool]:
        session = _session(handler, limits=_limits())
        session._session_id = "established-session"
        try:
            with pytest.raises(McpProtocolError, match="invalid data"):
                await session.call_tool("echo", {})
            session_id_after_rejection = session._session_id
            result = await session.call_tool("echo", {})
            return session_id_after_rejection, result.is_error
        finally:
            await session.close()

    session_id, is_error = asyncio.run(run())

    assert session_id == "established-session"
    assert is_error is False
    assert calls == 2


def test_http_concurrent_overflow_is_isolated_from_sibling_call() -> None:
    both_dispatched = asyncio.Event()
    dispatched = 0

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal dispatched
        request_id = json.loads(request.content)["id"]
        dispatched += 1
        if dispatched == 2:
            both_dispatched.set()
        await both_dispatched.wait()
        return httpx.Response(
            200,
            headers={"content-type": "application/json"},
            content=_jsonrpc_body(request_id, padding=500 if request_id == 1 else 0),
        )

    async def run() -> tuple[list[BaseException], list[bool], bool]:
        http_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        session = HttpMcpSession(
            server=McpServerSpec(name="concurrent", url="https://mcp.example/rpc"),
            http_client=http_client,
            url="https://mcp.example/rpc",
            client_name="cayu-test",
            client_version="1",
            transport_limits=_limits(max_message_bytes=256, max_response_bytes=1_024),
        )
        try:
            outcomes = await asyncio.gather(
                session.call_tool("first", {}),
                session.call_tool("second", {}),
                return_exceptions=True,
            )
            return (
                [item for item in outcomes if isinstance(item, BaseException)],
                [item.is_error for item in outcomes if not isinstance(item, BaseException)],
                session._closed,
            )
        finally:
            await session.close()

    errors, result_errors, session_closed = asyncio.run(run())

    assert len(errors) == 1
    assert isinstance(errors[0], McpMessageTooLargeError)
    assert result_errors == [False]
    assert session_closed is False


def test_http_oversized_outbound_is_rejected_before_copy_or_dispatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = 0

    def payload_copy_must_not_run(*args: Any, **kwargs: Any) -> dict[str, Any]:
        raise AssertionError("oversized request reached defensive copying")

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(
            200,
            headers={"content-type": "application/json"},
            content=_jsonrpc_body(2),
        )

    async def run() -> bool:
        session = _session(
            handler,
            limits=_limits(max_message_bytes=256, max_response_bytes=1_024),
        )
        try:
            with monkeypatch.context() as patcher:
                patcher.setattr(
                    mcp_http_module,
                    "jsonrpc_request_payload",
                    payload_copy_must_not_run,
                )
                with pytest.raises(McpMessageTooLargeError):
                    await session.call_tool("echo", {"text": "x" * 500})
            result = await session.call_tool("echo", {})
            return result.is_error
        finally:
            await session.close()

    assert asyncio.run(run()) is False
    assert calls == 1


def test_http_adapter_outbound_overflow_drops_known_secret_from_diagnostics(
    caplog: pytest.LogCaptureFixture,
    capsys: pytest.CaptureFixture[str],
    recwarn: pytest.WarningsRecorder,
) -> None:
    secret = "mcp-http-outbound-overflow-secret-canary"
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(500)

    async def run() -> BaseException:
        session = _session(
            handler,
            limits=_limits(max_message_bytes=256, max_response_bytes=1_024),
            secret_redactor=SecretRedactor(secret),
        )
        session._initialize_result = McpInitializeResult(protocol_version=MCP_PROTOCOL_VERSION)
        definition = McpToolDefinition(name="echo", input_schema={"type": "object"})
        toolset = McpToolset(
            server=session.server,
            session=session,
            definitions=(definition,),
        )
        try:
            with pytest.raises(McpMessageTooLargeError) as exc_info:
                await toolset.tools[0].run(
                    ToolContext(session_id="outbound-overflow", agent_name="test"),
                    {"text": (secret + "-") * 20},
                )
            return exc_info.value
        finally:
            await toolset.close()

    with caplog.at_level(logging.DEBUG):
        error = asyncio.run(run())

    assert secret not in "".join(traceback.format_exception(error))
    _assert_cayu_traceback_does_not_retain(error, secret)
    captured = capsys.readouterr()
    assert secret not in captured.out
    assert secret not in captured.err
    assert all(secret not in record.getMessage() for record in caplog.records)
    assert all(secret not in str(warning.message) for warning in recwarn)
    assert calls == 0


def test_http_adapter_rejects_oversized_arguments_before_defensive_copy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        request_id = json.loads(request.content)["id"]
        return httpx.Response(
            200,
            headers={"content-type": "application/json"},
            content=_jsonrpc_body(request_id),
        )

    async def run() -> bool:
        session = _session(
            handler,
            limits=_limits(max_message_bytes=256, max_response_bytes=1_024),
        )
        session._initialize_result = McpInitializeResult(protocol_version=MCP_PROTOCOL_VERSION)
        definition = McpToolDefinition(name="echo", input_schema={"type": "object"})
        toolset = McpToolset(server=session.server, session=session, definitions=(definition,))
        original_copy = mcp_http_module.copy_json_value

        def copied_request_must_fit(value: Any, field_name: str) -> Any:
            if field_name == "params":
                raise AssertionError("oversized adapter arguments reached defensive copying")
            return original_copy(value, field_name)

        def adapter_must_delegate_nesting_preflight(_value: Any) -> bool:
            raise AssertionError("built-in adapter performed an unbounded nesting scan")

        try:
            with monkeypatch.context() as patcher:
                patcher.setattr(mcp_http_module, "copy_json_value", copied_request_must_fit)
                patcher.setattr(mcp_tools_module, "copy_json_value", copied_request_must_fit)
                patcher.setattr(
                    mcp_tools_module,
                    "mcp_json_value_nesting_too_deep",
                    adapter_must_delegate_nesting_preflight,
                )
                with pytest.raises(McpMessageTooLargeError):
                    await toolset.tools[0].run(
                        ToolContext(session_id="adapter-overflow", agent_name="test"),
                        {"values": ["x" * 16 for _ in range(100)]},
                    )
            request_id = session._next_id
            payload = {
                "jsonrpc": "2.0",
                "id": request_id,
                "method": "tools/call",
                "params": {"name": "echo", "arguments": {"padding": ""}},
            }
            encoded = json.dumps(payload, separators=(",", ":")).encode()
            exact_arguments = {"padding": "x" * (256 - len(encoded))}
            result = await toolset.tools[0].run(
                ToolContext(session_id="adapter-boundary", agent_name="test"),
                exact_arguments,
            )
            return result.is_error
        finally:
            await toolset.close()

    assert asyncio.run(run()) is False
    assert calls == 1


@pytest.mark.parametrize("through_adapter", [False, True])
def test_http_invalid_outbound_arguments_are_typed_and_redacted(
    caplog: pytest.LogCaptureFixture,
    capsys: pytest.CaptureFixture[str],
    recwarn: pytest.WarningsRecorder,
    through_adapter: bool,
) -> None:
    secret = "mcp-http-invalid-outbound-secret-canary"
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        request_id = json.loads(request.content)["id"]
        return httpx.Response(
            200,
            headers={"content-type": "application/json"},
            content=_jsonrpc_body(request_id),
        )

    async def run() -> tuple[BaseException, bool]:
        session = _session(
            handler,
            limits=_limits(),
            secret_redactor=SecretRedactor(secret),
        )
        session._initialize_result = McpInitializeResult(protocol_version=MCP_PROTOCOL_VERSION)
        definition = McpToolDefinition(name="echo", input_schema={"type": "object"})
        toolset = McpToolset(server=session.server, session=session, definitions=(definition,))
        try:
            with pytest.raises(McpProtocolError) as exc_info:
                if through_adapter:
                    await toolset.tools[0].run(
                        ToolContext(session_id="invalid-outbound", agent_name="test"),
                        {secret: object()},
                    )
                else:
                    await session.call_tool("echo", {secret: object()})
            if through_adapter:
                result = await toolset.tools[0].run(
                    ToolContext(session_id="valid-outbound", agent_name="test"),
                    {},
                )
            else:
                result = await session.call_tool("echo", {})
            return exc_info.value, result.is_error
        finally:
            await toolset.close()

    with caplog.at_level(logging.DEBUG):
        error, result_is_error = asyncio.run(run())

    assert result_is_error is False
    assert calls == 1
    assert secret not in "".join(traceback.format_exception(error))
    _assert_cayu_traceback_does_not_retain(error, secret)
    captured = capsys.readouterr()
    assert secret not in captured.out
    assert secret not in captured.err
    assert all(secret not in record.getMessage() for record in caplog.records)
    assert all(secret not in str(warning.message) for warning in recwarn)


def test_http_large_integer_preflight_returns_typed_overflow_without_dispatch() -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(
            200,
            headers={"content-type": "application/json"},
            content=_jsonrpc_body(2),
        )

    async def run() -> bool:
        session = _session(
            handler,
            limits=_limits(max_message_bytes=256, max_response_bytes=1_024),
        )
        try:
            with pytest.raises(McpMessageTooLargeError):
                await session.call_tool("echo", {"value": 10**5_000})
            result = await session.call_tool("echo", {})
            return result.is_error
        finally:
            await session.close()

    assert asyncio.run(run()) is False
    assert calls == 1


def test_http_invalid_circular_arguments_keep_validation_error_and_session_reusable() -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        request_id = json.loads(request.content)["id"]
        return httpx.Response(
            200,
            headers={"content-type": "application/json"},
            content=_jsonrpc_body(request_id),
        )

    async def run() -> bool:
        session = _session(
            handler,
            limits=_limits(max_message_bytes=256, max_response_bytes=1_024),
        )
        arguments: dict[str, Any] = {}
        arguments["self"] = arguments
        try:
            with pytest.raises(McpProtocolError, match="circular references"):
                await session.call_tool("echo", arguments)
            result = await session.call_tool("echo", {})
            return result.is_error
        finally:
            arguments.clear()
            await session.close()

    assert asyncio.run(run()) is False
    assert calls == 1


@pytest.mark.parametrize(
    ("max_message_bytes", "error_type"),
    [
        (256, McpMessageTooLargeError),
        (8_192, McpProtocolError),
    ],
)
def test_http_deep_outbound_is_safely_rejected_without_dispatch_and_session_reuses(
    max_message_bytes: int,
    error_type: type[McpProtocolError],
    caplog: pytest.LogCaptureFixture,
    capsys: pytest.CaptureFixture[str],
    recwarn: pytest.WarningsRecorder,
) -> None:
    secret = f"mcp-http-deep-{max_message_bytes}-secret-canary"
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        request_id = json.loads(request.content)["id"]
        return httpx.Response(
            200,
            headers={"content-type": "application/json"},
            content=_jsonrpc_body(request_id),
        )

    async def run() -> tuple[BaseException, bool, bool]:
        session = _session(
            handler,
            limits=_limits(
                max_message_bytes=max_message_bytes,
                max_response_bytes=max(16_384, max_message_bytes),
            ),
        )
        arguments = _deep_json_arguments(1_500, secret)
        try:
            with pytest.raises(error_type) as exc_info:
                await session.call_tool("echo", arguments)
            result = await session.call_tool("echo", {})
            return exc_info.value, result.is_error, session._closed
        finally:
            arguments.clear()
            await session.close()

    with caplog.at_level(logging.DEBUG):
        error, result_is_error, session_closed = asyncio.run(run())

    assert calls == 1
    assert result_is_error is False
    assert session_closed is False
    assert secret not in "".join(traceback.format_exception(error))
    assert error.__cause__ is None
    assert error.__context__ is None
    _assert_cayu_traceback_does_not_retain(error, secret)
    captured = capsys.readouterr()
    assert secret not in captured.out
    assert secret not in captured.err
    assert all(secret not in record.getMessage() for record in caplog.records)
    assert all(secret not in str(warning.message) for warning in recwarn)


def test_http_deep_outbound_prefers_positive_size_overflow_at_the_depth_boundary() -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(500)

    async def run() -> None:
        session = _session(handler, limits=_limits(max_message_bytes=256))
        arguments = _deep_json_arguments(256, "boundary")
        try:
            with pytest.raises(McpMessageTooLargeError):
                await session.call_tool("echo", arguments)
        finally:
            arguments.clear()
            await session.close()

    asyncio.run(run())
    assert calls == 0


@pytest.mark.parametrize("media_type", ["application/json", "text/event-stream"])
@pytest.mark.parametrize("depth", [300, 1_500])
def test_http_deep_inbound_is_typed_secret_safe_and_keeps_completed_response_reusable(
    media_type: str,
    depth: int,
    caplog: pytest.LogCaptureFixture,
    capsys: pytest.CaptureFixture[str],
    recwarn: pytest.WarningsRecorder,
) -> None:
    secret = f"mcp-http-deep-inbound-{media_type}-{depth}-secret-canary"
    calls = 0
    hostile_stream: _ChunkStream | None = None

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls, hostile_stream
        calls += 1
        request_id = json.loads(request.content)["id"]
        if calls == 1:
            body = _deep_jsonrpc_response_body(
                request_id,
                depth=depth,
                canary=secret,
            )
            if media_type == "text/event-stream":
                body = b"data: " + body + b"\n\n"
            hostile_stream = _ChunkStream([body])
            return httpx.Response(
                200,
                headers={"content-type": media_type},
                stream=hostile_stream,
            )
        return httpx.Response(
            200,
            headers={"content-type": "application/json"},
            content=_jsonrpc_body(request_id),
        )

    async def run() -> tuple[BaseException, bool, bool]:
        session = _session(
            handler,
            limits=_limits(max_message_bytes=8_192, max_response_bytes=16_384),
            secret_redactor=SecretRedactor(secret),
        )
        try:
            with pytest.raises(McpProtocolError, match="supported JSON nesting") as exc_info:
                await session.call_tool("echo", {})
            result = await session.call_tool("echo", {})
            return exc_info.value, result.is_error, session._closed
        finally:
            await session.close()

    with caplog.at_level(logging.DEBUG):
        error, result_is_error, session_closed = asyncio.run(run())

    assert calls == 2
    assert hostile_stream is not None
    assert hostile_stream.closed is True
    assert hostile_stream.close_calls == 1
    assert result_is_error is False
    assert session_closed is False
    assert error.__cause__ is None
    assert error.__context__ is None
    assert secret not in "".join(traceback.format_exception(error))
    _assert_cayu_traceback_does_not_retain(error, secret)
    captured = capsys.readouterr()
    assert secret not in captured.out
    assert secret not in captured.err
    assert all(secret not in record.getMessage() for record in caplog.records)
    assert all(secret not in str(warning.message) for warning in recwarn)


def test_http_idle_timeout_closes_response_and_fences_logical_session() -> None:
    blocked = _BlockingStream()
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls == 1:
            return httpx.Response(
                200,
                headers={"content-type": "application/json"},
                stream=blocked,
            )
        return httpx.Response(
            200,
            headers={"content-type": "application/json"},
            content=_jsonrpc_body(2),
        )

    async def run() -> tuple[bool, bool, bool]:
        session = _session(
            handler,
            limits=_limits(idle_timeout_s=0.02, total_call_timeout_s=0.5),
        )
        try:
            with pytest.raises(McpIdleTimeoutError) as exc_info:
                await session.call_tool("echo", {})
            settlement = _http_settlement_task(exc_info.value)
            assert settlement is not None
            await settlement
            with pytest.raises(McpProtocolError, match="closed"):
                await session.call_tool("echo", {})
            return blocked.closed, session._closed, session._http.is_closed
        finally:
            await session.close()

    response_closed, session_closed, client_closed = asyncio.run(run())

    assert response_closed is True
    assert session_closed is True
    assert client_closed is True


def test_http_failed_response_cleanup_poisons_logical_session() -> None:
    blocked = _ReleaseClosesStream()

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"content-type": "application/json"},
            stream=blocked,
        )

    async def run() -> tuple[bool, bool]:
        session = _session(
            handler,
            limits=_limits(idle_timeout_s=0.02, total_call_timeout_s=0.5),
        )
        try:
            with pytest.raises(McpIdleTimeoutError) as exc_info:
                await session.call_tool("echo", {})
            settlement = _http_settlement_task(exc_info.value)
            assert settlement is not None
            assert settlement.done() is False
            client_close_task = session._client_close_task
            assert client_close_task is not None
            with pytest.raises(McpProtocolError, match="closed"):
                await session.call_tool("echo", {})
            blocked.release_close.set()
            await settlement
            return session._closed, blocked.closed
        finally:
            await session.close()

    session_closed, response_closed = asyncio.run(run())

    assert session_closed is True
    assert response_closed is True


def test_http_cleanup_only_failure_is_preserved_and_fences_session() -> None:
    cleanup_failure = RuntimeError("response cleanup failed")
    stream = _CleanupFailsStream(
        [b"data: " + _jsonrpc_body(1) + b"\n\n"],
        cleanup_failure,
    )

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            stream=stream,
        )

    async def run() -> tuple[BaseException | None, bool, bool]:
        session = _session(handler, limits=_limits())
        try:
            with pytest.raises(McpProtocolError, match="cleanup failed") as exc_info:
                await session.call_tool("echo", {})
            settlement = _http_settlement_task(exc_info.value)
            assert settlement is not None
            await settlement
            return exc_info.value.__cause__, session._closed, session._http.is_closed
        finally:
            await session.close()

    cause, session_closed, client_closed = asyncio.run(run())

    assert isinstance(cause, McpProtocolError)
    assert cause is not cleanup_failure
    assert "response cleanup failed" in str(cause)
    assert session_closed is True
    assert client_closed is True


def test_http_cleanup_failure_is_detached_and_redacted_from_diagnostics(
    caplog: pytest.LogCaptureFixture,
    capsys: pytest.CaptureFixture[str],
    recwarn: pytest.WarningsRecorder,
) -> None:
    secret = "mcp-http-cleanup-secret-canary"
    raw_cleanup_failure = RuntimeError(f"response cleanup exposed {secret}")
    stream = _CleanupFailsStream(
        [b"data: " + _jsonrpc_body(1) + b"\n\n"],
        raw_cleanup_failure,
    )

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            stream=stream,
        )

    async def run() -> BaseException:
        session = HttpMcpSession(
            server=McpServerSpec(name="cleanup-secret", url="https://mcp.example/rpc"),
            http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
            url="https://mcp.example/rpc",
            client_name="cayu-test",
            client_version="1",
            transport_limits=_limits(),
            secret_redactor=SecretRedactor(secret),
        )
        try:
            with pytest.raises(McpProtocolError) as exc_info:
                await session.call_tool("echo", {})
            settlement = _http_settlement_task(exc_info.value)
            assert settlement is not None
            await settlement
            return exc_info.value
        finally:
            await session.close()

    with caplog.at_level(logging.DEBUG):
        error = asyncio.run(run())

    assert error.__cause__ is not raw_cleanup_failure
    assert error.__context__ is None
    assert secret not in "".join(traceback.format_exception(error))
    traceback_cursor = error.__traceback__
    while traceback_cursor is not None:
        if is_cayu_source_filename(traceback_cursor.tb_frame.f_code.co_filename):
            assert all(
                secret not in repr(value) for value in traceback_cursor.tb_frame.f_locals.values()
            )
        traceback_cursor = traceback_cursor.tb_next
    captured = capsys.readouterr()
    assert secret not in captured.out
    assert secret not in captured.err
    assert all(secret not in record.getMessage() for record in caplog.records)
    assert all(secret not in str(warning.message) for warning in recwarn)


def test_http_primary_and_cleanup_failures_preserve_original_order() -> None:
    cleanup_failure = RuntimeError("response cleanup failed after protocol failure")
    stream = _CleanupFailsStream([b"data: not-json\n\n"], cleanup_failure)

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            stream=stream,
        )

    async def run() -> tuple[BaseException, BaseException | None]:
        session = _session(handler, limits=_limits())
        try:
            with pytest.raises(McpProtocolError, match="valid JSON") as exc_info:
                await session.call_tool("echo", {})
            settlement = _http_settlement_task(exc_info.value)
            assert settlement is not None
            await settlement
            return exc_info.value, exc_info.value.__cause__
        finally:
            await session.close()

    primary, cause = asyncio.run(run())

    assert type(primary) is McpProtocolError
    assert isinstance(cause, McpProtocolError)
    assert cause is not cleanup_failure
    assert "response cleanup failed after protocol failure" in str(cause)


def test_http_cancellation_preserves_cleanup_failure_in_settlement() -> None:
    cleanup_failure = RuntimeError("response cleanup failed after cancellation")
    stream = _BlockingCleanupFailsStream(cleanup_failure)

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"content-type": "application/json"},
            stream=stream,
        )

    async def run() -> tuple[int, bool]:
        session = _session(handler, limits=_limits())
        try:
            task = asyncio.create_task(session.call_tool("echo", {}))
            await stream.started.wait()
            task.cancel()
            cancelling = task.cancelling()
            with pytest.raises(asyncio.CancelledError) as exc_info:
                await task
            settlement = _http_settlement_task(exc_info.value)
            assert settlement is not None
            with pytest.raises(McpProtocolError) as settlement_exc:
                await settlement
            assert settlement_exc.value is not cleanup_failure
            assert "response cleanup failed after cancellation" in str(settlement_exc.value)
            return cancelling, task.cancelled()
        finally:
            await session.close()

    cancelling, cancelled = asyncio.run(run())

    assert cancelling == 1
    assert cancelled is True


def test_http_active_stream_still_hits_total_deadline() -> None:
    body = _jsonrpc_body(1, padding=300)
    stream = _ChunkStream([bytes((byte,)) for byte in body], delay_s=0.01)

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"content-type": "application/json"},
            stream=stream,
        )

    async def run() -> tuple[bool, bool, bool]:
        session = _session(
            handler,
            limits=_limits(idle_timeout_s=0.03, total_call_timeout_s=0.08),
        )
        try:
            with pytest.raises(McpCallDeadlineExceededError) as exc_info:
                await session.call_tool("echo", {})
            settlement = _http_settlement_task(exc_info.value)
            assert settlement is not None
            await settlement
            return stream.closed, session._closed, session._http.is_closed
        finally:
            await session.close()

    response_closed, session_closed, client_closed = asyncio.run(run())

    assert response_closed is True
    assert session_closed is True
    assert client_closed is True


@pytest.mark.parametrize(
    "preparation_stage",
    ["request_size_preflight", "payload_copy", "send_size_preflight"],
)
def test_http_pre_dispatch_deadline_scrubs_private_request_and_preserves_reuse(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    capsys: pytest.CaptureFixture[str],
    recwarn: pytest.WarningsRecorder,
    preparation_stage: str,
) -> None:
    secret = f"mcp-http-{preparation_stage}-deadline-secret-canary"
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        request_id = json.loads(request.content)["id"]
        return httpx.Response(
            200,
            headers={"content-type": "application/json"},
            content=_jsonrpc_body(request_id),
        )

    async def run() -> tuple[BaseException, bool, bool]:
        session = _session(
            handler,
            limits=_limits(idle_timeout_s=0.5, total_call_timeout_s=0.02),
            secret_redactor=SecretRedactor(secret),
        )
        session._tool_transport_names = {"echo": secret}
        try:
            with monkeypatch.context() as patcher:
                if preparation_stage == "request_size_preflight":
                    original_preparation = mcp_http_module.mcp_jsonrpc_request_preflight
                    preparation_name = "mcp_jsonrpc_request_preflight"
                elif preparation_stage == "payload_copy":
                    original_preparation = mcp_http_module.jsonrpc_request_payload
                    preparation_name = "jsonrpc_request_payload"
                else:
                    original_preparation = mcp_http_module.json_utf8_size_within_limit
                    preparation_name = "json_utf8_size_within_limit"

                def delayed_preparation(*args: Any, **kwargs: Any) -> Any:
                    time.sleep(0.04)
                    return original_preparation(*args, **kwargs)

                patcher.setattr(mcp_http_module, preparation_name, delayed_preparation)
                with pytest.raises(McpCallDeadlineExceededError) as exc_info:
                    await session.call_tool("echo", {})
            result = await session.call_tool("echo", {})
            return exc_info.value, result.is_error, session._closed
        finally:
            await session.close()

    with caplog.at_level(logging.DEBUG):
        error, result_is_error, session_closed = asyncio.run(run())

    assert calls == 1
    assert result_is_error is False
    assert session_closed is False
    assert secret not in "".join(traceback.format_exception(error))
    _assert_cayu_traceback_does_not_retain(error, secret)
    captured = capsys.readouterr()
    assert secret not in captured.out
    assert secret not in captured.err
    assert all(secret not in record.getMessage() for record in caplog.records)
    assert all(secret not in str(warning.message) for warning in recwarn)


def test_http_public_initialization_deadline_does_not_wait_for_retained_close(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cleanup_failure = RuntimeError("initialization cleanup failed")
    transport = _BlockingFailingCloseTransport(cleanup_failure)
    original_payload = mcp_http_module.jsonrpc_request_payload

    def delayed_payload(*args: Any, **kwargs: Any) -> dict[str, Any]:
        time.sleep(0.25)
        return original_payload(*args, **kwargs)

    async def run() -> tuple[BaseException, BaseException]:
        with monkeypatch.context() as patcher:
            patcher.setattr(mcp_http_module, "jsonrpc_request_payload", delayed_payload)
            connect_task = asyncio.create_task(
                HttpMcpClient(
                    transport=transport,
                    transport_limits=_limits(
                        idle_timeout_s=0.5,
                        total_call_timeout_s=0.2,
                    ),
                ).connect(McpServerSpec(name="bounded-init", url="https://mcp.example/rpc"))
            )
            await asyncio.sleep(0.3)
        assert connect_task.done() is True
        error = connect_task.exception()
        assert isinstance(error, McpCallDeadlineExceededError)
        settlement = _http_settlement_task(error)
        assert settlement is not None
        assert settlement.done() is False
        transport.release_close.set()
        with pytest.raises(McpProtocolError) as settlement_exc:
            await settlement
        return error, settlement_exc.value

    error, settlement_error = asyncio.run(run())

    assert isinstance(error, McpCallDeadlineExceededError)
    assert "initialization cleanup failed" in str(settlement_error)
    assert transport.close_calls == 1


def test_http_initialization_processing_deadline_retains_session_termination(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cleanup_failure = RuntimeError("post-response initialization cleanup failed")
    transport = _InitializationCleanupFailureTransport(cleanup_failure)
    original_redact = mcp_http_module.safely_redact_jsonrpc_response
    created_sessions: list[HttpMcpSession] = []

    def delayed_redaction(*args: Any, **kwargs: Any) -> Any:
        time.sleep(0.15)
        return original_redact(*args, **kwargs)

    def capture_session(**kwargs: Any) -> HttpMcpSession:
        session = HttpMcpSession(**kwargs)
        created_sessions.append(session)
        return session

    async def run() -> tuple[BaseException, BaseException]:
        with monkeypatch.context() as patcher:
            patcher.setattr(
                mcp_http_module,
                "safely_redact_jsonrpc_response",
                delayed_redaction,
            )
            patcher.setattr(mcp_http_module, "HttpMcpSession", capture_session)
            connect_task = asyncio.create_task(
                HttpMcpClient(
                    transport=transport,
                    transport_limits=_limits(
                        idle_timeout_s=0.5,
                        total_call_timeout_s=0.1,
                    ),
                ).connect(
                    McpServerSpec(
                        name="processing-deadline-init",
                        url="https://mcp.example/rpc",
                    )
                )
            )
            await transport.close_started.wait()
        assert connect_task.done() is True
        error = connect_task.exception()
        assert isinstance(error, McpCallDeadlineExceededError)
        settlement = _http_settlement_task(error)
        assert settlement is not None
        assert settlement.done() is False
        assert len(created_sessions) == 1
        assert created_sessions[0]._closed is True
        assert created_sessions[0]._fenced is True
        with pytest.raises(McpProtocolError, match="not been initialized"):
            _ = created_sessions[0].initialize_result
        assert transport.events == ["initialize", "delete", "close"]
        transport.release_close.set()
        with pytest.raises(McpProtocolError) as settlement_exc:
            await settlement
        return error, settlement_exc.value

    error, settlement_error = asyncio.run(run())

    assert isinstance(error, McpCallDeadlineExceededError)
    assert "post-response initialization cleanup failed" not in str(error)
    assert "post-response initialization cleanup failed" in str(settlement_error)
    assert transport.delete_headers is not None
    assert transport.delete_headers[MCP_SESSION_ID_HEADER] == "processing-deadline-session"
    assert transport.delete_headers[MCP_PROTOCOL_VERSION_HEADER] == MCP_PROTOCOL_VERSION
    assert created_sessions[0]._http.is_closed is True


@pytest.mark.parametrize("depth", [300, 1_500])
def test_http_deep_initialize_response_terminates_staged_server_session(depth: int) -> None:
    secret = "mcp-http-deep-initialize-secret-canary"
    transport = _DeepInitializationTransport(depth=depth, canary=secret)

    async def run() -> BaseException:
        client = HttpMcpClient(
            transport=transport,
            transport_limits=_limits(max_message_bytes=8_192, max_response_bytes=16_384),
            secret_resolver=StaticVault({"token": secret}),
        )
        spec = McpServerSpec(
            name="deep-initialize",
            url="https://mcp.example/rpc",
            secret_headers={"authorization": SecretRef(name="token")},
        )
        with pytest.raises(McpProtocolError, match="supported JSON nesting") as exc_info:
            await client.connect(spec)
        return exc_info.value

    error = asyncio.run(run())

    assert transport.events == ["initialize", "delete", "close"]
    assert transport.delete_headers is not None
    assert transport.delete_headers[MCP_SESSION_ID_HEADER] == "deep-initialize-session"
    if depth == 300:
        assert transport.delete_headers[MCP_PROTOCOL_VERSION_HEADER] == MCP_PROTOCOL_VERSION
    else:
        # Decoder recursion depth is interpreter-dependent. If decoding completed,
        # cleanup carries the allowlisted version; otherwise DELETE has only the
        # independently authoritative HTTP session identifier.
        assert transport.delete_headers.get(MCP_PROTOCOL_VERSION_HEADER) in {
            None,
            MCP_PROTOCOL_VERSION,
        }
    assert secret not in "".join(traceback.format_exception(error))
    _assert_cayu_traceback_does_not_retain(error, secret)


@pytest.mark.parametrize("media_type", ["application/json", "text/event-stream"])
@pytest.mark.parametrize(
    ("rejection", "error_match"),
    [
        ("invalid_envelope", "must use jsonrpc='2.0'"),
        ("mismatched_id", "response id did not match"),
    ],
)
def test_http_rejected_initialize_response_terminates_staged_server_session(
    media_type: str,
    rejection: str,
    error_match: str,
) -> None:
    transport = _RejectedInitializationTransport(
        media_type=media_type,
        rejection=rejection,
    )

    async def run() -> None:
        with pytest.raises(McpProtocolError, match=error_match):
            await HttpMcpClient(
                transport=transport,
                transport_limits=_limits(),
            ).connect(
                McpServerSpec(
                    name="rejected-initialize",
                    url="https://mcp.example/rpc",
                )
            )

    asyncio.run(run())

    assert transport.events == ["initialize", "delete", "close"]
    assert transport.delete_headers is not None
    assert transport.delete_headers[MCP_SESSION_ID_HEADER] == "rejected-initialize-session"
    assert transport.delete_headers[MCP_PROTOCOL_VERSION_HEADER] == MCP_PROTOCOL_VERSION


@pytest.mark.parametrize("delete_fails", [False, True])
def test_http_rejected_initialize_headers_terminate_staged_server_session(
    delete_fails: bool,
) -> None:
    secret = "rejected-initialize-delete-secret-canary"
    transport = _RejectedInitializationHeaderTransport(
        delete_failure=RuntimeError(secret) if delete_fails else None,
    )

    async def run() -> BaseException:
        with pytest.raises(McpProtocolError, match="unsupported content encoding") as exc_info:
            await HttpMcpClient(
                transport=transport,
                transport_limits=_limits(),
                secret_resolver=StaticVault({"token": secret}),
            ).connect(
                McpServerSpec(
                    name="rejected-header-initialize",
                    url="https://mcp.example/rpc",
                    secret_headers={"authorization": SecretRef(name="token")},
                )
            )
        return exc_info.value

    error = asyncio.run(run())

    assert transport.events == ["initialize", "delete", "close"]
    assert transport.delete_headers is not None
    assert transport.delete_headers[MCP_SESSION_ID_HEADER] == "rejected-header-initialize-session"
    assert MCP_PROTOCOL_VERSION_HEADER not in transport.delete_headers
    assert secret not in "".join(traceback.format_exception(error))
    _assert_cayu_traceback_does_not_retain(error, secret)
    if delete_fails:
        assert error.__cause__ is not None
        assert "server-session termination failed" in str(error.__cause__)
        assert REDACTED_SECRET in str(error.__cause__)


def test_http_initialize_sse_server_request_terminates_header_session() -> None:
    transport = _RejectedInitializationTransport(
        media_type="text/event-stream",
        rejection="server_request",
    )

    async def run() -> None:
        with pytest.raises(McpProtocolError, match="does not service MCP server requests"):
            await HttpMcpClient(
                transport=transport,
                transport_limits=_limits(),
            ).connect(
                McpServerSpec(
                    name="server-request-initialize",
                    url="https://mcp.example/rpc",
                )
            )

    asyncio.run(run())

    assert transport.events == ["initialize", "delete", "close"]
    assert transport.delete_headers is not None
    assert transport.delete_headers[MCP_SESSION_ID_HEADER] == "rejected-initialize-session"
    assert MCP_PROTOCOL_VERSION_HEADER not in transport.delete_headers


def test_http_initialize_decode_deadline_retains_delete_before_client_close(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    body = json.dumps(
        {
            "jsonrpc": "1.0",
            "id": 1,
            "result": {
                "protocolVersion": MCP_PROTOCOL_VERSION,
                "capabilities": {},
                "serverInfo": {"name": "late-invalid", "version": "1"},
            },
        },
        separators=(",", ":"),
    ).encode()
    transport = _RetainedInitializationSettlementTransport(_ChunkStream([body]))
    original_decode = mcp_http_module._decode_jsonrpc_bytes

    def delayed_decode(data: bytes) -> dict[str, Any]:
        time.sleep(0.15)
        return original_decode(data)

    async def run() -> BaseException:
        with monkeypatch.context() as patcher:
            patcher.setattr(mcp_http_module, "_decode_jsonrpc_bytes", delayed_decode)
            with pytest.raises(McpCallDeadlineExceededError) as exc_info:
                await HttpMcpClient(
                    transport=transport,
                    transport_limits=_limits(
                        idle_timeout_s=0.5,
                        total_call_timeout_s=0.1,
                    ),
                ).connect(
                    McpServerSpec(
                        name="decode-deadline-initialize",
                        url="https://mcp.example/rpc",
                    )
                )
        settlement = _http_settlement_task(exc_info.value)
        assert settlement is not None
        await settlement
        return exc_info.value

    error = asyncio.run(run())

    assert transport.events == ["initialize", "delete", "close"]
    assert transport.delete_headers is not None
    assert transport.delete_headers[MCP_SESSION_ID_HEADER] == "retained-initialize-session"
    assert transport.delete_headers[MCP_PROTOCOL_VERSION_HEADER] == MCP_PROTOCOL_VERSION
    assert "jsonrpc='2.0'" in repr(error.__cause__)


def test_http_initialize_response_close_deadline_settles_before_delete() -> None:
    body = json.dumps(
        {
            "jsonrpc": "1.0",
            "id": 1,
            "result": {
                "protocolVersion": MCP_PROTOCOL_VERSION,
                "capabilities": {},
                "serverInfo": {"name": "close-deadline", "version": "1"},
            },
        },
        separators=(",", ":"),
    ).encode()
    stream = _FirstCloseBlocksStream([body])
    transport = _RetainedInitializationSettlementTransport(stream)

    async def run() -> BaseException:
        with pytest.raises(McpCallDeadlineExceededError) as exc_info:
            await HttpMcpClient(
                transport=transport,
                transport_limits=_limits(
                    idle_timeout_s=0.5,
                    total_call_timeout_s=0.02,
                ),
            ).connect(
                McpServerSpec(
                    name="close-deadline-initialize",
                    url="https://mcp.example/rpc",
                )
            )
        settlement = _http_settlement_task(exc_info.value)
        assert settlement is not None
        assert settlement.done() is False
        assert transport.events == ["initialize"]
        stream.release_close.set()
        await settlement
        return exc_info.value

    error = asyncio.run(run())

    assert stream.closed is True
    assert stream.overlapping_close is False
    assert transport.events == ["initialize", "delete", "close"]
    assert transport.delete_headers is not None
    assert transport.delete_headers[MCP_SESSION_ID_HEADER] == "retained-initialize-session"
    assert MCP_PROTOCOL_VERSION_HEADER not in transport.delete_headers
    assert error.__cause__ is None


def test_http_nested_initialize_delete_settlement_reports_client_close_failure_once() -> None:
    body = json.dumps(
        {
            "jsonrpc": "1.0",
            "id": 1,
            "result": {
                "protocolVersion": MCP_PROTOCOL_VERSION,
                "capabilities": {},
                "serverInfo": {"name": "nested-delete", "version": "1"},
            },
        },
        separators=(",", ":"),
    ).encode()
    initialize_stream = _FirstCloseBlocksStream([body])
    delete_stream = _FirstCloseBlocksStream([b""])
    raw_close_failure = RuntimeError("nested initialize client close failed")
    transport = _RetainedInitializationSettlementTransport(
        initialize_stream,
        delete_stream=delete_stream,
        client_close_failure=raw_close_failure,
    )

    async def run() -> tuple[BaseException, BaseException]:
        with pytest.raises(McpCallDeadlineExceededError) as exc_info:
            await HttpMcpClient(
                transport=transport,
                transport_limits=_limits(
                    idle_timeout_s=0.5,
                    total_call_timeout_s=0.02,
                ),
            ).connect(
                McpServerSpec(
                    name="nested-delete-initialize",
                    url="https://mcp.example/rpc",
                )
            )
        settlement = _http_settlement_task(exc_info.value)
        assert settlement is not None
        assert transport.events == ["initialize"]
        initialize_stream.release_close.set()
        await asyncio.wait_for(delete_stream.close_started.wait(), timeout=0.1)
        assert transport.events == ["initialize", "delete"]
        delete_stream.release_close.set()
        settlement_outcome = (await asyncio.gather(settlement, return_exceptions=True))[0]
        assert isinstance(settlement_outcome, BaseException)
        return exc_info.value, settlement_outcome

    primary_error, settlement_error = asyncio.run(run())

    assert type(primary_error) is McpCallDeadlineExceededError
    assert type(settlement_error) is McpProtocolError
    assert settlement_error is not raw_close_failure
    assert "client cleanup failed" in str(settlement_error)
    assert transport.events == ["initialize", "delete", "close"]
    assert initialize_stream.close_calls == 2
    assert delete_stream.close_calls == 1
    assert initialize_stream.overlapping_close is False
    assert delete_stream.overlapping_close is False


def test_http_initialized_notification_timeout_retains_session_termination() -> None:
    transport = _InitializedNotificationSettlementTransport()

    async def run() -> BaseException:
        with pytest.raises(McpIdleTimeoutError) as exc_info:
            await HttpMcpClient(
                transport=transport,
                transport_limits=_limits(
                    idle_timeout_s=0.02,
                    total_call_timeout_s=0.2,
                ),
            ).connect(
                McpServerSpec(
                    name="initialized-notification-timeout",
                    url="https://mcp.example/rpc",
                )
            )
        settlement = _http_settlement_task(exc_info.value)
        assert settlement is not None
        await settlement
        return exc_info.value

    error = asyncio.run(run())

    assert error.__cause__ is None
    assert transport.stream.closed is True
    assert transport.events == ["initialize", "initialized", "delete", "close"]
    assert transport.delete_headers is not None
    assert transport.delete_headers[MCP_SESSION_ID_HEADER] == "initialized-notification-session"
    assert transport.delete_headers[MCP_PROTOCOL_VERSION_HEADER] == MCP_PROTOCOL_VERSION


def test_http_retained_initialize_delete_failure_is_secondary_and_client_closes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    delete_failure = RuntimeError("retained initialization DELETE failed")
    body = json.dumps(
        {
            "jsonrpc": "1.0",
            "id": 1,
            "result": {
                "protocolVersion": MCP_PROTOCOL_VERSION,
                "capabilities": {},
                "serverInfo": {"name": "delete-failure", "version": "1"},
            },
        },
        separators=(",", ":"),
    ).encode()
    transport = _RetainedInitializationSettlementTransport(
        _ChunkStream([body]),
        delete_failure=delete_failure,
    )
    original_decode = mcp_http_module._decode_jsonrpc_bytes

    def delayed_decode(data: bytes) -> dict[str, Any]:
        time.sleep(0.15)
        return original_decode(data)

    async def run() -> tuple[BaseException, BaseException]:
        with monkeypatch.context() as patcher:
            patcher.setattr(mcp_http_module, "_decode_jsonrpc_bytes", delayed_decode)
            with pytest.raises(McpCallDeadlineExceededError) as exc_info:
                await HttpMcpClient(
                    transport=transport,
                    transport_limits=_limits(
                        idle_timeout_s=0.5,
                        total_call_timeout_s=0.1,
                    ),
                ).connect(
                    McpServerSpec(
                        name="retained-delete-failure",
                        url="https://mcp.example/rpc",
                    )
                )
        settlement = _http_settlement_task(exc_info.value)
        assert settlement is not None
        with pytest.raises(McpProtocolError) as settlement_exc:
            await settlement
        return exc_info.value, settlement_exc.value

    error, settlement_error = asyncio.run(run())

    assert "DELETE failed" not in str(error)
    assert "initialization session cleanup failed" in str(settlement_error)
    assert transport.events == ["initialize", "delete", "close"]


def test_http_close_joins_retained_initialization_settlement_without_duplicate_delete() -> None:
    body = json.dumps(
        {
            "jsonrpc": "1.0",
            "id": 1,
            "result": {
                "protocolVersion": MCP_PROTOCOL_VERSION,
                "capabilities": {},
                "serverInfo": {"name": "concurrent-close", "version": "1"},
            },
        },
        separators=(",", ":"),
    ).encode()
    stream = _FirstCloseBlocksStream([body])
    transport = _RetainedInitializationSettlementTransport(stream)
    sessions: list[HttpMcpSession] = []

    def capture_session(**kwargs: Any) -> HttpMcpSession:
        session = HttpMcpSession(**kwargs)
        sessions.append(session)
        return session

    async def run() -> None:
        with pytest.MonkeyPatch.context() as patcher:
            patcher.setattr(mcp_http_module, "HttpMcpSession", capture_session)
            with pytest.raises(McpCallDeadlineExceededError) as exc_info:
                await HttpMcpClient(
                    transport=transport,
                    transport_limits=_limits(
                        idle_timeout_s=0.5,
                        total_call_timeout_s=0.02,
                    ),
                ).connect(
                    McpServerSpec(
                        name="concurrent-close-initialize",
                        url="https://mcp.example/rpc",
                    )
                )
        settlement = _http_settlement_task(exc_info.value)
        assert settlement is not None
        assert len(sessions) == 1
        close_task = asyncio.create_task(sessions[0].close())
        await asyncio.sleep(0)
        assert transport.events == ["initialize"]
        stream.release_close.set()
        await settlement
        await close_task

    asyncio.run(run())

    assert transport.events == ["initialize", "delete", "close"]


def test_http_total_deadline_remains_authoritative_during_response_processing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_redact = mcp_http_module.safely_redact_jsonrpc_response

    def delayed_redaction(*args: Any, **kwargs: Any) -> Any:
        time.sleep(0.04)
        return original_redact(*args, **kwargs)

    def handler(request: httpx.Request) -> httpx.Response:
        request_id = json.loads(request.content)["id"]
        return httpx.Response(
            200,
            headers={"content-type": "application/json"},
            content=_jsonrpc_body(request_id),
        )

    async def run() -> tuple[bool, bool, bool]:
        session = _session(
            handler,
            limits=_limits(idle_timeout_s=0.5, total_call_timeout_s=0.02),
        )
        with monkeypatch.context() as patcher:
            patcher.setattr(
                mcp_http_module,
                "safely_redact_jsonrpc_response",
                delayed_redaction,
            )
            with pytest.raises(McpCallDeadlineExceededError) as exc_info:
                await session.call_tool("echo", {})
        settlement = _http_settlement_task(exc_info.value)
        assert settlement is not None
        await settlement
        return session._closed, session._fenced, session._http.is_closed

    session_closed, session_fenced, client_closed = asyncio.run(run())

    assert session_closed is True
    assert session_fenced is True
    assert client_closed is True


def test_http_processing_deadline_terminates_established_server_session(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_redact = mcp_http_module.safely_redact_jsonrpc_response
    methods: list[str] = []

    def delayed_redaction(*args: Any, **kwargs: Any) -> Any:
        time.sleep(0.04)
        return original_redact(*args, **kwargs)

    def handler(request: httpx.Request) -> httpx.Response:
        methods.append(request.method)
        if request.method == "DELETE":
            return httpx.Response(200, content=b"")
        request_id = json.loads(request.content)["id"]
        return httpx.Response(
            200,
            headers={"content-type": "application/json"},
            content=_jsonrpc_body(request_id),
        )

    async def run() -> tuple[bool, bool, bool]:
        session = _session(
            handler,
            limits=_limits(idle_timeout_s=0.5, total_call_timeout_s=0.02),
        )
        session._initialize_result = McpInitializeResult(protocol_version=MCP_PROTOCOL_VERSION)
        session._negotiated_protocol_version = MCP_PROTOCOL_VERSION
        session._session_id = "established-session"
        with monkeypatch.context() as patcher:
            patcher.setattr(
                mcp_http_module,
                "safely_redact_jsonrpc_response",
                delayed_redaction,
            )
            with pytest.raises(McpCallDeadlineExceededError) as exc_info:
                await session.call_tool("echo", {})
        settlement = _http_settlement_task(exc_info.value)
        assert settlement is not None
        await settlement
        return session._closed, session._fenced, session._http.is_closed

    session_closed, session_fenced, client_closed = asyncio.run(run())

    assert methods == ["POST", "DELETE"]
    assert session_closed is True
    assert session_fenced is True
    assert client_closed is True


@pytest.mark.parametrize("client_close_fails", [False, True])
def test_http_processing_deadline_retains_established_cleanup_failures(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    capsys: pytest.CaptureFixture[str],
    recwarn: pytest.WarningsRecorder,
    client_close_fails: bool,
) -> None:
    delete_secret = "established-deadline-delete-secret-canary"
    close_secret = "established-deadline-close-secret-canary"
    transport = _EstablishedDeadlineCleanupTransport(
        delete_failure=RuntimeError(f"DELETE exposed {delete_secret}"),
        client_close_failure=(
            RuntimeError(f"client close exposed {close_secret}") if client_close_fails else None
        ),
    )
    original_redact = mcp_http_module.safely_redact_jsonrpc_response

    def delayed_redaction(*args: Any, **kwargs: Any) -> Any:
        time.sleep(0.04)
        return original_redact(*args, **kwargs)

    async def run() -> tuple[BaseException, BaseException, bool, bool]:
        http_client = httpx.AsyncClient(transport=transport)
        session = HttpMcpSession(
            server=McpServerSpec(name="cleanup-failure", url="https://mcp.example/rpc"),
            http_client=http_client,
            url="https://mcp.example/rpc",
            client_name="cayu-test",
            client_version="1",
            transport_limits=_limits(idle_timeout_s=0.5, total_call_timeout_s=0.02),
            secret_redactor=SecretRedactor([delete_secret, close_secret]),
        )
        session._initialize_result = McpInitializeResult(protocol_version=MCP_PROTOCOL_VERSION)
        session._negotiated_protocol_version = MCP_PROTOCOL_VERSION
        session._session_id = "established-cleanup-failure-session"
        with monkeypatch.context() as patcher:
            patcher.setattr(
                mcp_http_module,
                "safely_redact_jsonrpc_response",
                delayed_redaction,
            )
            with pytest.raises(McpCallDeadlineExceededError) as exc_info:
                await session.call_tool("echo", {})
        settlement = _http_settlement_task(exc_info.value)
        assert settlement is not None
        settlement_outcome = (await asyncio.gather(settlement, return_exceptions=True))[0]
        assert isinstance(settlement_outcome, BaseException)
        return exc_info.value, settlement_outcome, session._fenced, http_client.is_closed

    with caplog.at_level(logging.DEBUG):
        error, settlement_error, session_fenced, client_closed = asyncio.run(run())

    assert type(error) is McpCallDeadlineExceededError
    assert error.__cause__ is None
    assert delete_secret not in "".join(traceback.format_exception(error))
    assert close_secret not in "".join(traceback.format_exception(error))
    if client_close_fails:
        assert isinstance(settlement_error, BaseExceptionGroup)
        assert len(settlement_error.exceptions) == 2
        delete_error, close_error = settlement_error.exceptions
        assert isinstance(delete_error, McpProtocolError)
        assert isinstance(close_error, McpProtocolError)
        assert "server-session termination failed" in str(delete_error)
        assert "client cleanup failed" in str(close_error)
    else:
        assert isinstance(settlement_error, McpProtocolError)
        assert "server-session termination failed" in str(settlement_error)
    settlement_traceback = "".join(traceback.format_exception(settlement_error))
    assert delete_secret not in settlement_traceback
    assert close_secret not in settlement_traceback
    assert REDACTED_SECRET in settlement_traceback
    captured = capsys.readouterr()
    assert delete_secret not in captured.out
    assert delete_secret not in captured.err
    assert close_secret not in captured.out
    assert close_secret not in captured.err
    assert all(delete_secret not in record.getMessage() for record in caplog.records)
    assert all(close_secret not in record.getMessage() for record in caplog.records)
    assert all(delete_secret not in str(warning.message) for warning in recwarn)
    assert all(close_secret not in str(warning.message) for warning in recwarn)
    assert transport.events == ["request", "delete", "close"]
    assert session_fenced is True
    assert client_closed is True


def test_http_processing_deadline_settles_delete_response_before_client_close(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    delete_stream = _FirstCloseBlocksStream([b""])
    transport = _EstablishedDeadlineCleanupTransport(delete_stream=delete_stream)
    original_redact = mcp_http_module.safely_redact_jsonrpc_response

    def delayed_redaction(*args: Any, **kwargs: Any) -> Any:
        time.sleep(0.04)
        return original_redact(*args, **kwargs)

    async def run() -> tuple[list[str], list[str], bool]:
        http_client = httpx.AsyncClient(transport=transport)
        session = HttpMcpSession(
            server=McpServerSpec(name="ordered-cleanup", url="https://mcp.example/rpc"),
            http_client=http_client,
            url="https://mcp.example/rpc",
            client_name="cayu-test",
            client_version="1",
            transport_limits=_limits(idle_timeout_s=0.5, total_call_timeout_s=0.02),
        )
        session._initialize_result = McpInitializeResult(protocol_version=MCP_PROTOCOL_VERSION)
        session._negotiated_protocol_version = MCP_PROTOCOL_VERSION
        session._session_id = "ordered-cleanup-session"
        with monkeypatch.context() as patcher:
            patcher.setattr(
                mcp_http_module,
                "safely_redact_jsonrpc_response",
                delayed_redaction,
            )
            with pytest.raises(McpCallDeadlineExceededError) as exc_info:
                await session.call_tool("echo", {})
        settlement = _http_settlement_task(exc_info.value)
        assert settlement is not None
        await asyncio.wait_for(delete_stream.close_started.wait(), timeout=0.1)
        events_before_release = list(transport.events)
        delete_stream.release_close.set()
        await asyncio.wait_for(settlement, timeout=0.2)
        return events_before_release, list(transport.events), http_client.is_closed

    events_before_release, final_events, client_closed = asyncio.run(run())

    assert events_before_release == ["request", "delete"]
    assert final_events == ["request", "delete", "close"]
    assert delete_stream.close_calls == 1
    assert delete_stream.overlapping_close is False
    assert client_closed is True


def test_http_processing_deadline_reports_nested_delete_and_client_failures_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw_delete_close_failure = RuntimeError("nested DELETE response close failed")
    raw_client_close_failure = RuntimeError("nested DELETE client close failed")
    delete_stream = _FirstCloseBlocksThenFailsStream([b""], raw_delete_close_failure)
    transport = _EstablishedDeadlineCleanupTransport(
        delete_stream=delete_stream,
        client_close_failure=raw_client_close_failure,
    )
    original_redact = mcp_http_module.safely_redact_jsonrpc_response

    def delayed_redaction(*args: Any, **kwargs: Any) -> Any:
        time.sleep(0.04)
        return original_redact(*args, **kwargs)

    async def run() -> BaseException:
        session = HttpMcpSession(
            server=McpServerSpec(name="nested-failures", url="https://mcp.example/rpc"),
            http_client=httpx.AsyncClient(transport=transport),
            url="https://mcp.example/rpc",
            client_name="cayu-test",
            client_version="1",
            transport_limits=_limits(idle_timeout_s=0.5, total_call_timeout_s=0.02),
        )
        session._initialize_result = McpInitializeResult(protocol_version=MCP_PROTOCOL_VERSION)
        session._negotiated_protocol_version = MCP_PROTOCOL_VERSION
        session._session_id = "nested-failures-session"
        with monkeypatch.context() as patcher:
            patcher.setattr(
                mcp_http_module,
                "safely_redact_jsonrpc_response",
                delayed_redaction,
            )
            with pytest.raises(McpCallDeadlineExceededError) as exc_info:
                await session.call_tool("echo", {})
        settlement = _http_settlement_task(exc_info.value)
        assert settlement is not None
        await asyncio.wait_for(delete_stream.close_started.wait(), timeout=0.1)
        await asyncio.sleep(0.03)
        delete_stream.release_close.set()
        settlement_outcome = (await asyncio.gather(settlement, return_exceptions=True))[0]
        assert isinstance(settlement_outcome, BaseException)
        return settlement_outcome

    settlement_error = asyncio.run(run())

    assert isinstance(settlement_error, BaseExceptionGroup)
    assert len(settlement_error.exceptions) == 3
    termination_error, delete_error, client_error = settlement_error.exceptions
    assert type(termination_error) is McpCallDeadlineExceededError
    assert type(delete_error) is McpProtocolError
    assert type(client_error) is McpProtocolError
    assert termination_error is not raw_delete_close_failure
    assert delete_error is not raw_delete_close_failure
    assert client_error is not raw_client_close_failure
    assert "server-session termination failed" in str(termination_error)
    assert "stream close retry failed" in str(delete_error)
    assert "client cleanup failed" in str(client_error)
    assert transport.events == ["request", "delete", "close"]
    assert delete_stream.close_calls == 2
    assert delete_stream.overlapping_close is False


def test_http_total_deadline_includes_tool_result_validation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_parser = mcp_http_module.tool_result_from_payload

    def delayed_parser(value: object) -> Any:
        time.sleep(0.04)
        return original_parser(value)

    def handler(request: httpx.Request) -> httpx.Response:
        request_id = json.loads(request.content)["id"]
        return httpx.Response(
            200,
            headers={"content-type": "application/json"},
            content=_jsonrpc_body(request_id),
        )

    async def run() -> tuple[bool, bool, bool]:
        session = _session(
            handler,
            limits=_limits(idle_timeout_s=0.5, total_call_timeout_s=0.02),
        )
        with monkeypatch.context() as patcher:
            patcher.setattr(mcp_http_module, "tool_result_from_payload", delayed_parser)
            with pytest.raises(McpCallDeadlineExceededError) as exc_info:
                await session.call_tool("echo", {})
        settlement = _http_settlement_task(exc_info.value)
        assert settlement is not None
        await settlement
        return session._closed, session._fenced, session._http.is_closed

    session_closed, session_fenced, client_closed = asyncio.run(run())

    assert session_closed is True
    assert session_fenced is True
    assert client_closed is True


@pytest.mark.parametrize("processing_stage", ["malformed-json", "http-error"])
def test_http_total_deadline_wins_over_late_protocol_processing_failure(
    monkeypatch: pytest.MonkeyPatch,
    processing_stage: str,
) -> None:
    if processing_stage == "malformed-json":
        original_processing = mcp_http_module._decode_jsonrpc_bytes

        def delayed_processing(value: bytes) -> dict[str, Any]:
            time.sleep(0.04)
            return original_processing(value)

        expected_cause = "not valid JSON"
    else:
        original_processing = mcp_http_module._safe_body

        def delayed_processing(value: bytes, **kwargs: Any) -> str:
            time.sleep(0.04)
            return original_processing(value, **kwargs)

        expected_cause = "HTTP 400"

    def handler(request: httpx.Request) -> httpx.Response:
        if processing_stage == "malformed-json":
            return httpx.Response(
                200,
                headers={"content-type": "application/json"},
                content=b"not-json",
            )
        return httpx.Response(
            400,
            headers={"content-type": "text/plain"},
            content=b"bad request",
        )

    async def run() -> tuple[BaseException | None, bool, bool, bool]:
        session = _session(
            handler,
            limits=_limits(idle_timeout_s=0.5, total_call_timeout_s=0.02),
        )
        target = "_decode_jsonrpc_bytes" if processing_stage == "malformed-json" else "_safe_body"
        with monkeypatch.context() as patcher:
            patcher.setattr(mcp_http_module, target, delayed_processing)
            with pytest.raises(McpCallDeadlineExceededError) as exc_info:
                await session.call_tool("echo", {})
        settlement = _http_settlement_task(exc_info.value)
        assert settlement is not None
        await settlement
        return (
            exc_info.value.__cause__,
            session._closed,
            session._fenced,
            session._http.is_closed,
        )

    cause, session_closed, session_fenced, client_closed = asyncio.run(run())

    assert isinstance(cause, McpProtocolError)
    assert expected_cause in str(cause)
    assert session_closed is True
    assert session_fenced is True
    assert client_closed is True


def test_http_idle_timeout_does_not_charge_completed_body_processing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_decode = mcp_http_module._decode_jsonrpc_bytes

    def delayed_decode(data: bytes) -> dict[str, Any]:
        time.sleep(0.04)
        return original_decode(data)

    def handler(request: httpx.Request) -> httpx.Response:
        request_id = json.loads(request.content)["id"]
        return httpx.Response(
            200,
            headers={"content-type": "application/json"},
            content=_jsonrpc_body(request_id),
        )

    async def run() -> tuple[bool, bool]:
        session = _session(
            handler,
            limits=_limits(idle_timeout_s=0.02, total_call_timeout_s=0.5),
        )
        try:
            with monkeypatch.context() as patcher:
                patcher.setattr(mcp_http_module, "_decode_jsonrpc_bytes", delayed_decode)
                result = await session.call_tool("echo", {})
            return result.is_error, session._closed
        finally:
            await session.close()

    result_is_error, session_closed = asyncio.run(run())

    assert result_is_error is False
    assert session_closed is False


def test_http_expired_deadline_wins_over_completed_child_exception() -> None:
    async def run() -> tuple[tuple[asyncio.Future[Any], ...], BaseException]:
        budget = _HttpCallBudget(_limits(idle_timeout_s=10, total_call_timeout_s=10))

        async def fail_after_deadline() -> None:
            budget._deadline = asyncio.get_running_loop().time()
            raise RuntimeError("late child failure")

        with pytest.raises(McpCallDeadlineExceededError) as exc_info:
            await budget.wait(fail_after_deadline())
        abandoned = budget.take_abandoned_tasks()
        return abandoned, exc_info.value

    abandoned, error = asyncio.run(run())

    assert type(error) is McpCallDeadlineExceededError
    assert len(abandoned) == 1
    assert isinstance(abandoned[0].exception(), RuntimeError)


def test_http_deadline_before_response_headers_poisons_and_closes_client() -> None:
    secret = "mcp-http-late-handler-secret"
    handler_started = asyncio.Event()
    cancellation_seen = asyncio.Event()
    release_handler = asyncio.Event()
    handler_finished = asyncio.Event()

    async def handler(request: httpx.Request) -> httpx.Response:
        handler_started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            cancellation_seen.set()
            await release_handler.wait()
            handler_finished.set()
            raise httpx.ReadError(
                f"late cancelled handler failure {secret}", request=request
            ) from None

    async def run() -> tuple[bool, bool, bool, bool, float, BaseException, BaseException]:
        http_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        session = HttpMcpSession(
            server=McpServerSpec(name="headers", url="https://mcp.example/rpc"),
            http_client=http_client,
            url="https://mcp.example/rpc",
            client_name="cayu-test",
            client_version="1",
            transport_limits=_limits(idle_timeout_s=0.5, total_call_timeout_s=0.02),
            secret_redactor=SecretRedactor(secret),
        )
        try:
            started_at = asyncio.get_running_loop().time()
            with pytest.raises(McpCallDeadlineExceededError) as exc_info:
                await asyncio.wait_for(session.call_tool("echo", {}), timeout=0.2)
            elapsed = asyncio.get_running_loop().time() - started_at
            settlement = _http_settlement_task(exc_info.value)
            assert settlement is not None
            assert settlement.done() is False
            with pytest.raises(McpProtocolError, match="closed"):
                await session.call_tool("competing", {})
            release_handler.set()
            await asyncio.wait_for(handler_finished.wait(), timeout=0.2)
            with pytest.raises(
                McpProtocolError, match="late cancelled handler failure"
            ) as settlement_info:
                await settlement
            return (
                handler_started.is_set(),
                cancellation_seen.is_set(),
                session._closed,
                http_client.is_closed,
                elapsed,
                exc_info.value,
                settlement_info.value,
            )
        finally:
            await session.close()

    (
        started,
        cancelled,
        session_closed,
        client_closed,
        elapsed,
        public_error,
        settlement_error,
    ) = asyncio.run(run())

    assert started is True
    assert cancelled is True
    assert session_closed is True
    assert client_closed is True
    assert elapsed < 0.1
    assert secret not in repr(public_error)
    assert secret not in "".join(traceback.format_exception(settlement_error))
    traceback_cursor = settlement_error.__traceback__
    while traceback_cursor is not None:
        if is_cayu_source_filename(traceback_cursor.tb_frame.f_code.co_filename):
            assert all(
                secret not in repr(value) for value in traceback_cursor.tb_frame.f_locals.values()
            )
            if traceback_cursor.tb_frame.f_code.co_name == "_settle_registered_http_exchange":
                assert "exchange_owner" not in traceback_cursor.tb_frame.f_locals
                assert traceback_cursor.tb_frame.f_locals.get("kwargs") == {}
        traceback_cursor = traceback_cursor.tb_next


def test_http_late_response_after_header_deadline_exits_owned_context() -> None:
    handler_started = asyncio.Event()
    cancellation_seen = asyncio.Event()
    release_handler = asyncio.Event()
    stream = _ChunkStream([_jsonrpc_body(1)])

    async def handler(request: httpx.Request) -> httpx.Response:
        handler_started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            cancellation_seen.set()
            await release_handler.wait()
            return httpx.Response(
                200,
                headers={"content-type": "application/json"},
                stream=stream,
            )

    async def run() -> tuple[bool, bool, int, bool]:
        session = _session(
            handler,
            limits=_limits(idle_timeout_s=0.5, total_call_timeout_s=0.02),
        )
        try:
            with pytest.raises(McpCallDeadlineExceededError) as exc_info:
                await session.call_tool("echo", {})
            settlement = _http_settlement_task(exc_info.value)
            assert settlement is not None
            assert settlement.done() is False
            release_handler.set()
            await asyncio.wait_for(settlement, timeout=0.2)
            return (
                handler_started.is_set(),
                cancellation_seen.is_set(),
                stream.close_calls,
                session._http.is_closed,
            )
        finally:
            await session.close()

    started, cancelled, close_calls, client_closed = asyncio.run(run())

    assert started is True
    assert cancelled is True
    assert close_calls == 1
    assert client_closed is True


def test_http_late_initialize_response_is_deleted_before_client_close() -> None:
    transport = _LateInitializationResponseTransport()

    async def run() -> BaseException:
        with pytest.raises(McpCallDeadlineExceededError) as exc_info:
            await HttpMcpClient(
                transport=transport,
                transport_limits=_limits(
                    idle_timeout_s=0.5,
                    total_call_timeout_s=0.02,
                ),
            ).connect(
                McpServerSpec(
                    name="late-initialize",
                    url="https://mcp.example/rpc",
                )
            )
        error = exc_info.value
        settlement = _http_settlement_task(error)
        assert settlement is not None
        await asyncio.wait_for(transport.cancellation_seen.wait(), timeout=0.2)
        assert settlement.done() is False
        assert transport.events == ["initialize"]
        transport.release_response.set()
        await asyncio.wait_for(settlement, timeout=0.2)
        return error

    error = asyncio.run(run())

    assert type(error) is McpCallDeadlineExceededError
    assert transport.events == ["initialize", "delete", "close"]
    assert transport.delete_headers is not None
    assert transport.delete_headers[MCP_SESSION_ID_HEADER] == "late-initialize-session"
    assert MCP_PROTOCOL_VERSION_HEADER not in transport.delete_headers


def test_http_late_initialize_delete_settlement_does_not_wait_on_parent_exchange() -> None:
    delete_stream = _CancellationResistantCloseStream([b""])
    transport = _LateInitializationResponseTransport(delete_stream=delete_stream)

    async def run() -> tuple[bool, bool, bool]:
        with pytest.raises(McpCallDeadlineExceededError) as exc_info:
            await HttpMcpClient(
                transport=transport,
                transport_limits=_limits(
                    idle_timeout_s=0.5,
                    total_call_timeout_s=0.02,
                ),
            ).connect(
                McpServerSpec(
                    name="late-initialize-delete-settlement",
                    url="https://mcp.example/rpc",
                )
            )
        settlement = _http_settlement_task(exc_info.value)
        assert settlement is not None
        await asyncio.wait_for(transport.cancellation_seen.wait(), timeout=0.2)
        transport.release_response.set()
        await asyncio.wait_for(delete_stream.close_started.wait(), timeout=0.2)
        await asyncio.wait_for(delete_stream.cancellation_seen.wait(), timeout=0.2)
        # Let DELETE transfer its cancellation-resistant context close to the
        # retained child owner before allowing that exact owner to finish.
        await asyncio.sleep(0.03)
        assert settlement.done() is False
        delete_stream.release_close.set()
        settlement_outcome = (
            await asyncio.wait_for(
                asyncio.gather(settlement, return_exceptions=True),
                timeout=0.2,
            )
        )[0]
        return (
            delete_stream.overlapping_close,
            delete_stream.closed,
            isinstance(settlement_outcome, McpCallDeadlineExceededError),
        )

    overlapping_close, delete_closed, preserved_deadline = asyncio.run(run())

    assert transport.events == ["initialize", "delete", "close"]
    assert overlapping_close is False
    assert delete_closed is True
    assert preserved_deadline is True


def test_http_unexpected_transport_cancellation_is_not_caller_cancellation() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        raise asyncio.CancelledError

    async def run() -> tuple[bool, bool, int]:
        session = _session(handler, limits=_limits())
        try:
            with pytest.raises(McpProtocolError, match="cancelled unexpectedly") as exc_info:
                await session.call_tool("echo", {})
            settlement = _http_settlement_task(exc_info.value)
            assert settlement is not None
            await settlement
            current = asyncio.current_task()
            assert current is not None
            return session._closed, session._http.is_closed, current.cancelling()
        finally:
            await session.close()

    session_closed, client_closed, caller_cancelling = asyncio.run(run())

    assert session_closed is True
    assert client_closed is True
    assert caller_cancelling == 0


def test_http_historical_cancellation_does_not_reclassify_transport_cancellation() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        raise asyncio.CancelledError

    async def run() -> tuple[int, int, bool, bool]:
        current = asyncio.current_task()
        assert current is not None
        current.cancel("historical cancellation")
        with pytest.raises(asyncio.CancelledError, match="historical cancellation"):
            await asyncio.sleep(0)
        historical_count = current.cancelling()

        session = _session(handler, limits=_limits())
        try:
            with pytest.raises(McpProtocolError, match="cancelled unexpectedly") as exc_info:
                await session.call_tool("echo", {})
            settlement = _http_settlement_task(exc_info.value)
            assert settlement is not None
            await settlement
            return (
                historical_count,
                current.cancelling(),
                session._closed,
                session._http.is_closed,
            )
        finally:
            await session.close()
            current.uncancel()

    historical_count, final_count, session_closed, client_closed = asyncio.run(run())

    assert historical_count == 1
    assert final_count == historical_count
    assert session_closed is True
    assert client_closed is True


def test_http_peer_close_before_response_headers_fences_session() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.RemoteProtocolError("peer closed", request=request)

    async def run() -> tuple[bool, bool]:
        session = _session(handler, limits=_limits())
        try:
            with pytest.raises(McpPeerClosedError, match="peer closed") as exc_info:
                await session.call_tool("echo", {})
            settlement = _http_settlement_task(exc_info.value)
            assert settlement is not None
            await settlement
            return session._closed, session._http.is_closed
        finally:
            await session.close()

    session_closed, client_closed = asyncio.run(run())

    assert session_closed is True
    assert client_closed is True


def test_http_generic_transport_failure_detaches_raw_secret_cause() -> None:
    secret = "mcp-http-generic-transport-secret"
    raw_failure = RuntimeError(f"transport exposed {secret}")

    def handler(request: httpx.Request) -> httpx.Response:
        raise raw_failure

    async def run() -> BaseException:
        session = HttpMcpSession(
            server=McpServerSpec(name="generic-secret", url="https://mcp.example/rpc"),
            http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
            url="https://mcp.example/rpc",
            client_name="cayu-test",
            client_version="1",
            transport_limits=_limits(),
            secret_redactor=SecretRedactor(secret),
        )
        try:
            with pytest.raises(McpProtocolError) as exc_info:
                await session.call_tool("echo", {})
            settlement = _http_settlement_task(exc_info.value)
            assert settlement is not None
            await settlement
            return exc_info.value
        finally:
            await session.close()

    error = asyncio.run(run())

    assert error.__cause__ is not raw_failure
    assert error.__context__ is None
    assert secret not in "".join(traceback.format_exception(error))


def test_http_numeric_transport_failure_argument_is_redacted(
    caplog: pytest.LogCaptureFixture,
    capsys: pytest.CaptureFixture[str],
    recwarn: pytest.WarningsRecorder,
) -> None:
    secret = "4829017351642089"
    raw_failure = RuntimeError(int(secret))

    def handler(request: httpx.Request) -> httpx.Response:
        raise raw_failure

    async def run() -> BaseException:
        session = _session(
            handler,
            limits=_limits(),
            secret_redactor=SecretRedactor(secret),
        )
        try:
            with pytest.raises(McpProtocolError) as exc_info:
                await session.call_tool("echo", {})
            settlement = _http_settlement_task(exc_info.value)
            assert settlement is not None
            await settlement
            return exc_info.value
        finally:
            await session.close()

    with caplog.at_level(logging.DEBUG):
        error = asyncio.run(run())

    assert error.__cause__ is not raw_failure
    assert error.__context__ is None
    assert secret not in repr(error)
    assert secret not in "".join(traceback.format_exception(error))
    _assert_cayu_traceback_does_not_retain(error, secret)
    captured = capsys.readouterr()
    assert secret not in captured.out
    assert secret not in captured.err
    assert all(secret not in record.getMessage() for record in caplog.records)
    assert all(secret not in str(warning.message) for warning in recwarn)


@pytest.mark.parametrize("failure_type", [McpProtocolError, McpMessageTooLargeError])
def test_http_typed_extension_failure_fences_session_and_detaches_raw_secret_cause(
    failure_type: type[McpProtocolError],
) -> None:
    secret = "mcp-http-typed-extension-secret"
    raw_cause = RuntimeError(secret)
    raw_failure: McpProtocolError
    try:
        raise failure_type("typed stream failure") from raw_cause
    except McpProtocolError as exc:
        raw_failure = exc
    stream = _TransportFailureStream(raw_failure)
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(
            200,
            headers={"content-type": "application/json"},
            stream=stream,
        )

    async def run() -> tuple[BaseException, bool, bool, bool]:
        session = _session(
            handler,
            limits=_limits(),
            secret_redactor=SecretRedactor(secret),
        )
        try:
            with pytest.raises(McpProtocolError) as exc_info:
                await session.call_tool("echo", {})
            settlement = _http_settlement_task(exc_info.value)
            assert settlement is not None
            with pytest.raises(McpProtocolError, match="session is closed"):
                await session.call_tool("echo", {})
            await settlement
            return (
                exc_info.value,
                session._closed,
                session._fenced,
                session._http.is_closed,
            )
        finally:
            await session.close()

    error, session_closed, session_fenced, client_closed = asyncio.run(run())

    assert error is not raw_failure
    assert error.__cause__ is not raw_cause
    assert error.__context__ is None
    assert secret not in "".join(traceback.format_exception(error))
    assert session_closed is True
    assert session_fenced is True
    assert client_closed is True
    assert calls == 1


def test_http_grouped_extension_failure_fences_session_and_preserves_safe_order() -> None:
    secret = "mcp-http-grouped-extension-secret"
    raw_failure = BaseExceptionGroup(
        f"grouped stream failure exposed {secret}",
        [
            asyncio.CancelledError(f"stream cancelled with {secret}"),
            RuntimeError(f"stream failed with {secret}"),
        ],
    )
    stream = _TransportFailureStream(raw_failure)
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(
            200,
            headers={"content-type": "application/json"},
            stream=stream,
        )

    async def run() -> tuple[BaseExceptionGroup, bool, bool, bool]:
        session = _session(
            handler,
            limits=_limits(),
            secret_redactor=SecretRedactor(secret),
        )
        try:
            with pytest.raises(BaseExceptionGroup) as exc_info:
                await session.call_tool("echo", {})
            settlement = _http_settlement_task(exc_info.value)
            assert settlement is not None
            with pytest.raises(McpProtocolError, match="session is closed"):
                await session.call_tool("echo", {})
            await settlement
            return (
                exc_info.value,
                session._closed,
                session._fenced,
                session._http.is_closed,
            )
        finally:
            await session.close()

    error, session_closed, session_fenced, client_closed = asyncio.run(run())

    assert error is not raw_failure
    assert error.__context__ is None
    assert "CancelledError" in str(error.exceptions[0])
    assert "RuntimeError" in str(error.exceptions[1])
    assert secret not in "".join(traceback.format_exception(error))
    assert session_closed is True
    assert session_fenced is True
    assert client_closed is True
    assert calls == 1


def test_http_custom_scalar_base_exception_is_typed_redacted_and_fenced(
    caplog: pytest.LogCaptureFixture,
    capsys: pytest.CaptureFixture[str],
    recwarn: pytest.WarningsRecorder,
) -> None:
    secret = "mcp-http-custom-scalar-secret-canary"

    class CustomScalarSignal(BaseException):
        pass

    raw_failure = CustomScalarSignal(f"transport exposed {secret}")
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        raise raw_failure

    async def run() -> tuple[BaseException, bool, bool, bool]:
        session = _session(
            handler,
            limits=_limits(),
            secret_redactor=SecretRedactor(secret),
        )
        try:
            with pytest.raises(McpProtocolError) as exc_info:
                await session.call_tool("echo", {})
            error = exc_info.value
            settlement = _http_settlement_task(error)
            assert settlement is not None
            with pytest.raises(McpProtocolError, match="session is closed"):
                await session.call_tool("echo", {})
            await settlement
            return error, session._closed, session._fenced, session._http.is_closed
        finally:
            await session.close()

    with caplog.at_level(logging.DEBUG):
        error, session_closed, session_fenced, client_closed = asyncio.run(run())

    assert type(error) is McpProtocolError
    assert error is not raw_failure
    assert error.__cause__ is None
    assert error.__context__ is None
    assert secret not in repr(error)
    assert REDACTED_SECRET in repr(error)
    _assert_cayu_traceback_does_not_retain(error, secret)
    captured = capsys.readouterr()
    assert secret not in captured.out
    assert secret not in captured.err
    assert all(secret not in record.getMessage() for record in caplog.records)
    assert all(secret not in str(warning.message) for warning in recwarn)
    assert session_closed is True
    assert session_fenced is True
    assert client_closed is True
    assert calls == 1


@pytest.mark.parametrize("fatal_type", [KeyboardInterrupt, SystemExit, GeneratorExit])
def test_http_scalar_fatal_transport_failure_is_detached_and_fences_session(
    caplog: pytest.LogCaptureFixture,
    capsys: pytest.CaptureFixture[str],
    recwarn: pytest.WarningsRecorder,
    fatal_type: type[BaseException],
) -> None:
    secret = "mcp-http-scalar-fatal-secret-canary"
    raw_failure = fatal_type(f"transport exposed {secret}")
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        raise raw_failure

    async def run() -> tuple[BaseException, bool, bool, bool]:
        session = _session(
            handler,
            limits=_limits(),
            secret_redactor=SecretRedactor(secret),
        )
        try:
            with pytest.raises(fatal_type) as exc_info:
                await session.call_tool("echo", {})
            error = exc_info.value
            settlement = _http_settlement_task(error)
            assert settlement is not None
            with pytest.raises(McpProtocolError, match="session is closed"):
                await session.call_tool("echo", {})
            await settlement
            return error, session._closed, session._fenced, session._http.is_closed
        finally:
            await session.close()

    with caplog.at_level(logging.DEBUG):
        error, session_closed, session_fenced, client_closed = asyncio.run(run())

    assert error is not raw_failure
    assert error.__cause__ is None
    assert error.__context__ is None
    assert secret not in repr(error)
    assert REDACTED_SECRET in repr(error)
    _assert_cayu_traceback_does_not_retain(error, secret)
    captured = capsys.readouterr()
    assert secret not in captured.out
    assert secret not in captured.err
    assert all(secret not in record.getMessage() for record in caplog.records)
    assert all(secret not in str(warning.message) for warning in recwarn)
    assert session_closed is True
    assert session_fenced is True
    assert client_closed is True
    assert calls == 1


def test_http_numeric_scalar_fatal_argument_is_redacted_and_fenced(
    caplog: pytest.LogCaptureFixture,
    capsys: pytest.CaptureFixture[str],
    recwarn: pytest.WarningsRecorder,
) -> None:
    secret = "4829017351642089"
    raw_failure = SystemExit(int(secret))

    def handler(request: httpx.Request) -> httpx.Response:
        raise raw_failure

    async def run() -> tuple[BaseException, bool, bool]:
        session = _session(
            handler,
            limits=_limits(),
            secret_redactor=SecretRedactor(secret),
        )
        try:
            with pytest.raises(SystemExit) as exc_info:
                await session.call_tool("echo", {})
            error = exc_info.value
            settlement = _http_settlement_task(error)
            assert settlement is not None
            with pytest.raises(McpProtocolError, match="session is closed"):
                await session.call_tool("echo", {})
            await settlement
            return error, session._fenced, session._http.is_closed
        finally:
            await session.close()

    with caplog.at_level(logging.DEBUG):
        error, session_fenced, client_closed = asyncio.run(run())

    assert error is not raw_failure
    assert error.__cause__ is None
    assert error.__context__ is None
    assert error.args == (REDACTED_SECRET,)
    assert secret not in repr(error)
    assert secret not in "".join(traceback.format_exception(error))
    _assert_cayu_traceback_does_not_retain(error, secret)
    captured = capsys.readouterr()
    assert secret not in captured.out
    assert secret not in captured.err
    assert all(secret not in record.getMessage() for record in caplog.records)
    assert all(secret not in str(warning.message) for warning in recwarn)
    assert session_fenced is True
    assert client_closed is True


def test_http_system_exit_from_response_close_is_redacted_and_settled(
    caplog: pytest.LogCaptureFixture,
    capsys: pytest.CaptureFixture[str],
    recwarn: pytest.WarningsRecorder,
) -> None:
    secret = "mcp-http-response-close-fatal-secret-canary"
    raw_failure = SystemExit(f"response close exposed {secret}")
    stream = _ScalarFatalCleanupStream([_jsonrpc_body(1)], raw_failure)

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"content-type": "application/json"},
            stream=stream,
        )

    async def run() -> tuple[BaseException, bool, bool]:
        session = _session(
            handler,
            limits=_limits(),
            secret_redactor=SecretRedactor(secret),
        )
        try:
            with pytest.raises(SystemExit) as exc_info:
                await session.call_tool("echo", {})
            error = exc_info.value
            settlement = _http_settlement_task(error)
            assert settlement is not None
            await settlement
            with pytest.raises(McpProtocolError, match="session is closed"):
                await session.call_tool("echo", {})
            return error, session._fenced, session._http.is_closed
        finally:
            await session.close()

    with caplog.at_level(logging.DEBUG):
        error, session_fenced, client_closed = asyncio.run(run())

    assert stream.close_calls == 1
    assert error is not raw_failure
    assert error.__context__ is None
    assert secret not in "".join(traceback.format_exception(error))
    _assert_cayu_traceback_does_not_retain(error, secret)
    captured = capsys.readouterr()
    assert secret not in captured.out
    assert secret not in captured.err
    assert all(secret not in record.getMessage() for record in caplog.records)
    assert all(secret not in str(warning.message) for warning in recwarn)
    assert session_fenced is True
    assert client_closed is True


def test_http_request_error_never_renders_extension_diagnostic(
    capsys,
    caplog,
    recwarn,
) -> None:
    secret = "mcp-http-hostile-render-secret"
    raw_failure: _HostileRequestError | None = None

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal raw_failure
        raw_failure = _HostileRequestError(secret, request=request)
        raise raw_failure

    async def run() -> BaseException:
        session = _session(
            handler,
            limits=_limits(),
            secret_redactor=SecretRedactor(secret),
        )
        try:
            with pytest.raises(McpProtocolError) as exc_info:
                await session.call_tool("echo", {})
            settlement = _http_settlement_task(exc_info.value)
            assert settlement is not None
            await settlement
            return exc_info.value
        finally:
            await session.close()

    error = asyncio.run(run())
    captured = capsys.readouterr()

    assert raw_failure is not None
    assert raw_failure.render_calls == 0
    assert secret not in "".join(traceback.format_exception(error))
    assert secret not in captured.out
    assert secret not in captured.err
    assert all(secret not in record.getMessage() for record in caplog.records)
    assert all(secret not in str(warning.message) for warning in recwarn)


@pytest.mark.parametrize("diagnostic_mode", ["huge_integer", "many_strings"])
def test_http_request_error_diagnostics_are_source_bounded_and_fence_session(
    diagnostic_mode: str,
) -> None:
    secret = "mcp-http-bounded-exception-diagnostic-secret"

    class RecordingRedactor(SecretRedactor):
        def __init__(self) -> None:
            super().__init__(secret)
            self.source_bounds: list[tuple[int, int]] = []

        def redact_text_bounded(self, value: str, *, max_bytes: int) -> str:
            self.source_bounds.append((len(value), max_bytes))
            return super().redact_text_bounded(value, max_bytes=max_bytes)

    redactor = RecordingRedactor()

    def handler(request: httpx.Request) -> httpx.Response:
        failure = httpx.ConnectError("adapter failed", request=request)
        if diagnostic_mode == "huge_integer":
            failure.args = (10**5_000, secret)
        else:
            failure.args = tuple(f"{secret}-{'x' * 5_000}" for _ in range(100))
        raise failure

    async def run() -> tuple[BaseException, bool]:
        session = _session(
            handler,
            limits=_limits(),
            secret_redactor=redactor,
        )
        try:
            with pytest.raises(McpProtocolError) as exc_info:
                await session.call_tool("echo", {})
            settlement = _http_settlement_task(exc_info.value)
            assert settlement is not None
            await settlement
            return exc_info.value, session._closed
        finally:
            await session.close()

    error, session_closed = asyncio.run(run())

    assert session_closed is True
    assert error.__context__ is None
    assert len(str(error).encode()) <= 4_096
    assert secret not in "".join(traceback.format_exception(error))
    assert redactor.source_bounds
    assert all(source_chars <= max_bytes for source_chars, max_bytes in redactor.source_bounds)


def test_http_total_deadline_redacts_completed_response_during_cleanup(
    caplog: pytest.LogCaptureFixture,
    capsys: pytest.CaptureFixture[str],
    recwarn: pytest.WarningsRecorder,
) -> None:
    secret = "mcp-http-completed-response-deadline-secret"
    response_body = json.dumps(
        {
            "jsonrpc": "2.0",
            "id": 1,
            "result": {
                "content": [{"type": "text", "text": secret}],
                "structuredContent": {},
            },
        },
        separators=(",", ":"),
    ).encode()
    stream = _FirstCloseBlocksStream([response_body])

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"content-type": "application/json"},
            stream=stream,
        )

    async def run() -> tuple[BaseException, int, bool, bool, bool, float]:
        session = _session(
            handler,
            limits=_limits(idle_timeout_s=0.5, total_call_timeout_s=0.05),
            secret_redactor=SecretRedactor(secret),
        )
        try:
            started_at = asyncio.get_running_loop().time()
            with pytest.raises(McpCallDeadlineExceededError) as exc_info:
                await session.call_tool("echo", {})
            elapsed = asyncio.get_running_loop().time() - started_at
            settlement = _http_settlement_task(exc_info.value)
            assert settlement is not None
            assert settlement.done() is False
            stream.release_close.set()
            await settlement
            return (
                exc_info.value,
                stream.close_calls,
                stream.closed,
                stream.overlapping_close,
                session._closed,
                elapsed,
            )
        finally:
            await session.close()

    with caplog.at_level(logging.DEBUG):
        error, close_calls, response_closed, overlapping_close, session_closed, elapsed = (
            asyncio.run(run())
        )

    assert close_calls == 2
    assert response_closed is True
    assert overlapping_close is False
    assert session_closed is True
    assert elapsed < 0.2
    assert secret not in repr(error)
    assert secret not in "".join(traceback.format_exception(error))
    traceback_cursor = error.__traceback__
    while traceback_cursor is not None:
        if is_cayu_source_filename(traceback_cursor.tb_frame.f_code.co_filename):
            assert all(
                secret not in repr(value) for value in traceback_cursor.tb_frame.f_locals.values()
            )
        traceback_cursor = traceback_cursor.tb_next
    captured = capsys.readouterr()
    assert secret not in captured.out
    assert secret not in captured.err
    assert all(secret not in record.getMessage() for record in caplog.records)
    assert all(secret not in str(warning.message) for warning in recwarn)


def test_http_real_task_cancellation_closes_response_and_fences_session() -> None:
    secret = "mcp-http-cancellation-detail-secret"
    cancellation_detail = _HostileCancellationDetail(secret)
    blocked = _BlockingStream()
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls == 1:
            return httpx.Response(
                200,
                headers={"content-type": "application/json"},
                stream=blocked,
            )
        return httpx.Response(
            200,
            headers={"content-type": "application/json"},
            content=_jsonrpc_body(2),
        )

    async def run() -> tuple[int, bool, bool, bool, bool, BaseException]:
        session = _session(
            handler,
            limits=_limits(),
            secret_redactor=SecretRedactor(secret),
        )
        try:
            task = asyncio.create_task(session.call_tool("echo", {}))
            await blocked.started.wait()
            task.cancel(cancellation_detail)
            cancelling = task.cancelling()
            with pytest.raises(asyncio.CancelledError) as exc_info:
                await task
            settlement = _http_settlement_task(exc_info.value)
            assert settlement is not None
            await settlement
            with pytest.raises(McpProtocolError, match="closed"):
                await session.call_tool("echo", {})
            return (
                cancelling,
                task.cancelled(),
                blocked.closed,
                session._closed,
                session._http.is_closed,
                exc_info.value,
            )
        finally:
            await session.close()

    cancelling, cancelled, response_closed, session_closed, client_closed, error = asyncio.run(
        run()
    )

    assert cancelling == 1
    assert cancelled is True
    assert response_closed is True
    assert session_closed is True
    assert client_closed is True
    assert cancellation_detail.render_calls == 0
    assert error.args == ("MCP operation cancelled",)
    assert secret not in "".join(traceback.format_exception(error))


def test_http_repeated_cancellation_during_cleanup_fences_session_and_closes_client() -> None:
    stream = _FirstCloseBlocksStream([_jsonrpc_body(1)])

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"content-type": "application/json"},
            stream=stream,
        )

    async def run() -> tuple[int, bool, bool, bool]:
        session = _session(
            handler,
            limits=_limits(idle_timeout_s=0.5, total_call_timeout_s=2),
        )
        try:
            task = asyncio.create_task(session.call_tool("echo", {}))
            await stream.close_started.wait()
            task.cancel()
            task.cancel()
            cancelling = task.cancelling()
            with pytest.raises(asyncio.CancelledError) as exc_info:
                await task
            settlement = _http_settlement_task(exc_info.value)
            assert settlement is not None
            stream.release_close.set()
            await settlement
            return cancelling, task.cancelled(), session._closed, session._http.is_closed
        finally:
            await session.close()

    cancelling, cancelled, session_closed, client_closed = asyncio.run(run())

    assert cancelling == 2
    assert cancelled is True
    assert session_closed is True
    assert client_closed is True


@pytest.mark.parametrize(
    ("headers", "body", "error_type", "message"),
    [
        ({"content-encoding": "gzip"}, b"compressed", McpProtocolError, "encoding"),
        ({"content-length": "9999"}, b"", McpResponseTooLargeError, "response exceeded"),
        (
            {"content-length": "9" * 5_000},
            b"",
            McpResponseTooLargeError,
            "response exceeded",
        ),
        ({"content-type": "application/json"}, b"\xff", McpProtocolError, "UTF-8"),
        ({"content-type": "application/json"}, b"", McpPeerClosedError, "peer closed"),
    ],
)
def test_http_rejects_unsupported_or_malformed_bounded_responses(
    headers: dict[str, str],
    body: bytes,
    error_type: type[BaseException],
    message: str,
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        merged_headers = {"content-type": "application/json", **headers}
        if "content-length" in headers or "content-encoding" in headers:
            return httpx.Response(200, headers=merged_headers, stream=_ChunkStream([body]))
        return httpx.Response(200, headers=merged_headers, content=body)

    async def run() -> None:
        session = _session(
            handler,
            limits=_limits(max_message_bytes=256, max_response_bytes=512),
        )
        try:
            with pytest.raises(error_type, match=message):
                await session.call_tool("echo", {})
        finally:
            await session.close()

    asyncio.run(run())


def test_http_rejected_headers_cannot_bypass_404_session_fencing() -> None:
    methods: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "DELETE":
            methods.append("DELETE")
            return httpx.Response(200)
        payload = json.loads(request.content)
        method = payload["method"]
        methods.append(method)
        if method == "initialize":
            return httpx.Response(
                200,
                headers={
                    "content-type": "application/json",
                    MCP_SESSION_ID_HEADER: "expired-session",
                },
                content=_initialize_body(payload["id"]),
            )
        if method == "notifications/initialized":
            return httpx.Response(202)
        assert method == "tools/call"
        return httpx.Response(
            404,
            headers={
                "content-type": "application/json",
                "content-encoding": "gzip",
            },
            stream=_ChunkStream([b"encoded error"]),
        )

    async def run() -> None:
        session = await HttpMcpClient(
            transport=httpx.MockTransport(handler),
            transport_limits=_limits(),
        ).connect(McpServerSpec(name="expired-session", url="https://mcp.example/rpc"))
        try:
            with pytest.raises(McpProtocolError, match="unsupported content encoding"):
                await session.call_tool("first", {})
            with pytest.raises(McpProtocolError, match="closed"):
                await session.call_tool("second", {})
        finally:
            await session.close()

    asyncio.run(run())

    assert methods == ["initialize", "notifications/initialized", "tools/call"]


@pytest.mark.parametrize(
    ("header_mode", "error_type"),
    [
        ("content-encoding", McpProtocolError),
        ("content-length", McpProtocolError),
        ("session-authority", McpMessageTooLargeError),
    ],
)
def test_http_rejected_response_does_not_retain_private_header_values(
    header_mode: str,
    error_type: type[BaseException],
    caplog: pytest.LogCaptureFixture,
    capsys: pytest.CaptureFixture[str],
    recwarn: pytest.WarningsRecorder,
) -> None:
    secret = f"mcp-http-{header_mode}-header-canary"

    def handler(request: httpx.Request) -> httpx.Response:
        if header_mode == "content-encoding":
            headers = {
                "content-type": "application/json",
                "content-encoding": secret,
            }
            body = b"{}"
        elif header_mode == "content-length":
            headers = {
                "content-type": "application/json",
                "content-length": secret,
            }
            body = b"{}"
        else:
            headers = {
                "content-type": f"application/json; private={secret}",
                "mcp-session-id": secret,
            }
            body = b"x" * 300
        return httpx.Response(200, headers=headers, stream=_ChunkStream([body]))

    async def run() -> BaseException:
        session = HttpMcpSession(
            server=McpServerSpec(name="private-headers", url="https://mcp.example/rpc"),
            http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
            url="https://mcp.example/rpc",
            client_name="cayu-test",
            client_version="1",
            transport_limits=_limits(max_message_bytes=256, max_response_bytes=512),
            secret_redactor=SecretRedactor(secret),
        )
        try:
            with pytest.raises(error_type) as exc_info:
                await session.call_tool("echo", {})
            return exc_info.value
        finally:
            await session.close()

    with caplog.at_level(logging.DEBUG):
        error = asyncio.run(run())

    assert secret not in "".join(traceback.format_exception(error))
    traceback_cursor = error.__traceback__
    while traceback_cursor is not None:
        if is_cayu_source_filename(traceback_cursor.tb_frame.f_code.co_filename):
            assert all(
                secret not in repr(value) for value in traceback_cursor.tb_frame.f_locals.values()
            )
        traceback_cursor = traceback_cursor.tb_next
    captured = capsys.readouterr()
    assert secret not in captured.out
    assert secret not in captured.err
    assert all(secret not in record.getMessage() for record in caplog.records)
    assert all(secret not in str(warning.message) for warning in recwarn)


def test_http_forces_redirects_off_even_for_redirect_enabled_client() -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(302, headers={"location": "https://mcp.example/again"})

    async def run() -> None:
        client = httpx.AsyncClient(
            transport=httpx.MockTransport(handler),
            follow_redirects=True,
        )
        session = HttpMcpSession(
            server=McpServerSpec(name="redirect", url="https://mcp.example/rpc"),
            http_client=client,
            url="https://mcp.example/rpc",
            client_name="cayu-test",
            client_version="1",
            transport_limits=_limits(),
        )
        try:
            with pytest.raises(McpProtocolError, match="HTTP 302"):
                await session.call_tool("echo", {})
        finally:
            await session.close()

    asyncio.run(run())

    assert calls == 1


def test_http_redirect_jsonrpc_body_is_rejected_without_publishing_session_id() -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls == 1:
            return httpx.Response(
                302,
                headers={
                    "content-type": "application/json",
                    "location": "https://mcp.example/again",
                    MCP_SESSION_ID_HEADER: "redirect-session",
                },
                content=_jsonrpc_body(1),
            )
        return httpx.Response(
            200,
            headers={"content-type": "application/json"},
            content=_jsonrpc_body(2),
        )

    async def run() -> tuple[str | None, bool]:
        session = _session(handler, limits=_limits())
        try:
            with pytest.raises(McpProtocolError, match="HTTP 302"):
                await session.call_tool("redirected", {})
            session_id = session._session_id
            result = await session.call_tool("reused", {})
            return session_id, result.content[0]["text"] == ""
        finally:
            await session.close()

    session_id, reused = asyncio.run(run())

    assert session_id is None
    assert reused is True
    assert calls == 2


def test_http_initialized_notification_redirect_does_not_publish_initialization() -> None:
    events: list[str] = []
    delete_headers: httpx.Headers | None = None

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal delete_headers
        if request.method == "DELETE":
            events.append("delete")
            delete_headers = request.headers
            return httpx.Response(200)
        payload = json.loads(request.content)
        method = payload["method"]
        events.append(method)
        if method == "initialize":
            return httpx.Response(
                200,
                headers={
                    "content-type": "application/json",
                    MCP_SESSION_ID_HEADER: "redirect-initialize-session",
                },
                content=_initialize_body(payload["id"]),
            )
        return httpx.Response(
            302,
            headers={"location": "https://mcp.example/again"},
        )

    async def run() -> tuple[bool, bool]:
        session = _session(handler, limits=_limits())
        with pytest.raises(McpProtocolError, match="HTTP 302") as exc_info:
            await session.initialize()
        settlement = _http_settlement_task(exc_info.value)
        assert settlement is not None
        await settlement
        with pytest.raises(McpProtocolError, match="not been initialized"):
            _ = session.initialize_result
        return session._closed, session._http.is_closed

    session_closed, client_closed = asyncio.run(run())

    assert events == ["initialize", "notifications/initialized", "delete"]
    assert delete_headers is not None
    assert delete_headers[MCP_SESSION_ID_HEADER] == "redirect-initialize-session"
    assert delete_headers[MCP_PROTOCOL_VERSION_HEADER] == MCP_PROTOCOL_VERSION
    assert session_closed is True
    assert client_closed is True


def test_http_initialized_notification_404_clears_cleanup_authority_without_delete() -> None:
    events: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "DELETE":
            events.append("delete")
            raise AssertionError("an expired initialization session must not be deleted")
        payload = json.loads(request.content)
        method = payload["method"]
        events.append(method)
        if method == "initialize":
            return httpx.Response(
                200,
                headers={
                    "content-type": "application/json",
                    MCP_SESSION_ID_HEADER: "expired-initialize-session",
                },
                content=_initialize_body(payload["id"]),
            )
        return httpx.Response(
            404,
            headers={
                "content-type": "application/json",
                "content-encoding": "gzip",
            },
            stream=_ChunkStream([b"expired"]),
        )

    async def run() -> tuple[str | None, str | None, bool]:
        session = _session(handler, limits=_limits())
        with pytest.raises(McpProtocolError, match="unsupported content encoding") as exc_info:
            await session.initialize()
        settlement = _http_settlement_task(exc_info.value)
        assert settlement is not None
        await settlement
        return session._session_id, session._cleanup_session_id, session._http.is_closed

    session_id, cleanup_session_id, client_closed = asyncio.run(run())

    assert events == ["initialize", "notifications/initialized"]
    assert session_id is None
    assert cleanup_session_id is None
    assert client_closed is True


def test_http_initialize_keeps_session_authority_private_until_notification_succeeds() -> None:
    events: list[str] = []

    async def run() -> tuple[str | None, bool, httpx.Headers | None]:
        notification_started = asyncio.Event()
        release_notification = asyncio.Event()
        delete_headers: httpx.Headers | None = None

        async def handler(request: httpx.Request) -> httpx.Response:
            nonlocal delete_headers
            if request.method == "DELETE":
                events.append("delete")
                delete_headers = request.headers
                return httpx.Response(200)
            payload = json.loads(request.content)
            method = payload["method"]
            events.append(method)
            if method == "initialize":
                return httpx.Response(
                    200,
                    headers={
                        "content-type": "application/json",
                        MCP_SESSION_ID_HEADER: "private-initialize-session",
                    },
                    content=_initialize_body(payload["id"]),
                )
            if method == "notifications/initialized":
                assert request.headers[MCP_SESSION_ID_HEADER] == "private-initialize-session"
                assert request.headers[MCP_PROTOCOL_VERSION_HEADER] == MCP_PROTOCOL_VERSION
                notification_started.set()
                await release_notification.wait()
                return httpx.Response(302, headers={"location": "https://mcp.example/again"})
            raise AssertionError(f"unexpected MCP method {method}")

        session = HttpMcpSession(
            server=McpServerSpec(name="initializing", url="https://mcp.example/rpc"),
            http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
            url="https://mcp.example/rpc",
            client_name="cayu-test",
            client_version="1",
            transport_limits=_limits(),
        )
        initialize_task = asyncio.create_task(session.initialize())
        await asyncio.wait_for(notification_started.wait(), timeout=0.1)
        unpublished_session_id = session._session_id
        with pytest.raises(McpProtocolError, match="initialization is still in progress"):
            await session.call_tool("must-not-dispatch", {})
        rejected_before_dispatch = events == ["initialize", "notifications/initialized"]
        release_notification.set()
        with pytest.raises(McpProtocolError, match="HTTP 302") as exc_info:
            await initialize_task
        settlement = _http_settlement_task(exc_info.value)
        assert settlement is not None
        await settlement
        return unpublished_session_id, rejected_before_dispatch, delete_headers

    unpublished_session_id, rejected_before_dispatch, delete_headers = asyncio.run(run())

    assert unpublished_session_id is None
    assert rejected_before_dispatch is True
    assert events == ["initialize", "notifications/initialized", "delete"]
    assert delete_headers is not None
    assert delete_headers[MCP_SESSION_ID_HEADER] == "private-initialize-session"
    assert delete_headers[MCP_PROTOCOL_VERSION_HEADER] == MCP_PROTOCOL_VERSION


def test_http_close_fences_late_tool_response_and_waits_for_its_exchange() -> None:
    events: list[str] = []

    async def run() -> tuple[bool, bool, bool]:
        request_started = asyncio.Event()
        release_request = asyncio.Event()

        async def handler(request: httpx.Request) -> httpx.Response:
            payload = json.loads(request.content)
            events.append(payload["method"])
            request_started.set()
            await release_request.wait()
            return httpx.Response(
                200,
                headers={"content-type": "application/json"},
                content=_jsonrpc_body(payload["id"]),
            )

        session = _session(
            handler,
            limits=_limits(idle_timeout_s=0.5, total_call_timeout_s=1.0),
        )
        request_task = asyncio.create_task(session.call_tool("late", {}))
        await asyncio.wait_for(request_started.wait(), timeout=0.1)
        close_task = asyncio.create_task(session.close())
        await asyncio.sleep(0)
        close_waited_for_exchange = not close_task.done()
        release_request.set()
        with pytest.raises(McpProtocolError, match="closed before the response"):
            await request_task
        await asyncio.wait_for(close_task, timeout=0.2)
        return close_waited_for_exchange, session._closed, session._http.is_closed

    close_waited, session_closed, client_closed = asyncio.run(run())

    assert events == ["tools/call"]
    assert close_waited is True
    assert session_closed is True
    assert client_closed is True


def test_http_close_uses_late_tool_response_session_authority() -> None:
    events: list[str] = []

    async def run() -> tuple[str | None, str | None]:
        request_started = asyncio.Event()
        release_request = asyncio.Event()
        delete_session_id: str | None = None

        async def handler(request: httpx.Request) -> httpx.Response:
            nonlocal delete_session_id
            if request.method == "DELETE":
                events.append("delete")
                delete_session_id = request.headers.get(MCP_SESSION_ID_HEADER)
                return httpx.Response(200)
            payload = json.loads(request.content)
            events.append(payload["method"])
            request_started.set()
            await release_request.wait()
            return httpx.Response(
                200,
                headers={
                    "content-type": "application/json",
                    MCP_SESSION_ID_HEADER: "replacement-session",
                },
                content=_jsonrpc_body(payload["id"]),
            )

        session = _session(
            handler,
            limits=_limits(idle_timeout_s=0.5, total_call_timeout_s=1.0),
        )
        session._initialize_result = McpInitializeResult(protocol_version=MCP_PROTOCOL_VERSION)
        session._session_id = "original-session"
        request_task = asyncio.create_task(session.call_tool("late-authority", {}))
        await asyncio.wait_for(request_started.wait(), timeout=0.1)
        close_task = asyncio.create_task(session.close())
        await asyncio.sleep(0)
        release_request.set()
        with pytest.raises(McpProtocolError, match="closed before the response"):
            await request_task
        await asyncio.wait_for(close_task, timeout=0.2)
        return delete_session_id, session._session_id

    deleted_session_id, retained_session_id = asyncio.run(run())

    assert events == ["tools/call", "delete"]
    assert deleted_session_id == "replacement-session"
    assert retained_session_id is None


def test_http_close_uses_late_initialized_notification_session_authority() -> None:
    events: list[str] = []

    async def run() -> tuple[str | None, str | None]:
        notification_started = asyncio.Event()
        release_notification = asyncio.Event()
        delete_session_id: str | None = None

        async def handler(request: httpx.Request) -> httpx.Response:
            nonlocal delete_session_id
            if request.method == "DELETE":
                events.append("delete")
                delete_session_id = request.headers.get(MCP_SESSION_ID_HEADER)
                return httpx.Response(200)
            payload = json.loads(request.content)
            method = payload["method"]
            events.append(method)
            if method == "initialize":
                return httpx.Response(
                    200,
                    headers={
                        "content-type": "application/json",
                        MCP_SESSION_ID_HEADER: "initial-session",
                    },
                    content=_initialize_body(payload["id"]),
                )
            notification_started.set()
            await release_notification.wait()
            return httpx.Response(
                202,
                headers={MCP_SESSION_ID_HEADER: "notification-session"},
            )

        session = _session(
            handler,
            limits=_limits(idle_timeout_s=0.5, total_call_timeout_s=1.0),
        )
        initialize_task = asyncio.create_task(session.initialize())
        await asyncio.wait_for(notification_started.wait(), timeout=0.1)
        close_task = asyncio.create_task(session.close())
        await asyncio.sleep(0)
        release_notification.set()
        with pytest.raises(McpProtocolError, match="closed before the notification"):
            await initialize_task
        await asyncio.wait_for(close_task, timeout=0.2)
        return delete_session_id, session._cleanup_session_id

    deleted_session_id, retained_cleanup_session_id = asyncio.run(run())

    assert events == ["initialize", "notifications/initialized", "delete"]
    assert deleted_session_id == "notification-session"
    assert retained_cleanup_session_id is None


def test_http_fenced_close_scrubs_session_authority_before_client_close_failure(
    caplog: pytest.LogCaptureFixture,
    capsys: pytest.CaptureFixture[str],
    recwarn: pytest.WarningsRecorder,
) -> None:
    secret = "fenced-close-session-authority-secret"

    class FailingCloseTransport(httpx.AsyncBaseTransport):
        async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
            raise AssertionError("fenced close must not dispatch DELETE")

        async def aclose(self) -> None:
            raise RuntimeError(f"client close failed near {secret}")

    async def run() -> tuple[BaseException, str | None]:
        session = HttpMcpSession(
            server=McpServerSpec(name="fenced-close", url="https://mcp.example/rpc"),
            http_client=httpx.AsyncClient(transport=FailingCloseTransport()),
            url="https://mcp.example/rpc",
            client_name="cayu-test",
            client_version="1",
            transport_limits=_limits(),
            secret_redactor=SecretRedactor(secret),
        )
        session._initialize_result = McpInitializeResult(protocol_version=MCP_PROTOCOL_VERSION)
        session._session_id = secret
        session._fenced = True
        with pytest.raises(McpProtocolError, match="client cleanup failed") as exc_info:
            await session.close()
        assert session._close_task is not None
        retained_failure = session._close_task.exception()
        assert retained_failure is not None
        _assert_cayu_traceback_does_not_retain(retained_failure, secret)
        return exc_info.value, session._session_id

    with caplog.at_level(logging.DEBUG):
        error, retained_session_id = asyncio.run(run())

    assert retained_session_id is None
    assert secret not in repr(error)
    assert secret not in "".join(traceback.format_exception(error))
    _assert_cayu_traceback_does_not_retain(error, secret)
    captured = capsys.readouterr()
    assert secret not in captured.out
    assert secret not in captured.err
    assert all(secret not in record.getMessage() for record in caplog.records)
    assert all(secret not in str(warning.message) for warning in recwarn)


def test_http_fenced_sibling_settlement_waits_for_healthy_exchange() -> None:
    async def run() -> tuple[bool, bool, bool, int]:
        healthy_started = asyncio.Event()
        release_healthy = asyncio.Event()
        close_calls = 0

        async def handler(request: httpx.Request) -> httpx.Response:
            payload = json.loads(request.content)
            if payload["method"] == "tools/call" and payload["params"]["name"] == "healthy":
                healthy_started.set()
                await release_healthy.wait()
                return httpx.Response(
                    200,
                    headers={"content-type": "application/json"},
                    content=_jsonrpc_body(payload["id"]),
                )
            raise httpx.ReadError("injected sibling failure", request=request)

        class CountingTransport(httpx.MockTransport):
            async def aclose(self) -> None:
                nonlocal close_calls
                close_calls += 1
                await super().aclose()

        session = HttpMcpSession(
            server=McpServerSpec(name="sibling-fence", url="https://mcp.example/rpc"),
            http_client=httpx.AsyncClient(transport=CountingTransport(handler)),
            url="https://mcp.example/rpc",
            client_name="cayu-test",
            client_version="1",
            transport_limits=_limits(idle_timeout_s=0.5, total_call_timeout_s=1.0),
        )
        healthy_task = asyncio.create_task(session.call_tool("healthy", {}))
        await asyncio.wait_for(healthy_started.wait(), timeout=0.1)
        with pytest.raises(McpProtocolError) as failed:
            await session.call_tool("failing", {})
        settlement = _http_settlement_task(failed.value)
        assert settlement is not None
        await asyncio.sleep(0)
        cleanup_waited_for_sibling = not settlement.done() and close_calls == 0
        release_healthy.set()
        with pytest.raises(McpProtocolError, match="closed before the response"):
            await healthy_task
        await asyncio.gather(settlement, return_exceptions=True)
        return cleanup_waited_for_sibling, session._fenced, session._http.is_closed, close_calls

    cleanup_waited, session_fenced, client_closed, close_calls = asyncio.run(run())

    assert cleanup_waited is True
    assert session_fenced is True
    assert client_closed is True
    assert close_calls == 1


def test_http_notification_failure_scrubs_secret_cleanup_session_id(
    caplog: pytest.LogCaptureFixture,
    capsys: pytest.CaptureFixture[str],
    recwarn: pytest.WarningsRecorder,
) -> None:
    secret = "mcp-secret-cleanup-session-id-canary"
    events: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "DELETE":
            events.append("delete")
            assert request.headers[MCP_SESSION_ID_HEADER] == secret
            return httpx.Response(200)
        payload = json.loads(request.content)
        events.append(payload["method"])
        if payload["method"] == "initialize":
            return httpx.Response(
                200,
                headers={
                    "content-type": "application/json",
                    MCP_SESSION_ID_HEADER: secret,
                },
                content=_initialize_body(payload["id"]),
            )
        return httpx.Response(302, headers={"location": "https://mcp.example/again"})

    async def run() -> BaseException:
        session = _session(
            handler,
            limits=_limits(),
            secret_redactor=SecretRedactor(secret),
        )
        with pytest.raises(McpProtocolError, match="HTTP 302") as exc_info:
            await session.initialize()
        settlement = _http_settlement_task(exc_info.value)
        assert settlement is not None
        await settlement
        return exc_info.value

    with caplog.at_level(logging.DEBUG):
        error = asyncio.run(run())

    assert events == ["initialize", "notifications/initialized", "delete"]
    assert secret not in repr(error)
    assert secret not in "".join(traceback.format_exception(error))
    _assert_cayu_traceback_does_not_retain(error, secret)
    captured = capsys.readouterr()
    assert secret not in captured.out
    assert secret not in captured.err
    assert all(secret not in record.getMessage() for record in caplog.records)
    assert all(secret not in str(warning.message) for warning in recwarn)


def test_http_initialize_cleanup_reports_non_successful_delete_status() -> None:
    events: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "DELETE":
            events.append("delete")
            return httpx.Response(
                500,
                headers={"content-type": "application/json"},
                content=b'{"error":"cleanup rejected"}',
            )
        payload = json.loads(request.content)
        events.append(payload["method"])
        if payload["method"] == "initialize":
            return httpx.Response(
                200,
                headers={
                    "content-type": "application/json",
                    MCP_SESSION_ID_HEADER: "delete-status-session",
                },
                content=_initialize_body(payload["id"]),
            )
        return httpx.Response(302, headers={"location": "https://mcp.example/again"})

    async def run() -> tuple[BaseException, bool]:
        session = _session(handler, limits=_limits())
        with pytest.raises(McpProtocolError, match="HTTP 302") as exc_info:
            await session.initialize()
        settlement = _http_settlement_task(exc_info.value)
        assert settlement is not None
        with pytest.raises(McpProtocolError, match="HTTP 500") as settlement_exc:
            await settlement
        return settlement_exc.value, session._http.is_closed

    settlement_error, client_closed = asyncio.run(run())

    assert "cleanup rejected" in str(settlement_error)
    assert events == ["initialize", "notifications/initialized", "delete"]
    assert client_closed is True


@pytest.mark.parametrize("representation_failure", ["header", "body"])
def test_http_initialize_cleanup_preserves_status_before_representation_and_client_failures(
    caplog: pytest.LogCaptureFixture,
    capsys: pytest.CaptureFixture[str],
    recwarn: pytest.WarningsRecorder,
    representation_failure: str,
) -> None:
    representation_secret = f"delete-{representation_failure}-failure-secret"
    client_secret = "delete-status-client-close-secret"
    events: list[str] = []

    class FailingBodyStream(httpx.AsyncByteStream):
        async def __aiter__(self) -> AsyncIterator[bytes]:
            raise RuntimeError(f"DELETE body exposed {representation_secret}")
            yield b""  # pragma: no cover - makes this an async generator

        async def aclose(self) -> None:
            return None

    class CleanupFailureTransport(httpx.AsyncBaseTransport):
        async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
            if request.method == "DELETE":
                events.append("delete")
                if representation_failure == "header":
                    return httpx.Response(
                        500,
                        headers={"content-encoding": f"gzip-{representation_secret}"},
                        stream=_ChunkStream([b"rejected"]),
                    )
                return httpx.Response(
                    500,
                    headers={"content-type": "application/json"},
                    stream=FailingBodyStream(),
                )
            payload = json.loads(request.content)
            method = payload["method"]
            events.append(method)
            if method == "initialize":
                return httpx.Response(
                    200,
                    headers={
                        "content-type": "application/json",
                        MCP_SESSION_ID_HEADER: "status-evidence-session",
                    },
                    content=_initialize_body(payload["id"]),
                )
            return httpx.Response(302, headers={"location": "https://mcp.example/again"})

        async def aclose(self) -> None:
            events.append("close")
            raise RuntimeError(f"client close exposed {client_secret}")

    async def run() -> tuple[BaseException, BaseException]:
        session = HttpMcpSession(
            server=McpServerSpec(name="delete-status-evidence", url="https://mcp.example/rpc"),
            http_client=httpx.AsyncClient(transport=CleanupFailureTransport()),
            url="https://mcp.example/rpc",
            client_name="cayu-test",
            client_version="1",
            transport_limits=_limits(),
            secret_redactor=SecretRedactor([representation_secret, client_secret]),
        )
        with pytest.raises(McpProtocolError, match="HTTP 302") as exc_info:
            await session.initialize()
        settlement = _http_settlement_task(exc_info.value)
        assert settlement is not None
        settlement_outcome = (await asyncio.gather(settlement, return_exceptions=True))[0]
        assert isinstance(settlement_outcome, BaseException)
        return exc_info.value, settlement_outcome

    with caplog.at_level(logging.DEBUG):
        public_error, settlement_error = asyncio.run(run())

    rendered = "".join(traceback.format_exception(settlement_error))
    representation_text = (
        "unsupported content encoding"
        if representation_failure == "header"
        else "DELETE body exposed"
    )
    assert "HTTP 500" in rendered
    assert representation_text in rendered
    assert "client cleanup failed" in rendered
    assert rendered.index("HTTP 500") < rendered.index(representation_text)
    assert rendered.index(representation_text) < rendered.index("client cleanup failed")
    assert REDACTED_SECRET in rendered
    assert representation_secret not in rendered
    assert client_secret not in rendered
    _assert_cayu_traceback_does_not_retain(public_error, representation_secret, client_secret)
    _assert_cayu_traceback_does_not_retain(
        settlement_error,
        representation_secret,
        client_secret,
    )
    captured = capsys.readouterr()
    assert representation_secret not in captured.out
    assert representation_secret not in captured.err
    assert client_secret not in captured.out
    assert client_secret not in captured.err
    assert all(representation_secret not in record.getMessage() for record in caplog.records)
    assert all(client_secret not in record.getMessage() for record in caplog.records)
    assert all(representation_secret not in str(warning.message) for warning in recwarn)
    assert all(client_secret not in str(warning.message) for warning in recwarn)
    assert events == ["initialize", "notifications/initialized", "delete", "close"]


def test_http_failed_initialization_reports_nested_delete_settlement_failure() -> None:
    secret = "nested-initialize-delete-settlement-secret"
    delete_stream = _FirstCloseBlocksThenFailsStream([b""], RuntimeError(secret))
    events: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "DELETE":
            events.append("delete")
            return httpx.Response(200, stream=delete_stream)
        events.append("initialize")
        payload = json.loads(request.content)
        return httpx.Response(
            200,
            headers={
                "content-type": "application/json",
                "content-encoding": "gzip",
                MCP_SESSION_ID_HEADER: "nested-delete-session",
            },
            stream=_ChunkStream([_initialize_body(payload["id"])]),
        )

    async def run() -> tuple[BaseException, bool]:
        connect_task = asyncio.create_task(
            HttpMcpClient(
                transport=httpx.MockTransport(handler),
                transport_limits=_limits(idle_timeout_s=0.5, total_call_timeout_s=0.1),
                secret_resolver=StaticVault({"token": secret}),
            ).connect(
                McpServerSpec(
                    name="nested-delete-failure",
                    url="https://mcp.example/rpc",
                    secret_headers={"authorization": SecretRef(name="token")},
                )
            )
        )
        await asyncio.wait_for(delete_stream.close_started.wait(), timeout=0.5)
        assert connect_task.done() is False
        # Let the bounded DELETE owner transfer its still-running context exit
        # before releasing it, so the retained settlement retry is exercised.
        await asyncio.sleep(0.12)
        delete_stream.release_close.set()
        with pytest.raises(McpProtocolError, match="unsupported content encoding") as exc_info:
            await connect_task
        return exc_info.value, delete_stream.overlapping_close

    error, overlapping_close = asyncio.run(run())

    assert isinstance(error.__cause__, BaseExceptionGroup)
    assert len(error.__cause__.exceptions) == 2
    termination_error, settlement_error = error.__cause__.exceptions
    assert type(termination_error) is McpCallDeadlineExceededError
    assert type(settlement_error) is McpProtocolError
    assert "server-session termination failed" in str(termination_error)
    assert "server-session settlement failed" in str(settlement_error)
    assert "stream close retry failed" in str(settlement_error)
    assert REDACTED_SECRET in str(settlement_error)
    assert secret not in "".join(traceback.format_exception(error))
    _assert_cayu_traceback_does_not_retain(error, secret)
    assert events == ["initialize", "delete"]
    assert delete_stream.close_calls == 2
    assert overlapping_close is False


def test_http_session_delete_body_is_idle_bounded_and_cleaned_up() -> None:
    blocked = _BlockingStream()
    methods: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        methods.append(request.method)
        return httpx.Response(200, stream=blocked)

    async def run() -> bool:
        session = _session(
            handler,
            limits=_limits(idle_timeout_s=0.02, total_call_timeout_s=0.1),
        )
        session._session_id = "session-to-delete"
        await asyncio.wait_for(session.close(), timeout=0.2)
        return session._http.is_closed

    client_closed = asyncio.run(run())

    assert methods == ["DELETE"]
    assert blocked.closed is True
    assert client_closed is True


def test_http_session_delete_cleanup_uses_original_total_deadline() -> None:
    stream = _FirstCloseBlocksStream([b""])

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "DELETE"
        return httpx.Response(200, stream=stream)

    async def run() -> tuple[bool, bool, int, bool, bool]:
        session = _session(
            handler,
            limits=_limits(idle_timeout_s=0.5, total_call_timeout_s=0.02),
        )
        session._session_id = "session-to-delete"
        close_task = asyncio.create_task(session.close())
        await asyncio.wait_for(stream.close_started.wait(), timeout=0.1)
        done, _ = await asyncio.wait({close_task}, timeout=0.15)
        returned_before_release = close_task in done
        settlement_was_retained = any(not task.done() for task in session._settlement_tasks)
        stream.release_close.set()
        await asyncio.wait_for(close_task, timeout=0.2)
        pending_settlements = tuple(session._settlement_tasks)
        if pending_settlements:
            await asyncio.wait_for(
                asyncio.gather(*pending_settlements, return_exceptions=True),
                timeout=0.2,
            )
        return (
            returned_before_release,
            settlement_was_retained,
            stream.close_calls,
            stream.overlapping_close,
            session._http.is_closed,
        )

    (
        returned_before_release,
        settlement_was_retained,
        close_calls,
        overlapping_close,
        client_closed,
    ) = asyncio.run(run())

    assert returned_before_release is True
    assert settlement_was_retained is True
    assert close_calls == 2
    assert overlapping_close is False
    assert client_closed is True


def test_http_grouped_delete_failure_does_not_skip_client_close() -> None:
    transport = _GroupedDeleteFailureTransport()

    async def run() -> tuple[bool, bool]:
        http_client = httpx.AsyncClient(transport=transport)
        session = HttpMcpSession(
            server=McpServerSpec(name="grouped-delete", url="https://mcp.example/rpc"),
            http_client=http_client,
            url="https://mcp.example/rpc",
            client_name="cayu-test",
            client_version="1",
            transport_limits=_limits(),
        )
        session._session_id = "session-to-delete"
        await session.close()
        await session.close()
        return session._closed, http_client.is_closed

    session_closed, client_closed = asyncio.run(run())

    assert session_closed is True
    assert client_closed is True
    assert transport.close_calls == 1


def test_http_session_close_propagates_redacted_client_cleanup_failure(
    caplog: pytest.LogCaptureFixture,
    capsys: pytest.CaptureFixture[str],
    recwarn: pytest.WarningsRecorder,
) -> None:
    secret = "mcp-http-client-close-secret-canary"
    raw_close_failure = RuntimeError(f"client close exposed {secret}")
    transport = _FailingCloseTransport(raw_close_failure)

    async def run() -> tuple[BaseException, bool]:
        http_client = httpx.AsyncClient(transport=transport)
        session = HttpMcpSession(
            server=McpServerSpec(name="close-secret", url="https://mcp.example/rpc"),
            http_client=http_client,
            url="https://mcp.example/rpc",
            client_name="cayu-test",
            client_version="1",
            transport_limits=_limits(),
            secret_redactor=SecretRedactor(secret),
        )
        with pytest.raises(McpProtocolError, match="MCP HTTP client cleanup failed") as exc_info:
            await session.close()
        return exc_info.value, http_client.is_closed

    with caplog.at_level(logging.DEBUG):
        error, client_closed = asyncio.run(run())

    assert transport.close_calls == 1
    assert client_closed is True
    assert error is not raw_close_failure
    assert error.__cause__ is None
    assert error.__context__ is None
    assert secret not in "".join(traceback.format_exception(error))
    traceback_cursor = error.__traceback__
    while traceback_cursor is not None:
        if is_cayu_source_filename(traceback_cursor.tb_frame.f_code.co_filename):
            assert all(
                secret not in repr(value) for value in traceback_cursor.tb_frame.f_locals.values()
            )
        traceback_cursor = traceback_cursor.tb_next
    captured = capsys.readouterr()
    assert secret not in captured.out
    assert secret not in captured.err
    assert all(secret not in record.getMessage() for record in caplog.records)
    assert all(secret not in str(warning.message) for warning in recwarn)


def test_http_initialize_failure_remains_authoritative_when_client_cleanup_fails() -> None:
    cleanup_failure = RuntimeError("initialize client cleanup failed")
    transport = _FailingCloseTransport(cleanup_failure)

    async def run() -> BaseException:
        with pytest.raises(McpMessageTooLargeError) as exc_info:
            await HttpMcpClient(
                transport=transport,
                transport_limits=_limits(
                    max_message_bytes=32,
                    max_response_bytes=32,
                ),
            ).connect(McpServerSpec(name="bounded-init", url="https://mcp.example/rpc"))
        return exc_info.value

    primary_error = asyncio.run(run())

    assert type(primary_error) is McpMessageTooLargeError
    assert isinstance(primary_error.__cause__, McpProtocolError)
    assert "initialize client cleanup failed" in str(primary_error.__cause__)
    assert transport.close_calls == 1


def test_http_close_cancellation_remains_authoritative_when_client_cleanup_fails() -> None:
    secret = "mcp-http-cancelled-close-secret-canary"
    transport = _BlockingFailingCloseTransport(RuntimeError(f"cleanup exposed {secret}"))

    async def run() -> tuple[int, bool, BaseException | None]:
        session = HttpMcpSession(
            server=McpServerSpec(name="cancel-close", url="https://mcp.example/rpc"),
            http_client=httpx.AsyncClient(transport=transport),
            url="https://mcp.example/rpc",
            client_name="cayu-test",
            client_version="1",
            transport_limits=_limits(),
            secret_redactor=SecretRedactor(secret),
        )
        close_task = asyncio.create_task(session.close())
        await transport.close_started.wait()
        close_task.cancel("cancel MCP close")
        cancelling = close_task.cancelling()
        await asyncio.sleep(0)
        transport.release_close.set()
        with pytest.raises(asyncio.CancelledError, match="cancel MCP close") as exc_info:
            await close_task
        return cancelling, close_task.cancelled(), exc_info.value.__cause__

    cancelling, cancelled, cleanup_error = asyncio.run(run())

    assert cancelling == 1
    assert cancelled is True
    assert isinstance(cleanup_error, McpProtocolError)
    assert secret not in "".join(traceback.format_exception(cleanup_error))


def test_http_late_client_cleanup_failure_is_safe_and_visible_to_repeat_close() -> None:
    secret = "mcp-http-late-close-secret-canary"
    transport = _BlockingFailingCloseTransport(RuntimeError(f"late cleanup exposed {secret}"))

    async def run() -> tuple[BaseException, list[dict[str, Any]]]:
        loop = asyncio.get_running_loop()
        diagnostics: list[dict[str, Any]] = []
        previous_handler = loop.get_exception_handler()
        loop.set_exception_handler(lambda _loop, context: diagnostics.append(context))
        try:
            session = HttpMcpSession(
                server=McpServerSpec(name="late-close", url="https://mcp.example/rpc"),
                http_client=httpx.AsyncClient(transport=transport),
                url="https://mcp.example/rpc",
                client_name="cayu-test",
                client_version="1",
                transport_limits=_limits(
                    idle_timeout_s=0.5,
                    total_call_timeout_s=0.02,
                ),
                secret_redactor=SecretRedactor(secret),
            )
            first_close = asyncio.create_task(session.close())
            await transport.close_started.wait()
            await asyncio.wait_for(first_close, timeout=0.2)
            transport.release_close.set()
            client_close_task = session._client_close_task
            assert client_close_task is not None
            await asyncio.gather(client_close_task, return_exceptions=True)
            with pytest.raises(McpProtocolError) as exc_info:
                await session.close()
            await asyncio.sleep(0)
            return exc_info.value, diagnostics
        finally:
            loop.set_exception_handler(previous_handler)

    error, diagnostics = asyncio.run(run())

    assert secret not in "".join(traceback.format_exception(error))
    assert diagnostics == []
    assert transport.close_calls == 1


def test_http_oversize_failure_does_not_expose_secret_diagnostics(
    caplog: pytest.LogCaptureFixture,
    capsys: pytest.CaptureFixture[str],
    recwarn: pytest.WarningsRecorder,
) -> None:
    secret = "mcp-http-limit-secret-canary"
    oversized = _jsonrpc_body(1, padding=500) + secret.encode()

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"content-type": "application/json"},
            stream=_ChunkStream([oversized[:256], oversized[256:]]),
        )

    async def run() -> BaseException:
        client = HttpMcpClient(
            transport=httpx.MockTransport(handler),
            transport_limits=_limits(max_message_bytes=256, max_response_bytes=1_024),
            secret_resolver=StaticVault({"token": secret}),
        )
        with pytest.raises(McpMessageTooLargeError) as excinfo:
            await client.connect(
                McpServerSpec(
                    name="secret-http",
                    url="https://mcp.example/rpc",
                    secret_headers={"authorization": SecretRef(name="token")},
                )
            )
        return excinfo.value

    with caplog.at_level(logging.DEBUG):
        error = asyncio.run(run())

    assert secret not in repr(error)
    traceback = error.__traceback__
    while traceback is not None:
        if is_cayu_source_filename(traceback.tb_frame.f_code.co_filename):
            assert all(secret not in repr(value) for value in traceback.tb_frame.f_locals.values())
        traceback = traceback.tb_next
    captured = capsys.readouterr()
    assert secret not in captured.out
    assert secret not in captured.err
    assert all(secret not in record.getMessage() for record in caplog.records)
    assert all(secret not in str(warning.message) for warning in recwarn)


def test_http_timeout_drops_partial_secret_from_diagnostic_channels(
    caplog: pytest.LogCaptureFixture,
    capsys: pytest.CaptureFixture[str],
    recwarn: pytest.WarningsRecorder,
) -> None:
    secret = "mcp-http-timeout-secret-canary"
    stream = _ChunkThenBlocksStream(secret.encode())

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"content-type": "application/json"},
            stream=stream,
        )

    async def run() -> BaseException:
        client = HttpMcpClient(
            transport=httpx.MockTransport(handler),
            transport_limits=_limits(idle_timeout_s=0.02, total_call_timeout_s=0.2),
            secret_resolver=StaticVault({"token": secret}),
        )
        with pytest.raises(McpIdleTimeoutError) as exc_info:
            await client.connect(
                McpServerSpec(
                    name="secret-timeout-http",
                    url="https://mcp.example/rpc",
                    secret_headers={"authorization": SecretRef(name="token")},
                )
            )
        settlement = _http_settlement_task(exc_info.value)
        assert settlement is not None
        await settlement
        return exc_info.value

    with caplog.at_level(logging.DEBUG):
        error = asyncio.run(run())

    assert secret not in repr(error)
    traceback = error.__traceback__
    while traceback is not None:
        if is_cayu_source_filename(traceback.tb_frame.f_code.co_filename):
            assert all(secret not in repr(value) for value in traceback.tb_frame.f_locals.values())
        traceback = traceback.tb_next
    captured = capsys.readouterr()
    assert secret not in captured.out
    assert secret not in captured.err
    assert all(secret not in record.getMessage() for record in caplog.records)
    assert all(secret not in str(warning.message) for warning in recwarn)


def test_http_initialized_notification_timeout_redacts_response_headers(
    caplog: pytest.LogCaptureFixture,
    capsys: pytest.CaptureFixture[str],
    recwarn: pytest.WarningsRecorder,
) -> None:
    secret = "mcp-http-notification-timeout-secret"
    stream = _BlockingStream()
    request_count = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal request_count
        request_count += 1
        assert request.headers["authorization"] == secret
        if request_count == 1:
            return httpx.Response(
                200,
                headers={"content-type": "application/json"},
                content=_initialize_body(1),
            )
        return httpx.Response(
            202,
            headers={"content-type": secret},
            stream=stream,
        )

    async def run() -> tuple[BaseException, bool]:
        client = HttpMcpClient(
            transport=httpx.MockTransport(handler),
            transport_limits=_limits(idle_timeout_s=0.02, total_call_timeout_s=0.2),
            secret_resolver=StaticVault({"token": secret}),
        )
        with pytest.raises(McpIdleTimeoutError) as exc_info:
            await client.connect(
                McpServerSpec(
                    name="secret-notification-timeout-http",
                    url="https://mcp.example/rpc",
                    secret_headers={"authorization": SecretRef(name="token")},
                )
            )
        settlement = _http_settlement_task(exc_info.value)
        assert settlement is not None
        await settlement
        return exc_info.value, stream.closed

    with caplog.at_level(logging.DEBUG):
        error, response_closed = asyncio.run(run())

    assert request_count == 2
    assert response_closed is True
    assert error.__context__ is None
    assert secret not in repr(error)
    assert secret not in "".join(traceback.format_exception(error))
    traceback_cursor = error.__traceback__
    while traceback_cursor is not None:
        if is_cayu_source_filename(traceback_cursor.tb_frame.f_code.co_filename):
            assert all(
                secret not in repr(value) for value in traceback_cursor.tb_frame.f_locals.values()
            )
        traceback_cursor = traceback_cursor.tb_next
    captured = capsys.readouterr()
    assert secret not in captured.out
    assert secret not in captured.err
    assert all(secret not in record.getMessage() for record in caplog.records)
    assert all(secret not in str(warning.message) for warning in recwarn)
