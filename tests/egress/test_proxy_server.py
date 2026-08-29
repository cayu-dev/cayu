from __future__ import annotations

import asyncio
import base64
import contextlib
import os
import socket
import ssl
import tempfile
import threading
import time
import warnings
from datetime import timedelta
from typing import Any

import httpx
import pytest
from pydantic import SecretStr

import cayu.egress.broker as broker_module
import cayu.egress.proxy_server as proxy_server_module
from cayu.egress import (
    ApprovedEgressDestination,
    BrowserEgressPolicy,
    CapturedRequest,
    CapturedResponse,
    EgressUpstreamLimits,
    EgressUpstreamOperation,
    HttpEgressPolicy,
    TransparentEgressBroker,
    VirtualCredentialRegistry,
)
from cayu.egress.broker import CAYU_EGRESS_ERROR_HEADER
from cayu.vaults import ResolvedSecret, SecretRef, StaticVault

pytest.importorskip("cryptography")

from cryptography import x509

from cayu.egress.proxy_server import (
    DualStackLoopbackEgressProxyServer,
    SessionCertificateAuthority,
    TransparentEgressProxyServer,
    _CertificateCapacityError,
    _ProxyProtocolError,
    _read_head,
    _serialize_response,
    _serialize_response_head,
)

REAL_SECRET = "sk_test_51RealProxySwapSecret"


def _cooperative_upstream_operation(factory: Any) -> EgressUpstreamOperation:
    async def cancel_and_wait(task: asyncio.Task[CapturedResponse]) -> None:
        task.cancel()
        with contextlib.suppress(BaseException):
            await task

    return EgressUpstreamOperation(factory, cancel_and_wait=cancel_and_wait)


@pytest.mark.parametrize(
    ("status_code", "reason"),
    [(504, "Gateway Timeout"), (599, "Unknown Status")],
)
def test_response_serializer_uses_safe_http_reason(status_code: int, reason: str) -> None:
    response = _serialize_response(CapturedResponse(status_code=status_code))

    assert response.startswith(f"HTTP/1.1 {status_code} {reason}\r\n".encode())


def _read_test_head(payload: bytes, *, timeout_s: float = 1.0) -> tuple[bytes, dict[str, str]]:
    reader, writer = socket.socketpair()
    try:
        writer.sendall(payload)
        writer.shutdown(socket.SHUT_WR)
        return _read_head(reader, timeout_s=timeout_s)
    finally:
        reader.close()
        writer.close()


@pytest.mark.parametrize(
    ("limit_name", "exact_payload", "oversized_payload", "error_code"),
    [
        (
            "_MAX_REQUEST_LINE_BYTES",
            b"G" * 16 + b"\r\n\r\n",
            b"G" * 17 + b"\r\n\r\n",
            "request_line_too_large",
        ),
        (
            "_MAX_HEADER_LINE_BYTES",
            b"GET / HTTP/1.1\r\nX:" + b"a" * 14 + b"\r\n\r\n",
            b"GET / HTTP/1.1\r\nX:" + b"a" * 15 + b"\r\n\r\n",
            "header_line_too_large",
        ),
    ],
)
def test_request_head_line_limits_accept_exact_boundary_and_reject_next_byte(
    monkeypatch: pytest.MonkeyPatch,
    limit_name: str,
    exact_payload: bytes,
    oversized_payload: bytes,
    error_code: str,
) -> None:
    monkeypatch.setattr(proxy_server_module, limit_name, 16)

    assert _read_test_head(exact_payload)[0]
    with pytest.raises(_ProxyProtocolError) as caught:
        _read_test_head(oversized_payload)
    assert caught.value.error_code == error_code


def test_request_head_total_limit_accepts_exact_boundary_and_rejects_next_byte(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    exact = b"GET / HTTP/1.1\r\nX: 1\r\nY: 2\r\n\r\n"
    monkeypatch.setattr(proxy_server_module, "_MAX_REQUEST_HEAD_BYTES", len(exact))

    assert _read_test_head(exact)[1] == {"X": "1", "Y": "2"}
    with pytest.raises(_ProxyProtocolError) as oversized:
        _read_test_head(exact[:-4] + b"A\r\n\r\n")
    assert oversized.value.error_code == "request_head_too_large"


def test_request_header_count_accepts_exact_boundary_and_rejects_next_header(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(proxy_server_module, "_MAX_HEADER_COUNT", 2)

    assert _read_test_head(b"GET / HTTP/1.1\r\nX: 1\r\nY: 2\r\n\r\n")[1] == {
        "X": "1",
        "Y": "2",
    }
    with pytest.raises(_ProxyProtocolError) as oversized:
        _read_test_head(b"GET / HTTP/1.1\r\nX: 1\r\nY: 2\r\nZ: 3\r\n\r\n")
    assert oversized.value.error_code == "too_many_headers"


def test_request_head_uses_absolute_deadline_against_slowloris() -> None:
    reader, writer = socket.socketpair()

    def trickle() -> None:
        with writer:
            for byte in b"GET / HTTP/1.1\r\n\r\n":
                with contextlib.suppress(OSError):
                    writer.send(bytes((byte,)))
                time.sleep(0.02)

    sender = threading.Thread(target=trickle)
    sender.start()
    started = time.monotonic()
    try:
        with pytest.raises(_ProxyProtocolError) as caught:
            _read_head(reader, timeout_s=0.05)
        assert caught.value.error_code == "request_head_timeout"
        assert time.monotonic() - started < 0.25
    finally:
        reader.close()
        sender.join(timeout=1.0)


def test_live_proxy_slowloris_deadline_releases_worker_capacity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_read_head = proxy_server_module._read_head

    def bounded_read_head(
        sock: socket.socket,
        *,
        stop: threading.Event | None = None,
        timeout_s: float = 0.05,
    ) -> tuple[bytes, dict[str, str]]:
        del timeout_s
        return original_read_head(sock, stop=stop, timeout_s=0.05)

    monkeypatch.setattr(proxy_server_module, "_read_head", bounded_read_head)

    async def run() -> tuple[bytes, bytes]:
        broker, _registry = _broker(_CapturingUpstream())
        server = TransparentEgressProxyServer(
            broker,
            loop=asyncio.get_running_loop(),
            max_workers=1,
        )
        port = await server.start()

        def slow_request() -> bytes:
            with socket.create_connection(("127.0.0.1", port), timeout=2.0) as conn:
                conn.settimeout(1.0)
                for byte in b"GET / HTTP/1.1\r\n\r\n":
                    try:
                        conn.send(bytes((byte,)))
                    except OSError:
                        break
                    time.sleep(0.02)
                with contextlib.suppress(OSError):
                    conn.shutdown(socket.SHUT_WR)
                try:
                    return conn.recv(1024)
                except OSError:
                    return b""

        def ordinary_request() -> bytes:
            with socket.create_connection(("127.0.0.1", port), timeout=2.0) as conn:
                conn.sendall(b"GET / HTTP/1.1\r\n\r\n")
                return conn.recv(1024)

        try:
            timed_out = await asyncio.to_thread(slow_request)
            for _ in range(100):
                with server._connections_lock:
                    if not server._connections:
                        break
                await asyncio.sleep(0.01)
            else:
                raise AssertionError("Slow request retained the only proxy worker.")
            replacement = await asyncio.to_thread(ordinary_request)
            return timed_out, replacement
        finally:
            await server.close()

    timed_out, replacement = asyncio.run(run())

    assert timed_out.startswith(b"HTTP/1.1 408 Request Timeout")
    assert b"X-Cayu-Egress-Error: request_head_timeout" in timed_out
    assert replacement.startswith(b"HTTP/1.1 403 Forbidden")


def test_response_head_serialization_does_not_copy_body() -> None:
    body = b"bounded-body-canary"
    response = CapturedResponse(status_code=200, body=body)

    head = _serialize_response_head(response)

    assert body not in head
    assert b"Content-Length: 19\r\n" in head


class _CapturingUpstream:
    def __init__(self) -> None:
        self.sent: CapturedRequest | None = None

    def prepare(
        self,
        request: CapturedRequest,
        *,
        limits: EgressUpstreamLimits,
    ) -> EgressUpstreamOperation:
        async def send() -> CapturedResponse:
            self.sent = request
            return CapturedResponse(
                status_code=200,
                headers={"Request-Id": "req_123"},
                body=b'{"id":"cus_live","object":"customer"}',
            )

        return EgressUpstreamOperation(send)


def _broker(
    upstream: Any,
    *,
    max_active_upstream_operations: int = 16,
) -> tuple[TransparentEgressBroker, VirtualCredentialRegistry]:
    registry = VirtualCredentialRegistry()
    broker = TransparentEgressBroker(
        registry=registry,
        resolver=StaticVault({"stripe_test_key": REAL_SECRET}),
        policies={
            "stripe-example": HttpEgressPolicy(
                name="stripe-example",
                allowed_hosts=["api.stripe.com"],
                allowed_endpoints=[("POST", "/v1/customers")],
            )
        },
        upstream=upstream,
        max_active_upstream_operations=max_active_upstream_operations,
    )
    return broker, registry


def _mint(registry: VirtualCredentialRegistry) -> Any:
    from cayu.vaults import SecretRef

    return registry.mint(
        session_id="sess_1",
        env_name="STRIPE_SECRET_KEY",
        secret=SecretRef(name="stripe_test_key"),
        destination="api.stripe.com",
        credential_kind="stripe_bearer",
        policy_name="stripe-example",
    )


async def _run_through_proxy(
    path: str, form: dict[str, str]
) -> tuple[httpx.Response, _CapturingUpstream]:
    loop = asyncio.get_running_loop()
    upstream = _CapturingUpstream()
    broker, registry = _broker(upstream)
    grant = _mint(registry)
    server = TransparentEgressProxyServer(broker, loop=loop)
    port = await server.start()

    ca_dir = tempfile.mkdtemp(prefix="cayu-egress-catest-")
    ca_path = os.path.join(ca_dir, "ca.pem")
    with open(ca_path, "wb") as handle:
        handle.write(server.authority.ca_cert_pem())

    try:
        ssl_context = ssl.create_default_context(cafile=ca_path)
        async with httpx.AsyncClient(
            proxy=f"http://127.0.0.1:{port}",
            verify=ssl_context,
            timeout=15.0,
        ) as client:
            response = await client.post(
                f"https://api.stripe.com{path}",
                headers={"Authorization": f"Bearer {grant.presented_value}"},
                data=form,
            )
        return response, upstream
    finally:
        await server.close()
        import shutil

        shutil.rmtree(ca_dir, ignore_errors=True)


def test_tls_interception_swaps_credential_and_captures_traffic() -> None:
    response, upstream = asyncio.run(_run_through_proxy("/v1/customers", {"email": "a@b.co"}))

    # The HTTPS request completed against our minted leaf cert (MITM works) and
    # was captured by the broker rather than reaching real Stripe.
    assert response.status_code == 200
    assert upstream.sent is not None
    assert upstream.sent.path == "/v1/customers"
    # The real secret was injected only on the upstream leg.
    assert upstream.sent.headers["Authorization"] == f"Bearer {REAL_SECRET}"
    # The sandbox-facing response carries no real secret.
    assert REAL_SECRET not in response.text


def test_proxy_close_terminates_connection_established_before_authority_cutover() -> None:
    async def run() -> None:
        broker, _registry = _broker(_CapturingUpstream())
        _mint(_registry)
        server = TransparentEgressProxyServer(broker, loop=asyncio.get_running_loop())
        port = await server.start()
        stale = socket.create_connection(("127.0.0.1", port), timeout=5.0)
        try:
            for _ in range(100):
                with server._connections_lock:
                    if server._connections:
                        break
                await asyncio.sleep(0.01)
            else:
                raise AssertionError("Proxy did not accept the stale test connection.")
            await asyncio.wait_for(server.close(), timeout=2.0)
            stale.settimeout(1.0)
            try:
                stale_result = stale.recv(1)
            except OSError:
                stale_result = b""
            assert stale_result == b""
        finally:
            stale.close()

    asyncio.run(run())


def test_proxy_close_terminates_completed_tls_connection_without_blocking_loop() -> None:
    async def run() -> None:
        broker, _registry = _broker(_CapturingUpstream())
        _mint(_registry)
        server = TransparentEgressProxyServer(broker, loop=asyncio.get_running_loop())
        port = await server.start()
        tls_connected = threading.Event()

        def hold_tls_connection() -> bool:
            raw = socket.create_connection(("127.0.0.1", port), timeout=5.0)
            try:
                raw.sendall(
                    b"CONNECT api.stripe.com:443 HTTP/1.1\r\nHost: api.stripe.com:443\r\n\r\n"
                )
                assert raw.recv(1024).startswith(b"HTTP/1.1 200 Connection Established")
                context = ssl._create_unverified_context()
                with context.wrap_socket(raw, server_hostname="api.stripe.com") as tls:
                    tls_connected.set()
                    tls.settimeout(2.0)
                    try:
                        return tls.recv(1) == b""
                    except OSError:
                        return True
            finally:
                raw.close()

        client_task = asyncio.create_task(asyncio.to_thread(hold_tls_connection))
        try:
            assert await asyncio.to_thread(tls_connected.wait, 2.0)
            for _ in range(100):
                with server._connections_lock:
                    if any(isinstance(item, ssl.SSLSocket) for item in server._connections):
                        break
                await asyncio.sleep(0.01)
            else:
                raise AssertionError("Proxy did not track the completed TLS connection.")

            heartbeat = asyncio.create_task(asyncio.sleep(0.01))
            await asyncio.wait_for(server.close(), timeout=1.0)
            await asyncio.wait_for(heartbeat, timeout=0.1)
            assert await asyncio.wait_for(client_task, timeout=1.0)
        finally:
            if not client_task.done():
                client_task.cancel()
            if server._sockets:
                await server.close()

    asyncio.run(run())


def test_proxy_close_terminates_multiple_idle_tls_connections() -> None:
    async def run() -> None:
        broker, registry = _broker(_CapturingUpstream())
        _mint(registry)
        server = TransparentEgressProxyServer(broker, loop=asyncio.get_running_loop())
        port = await server.start()
        connected = [threading.Event() for _ in range(3)]

        def hold_tls_connection(ready: threading.Event) -> bool:
            raw = socket.create_connection(("127.0.0.1", port), timeout=5.0)
            try:
                raw.sendall(
                    b"CONNECT api.stripe.com:443 HTTP/1.1\r\nHost: api.stripe.com:443\r\n\r\n"
                )
                assert raw.recv(1024).startswith(b"HTTP/1.1 200 Connection Established")
                context = ssl._create_unverified_context()
                with context.wrap_socket(raw, server_hostname="api.stripe.com") as tls:
                    ready.set()
                    tls.settimeout(2.0)
                    try:
                        return tls.recv(1) == b""
                    except OSError:
                        return True
            finally:
                raw.close()

        clients = [
            asyncio.create_task(asyncio.to_thread(hold_tls_connection, ready))
            for ready in connected
        ]
        try:
            for ready in connected:
                assert await asyncio.to_thread(ready.wait, 2.0)
            await asyncio.wait_for(server.close(), timeout=1.0)
            assert all(await asyncio.wait_for(asyncio.gather(*clients), timeout=1.0))
        finally:
            for client in clients:
                if not client.done():
                    client.cancel()
            if server._sockets:
                await server.close()

    asyncio.run(run())


def test_proxy_shutdown_cancels_stalled_upstream_and_releases_grant_lease() -> None:
    async def run() -> tuple[bool, bool]:
        entered = asyncio.Event()
        settled = asyncio.Event()

        class _StallingUpstream:
            def prepare(
                self,
                request: CapturedRequest,
                *,
                limits: EgressUpstreamLimits,
            ) -> EgressUpstreamOperation:
                async def send() -> CapturedResponse:
                    entered.set()
                    try:
                        await asyncio.Event().wait()
                    finally:
                        settled.set()
                    raise AssertionError("Unreachable stalled upstream completed.")

                return _cooperative_upstream_operation(send)

        broker, registry = _broker(_StallingUpstream())
        grant = _mint(registry)
        server = TransparentEgressProxyServer(broker, loop=asyncio.get_running_loop())
        port = await server.start()

        def request() -> bool:
            raw = socket.create_connection(("127.0.0.1", port), timeout=5.0)
            try:
                raw.sendall(
                    b"CONNECT api.stripe.com:443 HTTP/1.1\r\nHost: api.stripe.com:443\r\n\r\n"
                )
                assert raw.recv(1024).startswith(b"HTTP/1.1 200 Connection Established")
                context = ssl._create_unverified_context()
                with context.wrap_socket(raw, server_hostname="api.stripe.com") as tls:
                    tls.sendall(
                        b"POST /v1/customers HTTP/1.1\r\n"
                        + f"Authorization: Bearer {grant.presented_value}\r\n".encode()
                        + b"Content-Length: 0\r\n\r\n"
                    )
                    tls.settimeout(2.0)
                    try:
                        return tls.recv(1) == b""
                    except OSError:
                        return True
            finally:
                raw.close()

        client = asyncio.create_task(asyncio.to_thread(request))
        try:
            await asyncio.wait_for(entered.wait(), timeout=2.0)
            await asyncio.wait_for(server.close(), timeout=1.0)
            await asyncio.wait_for(settled.wait(), timeout=1.0)
            return await asyncio.wait_for(client, timeout=1.0), not registry._active_counts
        finally:
            if not client.done():
                client.cancel()
            if server._sockets:
                await server.close()

    client_closed, leases_released = asyncio.run(run())

    assert client_closed is True
    assert leases_released is True


def test_client_disconnect_cancels_stalled_upstream_and_releases_worker() -> None:
    async def run() -> tuple[bool, bytes]:
        entered = asyncio.Event()
        settled = asyncio.Event()
        request_sent = threading.Event()
        disconnect_now = threading.Event()

        class _StallingUpstream:
            def prepare(
                self,
                request: CapturedRequest,
                *,
                limits: EgressUpstreamLimits,
            ) -> EgressUpstreamOperation:
                async def send() -> CapturedResponse:
                    entered.set()
                    try:
                        await asyncio.Event().wait()
                    finally:
                        settled.set()
                    raise AssertionError("Unreachable stalled upstream completed.")

                return _cooperative_upstream_operation(send)

        broker, registry = _broker(_StallingUpstream())
        grant = _mint(registry)
        server = TransparentEgressProxyServer(
            broker,
            loop=asyncio.get_running_loop(),
            max_workers=1,
        )
        port = await server.start()

        def disconnect() -> None:
            raw = socket.create_connection(("127.0.0.1", port), timeout=5.0)
            try:
                raw.sendall(
                    b"CONNECT api.stripe.com:443 HTTP/1.1\r\nHost: api.stripe.com:443\r\n\r\n"
                )
                assert raw.recv(1024).startswith(b"HTTP/1.1 200 Connection Established")
                context = ssl._create_unverified_context()
                tls = context.wrap_socket(raw, server_hostname="api.stripe.com")
                try:
                    tls.sendall(
                        b"POST /v1/customers HTTP/1.1\r\n"
                        + f"Authorization: Bearer {grant.presented_value}\r\n".encode()
                        + b"Content-Length: 0\r\n\r\n"
                    )
                    request_sent.set()
                    if not disconnect_now.wait(timeout=5.0):
                        raise TimeoutError("Client disconnect was not released.")
                finally:
                    tls.close()
            finally:
                raw.close()

        client_task = asyncio.create_task(asyncio.to_thread(disconnect))
        try:
            if not await asyncio.to_thread(request_sent.wait, 2.0):
                await client_task
                raise AssertionError("Client request did not reach the proxy.")
            await asyncio.wait_for(entered.wait(), timeout=2.0)
            disconnect_now.set()
            await client_task
            await asyncio.wait_for(settled.wait(), timeout=2.0)
            for _ in range(100):
                with server._connections_lock:
                    if not server._connections:
                        break
                await asyncio.sleep(0.01)
            else:
                raise AssertionError("Disconnected client retained the proxy worker.")
            for _ in range(100):
                if not registry._active_counts:
                    break
                await asyncio.sleep(0.01)
            else:
                raise AssertionError("Disconnected client retained its grant lease.")

            def replacement() -> bytes:
                with socket.create_connection(("127.0.0.1", port), timeout=2.0) as conn:
                    conn.sendall(b"GET / HTTP/1.1\r\n\r\n")
                    return conn.recv(1024)

            return not registry._active_counts, await asyncio.to_thread(replacement)
        finally:
            disconnect_now.set()
            if not client_task.done():
                with contextlib.suppress(BaseException):
                    await client_task
            await server.close()

    lease_released, replacement = asyncio.run(run())

    assert lease_released is True
    assert replacement.startswith(b"HTTP/1.1 403 Forbidden")


def test_response_overflow_releases_proxy_and_broker_capacity() -> None:
    async def run() -> tuple[httpx.Response, bytes, bool, bool, bool]:
        class _OversizedUpstream:
            def prepare(
                self,
                request: CapturedRequest,
                *,
                limits: EgressUpstreamLimits,
            ) -> EgressUpstreamOperation:
                assert request.host == "docs.example.com"
                assert limits.max_response_bytes == 1

                async def send() -> CapturedResponse:
                    return CapturedResponse(status_code=200, body=b"xx")

                return EgressUpstreamOperation(send)

        broker = TransparentEgressBroker(
            registry=VirtualCredentialRegistry(),
            resolver=None,
            policies={
                "browser": BrowserEgressPolicy(
                    name="browser",
                    allowed_hosts=["docs.example.com"],
                )
            },
            approved_destinations=[
                ApprovedEgressDestination(
                    destination="docs.example.com",
                    policy_name="browser",
                )
            ],
            upstream=_OversizedUpstream(),
            browser_max_response_bytes=1,
        )
        server = TransparentEgressProxyServer(
            broker,
            loop=asyncio.get_running_loop(),
            max_workers=1,
        )
        port = await server.start()
        ca_dir = tempfile.mkdtemp(prefix="cayu-egress-overflow-")
        ca_path = os.path.join(ca_dir, "ca.pem")
        with open(ca_path, "wb") as handle:
            handle.write(server.authority.ca_cert_pem())
        try:
            ssl_context = ssl.create_default_context(cafile=ca_path)
            async with httpx.AsyncClient(
                proxy=f"http://127.0.0.1:{port}",
                verify=ssl_context,
                timeout=5.0,
            ) as client:
                overflow = await client.get("https://docs.example.com/")
            for _ in range(100):
                with server._connections_lock:
                    if not server._connections:
                        break
                await asyncio.sleep(0.01)
            else:
                raise AssertionError("Overflow response retained the proxy worker.")

            def replacement() -> bytes:
                with socket.create_connection(("127.0.0.1", port), timeout=2.0) as conn:
                    conn.sendall(b"GET / HTTP/1.1\r\n\r\n")
                    return conn.recv(1024)

            return (
                overflow,
                await asyncio.to_thread(replacement),
                not server._connections,
                not broker._active_upstream_operations,
                broker._credentialless_active_requests == 0,
            )
        finally:
            await server.close()
            import shutil

            shutil.rmtree(ca_dir, ignore_errors=True)

    overflow, replacement, worker_released, upstream_released, authority_released = asyncio.run(
        run()
    )

    assert overflow.status_code == 502
    assert overflow.headers[CAYU_EGRESS_ERROR_HEADER] == "oversized_response"
    assert replacement.startswith(b"HTTP/1.1 403 Forbidden")
    assert worker_released is True
    assert upstream_released is True
    assert authority_released is True


def test_opaque_upstream_remains_capacity_counted_until_positive_settlement() -> None:
    async def run() -> tuple[bytes, bool, int, bool]:
        entered = threading.Event()
        release = threading.Event()

        class _OpaqueUpstream:
            def __init__(self) -> None:
                self.calls = 0

            def prepare(
                self,
                request: CapturedRequest,
                *,
                limits: EgressUpstreamLimits,
            ) -> EgressUpstreamOperation:
                assert isinstance(request, CapturedRequest)
                assert limits.max_response_bytes > 0

                def blocking_call() -> CapturedResponse:
                    self.calls += 1
                    entered.set()
                    if not release.wait(5.0):
                        raise TimeoutError("Opaque test operation was not released.")
                    return CapturedResponse(status_code=200, body=b"late")

                async def send() -> CapturedResponse:
                    return await asyncio.to_thread(blocking_call)

                return EgressUpstreamOperation(send)

        upstream = _OpaqueUpstream()
        broker, registry = _broker(upstream, max_active_upstream_operations=1)
        grant = _mint(registry)
        server = TransparentEgressProxyServer(
            broker,
            loop=asyncio.get_running_loop(),
            max_workers=1,
        )
        port = await server.start()

        def disconnect_after_dispatch() -> None:
            raw = socket.create_connection(("127.0.0.1", port), timeout=5.0)
            raw.sendall(b"CONNECT api.stripe.com:443 HTTP/1.1\r\nHost: api.stripe.com:443\r\n\r\n")
            assert raw.recv(1024).startswith(b"HTTP/1.1 200 Connection Established")
            context = ssl._create_unverified_context()
            tls = context.wrap_socket(raw, server_hostname="api.stripe.com")
            tls.sendall(
                b"POST /v1/customers HTTP/1.1\r\n"
                + f"Authorization: Bearer {grant.presented_value}\r\n".encode()
                + b"Content-Length: 0\r\n\r\n"
            )
            assert entered.wait(5.0), "Opaque upstream dispatch did not begin."
            tls.close()

        def request_while_capacity_is_held() -> bytes:
            raw = socket.create_connection(("127.0.0.1", port), timeout=5.0)
            try:
                raw.sendall(
                    b"CONNECT api.stripe.com:443 HTTP/1.1\r\nHost: api.stripe.com:443\r\n\r\n"
                )
                assert raw.recv(1024).startswith(b"HTTP/1.1 200 Connection Established")
                context = ssl._create_unverified_context()
                with context.wrap_socket(raw, server_hostname="api.stripe.com") as tls:
                    tls.sendall(
                        b"POST /v1/customers HTTP/1.1\r\n"
                        + f"Authorization: Bearer {grant.presented_value}\r\n".encode()
                        + b"Content-Length: 0\r\n\r\n"
                    )
                    return tls.recv(4096)
            finally:
                raw.close()

        close_task: asyncio.Task[None] | None = None
        try:
            await asyncio.to_thread(disconnect_after_dispatch)
            for _ in range(100):
                with server._connections_lock:
                    if not server._connections:
                        break
                await asyncio.sleep(0.01)
            else:
                raise AssertionError("Disconnected client retained the proxy worker.")

            exhausted = await asyncio.to_thread(request_while_capacity_is_held)
            close_task = asyncio.create_task(server.close())
            heartbeat = asyncio.create_task(asyncio.sleep(0.01))
            await heartbeat
            await asyncio.sleep(0.05)
            close_waited_for_settlement = not close_task.done()
            release.set()
            await asyncio.wait_for(close_task, timeout=2.0)
            return (
                exhausted,
                close_waited_for_settlement,
                upstream.calls,
                not registry._active_counts,
            )
        finally:
            release.set()
            if close_task is not None and not close_task.done():
                await close_task
            elif server._sockets:
                await server.close()

    exhausted, close_waited_for_settlement, calls, leases_released = asyncio.run(run())

    assert b" 503 " in exhausted
    assert b"X-Cayu-Egress-Error: upstream_capacity_exhausted" in exhausted
    assert close_waited_for_settlement is True
    assert calls == 1
    assert leases_released is True


def test_disconnected_client_retains_opaque_credential_resolution_until_settlement(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def run() -> tuple[bytes, bool, int, bool]:
        entered = threading.Event()
        release = threading.Event()

        class _OpaqueResolver:
            def __init__(self) -> None:
                self.calls = 0

            async def resolve(
                self,
                ref: SecretRef,
                *,
                scope: dict[str, Any] | None = None,
            ) -> ResolvedSecret:
                del scope

                def blocking_read() -> None:
                    self.calls += 1
                    entered.set()
                    if not release.wait(5.0):
                        raise TimeoutError("Opaque credential resolution was not released.")

                await asyncio.to_thread(blocking_read)
                return ResolvedSecret(name=ref.name, value=SecretStr(REAL_SECRET))

        resolver = _OpaqueResolver()
        registry = VirtualCredentialRegistry()
        broker = TransparentEgressBroker(
            registry=registry,
            resolver=resolver,
            policies={
                "stripe-example": HttpEgressPolicy(
                    name="stripe-example",
                    allowed_hosts=["api.stripe.com"],
                    allowed_endpoints=[("POST", "/v1/customers")],
                )
            },
            upstream=_CapturingUpstream(),
        )
        grant = _mint(registry)
        server = TransparentEgressProxyServer(
            broker,
            loop=asyncio.get_running_loop(),
            max_workers=1,
        )
        port = await server.start()

        def send_request(*, disconnect: bool) -> bytes:
            raw = socket.create_connection(("127.0.0.1", port), timeout=5.0)
            try:
                raw.sendall(
                    b"CONNECT api.stripe.com:443 HTTP/1.1\r\nHost: api.stripe.com:443\r\n\r\n"
                )
                assert raw.recv(1024).startswith(b"HTTP/1.1 200 Connection Established")
                context = ssl._create_unverified_context()
                with context.wrap_socket(raw, server_hostname="api.stripe.com") as tls:
                    tls.sendall(
                        b"POST /v1/customers HTTP/1.1\r\n"
                        + f"Authorization: Bearer {grant.presented_value}\r\n".encode()
                        + b"Content-Length: 0\r\n\r\n"
                    )
                    if disconnect:
                        assert entered.wait(5.0), "Credential resolution did not dispatch."
                        return b""
                    return tls.recv(4096)
            finally:
                raw.close()

        close_task: asyncio.Task[None] | None = None
        try:
            await asyncio.to_thread(send_request, disconnect=True)
            for _ in range(100):
                with server._connections_lock:
                    if not server._connections:
                        break
                await asyncio.sleep(0.01)
            else:
                raise AssertionError("Disconnected client retained the proxy worker.")

            exhausted = await asyncio.to_thread(send_request, disconnect=False)
            close_task = asyncio.create_task(server.close())
            await asyncio.sleep(0.05)
            close_waited_for_settlement = not close_task.done()
            release.set()
            await asyncio.wait_for(close_task, timeout=2.0)
            return (
                exhausted,
                close_waited_for_settlement,
                resolver.calls,
                not registry._active_counts,
            )
        finally:
            release.set()
            if close_task is not None and not close_task.done():
                await close_task
            elif server._sockets:
                await server.close()

    monkeypatch.setattr(
        broker_module,
        "_MAX_ACTIVE_CREDENTIAL_RESOLUTIONS",
        1,
    )
    exhausted, close_waited_for_settlement, calls, leases_released = asyncio.run(run())

    assert b" 503 " in exhausted
    assert b"X-Cayu-Egress-Error: credential_resolution_capacity_exhausted" in exhausted
    assert close_waited_for_settlement is True
    assert calls == 1
    assert leases_released is True


def test_proxy_leaf_certificate_uses_standards_compliant_validity_window() -> None:
    async def run() -> x509.Certificate:
        loop = asyncio.get_running_loop()
        broker, registry = _broker(_CapturingUpstream())
        _mint(registry)
        server = TransparentEgressProxyServer(broker, loop=loop)
        port = await server.start()

        def fetch_certificate() -> x509.Certificate:
            with socket.create_connection(("127.0.0.1", port), timeout=5.0) as conn:
                conn.sendall(
                    b"CONNECT api.stripe.com:443 HTTP/1.1\r\nHost: api.stripe.com:443\r\n\r\n"
                )
                assert conn.recv(1024).startswith(b"HTTP/1.1 200 Connection Established")
                context = ssl._create_unverified_context()
                with context.wrap_socket(conn, server_hostname="api.stripe.com") as tls:
                    cert_der = tls.getpeercert(binary_form=True)
            return x509.load_der_x509_certificate(cert_der)

        try:
            return await asyncio.to_thread(fetch_certificate)
        finally:
            await server.close()

    certificate = asyncio.run(run())

    assert certificate.not_valid_after_utc - certificate.not_valid_before_utc <= timedelta(days=398)


def test_denied_endpoint_blocked_through_proxy() -> None:
    response, upstream = asyncio.run(
        _run_through_proxy("/v1/payouts", {"amount": "100", "currency": "usd"})
    )

    assert response.status_code == 403
    assert upstream.sent is None  # never forwarded upstream
    assert REAL_SECRET not in response.text


def test_plain_http_requests_are_rejected_without_broker_call() -> None:
    async def run() -> tuple[bytes, _CapturingUpstream]:
        loop = asyncio.get_running_loop()
        upstream = _CapturingUpstream()
        broker, _registry = _broker(upstream)
        server = TransparentEgressProxyServer(broker, loop=loop)
        port = await server.start()

        def request_plain_http() -> bytes:
            with socket.create_connection(("127.0.0.1", port), timeout=5.0) as conn:
                conn.sendall(
                    b"GET http://api.stripe.com/v1/customers HTTP/1.1\r\n"
                    b"Host: api.stripe.com\r\n\r\n"
                )
                return conn.recv(1024)

        try:
            return await asyncio.to_thread(request_plain_http), upstream
        finally:
            await server.close()

    response, upstream = asyncio.run(run())

    assert response.startswith(b"HTTP/1.1 403 Forbidden")
    assert f"{CAYU_EGRESS_ERROR_HEADER}: destination_denied".encode() in response
    assert upstream.sent is None


def test_transport_auth_rejects_clients_without_the_sidecar_identity() -> None:
    async def run() -> tuple[bytes, bytes]:
        loop = asyncio.get_running_loop()
        broker, registry = _broker(_CapturingUpstream())
        _mint(registry)
        transport_auth_token = b"test-transport-secret"
        server = TransparentEgressProxyServer(
            broker,
            loop=loop,
            transport_auth_token=transport_auth_token,
        )
        port = await server.start()

        request = b"CONNECT api.stripe.com:443 HTTP/1.1\r\nHost: api.stripe.com:443\r\n\r\n"

        def unauthenticated() -> bytes:
            try:
                with socket.create_connection(("127.0.0.1", port), timeout=5.0) as conn:
                    conn.sendall(request)
                    return conn.recv(1024)
            except OSError:
                return b""

        def authenticated() -> bytes:
            with socket.create_connection(("127.0.0.1", port), timeout=5.0) as conn:
                encoded = base64.b64encode(b"cayu:" + transport_auth_token)
                conn.sendall(
                    b"CONNECT cayu-transport.invalid:443 HTTP/1.0\r\n"
                    b"Proxy-authorization: Basic " + encoded + b"\r\n\r\n"
                )
                outer_response = conn.recv(1024)
                if not outer_response.startswith(b"HTTP/1.1 200 Connection Established"):
                    return outer_response
                conn.sendall(request)
                return conn.recv(1024)

        try:
            return await asyncio.to_thread(unauthenticated), await asyncio.to_thread(authenticated)
        finally:
            await server.close()

    rejected, accepted = asyncio.run(run())

    assert rejected == b""
    assert accepted.startswith(b"HTTP/1.1 200 Connection Established")


def test_transport_auth_token_requires_nonempty_bytes() -> None:
    loop = asyncio.new_event_loop()
    try:
        broker, _registry = _broker(_CapturingUpstream())
        with pytest.raises(TypeError, match="must be bytes"):
            TransparentEgressProxyServer(
                broker,
                loop=loop,
                transport_auth_token="secret",  # type: ignore[arg-type]
            )
        with pytest.raises(ValueError, match="must be nonempty"):
            TransparentEgressProxyServer(
                broker,
                loop=loop,
                transport_auth_token=b"",
            )
    finally:
        loop.close()


@pytest.mark.parametrize(
    ("max_workers", "error_type"),
    [(True, TypeError), (0, ValueError), (65, ValueError)],
)
def test_proxy_worker_capacity_is_strictly_bounded(
    max_workers: Any,
    error_type: type[Exception],
) -> None:
    loop = asyncio.new_event_loop()
    try:
        broker, _registry = _broker(_CapturingUpstream())
        with pytest.raises(error_type):
            TransparentEgressProxyServer(
                broker,
                loop=loop,
                max_workers=max_workers,
            )
    finally:
        loop.close()


def test_connect_to_non_https_port_is_rejected_without_broker_call() -> None:
    async def run() -> tuple[bytes, _CapturingUpstream]:
        loop = asyncio.get_running_loop()
        upstream = _CapturingUpstream()
        broker, _registry = _broker(upstream)
        server = TransparentEgressProxyServer(broker, loop=loop)
        port = await server.start()

        def request_non_https_port() -> bytes:
            with socket.create_connection(("127.0.0.1", port), timeout=5.0) as conn:
                conn.sendall(
                    b"CONNECT api.stripe.com:8443 HTTP/1.1\r\nHost: api.stripe.com:8443\r\n\r\n"
                )
                return conn.recv(1024)

        try:
            return await asyncio.to_thread(request_non_https_port), upstream
        finally:
            await server.close()

    response, upstream = asyncio.run(run())

    assert response.startswith(b"HTTP/1.1 403 Forbidden")
    assert upstream.sent is None


@pytest.mark.parametrize(
    "target",
    ["secret-canary.example.com:443", "secret-canary.example.com:8443"],
)
def test_unauthorized_or_malformed_connect_is_rejected_before_certificate_generation(
    monkeypatch: pytest.MonkeyPatch,
    target: str,
    caplog: pytest.LogCaptureFixture,
    capsys: pytest.CaptureFixture[str],
) -> None:
    async def run() -> tuple[bytes, int, int]:
        broker, _registry = _broker(_CapturingUpstream())
        authority = SessionCertificateAuthority()
        generated = 0
        original = authority.server_ssl_context

        def counted(host: str, *, stop: threading.Event | None = None) -> ssl.SSLContext:
            nonlocal generated
            generated += 1
            return original(host, stop=stop)

        monkeypatch.setattr(authority, "server_ssl_context", counted)
        server = TransparentEgressProxyServer(
            broker,
            loop=asyncio.get_running_loop(),
            authority=authority,
        )
        port = await server.start()

        def connect() -> bytes:
            with socket.create_connection(("127.0.0.1", port), timeout=2.0) as conn:
                conn.sendall(
                    b"CONNECT "
                    + target.encode("ascii")
                    + b" HTTP/1.1\r\nHost: ignored.invalid\r\n\r\n"
                )
                return conn.recv(1024)

        try:
            response = await asyncio.to_thread(connect)
            return response, generated, len(authority._contexts)
        finally:
            await server.close()

    with warnings.catch_warnings(record=True) as caught_warnings:
        response, generated, cached = asyncio.run(run())
    captured = capsys.readouterr()

    assert response.startswith(b"HTTP/1.1 403 Forbidden")
    assert generated == 0
    assert cached == 0
    diagnostic_surfaces = "\n".join(
        (
            response.decode("latin-1"),
            caplog.text,
            captured.out,
            captured.err,
            *(str(item.message) for item in caught_warnings),
        )
    )
    assert "secret-canary" not in diagnostic_surfaces


def test_connect_certificate_admission_is_revoked_and_drained_before_success(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def run() -> tuple[bytes, int, int]:
        broker, registry = _broker(_CapturingUpstream())
        grant = _mint(registry)
        authority = SessionCertificateAuthority()
        generation_entered = threading.Event()
        release_generation = threading.Event()
        original = authority.server_ssl_context

        def blocked_generation(
            host: str,
            *,
            stop: threading.Event | None = None,
        ) -> ssl.SSLContext:
            generation_entered.set()
            if not release_generation.wait(2.0):
                raise TimeoutError("Certificate generation test gate was not released.")
            return original(host, stop=stop)

        monkeypatch.setattr(authority, "server_ssl_context", blocked_generation)
        server = TransparentEgressProxyServer(
            broker,
            loop=asyncio.get_running_loop(),
            authority=authority,
        )
        port = await server.start()

        def connect() -> bytes:
            with socket.create_connection(("127.0.0.1", port), timeout=2.0) as conn:
                conn.sendall(
                    b"CONNECT api.stripe.com:443 HTTP/1.1\r\nHost: api.stripe.com:443\r\n\r\n"
                )
                return conn.recv(1024)

        client = asyncio.create_task(asyncio.to_thread(connect))
        revocation: asyncio.Task[int] | None = None
        try:
            assert await asyncio.to_thread(generation_entered.wait, 1.0)
            revocation = asyncio.create_task(
                broker.revoke_authority_and_wait((grant.presented_value,))
            )
            for _ in range(100):
                if not await broker.authorize_connect_destination(
                    host="api.stripe.com",
                    port=443,
                ):
                    break
                await asyncio.sleep(0.01)
            else:
                raise AssertionError("Revocation did not remove CONNECT authority.")
            assert revocation.done() is False

            release_generation.set()
            response = await asyncio.wait_for(client, timeout=1.0)
            revoked = await asyncio.wait_for(revocation, timeout=1.0)
            return response, revoked, len(authority._contexts)
        finally:
            release_generation.set()
            if not client.done():
                client.cancel()
            await asyncio.gather(client, return_exceptions=True)
            if revocation is not None and not revocation.done():
                await revocation
            await server.close()

    response, revoked, cached = asyncio.run(run())

    assert response.startswith(b"HTTP/1.1 403 Forbidden")
    assert revoked == 1
    assert cached == 0


def test_connect_admission_completed_during_shutdown_is_released(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def run() -> tuple[int, int]:
        broker, registry = _broker(_CapturingUpstream())
        grant = _mint(registry)
        server = TransparentEgressProxyServer(broker, loop=asyncio.get_running_loop())
        original = broker.begin_connect_destination_admission
        acquired = asyncio.Event()
        allow_result_delivery = asyncio.Event()

        async def acquire_before_result_delivery(
            *,
            host: str,
            port: int,
        ) -> broker_module._ConnectDestinationAdmission | None:
            admission = await original(host=host, port=port)
            acquired.set()
            await allow_result_delivery.wait()
            return admission

        monkeypatch.setattr(
            broker,
            "begin_connect_destination_admission",
            acquire_before_result_delivery,
        )
        admission_call = asyncio.create_task(
            asyncio.to_thread(
                server._begin_connect_destination_admission,
                "api.stripe.com",
                443,
            )
        )
        try:
            await asyncio.wait_for(acquired.wait(), timeout=1.0)
            server._stop.set()
            await asyncio.sleep(0.3)
            assert admission_call.done() is False
            allow_result_delivery.set()
            with pytest.raises(RuntimeError, match="shutting down"):
                await asyncio.wait_for(admission_call, timeout=1.0)
            active_admissions = len(broker._active_connect_admissions)
            revoked = await asyncio.wait_for(
                broker.revoke_authority_and_wait((grant.presented_value,)),
                timeout=1.0,
            )
            return active_admissions, revoked
        finally:
            allow_result_delivery.set()
            await asyncio.gather(admission_call, return_exceptions=True)
            await server.close()

    active_admissions, revoked = asyncio.run(run())

    assert active_admissions == 0
    assert revoked == 1


def test_proxy_connection_capacity_rejects_without_executor_queue_and_recovers() -> None:
    async def run() -> tuple[bytes, bytes]:
        broker, _registry = _broker(_CapturingUpstream())
        server = TransparentEgressProxyServer(
            broker,
            loop=asyncio.get_running_loop(),
            max_workers=1,
        )
        port = await server.start()
        held = socket.create_connection(("127.0.0.1", port), timeout=2.0)
        held.sendall(b"G")
        try:
            for _ in range(100):
                with server._connections_lock:
                    if len(server._connections) == 1:
                        break
                await asyncio.sleep(0.01)
            else:
                raise AssertionError("Proxy did not occupy its only connection slot.")

            def rejected() -> bytes:
                with socket.create_connection(("127.0.0.1", port), timeout=2.0) as conn:
                    conn.sendall(b"GET / HTTP/1.1\r\n\r\n")
                    return conn.recv(1024)

            exhausted = await asyncio.to_thread(rejected)
            held.close()
            for _ in range(100):
                with server._connections_lock:
                    if not server._connections:
                        break
                await asyncio.sleep(0.01)
            else:
                raise AssertionError("Proxy did not release its connection slot.")
            recovered = await asyncio.to_thread(rejected)
            return exhausted, recovered
        finally:
            held.close()
            await server.close()

    exhausted, recovered = asyncio.run(run())

    assert exhausted.startswith(b"HTTP/1.1 503 Service Unavailable")
    assert b"proxy_capacity_exhausted" in exhausted
    assert recovered.startswith(b"HTTP/1.1 403 Forbidden")


def test_dual_stack_loopback_proxy_serves_and_closes_both_address_families() -> None:
    async def run() -> tuple[bytes, bytes, bool, bool]:
        loop = asyncio.get_running_loop()
        broker, _registry = _broker(_CapturingUpstream())
        server = DualStackLoopbackEgressProxyServer(broker, loop=loop)
        port = await server.start()

        def request(host: str) -> bytes:
            with socket.create_connection((host, port), timeout=5.0) as conn:
                conn.sendall(
                    b"GET http://api.stripe.com/v1/customers HTTP/1.1\r\n"
                    b"Host: api.stripe.com\r\n\r\n"
                )
                return conn.recv(1024)

        ipv4, ipv6 = await asyncio.gather(
            asyncio.to_thread(request, "127.0.0.1"),
            asyncio.to_thread(request, "::1"),
        )
        await server.close()

        def is_closed(host: str) -> bool:
            try:
                with socket.create_connection((host, port), timeout=0.25):
                    return False
            except OSError:
                return True

        return ipv4, ipv6, is_closed("127.0.0.1"), is_closed("::1")

    ipv4, ipv6, ipv4_closed, ipv6_closed = asyncio.run(run())

    assert ipv4.startswith(b"HTTP/1.1 403 Forbidden")
    assert ipv6.startswith(b"HTTP/1.1 403 Forbidden")
    diagnostic = f"{CAYU_EGRESS_ERROR_HEADER}: destination_denied".encode()
    assert diagnostic in ipv4
    assert diagnostic in ipv6
    assert ipv4_closed is True
    assert ipv6_closed is True


def test_start_closes_listener_when_bind_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    class _FakeSock:
        def __init__(self, *args: object, **kwargs: object) -> None:
            self.closed = False

        def setsockopt(self, *args: object) -> None:
            pass

        def bind(self, *args: object) -> None:
            raise OSError("address already in use")

        def listen(self, *args: object) -> None:  # pragma: no cover - not reached
            pass

        def settimeout(self, *args: object) -> None:  # pragma: no cover - not reached
            pass

        def close(self) -> None:
            self.closed = True

    created: list[_FakeSock] = []

    def _factory(*args: object, **kwargs: object) -> _FakeSock:
        sock = _FakeSock()
        created.append(sock)
        return sock

    # Create the loop BEFORE patching so asyncio's own sockets stay real; only
    # the proxy's start() socket is faked.
    loop = asyncio.new_event_loop()
    try:
        broker, _registry = _broker(_CapturingUpstream())
        server = TransparentEgressProxyServer(broker, loop=loop)
        monkeypatch.setattr("cayu.egress.proxy_server.socket.socket", _factory)
        with pytest.raises(OSError):
            loop.run_until_complete(server.start())
        loop.run_until_complete(server.close())
    finally:
        loop.close()

    # The listener socket was created but bind failed — it must be closed, not leaked.
    assert created and created[0].closed is True


def test_dual_stack_start_closes_ipv4_listener_when_ipv6_bind_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _FakeSock:
        def __init__(self, family: int) -> None:
            self.family = family
            self.closed = False

        def setsockopt(self, *args: object) -> None:
            pass

        def bind(self, *args: object) -> None:
            if self.family == socket.AF_INET6:
                raise OSError("IPv6 loopback unavailable")

        def listen(self, *args: object) -> None:
            pass

        def settimeout(self, *args: object) -> None:
            pass

        def getsockname(self) -> tuple[str, int]:
            return ("127.0.0.1", 8123)

        def close(self) -> None:
            self.closed = True

    created: list[_FakeSock] = []

    def resolve(host: str, port: int, **kwargs: object) -> list[tuple[Any, ...]]:
        family = socket.AF_INET6 if host == "::1" else socket.AF_INET
        return [(family, socket.SOCK_STREAM, 0, "", (host, port))]

    def create_socket(family: int, *args: object) -> _FakeSock:
        sock = _FakeSock(family)
        created.append(sock)
        return sock

    loop = asyncio.new_event_loop()
    try:
        broker, _registry = _broker(_CapturingUpstream())
        server = DualStackLoopbackEgressProxyServer(broker, loop=loop)
        monkeypatch.setattr("cayu.egress.proxy_server.socket.getaddrinfo", resolve)
        monkeypatch.setattr("cayu.egress.proxy_server.socket.socket", create_socket)
        with pytest.raises(OSError, match="IPv6 loopback unavailable"):
            loop.run_until_complete(server.start())
        loop.run_until_complete(server.close())
    finally:
        loop.close()

    assert len(created) == 2
    assert all(sock.closed for sock in created)


def test_leaf_cert_files_do_not_use_untrusted_connect_host(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with tempfile.TemporaryDirectory() as root:
        ca_dir = os.path.join(root, "ca")
        os.mkdir(ca_dir)
        monkeypatch.setattr("cayu.egress.proxy_server.tempfile.mkdtemp", lambda **_: ca_dir)

        authority = SessionCertificateAuthority()
        try:
            authority.server_ssl_context("../outside")
        finally:
            authority.close()

        assert not os.path.exists(os.path.join(root, "outside.cert.pem"))
        assert not os.path.exists(os.path.join(root, "outside.key.pem"))


def test_certificate_cache_is_lru_bounded_and_leaf_files_are_ephemeral(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(proxy_server_module, "_MAX_CERTIFICATE_CACHE_ENTRIES", 2)
    authority = SessionCertificateAuthority()
    tempdir = authority._tempdir
    try:
        first = authority.server_ssl_context("one.example.com")
        second = authority.server_ssl_context("two.example.com")
        assert authority.server_ssl_context("one.example.com") is first
        authority.server_ssl_context("three.example.com")

        assert tuple(authority._contexts) == ("one.example.com", "three.example.com")
        assert "two.example.com" not in authority._contexts
        assert second is not first
        assert os.listdir(tempdir) == []
    finally:
        authority.close()


def test_certificate_cache_entries_expire(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(proxy_server_module, "_CERTIFICATE_CACHE_TTL_S", 0.0)
    authority = SessionCertificateAuthority()
    try:
        first = authority.server_ssl_context("ttl.example.com")
        second = authority.server_ssl_context("ttl.example.com")
    finally:
        authority.close()

    assert second is not first


def test_certificate_generation_is_serialized_under_high_cardinality(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    authority = SessionCertificateAuthority()
    active = 0
    maximum = 0
    counter_lock = threading.Lock()

    def build(_host: str) -> ssl.SSLContext:
        nonlocal active, maximum
        with counter_lock:
            active += 1
            maximum = max(maximum, active)
        time.sleep(0.03)
        with counter_lock:
            active -= 1
        return ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)

    monkeypatch.setattr(authority, "_build_leaf_context", build)
    threads = [
        threading.Thread(target=authority.server_ssl_context, args=(f"{index}.example.com",))
        for index in range(4)
    ]
    try:
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=2.0)
            assert not thread.is_alive()
    finally:
        authority.close()

    assert maximum == 1


def test_certificate_revocation_during_generation_does_not_populate_cache(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    authority = SessionCertificateAuthority()
    generation_entered = threading.Event()
    release_generation = threading.Event()
    stop = threading.Event()
    failures: list[BaseException] = []

    def build(_host: str) -> ssl.SSLContext:
        generation_entered.set()
        if not release_generation.wait(2.0):
            raise TimeoutError("Certificate generation test gate was not released.")
        return ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)

    def generate() -> None:
        try:
            authority.server_ssl_context("revoked.example.com", stop=stop)
        except BaseException as exc:
            failures.append(exc)

    monkeypatch.setattr(authority, "_build_leaf_context", build)
    worker = threading.Thread(target=generate)
    cached_hosts: tuple[str, ...] = ()
    try:
        worker.start()
        assert generation_entered.wait(1.0)
        stop.set()
        release_generation.set()
        worker.join(timeout=1.0)
        assert not worker.is_alive()
        cached_hosts = tuple(authority._contexts)
    finally:
        release_generation.set()
        authority.close()

    assert len(failures) == 1
    assert isinstance(failures[0], _CertificateCapacityError)
    assert cached_hosts == ()


def test_cancelled_close_keeps_owned_cleanup_running() -> None:
    async def run() -> tuple[bool, bool]:
        broker, _registry = _broker(_CapturingUpstream())
        server = TransparentEgressProxyServer(broker, loop=asyncio.get_running_loop())
        original_executor = server._executor
        original_executor.shutdown(wait=True, cancel_futures=True)
        entered = threading.Event()
        release = threading.Event()

        class _BlockingExecutor:
            def shutdown(self, *, wait: bool, cancel_futures: bool) -> None:
                assert wait is True
                assert cancel_futures is False
                entered.set()
                release.wait(timeout=2.0)

        server._executor = _BlockingExecutor()  # type: ignore[assignment]
        close_call = asyncio.create_task(server.close())
        assert await asyncio.to_thread(entered.wait, 1.0)
        close_call.cancel()
        with pytest.raises(asyncio.CancelledError):
            await close_call
        still_settling = server._close_task is not None and not server._close_task.done()
        release.set()
        assert server._close_task is not None
        await asyncio.wait_for(server._close_task, timeout=1.0)
        await server.close()
        return still_settling, server.authority._closed

    still_settling, authority_closed = asyncio.run(run())

    assert still_settling is True
    assert authority_closed is True


def test_proxy_shutdown_settles_accepted_handler_queued_before_dispatch() -> None:
    async def run() -> tuple[bool, bool, bool]:
        broker, _registry = _broker(_CapturingUpstream())
        server = TransparentEgressProxyServer(
            broker,
            loop=asyncio.get_running_loop(),
            max_workers=1,
        )
        original_executor = server._executor
        original_executor.shutdown(wait=True, cancel_futures=True)
        accepted = threading.Event()
        pending: list[tuple[Any, tuple[Any, ...]]] = []

        class _DeferredExecutor:
            def submit(self, callback: Any, *args: Any) -> object:
                pending.append((callback, args))
                accepted.set()
                return object()

            def shutdown(self, *, wait: bool, cancel_futures: bool) -> None:
                assert wait is True
                assert cancel_futures is False
                while pending:
                    callback, args = pending.pop(0)
                    callback(*args)

        server._executor = _DeferredExecutor()  # type: ignore[assignment]
        port = await server.start()
        client = socket.create_connection(("127.0.0.1", port), timeout=2.0)
        try:
            assert await asyncio.to_thread(accepted.wait, 1.0)
            with server._connections_lock:
                tracked_before_close = bool(server._connections)

            await asyncio.wait_for(server.close(), timeout=1.0)

            with server._connections_lock:
                connections_released = not server._connections
            first_permit = server._connection_slots.acquire(blocking=False)
            second_permit = server._connection_slots.acquire(blocking=False)
            if first_permit:
                server._connection_slots.release()
            return tracked_before_close, connections_released, first_permit and not second_permit
        finally:
            client.close()
            if server._sockets:
                await server.close()

    tracked, connections_released, permit_released = asyncio.run(run())

    assert tracked is True
    assert connections_released is True
    assert permit_released is True


def test_ca_cleanup_failure_retains_ownership_and_can_be_retried(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    broker, _registry = _broker(_CapturingUpstream())
    loop = asyncio.new_event_loop()
    server = TransparentEgressProxyServer(broker, loop=loop)
    ca_dir = server.authority._tempdir
    original_rmtree = proxy_server_module.shutil.rmtree
    failed_once = False

    def fail_ca_cleanup_once(path: str, *args: object, **kwargs: object) -> None:
        nonlocal failed_once
        if path == ca_dir and not failed_once:
            failed_once = True
            raise OSError("certificate directory cleanup failed")
        original_rmtree(path, *args, **kwargs)

    monkeypatch.setattr(proxy_server_module.shutil, "rmtree", fail_ca_cleanup_once)
    try:
        with pytest.raises(BaseExceptionGroup, match="Egress proxy shutdown was incomplete"):
            loop.run_until_complete(server.close())

        assert server.authority._closed is True
        assert server.authority._cleanup_complete is False
        assert server._owns_authority is True
        assert os.path.isdir(ca_dir)
        with pytest.raises(_CertificateCapacityError, match="authority is closed"):
            server.authority.server_ssl_context("closed.example.com")

        loop.run_until_complete(server.close())
    finally:
        loop.close()

    assert server.authority._cleanup_complete is True
    assert server._owns_authority is False
    assert not os.path.exists(ca_dir)


def test_chunked_request_body_is_forwarded_intact() -> None:
    async def run() -> tuple[httpx.Response, _CapturingUpstream]:
        loop = asyncio.get_running_loop()
        upstream = _CapturingUpstream()
        broker, registry = _broker(upstream)
        grant = _mint(registry)
        server = TransparentEgressProxyServer(broker, loop=loop)
        port = await server.start()

        ca_dir = tempfile.mkdtemp(prefix="cayu-egress-catest-")
        ca_path = os.path.join(ca_dir, "ca.pem")
        with open(ca_path, "wb") as handle:
            handle.write(server.authority.ca_cert_pem())

        async def body() -> Any:
            # No Content-Length -> httpx sends Transfer-Encoding: chunked.
            for part in (b"email=", b"chunked", b"%40ex.co"):
                yield part

        try:
            ssl_context = ssl.create_default_context(cafile=ca_path)
            async with httpx.AsyncClient(
                proxy=f"http://127.0.0.1:{port}", verify=ssl_context, timeout=15.0
            ) as client:
                response = await client.post(
                    "https://api.stripe.com/v1/customers",
                    headers={
                        "Authorization": f"Bearer {grant.presented_value}",
                        "Content-Type": "application/x-www-form-urlencoded",
                    },
                    content=body(),
                )
            return response, upstream
        finally:
            await server.close()
            import shutil

            shutil.rmtree(ca_dir, ignore_errors=True)

    response, upstream = asyncio.run(run())

    assert response.status_code == 200
    assert upstream.sent is not None
    # The full chunked body was reassembled, not truncated.
    assert upstream.sent.body == b"email=chunked%40ex.co"
