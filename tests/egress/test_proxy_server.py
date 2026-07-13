from __future__ import annotations

import asyncio
import os
import socket
import ssl
import tempfile
from typing import Any

import httpx
import pytest

from cayu.egress import (
    CapturedRequest,
    CapturedResponse,
    HttpEgressPolicy,
    TransparentEgressBroker,
    VirtualCredentialRegistry,
)
from cayu.vaults import StaticVault

pytest.importorskip("cryptography")

from cayu.egress.proxy_server import (
    DualStackLoopbackEgressProxyServer,
    SessionCertificateAuthority,
    TransparentEgressProxyServer,
)

REAL_SECRET = "sk_test_51RealProxySwapSecret"


class _CapturingUpstream:
    def __init__(self) -> None:
        self.sent: CapturedRequest | None = None

    async def send(self, request: CapturedRequest) -> CapturedResponse:
        self.sent = request
        return CapturedResponse(
            status_code=200,
            headers={"Request-Id": "req_123"},
            body=b'{"id":"cus_live","object":"customer"}',
        )


def _broker(upstream: Any) -> tuple[TransparentEgressBroker, VirtualCredentialRegistry]:
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

    assert response.startswith(b"HTTP/1.1 405 Method Not Allowed")
    assert upstream.sent is None


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

    assert ipv4.startswith(b"HTTP/1.1 405 Method Not Allowed")
    assert ipv6.startswith(b"HTTP/1.1 405 Method Not Allowed")
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
