"""Protected channel establishment over real local TLS, without browser authority."""

from __future__ import annotations

import asyncio
import ipaddress
import ssl
from datetime import UTC, datetime, timedelta
from http import HTTPStatus

import pytest
from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.x509.oid import NameOID
from websockets.asyncio.server import serve
from websockets.exceptions import ConnectionClosedError, InvalidStatus

from cayu.tools._browser_control_transport import (
    CONTROL_MESSAGE_BYTES,
    CONTROL_SUBPROTOCOL,
    BrowserControlTransportUnavailable,
    open_guest_control_channel,
    validate_control_endpoint,
)


@pytest.fixture
def control_tls(tmp_path):
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    subject = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "control.test")])
    now = datetime.now(UTC)
    certificate = (
        x509.CertificateBuilder()
        .subject_name(subject)
        .issuer_name(subject)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - timedelta(minutes=1))
        .not_valid_after(now + timedelta(hours=1))
        .add_extension(
            x509.SubjectAlternativeName([x509.IPAddress(ipaddress.ip_address("127.0.0.1"))]),
            critical=False,
        )
        .sign(key, hashes.SHA256())
    )
    cert_path = tmp_path / "certificate.pem"
    key_path = tmp_path / "key.pem"
    cert_path.write_bytes(certificate.public_bytes(serialization.Encoding.PEM))
    key_path.write_bytes(
        key.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.PKCS8,
            serialization.NoEncryption(),
        )
    )
    server = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    server.load_cert_chain(cert_path, key_path)
    client = ssl.create_default_context(cafile=str(cert_path))
    return server, client


@pytest.mark.parametrize(
    "endpoint",
    [
        "ws://example.test/control",
        "wss://user:secret@example.test/control",
        "wss://example.test/control?token=secret",
        "wss://example.test/control#secret",
        "wss://example.test:0/control",
        "wss://example.test:99999/control",
        "wss://example.test/\ncontrol",
        "wss://example.test/\\control",
        "wss://example.test/\ud800",
    ],
)
def test_control_endpoint_rejects_unsafe_forms(endpoint) -> None:
    with pytest.raises(BrowserControlTransportUnavailable):
        validate_control_endpoint(endpoint)


def test_control_channel_uses_tls_and_does_not_log_wire_values(control_tls, caplog) -> None:
    async def scenario():
        server_tls, client_tls = control_tls
        credential = "private-capability-canary-" + "a" * 32
        payload = "sensitive-input-canary"
        observed = []

        async def peer(connection):
            observed.append(connection.request.headers["Authorization"])
            await connection.send(payload)
            await connection.wait_closed()

        # Only enable client wire logging globally. The controlled test server
        # necessarily receives the capability and must not log it either.
        caplog.set_level("DEBUG", logger="websockets.client")
        async with serve(
            peer, "127.0.0.1", 0, ssl=server_tls, subprotocols=[CONTROL_SUBPROTOCOL]
        ) as server:
            port = server.sockets[0].getsockname()[1]
            connection = await open_guest_control_channel(
                endpoint=f"wss://127.0.0.1:{port}/control",
                credential=credential,
                tls=client_tls,
            )
            try:
                assert connection.subprotocol == CONTROL_SUBPROTOCOL
                assert await connection.recv() == payload
                assert observed == [f"Bearer {credential}"]
            finally:
                await connection.close()
        assert credential not in caplog.text
        assert payload not in caplog.text

    asyncio.run(scenario())


def test_control_channel_never_follows_same_origin_redirect(control_tls) -> None:
    async def scenario():
        server_tls, client_tls = control_tls
        requests = []

        async def peer(connection):
            pytest.fail("Redirect must not establish a control channel.")

        def redirect(connection, request):
            requests.append(request.path)
            response = connection.respond(HTTPStatus.TEMPORARY_REDIRECT, "redirect")
            response.headers["Location"] = "/other"
            return response

        async with serve(peer, "127.0.0.1", 0, ssl=server_tls, process_request=redirect) as server:
            port = server.sockets[0].getsockname()[1]
            with pytest.raises(InvalidStatus):
                await open_guest_control_channel(
                    endpoint=f"wss://127.0.0.1:{port}/control",
                    credential="a" * 64,
                    tls=client_tls,
                )
        assert requests == ["/control"]

    asyncio.run(scenario())


def test_control_channel_rejects_oversized_inbound_message(control_tls) -> None:
    async def scenario():
        server_tls, client_tls = control_tls

        async def peer(connection):
            await connection.send(b"x" * (CONTROL_MESSAGE_BYTES + 1))
            await connection.wait_closed()

        async with serve(
            peer, "127.0.0.1", 0, ssl=server_tls, subprotocols=[CONTROL_SUBPROTOCOL]
        ) as server:
            port = server.sockets[0].getsockname()[1]
            connection = await open_guest_control_channel(
                endpoint=f"wss://127.0.0.1:{port}/control",
                credential="a" * 64,
                tls=client_tls,
            )
            try:
                with pytest.raises(ConnectionClosedError) as failure:
                    await connection.recv()
                assert failure.value.sent.code == 1009
            finally:
                await connection.close()

    asyncio.run(scenario())


def test_control_channel_refuses_disabled_certificate_validation() -> None:
    async def scenario():
        context = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
        context.check_hostname = False
        context.verify_mode = ssl.CERT_NONE
        with pytest.raises(BrowserControlTransportUnavailable):
            await open_guest_control_channel(
                endpoint="wss://127.0.0.1:1/control", credential="a" * 64, tls=context
            )

    asyncio.run(scenario())
