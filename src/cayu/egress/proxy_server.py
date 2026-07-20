from __future__ import annotations

import asyncio
import base64
import contextlib
import datetime as _dt
import hashlib
import ipaddress
import logging
import os
import secrets
import socket
import ssl
import tempfile
import threading
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import urlsplit

from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.x509.oid import NameOID

from cayu.egress.broker import CapturedRequest, CapturedResponse, TransparentEgressBroker

_ONE_DAY = _dt.timedelta(days=1)
_CA_VALIDITY = _dt.timedelta(days=825)
_LEAF_VALIDITY = _dt.timedelta(days=365)
_MAX_REQUEST_BYTES = 8 * 1024 * 1024
_BROKER_TIMEOUT_S = 60.0
_TRANSPORT_TUNNEL_TARGET = "cayu-transport.invalid:443"
_logger = logging.getLogger(__name__)


class SessionCertificateAuthority:
    """Per-session CA that mints leaf certs for TLS interception.

    The sandbox is configured to trust ``ca_cert_pem()``; the proxy presents a
    freshly-minted leaf for whatever host the sandbox connects to, so unmodified
    HTTPS clients complete the handshake without app changes. The CA lives only
    for the session and is discarded on ``close()``.
    """

    def __init__(self) -> None:
        self._key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
        self._cert = self._build_ca_cert()
        self._contexts: dict[str, ssl.SSLContext] = {}
        self._tempdir = tempfile.mkdtemp(prefix="cayu-egress-ca-")
        self._lock = threading.Lock()

    def ca_cert_pem(self) -> bytes:
        return self._cert.public_bytes(serialization.Encoding.PEM)

    def server_ssl_context(self, host: str) -> ssl.SSLContext:
        with self._lock:
            cached = self._contexts.get(host)
            if cached is not None:
                return cached
            context = self._build_leaf_context(host)
            self._contexts[host] = context
            return context

    def close(self) -> None:
        import shutil

        shutil.rmtree(self._tempdir, ignore_errors=True)

    def _now(self) -> _dt.datetime:
        return _dt.datetime.now(_dt.UTC)

    def _build_ca_cert(self) -> x509.Certificate:
        name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "Cayu Egress Session CA")])
        now = self._now()
        public_key = self._key.public_key()
        return (
            x509.CertificateBuilder()
            .subject_name(name)
            .issuer_name(name)
            .public_key(public_key)
            .serial_number(x509.random_serial_number())
            .not_valid_before(now - _ONE_DAY)
            .not_valid_after(now + _CA_VALIDITY)
            .add_extension(x509.BasicConstraints(ca=True, path_length=0), critical=True)
            .add_extension(x509.SubjectKeyIdentifier.from_public_key(public_key), critical=False)
            .add_extension(
                x509.KeyUsage(
                    digital_signature=True,
                    key_cert_sign=True,
                    crl_sign=True,
                    key_encipherment=False,
                    content_commitment=False,
                    data_encipherment=False,
                    key_agreement=False,
                    encipher_only=False,
                    decipher_only=False,
                ),
                critical=True,
            )
            .sign(self._key, hashes.SHA256())
        )

    def _build_leaf_context(self, host: str) -> ssl.SSLContext:
        leaf_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
        now = self._now()
        cert = (
            x509.CertificateBuilder()
            .subject_name(x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, host)]))
            .issuer_name(self._cert.subject)
            .public_key(leaf_key.public_key())
            .serial_number(x509.random_serial_number())
            .not_valid_before(now - _ONE_DAY)
            .not_valid_after(now + _LEAF_VALIDITY)
            .add_extension(x509.SubjectAlternativeName([x509.DNSName(host)]), critical=False)
            .add_extension(x509.BasicConstraints(ca=False, path_length=None), critical=True)
            .add_extension(
                x509.SubjectKeyIdentifier.from_public_key(leaf_key.public_key()), critical=False
            )
            .add_extension(
                x509.AuthorityKeyIdentifier.from_issuer_public_key(self._key.public_key()),
                critical=False,
            )
            .sign(self._key, hashes.SHA256())
        )
        file_stem = _leaf_file_stem(host)
        certfile = os.path.join(self._tempdir, f"{file_stem}.cert.pem")
        keyfile = os.path.join(self._tempdir, f"{file_stem}.key.pem")
        _write_private(certfile, cert.public_bytes(serialization.Encoding.PEM))
        _write_private(
            keyfile,
            leaf_key.private_bytes(
                serialization.Encoding.PEM,
                serialization.PrivateFormat.TraditionalOpenSSL,
                serialization.NoEncryption(),
            ),
        )
        context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        context.load_cert_chain(certfile, keyfile)
        return context


def _leaf_file_stem(host: str) -> str:
    """Return a path-safe cache filename stem for an untrusted CONNECT host."""
    return hashlib.sha256(host.encode("utf-8", "surrogatepass")).hexdigest()


def _validated_connect_target(target: str) -> tuple[str, int]:
    try:
        split = urlsplit(f"//{target}")
        port = split.port
    except ValueError as exc:
        raise ValueError("CONNECT target has an invalid port.") from exc
    host = split.hostname
    if (
        host is None
        or split.username is not None
        or split.password is not None
        or split.path
        or split.query
        or split.fragment
        or port != 443
    ):
        raise ValueError("CONNECT target must be a hostname on port 443.")
    try:
        ipaddress.ip_address(host)
    except ValueError:
        pass
    else:
        raise ValueError("CONNECT target must not be an IP address.")
    return host.lower().rstrip("."), port


class TransparentEgressProxyServer:
    """A threaded HTTP CONNECT proxy that terminates TLS and calls the broker.

    Runs blocking socket work on worker threads and bridges each captured
    request to the async ``broker.handle_request`` on the provided event loop,
    so the broker's async vault/upstream code runs on the main loop while the
    proxy handles raw sockets. Direct-egress blocking is the runner adapter's
    job; this server only captures and forwards what reaches it.
    """

    def __init__(
        self,
        broker: TransparentEgressBroker,
        *,
        loop: asyncio.AbstractEventLoop,
        authority: SessionCertificateAuthority | None = None,
        host: str | Sequence[str] = "127.0.0.1",
        port: int = 0,
        max_workers: int = 16,
        transport_auth_token: bytes | None = None,
    ) -> None:
        listen_hosts = (host,) if isinstance(host, str) else tuple(host)
        if not listen_hosts or any(not item.strip() for item in listen_hosts):
            raise ValueError("Egress proxy listen hosts must be nonblank.")
        if len(set(listen_hosts)) != len(listen_hosts):
            raise ValueError("Egress proxy listen hosts must be unique.")
        if transport_auth_token is not None:
            if type(transport_auth_token) is not bytes:
                raise TypeError("Egress proxy transport authentication token must be bytes.")
            if not transport_auth_token:
                raise ValueError("Egress proxy transport authentication token must be nonempty.")
        self._broker = broker
        self._loop = loop
        self._authority = authority or SessionCertificateAuthority()
        self._hosts = listen_hosts
        self._port = port
        self._transport_auth_token = transport_auth_token
        self._sockets: list[socket.socket] = []
        self._executor = ThreadPoolExecutor(max_workers=max_workers)
        self._accept_threads: list[threading.Thread] = []
        self._stop = threading.Event()

    @property
    def authority(self) -> SessionCertificateAuthority:
        return self._authority

    @property
    def port(self) -> int:
        if not self._sockets:
            raise RuntimeError("Proxy server is not started.")
        return self._sockets[0].getsockname()[1]

    async def start(self) -> int:
        listeners: list[socket.socket] = []
        port = self._port
        try:
            for host in self._hosts:
                listener = self._open_listener(host, port)
                listeners.append(listener)
                if port == 0:
                    port = listener.getsockname()[1]
        except BaseException:
            for listener in listeners:
                listener.close()
            raise

        self._sockets = listeners
        self._accept_threads = [
            threading.Thread(target=self._accept_loop, args=(listener,), daemon=True)
            for listener in listeners
        ]
        for thread in self._accept_threads:
            thread.start()
        return self.port

    async def close(self) -> None:
        self._stop.set()
        for listener in self._sockets:
            listener.close()
        self._sockets = []
        if self._accept_threads:
            await asyncio.gather(
                *(asyncio.to_thread(thread.join, 5.0) for thread in self._accept_threads)
            )
            self._accept_threads = []
        self._executor.shutdown(wait=True, cancel_futures=True)
        self._authority.close()

    @staticmethod
    def _open_listener(host: str, port: int) -> socket.socket:
        addresses = socket.getaddrinfo(
            host,
            port,
            type=socket.SOCK_STREAM,
            flags=socket.AI_PASSIVE,
        )
        if not addresses:
            raise OSError(f"No listen address resolved for {host!r}.")
        family, sock_type, protocol, _, address = addresses[0]
        listener = socket.socket(family, sock_type, protocol)
        try:
            listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            if family == socket.AF_INET6:
                listener.setsockopt(socket.IPPROTO_IPV6, socket.IPV6_V6ONLY, 1)
            listener.bind(address)
            listener.listen(128)
            listener.settimeout(0.5)
        except BaseException:
            listener.close()
            raise
        return listener

    def _accept_loop(self, listener: socket.socket) -> None:
        while not self._stop.is_set():
            try:
                conn, _ = listener.accept()
            except TimeoutError:
                continue
            except OSError:
                break
            self._executor.submit(self._safe_handle, conn)

    def _safe_handle(self, conn: socket.socket) -> None:
        try:
            self._handle_connection(conn)
        except Exception:
            # A single malformed connection must never take down the proxy.
            pass
        finally:
            with contextlib.suppress(OSError):
                conn.close()

    def _handle_connection(self, conn: socket.socket) -> None:
        conn.settimeout(30.0)
        if self._transport_auth_token is not None:
            request_line, headers = _read_head(conn)
            if not self._transport_is_authenticated(request_line, headers):
                _logger.debug("Rejected unauthenticated egress proxy transport connection.")
                return
            _logger.debug("Accepted authenticated egress proxy transport connection.")
            conn.sendall(b"HTTP/1.1 200 Connection Established\r\n\r\n")
        # Read the head WITHOUT over-reading: for CONNECT, any following bytes are
        # the tunneled TLS ClientHello and must stay in the socket for wrap_socket.
        request_line, _headers = _read_head(conn)
        if not request_line:
            return
        method, target, _ = request_line.decode("latin-1").split(" ", 2)
        if method.upper() == "CONNECT":
            self._handle_connect(conn, target)
        else:
            conn.sendall(
                b"HTTP/1.1 405 Method Not Allowed\r\nContent-Length: 0\r\nConnection: close\r\n\r\n"
            )

    def _transport_is_authenticated(
        self,
        request_line: bytes,
        headers: dict[str, str],
    ) -> bool:
        if self._transport_auth_token is None:
            return True
        try:
            method, target, _version = request_line.decode("latin-1").split(" ", 2)
        except ValueError:
            return False
        if method.upper() != "CONNECT" or target != _TRANSPORT_TUNNEL_TARGET:
            return False
        presented = next(
            (value for key, value in headers.items() if key.lower() == "proxy-authorization"),
            "",
        )
        encoded = base64.b64encode(b"cayu:" + self._transport_auth_token).decode("ascii")
        return secrets.compare_digest(presented, f"Basic {encoded}")

    def _handle_connect(self, conn: socket.socket, target: str) -> None:
        try:
            host, port = _validated_connect_target(target)
        except ValueError:
            conn.sendall(
                b"HTTP/1.1 403 Forbidden\r\nContent-Length: 0\r\nConnection: close\r\n\r\n"
            )
            return
        conn.sendall(b"HTTP/1.1 200 Connection Established\r\n\r\n")
        context = self._authority.server_ssl_context(host)
        tls = context.wrap_socket(conn, server_side=True)
        try:
            request_line, headers = _read_head(tls)
            if not request_line:
                return
            method, path, _ = request_line.decode("latin-1").split(" ", 2)
            body = _read_body(tls, headers)
            split = urlsplit(path)
            captured = CapturedRequest(
                method=method,
                host=host,
                path=split.path or "/",
                protocol="https",
                port=port,
                query=split.query,
                headers=headers,
                body=body,
            )
            response = self._call_broker(captured)
            tls.sendall(_serialize_response(response))
        finally:
            _shutdown(tls)

    def _call_broker(self, request: CapturedRequest) -> CapturedResponse:
        future = asyncio.run_coroutine_threadsafe(self._broker.handle_request(request), self._loop)
        # Poll so a concurrent close() (which sets _stop) aborts the wait quickly
        # instead of blocking the whole shutdown for up to _BROKER_TIMEOUT_S.
        waited = 0.0
        while waited < _BROKER_TIMEOUT_S:
            if self._stop.is_set():
                future.cancel()
                raise RuntimeError("Egress proxy is shutting down.")
            try:
                return future.result(timeout=0.25)
            except TimeoutError:
                waited += 0.25
        future.cancel()
        raise TimeoutError("Broker did not respond within the timeout.")


class DualStackLoopbackEgressProxyServer(TransparentEgressProxyServer):
    """Expose one proxy port on both host loopback address families."""

    def __init__(
        self,
        broker: TransparentEgressBroker,
        *,
        loop: asyncio.AbstractEventLoop,
        authority: SessionCertificateAuthority | None = None,
        host: str = "127.0.0.1",
        port: int = 0,
        max_workers: int = 16,
    ) -> None:
        if host != "127.0.0.1":
            raise ValueError("Dual-stack loopback proxy host must be '127.0.0.1'.")
        super().__init__(
            broker,
            loop=loop,
            authority=authority,
            host=(host, "::1"),
            port=port,
            max_workers=max_workers,
        )


def _write_private(path: str, data: bytes) -> None:
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "wb") as handle:
        handle.write(data)


def _read_head(sock: socket.socket) -> tuple[bytes, dict[str, str]]:
    """Read request line + headers, stopping exactly at the blank line.

    Reads a byte at a time so it never consumes bytes past the header
    terminator — important for CONNECT, where the following bytes are the
    tunneled TLS ClientHello that must reach ``wrap_socket``.
    """
    buffer = bytearray()
    while not buffer.endswith(b"\r\n\r\n"):
        byte = sock.recv(1)
        if not byte:
            break
        buffer += byte
        if len(buffer) > _MAX_REQUEST_BYTES:
            raise ValueError("Request head too large.")
    head = bytes(buffer).split(b"\r\n\r\n", 1)[0]
    lines = head.split(b"\r\n")
    request_line = lines[0] if lines else b""
    headers: dict[str, str] = {}
    for line in lines[1:]:
        if not line:
            continue
        key, _, value = line.partition(b":")
        headers[key.decode("latin-1").strip()] = value.decode("latin-1").strip()
    return request_line, headers


def _read_body(sock: socket.socket, headers: dict[str, str]) -> bytes:
    if _is_chunked(headers):
        return _read_chunked(sock)
    length = _content_length(headers)
    if length is None:
        return b""
    return _recv_exact(sock, length)


def _recv_exact(sock: socket.socket, n: int) -> bytes:
    if n > _MAX_REQUEST_BYTES:
        raise ValueError("Request body too large.")
    buffer = bytearray()
    while len(buffer) < n:
        chunk = sock.recv(min(65536, n - len(buffer)))
        if not chunk:
            raise ValueError("Connection closed before the full body was received.")
        buffer += chunk
    return bytes(buffer)


def _read_line(sock: socket.socket) -> bytes:
    buffer = bytearray()
    while not buffer.endswith(b"\n"):
        byte = sock.recv(1)
        if not byte:
            break
        buffer += byte
        if len(buffer) > _MAX_REQUEST_BYTES:
            raise ValueError("Header line too long.")
    return bytes(buffer).rstrip(b"\r\n")


def _is_chunked(headers: dict[str, str]) -> bool:
    for key, value in headers.items():
        if key.lower() == "transfer-encoding" and "chunked" in value.lower():
            return True
    return False


def _read_chunked(sock: socket.socket) -> bytes:
    body = bytearray()
    while True:
        size_line = _read_line(sock)
        if not size_line:
            break
        token = size_line.split(b";", 1)[0].strip()
        try:
            size = int(token, 16)
        except ValueError as exc:
            raise ValueError("Invalid chunk size.") from exc
        if size == 0:
            # Consume any trailer headers up to the final blank line.
            while _read_line(sock):
                continue
            break
        body += _recv_exact(sock, size)
        _read_line(sock)  # trailing CRLF after the chunk data
        if len(body) > _MAX_REQUEST_BYTES:
            raise ValueError("Request body too large.")
    return bytes(body)


def _content_length(headers: dict[str, str]) -> int | None:
    for key, value in headers.items():
        if key.lower() == "content-length":
            try:
                return int(value.strip())
            except ValueError as exc:
                # A malformed Content-Length is an error, not a zero-length body:
                # silently dropping the body would forward a truncated request.
                raise ValueError(f"Invalid Content-Length: {value!r}") from exc
    return None


def _serialize_response(response: CapturedResponse) -> bytes:
    reason = _STATUS_REASON.get(response.status_code, "OK")
    lines = [f"HTTP/1.1 {response.status_code} {reason}"]
    headers = {k: v for k, v in response.headers.items() if k.lower() != "content-length"}
    headers["Content-Length"] = str(len(response.body))
    headers["Connection"] = "close"
    for key, value in headers.items():
        lines.append(f"{key}: {value}")
    head = ("\r\n".join(lines) + "\r\n\r\n").encode("latin-1")
    return head + response.body


def _shutdown(sock: socket.socket) -> None:
    with contextlib.suppress(OSError):
        sock.shutdown(socket.SHUT_RDWR)
    with contextlib.suppress(OSError):
        sock.close()


_STATUS_REASON = {
    200: "OK",
    201: "Created",
    400: "Bad Request",
    401: "Unauthorized",
    403: "Forbidden",
    404: "Not Found",
    500: "Internal Server Error",
    502: "Bad Gateway",
}
