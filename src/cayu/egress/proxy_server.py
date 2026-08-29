from __future__ import annotations

import asyncio
import base64
import contextlib
import datetime as _dt
import hashlib
import logging
import os
import secrets
import select
import shutil
import socket
import ssl
import tempfile
import threading
import time
from collections import OrderedDict
from collections.abc import Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from http import HTTPStatus
from urllib.parse import urlsplit

from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.x509.oid import NameOID

from cayu.egress.broker import (
    CAYU_EGRESS_ERROR_HEADER,
    CapturedRequest,
    CapturedResponse,
    TransparentEgressBroker,
    _ConnectDestinationAdmission,
)
from cayu.egress.destinations import normalize_egress_hostname

_ONE_DAY = _dt.timedelta(days=1)
_CA_VALIDITY = _dt.timedelta(days=825)
_LEAF_VALIDITY = _dt.timedelta(days=365)
_MAX_REQUEST_BYTES = 8 * 1024 * 1024
_MAX_REQUEST_HEAD_BYTES = 64 * 1024
_MAX_REQUEST_LINE_BYTES = 8 * 1024
_MAX_HEADER_LINE_BYTES = 8 * 1024
_MAX_HEADER_COUNT = 100
_REQUEST_HEAD_TIMEOUT_S = 10.0
_REQUEST_BODY_TIMEOUT_S = 30.0
_TLS_HANDSHAKE_TIMEOUT_S = 10.0
_SOCKET_POLL_INTERVAL_S = 0.1
_MAX_PROXY_WORKERS = 64
_MAX_CERTIFICATE_CACHE_ENTRIES = 128
_CERTIFICATE_CACHE_TTL_S = 15 * 60.0
_CERTIFICATE_GENERATION_WAIT_S = 5.0
_BROKER_TIMEOUT_S = 60.0
_TRANSPORT_TUNNEL_TARGET = "cayu-transport.invalid:443"
_PLAIN_HTTP_DENIAL_RESPONSE = (
    "HTTP/1.1 403 Forbidden\r\n"
    f"{CAYU_EGRESS_ERROR_HEADER}: destination_denied\r\n"
    "Content-Length: 0\r\n"
    "Connection: close\r\n"
    "\r\n"
).encode("ascii")
_CONNECT_DENIAL_RESPONSE = (
    "HTTP/1.1 403 Forbidden\r\n"
    f"{CAYU_EGRESS_ERROR_HEADER}: destination_denied\r\n"
    "Content-Length: 0\r\n"
    "Connection: close\r\n"
    "\r\n"
).encode("ascii")
_CAPACITY_EXHAUSTED_RESPONSE = (
    "HTTP/1.1 503 Service Unavailable\r\n"
    f"{CAYU_EGRESS_ERROR_HEADER}: proxy_capacity_exhausted\r\n"
    "Content-Length: 0\r\n"
    "Connection: close\r\n"
    "\r\n"
).encode("ascii")
_logger = logging.getLogger(__name__)


class _ProxyProtocolError(ValueError):
    """A fixed-diagnostic protocol rejection with no caller input attached."""

    def __init__(self, error_code: str, message: str) -> None:
        super().__init__(message)
        self.error_code = error_code


class _CertificateCapacityError(RuntimeError):
    pass


@dataclass(frozen=True)
class _CombinedStopSignal:
    first: threading.Event
    second: threading.Event

    def is_set(self) -> bool:
        return self.first.is_set() or self.second.is_set()


@dataclass(frozen=True)
class _CertificateCacheEntry:
    context: ssl.SSLContext
    expires_at: float


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
        self._contexts: OrderedDict[str, _CertificateCacheEntry] = OrderedDict()
        self._tempdir = tempfile.mkdtemp(prefix="cayu-egress-ca-")
        self._lock = threading.Lock()
        self._closed = False
        self._cleanup_complete = False

    def ca_cert_pem(self) -> bytes:
        return self._cert.public_bytes(serialization.Encoding.PEM)

    def server_ssl_context(
        self,
        host: str,
        *,
        stop: threading.Event | _CombinedStopSignal | None = None,
    ) -> ssl.SSLContext:
        deadline = time.monotonic() + _CERTIFICATE_GENERATION_WAIT_S
        while True:
            if stop is not None and stop.is_set():
                raise _CertificateCapacityError("Certificate generation was stopped.")
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise _CertificateCapacityError("Certificate generation capacity is exhausted.")
            if self._lock.acquire(timeout=min(_SOCKET_POLL_INTERVAL_S, remaining)):
                break
        try:
            if self._closed:
                raise _CertificateCapacityError("Certificate authority is closed.")
            if stop is not None and stop.is_set():
                raise _CertificateCapacityError("Certificate generation was stopped.")
            now = time.monotonic()
            self._prune_expired_contexts(now)
            cached = self._contexts.pop(host, None)
            if cached is not None:
                self._contexts[host] = cached
                return cached.context
            context = self._build_leaf_context(host)
            if stop is not None and stop.is_set():
                raise _CertificateCapacityError("Certificate generation was stopped.")
            while len(self._contexts) >= _MAX_CERTIFICATE_CACHE_ENTRIES:
                self._contexts.popitem(last=False)
            self._contexts[host] = _CertificateCacheEntry(
                context=context,
                expires_at=now + _CERTIFICATE_CACHE_TTL_S,
            )
            return context
        finally:
            self._lock.release()

    def close(self) -> None:
        with self._lock:
            if self._cleanup_complete:
                return
            self._closed = True
            self._contexts.clear()
            with contextlib.suppress(FileNotFoundError):
                shutil.rmtree(self._tempdir)
            self._cleanup_complete = True

    def _prune_expired_contexts(self, now: float) -> None:
        expired = [host for host, entry in self._contexts.items() if entry.expires_at <= now]
        for host in expired:
            self._contexts.pop(host, None)

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
        try:
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
        finally:
            with contextlib.suppress(OSError):
                os.unlink(certfile)
            with contextlib.suppress(OSError):
                os.unlink(keyfile)


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
    return normalize_egress_hostname(host, field_name="CONNECT target"), port


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
        owns_authority: bool = True,
    ) -> None:
        if type(max_workers) is not int:
            raise TypeError("max_workers must be an integer.")
        if not 1 <= max_workers <= _MAX_PROXY_WORKERS:
            raise ValueError(f"max_workers must be between 1 and {_MAX_PROXY_WORKERS}.")
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
        if type(owns_authority) is not bool:
            raise TypeError("owns_authority must be a bool.")
        if authority is None and not owns_authority:
            raise ValueError("A non-owning proxy server requires an existing authority.")
        self._broker = broker
        self._loop = loop
        self._authority = authority or SessionCertificateAuthority()
        self._owns_authority = owns_authority
        self._hosts = listen_hosts
        self._port = port
        self._transport_auth_token = transport_auth_token
        self._sockets: list[socket.socket] = []
        self._executor = ThreadPoolExecutor(max_workers=max_workers)
        self._connection_slots = threading.BoundedSemaphore(max_workers)
        self._accept_threads: list[threading.Thread] = []
        self._stop = threading.Event()
        self._connections: set[socket.socket] = set()
        self._connections_lock = threading.Lock()
        self._close_task: asyncio.Task[None] | None = None

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
        retry_incomplete = False
        if self._close_task is not None and self._close_task.done():
            try:
                retry_incomplete = self._close_task.exception() is not None
            except asyncio.CancelledError:
                retry_incomplete = True
        if self._close_task is None or retry_incomplete:
            self._close_task = asyncio.create_task(self._close_owned_resources())
        await asyncio.shield(self._close_task)

    async def _close_owned_resources(self) -> None:
        self._stop.set()
        for listener in self._sockets:
            listener.close()
        self._sockets = []
        with self._connections_lock:
            connections = tuple(self._connections)
        for connection in connections:
            _interrupt(connection)
        errors: list[BaseException] = []
        try:
            if self._accept_threads:
                try:
                    await asyncio.gather(
                        *(asyncio.to_thread(thread.join, 5.0) for thread in self._accept_threads)
                    )
                except BaseException as exc:
                    errors.append(exc)
                self._accept_threads = []
            try:
                # Every submitted handler already owns one bounded connection
                # slot and its socket is tracked above. Let queued handlers enter
                # ``_safe_handle`` so that ownership is released by the same
                # exactly-once cleanup path as handlers that already started.
                await asyncio.to_thread(self._executor.shutdown, wait=True, cancel_futures=False)
            except BaseException as exc:
                errors.append(exc)
            try:
                await self._broker.settle_active_operations()
            except BaseException as exc:
                errors.append(exc)
        finally:
            for connection in connections:
                with contextlib.suppress(OSError):
                    connection.close()
            if self._owns_authority:
                try:
                    self._authority.close()
                except BaseException as exc:
                    errors.append(exc)
                else:
                    self._owns_authority = False
        if errors:
            raise BaseExceptionGroup("Egress proxy shutdown was incomplete.", errors)

    def adopt_authority_ownership(self, authority: SessionCertificateAuthority) -> None:
        """Take teardown ownership during an atomic fresh-path handoff."""

        if authority is not self._authority:
            raise ValueError("Proxy authority ownership can only transfer to the same CA.")
        if self._owns_authority:
            raise RuntimeError("Proxy server already owns its certificate authority.")
        self._owns_authority = True

    def relinquish_authority_ownership(self, authority: SessionCertificateAuthority) -> None:
        """Relinquish teardown ownership after a replacement server takes it."""

        if authority is not self._authority:
            raise ValueError("Proxy authority ownership can only transfer for the same CA.")
        if not self._owns_authority:
            raise RuntimeError("Proxy server does not own its certificate authority.")
        self._owns_authority = False

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
            if not self._connection_slots.acquire(blocking=False):
                with contextlib.suppress(Exception):
                    _send_all(
                        conn,
                        _CAPACITY_EXHAUSTED_RESPONSE,
                        stop=self._stop,
                        timeout_s=_SOCKET_POLL_INTERVAL_S,
                    )
                with contextlib.suppress(OSError):
                    conn.close()
                continue
            with self._connections_lock:
                if self._stop.is_set():
                    should_close = True
                else:
                    self._connections.add(conn)
                    should_close = False
            if should_close:
                conn.close()
                self._connection_slots.release()
                break
            try:
                self._executor.submit(self._safe_handle, conn)
            except BaseException:
                with self._connections_lock:
                    self._connections.discard(conn)
                with contextlib.suppress(OSError):
                    conn.close()
                self._connection_slots.release()
                if self._stop.is_set():
                    break
                raise

    def _safe_handle(self, conn: socket.socket) -> None:
        try:
            self._handle_connection(conn)
        except _ProxyProtocolError as exc:
            _send_protocol_rejection(conn, exc, stop=self._stop)
        except Exception:
            # A single malformed connection must never take down the proxy.
            pass
        finally:
            with self._connections_lock:
                self._connections.discard(conn)
            with contextlib.suppress(OSError):
                conn.close()
            self._connection_slots.release()

    def _replace_tracked_connection(
        self,
        previous: socket.socket,
        replacement: socket.socket,
    ) -> bool:
        """Atomically transfer shutdown ownership after TLS consumes a raw socket."""

        with self._connections_lock:
            if self._stop.is_set() or previous not in self._connections:
                accepted = False
            else:
                self._connections.remove(previous)
                self._connections.add(replacement)
                accepted = True
        if not accepted:
            _shutdown(replacement)
        return accepted

    def _discard_tracked_connection(self, connection: socket.socket) -> None:
        with self._connections_lock:
            self._connections.discard(connection)

    def _handle_connection(self, conn: socket.socket) -> None:
        if self._transport_auth_token is not None:
            request_line, headers = _read_head(conn, stop=self._stop)
            if not self._transport_is_authenticated(request_line, headers):
                _logger.debug("Rejected unauthenticated egress proxy transport connection.")
                return
            _logger.debug("Accepted authenticated egress proxy transport connection.")
            _send_all(
                conn,
                b"HTTP/1.1 200 Connection Established\r\n\r\n",
                stop=self._stop,
            )
        # Read the head WITHOUT over-reading: for CONNECT, any following bytes are
        # the tunneled TLS ClientHello and must stay in the socket for wrap_socket.
        request_line, _headers = _read_head(conn, stop=self._stop)
        if not request_line:
            return
        method, target, _ = _parse_request_line(request_line)
        if method.upper() == "CONNECT":
            self._handle_connect(conn, target)
        else:
            _send_all(conn, _PLAIN_HTTP_DENIAL_RESPONSE, stop=self._stop)

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
            _send_all(conn, _CONNECT_DENIAL_RESPONSE, stop=self._stop)
            return
        admission = self._begin_connect_destination_admission(host, port)
        if admission is None:
            _send_all(conn, _CONNECT_DENIAL_RESPONSE, stop=self._stop)
            return
        try:
            admission_stop = _CombinedStopSignal(self._stop, admission.revoked)
            try:
                context = self._authority.server_ssl_context(host, stop=admission_stop)
            except _CertificateCapacityError:
                response = (
                    _CONNECT_DENIAL_RESPONSE
                    if admission.revoked.is_set()
                    else _CAPACITY_EXHAUSTED_RESPONSE
                )
                _send_all(conn, response, stop=self._stop)
                return
            if admission.revoked.is_set() or not self._connect_destination_admission_is_active(
                admission
            ):
                _send_all(conn, _CONNECT_DENIAL_RESPONSE, stop=self._stop)
                return
            _send_all(
                conn,
                b"HTTP/1.1 200 Connection Established\r\n\r\n",
                stop=self._stop,
            )
        finally:
            self._end_connect_destination_admission(admission)
        tls = context.wrap_socket(conn, server_side=True, do_handshake_on_connect=False)
        if not self._replace_tracked_connection(conn, tls):
            return
        try:
            _complete_tls_handshake(tls, stop=self._stop)
            request_line, headers = _read_head(tls, stop=self._stop)
            if not request_line:
                return
            method, path, _ = _parse_request_line(request_line)
            body = _read_body(tls, headers, stop=self._stop)
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
            response = self._call_broker(captured, client=tls)
            response_head = _serialize_response_head(response)
            _send_all(tls, response_head, stop=self._stop)
            _send_all(tls, response.body, stop=self._stop)
        except _ProxyProtocolError as exc:
            _send_protocol_rejection(tls, exc, stop=self._stop)
        finally:
            self._discard_tracked_connection(tls)
            _shutdown(tls)

    def _begin_connect_destination_admission(
        self,
        host: str,
        port: int,
    ) -> _ConnectDestinationAdmission | None:
        future = asyncio.run_coroutine_threadsafe(
            self._broker.begin_connect_destination_admission(host=host, port=port),
            self._loop,
        )
        waited = 0.0
        while waited < _BROKER_TIMEOUT_S:
            if self._stop.is_set():
                self._settle_aborted_connect_admission(future)
                raise RuntimeError("Egress proxy is shutting down.")
            try:
                return future.result(timeout=0.25)
            except TimeoutError:
                waited += 0.25
            except Exception as exc:
                raise _ProxyProtocolError(
                    "proxy_unavailable",
                    "CONNECT admission failed.",
                ) from exc
        self._settle_aborted_connect_admission(future)
        raise _ProxyProtocolError("proxy_unavailable", "CONNECT admission timed out.")

    def _settle_aborted_connect_admission(
        self,
        future: Future[_ConnectDestinationAdmission | None],
    ) -> None:
        """Positively release an admission even if result delivery is delayed."""

        try:
            admission = future.result(timeout=_BROKER_TIMEOUT_S)
        except TimeoutError:
            future.add_done_callback(self._release_completed_connect_admission)
            return
        except Exception:
            return
        if admission is not None:
            self._end_connect_destination_admission(admission)

    def _release_completed_connect_admission(
        self,
        future: Future[_ConnectDestinationAdmission | None],
    ) -> None:
        """Schedule cleanup without blocking the broker event-loop callback thread."""

        try:
            admission = future.result()
        except Exception:
            return
        if admission is not None:
            asyncio.run_coroutine_threadsafe(
                self._broker.end_connect_destination_admission(admission),
                self._loop,
            )

    def _connect_destination_admission_is_active(
        self,
        admission: _ConnectDestinationAdmission,
    ) -> bool:
        future = asyncio.run_coroutine_threadsafe(
            self._broker.connect_destination_admission_is_active(admission),
            self._loop,
        )
        waited = 0.0
        while waited < _BROKER_TIMEOUT_S:
            if self._stop.is_set():
                future.cancel()
                raise RuntimeError("Egress proxy is shutting down.")
            try:
                return future.result(timeout=0.25)
            except TimeoutError:
                waited += 0.25
            except Exception as exc:
                raise _ProxyProtocolError(
                    "proxy_unavailable",
                    "CONNECT admission revalidation failed.",
                ) from exc
        future.cancel()
        raise _ProxyProtocolError("proxy_unavailable", "CONNECT admission timed out.")

    def _end_connect_destination_admission(
        self,
        admission: _ConnectDestinationAdmission,
    ) -> None:
        future = asyncio.run_coroutine_threadsafe(
            self._broker.end_connect_destination_admission(admission),
            self._loop,
        )
        try:
            future.result(timeout=_BROKER_TIMEOUT_S)
        except TimeoutError as exc:
            raise _ProxyProtocolError(
                "proxy_unavailable",
                "CONNECT admission release timed out.",
            ) from exc
        except Exception as exc:
            raise _ProxyProtocolError(
                "proxy_unavailable",
                "CONNECT admission release failed.",
            ) from exc

    def _call_broker(
        self,
        request: CapturedRequest,
        *,
        client: ssl.SSLSocket,
    ) -> CapturedResponse:
        future = asyncio.run_coroutine_threadsafe(self._broker.handle_request(request), self._loop)
        # Poll so a concurrent close() (which sets _stop) aborts the wait quickly
        # instead of blocking the whole shutdown for up to _BROKER_TIMEOUT_S.
        waited = 0.0
        while waited < _BROKER_TIMEOUT_S:
            if self._stop.is_set():
                future.cancel()
                raise RuntimeError("Egress proxy is shutting down.")
            if _client_disconnected(client):
                future.cancel()
                raise _ProxyProtocolError(
                    "client_disconnected",
                    "Client disconnected before the broker response was available.",
                )
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


def _read_head(
    sock: socket.socket,
    *,
    stop: threading.Event | None = None,
    timeout_s: float = _REQUEST_HEAD_TIMEOUT_S,
) -> tuple[bytes, dict[str, str]]:
    """Read request line + headers, stopping exactly at the blank line.

    Reads a byte at a time so it never consumes bytes past the header
    terminator — important for CONNECT, where the following bytes are the
    tunneled TLS ClientHello that must reach ``wrap_socket``.
    """
    deadline = time.monotonic() + timeout_s
    buffer = bytearray()
    line = bytearray()
    line_index = 0
    header_count = 0
    while not buffer.endswith(b"\r\n\r\n"):
        byte = _recv_with_deadline(
            sock,
            1,
            stop=stop,
            deadline=deadline,
            timeout_error_code="request_head_timeout",
            timeout_message="Request head deadline exceeded.",
        )
        if not byte:
            if not buffer:
                return b"", {}
            raise _ProxyProtocolError(
                "malformed_request_head",
                "Connection closed before the request head was complete.",
            )
        buffer += byte
        line += byte
        if len(buffer) > _MAX_REQUEST_HEAD_BYTES:
            raise _ProxyProtocolError("request_head_too_large", "Request head too large.")
        if line.endswith(b"\r\n"):
            content_length = len(line) - 2
            if line_index == 0:
                if content_length > _MAX_REQUEST_LINE_BYTES:
                    raise _ProxyProtocolError(
                        "request_line_too_large",
                        "Request line too large.",
                    )
            elif content_length:
                header_count += 1
                if header_count > _MAX_HEADER_COUNT:
                    raise _ProxyProtocolError(
                        "too_many_headers",
                        "Request has too many headers.",
                    )
                if content_length > _MAX_HEADER_LINE_BYTES:
                    raise _ProxyProtocolError(
                        "header_line_too_large",
                        "Request header line too large.",
                    )
            line_index += 1
            line.clear()
        else:
            line_limit = _MAX_REQUEST_LINE_BYTES if line_index == 0 else _MAX_HEADER_LINE_BYTES
            # One extra byte is allowed only for the possible leading CR in CRLF.
            if len(line) > line_limit + int(line.endswith(b"\r")):
                error_code = (
                    "request_line_too_large" if line_index == 0 else "header_line_too_large"
                )
                raise _ProxyProtocolError(error_code, "Request head line too large.")
    head = bytes(buffer).split(b"\r\n\r\n", 1)[0]
    lines = head.split(b"\r\n")
    request_line = lines[0] if lines else b""
    headers: dict[str, str] = {}
    for line in lines[1:]:
        if not line:
            continue
        key, separator, value = line.partition(b":")
        if not separator or not key.strip() or key[:1].isspace():
            raise _ProxyProtocolError("malformed_request_head", "Malformed request header.")
        headers[key.decode("latin-1").strip()] = value.decode("latin-1").strip()
    return request_line, headers


def _parse_request_line(request_line: bytes) -> tuple[str, str, str]:
    malformed = False
    try:
        method, target, version = request_line.decode("latin-1").split(" ", 2)
    except ValueError:
        malformed = True
        method = target = version = ""
    if malformed or not method or not target or version not in {"HTTP/1.0", "HTTP/1.1"}:
        raise _ProxyProtocolError("malformed_request_head", "Malformed request line.")
    return method, target, version


def _read_body(
    sock: socket.socket,
    headers: dict[str, str],
    *,
    stop: threading.Event | None = None,
) -> bytes:
    deadline = time.monotonic() + _REQUEST_BODY_TIMEOUT_S
    if _is_chunked(headers):
        return _read_chunked(sock, stop=stop, deadline=deadline)
    length = _content_length(headers)
    if length is None:
        return b""
    return _recv_exact(sock, length, stop=stop, deadline=deadline)


def _recv_exact(
    sock: socket.socket,
    n: int,
    *,
    stop: threading.Event | None = None,
    deadline: float | None = None,
) -> bytes:
    if n < 0:
        raise _ProxyProtocolError("malformed_request_body", "Negative body length.")
    if n > _MAX_REQUEST_BYTES:
        raise _ProxyProtocolError("request_body_too_large", "Request body too large.")
    deadline = deadline or (time.monotonic() + _REQUEST_BODY_TIMEOUT_S)
    buffer = bytearray()
    while len(buffer) < n:
        chunk = _recv_with_deadline(
            sock,
            min(65536, n - len(buffer)),
            stop=stop,
            deadline=deadline,
            timeout_error_code="request_body_timeout",
            timeout_message="Request body deadline exceeded.",
        )
        if not chunk:
            raise _ProxyProtocolError(
                "malformed_request_body",
                "Connection closed before the full body was received.",
            )
        buffer += chunk
    return bytes(buffer)


def _read_line(
    sock: socket.socket,
    *,
    stop: threading.Event | None,
    deadline: float,
) -> bytes:
    buffer = bytearray()
    while not buffer.endswith(b"\n"):
        byte = _recv_with_deadline(
            sock,
            1,
            stop=stop,
            deadline=deadline,
            timeout_error_code="request_body_timeout",
            timeout_message="Request body deadline exceeded.",
        )
        if not byte:
            raise _ProxyProtocolError(
                "malformed_request_body",
                "Connection closed before a body line was complete.",
            )
        buffer += byte
        if len(buffer) > _MAX_HEADER_LINE_BYTES + 2:
            raise _ProxyProtocolError("header_line_too_large", "Body framing line too long.")
    return bytes(buffer).rstrip(b"\r\n")


def _is_chunked(headers: dict[str, str]) -> bool:
    for key, value in headers.items():
        if key.lower() == "transfer-encoding" and "chunked" in value.lower():
            return True
    return False


def _read_chunked(
    sock: socket.socket,
    *,
    stop: threading.Event | None,
    deadline: float,
) -> bytes:
    body = bytearray()
    while True:
        size_line = _read_line(sock, stop=stop, deadline=deadline)
        if not size_line:
            raise _ProxyProtocolError("malformed_request_body", "Missing chunk size.")
        token = size_line.split(b";", 1)[0].strip()
        invalid_size = False
        try:
            size = int(token, 16)
        except ValueError:
            invalid_size = True
            size = -1
        if invalid_size or size < 0:
            raise _ProxyProtocolError("malformed_request_body", "Invalid chunk size.")
        if size == 0:
            # Consume any trailer headers up to the final blank line.
            trailer_count = 0
            while _read_line(sock, stop=stop, deadline=deadline):
                trailer_count += 1
                if trailer_count > _MAX_HEADER_COUNT:
                    raise _ProxyProtocolError("too_many_headers", "Too many trailer headers.")
            break
        if size > _MAX_REQUEST_BYTES - len(body):
            raise _ProxyProtocolError("request_body_too_large", "Request body too large.")
        body += _recv_exact(sock, size, stop=stop, deadline=deadline)
        if _read_line(sock, stop=stop, deadline=deadline):
            raise _ProxyProtocolError("malformed_request_body", "Malformed chunk terminator.")
    return bytes(body)


def _content_length(headers: dict[str, str]) -> int | None:
    for key, value in headers.items():
        if key.lower() == "content-length":
            invalid_length = False
            try:
                length = int(value.strip())
            except ValueError:
                invalid_length = True
                length = -1
            # A malformed Content-Length is an error, not a zero-length body:
            # silently dropping the body would forward a truncated request.
            if invalid_length or length < 0:
                raise _ProxyProtocolError(
                    "malformed_request_body",
                    "Invalid Content-Length.",
                )
            return length
    return None


def _serialize_response_head(response: CapturedResponse) -> bytes:
    try:
        reason = HTTPStatus(response.status_code).phrase
    except ValueError:
        reason = "Unknown Status"
    lines = [f"HTTP/1.1 {response.status_code} {reason}"]
    headers = {k: v for k, v in response.headers.items() if k.lower() != "content-length"}
    headers["Content-Length"] = str(len(response.body))
    headers["Connection"] = "close"
    for key, value in headers.items():
        lines.append(f"{key}: {value}")
    return ("\r\n".join(lines) + "\r\n\r\n").encode("latin-1")


def _serialize_response(response: CapturedResponse) -> bytes:
    """Compatibility helper for tests; production sends the bounded body separately."""

    return _serialize_response_head(response) + response.body


def _complete_tls_handshake(
    sock: ssl.SSLSocket,
    *,
    stop: threading.Event | None,
) -> None:
    deadline = time.monotonic() + _TLS_HANDSHAKE_TIMEOUT_S
    while True:
        if stop is not None and stop.is_set():
            raise _ProxyProtocolError("proxy_shutting_down", "Proxy is shutting down.")
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise _ProxyProtocolError("tls_handshake_timeout", "TLS handshake timed out.")
        sock.settimeout(min(_SOCKET_POLL_INTERVAL_S, remaining))
        try:
            sock.do_handshake()
            return
        except (TimeoutError, ssl.SSLWantReadError, ssl.SSLWantWriteError):
            continue


def _recv_with_deadline(
    sock: socket.socket,
    size: int,
    *,
    stop: threading.Event | None,
    deadline: float,
    timeout_error_code: str,
    timeout_message: str,
) -> bytes:
    while True:
        if stop is not None and stop.is_set():
            raise _ProxyProtocolError("proxy_shutting_down", "Proxy is shutting down.")
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise _ProxyProtocolError(timeout_error_code, timeout_message)
        sock.settimeout(min(_SOCKET_POLL_INTERVAL_S, remaining))
        try:
            return sock.recv(size)
        except (TimeoutError, ssl.SSLWantReadError, ssl.SSLWantWriteError):
            continue


def _send_all(
    sock: socket.socket,
    data: bytes,
    *,
    stop: threading.Event | None,
    timeout_s: float = _REQUEST_BODY_TIMEOUT_S,
) -> None:
    if not data:
        return
    deadline = time.monotonic() + timeout_s
    view = memoryview(data)
    sent = 0
    while sent < len(view):
        if stop is not None and stop.is_set():
            raise _ProxyProtocolError("proxy_shutting_down", "Proxy is shutting down.")
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise _ProxyProtocolError("response_write_timeout", "Response write timed out.")
        sock.settimeout(min(_SOCKET_POLL_INTERVAL_S, remaining))
        try:
            written = sock.send(view[sent:])
        except (TimeoutError, ssl.SSLWantReadError, ssl.SSLWantWriteError):
            continue
        if written <= 0:
            raise OSError("Socket closed during response write.")
        sent += written


def _send_protocol_rejection(
    sock: socket.socket,
    error: _ProxyProtocolError,
    *,
    stop: threading.Event | None,
) -> None:
    if error.error_code in {"client_disconnected", "proxy_shutting_down"}:
        return
    if error.error_code.endswith("_timeout"):
        status, reason = 408, "Request Timeout"
    elif error.error_code == "proxy_unavailable":
        status, reason = 503, "Service Unavailable"
    elif error.error_code in {
        "request_head_too_large",
        "request_line_too_large",
        "header_line_too_large",
        "too_many_headers",
    }:
        status, reason = 431, "Request Header Fields Too Large"
    else:
        status, reason = 400, "Bad Request"
    response = (
        f"HTTP/1.1 {status} {reason}\r\n"
        f"{CAYU_EGRESS_ERROR_HEADER}: {error.error_code}\r\n"
        "Content-Length: 0\r\n"
        "Connection: close\r\n"
        "\r\n"
    ).encode("ascii")
    with contextlib.suppress(Exception):
        _send_all(sock, response, stop=stop, timeout_s=1.0)


def _shutdown(sock: socket.socket) -> None:
    with contextlib.suppress(OSError):
        sock.shutdown(socket.SHUT_RDWR)
    with contextlib.suppress(OSError):
        sock.close()


def _interrupt(sock: socket.socket) -> None:
    """Wake a blocking handler without mutating its SSL wrapper concurrently."""

    try:
        duplicate = socket.socket(
            sock.family,
            sock.type,
            sock.proto,
            fileno=socket.dup(sock.fileno()),
        )
    except OSError:
        return
    try:
        with contextlib.suppress(OSError):
            duplicate.shutdown(socket.SHUT_RDWR)
    finally:
        duplicate.close()


def _client_disconnected(sock: ssl.SSLSocket) -> bool:
    """Observe peer closure while an async broker call owns the request.

    One proxy connection carries exactly one request and is always closed after
    its response, so consuming unexpected pipelined input cannot affect a later
    request. A readable TLS socket is probed without blocking; EOF or a socket
    failure means the broker operation no longer has a response consumer.
    """

    try:
        readable, _, _ = select.select((sock,), (), (), 0)
    except (OSError, ValueError):
        return True
    if not readable:
        return False
    try:
        return sock.recv(1) == b""
    except (TimeoutError, ssl.SSLWantReadError, ssl.SSLWantWriteError):
        return False
    except OSError:
        return True
