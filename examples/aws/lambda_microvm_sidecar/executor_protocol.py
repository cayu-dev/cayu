"""Wire protocol between the sidecar's supervisor and its command executor.

In Kubernetes mode the supervisor and the executor run in two containers of one
pod that share only its network namespace. The executor listens on a Unix socket
in the abstract namespace, which that network namespace scopes, so no volume is
shared. Each request is one connection: the supervisor connects, sends one
frame, and reads one frame back. A frame is a 4-byte big-endian length followed
by that many bytes of UTF-8 JSON; bytes travel base64-encoded.

The executor authenticates the caller by its peer credentials (``SO_PEERCRED``):
only the configured supervisor UID is served. Every request is idempotent, so a
reset connection, such as one cut by a snapshot and restore, only causes a retry.
Only this module knows the transport; replacing it, for example with loopback
TCP and a token, leaves both sides unchanged.
"""

from __future__ import annotations

import contextlib
import json
import socket
import struct
import threading
import time
from collections.abc import Callable
from typing import Any

PROTOCOL_VERSION = 1
EXECUTOR_SOCKET_ENV = "CAYU_EXECUTOR_SOCKET"
# A leading "@" names an abstract socket; anything else is a filesystem path.
DEFAULT_EXECUTOR_SOCKET = "@cayu-sidecar-executor"
# Bytes moved by one stdin write or output read.
MAX_CHUNK_BYTES = 1024 * 1024
# The longest an executor holds a read, wait, or tree confirmation open.
MAX_WAIT_SECONDS = 20.0
# A chunk's base64 plus the request's other fields.
MAX_FRAME_BYTES = 2 * MAX_CHUNK_BYTES
DEFAULT_RECONNECT_SECONDS = 30.0
_REQUEST_TIMEOUT_SECONDS = 10.0
_RETRY_INTERVAL_SECONDS = 0.1
# Requests the executor answers only once the command lets it.
_BLOCKING_OPERATIONS = frozenset({"write"})
_LENGTH = struct.Struct(">I")
_PEER_CREDENTIALS = struct.Struct("3i")


class ExecutorError(OSError):
    """The executor refused a request; ``kind`` names why."""

    def __init__(self, kind: str, message: str) -> None:
        super().__init__(message)
        self.kind = kind


class ExecutorUnavailableError(ExecutorError):
    """The executor could not be reached within the reconnect window."""

    def __init__(self, message: str) -> None:
        super().__init__("unavailable", message)


class FrameError(ValueError):
    """A frame is oversized or is not a JSON object."""


def socket_address(address: str) -> str | bytes:
    if address.startswith("@"):
        if len(address) < 2:
            raise ValueError("an abstract executor socket needs a name after '@'")
        return b"\0" + address[1:].encode("utf-8")
    if not address.startswith("/"):
        raise ValueError("the executor socket must be '@name' or an absolute path")
    return address


def listen(address: str) -> socket.socket:
    listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        listener.bind(socket_address(address))
        listener.listen(64)
    except BaseException:
        listener.close()
        raise
    return listener


def peer_uid(connection: socket.socket) -> int:
    """The UID of the process on the other end, from the kernel (Linux only).

    The peer PID is not used: across containers under gVisor it can read 0.
    """
    raw = connection.getsockopt(socket.SOL_SOCKET, socket.SO_PEERCRED, _PEER_CREDENTIALS.size)
    return _PEER_CREDENTIALS.unpack(raw)[1]


def send_frame(connection: socket.socket, message: dict[str, Any]) -> None:
    body = json.dumps(message, separators=(",", ":")).encode("utf-8")
    if len(body) > MAX_FRAME_BYTES:
        raise FrameError(f"executor frame exceeds {MAX_FRAME_BYTES} bytes")
    connection.sendall(_LENGTH.pack(len(body)) + body)


def recv_frame(connection: socket.socket) -> dict[str, Any]:
    (length,) = _LENGTH.unpack(_recv_exact(connection, _LENGTH.size))
    if length > MAX_FRAME_BYTES:
        raise FrameError(f"executor frame exceeds {MAX_FRAME_BYTES} bytes")
    try:
        message = json.loads(_recv_exact(connection, length))
    except ValueError as exc:
        raise FrameError("executor frame is not JSON") from exc
    if type(message) is not dict:
        raise FrameError("executor frame is not a JSON object")
    return message


def _recv_exact(connection: socket.socket, size: int) -> bytes:
    data = bytearray()
    while len(data) < size:
        chunk = connection.recv(size - len(data))
        if not chunk:
            raise ConnectionResetError("executor connection closed mid-frame")
        data.extend(chunk)
    return bytes(data)


class ExecutorClient:
    """Send requests to the executor, reconnecting for ``reconnect_s`` after a failure.

    ``session`` is the supervisor's boot ID. The executor stops every command of
    an earlier session when a new one first calls, so a restarted supervisor
    never leaves processes nobody can reach. ``boot_id`` is the executor's boot
    ID from its latest answer; when it changes, every command it ran is gone,
    and ``on_restart`` is called on a thread of its own.
    """

    def __init__(
        self,
        address: str,
        *,
        session: str,
        reconnect_s: float = DEFAULT_RECONNECT_SECONDS,
        on_restart: Callable[[], object] | None = None,
    ) -> None:
        self.address = address
        self._socket_address = socket_address(address)
        self.session = session
        self.reconnect_s = reconnect_s
        self.on_restart = on_restart
        self.boot_id: str | None = None
        self._boot_lock = threading.Lock()

    def call(
        self,
        op: str,
        *,
        wait_s: float = 0.0,
        reconnect_s: float | None = None,
        **fields: Any,
    ) -> dict[str, Any]:
        """Send one request and return its answer.

        A ``write`` has no answer timeout: the executor answers once the command
        has taken the chunk, so a slow reader holds it back as a local pipe does.
        """
        request = {"op": op, "session": self.session, **fields}
        if wait_s:
            request["wait_s"] = wait_s
        window = self.reconnect_s if reconnect_s is None else reconnect_s
        timeout_s = None if op in _BLOCKING_OPERATIONS else _REQUEST_TIMEOUT_SECONDS + wait_s
        deadline: float | None = None
        while True:
            try:
                response = self._exchange(request, timeout_s=timeout_s)
                break
            except OSError as exc:
                if deadline is None:
                    deadline = time.monotonic() + window
                if time.monotonic() >= deadline:
                    raise ExecutorUnavailableError(
                        f"the command executor at {self.address} is unreachable: {exc}"
                    ) from exc
                time.sleep(_RETRY_INTERVAL_SECONDS)
        self._observe_boot(response.get("boot_id"))
        error = response.get("error")
        if error is None:
            return response
        if type(error) is not dict:
            raise FrameError("executor error is not an object")
        if error.get("kind") == "os_error" and type(error.get("errno")) is int:
            # The spawn's own failure, such as a missing program, as Popen raises it.
            raise OSError(error["errno"], str(error.get("message")), error.get("filename"))
        raise ExecutorError(str(error.get("kind")), str(error.get("message")))

    def _observe_boot(self, boot_id: object) -> None:
        if type(boot_id) is not str:
            raise FrameError("executor answer carries no boot_id")
        with self._boot_lock:
            previous, self.boot_id = self.boot_id, boot_id
        if previous is not None and previous != boot_id and self.on_restart is not None:
            # Not on the caller's thread, which may hold locks the handler needs.
            threading.Thread(target=self.on_restart, name="cayu-executor-restart").start()

    def _exchange(self, request: dict[str, Any], *, timeout_s: float | None) -> dict[str, Any]:
        connection = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        with contextlib.closing(connection):
            connection.settimeout(_REQUEST_TIMEOUT_SECONDS)
            connection.connect(self._socket_address)
            connection.settimeout(timeout_s)
            send_frame(connection, request)
            return recv_frame(connection)
