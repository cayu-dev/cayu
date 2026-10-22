"""Guest listeners whose TCP connections the owner relays to a host service."""

from __future__ import annotations

import base64
import binascii
import contextlib
import socket
import threading
import time
from collections.abc import Callable
from typing import Any

from .supervisor import (
    CommandConflictError,
    CommandNotFoundError,
    CommandRequestError,
)

# The entrypoint accepts exactly this range from the agent namespace, on the
# veth gateway only, so a forward is reachable from agent commands and from
# nowhere else.
FORWARD_LISTEN_HOST = "192.0.2.1"
FORWARD_PORTS = range(18500, 18564)
MAX_FORWARD_CONNECTIONS = 32
# Unread guest bytes per connection. Beyond it the reader stops draining the
# socket, so a slow host applies TCP backpressure instead of growing memory.
FORWARD_WINDOW_BYTES = 4 * 1024 * 1024
MAX_FORWARD_CHUNK_BYTES = 4 * 1024 * 1024
MAX_FORWARD_WAIT_SECONDS = 20.0
MAX_FORWARD_LIFETIME_SECONDS = 7 * 24 * 3600
_SEND_TIMEOUT_SECONDS = 30.0
_TOMBSTONE_TTL_SECONDS = 300.0
_MAX_TOMBSTONES = 1024
_RECEIVE_CHUNK_BYTES = 64 * 1024


class _Connection:
    """One accepted guest connection, addressed by byte offsets in both directions.

    Reading at an offset acknowledges every earlier guest byte, which is then
    dropped; reading the same offset again recovers a lost response. A write
    names the offset it starts at, so a replayed chunk is skipped.
    """

    def __init__(self, client: socket.socket, window_bytes: int) -> None:
        self.socket = client
        self.window_bytes = window_bytes
        self.condition = threading.Condition()
        self.buffer = bytearray()
        self.base = 0
        self.eof = False
        self.closed = False
        self.write_lock = threading.Lock()
        self.write_offset = 0
        self.write_closed = False
        self.close_lock = threading.Lock()
        self.close_complete = False
        self.reader = threading.Thread(target=self._receive, daemon=True)

    def _receive(self) -> None:
        try:
            while True:
                with self.condition:
                    while len(self.buffer) >= self.window_bytes and not self.closed:
                        self.condition.wait()
                    if self.closed:
                        return
                try:
                    chunk = self.socket.recv(_RECEIVE_CHUNK_BYTES)
                except TimeoutError:
                    continue
                with self.condition:
                    if not chunk:
                        self.eof = True
                        self.condition.notify_all()
                        return
                    self.buffer.extend(chunk)
                    self.condition.notify_all()
        except OSError:
            with self.condition:
                self.eof = True
                self.condition.notify_all()

    def read(self, offset: int, *, max_bytes: int, wait_s: float) -> dict[str, Any]:
        with self.condition:
            end = self.base + len(self.buffer)
            if not self.base <= offset <= end:
                raise CommandRequestError("forward read offset is outside the retained window")
            del self.buffer[: offset - self.base]
            self.base = offset
            self.condition.notify_all()
            deadline = time.monotonic() + wait_s
            while not self.buffer and not self.eof and not self.closed:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                self.condition.wait(remaining)
            data = bytes(self.buffer[:max_bytes])
            return {
                "offset": offset,
                "data_base64": base64.b64encode(data).decode("ascii"),
                "eof": self.closed or (self.eof and len(data) == len(self.buffer)),
            }

    def write(self, offset: int, data: bytes, *, eof: bool) -> dict[str, Any]:
        with self.write_lock:
            if offset > self.write_offset:
                raise CommandRequestError(
                    "forward write offset skips bytes that were never written"
                )
            pending = data[self.write_offset - offset :]
            if self.write_closed or self.closed:
                return {"offset": self.write_offset, "closed": True}
            try:
                if pending:
                    self.socket.sendall(pending)
                self.write_offset += len(pending)
                if eof:
                    self.write_closed = True
                    self.socket.shutdown(socket.SHUT_WR)
            except OSError:
                self.write_closed = True
                self.close()
            return {"offset": self.write_offset, "closed": self.write_closed}

    def close(self) -> None:
        with self.close_lock:
            if self.close_complete:
                return
            with self.condition:
                self.closed = True
                self.buffer.clear()
                self.condition.notify_all()
            with contextlib.suppress(OSError):
                self.socket.shutdown(socket.SHUT_RDWR)
            self.socket.close()
            self.close_complete = True


class _Forward:
    def __init__(
        self,
        forward_id: str,
        listener: socket.socket,
        *,
        owner_generation: int | None,
        expires_at: float | None,
        max_connections: int,
        window_bytes: int,
        on_expire: Callable[[str], object],
    ) -> None:
        self.forward_id = forward_id
        self.listener = listener
        self.host, self.port = listener.getsockname()[:2]
        self.owner_generation = owner_generation
        self.expires_at = expires_at
        self.max_connections = max_connections
        self.window_bytes = window_bytes
        self.on_expire = on_expire
        self.condition = threading.Condition()
        self.connections: dict[int, _Connection] = {}
        self.pending_connections: set[_Connection] = set()
        self.last_connection = 0
        self.closed = False
        self.thread = threading.Thread(
            target=self._accept, name=f"cayu-forward-{self.port}", daemon=True
        )

    def _accept(self) -> None:
        while True:
            with self.condition:
                if self.closed:
                    return
            if self.expires_at is not None and time.monotonic() >= self.expires_at:
                self.on_expire(self.forward_id)
                return
            try:
                client, _address = self.listener.accept()
            except TimeoutError:
                continue
            except OSError:
                return
            connection = _Connection(client, self.window_bytes)
            with self.condition:
                self.pending_connections.add(connection)
            try:
                client.settimeout(_SEND_TIMEOUT_SECONDS)
                with self.condition:
                    if not self.closed and len(self.connections) < self.max_connections:
                        self.last_connection += 1
                        self.connections[self.last_connection] = connection
                        self.pending_connections.remove(connection)
                        connection.reader.start()
                        self.condition.notify_all()
                        continue
                connection.close()
                with self.condition:
                    self.pending_connections.discard(connection)
            except OSError:
                # Retain even a rejected client's socket when setup/close
                # fails. Stop accepting so the retained set stays bounded;
                # immediately attempt the owning teardown, including healthy
                # connections, rather than abandoning lifetime cleanup.
                with self.condition:
                    self.closed = True
                    self.condition.notify_all()
                self.on_expire(self.forward_id)
                return

    def accept(self, after: int, *, wait_s: float) -> dict[str, Any]:
        with self.condition:
            deadline = time.monotonic() + wait_s
            while not self.closed and self.last_connection <= after:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                self.condition.wait(remaining)
            return {
                "connections": sorted(
                    identifier for identifier in self.connections if identifier > after
                ),
                "closed": self.closed,
            }

    def connection(self, identifier: int) -> _Connection:
        with self.condition:
            connection = self.connections.get(identifier)
        if connection is None:
            raise CommandNotFoundError("forward connection not found")
        return connection

    def close_connection(self, identifier: int) -> None:
        with self.condition:
            connection = self.connections.get(identifier)
        if connection is not None:
            connection.close()
            with self.condition:
                self.connections.pop(identifier, None)

    def close(self) -> None:
        # The registry's closing record serializes calls, including retries
        # after an incomplete teardown. ``closed`` fences I/O, not completion.
        with self.condition:
            self.closed = True
            self.condition.notify_all()
        failure: Exception | None = None
        # Attempt each independent resource even when another cannot close.
        # The last drain includes clients returned by accept during teardown.
        for close in (
            self.listener.close,
            self._close_connections,
            self._join_accept_thread,
            self._close_connections,
        ):
            try:
                close()
            except Exception as exc:
                failure = failure or exc
        if failure is not None:
            raise failure

    def _join_accept_thread(self) -> None:
        if threading.current_thread() is not self.thread:
            self.thread.join(timeout=2)
            if self.thread.is_alive():
                raise CommandConflictError("forward teardown is still running; retry close")

    def _close_connections(self) -> None:
        with self.condition:
            identifiers = tuple(self.connections)
            pending = tuple(self.pending_connections)
        failure: Exception | None = None
        for identifier in identifiers:
            try:
                self.close_connection(identifier)
            except Exception as exc:
                failure = failure or exc
        for connection in pending:
            try:
                connection.close()
                with self.condition:
                    self.pending_connections.discard(connection)
            except Exception as exc:
                failure = failure or exc
        if failure is not None:
            raise failure


class _ClosingForward:
    """One retained teardown, shared by individual and lifecycle closes."""

    def __init__(self, forward: _Forward) -> None:
        self.forward = forward
        self.condition = threading.Condition()
        self.running = False
        self.complete = False

    def close(self, *, wait: bool = True) -> bool:
        with self.condition:
            while self.running:
                if not wait:
                    return False
                self.condition.wait()
            if self.complete:
                return True
            self.running = True
        complete = False
        try:
            self.forward.close()
            complete = True
        finally:
            with self.condition:
                self.complete = complete
                self.running = False
                self.condition.notify_all()
        return True


class PortForwardRegistry:
    """Owner-scoped guest listeners relayed by the host over the sidecar endpoint.

    Each forward listens on the agent veth gateway, so only agent commands can
    connect. The host long-polls for accepted connections and moves each
    connection's bytes by offset, which keeps retries idempotent. A superseding
    owner claim, suspension, termination, or the forward's lifetime closes it.
    Closing an unknown forward records a short-lived tombstone, so an open that
    arrives after its own cleanup is refused instead of leaving a listener.
    """

    def __init__(
        self,
        *,
        listen_host: str = FORWARD_LISTEN_HOST,
        ports: range = FORWARD_PORTS,
        max_connections: int = MAX_FORWARD_CONNECTIONS,
        window_bytes: int = FORWARD_WINDOW_BYTES,
    ) -> None:
        self.listen_host = listen_host
        self.ports = ports
        self.max_connections = max_connections
        self.window_bytes = window_bytes
        self._lock = threading.Lock()
        self._forwards: dict[str, _Forward] = {}
        self._closing: dict[str, _ClosingForward] = {}
        self._tombstones: dict[str, float] = {}

    def open(
        self,
        forward_id: object,
        *,
        owner_generation: int | None = None,
        timeout_s: object = None,
    ) -> dict[str, Any]:
        identifier = _forward_id(forward_id)
        lifetime = _lifetime(timeout_s)
        with self._lock:
            self._prune_tombstones()
            if identifier in self._tombstones or identifier in self._closing:
                raise CommandConflictError("forward was already closed")
            existing = self._forwards.get(identifier)
            if existing is not None:
                if existing.owner_generation != owner_generation:
                    raise CommandConflictError("forward belongs to another owner")
                if existing.closed:
                    raise CommandConflictError("forward teardown is incomplete; retry close")
                return self._describe(existing)
            # Reserve the close fence at admission so expiry never depends on
            # spare tombstone capacity, even when the host has disappeared.
            if self._entry_count() >= _MAX_TOMBSTONES:
                raise CommandConflictError("forward closure fencing capacity exhausted")
            listener = self._bind_locked()
            forward = _Forward(
                identifier,
                listener,
                owner_generation=owner_generation,
                expires_at=None if lifetime is None else time.monotonic() + lifetime,
                max_connections=self.max_connections,
                window_bytes=self.window_bytes,
                on_expire=self._expire,
            )
            self._forwards[identifier] = forward
            forward.thread.start()
            return self._describe(forward)

    def accept(
        self, forward_id: object, *, after: object, wait_s: object, owner_generation: int | None
    ) -> dict[str, Any]:
        forward = self._forward(forward_id, owner_generation)
        return forward.accept(_non_negative_int(after, "after"), wait_s=_wait(wait_s))

    def read(
        self,
        forward_id: object,
        connection: object,
        *,
        offset: object,
        max_bytes: object,
        wait_s: object,
        owner_generation: int | None,
    ) -> dict[str, Any]:
        forward = self._forward(forward_id, owner_generation)
        if type(max_bytes) is not int or not 1 <= max_bytes <= MAX_FORWARD_CHUNK_BYTES:
            raise CommandRequestError("max_bytes must be an integer from 1 to 4 MiB")
        return forward.connection(_non_negative_int(connection, "connection")).read(
            _non_negative_int(offset, "offset"), max_bytes=max_bytes, wait_s=_wait(wait_s)
        )

    def write(
        self,
        forward_id: object,
        connection: object,
        *,
        offset: object,
        data_base64: object,
        eof: object,
        owner_generation: int | None,
    ) -> dict[str, Any]:
        forward = self._forward(forward_id, owner_generation)
        if type(data_base64) is not str or len(data_base64) > 4 * (
            (MAX_FORWARD_CHUNK_BYTES + 2) // 3
        ):
            raise CommandRequestError("data_base64 must be base64 of at most 4 MiB")
        try:
            data = base64.b64decode(data_base64, validate=True)
        except (binascii.Error, ValueError) as exc:
            raise CommandRequestError("data_base64 must be valid base64") from exc
        if type(eof) is not bool:
            raise CommandRequestError("eof must be a boolean")
        return forward.connection(_non_negative_int(connection, "connection")).write(
            _non_negative_int(offset, "offset"), data, eof=eof
        )

    def close_connection(
        self, forward_id: object, connection: object, *, owner_generation: int | None
    ) -> dict[str, Any]:
        identifier = _non_negative_int(connection, "connection")
        with contextlib.suppress(CommandNotFoundError):
            self._forward(forward_id, owner_generation, include_closing=True).close_connection(
                identifier
            )
        return {"connection": identifier, "closed": True}

    def close(self, forward_id: object) -> dict[str, Any]:
        identifier = _forward_id(forward_id)
        with self._lock:
            self._prune_tombstones()
            if (
                identifier not in self._forwards
                and identifier not in self._closing
                and identifier not in self._tombstones
                and self._entry_count() >= _MAX_TOMBSTONES
            ):
                raise CommandConflictError(
                    "forward closure fencing capacity exhausted; retry close"
                )
            self._tombstones[identifier] = time.monotonic() + _TOMBSTONE_TTL_SECONDS
            closing = self._retire(identifier)
        if closing is not None:
            self._finish_close(identifier, closing)
        return {"forward_id": identifier, "closed": True}

    def close_all(self) -> None:
        with self._lock:
            for identifier in tuple(self._forwards):
                self._retire(identifier)
            closing = tuple(self._closing.items())
        failure: Exception | None = None
        for identifier, record in closing:
            try:
                self._finish_close(identifier, record)
            except Exception as exc:
                failure = failure or exc
        if failure is not None:
            raise failure

    def _expire(self, identifier: str) -> None:
        # A different closer may be joining this accept thread. Expiry does
        # not acknowledge closure: leave that closer's record for all waiters
        # and return so its join can finish.
        with self._lock:
            if identifier in self._closing:
                return
            self._tombstones[identifier] = time.monotonic() + _TOMBSTONE_TTL_SECONDS
            closing = self._retire(identifier)
        if closing is not None:
            try:
                self._finish_close(identifier, closing, wait=False)
            except (OSError, CommandConflictError):
                # Internal expiry/fault teardown acknowledges nothing. Keep
                # uncertain resources fenced for an explicit positive close.
                return

    def _retire(self, identifier: str) -> _ClosingForward | None:
        """Move admission to the retained closing set while holding ``_lock``."""
        forward = self._forwards.pop(identifier, None)
        if forward is not None:
            self._closing[identifier] = _ClosingForward(forward)
        return self._closing.get(identifier)

    def _finish_close(
        self, identifier: str, closing: _ClosingForward, *, wait: bool = True
    ) -> None:
        if closing.close(wait=wait):
            with self._lock:
                if self._closing.get(identifier) is closing:
                    del self._closing[identifier]

    def _entry_count(self) -> int:
        # A closing forward may already have its reserved tombstone. Count
        # that identifier once, and retain its capacity until teardown ends.
        return len(self._forwards) + len(self._tombstones.keys() | self._closing.keys())

    def _forward(
        self, forward_id: object, owner_generation: int | None, *, include_closing: bool = False
    ) -> _Forward:
        identifier = _forward_id(forward_id)
        with self._lock:
            forward = self._forwards.get(identifier)
            if forward is None and include_closing:
                closing = self._closing.get(identifier)
                forward = None if closing is None else closing.forward
        if forward is None or forward.owner_generation != owner_generation:
            raise CommandNotFoundError("forward not found")
        return forward

    def _bind_locked(self) -> socket.socket:
        used = {forward.port for forward in self._forwards.values()} | {
            closing.forward.port for closing in self._closing.values()
        }
        for port in self.ports:
            if port in used:
                continue
            listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            try:
                listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                listener.bind((self.listen_host, port))
                listener.listen(self.max_connections)
            except OSError:
                listener.close()
                continue
            listener.settimeout(0.2)
            return listener
        raise CommandConflictError("no forward port is free")

    def _prune_tombstones(self) -> None:
        now = time.monotonic()
        for identifier, expires_at in tuple(self._tombstones.items()):
            if expires_at <= now:
                del self._tombstones[identifier]

    @staticmethod
    def _describe(forward: _Forward) -> dict[str, Any]:
        return {"forward_id": forward.forward_id, "host": forward.host, "port": forward.port}


def _forward_id(value: object) -> str:
    if (
        type(value) is not str
        or not value.strip()
        or len(value.encode("utf-8")) > 256
        or any(character in value for character in "/\\")
    ):
        raise CommandRequestError("forward_id is invalid")
    return value


def _lifetime(value: object) -> float | None:
    if value is None:
        return None
    if (
        isinstance(value, bool)
        or not isinstance(value, int | float)
        or not 0 < value <= MAX_FORWARD_LIFETIME_SECONDS
    ):
        raise CommandRequestError("forward timeout_s must be a positive number of seconds")
    return float(value)


def _wait(value: object) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, int | float)
        or not 0 <= value <= MAX_FORWARD_WAIT_SECONDS
    ):
        raise CommandRequestError("wait_s must be from 0 to 20 seconds")
    return float(value)


def _non_negative_int(value: object, field_name: str) -> int:
    if type(value) is not int or value < 0:
        raise CommandRequestError(f"{field_name} must be a non-negative integer")
    return value
