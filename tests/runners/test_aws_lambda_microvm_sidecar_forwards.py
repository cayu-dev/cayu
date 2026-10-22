"""Sidecar port forwards: guest listeners whose connections the owner relays."""

from __future__ import annotations

import asyncio
import base64
import importlib
import socket
import sys
import threading
import time
from pathlib import Path

import pytest
from examples.aws.lambda_microvm_sidecar.forwards import PortForwardRegistry
from examples.aws.lambda_microvm_sidecar.supervisor import (
    CommandConflictError,
    CommandNotFoundError,
    CommandRequestError,
    CommandSupervisor,
)


def _free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


def _registry(**options) -> PortForwardRegistry:
    port = _free_port()
    return PortForwardRegistry(listen_host="127.0.0.1", ports=range(port, port + 1), **options)


def _connect(opened: dict) -> socket.socket:
    client = socket.create_connection((opened["host"], opened["port"]), timeout=5)
    client.settimeout(5)
    return client


def _accept(registry: PortForwardRegistry, forward_id: str, after: int = 0) -> list[int]:
    return registry.accept(forward_id, after=after, wait_s=5, owner_generation=1)["connections"]


def _read(registry, forward_id, connection, offset, wait_s=5) -> tuple[bytes, bool]:
    response = registry.read(
        forward_id,
        connection,
        offset=offset,
        max_bytes=1024,
        wait_s=wait_s,
        owner_generation=1,
    )
    assert response["offset"] == offset
    return base64.b64decode(response["data_base64"]), response["eof"]


def _write(registry, forward_id, connection, offset, data, eof=False) -> dict:
    return registry.write(
        forward_id,
        connection,
        offset=offset,
        data_base64=base64.b64encode(data).decode("ascii"),
        eof=eof,
        owner_generation=1,
    )


def test_connection_bytes_move_by_offset_in_both_directions() -> None:
    registry = _registry()
    try:
        opened = registry.open("fwd", owner_generation=1)
        assert opened == {"forward_id": "fwd", "host": "127.0.0.1", "port": opened["port"]}
        # Reopening with the same owner is idempotent.
        assert registry.open("fwd", owner_generation=1) == opened

        client = _connect(opened)
        client.sendall(b"request")
        [connection] = _accept(registry, "fwd")
        assert _read(registry, "fwd", connection, 0) == (b"request", False)
        # Reading the same offset again recovers a lost response.
        assert _read(registry, "fwd", connection, 0) == (b"request", False)

        assert _write(registry, "fwd", connection, 0, b"resp") == {"offset": 4, "closed": False}
        # A replayed write is skipped; only the new suffix reaches the guest.
        assert _write(registry, "fwd", connection, 0, b"response", eof=True) == {
            "offset": 8,
            "closed": True,
        }
        received = b""
        while chunk := client.recv(1024):
            received += chunk
        assert received == b"response"

        client.close()
        assert _read(registry, "fwd", connection, 7) == (b"", True)
        with pytest.raises(CommandRequestError):
            _write(registry, "fwd", connection, 20, b"gap")
        registry.close_connection("fwd", connection, owner_generation=1)
        with pytest.raises(CommandNotFoundError):
            _read(registry, "fwd", connection, 7)
    finally:
        registry.close_all()


def test_unread_guest_bytes_apply_backpressure() -> None:
    registry = _registry(window_bytes=1024)
    try:
        opened = registry.open("fwd", owner_generation=1)
        client = _connect(opened)
        payload = b"x" * (1024 * 1024)
        sender = threading.Thread(target=client.sendall, args=(payload,))
        sender.start()
        [connection] = _accept(registry, "fwd")
        time.sleep(0.3)
        held = registry._forwards["fwd"].connections[connection].buffer
        # The reader stops draining the socket once the window is full.
        assert 1024 <= len(held) <= 1024 + 64 * 1024
        received = b""
        while len(received) < len(payload):
            response = registry.read(
                "fwd",
                connection,
                offset=len(received),
                max_bytes=64 * 1024,
                wait_s=5,
                owner_generation=1,
            )
            received += base64.b64decode(response["data_base64"])
        sender.join(5)
        assert received == payload
        client.close()
    finally:
        registry.close_all()


def test_close_ends_connections_and_refuses_a_late_open() -> None:
    registry = _registry()
    opened = registry.open("fwd", owner_generation=1)
    client = _connect(opened)
    [connection] = _accept(registry, "fwd")

    assert registry.close("fwd") == {"forward_id": "fwd", "closed": True}
    assert client.recv(1024) == b""
    with pytest.raises(OSError):
        _connect(opened).recv(1)
    with pytest.raises(CommandNotFoundError):
        _read(registry, "fwd", connection, 0)

    # A close that arrived before its open leaves a tombstone.
    assert registry.close("never-opened")["closed"] is True
    with pytest.raises(CommandConflictError):
        registry.open("never-opened", owner_generation=1)


def test_forwards_belong_to_one_owner_generation() -> None:
    registry = _registry()
    try:
        registry.open("fwd", owner_generation=1)
        with pytest.raises(CommandConflictError):
            registry.open("fwd", owner_generation=2)
        with pytest.raises(CommandNotFoundError):
            registry.accept("fwd", after=0, wait_s=0, owner_generation=2)
        # The only port in the range is taken.
        with pytest.raises(CommandConflictError, match="no forward port"):
            registry.open("other", owner_generation=1)
    finally:
        registry.close_all()


def test_forward_closes_itself_when_its_lifetime_ends() -> None:
    registry = _registry()
    opened = registry.open("fwd", owner_generation=1, timeout_s=0.3)
    deadline = time.monotonic() + 5
    while True:
        try:
            registry.accept("fwd", after=0, wait_s=0, owner_generation=1)
        except CommandNotFoundError:
            break
        assert time.monotonic() < deadline
        time.sleep(0.05)
    with pytest.raises(OSError):
        _connect(opened).recv(1)


def test_connections_beyond_the_limit_are_refused() -> None:
    registry = _registry(max_connections=1)
    try:
        opened = registry.open("fwd", owner_generation=1)
        first = _connect(opened)
        assert _accept(registry, "fwd") == [1]
        second = _connect(opened)
        assert second.recv(1) == b""
        assert registry.accept("fwd", after=1, wait_s=0.3, owner_generation=1)["connections"] == []
        first.close()
    finally:
        registry.close_all()


@pytest.mark.parametrize(
    ("call", "error"),
    [
        (lambda r: r.open("a/b", owner_generation=1), "forward_id"),
        (lambda r: r.open("fwd", owner_generation=1, timeout_s=0), "timeout_s"),
        (lambda r: r.accept("fwd", after=-1, wait_s=0, owner_generation=1), "after"),
        (lambda r: r.accept("fwd", after=0, wait_s=30, owner_generation=1), "wait_s"),
        (
            lambda r: r.write("fwd", 1, offset=0, data_base64="!", eof=False, owner_generation=1),
            "base64",
        ),
    ],
)
def test_malformed_forward_requests_are_refused(call, error) -> None:
    registry = _registry()
    try:
        registry.open("fwd", owner_generation=1)
        with pytest.raises(CommandRequestError, match=error):
            call(registry)
    finally:
        registry.close_all()


def test_sidecar_routes_fence_forwards_to_the_current_owner(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from fastapi import HTTPException

    monkeypatch.setenv("CAYU_MICROVM_WORKSPACE_ROOT", str(tmp_path))
    sys.modules.pop("examples.aws.lambda_microvm_sidecar.app", None)
    app = importlib.import_module("examples.aws.lambda_microvm_sidecar.app")
    monkeypatch.setattr(app, "SUPERVISOR", CommandSupervisor(root=tmp_path))
    monkeypatch.setattr(app, "FORWARDS", _registry())
    claim, successor = "c" * 40, "d" * 40
    try:
        assert asyncio.run(app.health())["protocol_version"] == "7"
        asyncio.run(app.claim_owner({"claim_id": claim}))
        opened = asyncio.run(app.open_forward({"forward_id": "fwd", "owner_claim": claim}))
        client = _connect(opened)
        client.sendall(b"hi")
        accepted = asyncio.run(
            app.accept_forward("fwd", {"owner_claim": claim, "after": 0, "wait_s": 5})
        )
        assert accepted == {"connections": [1], "closed": False}
        read = asyncio.run(
            app.read_forward(
                "fwd", 1, {"owner_claim": claim, "offset": 0, "max_bytes": 16, "wait_s": 5}
            )
        )
        assert base64.b64decode(read["data_base64"]) == b"hi"
        written = asyncio.run(
            app.write_forward(
                "fwd", 1, {"owner_claim": claim, "offset": 0, "data_base64": "b2s=", "eof": True}
            )
        )
        assert written == {"offset": 2, "closed": True}
        assert client.recv(16) == b"ok"

        with pytest.raises(HTTPException) as stale:
            asyncio.run(app.accept_forward("fwd", {"owner_claim": successor, "after": 0}))
        assert stale.value.status_code == 412
        with pytest.raises(HTTPException) as missing:
            asyncio.run(app.accept_forward("other", {"owner_claim": claim, "after": 0}))
        assert missing.value.status_code == 404

        # A superseding claim closes every forward of the earlier owner.
        asyncio.run(app.claim_owner({"claim_id": successor}))
        assert client.recv(16) == b""
        with pytest.raises(HTTPException) as gone:
            asyncio.run(app.accept_forward("fwd", {"owner_claim": successor, "after": 0}))
        assert gone.value.status_code == 404

        asyncio.run(app.open_forward({"forward_id": "next", "owner_claim": successor}))
        asyncio.run(app.suspend_hook())
        with pytest.raises(HTTPException) as suspended:
            asyncio.run(app.accept_forward("next", {"owner_claim": successor, "after": 0}))
        assert suspended.value.status_code == 404
        closed = asyncio.run(app.close_forward("next", {"owner_claim": successor}))
        assert closed == {"forward_id": "next", "closed": True}
    finally:
        app.FORWARDS.close_all()
        app.SUPERVISOR.cancel_all()
        sys.modules.pop("examples.aws.lambda_microvm_sidecar.app", None)


def test_idle_receive_timeout_keeps_connection_open(monkeypatch) -> None:
    monkeypatch.setattr("examples.aws.lambda_microvm_sidecar.forwards._SEND_TIMEOUT_SECONDS", 0.02)
    registry = _registry()
    try:
        opened = registry.open("idle", owner_generation=1)
        with _connect(opened) as client:
            [connection] = _accept(registry, "idle")
            # A long-poll outlasting several socket timeouts must not report EOF.
            assert _read(registry, "idle", connection, 0, wait_s=0.1) == (b"", False)
            client.sendall(b"still here")
            assert _read(registry, "idle", connection, 0) == (b"still here", False)
    finally:
        registry.close_all()


def test_successful_close_fences_late_open_retry() -> None:
    registry = _registry()
    try:
        registry.open("retry", owner_generation=1)
        assert registry.close("retry")["closed"] is True
        with pytest.raises(CommandConflictError, match="already closed"):
            registry.open("retry", owner_generation=1)
        assert registry._forwards == {}
    finally:
        registry.close_all()


def test_expiry_uses_reserved_fence_at_full_capacity(monkeypatch) -> None:
    monkeypatch.setattr("examples.aws.lambda_microvm_sidecar.forwards._MAX_TOMBSTONES", 2)
    registry = _registry()
    try:
        registry.close("earlier")
        opened = registry.open("live", owner_generation=1, timeout_s=0.1)
        forward = registry._forwards["live"]
        with _connect(opened) as client:
            _accept(registry, "live")
            with pytest.raises(CommandConflictError, match="fencing capacity"):
                registry.close("other")
            with pytest.raises(CommandConflictError, match="fencing capacity"):
                registry.open("another", owner_generation=1)
            forward.thread.join(timeout=3)
            assert not forward.thread.is_alive()
            assert forward.closed
            assert client.recv(1) == b""
        assert registry._forwards == {}
        assert len(registry._tombstones) == 2
        assert registry.close("live")["closed"] is True
        with pytest.raises(CommandConflictError, match="already closed"):
            registry.open("live", owner_generation=1)
    finally:
        registry.close_all()


def test_partial_write_timeout_closes_both_connection_directions() -> None:
    from examples.aws.lambda_microvm_sidecar.forwards import _Connection

    server, guest = socket.socketpair()
    guest.settimeout(2)

    class TimedOutWriter:
        def sendall(self, data):
            server.sendall(data[:2])
            raise TimeoutError("guest stopped reading")

        def __getattr__(self, name):
            return getattr(server, name)

    connection = _Connection(TimedOutWriter(), 1024)
    connection.reader.start()
    try:
        assert connection.write(0, b"payload", eof=False) == {"offset": 0, "closed": True}
        assert guest.recv(10) == b"pa"
        assert guest.recv(10) == b""
        assert connection.read(0, max_bytes=1024, wait_s=0)["eof"] is True
        connection.reader.join(timeout=2)
        assert not connection.reader.is_alive()
    finally:
        connection.close()
        guest.close()
