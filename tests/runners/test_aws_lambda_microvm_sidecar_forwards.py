"""Sidecar port forwards: guest listeners whose connections the owner relays."""

from __future__ import annotations

import asyncio
import base64
import importlib
import socket
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
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


@pytest.mark.parametrize("first_all", [False, True], ids=["close", "close_all"])
@pytest.mark.parametrize("second_all", [False, True], ids=["duplicate", "close_all"])
def test_concurrent_closes_wait_for_socket_teardown(monkeypatch, first_all, second_all):
    registry = _registry()
    opened = registry.open("fwd", owner_generation=1)
    forward = registry._forwards["fwd"]
    entered, release, second_started = threading.Event(), threading.Event(), threading.Event()
    original_close = forward.close

    def paused_close():
        entered.set()
        assert release.wait(5), "test did not release socket teardown"
        original_close()

    def close(all_forwards):
        return registry.close_all() if all_forwards else registry.close("fwd")

    def second_close():
        second_started.set()
        return close(second_all)

    monkeypatch.setattr(forward, "close", paused_close)
    with _connect(opened) as client, ThreadPoolExecutor(max_workers=2) as workers:
        [connection] = _accept(registry, "fwd")
        accepted = forward.connections[connection].socket
        first = workers.submit(close, first_all)
        try:
            assert entered.wait(5)
            second = workers.submit(second_close)
            assert second_started.wait(5)
            # The teardown owner is held at a known barrier, not merely slow.
            # Neither another close nor a lifecycle close_all may certify it.
            with pytest.raises(TimeoutError):
                second.result(timeout=0.1)
        finally:
            release.set()
            first.result(timeout=5)
            registry.close_all()
        second.result(timeout=5)
        assert client.recv(1) == b""
        assert accepted.fileno() == -1
        assert forward.listener.fileno() == -1
        with pytest.raises(OSError):
            _connect(opened)


def test_close_all_waits_for_expiry_teardown(monkeypatch):
    registry = _registry()
    opened = registry.open("fwd", owner_generation=1)
    forward = registry._forwards["fwd"]
    entered, release, second_started = threading.Event(), threading.Event(), threading.Event()
    original_close = forward.close

    def paused_close():
        entered.set()
        assert release.wait(5), "test did not release expiry teardown"
        original_close()

    def close_all():
        second_started.set()
        registry.close_all()

    monkeypatch.setattr(forward, "close", paused_close)
    with _connect(opened) as client, ThreadPoolExecutor(max_workers=1) as workers:
        _accept(registry, "fwd")
        forward.expires_at = time.monotonic() - 1
        try:
            assert entered.wait(5)
            closed = workers.submit(close_all)
            assert second_started.wait(5)
            with pytest.raises(TimeoutError):
                closed.result(timeout=0.1)
        finally:
            release.set()
            registry.close_all()
            forward.thread.join(timeout=5)
        closed.result(timeout=5)
        assert not forward.thread.is_alive()
        assert client.recv(1) == b""
        assert forward.listener.fileno() == -1


def test_expiry_yields_to_the_closer_joining_its_thread(monkeypatch):
    registry = _registry()
    opened = registry.open("fwd", owner_generation=1)
    forward = registry._forwards["fwd"]
    expiry_ready, release_expiry = threading.Event(), threading.Event()
    finish_close, forward_close = registry._finish_close, forward.close

    def paused_expiry(identifier, closing, *, wait=True):
        if threading.current_thread() is forward.thread:
            expiry_ready.set()
            assert release_expiry.wait(5)
        finish_close(identifier, closing, wait=wait)

    def other_close():
        # This call already owns teardown; expiry must yield before it joins
        # the accept thread, even though expiry created the closing record.
        release_expiry.set()
        forward_close()

    monkeypatch.setattr(registry, "_finish_close", paused_expiry)
    monkeypatch.setattr(forward, "close", other_close)
    with _connect(opened) as client, ThreadPoolExecutor(max_workers=1) as workers:
        _accept(registry, "fwd")
        forward.expires_at = time.monotonic() - 1
        try:
            assert expiry_ready.wait(5)
            assert workers.submit(registry.close, "fwd").result(timeout=5)["closed"] is True
        finally:
            release_expiry.set()
            registry.close_all()
            forward.thread.join(timeout=5)
        assert not forward.thread.is_alive()
        assert client.recv(1) == b""
        assert not registry._closing


class _SocketCloseFault:
    def __init__(self, original):
        self.original = original
        self.fail = True

    def __getattr__(self, name):
        return getattr(self.original, name)

    def close(self):
        if self.fail:
            raise OSError("injected socket close failure")
        self.original.close()


@pytest.mark.parametrize("rejection", ["closing", "capacity"])
def test_rejected_accepted_socket_is_retained_until_close_succeeds(monkeypatch, rejection):
    registry = _registry(max_connections=1)
    opened = registry.open("fwd", owner_generation=1)
    forward = registry._forwards["fwd"]
    listening, accepted, release, joining, attempted = (threading.Event() for _ in range(5))
    retained = []
    clients = []
    listener = forward.listener
    original_join = forward.thread.join

    class LateClient(_SocketCloseFault):
        def close(self):
            attempted.set()
            super().close()

    class PausedAccept:
        def __getattr__(self, name):
            return getattr(listener, name)

        def accept(self):
            listening.set()
            client, address = listener.accept()
            fault = LateClient(client)
            retained.append(fault)  # Prevent garbage collection from closing the socket.
            accepted.set()
            assert release.wait(5)
            return fault, address

    def join(*args, **kwargs):
        joining.set()
        return original_join(*args, **kwargs)

    try:
        if rejection == "capacity":
            clients.append(_connect(opened))
            _accept(registry, "fwd")
        monkeypatch.setattr(forward, "listener", PausedAccept())
        monkeypatch.setattr(forward.thread, "join", join)
        assert listening.wait(5)
        clients.append(_connect(opened))
        assert accepted.wait(5)
        with ThreadPoolExecutor(max_workers=1) as workers:
            if rejection == "closing":
                closed = workers.submit(registry.close, "fwd")
                try:
                    assert joining.wait(5)
                finally:
                    release.set()
                with pytest.raises(OSError, match="injected socket close failure"):
                    closed.result(timeout=5)
            else:
                release.set()
                assert attempted.wait(5)
                with pytest.raises(OSError, match="injected socket close failure"):
                    registry.close("fwd")
        assert retained[0].original.fileno() != -1
        assert "fwd" in registry._closing
        with pytest.raises(CommandConflictError):
            registry.open("fwd", owner_generation=1)
    finally:
        release.set()
        for fault in retained:
            fault.fail = False
        try:
            registry.close_all()
            original_join(timeout=5)
            assert all(fault.original.fileno() == -1 for fault in retained)
        finally:
            for fault in retained:
                fault.original.close()
            for client in clients:
                client.close()
    assert not forward.thread.is_alive()
    assert not registry._closing


@pytest.mark.parametrize("timeout_s", [None, 60], ids=["no-deadline", "deadline"])
@pytest.mark.parametrize("persistent", [False, True], ids=["transient", "persistent"])
def test_rejected_client_failure_still_tears_down_forward(monkeypatch, timeout_s, persistent):
    registry = _registry(max_connections=1)
    opened = registry.open("fwd", owner_generation=1, timeout_s=timeout_s)
    forward = registry._forwards["fwd"]
    listener = forward.listener
    listening = threading.Event()
    retained = []

    class RejectedClient(_SocketCloseFault):
        def close(self):
            fail = self.fail
            if not persistent:
                self.fail = False
            if fail:
                raise OSError("injected rejected-client close failure")
            super().close()

    class RejectingListener:
        def __getattr__(self, name):
            return getattr(listener, name)

        def accept(self):
            listening.set()
            client, address = listener.accept()
            fault = RejectedClient(client)
            retained.append(fault)
            return fault, address

    with _connect(opened) as healthy:
        [identifier] = _accept(registry, "fwd")
        accepted = forward.connections[identifier]
        monkeypatch.setattr(forward, "listener", RejectingListener())
        try:
            assert listening.wait(5)
            with _connect(opened):
                forward.thread.join(timeout=5)
                assert not forward.thread.is_alive()
                assert forward.closed  # Admission refusal alone is not confirmed teardown.
                assert listener.fileno() == -1
                assert accepted.socket.fileno() == -1
                assert healthy.recv(1) == b""
                with pytest.raises(OSError):
                    _connect(opened)
                assert bool(registry._closing) is persistent
                if persistent:
                    assert retained[0].original.fileno() != -1
                    with pytest.raises(OSError, match="rejected-client close failure"):
                        registry.close("fwd")
                    assert "fwd" in registry._closing
                else:
                    assert retained[0].original.fileno() == -1
        finally:
            for fault in retained:
                fault.fail = False
            registry.close_all()
            forward.thread.join(timeout=5)
    assert not registry._closing
    assert registry.close("fwd")["closed"] is True
    assert all(fault.original.fileno() == -1 for fault in retained)


@pytest.mark.parametrize("target", ["listener", "connection"])
def test_close_all_attempts_every_forward_and_connection_after_failure(monkeypatch, target):
    ports = (_free_port(), _free_port())
    assert ports[0] != ports[1]
    registry = PortForwardRegistry(listen_host="127.0.0.1", ports=ports)
    first = registry.open("first", owner_generation=1)
    second = registry.open("second", owner_generation=1)
    first_forward = registry._forwards["first"]
    second_forward = registry._forwards["second"]
    with _connect(first), _connect(first), _connect(second):
        identifiers = _accept(registry, "first")
        if len(identifiers) == 1:
            identifiers.extend(_accept(registry, "first", after=identifiers[0]))
        [other_id] = _accept(registry, "second")
        first_connection, healthy = (first_forward.connections[i] for i in identifiers)
        second_connection = second_forward.connections[other_id]
        owner, name = (
            (first_forward, "listener") if target == "listener" else (first_connection, "socket")
        )
        fault = _SocketCloseFault(getattr(owner, name))
        monkeypatch.setattr(owner, name, fault)
        try:
            with pytest.raises(OSError, match="injected socket close failure"):
                registry.close_all()
            assert set(registry._closing) == {"first"}
            assert healthy.socket.fileno() == -1
            assert second_connection.socket.fileno() == -1
            assert second_forward.listener.fileno() == -1
            assert not second_forward.thread.is_alive()
            assert fault.original.fileno() != -1
            if target == "connection":
                assert first_forward.listener.fileno() == -1
                with pytest.raises(OSError):
                    _connect(first)
            with pytest.raises(OSError):
                _connect(second)
        finally:
            fault.fail = False
            registry.close_all()
    assert not registry._closing


@pytest.mark.parametrize("target", ["listener", "connection"])
def test_socket_close_failure_remains_fenced_and_retryable(monkeypatch, target):
    monkeypatch.setattr("examples.aws.lambda_microvm_sidecar.forwards._MAX_TOMBSTONES", 1)
    registry = _registry()
    opened = registry.open("fwd", owner_generation=1)
    forward = registry._forwards["fwd"]
    with _connect(opened) as client:
        [identifier] = _accept(registry, "fwd")
        connection = forward.connections[identifier]
        socket_owner, socket_name = (
            (forward, "listener") if target == "listener" else (connection, "socket")
        )
        fault = _SocketCloseFault(getattr(socket_owner, socket_name))
        monkeypatch.setattr(socket_owner, socket_name, fault)
        try:
            for close in (lambda: registry.close("fwd"), registry.close_all):
                with pytest.raises(OSError, match="injected socket close failure"):
                    close()
                assert "fwd" in registry._closing
                # The tombstone and draining forward share one capacity slot.
                assert registry._entry_count() == 1
                with pytest.raises(CommandConflictError):
                    registry.open("fwd", owner_generation=1)
                with pytest.raises(CommandConflictError, match="capacity"):
                    registry.close("other")
        finally:
            fault.fail = False
            registry.close_all()
        assert not registry._closing
        assert registry.close("fwd")["closed"] is True
        assert client.recv(1) == b""
        assert forward.listener.fileno() == connection.socket.fileno() == -1


def test_join_timeout_never_certifies_teardown(monkeypatch):
    registry = _registry()
    opened = registry.open("fwd", owner_generation=1)
    forward = registry._forwards["fwd"]
    with _connect(opened) as client:
        _accept(registry, "fwd")
        try:
            with monkeypatch.context() as fault:
                fault.setattr(forward.thread, "join", lambda **_options: None)
                fault.setattr(forward.thread, "is_alive", lambda: True)
                with pytest.raises(CommandConflictError, match="teardown is still running"):
                    registry.close("fwd")
                assert "fwd" in registry._closing
                with pytest.raises(CommandConflictError, match="teardown is still running"):
                    registry.close_all()
        finally:
            registry.close_all()
        assert not registry._closing
        assert not forward.thread.is_alive()
        assert client.recv(1) == b""


@pytest.mark.parametrize("whole_forward", [False, True], ids=["connection", "forward"])
@pytest.mark.parametrize("first_whole_forward", [False, True], ids=["connection", "forward"])
def test_connection_teardown_is_shared_with_concurrent_closes(
    monkeypatch, whole_forward, first_whole_forward
):
    registry = _registry()
    opened = registry.open("fwd", owner_generation=1)
    forward = registry._forwards["fwd"]
    entered, release, second_started = threading.Event(), threading.Event(), threading.Event()
    with _connect(opened) as client, ThreadPoolExecutor(max_workers=2) as workers:
        [identifier] = _accept(registry, "fwd")
        connection = forward.connections[identifier]
        original = connection.socket

        class PausedSocket:
            def __getattr__(self, name):
                return getattr(original, name)

            def close(self):
                entered.set()
                assert release.wait(5)
                original.close()

        monkeypatch.setattr(connection, "socket", PausedSocket())

        def close_connection():
            return registry.close_connection("fwd", identifier, owner_generation=1)

        def second_close():
            second_started.set()
            return registry.close("fwd") if whole_forward else close_connection()

        first = (
            workers.submit(registry.close, "fwd")
            if first_whole_forward
            else workers.submit(close_connection)
        )
        try:
            assert entered.wait(5)
            second = workers.submit(second_close)
            assert second_started.wait(5)
            with pytest.raises(TimeoutError):
                second.result(timeout=0.1)
        finally:
            release.set()
            first.result(timeout=5)
            registry.close_all()
        second.result(timeout=5)
        assert client.recv(1) == b""
        assert original.fileno() == -1


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


@pytest.mark.parametrize("third_claim", [False, True], ids=["same_claim", "third_claim"])
def test_owner_route_cannot_acknowledge_an_incomplete_forward_close(
    tmp_path, monkeypatch, third_claim
):
    from fastapi import HTTPException

    monkeypatch.setenv("CAYU_MICROVM_WORKSPACE_ROOT", str(tmp_path))
    sys.modules.pop("examples.aws.lambda_microvm_sidecar.app", None)
    app = importlib.import_module("examples.aws.lambda_microvm_sidecar.app")
    registry = _registry()
    monkeypatch.setattr(app, "FORWARDS", registry)
    first_claim, pending_claim = "a" * 40, "b" * 40
    retry_claim = "c" * 40 if third_claim else pending_claim
    asyncio.run(app.claim_owner({"claim_id": first_claim}))
    opened = asyncio.run(app.open_forward({"forward_id": "fwd", "owner_claim": first_claim}))
    forward = registry._forwards["fwd"]
    with _connect(opened) as client:
        [identifier] = _accept(registry, "fwd")
        connection = forward.connections[identifier]
        fault = _SocketCloseFault(connection.socket)
        monkeypatch.setattr(connection, "socket", fault)
        replacements = []
        try:
            with pytest.raises(OSError, match="injected socket close failure"):
                asyncio.run(app.claim_owner({"claim_id": pending_claim}))
            assert forward.listener.fileno() == -1
            for claim in (first_claim, pending_claim):
                assert asyncio.run(app.check_owner({"claim_id": claim})) == {"current": False}
                with pytest.raises(HTTPException) as refused:
                    asyncio.run(app.open_forward({"forward_id": "late", "owner_claim": claim}))
                assert refused.value.status_code == 412
                with pytest.raises(HTTPException) as refused:
                    asyncio.run(app.acquire_lifecycle({"claim_id": claim, "action": "suspend"}))
                assert refused.value.status_code == 412

            # Even changing the callback supplied by the route cannot replace
            # the exact mandatory teardown retained by the failed takeover.
            monkeypatch.setattr(app, "_close_guest_relays", lambda: replacements.append(True))
            with pytest.raises(OSError, match="injected socket close failure"):
                asyncio.run(app.claim_owner({"claim_id": retry_claim}))
            assert replacements == []
            fault.fail = False
            generation = 3 if third_claim else 2
            assert asyncio.run(app.claim_owner({"claim_id": retry_claim})) == {
                "generation": generation,
                "superseded_previous": True,
            }
            assert app.OWNER.require(retry_claim) == generation
            assert len(replacements) == int(third_claim)
            assert not registry._closing
            assert connection.socket.fileno() == -1
            assert client.recv(1) == b""
        finally:
            fault.fail = False
            registry.close_all()
            app.SUPERVISOR.cancel_all()
            sys.modules.pop("examples.aws.lambda_microvm_sidecar.app", None)


@pytest.mark.parametrize("lease_action", ["suspend", "terminate"])
@pytest.mark.parametrize("hook_action", ["suspend", "terminate"])
def test_failed_forward_teardown_still_settles_matching_lifecycle_hook(
    tmp_path, monkeypatch, lease_action, hook_action
):
    from fastapi import HTTPException

    monkeypatch.setenv("CAYU_MICROVM_WORKSPACE_ROOT", str(tmp_path))
    sys.modules.pop("examples.aws.lambda_microvm_sidecar.app", None)
    app = importlib.import_module("examples.aws.lambda_microvm_sidecar.app")
    registry = _registry()
    monkeypatch.setattr(app, "FORWARDS", registry)
    claim, successor = "a" * 40, "b" * 40
    asyncio.run(app.claim_owner({"claim_id": claim}))
    opened = asyncio.run(app.open_forward({"forward_id": "fwd", "owner_claim": claim}))
    forward = registry._forwards["fwd"]
    with _connect(opened):
        [identifier] = _accept(registry, "fwd")
        connection = forward.connections[identifier]
        fault = _SocketCloseFault(connection.socket)
        monkeypatch.setattr(connection, "socket", fault)
        try:
            asyncio.run(app.acquire_lifecycle({"claim_id": claim, "action": lease_action}))
            with pytest.raises(OSError, match="injected socket close failure"):
                asyncio.run(getattr(app, f"{hook_action}_hook")())
            assert forward.listener.fileno() == -1
            assert fault.original.fileno() != -1
            assert "fwd" in registry._closing
            settles = hook_action == "terminate" or lease_action == "suspend"
            if settles:
                # The provider applied this lifecycle operation, but a new
                # owner still cannot bypass the separately retained teardown.
                with pytest.raises(OSError, match="injected socket close failure"):
                    asyncio.run(app.claim_owner({"claim_id": successor}))
                assert not app.OWNER.is_current(claim)
                assert not app.OWNER.is_current(successor)
            else:
                with pytest.raises(HTTPException) as refused:
                    asyncio.run(app.claim_owner({"claim_id": successor}))
                assert refused.value.status_code == app.OWNER_LIFECYCLE_LEASED_STATUS
            fault.fail = False
            if not settles:
                assert asyncio.run(app.terminate_hook()) == {"status": "ok"}
            assert asyncio.run(app.claim_owner({"claim_id": successor})) == {
                "generation": 2,
                "superseded_previous": True,
            }
            assert not registry._closing
            assert fault.original.fileno() == -1
        finally:
            fault.fail = False
            registry.close_all()
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


def test_sidecar_writes_never_queue_behind_long_polls(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from concurrent.futures import ThreadPoolExecutor

    from examples.aws.lambda_microvm_sidecar.forwards import (
        FORWARD_PORTS,
        MAX_FORWARD_CONNECTIONS,
    )

    monkeypatch.setenv("CAYU_MICROVM_WORKSPACE_ROOT", str(tmp_path))
    sys.modules.pop("examples.aws.lambda_microvm_sidecar.app", None)
    app = importlib.import_module("examples.aws.lambda_microvm_sidecar.app")
    # Every long poll the forward limits allow fits in the poll pool at once.
    assert app.FORWARD_POLL_IO._max_workers == len(FORWARD_PORTS) * (MAX_FORWARD_CONNECTIONS + 1)
    assert app.FORWARD_IO._max_workers == len(FORWARD_PORTS) * MAX_FORWARD_CONNECTIONS
    # Small pools make saturation cheap: the polls below fill the poll pool.
    polls, writes = ThreadPoolExecutor(max_workers=2), ThreadPoolExecutor(max_workers=2)
    monkeypatch.setattr(app, "FORWARD_POLL_IO", polls)
    monkeypatch.setattr(app, "FORWARD_IO", writes)
    monkeypatch.setattr(app, "SUPERVISOR", CommandSupervisor(root=tmp_path))
    monkeypatch.setattr(app, "FORWARDS", _registry())
    claim = "c" * 40

    async def scenario() -> None:
        await app.claim_owner({"claim_id": claim})
        opened = await app.open_forward({"forward_id": "fwd", "owner_claim": claim})
        client = await asyncio.to_thread(_connect, opened)
        try:
            await app.accept_forward("fwd", {"owner_claim": claim, "after": 0, "wait_s": 5})
            long_polls = [
                asyncio.create_task(
                    app.read_forward(
                        "fwd", 1, {"owner_claim": claim, "offset": 0, "max_bytes": 16, "wait_s": 5}
                    )
                )
                for _ in range(2)
            ]
            await asyncio.sleep(0.2)
            written = await asyncio.wait_for(
                app.write_forward(
                    "fwd", 1, {"owner_claim": claim, "offset": 0, "data_base64": "b2s="}
                ),
                2,
            )
            assert written == {"offset": 2, "closed": False}
            assert await asyncio.to_thread(client.recv, 16) == b"ok"
            app.FORWARDS.close_all()
            await asyncio.gather(*long_polls)
        finally:
            client.close()

    try:
        asyncio.run(scenario())
    finally:
        app.FORWARDS.close_all()
        app.SUPERVISOR.cancel_all()
        polls.shutdown(wait=True)
        writes.shutdown(wait=True)
        sys.modules.pop("examples.aws.lambda_microvm_sidecar.app", None)
