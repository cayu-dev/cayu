"""LambdaMicroVMRunner port forwards over sidecar protocol 7 (fake AWS, real sidecar)."""

from __future__ import annotations

import asyncio
import importlib
import json
import socket
import statistics
import sys
import time
from pathlib import Path
from typing import Any

import httpx
import pytest
from examples.aws.lambda_microvm_sidecar.forwards import PortForwardRegistry
from examples.aws.lambda_microvm_sidecar.supervisor import (
    CommandConflictError,
    CommandNotFoundError,
    CommandRequestError,
    CommandSupervisor,
)
from tests.runners.lambda_microvm_harness import ConformanceLambdaClient
from tests.runners.test_aws_lambda_microvm_interactive import InteractiveTransport

from cayu.runners import (
    HttpxLambdaMicroVMEndpointTransport,
    LambdaMicroVMOwnershipSuperseded,
    LambdaMicroVMRunner,
    RunnerPortForwardCapability,
)
from cayu.runners.aws_lambda_microvm import (
    LambdaMicroVMEndpointTransientError,
    LambdaMicroVMError,
)


class ForwardingTransport(InteractiveTransport):
    """The real sidecar forward registry behind the sidecar's owner fence."""

    def __init__(self, root: Path, *, isolated: bool = True) -> None:
        super().__init__(root, isolated=isolated)
        self.forwards = PortForwardRegistry(listen_host="127.0.0.1", ports=range(0, 1))
        self.fail_next_forward_write = 0
        self.fail_next_open = 0
        self.forward_writes: list[int] = []

    def on_owner_superseded(self, generation: int) -> None:
        super().on_owner_superseded(generation)
        self.forwards.close_all()

    async def _forward(self, owner_claim: str, operation, *arguments, **keywords) -> Any:
        _, generation = self.admit_start_with_generation({"owner_claim": owner_claim})
        try:
            return await asyncio.to_thread(
                operation, *arguments, owner_generation=generation, **keywords
            )
        except (CommandNotFoundError, CommandRequestError, CommandConflictError) as error:
            raise LambdaMicroVMError(f"fake sidecar refused the forward call: {error}") from None

    async def open_forward(self, *, forward_id, owner_claim, lifetime_s, **_) -> dict[str, Any]:
        response = await self._forward(
            owner_claim, self.forwards.open, forward_id, timeout_s=lifetime_s
        )
        if self.fail_next_open:
            # The sidecar opened the listener but the response was lost.
            self.fail_next_open -= 1
            raise LambdaMicroVMEndpointTransientError("lost response")
        return response

    async def accept_forward(self, *, forward_id, owner_claim, after, wait_s, **_):
        return await self._forward(
            owner_claim, self.forwards.accept, forward_id, after=after, wait_s=wait_s
        )

    async def read_forward(
        self, *, forward_id, connection, owner_claim, offset, max_bytes, wait_s, **_
    ):
        return await self._forward(
            owner_claim,
            self.forwards.read,
            forward_id,
            connection,
            offset=offset,
            max_bytes=max_bytes,
            wait_s=wait_s,
        )

    async def write_forward(
        self, *, forward_id, connection, owner_claim, offset, data_base64, eof, **_
    ):
        self.forward_writes.append(offset)
        response = await self._forward(
            owner_claim,
            self.forwards.write,
            forward_id,
            connection,
            offset=offset,
            data_base64=data_base64,
            eof=eof,
        )
        if self.fail_next_forward_write:
            self.fail_next_forward_write -= 1
            raise LambdaMicroVMEndpointTransientError("lost response")
        return response

    async def close_forward_connection(self, *, forward_id, connection, owner_claim, **_):
        return await self._forward(
            owner_claim, self.forwards.close_connection, forward_id, connection
        )

    async def close_forward(self, *, forward_id, owner_claim, **_) -> dict[str, Any]:
        self.admit_start({"owner_claim": owner_claim})
        return await asyncio.to_thread(self.forwards.close, forward_id)


def _runner(root: Path, transport: Any) -> LambdaMicroVMRunner:
    return LambdaMicroVMRunner(
        ConformanceLambdaClient(),
        microvm_id="mvm-forward",
        endpoint="forward.lambda-microvm.invalid",
        region_name="us-east-1",
        default_cwd=str(root),
        endpoint_transport=transport,
        poll_interval_s=0,
    )


async def _upper_server() -> asyncio.Server:
    """A host service that answers each line upper-cased and closes after end of input."""

    async def serve(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        while line := await reader.readline():
            writer.write(line.upper())
            await writer.drain()
        writer.close()

    return await asyncio.start_server(serve, "127.0.0.1", 0)


def _guest_exchange(host: str, port: int, request: bytes) -> bytes:
    with socket.create_connection((host, port), timeout=10) as client:
        client.settimeout(10)
        client.sendall(request)
        client.shutdown(socket.SHUT_WR)
        received = b""
        while chunk := client.recv(65536):
            received += chunk
        return received


def test_forward_relays_guest_connections_to_the_host_service(tmp_path: Path) -> None:
    async def scenario() -> None:
        transport = ForwardingTransport(tmp_path)
        runner = _runner(tmp_path, transport)
        capability = runner.port_forward_capability()
        assert isinstance(capability, RunnerPortForwardCapability)
        server = await _upper_server()
        target_port = server.sockets[0].getsockname()[1]
        async with server:
            forward = await capability.open_port_forward(
                host="127.0.0.1", port=target_port, timeout_s=60
            )
            assert forward.guest_url == f"http://127.0.0.1:{forward.guest_port}"
            large = (b"x" * 1023 + b"\n") * 3072
            first, second = await asyncio.gather(
                asyncio.to_thread(
                    _guest_exchange, forward.guest_host, forward.guest_port, b"one\ntwo\n"
                ),
                asyncio.to_thread(_guest_exchange, forward.guest_host, forward.guest_port, large),
            )
            assert first == b"ONE\nTWO\n"
            assert second == large.upper()

            await forward.close()
            await forward.close()
            with pytest.raises(OSError):
                await asyncio.to_thread(
                    _guest_exchange, forward.guest_host, forward.guest_port, b"late\n"
                )
            assert transport.forwards._forwards == {}

    asyncio.run(scenario())


def test_lost_write_response_is_retried_without_duplicating_bytes(tmp_path: Path) -> None:
    async def scenario() -> None:
        transport = ForwardingTransport(tmp_path)
        server = await _upper_server()
        async with server:
            forward = (
                await _runner(tmp_path, transport)
                .port_forward_capability()
                .open_port_forward(host="127.0.0.1", port=server.sockets[0].getsockname()[1])
            )
            transport.fail_next_forward_write = 1
            reply = await asyncio.to_thread(
                _guest_exchange, forward.guest_host, forward.guest_port, b"once\n"
            )
            assert reply == b"ONCE\n"
            assert transport.forward_writes[:2] == [0, 0]
            await forward.close()

    asyncio.run(scenario())


def test_a_lost_open_response_leaves_no_guest_listener(tmp_path: Path) -> None:
    async def scenario() -> None:
        transport = ForwardingTransport(tmp_path)
        transport.fail_next_open = 3
        capability = _runner(tmp_path, transport).port_forward_capability()
        with pytest.raises(LambdaMicroVMEndpointTransientError):
            await capability.open_port_forward(host="127.0.0.1", port=1)
        assert transport.forwards._forwards == {}

    asyncio.run(scenario())


def test_a_successor_owner_closes_the_forward(tmp_path: Path) -> None:
    async def scenario() -> None:
        transport = ForwardingTransport(tmp_path)
        runner = _runner(tmp_path, transport)
        server = await _upper_server()
        async with server:
            forward = await runner.port_forward_capability().open_port_forward(
                host="127.0.0.1", port=server.sockets[0].getsockname()[1]
            )
            await transport.claim_owner(claim_id="s" * 40)
            with pytest.raises(OSError):
                await asyncio.to_thread(
                    _guest_exchange, forward.guest_host, forward.guest_port, b"x\n"
                )
            with pytest.raises(LambdaMicroVMOwnershipSuperseded):
                await forward.close()
            assert runner.owner_fence_generation is None

    asyncio.run(scenario())


def test_runner_without_forward_transport_reports_no_capability(tmp_path: Path) -> None:
    assert _runner(tmp_path, InteractiveTransport(tmp_path)).port_forward_capability() is None


@pytest.mark.parametrize(
    ("options", "error"),
    [
        ({"host": "", "port": 80}, ValueError),
        ({"host": "127.0.0.1", "port": 0}, ValueError),
        ({"host": "127.0.0.1", "port": 80, "timeout_s": 0}, ValueError),
    ],
)
def test_open_refuses_malformed_targets(tmp_path: Path, options, error) -> None:
    capability = _runner(tmp_path, ForwardingTransport(tmp_path)).port_forward_capability()
    with pytest.raises(error):
        asyncio.run(capability.open_port_forward(**options))


class _RoundTripTransport(httpx.AsyncBaseTransport):
    """The real sidecar app in process, with a fixed network round trip per request."""

    def __init__(self, app: Any, round_trip_s: float) -> None:
        self.inner = httpx.ASGITransport(app=app)
        self.half = round_trip_s / 2

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        await asyncio.sleep(self.half)
        await request.aread()
        response = await self.inner.handle_async_request(request)
        await response.aread()
        await asyncio.sleep(self.half)
        return response


async def _sse_upstream(events: int, interval_s: float) -> asyncio.Server:
    """A fake model upstream that streams timestamped server-sent events."""

    async def serve(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        await reader.readuntil(b"\r\n\r\n")
        writer.write(
            b"HTTP/1.1 200 OK\r\ncontent-type: text/event-stream\r\nconnection: close\r\n\r\n"
        )
        await writer.drain()
        for index in range(events):
            payload = json.dumps({"index": index, "sent": time.perf_counter()})
            writer.write(f"event: content_block_delta\ndata: {payload}\n\n".encode())
            await writer.drain()
            await asyncio.sleep(interval_s)
        writer.close()

    return await asyncio.start_server(serve, "127.0.0.1", 0)


def _guest_sse_latencies(host: str, port: int) -> list[float]:
    latencies = []
    with socket.create_connection((host, port), timeout=30) as client:
        client.settimeout(30)
        client.sendall(b"POST /v1/messages HTTP/1.1\r\nhost: gateway\r\ncontent-length: 0\r\n\r\n")
        buffer = b""
        while chunk := client.recv(65536):
            buffer += chunk
            while b"\n\n" in buffer:
                event, buffer = buffer.split(b"\n\n", 1)
                for line in event.split(b"\n"):
                    if line.startswith(b"data: "):
                        sent = json.loads(line[6:])["sent"]
                        latencies.append(time.perf_counter() - sent)
    return latencies


@pytest.mark.parametrize("round_trip_s", [0.0, 0.05])
def test_streamed_events_cross_the_relay_promptly(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, round_trip_s: float
) -> None:
    monkeypatch.setenv("CAYU_MICROVM_WORKSPACE_ROOT", str(tmp_path))
    sys.modules.pop("examples.aws.lambda_microvm_sidecar.app", None)
    app = importlib.import_module("examples.aws.lambda_microvm_sidecar.app")
    monkeypatch.setattr(app, "SUPERVISOR", CommandSupervisor(root=tmp_path))
    monkeypatch.setattr(
        app, "FORWARDS", PortForwardRegistry(listen_host="127.0.0.1", ports=range(0, 1))
    )
    events, interval_s = 40, 0.05

    async def scenario() -> list[float]:
        transport = HttpxLambdaMicroVMEndpointTransport()
        transport._client._client = httpx.AsyncClient(
            transport=_RoundTripTransport(app.app, round_trip_s)
        )
        runner = LambdaMicroVMRunner(
            ConformanceLambdaClient(),
            microvm_id="mvm-stream",
            endpoint="stream.lambda-microvm.invalid",
            region_name="us-east-1",
            default_cwd=str(tmp_path),
            endpoint_transport=transport,
            poll_interval_s=0,
        )
        upstream = await _sse_upstream(events, interval_s)
        try:
            async with upstream:
                forward = await runner.port_forward_capability().open_port_forward(
                    host="127.0.0.1", port=upstream.sockets[0].getsockname()[1], timeout_s=120
                )
                try:
                    return await asyncio.to_thread(
                        _guest_sse_latencies, forward.guest_host, forward.guest_port
                    )
                finally:
                    await forward.close()
        finally:
            await transport.aclose()

    try:
        latencies = asyncio.run(scenario())
    finally:
        app.FORWARDS.close_all()
        sys.modules.pop("examples.aws.lambda_microvm_sidecar.app", None)
    assert len(latencies) == events
    median = statistics.median(latencies)
    worst = max(latencies)
    print(
        f"relay SSE latency with {round_trip_s * 1000:.0f} ms round trip: "
        f"median {median * 1000:.1f} ms, p90 "
        f"{statistics.quantiles(latencies, n=10)[-1] * 1000:.1f} ms, max {worst * 1000:.1f} ms"
    )
    # One event waits for at most the response half of a long poll plus the
    # write that follows it; the bound leaves room for a loaded CI host.
    assert median < round_trip_s + 0.25
    assert worst < 2 * round_trip_s + 1.0


def test_failed_open_retains_cleanup_until_endpoint_recovers(tmp_path: Path) -> None:
    class UnavailableCloseTransport(ForwardingTransport):
        unavailable = True

        async def close_forward(self, **kwargs):
            if self.unavailable:
                raise LambdaMicroVMEndpointTransientError("offline")
            return await super().close_forward(**kwargs)

    async def scenario() -> None:
        transport = UnavailableCloseTransport(tmp_path)
        transport.fail_next_open = 3
        capability = _runner(tmp_path, transport).port_forward_capability()
        with pytest.raises(LambdaMicroVMEndpointTransientError) as caught:
            await capability.open_port_forward(host="127.0.0.1", port=1)
        handle = capability.failed_open_forward(caught.value)
        assert handle is not None
        assert len(transport.forwards._forwards) == 1
        transport.unavailable = False
        await handle.close()
        assert transport.forwards._forwards == {}

    asyncio.run(scenario())


def test_accept_recovers_after_transient_retry_budget_exhausted(tmp_path: Path) -> None:
    class UnavailableAcceptTransport(ForwardingTransport):
        failures = 4

        async def accept_forward(self, **kwargs):
            if self.failures:
                self.failures -= 1
                raise LambdaMicroVMEndpointTransientError("offline")
            return await super().accept_forward(**kwargs)

    async def scenario() -> None:
        transport = UnavailableAcceptTransport(tmp_path)
        runner = _runner(tmp_path, transport)
        server = await _upper_server()
        async with server:
            forward = await runner.port_forward_capability().open_port_forward(
                host="127.0.0.1", port=server.sockets[0].getsockname()[1]
            )
            try:
                response = await asyncio.to_thread(
                    _guest_exchange, forward.guest_host, forward.guest_port, b"recovered\n"
                )
                assert response == b"RECOVERED\n"
                assert transport.failures == 0
            finally:
                await forward.close()

    asyncio.run(scenario())


@pytest.mark.parametrize("failure", ["transient", "unconfirmed", "wrong_connection"])
def test_connection_cleanup_retains_slot_until_endpoint_recovers(
    tmp_path: Path, failure: str
) -> None:
    class FailingCleanupTransport(ForwardingTransport):
        def __init__(self, root):
            super().__init__(root)
            self.forwards.max_connections = 1
            self.cleanup_outage = True
            self.cleanup_attempts = 0

        async def close_forward_connection(self, **kwargs):
            self.cleanup_attempts += 1
            if self.cleanup_outage:
                if failure == "transient":
                    raise LambdaMicroVMEndpointTransientError("cleanup endpoint unavailable")
                return {
                    "connection": kwargs["connection"] + (failure == "wrong_connection"),
                    "closed": failure == "wrong_connection",
                }
            return await super().close_forward_connection(**kwargs)

    async def wait_until(predicate):
        async with asyncio.timeout(5):
            while not predicate():
                await asyncio.sleep(0.01)

    async def scenario():
        transport = FailingCleanupTransport(tmp_path)
        server = await _upper_server()
        async with server:
            forward = (
                await _runner(tmp_path, transport)
                .port_forward_capability()
                .open_port_forward(host="127.0.0.1", port=server.sockets[0].getsockname()[1])
            )
            try:
                assert (
                    await asyncio.to_thread(
                        _guest_exchange, forward.guest_host, forward.guest_port, b"first\n"
                    )
                    == b"FIRST\n"
                )
                await wait_until(lambda: transport.cleanup_attempts >= 4)
                guest = transport.forwards._forwards[forward.forward_id]
                assert len(guest.connections) == 1
                assert len(forward._connections) == 1
                transport.cleanup_outage = False
                await wait_until(lambda: not guest.connections and not forward._connections)
                assert (
                    await asyncio.to_thread(
                        _guest_exchange, forward.guest_host, forward.guest_port, b"second\n"
                    )
                    == b"SECOND\n"
                )
            finally:
                await forward.close()
            assert not transport.forwards._forwards

    asyncio.run(scenario())


def test_forward_close_ends_connection_cleanup_retry(tmp_path: Path) -> None:
    class FailingCleanupTransport(ForwardingTransport):
        async def close_forward_connection(self, **kwargs):
            raise LambdaMicroVMEndpointTransientError("cleanup endpoint unavailable")

    async def scenario():
        transport = FailingCleanupTransport(tmp_path)
        server = await _upper_server()
        async with server:
            forward = (
                await _runner(tmp_path, transport)
                .port_forward_capability()
                .open_port_forward(host="127.0.0.1", port=server.sockets[0].getsockname()[1])
            )
            await asyncio.to_thread(
                _guest_exchange, forward.guest_host, forward.guest_port, b"done\n"
            )
            await asyncio.wait_for(forward.close(), 3)
            assert not forward._connections
            assert not transport.forwards._forwards

    asyncio.run(scenario())
