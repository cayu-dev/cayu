"""Untrusted stdio is bounded before the vendor SDK can retain frames."""

from __future__ import annotations

import asyncio
import io
import json
from contextlib import asynccontextmanager
from types import SimpleNamespace

import pytest

from cayu.deadlines import ExecutionDeadline
from cayu.external_agents import ExternalAgentContext, ExternalAgentError, ExternalAgentLaunch
from cayu.external_agents._codex_transport import BoundedCodexTransport
from cayu.external_agents.codex import CodexAgentAdapter


@pytest.mark.parametrize("limit", ["frame", "bytes", "messages"])
def test_codex_adapter_bounds_global_notifications_during_initialize(monkeypatch, limit):
    import cayu.external_agents._codex_transport as transport

    monkeypatch.setattr(transport, "_MAX_FRAME_BYTES", 256)
    monkeypatch.setattr(transport, "_MAX_TRANSPORT_BYTES", 512)
    monkeypatch.setattr(transport, "_MAX_TRANSPORT_MESSAGES", 3 if limit == "messages" else 100)
    payload = "x" * (257 if limit == "frame" else 80 if limit == "bytes" else 1)
    line = json.dumps({"method": "guest/unknown", "params": {"data": payload}}).encode() + b"\n"
    clients, closed = [], []

    class SDKClient:
        def __init__(self, config, approval_handler):
            self.global_notifications = []
            self.closed = False
            self._proc = SimpleNamespace(
                stdout=SimpleNamespace(buffer=io.BytesIO(line * 10)),
                kill=lambda: None,
                wait=lambda: None,
            )
            clients.append(self)

        def start(self):
            pass

        def initialize(self):
            # Mirrors the SDK's global queue before any turn is registered.
            for _ in range(10):
                self.global_notifications.append(self._read_message())
            raise AssertionError("guest notifications escaped the transport bound")

        def _read_message(self):
            return json.loads(self._proc.stdout.buffer.readline())

        def close(self):
            self.closed = True

    monkeypatch.setattr(
        "cayu.external_agents.codex.import_module",
        lambda _: SimpleNamespace(CodexClient=SDKClient, CodexConfig=SimpleNamespace),
    )

    async def scenario():
        async def close():
            closed.append(True)

        @asynccontextmanager
        async def launch(context):
            yield ExternalAgentLaunch("/guest/codex", "/checkout", close)

        adapter = CodexAgentAdapter(model="test", base_url="http://gateway", launch=launch)
        with pytest.raises(ExternalAgentError, match="size limit"):
            _ = [
                event
                async for event in adapter.stream(
                    ExternalAgentContext("run", "test", ExecutionDeadline.after(10))
                )
            ]
        [client] = clients
        assert client.closed and closed == [True]
        assert len(client.global_notifications) * len(line) <= 512
        assert len(client.global_notifications) <= transport._MAX_TRANSPORT_MESSAGES
        if limit == "frame":
            assert not client.global_notifications

    asyncio.run(scenario())


@pytest.mark.parametrize("line", [b"secret invalid JSON\n", b'["secret"]\n', b'"\xffsecret"\n'])
def test_codex_transport_does_not_quote_invalid_guest_frames(line):
    client = BoundedCodexTransport()
    client._proc = SimpleNamespace(stdout=SimpleNamespace(buffer=io.BytesIO(line)))
    with pytest.raises(ExternalAgentError, match="invalid transport frame") as failure:
        client._read_message()
    assert "secret" not in str(failure.value)


def test_codex_stderr_is_drained_in_bounded_chunks_without_retention():
    reads = []

    class Stderr(io.BytesIO):
        def read(self, size=-1):
            reads.append(size)
            assert 0 < size <= 64 * 1024
            return super().read(size)

    stderr = Stderr(b"private capability" * 10_000)
    client = BoundedCodexTransport()
    client._proc = SimpleNamespace(stderr=SimpleNamespace(buffer=stderr))
    client._start_stderr_drain_thread()
    client._stderr_thread.join(timeout=2)
    assert not client._stderr_thread.is_alive()
    assert stderr.tell() == len(stderr.getvalue()) and len(reads) > 2
    assert not hasattr(client, "_stderr_lines")
